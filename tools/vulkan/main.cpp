// tools/vulkan/main.cpp - host for the Vulkan port of the S2 GEMV (P2.S2), a proof-of-concept for running
// Strata's kernels on AMD/RDNA2 (issue: "make it work on my 6850M XT").
//
// The test data and the CPU reference mirror src/kernels/s2_gemv_parity.cpp: same fp16 pattern table, same
// mt19937 seed, same accumulation order as the kernel, same non-vacuity checks.  The CUDA parity test measured
// its worst relative error against ggml's scalar decode at well under 1e-5; this asserts the same tolerance
// against the Vulkan kernel on the user's GPU.
#include <vulkan/vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <windows.h>

namespace {

constexpr int QK = 64;

void die(const char* what, VkResult r) {
    std::fprintf(stderr, "%s: VkResult %d\n", what, (int) r);
    std::exit(1);
}
void die_str(const char* what) {
    std::fprintf(stderr, "%s\n", what);
    std::exit(1);
}

#define VK_CHECK(call, what)                \
    do {                                    \
        const VkResult r_ = (call);         \
        if (r_ != VK_SUCCESS) die(what, r_); \
    } while (0)

float fp16_to_fp32(uint16_t h) {
    const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 0) v = std::ldexp((float) m, -24);                       // subnormal or zero
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float) (m + 1024u), (int) e - 25);           // (1+m/1024)*2^(e-15) = (1024+m)*2^(e-25)
    return (h & 0x8000u) ? -v : v;
}

// The exponent algebra above (e-25, not the fp32-looking e-125) was gotten wrong once and it zeroed every
// reference value while still "running fine"; these constants are what catch that class of bug at start-up.
void selftest_fp16() {
    const uint16_t bits[] = {0x3C00, 0xB900, 0x477A, 0x0001, 0x0C00};  //   1.0, -0.625, 7.4765625, 2^-24, 2^-12
    const float want[]    = {     1.0f,  -0.625f,  7.4765625f, 5.96046448e-8f, 2.44140625e-4f};
    for (size_t i = 0; i < sizeof(bits) / sizeof(bits[0]); ++i) {
        const float got = fp16_to_fp32(bits[i]);
        if (std::fabs(got - want[i]) > std::fabs(want[i]) * 1e-6f) {
            std::fprintf(stderr, "fp16_to_fp32(0x%04X) = %.9g, want %.9g\n", bits[i], got, want[i]);
            std::exit(1);
        }
    }
}

const uint16_t kScales[] = {                                          // the parity test's table, verbatim
    0x3C00, 0x3800, 0x3400, 0x3000, 0x2C00, 0x2800, 0x2000,
    0xBC00, 0xB800, 0xB400, 0xB000, 0xAC00, 0x1800, 0x1400,
    0x3E00, 0x3555, 0x3C01, 0x4248, 0x4123, 0x2AAA, 0x4A2B,
    0xBE00, 0xB555, 0xC248, 0x2AAB, 0x4A2C, 0x2AAB};

std::string exe_dir() {
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string s(path);
    return s.substr(0, s.find_last_of("\\/") + 1);
}

std::vector<char> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> v((size_t) n);
    if (n > 0 && fread(v.data(), 1, (size_t) n, f) != (size_t) n) { fclose(f); return {}; }
    fclose(f);
    return v;
}

VkPhysicalDevice g_phys_{};
VkPhysicalDeviceMemoryProperties g_memprops{};

uint32_t find_mem_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_memprops.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (g_memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    die_str("no memory type");
    return 0;
}

VkBuffer create_buf(VkDevice dev, VkDeviceSize bytes, uint32_t modes, VkBufferUsageFlags usage) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = bytes;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer b{};
    VK_CHECK(vkCreateBuffer(dev, &ci, nullptr, &b), "vkCreateBuffer");
    (void) modes;
    return b;
}

VkDeviceMemory alloc_mem(VkDevice dev, VkBuffer b, VkMemoryPropertyFlags props) {
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, b, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_mem_type(g_phys_, req.memoryTypeBits, props);
    VkDeviceMemory m{};
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &m), "vkAllocateMemory");
    VK_CHECK(vkBindBufferMemory(dev, b, m, 0), "vkBindBufferMemory");
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    selftest_fp16();
    long long n_in = 2560, n_out = 640;
    double tol = 1e-5;
    int gpu = -1;
    long iters = 300;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n-in") && i + 1 < argc) n_in = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--n-out") && i + 1 < argc) n_out = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = std::atoi(argv[++i]);
        else { die_str("usage: s2_gemv_vk [--n-in N] [--n-out N] [--tol T] [--iters N] [--gpu IDX]"); return 2; }
    }
    if (n_in % QK) { die_str("--n-in must be a multiple of 64"); return 2; }
    const long long nb = n_in / QK;

    // ---- test data: identical to src/kernels/s2_gemv_parity.cpp ----
    std::mt19937 rng(999);
    const int n_scales = (int) (sizeof(kScales) / sizeof(kScales[0]));
    std::vector<uint16_t> x((size_t) n_in);
    for (long long i = 0; i < n_in; ++i) x[(size_t) i] = kScales[rng() % n_scales];
    std::vector<uint8_t> codes((size_t) n_out * nb * 16);
    std::vector<float> scales((size_t) n_out * nb);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = (uint8_t) (rng() & 0xFF);
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = fp16_to_fp32(kScales[rng() % n_scales]);

    // ---- CPU reference, same accumulation order as both kernels ----
    // `ub` is the row's sum of |term|: the cancellation scale.  A kernel that keeps the scalar ADD ORDER (the
    // vector-load port) is judged relative to |ref| and must be bit-identical; a kernel that REOPENS the order
    // (the workgroup-reduction variant) cannot hold any relative bound on rows that cancel - its own fp32
    // rounding differs from the sequential reference by ~eps*max|partial| there - so it is judged against the
    // standard condition-aware budget tol * ub instead.  See docs/VULKAN-PORT.md.
    std::vector<float> ref((size_t) n_out);
    std::vector<double> ub((size_t) n_out, 0.0);
    const auto t_cpu0 = std::chrono::steady_clock::now();
    for (long long o = 0; o < n_out; ++o) {
        float acc = 0.0f;
        double sum_abs = 0.0;
        for (long long b = 0; b < nb; ++b) {
            const float d = scales[(size_t) (o * nb + b)];
            const uint8_t* c = &codes[(size_t) ((o * nb + b) * 16)];
            const uint16_t* xb_ = &x[(size_t) (b * QK)];
            for (int j = 0; j < QK; ++j) {
                const int cm1 = ((c[j >> 2] >> ((j & 3) * 2)) & 3) - 1;
                acc += (float) cm1 * d * fp16_to_fp32(xb_[j]);
                sum_abs += std::fabs((double) cm1 * d * fp16_to_fp32(xb_[j]));
            }
        }
        ref[(size_t) o] = acc;
        ub[(size_t) o] = sum_abs;
    }
    const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cpu0).count();

    // ---- Vulkan: instance ----
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst{};
    VK_CHECK(vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance (is the driver current?)");

    uint32_t npd = 0;
    vkEnumeratePhysicalDevices(inst, &npd, nullptr);
    std::vector<VkPhysicalDevice> phys(npd);
    vkEnumeratePhysicalDevices(inst, &npd, phys.data());
    if (!npd) die_str("no Vulkan physical devices - the AMD driver does not expose Vulkan");
    for (uint32_t i = 0; i < npd; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(phys[i], &p);
        std::printf("gpu %u: %s\n", i, p.deviceName);
    }
    if (gpu < 0) {                                       // the enumeration order is not stable across runs; the
        gpu = 0;                                         // default is a discrete GPU, never the iGPU (whose UMA
        for (uint32_t i = 0; i < npd; ++i) {             // "device local" heap can out-size real VRAM)
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(phys[i], &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { gpu = (int) i; break; }
        }
    }
    if (gpu >= (int) npd) die_str("--gpu out of range");
    g_phys_ = phys[(size_t) gpu];
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g_phys_, &props);
    vkGetPhysicalDeviceMemoryProperties(g_phys_, &g_memprops);

    uint32_t qf = VK_QUEUE_FAMILY_IGNORED, qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys_, &qfc, nullptr);
    std::vector<VkQueueFamilyProperties> qfps(qfc);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys_, &qfc, qfps.data());
    for (uint32_t i = 0; i < qfc; ++i)
        if (qfps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    if (qf == VK_QUEUE_FAMILY_IGNORED) die_str("no compute queue family");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qf;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qi;
    VkDevice dev{};
    VK_CHECK(vkCreateDevice(g_phys_, &dci, nullptr, &dev), "vkCreateDevice");
    VkQueue queue{};
    vkGetDeviceQueue(dev, qf, 0, &queue);

    // ---- shader modules: the .spv files next to the executable.  s2_gemv is the vector-load port (bit-exact
    // add order); s2_gemv_wg is the workgroup-reduction variant, tolerance-bound like the CUDA fast path.
    struct Variant {
        std::string name, spv;
        bool per_row;   // true: one invocation per row; false: one WORKGROUP per row
        bool add_order; // true: keeps the scalar ADD ORDER -> judged relative to |ref|, expect bit-exactness
        VkShaderModule mod{};
        VkPipeline pipe{};
    };
    Variant variants[2] = {{"s2_gemv", "s2_gemv.spv", true, true, {}, {}},
                           {"s2_gemv_wg", "s2_gemv_wg.spv", false, false, {}, {}}};

    // ---- buffers: device-local + host-visible staging ----
    const VkDeviceSize x_bytes = (VkDeviceSize) n_in * 2;
    const VkDeviceSize c_bytes = (VkDeviceSize) codes.size();
    const VkDeviceSize s_bytes = (VkDeviceSize) scales.size() * 4;
    const VkDeviceSize y_bytes = (VkDeviceSize) n_out * 4;

    struct Slot {
        VkBuffer dev{}, host{};
        VkDeviceMemory dmem{}, hmem{};
    };
    Slot slots[4];
    const VkDeviceSize szs[4] = {x_bytes, c_bytes, s_bytes, y_bytes};
    const void* srcs[4] = {x.data(), codes.data(), scales.data(), nullptr};
    const VkDeviceSize srcn[4] = {x_bytes, c_bytes, s_bytes, 0};

    for (int i = 0; i < 4; ++i) {
        slots[i].dev = create_buf(dev, szs[i], 0,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      (i == 3 ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0));
        slots[i].dmem = alloc_mem(dev, slots[i].dev, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
    // staging buffers: host-visible coherent, transfer-source
    for (int i = 0; i < 3; ++i) {
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = szs[i];
        ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(dev, &ci, nullptr, &slots[i].host), "vkCreateBuffer staging");
        slots[i].hmem = alloc_mem(dev, slots[i].host,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* p{};
        VK_CHECK(vkMapMemory(dev, slots[i].hmem, 0, szs[i], 0, &p), "vkMapMemory");
        memcpy(p, srcs[i], (size_t) srcn[i]);
        vkUnmapMemory(dev, slots[i].hmem);
    }
    // output staging
    VkBufferCreateInfo yci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    yci.size = y_bytes;
    yci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    yci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &yci, nullptr, &slots[3].host), "vkCreateBuffer y staging");
    slots[3].hmem = alloc_mem(dev, slots[3].host,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // ---- descriptor set ----
    VkDescriptorSetLayoutBinding bnds[4]{};
    for (int i = 0; i < 4; ++i) {
        bnds[i].binding = (uint32_t) i;
        bnds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bnds[i].descriptorCount = 1;
        bnds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 4;
    lci.pBindings = bnds;
    VkDescriptorSetLayout layout{};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &layout), "vkCreateDescriptorSetLayout");

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout playout{};
    VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &playout), "vkCreatePipelineLayout");

    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    dpci.pPoolSizes = &ps;
    VkDescriptorPool dp{};
    VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &layout;
    VkDescriptorSet set{};
    VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &set), "vkAllocateDescriptorSets");

    VkWriteDescriptorSet writes[4]{};
    VkDescriptorBufferInfo binfo[4]{};
    for (int i = 0; i < 4; ++i) {
        binfo[i].buffer = slots[i].dev;
        binfo[i].offset = 0;
        binfo[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t) i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfo[i];
    }
    vkUpdateDescriptorSets(dev, 4, writes, 0, nullptr);

    // ---- uploads in their own command buffer, so each variant's timing window is dispatches only ----
    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.queueFamilyIndex = qf;
    VkCommandPool cp{};
    VK_CHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cp), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmdb{};
    VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmdb), "vkAllocateCommandBuffers");
    {
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmdb, &cbi), "vkBeginCommandBuffer");
        for (int i = 0; i < 3; ++i) {
            const VkBufferCopy r{0, 0, szs[i]};
            vkCmdCopyBuffer(cmdb, slots[i].host, slots[i].dev, 1, &r);
        }
        vkCmdPipelineBarrier(cmdb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 0, nullptr);
        VK_CHECK(vkEndCommandBuffer(cmdb), "vkEndCommandBuffer upload");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmdb;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, nullptr), "vkQueueSubmit upload");
        VK_CHECK(vkQueueWaitIdle(queue), "vkQueueWaitIdle upload");
    }

    struct { uint32_t nb, n_out; } pc{(uint32_t) nb, (uint32_t) n_out};
    const double weight_bytes = (double) n_out * nb * (16 + 4);   // codes + scales streamed per dispatch
    double lo = ref[0], hi = ref[0];
    for (float v : ref) { lo = std::fmin(lo, v); hi = std::fmax(hi, v); }
    if (lo == hi) die_str("VACUOUS: the reference is constant, so agreement proves nothing");
    std::printf("\nVulkan S2 GEMV on: %s\n", props.deviceName);
    std::printf("reference spread [%.4g, %.4g]\n", lo, hi);

    // ---- each variant: compile, dispatch `iters` times timed, read back, compare against the reference ----
    bool failed = false;
    for (Variant& va : variants) {
        const std::vector<char> spv = read_file(exe_dir() + va.spv);
        if (spv.empty()) {
            std::fprintf(stderr, "cannot read %s next to the executable (did the shader compile step run?)\n",
                         va.spv.c_str());
            return 1;
        }
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv.size();
        smci.pCode = (const uint32_t*) spv.data();
        VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &va.mod), "vkCreateShaderModule");
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = va.mod;
        cpci.stage.pName = "main";
        cpci.layout = playout;
        VK_CHECK(vkCreateComputePipelines(dev, nullptr, 1, &cpci, nullptr, &va.pipe), "vkCreateComputePipelines");

        vkResetCommandBuffer(cmdb, 0);
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmdb, &cbi), "vkBeginCommandBuffer");
        vkCmdBindPipeline(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, va.pipe);
        vkCmdBindDescriptorSets(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmdb, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &pc);
        const uint32_t groups = va.per_row ? (uint32_t) ((n_out + 63) / 64) : (uint32_t) n_out;
        for (long it = 0; it < iters; ++it) vkCmdDispatch(cmdb, groups, 1, 1);
        {
            const VkBufferCopy r{0, 0, y_bytes};
            vkCmdPipelineBarrier(cmdb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 0, nullptr);
            vkCmdCopyBuffer(cmdb, slots[3].dev, slots[3].host, 1, &r);
        }
        VK_CHECK(vkEndCommandBuffer(cmdb), "vkEndCommandBuffer");

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmdb;
        const auto t0 = std::chrono::steady_clock::now();
        VK_CHECK(vkQueueSubmit(queue, 1, &si, nullptr), "vkQueueSubmit");
        VK_CHECK(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        const double gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        std::vector<float> got((size_t) n_out);
        void* ymap{};
        VK_CHECK(vkMapMemory(dev, slots[3].hmem, 0, y_bytes, 0, &ymap), "vkMapMemory y");
        memcpy(got.data(), ymap, (size_t) y_bytes);
        vkUnmapMemory(dev, slots[3].hmem);

        long long bad = 0, first_bad = -1;
        double worst = 0.0;
        for (long long o = 0; o < n_out; ++o) {
            const double a = ref[(size_t) o], b = got[(size_t) o];
            // in-order kernels: relative to |ref|; reordered ones: relative to the row's cancellation scale
            const double denom = va.add_order ? (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30)
                                              : std::fmax(ub[(size_t) o], 1e-30);
            const double rel = std::fabs(a - b) / denom;
            if (!(rel <= worst)) worst = rel;
            if (!(rel <= tol)) {
                if (first_bad < 0) first_bad = o;
                ++bad;
            }
        }

        std::printf("%s: n_in %lld, n_out %lld, %lld rows over tolerance, worst %.3e of %s (tol %.1e)\n",
                    va.name.c_str(), n_in, n_out, bad, worst,
                    va.add_order ? "rel vs ref" : "rel vs sum|terms|", tol);
        if (bad) {
            std::printf("  first bad row %lld: ref %.9g got %.9g\n", first_bad, (double) ref[(size_t) first_bad],
                        (double) got[(size_t) first_bad]);
            failed = true;
        } else {
            std::printf("  gpu: %.3f ms per dispatch (%ld dispatches), %.1f MB of weights streamed -> %.1f GB/s\n",
                        gpu_ms / iters, iters, weight_bytes / 1e6, weight_bytes * iters / (gpu_ms * 1e6));
        }
    }

    std::printf("cpu reference: %.3f ms per pass (single thread)\n", cpu_ms);
    if (failed) return 1;

    for (Variant& va : variants) {
        vkDestroyPipeline(dev, va.pipe, nullptr);
        vkDestroyShaderModule(dev, va.mod, nullptr);
    }
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyDescriptorSetLayout(dev, layout, nullptr);
    vkDestroyPipelineLayout(dev, playout, nullptr);
    for (int i = 0; i < 4; ++i) {
        if (slots[i].dev) { vkDestroyBuffer(dev, slots[i].dev, nullptr); vkFreeMemory(dev, slots[i].dmem, nullptr); }
        if (slots[i].host) { vkDestroyBuffer(dev, slots[i].host, nullptr); vkFreeMemory(dev, slots[i].hmem, nullptr); }
    }
    vkDestroyCommandPool(dev, cp, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    std::printf("s2_gemv_vk OK\n");
    return 0;
}
