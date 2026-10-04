// tools/vulkan/vk_util.hpp - the Vulkan host boilerplate shared by the port tests: instance/device creation
// with an explicit GPU choice, and device-local + host-visible staging buffer allocation.
//
// WHY A HEADER AND NOT A LIBRARY: these files are proof-of-concept hosts that must stay readable next to the
// kernel they test; a header keeps the build one step.  The rules encoded here come from docs/VULKAN-PORT.md:
// never default to physical device index 0 (the enumeration order is not stable and a laptop iGPU looks like
// any other device), print every device name at start-up, allocate DEVICE_LOCAL for compute buffers and
// HOST_VISIBLE|HOST_COHERENT for staging.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <windows.h>

namespace vku {

inline void die(const char* what, VkResult r) {
    std::fprintf(stderr, "%s: VkResult %d\n", what, (int) r);
    std::exit(1);
}
inline void die_str(const char* what) {
    std::fprintf(stderr, "%s\n", what);
    std::exit(1);
}

#define VKU_CHECK(call, what)                                  \
    do {                                                       \
        const VkResult r_ = (call);                            \
        if (r_ != VK_SUCCESS) ::vku::die(what, r_);            \
    } while (0)

inline std::string exe_dir() {
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    const std::string s(path);
    return s.substr(0, s.find_last_of("\\/") + 1);
}

inline std::vector<char> read_file(const std::string& p) {
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

struct Gpu {
    VkInstance inst{};
    VkPhysicalDevice phys{};
    VkDevice dev{};
    VkQueue queue{};
    uint32_t qf = VK_QUEUE_FAMILY_IGNORED;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memprops{};
    bool float64 = false;   // whether shaderFloat64 is enabled on `dev` (queried, never assumed)
};

// `gpu` < 0 selects the first DISCRETE_GPU (never index 0, never "biggest heap"); every device name is
// printed first so the user can see what is there and override with --gpu N.  `want_float64` enables
// shaderFloat64 at vkCreateDevice IF THE DEVICE HAS IT; the result lands in g.float64 and is printed,
// because "double then cast" contracts (silu) live or die on it and RDNA2's answer was measured, not
// recalled.
inline Gpu open_gpu(int gpu, bool want_float64 = false, bool validate = false) {
    Gpu g;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (validate) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = layers;
    }
    VKU_CHECK(vkCreateInstance(&ici, nullptr, &g.inst), "vkCreateInstance (is the driver current?)");

    uint32_t npd = 0;
    vkEnumeratePhysicalDevices(g.inst, &npd, nullptr);
    std::vector<VkPhysicalDevice> phys(npd);
    vkEnumeratePhysicalDevices(g.inst, &npd, phys.data());
    if (!npd) die_str("no Vulkan physical devices - the AMD driver does not expose Vulkan");
    for (uint32_t i = 0; i < npd; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(phys[i], &p);
        std::printf("gpu %u: %s\n", i, p.deviceName);
    }
    if (gpu < 0) {
        gpu = 0;
        for (uint32_t i = 0; i < npd; ++i) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(phys[i], &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { gpu = (int) i; break; }
        }
    }
    if (gpu >= (int) npd) die_str("--gpu out of range");
    g.phys = phys[(size_t) gpu];
    vkGetPhysicalDeviceProperties(g.phys, &g.props);
    vkGetPhysicalDeviceMemoryProperties(g.phys, &g.memprops);

    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qfc, nullptr);
    std::vector<VkQueueFamilyProperties> qfps(qfc);
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qfc, qfps.data());
    for (uint32_t i = 0; i < qfc; ++i)
        if (qfps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g.qf = i; break; }
    if (g.qf == VK_QUEUE_FAMILY_IGNORED) die_str("no compute queue family");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = g.qf;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures2 supp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    vkGetPhysicalDeviceFeatures2(g.phys, &supp);
    VkPhysicalDeviceFeatures2 en{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    if (want_float64 && supp.features.shaderFloat64) en.features.shaderFloat64 = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qi;
    dci.pEnabledFeatures = want_float64 ? &en.features : nullptr;   // Features2's .features is a valid type
    VKU_CHECK(vkCreateDevice(g.phys, &dci, nullptr, &g.dev), "vkCreateDevice");
    g.float64 = want_float64 && en.features.shaderFloat64 == VK_TRUE;
    std::printf("shaderFloat64: %s\n", g.float64 ? "enabled"
                    : (want_float64 ? "*** REQUESTED, NOT AVAILABLE ON THIS DEVICE ***" : "(not requested)"));
    vkGetDeviceQueue(g.dev, g.qf, 0, &g.queue);
    return g;
}

inline uint32_t find_mem_type(const Gpu& g, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g.memprops.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (g.memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    die_str("no memory type");
    return 0;
}

inline VkBuffer create_buf(const Gpu& g, VkDeviceSize bytes, VkBufferUsageFlags usage) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = bytes;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer b{};
    VKU_CHECK(vkCreateBuffer(g.dev, &ci, nullptr, &b), "vkCreateBuffer");
    return b;
}

inline VkDeviceMemory alloc_mem(const Gpu& g, VkBuffer b, VkMemoryPropertyFlags props) {
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.dev, b, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_mem_type(g, req.memoryTypeBits, props);
    VkDeviceMemory m{};
    VKU_CHECK(vkAllocateMemory(g.dev, &ai, nullptr, &m), "vkAllocateMemory");
    VKU_CHECK(vkBindBufferMemory(g.dev, b, m, 0), "vkBindBufferMemory");
    return m;
}

// A device-local compute buffer with a host-visible coherent staging twin already loaded with `src`.
struct Slot {
    VkBuffer dev{}, host{};
    VkDeviceMemory dmem{}, hmem{};
};

inline Slot make_upload_slot(const Gpu& g, const void* src, VkDeviceSize bytes) {
    Slot s;
    s.dev = create_buf(g, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    s.dmem = alloc_mem(g, s.dev, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    s.host = create_buf(g, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    s.hmem = alloc_mem(g, s.host, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* p{};
    VKU_CHECK(vkMapMemory(g.dev, s.hmem, 0, bytes, 0, &p), "vkMapMemory upload");
    memcpy(p, src, (size_t) bytes);
    vkUnmapMemory(g.dev, s.hmem);
    return s;
}

inline Slot make_download_slot(const Gpu& g, VkDeviceSize bytes) {    Slot s;
    s.dev = create_buf(g, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    s.dmem = alloc_mem(g, s.dev, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    s.host = create_buf(g, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    s.hmem = alloc_mem(g, s.host, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    return s;
}

// In-place buffers (scale, silu, rms_norm) need BOTH directions on one buffer: upload before the
// dispatch, read back after. The upload/download twins above each lack one transfer direction.
inline Slot make_rw_slot(const Gpu& g, const void* src, VkDeviceSize bytes) {
    Slot s;
    s.dev = create_buf(g, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    s.dmem = alloc_mem(g, s.dev, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    s.host = create_buf(g, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    s.hmem = alloc_mem(g, s.host, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (src) {
        void* p{};
        VKU_CHECK(vkMapMemory(g.dev, s.hmem, 0, bytes, 0, &p), "vkMapMemory rw");
        memcpy(p, src, (size_t) bytes);
        vkUnmapMemory(g.dev, s.hmem);
    }
    return s;
}

inline void destroy_slot(const Gpu& g, Slot& s) {
    if (s.dev) { vkDestroyBuffer(g.dev, s.dev, nullptr); vkFreeMemory(g.dev, s.dmem, nullptr); }
    if (s.host) { vkDestroyBuffer(g.dev, s.host, nullptr); vkFreeMemory(g.dev, s.hmem, nullptr); }
    s = Slot{};
}

// shader module from a .spv sitting next to the executable (the build compiles them there)
inline VkShaderModule load_shader(const Gpu& g, const std::string& spv_name) {
    const std::vector<char> spv = read_file(exe_dir() + spv_name);
    if (spv.empty()) {
        std::fprintf(stderr, "cannot read %s next to the executable (did the shader compile step run?)\n",
                     spv_name.c_str());
        std::exit(1);
    }
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = spv.size();
    ci.pCode = (const uint32_t*) spv.data();
    VkShaderModule m{};
    VKU_CHECK(vkCreateShaderModule(g.dev, &ci, nullptr, &m), "vkCreateShaderModule");
    return m;
}

}  // namespace vku
