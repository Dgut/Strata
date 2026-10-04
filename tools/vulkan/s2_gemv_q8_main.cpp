// tools/vulkan/s2_gemv_q8_main.cpp - host for the Vulkan port of the S2 GEMV with Q8_0 activations (round 186).
//
// The test mirrors src/kernels/s2_gemv_q8_parity.cpp and its checks: (1) the quantized-activation GEMV against
// a host reference that quantizes the activation exactly as `quantize_q8_0_kernel` does and then dots it
// against the same weights, per-row relative, tolerance 1e-4; (2) THE GAP IS REAL: the same weights driven by
// an fp16 activation - through the ALREADY-PORTED s2_gemv shader, a real second kernel, not a reference - and
// by the Q8_0 path must DIFFER by ~the activation contract's 1%, or this test cannot see the difference it
// exists to measure.  Same shape (n_in 2560, n_out 64), same seed 31, same generator order as the CUDA test.
//
// WHY THE QUANTIZER STAYS ON THE HOST: `quantize_q8_0_kernel` divides in double (`rint((double)x/(double)d32)`)
// to reproduce ggml's bytes, and RDNA2 has no shader fp64 - porting the quantizer is its own decision, not a
// detail of this kernel. The GEMV under test consumes Q8_0 BYTES either way; the host quantizer here mirrors
// the CUDA kernel byte-for-byte and is self-tested at start-up like every other conversion helper.
#include "vk_util.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using vku::Gpu;
using vku::Slot;

namespace {

void check(VkResult r, const char* what) { if (r != VK_SUCCESS) vku::die(what, r); }

constexpr long long n_in = 2560, n_out = 64;   // the parity test's shape; the shader runs tpr=32 lanes/row

// ---- fp16 conversions, mirrored from include/strata/kernels/f16_bits.hpp via the two hosts already here ----
float f32_from_f16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t ex = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t out;
    if (ex == 0) {
        if (man == 0) {
            out = sign;
        } else {
            uint32_t m = man;
            int s = 0;
            while ((m & 0x400u) == 0) { m <<= 1; ++s; }
            out = sign | ((uint32_t) (113 - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (ex == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((ex - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

uint16_t f16_from_f32(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    int exp = (int) rawexp - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (rawexp == 0xFFu) return (uint16_t) (sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 31) return (uint16_t) (sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const uint32_t sh = (uint32_t) (14 - exp);
        uint32_t h = (man >> sh) & 0x3FFu;
        const uint32_t rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    uint16_t h = (uint16_t) (sign | ((uint32_t) exp << 10) | (man >> 13));
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return h;
}

// start-up self-test, per docs/VULKAN-PORT.md: the patterns are the ones already established in this repo's
// hosts (main.cpp's table) plus the two directions agreeing with each other.
void selftest_fp16() {
    const uint16_t bits[] = {0x3C00, 0xB900, 0x477A, 0x0001, 0x0C00};  //  1.0, -0.625, 7.4765625, 2^-24, 2^-12
    const float want[]    = {     1.0f,  -0.625f,  7.4765625f, 5.96046448e-8f, 2.44140625e-4f};
    for (size_t i = 0; i < sizeof(bits) / sizeof(bits[0]); ++i) {
        const float got = f32_from_f16(bits[i]);
        if (std::fabs(got - want[i]) > std::fabs(want[i]) * 1e-6f) {
            std::fprintf(stderr, "selftest: f32_from_f16(0x%04X) = %.9g, want %.9g\n", bits[i], got, want[i]);
            std::exit(1);
        }
    }
    const float vals[] = {                     1.0f, -0.625f, 7.4765625f, 5.96046448e-8f};
    const uint16_t wbits[] = {                       0x3C00,   0xB900,     0x477A,      0x0001};
    for (size_t i = 0; i < 4; ++i) {
        const uint16_t got = f16_from_f32(vals[i]);
        if (got != wbits[i]) {
            std::fprintf(stderr, "selftest: f16_from_f32(%.9g) = 0x%04X, want 0x%04X\n", vals[i], got, wbits[i]);
            std::exit(1);
        }
    }
}

// ---- the host mirror of quantize_q8_0_kernel (src/kernels/cuda/quantize_act.cu), byte for byte: amax loop,
// d32 = amax/127 in fp32, d stored as fp16, qs = rint(x/d32) IN DOUBLE with the same clamps. ----
void quantize_q8_0(const float* x, uint8_t* blocks, long long n) {
    for (long long b = 0; b < n / 32; ++b) {
        const float* xb = x + b * 32;
        uint8_t* out = blocks + b * 34;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
        if (amax == 0.0f) {
            out[0] = 0; out[1] = 0;
            for (int i = 0; i < 32; ++i) out[2 + i] = 0;
            continue;
        }
        const float d32 = amax / 127.0f;
        const uint16_t d16 = f16_from_f32(d32);
        out[0] = (uint8_t) (d16 & 0xFF);
        out[1] = (uint8_t) (d16 >> 8);
        for (int i = 0; i < 32; ++i) {
            double q = std::rint((double) xb[i] / (double) d32);
            if (q > 127.0) q = 127.0;
            if (q < -128.0) q = -128.0;
            out[2 + i] = (uint8_t) (int8_t) q;
        }
    }
}

// The quantizer is a conversion helper, so it gets its own start-up check too: an all-ones block has amax 1,
// d32 = fp32(1/127), whose nearest fp16 is (1 + 8/1024)*2^-7 = 0x2008 (NOT recalled - derived: 1/127 / 2^-7 =
// 128/127 = 1.00787, mantissa step 1/1024, so 8 steps with remainder .063 -> rounds to 8), and every qs is
// rint(+-~127) = ~127.  The zero block must come back all-zero bytes.
void selftest_quantizer() {
    std::vector<uint8_t> blk(34);
    float ones[32];
    for (int i = 0; i < 32; ++i) ones[i] = 1.0f;
    quantize_q8_0(ones, blk.data(), 32);
    const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
    if (dbits != 0x2008u) {
        std::fprintf(stderr, "selftest: quantize(ones) d = 0x%04X, want 0x2008\n", dbits);
        std::exit(1);
    }
    for (int i = 0; i < 32; ++i)
        if (blk[2 + i] != (uint8_t) 127) { std::fprintf(stderr, "selftest: quantize(ones) qs[%d]\n", i); std::exit(1); }
    float zeros[32] = {};
    quantize_q8_0(zeros, blk.data(), 32);
    for (int i = 0; i < 34; ++i)
        if (blk[i] != 0) { std::fprintf(stderr, "selftest: quantize(zeros) byte %d nonzero\n", i); std::exit(1); }
}

struct Pipeline {
    VkPipeline pipe{};
    VkDescriptorSet set{};
};

Pipeline make_pipeline(const Gpu& g, VkPipelineLayout playout, VkDescriptorSetLayout layout, VkDescriptorPool dp,
                       const char* spv, const VkDescriptorBufferInfo binfo[4]) {
    Pipeline p;
    const VkShaderModule mod = vku::load_shader(g, spv);
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = playout;
    check(vkCreateComputePipelines(g.dev, nullptr, 1, &cpci, nullptr, &p.pipe), spv);
    vkDestroyShaderModule(g.dev, mod, nullptr);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &layout;
    check(vkAllocateDescriptorSets(g.dev, &dsai, &p.set), "allocate sets");
    VkWriteDescriptorSet writes[4]{};
    for (int i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = p.set;   // forgetting this cost a crash on the bf16 host (docs/VULKAN-PORT.md)
        writes[i].dstBinding = (uint32_t) i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfo[i];
    }
    vkUpdateDescriptorSets(g.dev, 4, writes, 0, nullptr);
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    selftest_fp16();
    selftest_quantizer();
    int gpu = -1;
    long iters = 200;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = std::atoi(argv[++i]);
        else { vku::die_str("usage: s2_gemv_q8_vk [--iters N] [--gpu IDX]"); return 2; }
    }

    // ---- the fixture, verbatim from src/kernels/s2_gemv_q8_parity.cpp: same seed, same generation order ----
    std::mt19937 rng(31);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) n_in);
    for (auto& v : x) v = gauss(rng);
    std::vector<uint8_t> codes((size_t) n_out * n_in / 4);
    for (auto& v : codes) v = (uint8_t) (rng() & 0xFF);
    std::vector<float> scales((size_t) n_out * (n_in / 64));
    for (auto& v : scales) v = 0.001f + 0.0001f * (float) (rng() % 100);

    std::vector<uint8_t> act((size_t) (n_in / 32) * 34);
    quantize_q8_0(x.data(), act.data(), n_in);

    // ---- host reference: the parity test's, verbatim - decode the act bytes and dot sequentially ----
    auto act_val = [&](long long i) {
        const uint8_t* blk = &act[(size_t) (i / 32) * 34];
        const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
        float d;
        const uint32_t sign = (uint32_t) (dbits & 0x8000) << 16;
        uint32_t ex = (dbits >> 10) & 0x1F, man = dbits & 0x3FF;
        uint32_t out;
        if (ex == 0) { out = sign; }
        else { out = sign | ((ex - 15 + 127) << 23) | (man << 13); }
        std::memcpy(&d, &out, 4);
        return (float) ((int8_t) blk[2 + (i % 32)]) * d;
    };
    std::vector<float> ref((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        float acc = 0.0f;
        for (long long g = 0; g < n_in / 64; ++g) {
            const float d = scales[(size_t) (o * (n_in / 64) + g)];
            for (int j = 0; j < 64; ++j) {
                const long long i = g * 64 + j;
                const uint8_t byte = codes[(size_t) (o * (n_in / 4) + i / 4)];
                const int code = (byte >> ((i % 4) * 2)) & 3;
                acc += (float) (code - 1) * d * act_val(i);
            }
        }
        ref[(size_t) o] = acc;
    }

    // the fp16 activation for check 2: converted EXACTLY, as the CUDA test insists, so the comparison is about
    // the activation FORMAT and nothing else.
    std::vector<uint16_t> hx((size_t) n_in);
    for (long long i = 0; i < n_in; ++i) hx[(size_t) i] = f16_from_f32(x[(size_t) i]);

    Gpu g = vku::open_gpu(gpu);
    std::printf("\nVulkan S2 GEMV (Q8_0 activations) on: %s\n", g.props.deviceName);

    const size_t act_bytes = act.size(), codes_bytes = codes.size(), scales_bytes = scales.size() * 4;
    Slot s_act = vku::make_upload_slot(g, act.data(), act_bytes);
    Slot s_codes = vku::make_upload_slot(g, codes.data(), codes_bytes);
    Slot s_scales = vku::make_upload_slot(g, scales.data(), scales_bytes);
    Slot s_y = vku::make_download_slot(g, (VkDeviceSize) n_out * 4);
    Slot s_hx = vku::make_upload_slot(g, hx.data(), hx.size() * 2);
    Slot s_y16 = vku::make_download_slot(g, (VkDeviceSize) n_out * 4);

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
    check(vkCreateDescriptorSetLayout(g.dev, &lci, nullptr, &layout), "layout");
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};   // both shaders: {uint, uint}
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout playout{};
    check(vkCreatePipelineLayout(g.dev, &plci, nullptr, &playout), "pipeline layout");
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 2;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VkDescriptorPool dp{};
    check(vkCreateDescriptorPool(g.dev, &dpci, nullptr, &dp), "descriptor pool");

    const VkDescriptorBufferInfo q8_binfo[4] = {{s_act.dev, 0, VK_WHOLE_SIZE},   {s_codes.dev, 0, VK_WHOLE_SIZE},
                                                {s_scales.dev, 0, VK_WHOLE_SIZE}, {s_y.dev, 0, VK_WHOLE_SIZE}};
    const VkDescriptorBufferInfo f16_binfo[4] = {{s_hx.dev, 0, VK_WHOLE_SIZE},   {s_codes.dev, 0, VK_WHOLE_SIZE},
                                                 {s_scales.dev, 0, VK_WHOLE_SIZE}, {s_y16.dev, 0, VK_WHOLE_SIZE}};
    Pipeline q8 = make_pipeline(g, playout, layout, dp, "s2_gemv_q8.spv", q8_binfo);
    Pipeline f16 = make_pipeline(g, playout, layout, dp, "s2_gemv.spv", f16_binfo);

    VkCommandPoolCreateInfo cpooli{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpooli.queueFamilyIndex = g.qf;
    cpooli.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cp{};
    check(vkCreateCommandPool(g.dev, &cpooli, nullptr, &cp), "command pool");
    VkCommandBuffer cmdb{};
    {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cp;
        cbai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(g.dev, &cbai, &cmdb), "allocate cmd buffer");
    }

    // upload everything once
    {
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cmdb, &cbi), "begin upload");
        const VkBufferCopy r_act{0, 0, act_bytes}, r_codes{0, 0, codes_bytes}, r_scales{0, 0, scales_bytes},
                         r_hx{0, 0, (VkDeviceSize) hx.size() * 2};
        vkCmdCopyBuffer(cmdb, s_act.host, s_act.dev, 1, &r_act);
        vkCmdCopyBuffer(cmdb, s_codes.host, s_codes.dev, 1, &r_codes);
        vkCmdCopyBuffer(cmdb, s_scales.host, s_scales.dev, 1, &r_scales);
        vkCmdCopyBuffer(cmdb, s_hx.host, s_hx.dev, 1, &r_hx);
        vkCmdPipelineBarrier(cmdb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(cmdb), "end upload");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmdb;
        check(vkQueueSubmit(g.queue, 1, &si, nullptr), "submit upload");
        check(vkQueueWaitIdle(g.queue), "wait upload");
    }

    // ---- run the Q8 GEMV: `iters` timed dispatches + a read-back copy in one command buffer ----
    double gpu_ms = 0.0;
    {
        const uint32_t pcq[2] = {(uint32_t) (n_in / 4), (uint32_t) n_out};
        vkResetCommandBuffer(cmdb, 0);
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cmdb, &cbi), "begin dispatch");
        vkCmdBindPipeline(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, q8.pipe);
        vkCmdBindDescriptorSets(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &q8.set, 0, nullptr);
        vkCmdPushConstants(cmdb, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pcq);
        for (long it = 0; it < iters; ++it) vkCmdDispatch(cmdb, (uint32_t) n_out, 1, 1);
        const VkBufferCopy ry{0, 0, (VkDeviceSize) n_out * 4};
        vkCmdPipelineBarrier(cmdb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 0, nullptr);
        vkCmdCopyBuffer(cmdb, s_y.dev, s_y.host, 1, &ry);
        check(vkEndCommandBuffer(cmdb), "end dispatch");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmdb;
        const auto t0 = std::chrono::steady_clock::now();
        check(vkQueueSubmit(g.queue, 1, &si, nullptr), "submit");
        check(vkQueueWaitIdle(g.queue), "wait");
        gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    std::vector<float> got((size_t) n_out);
    {
        void* p{};
        check(vkMapMemory(g.dev, s_y.hmem, 0, (VkDeviceSize) n_out * 4, 0, &p), "map y");
        memcpy(got.data(), p, (size_t) n_out * 4);
        vkUnmapMemory(g.dev, s_y.hmem);
    }

    int bad = 0;
    double worst = 0.0, sum_abs = 0.0;
    for (long long o = 0; o < n_out; ++o) {
        const double a = ref[(size_t) o], b = got[(size_t) o];
        sum_abs += std::fabs(a);
        const double rel = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
        if (!(rel <= worst)) worst = rel;
        if (!(rel <= 1e-4)) ++bad;   // the parity test's tolerance: 1e-4, ten times tighter than the phase spec
    }
    std::printf("  %-38s %s (%d of %lld over 1e-4, worst rel %.3e, mean |ref| %.3f)\n",
                "q8-activation GEMV vs host reference", bad ? "*** WRONG ***" : "matches", bad, n_out, worst,
                sum_abs / (double) n_out);

    // ---- CHECK 2: THE GAP IS REAL - the same weights driven through the fp16-activation kernel must differ.
    // The rival here is the OTHER PORTED SHADER (s2_gemv), not a reference computation: a q8 path that quietly
    // did fp16 arithmetic could still match a reference, but it cannot make two different kernels agree.
    {
        const uint32_t pcf[2] = {(uint32_t) (n_in / 64), (uint32_t) n_out};   // s2_gemv: nb blocks per row
        vkResetCommandBuffer(cmdb, 0);
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cmdb, &cbi), "begin fp16 run");
        vkCmdBindPipeline(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, f16.pipe);
        vkCmdBindDescriptorSets(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &f16.set, 0, nullptr);
        vkCmdPushConstants(cmdb, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pcf);
        vkCmdDispatch(cmdb, (uint32_t) ((n_out + 63) / 64), 1, 1);   // s2_gemv: one invocation per row
        const VkBufferCopy ry{0, 0, (VkDeviceSize) n_out * 4};
        vkCmdPipelineBarrier(cmdb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 0, nullptr);
        vkCmdCopyBuffer(cmdb, s_y16.dev, s_y16.host, 1, &ry);
        check(vkEndCommandBuffer(cmdb), "end fp16 run");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmdb;
        check(vkQueueSubmit(g.queue, 1, &si, nullptr), "submit fp16");
        check(vkQueueWaitIdle(g.queue), "wait fp16");
    }
    std::vector<float> y16((size_t) n_out);
    {
        void* p{};
        check(vkMapMemory(g.dev, s_y16.hmem, 0, (VkDeviceSize) n_out * 4, 0, &p), "map y16");
        memcpy(y16.data(), p, (size_t) n_out * 4);
        vkUnmapMemory(g.dev, s_y16.hmem);
    }
    double diff = 0.0;
    for (long long o = 0; o < n_out; ++o)
        diff += std::fabs((double) y16[(size_t) o] - (double) got[(size_t) o]);
    const double rel_gap = diff / (sum_abs > 1e-30 ? sum_abs : 1e-30);
    std::printf("  %-38s %.4f%% of the reference magnitude\n", "fp16-activation vs q8-activation", rel_gap * 100);
    if (rel_gap < 1e-4) {
        std::fprintf(stderr,
                     "the two activation paths agree to %.2e, so this test CANNOT see the difference it exists "
                     "to measure - either the q8 path silently used fp16, or the fixture is degenerate\n",
                     rel_gap);
        ++bad;
    }

    const double bytes = (double) codes_bytes + (double) act_bytes + (double) n_out * 4;
    std::printf("  %-38s %.3f ms/dispatch, %.1f GB/s\n", "q8 GEMV bandwidth", gpu_ms / iters,
                bytes * (double) iters / (gpu_ms * 1e6));

    vkDestroyDescriptorPool(g.dev, dp, nullptr);
    vkDestroyPipeline(g.dev, q8.pipe, nullptr);
    vkDestroyPipeline(g.dev, f16.pipe, nullptr);
    vkDestroyCommandPool(g.dev, cp, nullptr);
    vkDestroyPipelineLayout(g.dev, playout, nullptr);
    vkDestroyDescriptorSetLayout(g.dev, layout, nullptr);
    vku::destroy_slot(g, s_act);
    vku::destroy_slot(g, s_codes);
    vku::destroy_slot(g, s_scales);
    vku::destroy_slot(g, s_y);
    vku::destroy_slot(g, s_hx);
    vku::destroy_slot(g, s_y16);
    vkDestroyDevice(g.dev, nullptr);
    vkDestroyInstance(g.inst, nullptr);
    std::printf("s2_gemv_q8_vk: %d failures\n", bad);
    return bad ? 1 : 0;
}
