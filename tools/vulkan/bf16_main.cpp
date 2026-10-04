// tools/vulkan/bf16_main.cpp - host for the Vulkan port of the BF16 GEMV (P2.S5).
//
// The test mirrors src/kernels/bf16_gemv_parity.cpp and its three checks: (1) both Vulkan modes against a
// DOUBLE host reference over the same bf16 bits, judged by rel-L1 <= 1e-5 - products are exact in f32, so
// what is under test is summation order only; (2) the two modes agree with each other; (3) THE ACTIVATION
// CONTRACT IS OBSERVABLE: an fp16-rounded activation instead of a bf16 one must move the output by > 1e-4
// rel-L1, or the kernel is quietly consuming the wrong width.  Same shapes (the real ones), same seed, same
// generator code as the CUDA test; the values themselves are STL-dependent, and this host runs on the same
// STL as everything else here.
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

// ---- the bf16 conversions from include/strata/kernels/bf16_bits.hpp, verbatim ----
uint16_t bf16_from_f32(float f) {
    uint32_t i;
    memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}
float f32_from_bf16(uint16_t h) {
    const uint32_t i = (uint32_t) h << 16;
    float f;
    memcpy(&f, &i, 4);
    return f;
}

// The header's own warning: the `& 1` is on BIT 16, the lowest KEPT bit; using bit 15 differs only on exact
// ties - i.e. almost never on real data - which is exactly why it survives any end-to-end testing. These
// constants are the tie cases, checked at start-up per docs/VULKAN-PORT.md.
void selftest_bf16() {
    if (!(bf16_from_f32(1.0f) == 0x3F80)) vku::die_str("selftest: bf16_from_f32(1.0)");
    if (!(bf16_from_f32(0.5f) == 0x3F00)) vku::die_str("selftest: bf16_from_f32(0.5)");
    if (!(bf16_from_f32(-1.5f) == 0xBFC0)) vku::die_str("selftest: bf16_from_f32(-1.5)");
    uint32_t tie_even = 0x3F808000u, tie_odd = 0x3F818000u;   // discarded half exactly 0x8000
    float fe, fo;
    memcpy(&fe, &tie_even, 4);
    memcpy(&fo, &tie_odd, 4);
    if (!(bf16_from_f32(fe) == 0x3F80)) vku::die_str("selftest: tie must round to EVEN (down here)");
    if (!(bf16_from_f32(fo) == 0x3F82)) vku::die_str("selftest: tie must round to EVEN (up here)");
    const uint32_t pat_bits = (uint32_t) 0xC249u << 16;       // -50.25
    float pv;
    memcpy(&pv, &pat_bits, 4);
    if (!(f32_from_bf16(0xC249u) == pv)) vku::die_str("selftest: f32_from_bf16");
}

// ---- the fp16 conversions from include/strata/kernels/f16_bits.hpp (normal path only is exercised here:
// activations are N(0,1), no subnormals or overflow in sight) - needed for check 3's rival activation ----
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

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

void reference(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w, long long n_in, long long n_out,
               std::vector<float>& y) {
    y.assign((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        double acc = 0;
        for (long long i = 0; i < n_in; ++i)
            acc += (double) f32_from_bf16(x[(size_t) i]) * (double) f32_from_bf16(w[(size_t) (o * n_in + i)]);
        y[(size_t) o] = (float) acc;
    }
}

}  // namespace

int main(int argc, char** argv) {
    selftest_bf16();
    int gpu = -1;
    long iters = 200;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = std::atoi(argv[++i]);
        else { vku::die_str("usage: bf16_gemv_vk [--iters N] [--gpu IDX]"); return 2; }
    }

    Gpu g = vku::open_gpu(gpu);

    const VkShaderModule mod = vku::load_shader(g, "bf16_gemv.spv");
    VkDescriptorSetLayoutBinding bnds[3]{};
    for (int i = 0; i < 3; ++i) {
        bnds[i].binding = (uint32_t) i;
        bnds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bnds[i].descriptorCount = 1;
        bnds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 3;
    lci.pBindings = bnds;
    VkDescriptorSetLayout layout{};
    check(vkCreateDescriptorSetLayout(g.dev, &lci, nullptr, &layout), "layout");
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout playout{};
    check(vkCreatePipelineLayout(g.dev, &plci, nullptr, &playout), "pipeline layout");

    struct Mode { const char* name; uint32_t split; };
    const Mode modes[2] = {{"warp", 0u}, {"split256", 1u}};
    // one pipeline for both modes: the mode is a push constant and only the GRID differs, which is dispatch
    // state, not pipeline state.
    VkPipeline pipes[2];
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = playout;
    check(vkCreateComputePipelines(g.dev, nullptr, 1, &cpci, nullptr, &pipes[0]), "pipeline");
    pipes[1] = pipes[0];

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

    // THE REAL SHAPES, copied from src/kernels/bf16_gemv_parity.cpp: a kernel that only works on a square toy
    // is not a kernel this engine can use.
    struct Shape { long long n_in, n_out; const char* what; };
    const Shape shapes[] = {
        {2560, 48, "ssm_alpha.weight  [2560, 48]"},
        {2560, 128, "indexer.k_proj   [2560, 128]"},
        {2560, 512, "indexer.q_proj   [2560, 512]"},
        {2560, 2560, "ple_value        [2560, 2560]"},
    };

    int bad = 0;
    std::printf("\nVulkan BF16 GEMV on: %s\n", g.props.deviceName);

    for (const Shape& s : shapes) {
        std::mt19937 rng(4242);
        std::normal_distribution<float> gen(0.0f, 1.0f);
        std::vector<float> fx((size_t) s.n_in), fw((size_t) (s.n_in * s.n_out));
        for (auto& v : fx) v = gen(rng);
        for (auto& v : fw) v = gen(rng) * 0.05f;

        std::vector<uint16_t> x((size_t) s.n_in), w((size_t) (s.n_in * s.n_out));
        for (size_t i = 0; i < fx.size(); ++i) x[i] = bf16_from_f32(fx[i]);
        for (size_t i = 0; i < fw.size(); ++i) w[i] = bf16_from_f32(fw[i]);

        std::vector<float> want;
        reference(x, w, s.n_in, s.n_out, want);

        Slot sx = vku::make_upload_slot(g, x.data(), (VkDeviceSize) x.size() * 2);
        Slot sw = vku::make_upload_slot(g, w.data(), (VkDeviceSize) w.size() * 2);
        Slot sy = vku::make_download_slot(g, (VkDeviceSize) s.n_out * 4);

        VkDescriptorBufferInfo binfo[3] = {{sx.dev, 0, VK_WHOLE_SIZE}, {sw.dev, 0, VK_WHOLE_SIZE},
                                           {sy.dev, 0, VK_WHOLE_SIZE}};
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VkDescriptorPool dp{};
        check(vkCreateDescriptorPool(g.dev, &dpci, nullptr, &dp), "descriptor pool");
        VkDescriptorSet set{};
        {
            VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dsai.descriptorPool = dp;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &layout;
            check(vkAllocateDescriptorSets(g.dev, &dsai, &set), "allocate sets");
        }
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;   // the s2 host sets this; forgetting it here cost a crash
            writes[i].dstBinding = (uint32_t) i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &binfo[i];
        }
        vkUpdateDescriptorSets(g.dev, 3, writes, 0, nullptr);

        // upload once; then each mode: `iters` timed dispatches + a read-back copy
        {
            VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            check(vkBeginCommandBuffer(cmdb, &cbi), "begin upload");
            const VkBufferCopy r1{0, 0, (VkDeviceSize) x.size() * 2}, r2{0, 0, (VkDeviceSize) w.size() * 2};
            vkCmdCopyBuffer(cmdb, sx.host, sx.dev, 1, &r1);
            vkCmdCopyBuffer(cmdb, sw.host, sw.dev, 1, &r2);
            vkCmdPipelineBarrier(cmdb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 0, nullptr, 0, nullptr, 0, nullptr);
            check(vkEndCommandBuffer(cmdb), "end upload");
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmdb;
            check(vkQueueSubmit(g.queue, 1, &si, nullptr), "submit upload");
            check(vkQueueWaitIdle(g.queue), "wait upload");
        }

        std::vector<float> last_got;   // for the mode-vs-mode agreement check
        double gpu_ms_total = 0.0;
        std::printf("  %-34s", s.what);
        for (int m = 0; m < 2; ++m) {
            struct PC { uint32_t n_in, n_out, split; } pc{
                (uint32_t) s.n_in, (uint32_t) s.n_out, modes[m].split};
            const uint32_t groups =
                modes[m].split ? (uint32_t) s.n_out : (uint32_t) ((s.n_out + 7) / 8);

            vkResetCommandBuffer(cmdb, 0);
            VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            check(vkBeginCommandBuffer(cmdb, &cbi), "begin dispatch");
            vkCmdBindPipeline(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[0]);
            vkCmdBindDescriptorSets(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(cmdb, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pc);
            for (long it = 0; it < iters; ++it) vkCmdDispatch(cmdb, groups, 1, 1);
            const VkBufferCopy ry{0, 0, (VkDeviceSize) s.n_out * 4};
            vkCmdPipelineBarrier(cmdb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 0, nullptr, 0, nullptr, 0, nullptr);
            vkCmdCopyBuffer(cmdb, sy.dev, sy.host, 1, &ry);
            check(vkEndCommandBuffer(cmdb), "end dispatch");

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmdb;
            const auto t0 = std::chrono::steady_clock::now();
            check(vkQueueSubmit(g.queue, 1, &si, nullptr), "submit");
            check(vkQueueWaitIdle(g.queue), "wait");
            gpu_ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

            std::vector<float> got((size_t) s.n_out);
            void* ymap{};
            check(vkMapMemory(g.dev, sy.hmem, 0, (VkDeviceSize) s.n_out * 4, 0, &ymap), "map y");
            memcpy(got.data(), ymap, (size_t) s.n_out * 4);
            vkUnmapMemory(g.dev, sy.hmem);

            const double r = rel_l1(want, got);
            std::printf(" %s %.2e", modes[m].name, r);
            if (!(r <= 1e-5)) { std::printf("(*** over 1e-5 ***)"); ++bad; }
            if (m == 1) {   // check 2: the two reduction orders must agree with EACH OTHER too
                const double rr = rel_l1(last_got, got);
                std::printf("  x-agree %.2e", rr);
                if (!(rr <= 1e-5)) { std::printf("(*** over ***)"); ++bad; }
            }
            last_got = got;
        }

        // check 3: THE ACTIVATION CONTRACT IS OBSERVABLE. Feed the KERNEL an activation that went through
        // fp16 first - the engine's old behavior - and demand that it MOVES the output. Perfectly plausible
        // numbers are exactly what this catches.
        {
            std::vector<uint16_t> x_rival((size_t) s.n_in);
            for (size_t i = 0; i < fx.size(); ++i) {
                const float v = f32_from_f16(f16_from_f32(fx[i]));   // the old contract: through fp16 first
                x_rival[i] = bf16_from_f32(v);
            }
            // re-upload the rival activation and run it back through the kernel, like the CUDA test does
            {
                void* p{};
                check(vkMapMemory(g.dev, sx.hmem, 0, (VkDeviceSize) x.size() * 2, 0, &p), "map rival");
                memcpy(p, x_rival.data(), (size_t) x.size() * 2);
                vkUnmapMemory(g.dev, sx.hmem);
                vkResetCommandBuffer(cmdb, 0);
                VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                check(vkBeginCommandBuffer(cmdb, &cbi), "begin rival");
                const VkBufferCopy rx{0, 0, (VkDeviceSize) x.size() * 2};
                vkCmdCopyBuffer(cmdb, sx.host, sx.dev, 1, &rx);
                vkCmdPipelineBarrier(cmdb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 0, nullptr, 0, nullptr, 0, nullptr);
                const struct { uint32_t n_in, n_out, split; } pc{
                    (uint32_t) s.n_in, (uint32_t) s.n_out, 0u};
                vkCmdBindPipeline(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[0]);
                vkCmdBindDescriptorSets(cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &set, 0, nullptr);
                vkCmdPushConstants(cmdb, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pc);
                vkCmdDispatch(cmdb, (uint32_t) ((s.n_out + 7) / 8), 1, 1);
                const VkBufferCopy ry{0, 0, (VkDeviceSize) s.n_out * 4};
                vkCmdPipelineBarrier(cmdb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 0, nullptr, 0, nullptr, 0, nullptr);
                vkCmdCopyBuffer(cmdb, sy.dev, sy.host, 1, &ry);
                check(vkEndCommandBuffer(cmdb), "end rival");
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cmdb;
                check(vkQueueSubmit(g.queue, 1, &si, nullptr), "submit rival");
                check(vkQueueWaitIdle(g.queue), "wait rival");
            }
            std::vector<float> got_rival((size_t) s.n_out);
            void* ymap{};
            check(vkMapMemory(g.dev, sy.hmem, 0, (VkDeviceSize) s.n_out * 4, 0, &ymap), "map rival y");
            memcpy(got_rival.data(), ymap, (size_t) s.n_out * 4);
            vkUnmapMemory(g.dev, sy.hmem);

            const double r = rel_l1(want, got_rival);
            const bool visible = r > 1e-4;
            std::printf("\n      %-30s %s (%.4f%% apart)", "bf16 vs fp16 activation", visible ? "yes" : "*** NO ***",
                        r * 100.0);
            if (!visible) ++bad;
        }

        const double bytes = (double) s.n_in * s.n_out * 2 + (double) s.n_in * 2;
        std::printf("\n      %.3f ms/dispatch warp-mode-window, %.1f GB/s\n", gpu_ms_total / (2.0 * iters),
                    bytes * 2.0 * iters / (gpu_ms_total * 1e6));
        std::printf("\n");

        vkDestroyDescriptorPool(g.dev, dp, nullptr);
        vku::destroy_slot(g, sx);
        vku::destroy_slot(g, sw);
        vku::destroy_slot(g, sy);
    }

    vkDestroyCommandPool(g.dev, cp, nullptr);
    vkDestroyPipeline(g.dev, pipes[0], nullptr);
    vkDestroyShaderModule(g.dev, mod, nullptr);
    vkDestroyPipelineLayout(g.dev, playout, nullptr);
    vkDestroyDescriptorSetLayout(g.dev, layout, nullptr);
    vkDestroyDevice(g.dev, nullptr);
    vkDestroyInstance(g.inst, nullptr);
    std::printf("bf16_gemv_vk: %d failures\n", bad);
    return bad ? 1 : 0;
}
