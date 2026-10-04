// tools/vulkan/elementwise_main.cpp - host for the Vulkan port of the layer glue (elementwise.cu).
//
// Mirrors src/kernels/elementwise_parity.cpp check by check: same fixtures (same seeds, same generator
// code), same tolerances, same non-vacuity checks. What does NOT transfer is the CUDA-graph capture of
// embedding_gather - a stream-capture contract on the CUDA side with no Vulkan counterpart; it is noted
// in the output rather than silently dropped.
//
// Start-up prints whether the driver exposes shaderFloat64 and DIES without it: the silu contract is
// "double then cast" (ref/gdn.py's numpy is the oracle) and there is no honest f32 fallback for a
// tolerance of 1e-7. The RX 6850M XT (RDNA2) exposes it; that was measured, not assumed.
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

// ---- f16_from_f32 from include/strata/kernels/f16_bits.hpp, verbatim (as bf16_main.cpp carries it).
// The parity contract for f32_to_f16_bulk is bit-exact agreement with THIS function; the shader carries
// its own port, and check 3 is where the two are compared.
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

// Self-test at start-up (docs/VULKAN-PORT.md rule 3). Patterns DERIVED, not recalled - and caught in the
// act here: the first version of this self-test claimed "2^-24 is the smallest normal half"; it is the
// smallest SUBNORMAL one (0x0001), the smallest NORMAL is 2^-14 (0x0400), and 2^-25 is an exact tie to
// even and rounds DOWN to zero. All three facts come out of the subnormal step 2^-24 = LSB weight.
void selftest_f16() {
    if (!(f16_from_f32(1.0f) == 0x3C00)) vku::die_str("selftest: f16(1.0)");
    if (!(f16_from_f32(0.5f) == 0x3800)) vku::die_str("selftest: f16(0.5)");
    if (!(f16_from_f32(-0.625f) == 0xB900)) vku::die_str("selftest: f16(-0.625) = 0xB900");
    if (!(f16_from_f32(0x1p-14f) == 0x0400)) vku::die_str("selftest: f16(smallest normal half) = 0x0400");
    if (!(f16_from_f32(0x1p-24f) == 0x0001)) vku::die_str("selftest: f16(smallest subnormal half) = 0x0001");
    if (!(f16_from_f32(0x1p-25f) == 0x0000)) vku::die_str("selftest: f16(tie below the smallest subnormal) = 0x0000");
    if (!(f16_from_f32(INFINITY) == 0x7C00)) vku::die_str("selftest: f16(+inf)");
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

// ---- one shader, mode by push constant; 7 storage buffers, dummies where a mode is silent. The
// push block mirrors the shader's, field for field.
struct PC {
    int32_t mode, n, h_v, bits, bias, group, out_shift, flags;
    float s, eps;
};

enum Mode : int32_t { GATE = 0, SILU = 1, SCALE = 2, TO_F16 = 3, RMS = 4, GATHER = 5 };

struct Ctx {
    Gpu* g{};
    VkDescriptorSetLayout layout{};
    VkPipelineLayout playout{};
    VkPipeline pipe{};
    VkCommandPool cp{};
    VkCommandBuffer cmdb{};
    VkDescriptorPool dpool{};
};

Ctx make_ctx(Gpu& g, VkShaderModule mod) {
    Ctx c;
    c.g = &g;
    VkDescriptorSetLayoutBinding bnds[7]{};
    for (int i = 0; i < 7; ++i) {
        bnds[i].binding = (uint32_t) i;
        bnds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bnds[i].descriptorCount = 1;
        bnds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 7;
    lci.pBindings = bnds;
    check(vkCreateDescriptorSetLayout(g.dev, &lci, nullptr, &c.layout), "layout");
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &c.layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    check(vkCreatePipelineLayout(g.dev, &plci, nullptr, &c.playout), "pipeline layout");
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = c.playout;
    check(vkCreateComputePipelines(g.dev, nullptr, 1, &cpci, nullptr, &c.pipe), "pipeline");
    VkCommandPoolCreateInfo cpooli{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpooli.queueFamilyIndex = g.qf;
    cpooli.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(g.dev, &cpooli, nullptr, &c.cp), "command pool");
    {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = c.cp;
        cbai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(g.dev, &cbai, &c.cmdb), "allocate cmd buffer");
    }
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 * 320};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 320;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    check(vkCreateDescriptorPool(g.dev, &dpci, nullptr, &c.dpool), "descriptor pool");
    return c;
}

// Upload the slots flagged in upload_mask (download-only buffers lack TRANSFER_DST and would fail the
// copy), dispatch `iters` times in one timed window, copy back the slots flagged in readback_mask.
// Sizes travel alongside the slots because vku::Slot does not record them. In-place ops must run with
// iters == 1 unless they are idempotent - a timed window of N dispatches applies the op N times before
// anything is read back, which for rms_norm meant "normalized 200 times".
void exec(Ctx& c, const PC& pc, uint32_t groups, Slot (&slots)[7], const VkDeviceSize sz[7],
          uint32_t upload_mask, uint32_t readback_mask, long iters, double* ms_out) {
    Gpu& g = *c.g;
    VkDescriptorBufferInfo binfo[7];
    for (int i = 0; i < 7; ++i) binfo[i] = {slots[i].dev, 0, VK_WHOLE_SIZE};
    VkDescriptorSet set{};
    {
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = c.dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &c.layout;
        check(vkAllocateDescriptorSets(g.dev, &dsai, &set), "allocate sets");
    }
    VkWriteDescriptorSet writes[7]{};
    for (int i = 0; i < 7; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;   // forgetting this cost bf16_main an access violation
        writes[i].dstBinding = (uint32_t) i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfo[i];
    }
    vkUpdateDescriptorSets(g.dev, 7, writes, 0, nullptr);

    {   // upload phase
        vkResetCommandBuffer(c.cmdb, 0);
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(c.cmdb, &cbi), "begin upload");
        for (int i = 0; i < 7; ++i) {
            if (!(upload_mask & (1u << i)) || !sz[i]) continue;
            const VkBufferCopy r{0, 0, sz[i]};
            vkCmdCopyBuffer(c.cmdb, slots[i].host, slots[i].dev, 1, &r);
        }
        vkCmdPipelineBarrier(c.cmdb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(c.cmdb), "end upload");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c.cmdb;
        check(vkQueueSubmit(g.queue, 1, &si, nullptr), "submit upload");
        check(vkQueueWaitIdle(g.queue), "wait upload");
    }

    {   // dispatch window + read-back copies
        vkResetCommandBuffer(c.cmdb, 0);
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(c.cmdb, &cbi), "begin dispatch");
        vkCmdBindPipeline(c.cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipe);
        vkCmdBindDescriptorSets(c.cmdb, VK_PIPELINE_BIND_POINT_COMPUTE, c.playout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(c.cmdb, c.playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        for (long it = 0; it < iters; ++it) vkCmdDispatch(c.cmdb, groups, 1, 1);
        vkCmdPipelineBarrier(c.cmdb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 0, nullptr);
        for (int i = 0; i < 7; ++i) {
            if (!(readback_mask & (1u << i))) continue;
            const VkBufferCopy r{0, 0, sz[i]};
            vkCmdCopyBuffer(c.cmdb, slots[i].dev, slots[i].host, 1, &r);
        }
        check(vkEndCommandBuffer(c.cmdb), "end dispatch");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c.cmdb;
        const auto t0 = std::chrono::steady_clock::now();
        check(vkQueueSubmit(g.queue, 1, &si, nullptr), "submit");
        check(vkQueueWaitIdle(g.queue), "wait");
        if (ms_out) *ms_out = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0).count() / (double) iters;
    }

    vkFreeDescriptorSets(g.dev, c.dpool, 1, &set);
}

// Map a slot's host memory into `out` and unmap.
void read_back(Gpu& g, const Slot& s, VkDeviceSize bytes, void* out) {
    void* p{};
    check(vkMapMemory(g.dev, s.hmem, 0, bytes, 0, &p), "map read-back");
    memcpy(out, p, (size_t) bytes);
    vkUnmapMemory(g.dev, s.hmem);
}

// Put new content into a slot's host staging memory; the next exec re-uploads it.
void re_stage(Gpu& g, const Slot& s, VkDeviceSize bytes, const void* src) {
    void* p{};
    check(vkMapMemory(g.dev, s.hmem, 0, bytes, 0, &p), "map re-stage");
    memcpy(p, src, (size_t) bytes);
    vkUnmapMemory(g.dev, s.hmem);
}

}  // namespace

int main(int argc, char** argv) {
    selftest_f16();
    int gpu = -1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = std::atoi(argv[++i]);
        else { vku::die_str("usage: elementwise_vk [--gpu IDX]"); return 2; }
    }

    Gpu g = vku::open_gpu(gpu, /*want_float64*/ true);
    if (!g.float64) vku::die_str("silu's contract is double-then-cast; without shaderFloat64 there is "
                                 "nothing honest to test - refusing to run an f32 stand-in");

    const VkShaderModule mod = vku::load_shader(g, "elementwise.spv");
    Ctx c = make_ctx(g, mod);

    int bad = 0;
    std::printf("\nVulkan elementwise on: %s\n", g.props.deviceName);

    // a silent dummy for bindings a mode does not use (descriptors must be bound even when dynamically
    // unused); one per exec is wasteful and free at this scale
    auto dummy = [&](Slot& s, VkDeviceSize& sz) {
        static const uint32_t z[16] = {};
        s = vku::make_upload_slot(g, z, 64);
        sz = 64;
    };

    // ---- 1. gdn_gate, with the reference's own softplus (fixture copied from elementwise_parity.cpp:
    // same seed 11, same generator call order, mixture that crosses the branch)
    {
        const int n = 48;
        std::mt19937 rng(11);
        std::normal_distribution<float> gen(0.0f, 1.0f);
        std::vector<float> alpha(n), dt(n), a(n), want(n);
        for (int i = 0; i < n; ++i) {
            const bool large = (i % 3) == 0;
            alpha[i] = large ? (22.0f + 6.0f * (float) (i % 17)) : gen(rng) * 3.0f;
            dt[i] = gen(rng) * 0.5f;
            a[i] = -(std::fabs(gen(rng)) + 0.1f);   // NEGATIVE, as the artifact's ssm_a is
            const double x = (double) alpha[i] + (double) dt[i];
            const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
            want[i] = (float) (sp * (double) a[i]);
        }
        {
            int over = 0, past_overflow = 0;
            for (int i = 0; i < n; ++i) {
                const float x = alpha[i] + dt[i];
                if (x > 20.0f) ++over;
                if (x > 88.0f) ++past_overflow;
            }
            std::printf("  %-44s %s (%d above 20, %d above 88)\n", "the fixture crosses the branch",
                        over ? "yes" : "*** NO ***", over, past_overflow);
            if (!over) ++bad;
        }
        {
            int positive = 0;
            for (int i = 0; i < n; ++i) if (a[i] >= 0.0f) ++positive;
            std::printf("  %-44s %s (%d of %d non-negative)\n", "the fixture's ssm_a is negative",
                        positive ? "*** NO ***" : "yes", positive, n);
            if (positive) ++bad;
        }

        Slot slots[7];
        VkDeviceSize sz[7]{};
        slots[0] = vku::make_upload_slot(g, alpha.data(), n * 4); sz[0] = n * 4;
        slots[1] = vku::make_upload_slot(g, dt.data(), n * 4);    sz[1] = n * 4;
        slots[2] = vku::make_upload_slot(g, a.data(), n * 4);     sz[2] = n * 4;
        dummy(slots[3], sz[3]); dummy(slots[4], sz[4]); dummy(slots[5], sz[5]);
        slots[6] = vku::make_download_slot(g, n * 4);             sz[6] = n * 4;
        PC pc{GATE, n, 48, 0, 0, 0, 0, 0, 0.0f, 0.0f};
        double ms = 0.0;
        exec(c, pc, (n + 31) / 32, slots, sz, 0b000111, 1u << 6, 200, &ms);   // gate is idempotent
        std::vector<float> got(n);
        read_back(g, slots[6], n * 4, got.data());

        const double rel = rel_l1(want, got);
        std::printf("  %-44s rel %.3e  (%.4f ms/dispatch)\n", "gdn_gate vs the reference", rel, ms);
        if (!(rel <= 1e-6)) { std::printf("    *** over 1e-6 ***\n"); ++bad; }

        // TRAP: no large-x branch - host-side rival, exactly as the CUDA test computes it
        {
            int wrong = 0, differing = 0;
            float first_differ = -1.0f;
            std::vector<float> rival(n);
            for (int i = 0; i < n; ++i) {
                const float x = alpha[i] + dt[i];
                const float sp = std::log1p(std::exp(x));   // no branch, f32
                rival[i] = sp * a[i];
                if (!std::isfinite(sp)) ++wrong;
                const float branched = x > 20.0f ? x : std::log1pf(std::exp(x));
                if (sp != branched) {
                    ++differing;
                    if (first_differ < 0.0f || x < first_differ) first_differ = x;
                }
            }
            const double r = rel_l1(want, rival);
            const bool visible = r > 0.05 || wrong > 0;
            std::printf("  %-44s %s (%.2f%% apart, %d non-finite, %d heads differ, first at x = %.1f)\n",
                        "the softplus large-x branch is observable", visible ? "yes" : "*** NO ***",
                        r * 100, wrong, differing, (double) first_differ);
            if (!visible) ++bad;
        }
        {
            int over = 0;
            for (float v : got) if (!(std::exp(v) < 1.0f)) ++over;
            std::printf("  %-44s %s (%d of %d not < 1)\n", "exp(gate) < 1 for every head",
                        over ? "*** NO ***" : "yes", over, n);
            if (over) ++bad;
        }
        for (int i = 0; i < 7; ++i) vku::destroy_slot(g, slots[i]);
    }

    // ---- 2. silu, in double then cast (in place: iters must stay at 1, the op is not idempotent)
    {
        const int n = 4096;
        std::mt19937 rng(22);
        std::normal_distribution<float> gen(0.0f, 3.0f);
        std::vector<float> x(n), want(n);
        for (int i = 0; i < n; ++i) {
            x[i] = gen(rng);
            const double v = (double) x[i];
            want[i] = (float) (v / (1.0 + std::exp(-v)));
        }
        Slot slots[7];
        VkDeviceSize sz[7]{};
        slots[0] = vku::make_rw_slot(g, x.data(), n * 4); sz[0] = n * 4;
        for (int i = 1; i < 7; ++i) dummy(slots[i], sz[i]);
        PC pc{SILU, n, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f};
        double ms = 0.0;
        exec(c, pc, (n + 31) / 32, slots, sz, 1u << 0, 1u << 0, 1, &ms);
        std::vector<float> got(n);
        read_back(g, slots[0], n * 4, got.data());
        const double rel = rel_l1(want, got);
        std::printf("\n  %-44s rel %.3e  (%.4f ms)\n", "silu (double) vs the reference", rel, ms);
        if (!(rel <= 1e-7)) { std::printf("    *** over 1e-7 ***\n"); ++bad; }
        for (int i = 0; i < 7; ++i) vku::destroy_slot(g, slots[i]);
    }

    // ---- 3. scale (exact) and the f32->f16 bridge (bit-exact against the shared conversion)
    {
        const int n = 1024;
        std::vector<float> x(n), want(n);
        for (int i = 0; i < n; ++i) x[i] = (float) (i - n / 2) * 0.013f;
        const float s = 1.0f / std::sqrt(128.0f);
        for (int i = 0; i < n; ++i) want[i] = x[i] * s;

        Slot slots[7];
        VkDeviceSize sz[7]{};
        slots[0] = vku::make_rw_slot(g, x.data(), n * 4); sz[0] = n * 4;
        for (int i = 1; i < 5; ++i) dummy(slots[i], sz[i]);
        slots[5] = vku::make_download_slot(g, n * 2);     sz[5] = n * 2;   // two halves per word
        dummy(slots[6], sz[6]);
        PC pc{SCALE, n, 0, 0, 0, 0, 0, 0, s, 0.0f};
        double ms = 0.0;
        exec(c, pc, (n + 31) / 32, slots, sz, 1u << 0, 1u << 0, 1, &ms);
        std::vector<float> got(n);
        read_back(g, slots[0], n * 4, got.data());
        int diff = 0;
        for (int i = 0; i < n; ++i) if (got[i] != want[i]) ++diff;
        std::printf("  %-44s %d of %d differ\n", "scale_inplace is exact", diff, n);
        if (diff) ++bad;

        // f32_to_f16 runs on the ALREADY-SCALED data: exec re-uploads from host staging, so stage the
        // scaled result first or the shader would convert the original x while we compare against f16(got)
        re_stage(g, slots[0], n * 4, got.data());
        pc.mode = TO_F16;
        exec(c, pc, (n / 2 + 31) / 32, slots, sz, 1u << 0, 1u << 5, 200, &ms);
        std::vector<uint16_t> h(n);
        read_back(g, slots[5], n * 2, h.data());
        int hbad = 0;
        for (int i = 0; i < n; ++i) if (h[i] != f16_from_f32(got[i])) ++hbad;
        std::printf("  %-44s %d of %d differ\n", "f32_to_f16_bulk uses the shared conversion", hbad, n);
        if (hbad) ++bad;
        for (int i = 0; i < 7; ++i) vku::destroy_slot(g, slots[i]);
    }

    // ---- 4. rms_norm_weighted and the TWO RIVAL READINGS it exists to be told apart from
    {
        const int rows = 24, cols = 256;
        const float eps = 1e-6f;
        std::vector<float> x((size_t) rows * cols);
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float) std::sin((double) i * 0.017) * 1.7f + 0.4f;
        std::vector<float> w(cols);
        for (int cc = 0; cc < cols; ++cc) w[cc] = 0.5f + 0.002f * (float) (cc % 97);

        auto ref = [&](bool use_sum, bool use_w) {
            std::vector<double> y((size_t) rows * cols);
            for (int r = 0; r < rows; ++r) {
                double ss = 0;
                for (int cc = 0; cc < cols; ++cc) {
                    const double v = x[(size_t) r * cols + cc];
                    ss += v * v;
                }
                const double den = std::sqrt((use_sum ? ss : ss / (double) cols) + (double) eps);
                for (int cc = 0; cc < cols; ++cc) {
                    const double v = x[(size_t) r * cols + cc];
                    y[(size_t) r * cols + cc] = (v / den) * (use_w ? (double) w[cc] : 1.0);
                }
            }
            std::vector<float> out(x.size());
            for (size_t i = 0; i < out.size(); ++i) out[i] = (float) y[i];
            return out;
        };
        const std::vector<float> want = ref(false, true);
        const std::vector<float> rival_sum = ref(true, false);
        const std::vector<float> rival_now = ref(false, false);
        const double r_sum = rel_l1(want, rival_sum), r_now = rel_l1(want, rival_now);
        std::printf("\n  %-44s %.2f%% apart\n", "sum-vs-mean is observable", r_sum * 100);
        std::printf("  %-44s %.2f%% apart\n", "with-w-vs-without-w is observable", r_now * 100);
        if (!(r_sum > 0.5)) { std::printf("  *** the sum reading is NOT observable ***\n"); ++bad; }
        if (!(r_now > 0.5)) { std::printf("  *** the no-weight reading is NOT observable ***\n"); ++bad; }

        Slot slots[7];
        VkDeviceSize sz[7]{};
        slots[0] = vku::make_rw_slot(g, x.data(), x.size() * 4); sz[0] = x.size() * 4;
        dummy(slots[1], sz[1]);
        slots[2] = vku::make_upload_slot(g, w.data(), cols * 4); sz[2] = cols * 4;
        for (int i = 3; i < 7; ++i) dummy(slots[i], sz[i]);
        PC pc{RMS, cols, 0, 0, 0, 0, 0, 1 /* has_w */, 0.0f, eps};
        double ms = 0.0;
        // iters stays at 1: rms is IN PLACE, and a timed window of 200 dispatches would normalize the
        // same buffer 200 times before the read-back - the timing legs of this host must be idempotent
        exec(c, pc, rows, slots, sz, 0b0000101, 1u << 0, 1, &ms);   // one workgroup per row
        std::vector<float> got(x.size());
        read_back(g, slots[0], x.size() * 4, got.data());
        const double r_got = rel_l1(want, got);
        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double m = std::fabs((double) want[i]);
            const double e = std::fabs((double) got[i] - (double) want[i]) / (m > 1e-30 ? m : 1e-30);
            if (e > worst) worst = e;
        }
        int nonfinite = 0;
        for (float v : got) if (!std::isfinite(v)) ++nonfinite;
        std::printf("  %-44s rel %.3e  worst %.3e  nonfinite %d  (%.4f ms/dispatch)\n",
                    "rms_norm_weighted vs f64 oracle", r_got, worst, nonfinite, ms);
        if (nonfinite) ++bad;
        if (!(r_got < 1e-6)) { std::printf("  *** rms_norm_weighted is WRONG ***\n"); ++bad; }

        // a null `w` must be legal - and elementwise_parity's fixture lesson applies VERBATIM: the
        // read-back of the leg above copied the NORMALISED tensor into slot 0's staging memory, so the
        // original x has to be re-staged before this exec re-uploads it, or this call normalizes twice.
        // (The first version of this host claimed the lesson came free from exec(); it does not - the
        // read-back is a copy too, and it lands in the staging buffer.)
        const std::vector<float> want_null = ref(false, false);
        re_stage(g, slots[0], x.size() * 4, x.data());
        pc.flags = 0;
        exec(c, pc, rows, slots, sz, 0b0000101, 1u << 0, 1, &ms);
        read_back(g, slots[0], x.size() * 4, got.data());
        int null_bad = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (std::fabs((double) got[i] - (double) want_null[i]) > 1e-6) ++null_bad;
        std::printf("  %-44s %d of %zu differ\n", "a null weight is legal", null_bad, got.size());
        if (null_bad) ++bad;
        for (int i = 0; i < 7; ++i) vku::destroy_slot(g, slots[i]);
    }

    // ---- 5. embedding gather: every code width, scale-group size, optional offset and row, bit-exact
    // against the volatile (non-fused) oracle. The CUDA test's graph-capture leg does not transfer -
    // no Vulkan counterpart - and is reported as skipped rather than quietly dropped.
    {
        std::mt19937 rng(77);
        std::uniform_real_distribution<float> values(-2.0f, 2.0f);
        int mismatches = 0, guards = 0, fma_diff = 0, cases = 0;
        for (const int bits : {2, 4, 8}) {
            const int bias = bits == 2 ? -1 : bits == 4 ? -7 : -16;
            const int per_byte = 8 / bits;
            const unsigned mask = (1u << bits) - 1u;
            for (const int group : {16, 32, 64}) {
                for (const int n : {320, 2560}) {
                    const int row_bytes = n / per_byte, row_groups = n / group;
                    std::vector<uint8_t> codes((size_t) 3 * row_bytes, 0);
                    std::vector<float> scales((size_t) 3 * row_groups), offsets(scales.size());
                    std::vector<int> unpacked((size_t) 3 * n);
                    for (size_t i = 0; i < unpacked.size(); ++i) {
                        const unsigned code = ((unsigned) i * 13u + (unsigned) (i / n) * 7u) & mask;
                        unpacked[i] = (int) code;
                        codes[i / per_byte] |= (uint8_t) (code << ((i % per_byte) * bits));
                    }
                    for (size_t i = 0; i < scales.size(); ++i) {
                        scales[i] = values(rng);
                        offsets[i] = values(rng);
                    }
                    scales[0] = -0.0f;
                    offsets[0] = 0.0f;
                    for (const bool with_offset : {false, true}) {
                        for (int row = 0; row < 3; ++row) {
                            std::vector<float> want(n), got(n + 2, -12345.0f);
                            for (int i = 0; i < n; ++i) {
                                const size_t gi = (size_t) row * row_groups + i / group;
                                const float code = (float) (unpacked[(size_t) row * n + i] + bias);
                                volatile float product = code * scales[gi];
                                const float offset = with_offset ? offsets[gi] : 0.0f;
                                want[i] = product + offset;
                                const float fused = std::fma(code, scales[gi], offset);
                                if (std::memcmp(&fused, &want[i], sizeof(float)) != 0) ++fma_diff;
                            }
                            Slot slots[7];
                            VkDeviceSize sz[7]{};
                            dummy(slots[0], sz[0]);
                            dummy(slots[1], sz[1]);
                            slots[2] = vku::make_upload_slot(g, scales.data() + (size_t) row * row_groups,
                                                             row_groups * 4);
                            sz[2] = row_groups * 4;
                            if (with_offset) {
                                slots[3] = vku::make_upload_slot(g, offsets.data() + (size_t) row * row_groups,
                                                                 row_groups * 4);
                                sz[3] = row_groups * 4;
                            } else {
                                dummy(slots[3], sz[3]);
                            }
                            const uint32_t code_words = (uint32_t) (row_bytes + 3) / 4u;
                            std::vector<uint32_t> cw(code_words, 0);
                            memcpy(cw.data(), codes.data() + (size_t) row * row_bytes, row_bytes);
                            slots[4] = vku::make_upload_slot(g, cw.data(), code_words * 4);
                            sz[4] = code_words * 4;
                            dummy(slots[5], sz[5]);
                            // the guard bytes: output buffer of n+2 floats, pre-filled -12345, kernel writes [1,n]
                            slots[6] = vku::make_rw_slot(g, got.data(), (n + 2) * 4);
                            sz[6] = (n + 2) * 4;
                            PC pc{GATHER, n, 0, bits, bias, group, 1 /* out_shift */, with_offset ? 2 : 0,
                                  0.0f, 0.0f};
                            double ms = 0.0;
                            exec(c, pc, (uint32_t) (n + 31) / 32u, slots, sz, 0b1111111, 1u << 6, 50, &ms);
                            read_back(g, slots[6], (size_t) (n + 2) * 4, got.data());
                            for (int i = 0; i < n; ++i)
                                if (std::memcmp(&want[i], &got[i + 1], sizeof(float)) != 0) ++mismatches;
                            if (got.front() != -12345.0f || got.back() != -12345.0f) ++guards;
                            ++cases;
                            for (int i = 0; i < 7; ++i) vku::destroy_slot(g, slots[i]);
                        }
                    }
                }
            }
        }
        std::printf("\n  embedding gather: %d row cases, %d bit mismatches, %d guard failures, "
                    "%d FMA differences\n", cases, mismatches, guards, fma_diff);
        std::printf("  embedding gather: note - the CUDA test's graph-capture leg has no Vulkan "
                    "counterpart and is SKIPPED, not passed\n");
        if (mismatches || guards || fma_diff == 0) ++bad;
    }

    vkDestroyDescriptorPool(g.dev, c.dpool, nullptr);
    vkDestroyCommandPool(g.dev, c.cp, nullptr);
    vkDestroyPipeline(g.dev, c.pipe, nullptr);
    vkDestroyShaderModule(g.dev, mod, nullptr);
    vkDestroyPipelineLayout(g.dev, c.playout, nullptr);
    vkDestroyDescriptorSetLayout(g.dev, c.layout, nullptr);
    vkDestroyDevice(g.dev, nullptr);
    vkDestroyInstance(g.inst, nullptr);
    std::printf("\nelementwise_vk: %d failures\n", bad);
    return bad ? 1 : 0;
}
