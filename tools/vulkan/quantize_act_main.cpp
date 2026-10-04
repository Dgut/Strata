// tools/vulkan/quantize_act_main.cpp - host for the Vulkan port of the activation quantizer (quantize_act.cu).
//
// Mirrors src/kernels/quantize_act_parity.cpp check by check: same fixtures in the same order off the same
// std::mt19937(7) stream, same byte-exact contract ("THE CHECK IS ON THE BYTES"), same bit-exact round-trip
// contract, same host transcriptions of ggml's q8_0/q8_K references including nearest_int and the -127 scale.
// The CUDA test's copy PROBE (a debugging aid for a device/host mismatch) is not part of the contract and is
// not ported; the graph-capture-free CUDA `quantize_q8_0_scaled` variant is not exercised by that parity test
// either, so it is not ported here - this host ports what the parity test checks.
//
// Needs shaderFloat64: subtlety 2 of the CUDA source is a DOUBLE division before rint, and there is no honest
// f32 stand-in for a byte-exact contract. The RX 6850M XT exposes it; probed at start-up, refuses otherwise.
#define NOMINMAX   // vk_util pulls in windows.h, whose min/max macros poison `std::min/std::max`
#include "vk_util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using vku::Gpu;
using vku::Slot;

namespace {

void check(VkResult r, const char* what) { if (r != VK_SUCCESS) vku::die(what, r); }

// ---- f16_from_f32 / f32_from_f16 from include/strata/kernels/f16_bits.hpp, verbatim. The references below
// use them; the shader carries its own port of the encoder and the hardware decoder, and the table modes
// (4/5) are where the two implementations are compared against the same numpy-quoted patterns.
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

// The numpy-quoted patterns from quantize_act_parity.cpp, quoted rather than reasoned (the file that OWNS
// the Q8_0/Q8_K scales asserts them against what numpy's float16 says; one of the original expectations was
// itself wrong and the kernel was right). Used for BOTH converters' self-tests: host at start-up, shader via
// table modes 4/5.
struct F16Case { float in; uint16_t want; const char* what; };
const F16Case kF16Cases[] = {
    {1e30f, 0x7C00u, "1e30 -> +inf (finite overflow)"},
    {-1e30f, 0xFC00u, "-1e30 -> -inf"},
    {65536.0f, 0x7C00u, "65536 -> +inf (just past fp16 max)"},
    {3.0e38f, 0x7C00u, "3e38 (near FLT_MAX) -> +inf"},
    {65504.0f, 0x7BFFu, "65504 -> the largest finite fp16, NOT inf"},
    {65519.996f, 0x7BFFu, "65519.996 -> just below the boundary, still finite"},
    {65520.0f, 0x7C00u, "65520.0 -> the exact tie, rounds to even = inf"},
    {5.9604645e-8f, 0x0001u, "2^-24 -> the smallest subnormal, NOT zero"},
    {1.0f, 0x3C00u, "1.0"},
    {0.0f, 0x0000u, "0.0"},
};

void selftest_f16_host() {
    int bad = 0;
    for (const F16Case& c : kF16Cases) {
        if (f16_from_f32(c.in) != c.want) {
            std::printf("    *** host f16 encoder: %s: want 0x%04X got 0x%04X\n", c.what, c.want,
                        f16_from_f32(c.in));
            ++bad;
        }
    }
    if (f32_from_f16(0x0001u) != 5.9604645e-8f) { std::printf("    *** host f16 decoder flushed the subnormal\n"); ++bad; }
    if (bad) vku::die_str("host f16 self-test failed - the references below are meaningless without it");
}

// ---- references, verbatim from quantize_act_parity.cpp
void reference_q8_0(const float* x, long long n, std::vector<uint8_t>& blocks, std::vector<float>& back) {
    blocks.assign((size_t) (n / 32) * 34, 0);
    back.assign((size_t) n, 0.0f);
    for (long long b = 0; b < n / 32; ++b) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
        uint8_t* out = &blocks[(size_t) b * 34];
        if (amax == 0.0f) continue;
        const float d32 = amax / 127.0f;
        const uint16_t dbits = f16_from_f32(d32);
        out[0] = (uint8_t) (dbits & 0xFF);
        out[1] = (uint8_t) (dbits >> 8);
        const float d16 = f32_from_f16(dbits);
        for (int i = 0; i < 32; ++i) {
            double q = std::rint((double) xb[i] / (double) d32);
            if (q > 127.0) q = 127.0;
            if (q < -128.0) q = -128.0;
            out[2 + i] = (uint8_t) (int8_t) q;
            back[(size_t) (b * 32 + i)] = (float) (int8_t) out[2 + i] * d16;
        }
    }
}

int nearest_int_host(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    std::memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

void reference_q8_K(const float* x, long long n, std::vector<uint8_t>& blocks, std::vector<float>& back,
                    float scale_num) {
    const int QK = 256;
    blocks.assign((size_t) (n / QK) * 292, 0);
    back.assign((size_t) n, 0.0f);
    for (long long b = 0; b < n / QK; ++b) {
        const float* xb = x + b * QK;
        uint8_t* out = &blocks[(size_t) b * 292];
        float max = 0.0f, amax = 0.0f;
        for (int j = 0; j < QK; ++j) {
            const float ax = std::fabs(xb[j]);
            if (ax > amax) { amax = ax; max = xb[j]; }
        }
        if (amax == 0.0f) continue;
        const float iscale = scale_num / max;
        int8_t* qs = (int8_t*) (out + 4);
        for (int j = 0; j < QK; ++j) {
            const int v = nearest_int_host(iscale * xb[j]);
            qs[j] = (int8_t) std::min(127, v);
        }
        int16_t* bsums = (int16_t*) (out + 4 + QK);
        for (int j = 0; j < QK / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
            bsums[j] = (int16_t) sum;
        }
        const float d = 1.0f / iscale;
        std::memcpy(out, &d, 4);
        for (int j = 0; j < QK; ++j) back[(size_t) (b * QK + j)] = (float) qs[j] * d;
    }
}

// ---- the shader's context: 3 storage buffers (fx floats in, blk block-words, fo floats out), push {mode,nb}
struct PC { int32_t mode, nb; };
enum : int32_t { QUANT_Q8_0 = 0, DEQUANT_Q8_0 = 1, QUANT_Q8K = 2, DEQUANT_Q8K = 3, F16_ENC = 4, F16_DEC = 5 };

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
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 64};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;   // validation: vkFreeDescriptorSets needs it
    dpci.maxSets = 64;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    check(vkCreateDescriptorPool(g.dev, &dpci, nullptr, &c.dpool), "descriptor pool");
    return c;
}

struct Step { int32_t mode, nb, groups; };

// Upload the flagged slots, run `steps` in order in ONE command buffer (device-side chaining: the dequant
// leg consumes what the quantize leg wrote, exactly like the CUDA test's two back-to-back calls), copy back
// the flagged slots. iters > 1 is only honest for idempotent windows; correctness legs run iters == 1.
void exec(Ctx& c, const std::vector<Step>& steps, Slot (&slots)[3], const VkDeviceSize sz[3],
          uint32_t upload_mask, uint32_t readback_mask, long iters, double* ms_out) {
    Gpu& g = *c.g;
    VkDescriptorBufferInfo binfo[3];
    for (int i = 0; i < 3; ++i) binfo[i] = {slots[i].dev, 0, VK_WHOLE_SIZE};
    VkDescriptorSet set{};
    {
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = c.dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &c.layout;
        check(vkAllocateDescriptorSets(g.dev, &dsai, &set), "allocate sets");
    }
    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;   // forgetting this cost bf16_main an access violation
        writes[i].dstBinding = (uint32_t) i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfo[i];
    }
    vkUpdateDescriptorSets(g.dev, 3, writes, 0, nullptr);

    {   // upload phase
        vkResetCommandBuffer(c.cmdb, 0);
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(c.cmdb, &cbi), "begin upload");
        for (int i = 0; i < 3; ++i) {
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
        for (long it = 0; it < iters; ++it) {
            for (size_t si = 0; si < steps.size(); ++si) {
                const Step& s = steps[si];
                if (si > 0)   // the dequant leg READS what the quantize leg wrote: in-order execution is not
                    vkCmdPipelineBarrier(c.cmdb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // write VISIBILITY -
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,           // dispatch to dispatch
                                         0, 0, nullptr, 0, nullptr, 0, nullptr);         // needs a barrier too
                const PC pc{s.mode, s.nb};
                vkCmdPushConstants(c.cmdb, c.playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
                vkCmdDispatch(c.cmdb, (uint32_t) s.groups, 1, 1);
            }
        }
        vkCmdPipelineBarrier(c.cmdb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 0, nullptr);
        for (int i = 0; i < 3; ++i) {
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
                                 std::chrono::steady_clock::now() - t0).count() /
                             (double) (iters * (long) steps.size());
    }
    vkFreeDescriptorSets(g.dev, c.dpool, 1, &set);
}

// Fill a download slot's host staging with a sentinel BEFORE the dispatch: regions the kernel never wrote
// come back as 0xA5 instead of undefined memory, which is the difference between "wrong value" and "no
// writer" - and that difference decides which half of the shader to suspect.
void stage_sentinel(Gpu& g, const Slot& s, VkDeviceSize bytes) {
    void* p{};
    check(vkMapMemory(g.dev, s.hmem, 0, bytes, 0, &p), "map sentinel");
    memset(p, 0xA5, (size_t) bytes);
    vkUnmapMemory(g.dev, s.hmem);
}

void read_back(Gpu& g, const Slot& s, VkDeviceSize bytes, void* out) {
    void* p{};
    check(vkMapMemory(g.dev, s.hmem, 0, bytes, 0, &p), "map read-back");
    memcpy(out, p, (size_t) bytes);
    vkUnmapMemory(g.dev, s.hmem);
}

std::vector<uint8_t> words_to_bytes(const std::vector<uint32_t>& words) {
    std::vector<uint8_t> b(words.size() * 4);
    for (size_t i = 0; i < words.size(); ++i)
        for (int k = 0; k < 4; ++k) b[i * 4 + k] = (uint8_t) ((words[i] >> (k * 8)) & 0xFFu);
    return b;
}

uint32_t groups_of(long long n) { return (uint32_t) ((n + 127) / 128); }

// ---- the CUDA test's run_case, with the CUDA calls replaced by one chained exec. Blocks are compared as
// BYTES (the contract), the round trip as bits.
int run_case(Ctx& c, const char* name, const std::vector<float>& x, bool check_bytes, double* ms_out) {
    Gpu& g = *c.g;
    const long long n = (long long) x.size();
    const long long nb = n / 32;
    const long long npairs = (nb + 1) / 2;
    std::vector<uint8_t> r_blocks;
    std::vector<float> r_back;
    reference_q8_0(x.data(), n, r_blocks, r_back);

    Slot slots[3]{};
    VkDeviceSize sz[3] = {0, 0, 0};
    slots[0] = vku::make_upload_slot(g, x.data(), (VkDeviceSize) n * 4);          sz[0] = (VkDeviceSize) n * 4;
    slots[1] = vku::make_rw_slot(g, nullptr, (VkDeviceSize) npairs * 68);         sz[1] = (VkDeviceSize) npairs * 68;
    slots[2] = vku::make_download_slot(g, (VkDeviceSize) n * 4);                  sz[2] = (VkDeviceSize) n * 4;

    stage_sentinel(g, slots[1], sz[1]);   // uploaded with the exec: words no thread wrote come back as A5
    const Step steps[] = {{QUANT_Q8_0, (int32_t) nb, (int32_t) groups_of(npairs)},
                          {DEQUANT_Q8_0, (int32_t) nb, (int32_t) groups_of(nb)}};
    double ms = 0;
    exec(c, std::vector<Step>(steps, steps + 2), slots, sz, /*upload*/ 1u | 2u, /*readback*/ 2u | 4u, 1, &ms);

    std::vector<uint32_t> g_words((size_t) npairs * 17);
    read_back(g, slots[1], sz[1], g_words.data());
    std::vector<float> g_back((size_t) n);
    read_back(g, slots[2], sz[2], g_back.data());
    const std::vector<uint8_t> g_bytes = words_to_bytes(g_words);   // pair layout is contiguous: byte 34*b

    long long byte_bad = 0, val_bad = 0;
    std::vector<long long> bad_blk;   // which blocks are bad - the INDICES tell whether this is a pattern
    if (check_bytes)
        for (size_t i = 0; i < r_blocks.size(); ++i)
            if (g_bytes[i] != r_blocks[i]) { byte_bad++; if (bad_blk.empty() || bad_blk.back() != (long long)(i / 34)) bad_blk.push_back((long long)(i / 34)); }
    if (byte_bad && bad_blk.size() > 1) {
        std::printf("  bad blocks (#%zu):", bad_blk.size());
        for (size_t i = 0; i < bad_blk.size() && i < 12; ++i) std::printf(" %lld", bad_blk[i]);
        std::printf("\n");
    }
    long long first_val = -1;
    for (long long i = 0; i < n; ++i) {
        const float a = r_back[(size_t) i], b = g_back[(size_t) i];
        if (std::memcmp(&a, &b, sizeof(float)) != 0) { ++val_bad; if (first_val < 0) first_val = i; }
    }
    if (first_val >= 0) {
        const size_t blk = (size_t) (first_val / 32), qi = (size_t) (first_val % 32);
        uint32_t ra, rb;
        memcpy(&ra, &r_back[(size_t) first_val], 4);
        memcpy(&rb, &g_back[(size_t) first_val], 4);
        const uint16_t dbits = (uint16_t) (r_blocks[blk * 34] | (r_blocks[blk * 34 + 1] << 8));
        std::printf("  first round-trip diff: elem %lld block %zu q[%zu]=%02x d16bits=%04x d16=%.9g"
                    "  ref %.9g (%08x) got %.9g (%08x)\n",
                    first_val, blk, qi, r_blocks[blk * 34 + 2 + qi], dbits, f32_from_f16(dbits),
                    (double) r_back[(size_t) first_val], ra, (double) g_back[(size_t) first_val], rb);
    }
    if (byte_bad) {
        for (size_t i = 0; i < r_blocks.size(); ++i) {
            if (g_bytes[i] != r_blocks[i]) {
                const size_t blk = i / 34, off = i % 34;
                float amax = 0; for (int k = 0; k < 32; ++k) amax = std::fmax(amax, std::fabs(x[(size_t)(blk*32+k)]));
                std::printf("    first bad byte: block %zu offset %zu%s  amax %.9g  d32 %.9g\n", blk, off,
                            off < 2 ? " (the fp16 scale)" : " (a quant)", amax, amax/127.0f);
                std::printf("      ref[0..9]:"); for (int k = 0; k < 10; ++k) std::printf(" %02x", r_blocks[blk*34+k]);
                std::printf("\n      got[0..9]:"); for (int k = 0; k < 10; ++k) std::printf(" %02x", g_bytes[blk*34+k]);
                const size_t qi = off >= 2 ? off - 2 : 0;
                const double qv = (double) x[(size_t)(blk*32+qi)] / (double) (amax/127.0f);
                std::printf("\n      x[%zu] %.9g  x/d32 %.9g   ref %02x got %02x\n", qi,
                            (double) x[(size_t)(blk*32+qi)], qv, r_blocks[i], g_bytes[i]);
                break;
            }
        }
    }
    std::printf("  %-26s blocks %s (%lld bad bytes)   round trip %s (%lld differ)", name,
                !check_bytes ? "not compared" : (byte_bad ? "*** WRONG ***" : "byte-exact"), byte_bad,
                val_bad ? "*** WRONG ***" : "bit-exact", val_bad);
    if (ms > 0) std::printf("   (%.4f ms/dispatch)", ms);
    std::printf("\n");
    if (ms_out) *ms_out = ms;
    for (int i = 0; i < 3; ++i) vku::destroy_slot(g, slots[i]);
    return (int) (byte_bad + val_bad);
}

int run_case_k(Ctx& c, const char* name, const std::vector<float>& x, bool check_bytes, double* ms_out) {
    Gpu& g = *c.g;
    const long long n = (long long) x.size();
    const long long nb = n / 256;
    std::vector<uint8_t> r_blocks;
    std::vector<float> r_back;
    reference_q8_K(x.data(), n, r_blocks, r_back, -127.0f);

    Slot slots[3]{};
    VkDeviceSize sz[3] = {0, 0, 0};
    slots[0] = vku::make_upload_slot(g, x.data(), (VkDeviceSize) n * 4);          sz[0] = (VkDeviceSize) n * 4;
    slots[1] = vku::make_rw_slot(g, nullptr, (VkDeviceSize) nb * 292);            sz[1] = (VkDeviceSize) nb * 292;
    slots[2] = vku::make_download_slot(g, (VkDeviceSize) n * 4);                  sz[2] = (VkDeviceSize) n * 4;

    stage_sentinel(g, slots[1], sz[1]);
    const Step steps[] = {{QUANT_Q8K, (int32_t) nb, (int32_t) groups_of(nb)},
                          {DEQUANT_Q8K, (int32_t) nb, (int32_t) groups_of(nb)}};
    double ms = 0;
    exec(c, std::vector<Step>(steps, steps + 2), slots, sz, 1u | 2u, 2u | 4u, 1, &ms);

    std::vector<uint32_t> g_words((size_t) nb * 73);
    read_back(g, slots[1], sz[1], g_words.data());
    std::vector<float> g_back((size_t) n);
    read_back(g, slots[2], sz[2], g_back.data());
    const std::vector<uint8_t> g_bytes = words_to_bytes(g_words);

    long long byte_bad = 0, val_bad = 0, first_bad = -1;
    if (check_bytes)
        for (size_t i = 0; i < r_blocks.size(); ++i)
            if (g_bytes[i] != r_blocks[i]) { if (first_bad < 0) first_bad = (long long) i; ++byte_bad; }
    for (long long i = 0; i < n; ++i)
        if (std::memcmp(&r_back[(size_t) i], &g_back[(size_t) i], 4) != 0) ++val_bad;
    std::printf("  %-26s blocks %s (%lld bad bytes)   round trip %s (%lld differ)", name,
                !check_bytes ? "not compared" : (byte_bad ? "*** WRONG ***" : "byte-exact"), byte_bad,
                val_bad ? "*** WRONG ***" : "bit-exact", val_bad);
    if (ms > 0) std::printf("   (%.4f ms/dispatch)", ms);
    std::printf("\n");
    if (first_bad >= 0) {
        const size_t blk = (size_t) first_bad / 292, off = (size_t) first_bad % 292;
        const char* where = off < 4 ? "d (the f32 scale)" : (off < 260 ? "qs (the quants)" : "bsums");
        std::printf("      first bad byte: block %zu offset %zu = %s   ref %02x  got %02x\n", blk, off,
                    where, r_blocks[(size_t) first_bad], g_bytes[(size_t) first_bad]);
    }
    if (ms_out) *ms_out = ms;
    for (int i = 0; i < 3; ++i) vku::destroy_slot(g, slots[i]);
    return (int) (byte_bad + val_bad);
}

}  // namespace

VKAPI_ATTR VkBool32 VKAPI_CALL dbg(VkDebugUtilsMessageSeverityFlagBitsEXT s,
                                   VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* d,
                                   void*) {
    // the lesson of this port's longest debug session: RADV creates an over-LDS-limit pipeline WITHOUT failing,
    // and it corrupts output in ways that look like lost stores. Validation says it in one line; print it.
    std::printf("VALIDATION(%x): %s\n", (unsigned) s, d->pMessage);
    return VK_FALSE;
}

int main(int argc, char** argv) {
    selftest_f16_host();
    int gpu = -1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = std::atoi(argv[++i]);
        else { vku::die_str("usage: quantize_act_vk [--gpu IDX]"); return 2; }
    }

    Gpu g = vku::open_gpu(gpu, /*want_float64*/ true, /*validate*/ true);
    VkDebugUtilsMessengerEXT messenger{};
    {
        VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        mi.pfnUserCallback = dbg;
        auto fn = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(g.inst,
                                                                             "vkCreateDebugUtilsMessengerEXT");
        if (fn) fn(g.inst, &mi, nullptr, &messenger);
    }
    if (!g.float64) vku::die_str("subtlety 2 is a DOUBLE division before rint; without shaderFloat64 there "
                                 "is nothing honest to compare byte-exact against - refusing to run an f32 stand-in");

    const VkShaderModule mod = vku::load_shader(g, "quantize_act.spv");
    Ctx c = make_ctx(g, mod);

    int bad = 0;
    std::printf("\nVulkan quantize_act on: %s\n", g.props.deviceName);

    // ---- the SHADER's f16 conversions, self-tested at start-up (rule 3) against the same numpy-quoted
    // table the host just passed: mode 4 encodes fx -> blk low halves, mode 5 decodes blk low halves -> fo.
    {
        std::vector<float> in;
        for (const F16Case& cc : kF16Cases) in.push_back(cc.in);
        const long long nc = (long long) in.size();
        Slot slots[3]{};
        VkDeviceSize sz[3] = {0, 0, 0};
        slots[0] = vku::make_upload_slot(g, in.data(), (VkDeviceSize) nc * 4);   sz[0] = (VkDeviceSize) nc * 4;
        slots[1] = vku::make_download_slot(g, (VkDeviceSize) nc * 4);            sz[1] = (VkDeviceSize) nc * 4;
        slots[2] = vku::make_download_slot(g, (VkDeviceSize) nc * 4);            sz[2] = (VkDeviceSize) nc * 4;
        const Step enc{F16_ENC, (int32_t) nc, (int32_t) groups_of(nc)};
        exec(c, {enc}, slots, sz, 1u, 2u, 1, nullptr);
        std::vector<uint32_t> got((size_t) nc);
        read_back(g, slots[1], sz[1], got.data());
        int fp16_bad = 0;
        for (size_t i = 0; i < sizeof(kF16Cases)/sizeof(kF16Cases[0]); ++i) {
            const uint16_t got_bits = (uint16_t) (got[i] & 0xFFFFu);
            if (got_bits != kF16Cases[i].want) {
                std::printf("    *** shader f16 encoder: %s: want 0x%04X got 0x%04X\n", kF16Cases[i].what,
                            kF16Cases[i].want, got_bits);
                ++fp16_bad;
            }
        }
        std::printf("  %-44s %s (%d of %d wrong)\n", "shader f32->f16 overflow saturates, not NaN",
                    fp16_bad ? "*** NO ***" : "yes", fp16_bad, (int) (sizeof kF16Cases / sizeof kF16Cases[0]));
        if (fp16_bad) bad += fp16_bad;

        // the decoder: feed the pattern 0x0001 back through mode 5 and require the exact subnormal
        const uint32_t pat = 0x0001u;
        vku::destroy_slot(g, slots[1]);
        slots[1] = vku::make_upload_slot(g, &pat, 4);   // blk becomes an upload slot for the decode leg
        sz[1] = 4;
        const Step dec{F16_DEC, 1, 1};   // one entry: groups 1

        exec(c, {dec}, slots, sz, 2u, 4u, 1, nullptr);
        float sub = 0.0f;
        read_back(g, slots[2], 4, &sub);
        const bool sub_ok = (sub == 5.9604645e-8f);
        std::printf("  %-44s %s (0x0001 -> %.9g, want 2^-24)\n", "shader f16->f32 keeps subnormals",
                    sub_ok ? "yes" : "*** NO ***", (double) sub);
        if (!sub_ok) ++bad;
        for (int i = 0; i < 3; ++i) vku::destroy_slot(g, slots[i]);
    }

    std::mt19937 rng(7);   // the CUDA test's stream: fixtures are generated in the SAME ORDER off the SAME
                          // generator, so every value below is bit-identical to the one the CUDA test used.
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const long long N = 32 * 4096;

    std::vector<float> a((size_t) N);
    for (auto& v : a) v = gauss(rng);
    double ms_q8_0 = 0;
    bad += run_case(c, "random normal", a, true, &ms_q8_0);

    std::vector<float> b((size_t) N);
    for (auto& v : b) v = gauss(rng) * 1e-6f;
    bad += run_case(c, "small magnitude (1e-6)", b, true, nullptr);

    std::vector<float> cc((size_t) N, 0.0f);
    bad += run_case(c, "all zero", cc, true, nullptr);

    std::vector<float> d((size_t) N);
    for (long long i = 0; i < N; ++i) d[(size_t) i] = ((i % 200) - 100) + 0.5f;
    d[0] = 127.0f;
    bad += run_case(c, "exact .5 boundaries", d, true, nullptr);

    std::vector<float> e((size_t) N, 0.0f);
    for (long long blk = 0; blk < N / 32; ++blk) e[(size_t) (blk * 32)] = 1000.0f + (float) (blk % 7);
    bad += run_case(c, "one extreme per block", e, true, nullptr);

    std::printf("\nquantize_q8_0: %d failures over 5 distributions x %lld elements\n", bad, N);

    std::printf("\nQ8_K  (ggml block_q8_K: 292 bytes / 256 elements, {f32 d; i8 qs[256]; i16 bsums[16]})\n");
    int bad_k = 0;
    const long long NK = 256 * 2048;

    std::vector<float> ka((size_t) NK);
    for (auto& v : ka) v = gauss(rng);
    double ms_q8k = 0;
    bad_k += run_case_k(c, "random normal", ka, true, &ms_q8k);

    std::vector<float> kb((size_t) NK);
    for (auto& v : kb) v = gauss(rng) * 1e-7f;
    bad_k += run_case_k(c, "small magnitude (1e-7)", kb, true, nullptr);

    std::vector<float> kc((size_t) NK, 0.0f);
    bad_k += run_case_k(c, "all zero", kc, true, nullptr);

    std::vector<float> kd((size_t) NK, 0.0f);
    for (long long blk = 0; blk < NK / 256; ++blk)
        for (int j = 0; j < 256; ++j) kd[(size_t) (blk * 256 + j)] = -(float) ((j % 100) + 0.5f);
    for (long long blk = 0; blk < NK / 256; ++blk) kd[(size_t) (blk * 256)] = 127.0f;
    bad_k += run_case_k(c, "exact .5 ties (half-to-even)", kd, true, nullptr);
    {
        std::vector<uint8_t> rb;
        std::vector<float> rback;
        reference_q8_K(kd.data(), NK, rb, rback, -127.0f);
        long long ties = 0, differ = 0;
        for (long long i = 0; i < NK; ++i) {
            const float t = -kd[(size_t) i];
            if (t - std::floor(t) == 0.5f) {
                ++ties;
                const int even = nearest_int_host(t);
                const int away = (int) std::round(t);
                if (even != away) ++differ;
            }
        }
        std::printf("  %-26s %lld ties, %lld where half-to-even != half-away (%.1f%%)\n",
                    "  tie fixture discriminates", ties, differ, ties ? 100.0 * (double) differ / ties : 0.0);
        if (differ < ties / 3) { std::printf("    *** the tie fixture cannot see the rounding rule ***\n"); ++bad_k; }
    }
    {
        std::vector<uint8_t> r127, r128;
        std::vector<float> b127, b128;
        reference_q8_K(ka.data(), NK, r127, b127, -127.0f);
        reference_q8_K(ka.data(), NK, r128, b128, -128.0f);
        double dd = 0, m = 0;
        long long qdiffer = 0;
        for (long long i = 0; i < NK; ++i) { dd += std::fabs(b127[(size_t) i] - b128[(size_t) i]); m += std::fabs(b127[(size_t) i]); }
        for (size_t i = 4; i < r127.size(); ++i) if (r127[i] != r128[i]) ++qdiffer;
        std::printf("  %-26s %-4s (%.3f%% of magnitude, %lld bytes differ)\n", "-127 vs -128 scale observable",
                    dd / m > 0.005 ? "yes" : "*** NO ***", 100.0 * dd / m, qdiffer);
        if (!(dd / m > 0.005)) ++bad_k;
    }

    // bandwidth with its shape (docs/VULKAN-PORT.md): quantize-only dispatches over the two big fixtures
    {
        auto bench = [&](const char* name, const std::vector<float>& x, int32_t mode, long long block_elems) {
            static const uint32_t z[16] = {};   // descriptors must be bound even when a mode ignores them
            const long long n = (long long) x.size(), nb = n / block_elems;
            const VkDeviceSize blk_bytes = (VkDeviceSize) nb * (block_elems == 32 ? 34 : 292);
            Slot slots[3]{};
            VkDeviceSize sz[3] = {0, 0, 0};
            slots[0] = vku::make_upload_slot(g, x.data(), (VkDeviceSize) n * 4);  sz[0] = (VkDeviceSize) n * 4;
            slots[1] = vku::make_download_slot(g, blk_bytes);                     sz[1] = blk_bytes;
            slots[2] = vku::make_upload_slot(g, z, 64);                           sz[2] = 64;
            const long long iters = 50;   // idempotent: re-quantizing the same input writes the same bytes
            const Step st{mode, (int32_t) nb, (int32_t) groups_of(block_elems == 32 ? (nb + 1) / 2 : nb)};
            double ms = 0;
            exec(c, {st}, slots, sz, 1u, 0u, iters, &ms);
            const double bytes = (double) n * 4.0 + (double) nb * (block_elems == 32 ? 34 : 292);
            std::printf("  %-52s %.4f ms/dispatch, %.1f GB/s\n", name, ms, bytes / (ms * 1e-3) / 1e9);
            for (int i = 0; i < 3; ++i) vku::destroy_slot(g, slots[i]);
        };
        std::printf("\nbandwidth, quantize-only dispatches (quote with their shape):\n");
        bench("q8_0 quantize, n=131072", a, QUANT_Q8_0, 32);
        bench("q8_K quantize, n=524288", ka, QUANT_Q8K, 256);
    }

    std::printf("\nquantize_act_vk: %d failures over 5 Q8_0 + 4 Q8_K distributions\n", bad + bad_k);
    vkDestroyShaderModule(g.dev, mod, nullptr);
    vkDestroyDescriptorSetLayout(g.dev, c.layout, nullptr);
    vkDestroyPipelineLayout(g.dev, c.playout, nullptr);
    vkDestroyPipeline(g.dev, c.pipe, nullptr);
    vkDestroyCommandPool(g.dev, c.cp, nullptr);
    vkDestroyDescriptorPool(g.dev, c.dpool, nullptr);
    vkDestroyDevice(g.dev, nullptr);
    vkDestroyInstance(g.inst, nullptr);
    return (bad + bad_k) ? 1 : 0;
}
