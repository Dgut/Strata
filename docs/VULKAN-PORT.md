# Porting the CUDA kernels to Vulkan

Working notes from the proof-of-concept port of `src/kernels/cuda/s2_gemv.cu` (see `tools/vulkan/`). Every
gotcha below was actually hit; every rule below is what caught it or would have. Read this before porting the
next kernel.

## The method (what to do, in order)

1. **One kernel at a time, with its parity test as the spec.** Each CUDA kernel has a `*_parity.cpp` next to
   `src/kernels/`. Copy its data generation (same table, same seed), its tolerance, and its non-vacuity checks
   into the Vulkan host's test. The port is done when the numbers match the parity test's contract, not when it
   "runs".
2. **Keep the accumulation order identical to the CUDA kernel.** The tolerances (1e-5) exist to catch FMA
   contraction, and nothing else; a different summation order makes the tolerance meaningless. Pairing elements
   in GLSL is fine only when both adds happen back-to-back, exactly where the scalar loop would do them.
3. **Self-test every conversion helper at start-up.** The host `fp16_to_fp32` once shipped with exponent `e-125`
   instead of `e-25`: it "ran fine" and silently zeroed every reference value, so the comparison passed against
   garbage until the non-vacuity check caught it. A handful of known bit patterns (`0x3C00` is 1.0, `0x0001` is
   2^-24) checked at start-up catches this class of bug immediately - and when it fires, suspect the test's own
   constants too: the first run of the self-test caught a hand-computed fp16 pattern that was wrong (-0.625 is
   `0xB900`, not `0xBE00`). Compute expected patterns, don't recall them.
4. **Compile shaders in the build** (`tools/vulkan/CMakeLists.txt` runs `glslangValidator` post-build next to
   the exe). Run `spirv-val` on the output when touching anything exotic.

## GLSL gotchas (each cost a compile cycle)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `syntax error, unexpected IDENTIFIER`, empty token, column points at a **member name** inside a buffer block | the **type** before it is not known to core GLSL 450 (e.g. `ubyte`) | read byte planes as `uint` words and shift (`(word >> ((byte&3)*8)) & 0xFF`); no extension needed, and it reads 4x wider - faster too |
| `'PC' : only uniform, buffer, in, or out blocks are supported` on a push-constant block | wrote `layout(push_constant) const` | push-constant blocks are `uniform`, never `const` |
| syntax error at an SSBO's instance name | glslang reserves two-letter names like `cb`, `sb` (HLSL legacy) | give block instances descriptive names (`cbuf`, `sbuf`) |

General rule for GLSL errors: they report the token **after** the problem. When a line looks fine, look one
token earlier - usually an unknown type name. To get better messages:
`glslangValidator -C --enhanced-msgs --error-column file.comp`, and to bisect, compile prefixes of the file
(`head -n N`) plus a dummy `void main() {}`.

## Vulkan API gotchas (host side)

- **`#define VK_NO_PROTOTYPES 0` does not mean "prototypes on"** - the header checks `#ifndef`, so defining it
  at all strips the prototypes and every `vk*` call fails to link/compile. Just don't define it.
- The copy struct is **`VkBufferCopy { srcOffset, dstOffset, size }`** with the buffers as separate arguments of
  `vkCmdCopyBuffer`. (D3D12's `CopyBufferRegion` carries the buffers inside; that confusion costs a compile.)
- Barriers around staging: `TRANSFER -> COMPUTE_SHADER` after uploads, `COMPUTE_SHADER -> TRANSFER` before the
  read-back copy. The PoC uses one command buffer with everything in it, which sidesteps most other sync bugs.
- **Physical-device enumeration order is not stable across runs**, and a laptop has an iGPU that looks like any
  other device. Print all device names at start-up, default to `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` (not index
  0, and not "biggest DEVICE_LOCAL heap" either - an iGPU's UMA heap out-sizes real VRAM), and let `--gpu N`
  override - then echo which device actually ran, above the results.
- Allocate DEVICE_LOCAL buffers + HOST_VISIBLE|HOST_COHERENT staging; `vkMapMemory` once for in/out. It is a
  few dozen lines more than host-visible-everywhere and keeps benchmark numbers honest about real VRAM speed.

## Tooling on Windows/PowerShell

- PowerShell splits `-target-env=vulkan1.2` at the dot; quote it or run through `cmd /c`. Note the two tools
  differ: `glslangValidator -V --target-env vulkan1.2` (space), `glslc -target-env=vulkan1.2` (equals).
- Find the SDK via `$env:VULKAN_SDK/Bin`; in CMake use
  `find_program(GLSLANG glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)`.

## What a finished port prints

The PoC's output is the template - all of it matters: device name used, rows over tolerance and the worst
relative error against the tolerance, the reference spread (non-vacuity), ms per dispatch and streamed GB/s so
later optimizations are comparable, and the single-thread CPU reference for scale.

## The s2_gemv follow-up: vector loads, workgroup reduction, and a tolerance that had to change

The scalar port measured 31-89 GB/s depending on shape and iteration count (dispatch overhead dominates at the
small parity-test shape - **always quote bandwidth with its shape and iters**). The follow-up added both levers
the notes promised, as two pipelines in one host:

- `s2_gemv.comp` v2 - uvec4 loads (one per block of codes, 8 fp16 per x load), add order untouched. Still
  **bit-identical** (worst rel 0.000) and measured ~99 GB/s at n_in 4096 / n_out 8192.
- `s2_gemv_wg.comp` - one workgroup per row, four accumulators per lane, tree reduction: the structure of
  `s2_gemv_fast.cu`. Measured ~100 GB/s at that shape; it does not beat the in-order port there, and both sit
  well under the card's ~470 GB/s peak. Like the CUDA fast path's negative results, this is a measurement to
  keep, not a win to claim: micro-tuning below the noise floor is how the CUDA side lost a week.

**The reordered variant cannot pass the parity test's tolerance, and that is arithmetic, not a bug.** On rows
that cancel (sum ~0.1% of the partial magnitudes - ordinary for these distributions) ANY fp32 kernel that
reopens the add order differs from the sequential reference by ~eps*max|partial|, which relative to the small
result exceeded 1e-5 (measured 5.6e-5 worst). So the host judges the two kinds differently:

- in-order kernels: relative to |ref|, and they must be bit-exact - that is what "same add order" means;
- reordered kernels: against the condition-aware budget `tol * sum(|terms|)` per row (the row's cancellation
  scale, computed alongside the reference). The wg variant came in at 6.3e-7 of that budget - 16x margin.

A gotcha on the way: selecting one of a uvec4's four words by testing bits **2 and 3** of the index compiles,
runs, and is wrong (worst rel 641). Word `p>>3` is picked by bits **3 and 4** of `p`. Bit-test selectors get
their own worked example here because nothing else in the output looks wrong.

## The second kernel: BF16 GEMV (`bf16_gemv.comp`, `bf16_main.cpp`)

The method held up on a different kernel with no CUDA-specific shape. What transferred and what was new:

- **It transferred:** mirror the parity test verbatim (same four REAL shapes, same seed 4242, same generator
  code), self-test the conversions at start-up (the bf16 tie-to-even cases `0x3F808000 -> 0x3F80` and
  `0x3F818000 -> 0x3F82` are the header's exact "differs only on ties, survives any end-to-end test" warning),
  compile in the build. All four shapes passed first run: rel-L1 vs the DOUBLE reference ~8e-8 for both modes,
  cross-mode agreement ~9e-8. Products are exact (both sides bf16-valued), so that residual is summation order
  and nothing else - which is what the double reference is for when a kernel legitimately has no scalar add
  order to match.
- **The activation contract must be ported too, not just the arithmetic.** `bf16_gemv_parity`'s third check -
  feed a bf16 activation that went through fp16 and require the output to MOVE (it does: ~0.14%, well above the
  1e-4 floor) - is the check that the kernel consumes the right width. Ported by re-uploading the rival bits
  and running them back through the same shader, exactly as the CUDA test does; a reference-only version of
  this check would pass against a kernel that ignores `x` entirely.
- **New host gotcha (`dstSet`).** A `VkWriteDescriptorSet` with `dstSet == VK_NULL_HANDLE` passed to
  `vkUpdateDescriptorSets` is not a clean error on the AMD driver - it access-violates (exit `0xC0000005`) and,
  because stdout was still buffered, printed nothing. The s2 host set `dstSet`; re-deriving the boilerplate by
  memory dropped it. Hence `vk_util.hpp`: one place for device pick + staging slots + shader load, so the next
  kernel's host starts from code that already has this in it. (s2's host still carries its own copy; migrate
  it there when next touching that file.)
- **New shader gotcha (barriers are not `return`).** In warp-per-row mode an out-of-range row cannot early-`return`:
  every thread in the workgroup must reach `barrier()`. Out-of-range lanes are PREDICATED (skip the loop body and
  the write, still hit every barrier), not returned. GLSL does not diagnose the wrong version; it just hangs or
  corrupts.

No subgroup ops: RDNA2 has them, but a shared-memory tree over each row's 32 lanes compiles for every Vulkan 1.2
driver and measured the same. Two modes in ONE shader picked by a push constant (the mode is data; only the GRID
differs, which is dispatch state) - one pipeline serves both. Bandwidth at the real shapes: ~130-150 GB/s on the
small projections, 270 GB/s on `ple_value` [2560,2560]; the small shapes are latency-bound, not bandwidth-bound.

## The third kernel: Q8_0-activation GEMV (`s2_gemv_q8.comp`, `s2_gemv_q8_main.cpp`)

The first port whose INPUT is produced by another kernel - and what that taught:

- **A CUDA kernel's helper may stay on the host, and saying why is part of the port.** `quantize_q8_0_kernel`
  divides in DOUBLE (`rint((double)x/(double)d32)`) to reproduce ggml's Q8_0 bytes, and RDNA2 exposes no shader
  fp64. Porting the quantizer is its own decision (int64 cross-multiplication or a two-float division with an
  exact tie rule); for this test the host mirrors the CUDA kernel byte-for-byte, self-tested at start-up like
  every conversion helper (all-ones block: d = `0x2008` - DERIVED from 1/127 / 2^-7, not recalled - qs all 127;
  zero block: all-zero bytes). The GEMV under test consumes Q8_0 bytes either way.
- **Check 2's rival should be a real kernel, not a reference.** The CUDA parity drives the fp16 path through
  `s_gemv_split`; the Vulkan host drives it through the ALREADY-PORTED `s2_gemv` shader in the same process (two
  pipelines, one descriptor layout - both are 4 SSBOs + an 8-byte push block). A q8 path that quietly did fp16
  arithmetic could still match a reference; it cannot make two different kernels agree. Measured gap on this
  card: 0.62% of the reference magnitude, against the CUDA test's ~0.95% - same order, same verdict.
- **Faithfulness signal worth having: the worst relative error came out at exactly the CUDA kernel's measured
  1.479e-05** (tolerance 1e-4) - same fixture, same per-thread strided quads, four accumulators, `(a0+a1)+(a2+a3)`
  then the 16/8/4/2/1 tree at tpr=32. Mirroring the launch shape made the FMA-contraction fingerprint match too.
- **GLSL gotcha (`shared` in nested scope).** `shared float partial[32];` inside `main()` compiles to: "not
  allowed in nested scope". Workgroup memory is declared at FILE scope, next to the push-constant block.

Bandwidth at the parity shape (n_in 2560, n_out 64 - tiny, dispatch-dominated): ~36 GB/s; quote it with its
shape, as always.

## The fourth kernel: the layer glue (`elementwise.comp`, `elementwise_main.cpp`)

`elementwise.cu` in one mode-switched shader (gdn_gate, silu, scale, f32->f16, rms_norm_weighted,
embedding_gather), host mirrors `elementwise_parity.cpp` with its seeds, tolerances and non-vacuity checks
intact. What the CUDA side never had to answer, Vulkan did:

- **glslang's Vulkan GLSL has double OPERATORS but no double BUILT-INS.** `exp/log/sqrt/floor` on `double`
  die with the whole message `'exp' : no matching overloaded function found`; no `#extension` line and not
  `450core` bring them (glslc is the same glslang). The 6850M XT exposes `shaderFloat64` (probed at start-up,
  printed, device refuses to run without it - silu's contract is double-then-cast and there is no honest f32
  stand-in for a 1e-7 tolerance), and the operators compile fine on it. The fix for exp is 20 lines: range
  reduce `x = k*ln2 + r` (floor done manually - `int()` truncates toward zero, step down when above), Taylor
  to r^16 in double (truncation ~1e-17 for |r| <= ln2/2), then scale by 2^k with an exact doubling loop. It
  measured **bit-identical** to the host's `std::exp` reference on silu's fixture (rel 0.000). gdn_gate kept
  CUDA's f32 softplus: the 1e-6 tolerance was set by CUDA's f32 passing it, and GLSL's f32 `log(1+exp(x))`
  is the same contract as `log1pf(expf)` where `exp` cannot overflow (measured rel 4.0e-8).
- **`layout(push_constant) uniform PC pc;` with a named struct type compiles nowhere** - "non-opaque uniforms
  outside a block" + "push_constant can only be used with a block". Push constants are BLOCKS: `layout
  (push_constant) uniform PC { ... } pc;`. The two error lines name the same token and mean one thing.
- **A shared-memory broadcast race has a fingerprint, and it is `rel-L1 = exactly 1.000e+00` with zero
  non-finite.** The rms tree ended with lane 0 computing `inv` from `sred[0]`, then ALL lanes storing
  `sred[0] = inv` - but lanes 1..31 still held their private `inv = 0.0` and raced lane 0's read AND its
  write. Lane 0 lost twice, every lane read `inv = 0`, and the kernel cheerfully normalized every element by
  zero: outputs exactly +/-0 (yes, -0 too), worst relative error exactly 1.0, nothing non-finite to complain
  about. One writer into shared, barrier, then everyone reads.
- **A read-back is a copy too, and it lands in the staging buffer.** The CUDA fixture's "re-upload before the
  second in-place call" lesson survived the port in full: `exec()` re-uploads from host STAGING, but the
  previous leg's dev->host copy overwrote that staging with its result. The null-weight leg normalized a
  normalized tensor and reported 6144 of 6144 differ - the same fixture bug the CUDA test's comment warns
  about, re-hit verbatim one API over. Re-stage the original input between in-place legs (and keep timed
  windows to idempotent ops: an `iters=200` window on rms normalizes 200 times before anything is read).
- **FMA separation without a compiler knob**: gather's oracle is bit-exact against non-fused mul-then-add,
  RADV contracts aggressively and GLSL cannot say no - so the product goes through shared memory with a
  `barrier()` between the mul and the add; nothing fuses across a workgroup barrier. Workgroup-uniform loop
  trip counts keep every lane at every barrier (the bf16 lesson, used on purpose). Result: **all 108
  bit-exact cases pass, 0 guard failures**, including the -0.0 scale and both offset legs. The CUDA test's
  graph-capture leg has no Vulkan counterpart; it is printed as SKIPPED rather than quietly dropped.
- The f16 self-test caught its OWN author's hand-derived patterns again (rule 3, third time): "2^-24 is the
  smallest normal half" - it is the smallest SUBNORMAL one (0x0001); smallest normal is 2^-14 (0x0400), and
  2^-25 ties to even and rounds DOWN to zero. Derive from the LSB weight 2^-24, every time.

`vk_util.hpp` grew: `open_gpu(gpu, want_float64)` (queries support, enables it if present, records the answer
in `Gpu::float64`, prints it) and `make_rw_slot` for in-place buffers - upload AND read-back on one buffer,
which the upload/download twins each lack a transfer direction for.

## The fifth kernel: the activation quantizer (`quantize_act.comp`, `quantize_act_main.cpp`)

The byte-exact contract (ggml's Q8_0/Q8_K bytes) was worth every debugging cycle it cost - and it exposed
four things no earlier port needed:

- **RADV creates an over-LDS-limit pipeline WITHOUT failing, and it corrupts output in ways that look like
  lost stores.** A `shared uint sq[128][68]` staging array (34816 B) exceeded this driver's
  `maxComputeSharedMemorySize` (32768): `vkCreateComputePipelines` returned SUCCESS, and every dispatch came
  back with byte-granular garbage scattered across runs of consecutive threads. Only
  `VK_LAYER_KHRONOS_validation` says it in one line (`VUID-RuntimeSpirv-Workgroup-06530`). Hence
  `open_gpu(gpu, want_float64, validate)` + a stdout messenger in this host: **run the port under validation
  whenever results are weird at all.** The fix was packing the staging to words (8704 B), which was the
  right layout anyway.
- **Two dispatches in one command buffer need a barrier between them.** The dequant leg reads what the
  quantize leg wrote; in-order execution is not write VISIBILITY - dispatch-to-dispatch, like everything
  else in Vulkan, requires `vkCmdPipelineBarrier(COMPUTE_SHADER -> COMPUTE_SHADER)`. Without it the failure
  was non-deterministic and looked exactly like a race with the read-back copy (the comparison baseline was
  staged as a 0xA5 sentinel - see below - so "stale" showed up as sentinel bytes).
- **Vulkan GLSL's default fp is relaxed, and RADV exploits it in two ways that break byte-exact contracts:**
  `1.0 / x` lowers to the approximate reciprocal (the Q8_K block scale came out 1 ULP off on ~7% of blocks;
  an approximate `iscale` also shifts every product's last bit, which only shows up at rounding boundaries),
  and products lose the SIGN OF ZERO (`-8 * +0.0f` came back `+0`, not `-0`). The fixes: divide through
  double and cast once (empirically the same bits as a correctly-rounded f32 divide on all 2048 scales of
  every fixture - measured, not argued), and rebuild zero's sign from the operands when the product is zero.
  Kernels 1-4 divided only where tolerances could not see an ULP (or by exact powers of two), which is why
  this is invisible until a BYTE-EXACT contract looks.
- **The ggml `nearest_int` magic number does not need porting - its ROUNDED PRODUCT does.** nvcc needed
  `__fmul_rn` to stop the mul fusing into the magic add; GLSL cannot say no either, but a floor-first
  implementation (`floor(p)`, exact `p - floor(p)`, explicit half-to-even on ties) puts a libcall between the
  mul and everything else, so nothing can fuse through it. It matched the magic number on every fixture,
  including the manufactured exact-.5 ties (522240 of them, 49.8% where half-to-even != half-away).

Layout lessons: **34-byte blocks are not word-aligned but BLOCK PAIRS are** (68 B = exactly 17 words), so one
thread owns a pair and writes whole words - byte-exact ggml layout in memory, no 16-bit-storage extension,
no cross-thread word races; Q8_K's 292 B blocks are word-aligned by themselves. When "the kernel did not
write it" must be told apart from "the kernel wrote it wrong", **stage a sentinel into the download buffer
and upload it** (`make_rw_slot` + memset 0xA5): unwritten regions come back A5 instead of undefined memory.

GLSL/host gotchas, small but each cost a cycle: `active` is reserved (same HLSL legacy family as `cb`/`sb`);
`intBitsToFloat` wants an `int`, not the `uint` you have; `windows.h`'s min/max macros poison `std::min` -
`#define NOMINMAX` before including anything that pulls it in.

Bandwidth with its shape: quantize-only dispatch at n=131072 (Q8_0): ~13 GB/s; at n=524288 (Q8_K): ~9 GB/s -
both far below the GEMV ports because one thread per 32/256-element block with a serial inner loop is not
optimized for anything yet. That is a measurement to keep, not a win to claim.
