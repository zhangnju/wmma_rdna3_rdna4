# Wave Matrix Multiply-Accumulate (WMMA) Programming on AMD RDNA 3 and RDNA 4 GPUs

## Overview

AMD's RDNA 3 (GFX11) architecture introduced Wave Matrix Multiply-Accumulate (WMMA) — a hardware-accelerated matrix multiplication instruction set targeting AI and compute workloads. RDNA 4 (GFX12) takes this further by doubling throughput, simplifying the register layout, and adding new data types including FP8 and structured sparsity support.

This article walks through how WMMA works, how to write HIP kernels using WMMA compiler intrinsics, and what you need to change when moving from RDNA 3 to RDNA 4.

> Note: WMMA is a consumer GPU analog to the MFMA instructions found on AMD's datacenter CDNA architecture (MI-series accelerators). While both perform the same D = A × B + C matrix fused multiply-add, the intrinsic names, tile sizes, and register conventions differ significantly between the two families.

---

## Background: What Is WMMA?

WMMA stands for Wave Matrix Multiply-Accumulate. It computes:

```
D = A × B + C
```

where A, B, C, and D are small 2D matrix tiles, and the computation is performed cooperatively by all lanes in a wavefront simultaneously. Unlike regular GPU arithmetic where each thread independently processes scalar or vector data, a WMMA instruction is a *wavefront-level operation* — the entire wavefront works together to produce a single matrix output tile.

This cooperative model lets the hardware optimize data reuse, schedule operand fetches in parallel with math, and achieve peak arithmetic throughput, very similar to NVIDIA's Tensor Core approach.

### GEMM Notation

The tile operation follows the GEMM convention M×N×K:

| Matrix | Shape | Role |
|--------|-------|------|
| A      | M × K | Left operand |
| B      | K × N | Right operand |
| C / D  | M × N | Accumulator input / output |

On both RDNA 3 and RDNA 4, only 16×16 tile sizes are supported, meaning M = N = K = 16 for all WMMA instructions. Larger matrices must be decomposed into 16×16 sub-tiles.

### Memory Layout Conventions

| Matrix | Layout |
|--------|--------|
| A      | Column-major |
| B      | Row-major |
| C / D  | Row-major |

---

## RDNA 3 (GFX11) WMMA

### Supported Data Types

RDNA 3 supports the following WMMA type combinations:

| Input (A, B) | Accumulator (C, D) | Intrinsic Suffix |
|--------------|--------------------|------------------|
| FP16         | FP16               | `f16_16x16x16_f16` |
| FP16         | FP32               | `f32_16x16x16_f16` |
| BF16         | FP32               | `f32_16x16x16_bf16` |
| BF16         | BF16               | `bf16_16x16x16_bf16` |
| INT8         | INT32              | `i32_16x16x16_iu8` |
| INT4         | INT32              | `i32_16x16x16_iu4` |

### Wavefront Mode: Wave32 vs Wave64

**Wave32** and **Wave64** are *wavefront sizes*: how many threads (lanes) in a group execute the same instruction in lockstep, analogous to a CUDA warp. Wave32 uses 32 lanes per wave; Wave64 uses 64. RDNA 3 WMMA can be issued in either mode via `*_w32` vs `*_w64` compiler intrinsics. Wave32 is the usual default setting in RDNA 3 sample codes and documents. 

when programming wmma through Wave32 or Wave64, developers need to take the below factors into consideration:

- Occupancy and launch shape: Wider waves change how many waves can be resident given fixed register and SIMD resources, so peak occupancy is not automatically better with one width or the other—it depends on your kernel. It is good practice to size thread blocks as a multiple of 32 or 64, so you do not leave most of a final wave idle. 

- Control flow and wave-scoped operations: Branches that diverge *within* a wave still execute both paths under hardware masking; wider waves mean more lanes can be affected by a single divergent region. Intrinsics and patterns that are defined over “the current wave” (e.g. ballot- or shuffle-style operations) also depend on wave width: a Wave64 kernel’s wave has 64 participating lanes, not 32.

- Consistency: Within one kernel, wave mode must be consistent, please do not mix `*_w32` and `*_w64` WMMA builtins in the same compilation unit for the same entry point. HIP/Clang options and attributes can select or hint wave mode.

- Fragment and VGPR:A fragment (`*_frag`) is the per-lane slice of A/B/C/D register image placed in VGPRs for one WMMA; rocWMMA and CUDA WMMA use the same term. Wave32 and Wave64 have the same VGPR number for A/B fraement, but different size for C/D. 
For FP16/BF16 inputs, each lane’s `A_frag` and `B_frag` each hold 16 half-precision elements packed into VGPRs, which means 8 VGPRS per lane for Wave32 and Wave64. Accordingly,each `A_frag` and `B_frag` needs 4 VGPRS per lane for Int8 input,and 2 VGPRs per lane for Int4 datatype. How many 32-bit VGPRs each lane needs for `C_frag` / `D_frag` depends on the accumulator element type and wave mode. FP32/INT32 accumulation needs 8 VGPRs per lane in Wave32, 4 in Wave64. Packed FP16 / BF16 accumulation needs 4 VGPRs per lane in Wave32, 2 in Wave64. 

### RDNA 3 Register Layout

**Register layout** here means how each matrix element in the WMMA tile maps to a specific lane and VGPR, including `A`/`B` replication across lanes and how `C`/`D` rows and columns bind to `thread`/`lane` — in other words, which thread loads which memory element before the intrinsic and which thread stores which element afterward at the hardware register level.

We take Wave32 FP16 input/FP32 accumulator as an example to introduce RDNA3 WMMA layout.

In Wave32 FP16 WMMA there are 32 threads, but A and B only need 16 different operand inputs; each input is duplicated on lanes *i* and *i*+16 (same registers on both). Number those setups t = 0…15: t is one row index into the A tile and one column index into the B tile, and you use the same t for both.This A/B replication quirk is eliminated on RDNA 4 — a key improvement discussed below.

With A column-major (`A[m,k]` at `k*16+m`) and B row-major (`B[k,n]` at `k*16+n`), AMD GPUOpen and Composable Kernel tests use:

- `a_frag[k]` = A[t,k] for k = 0…15 — row t of the A tile.
- `b_frag[k]` = B[k,t] for k = 0…15 — column t of the B tile.


For FP32 C and D (accumulator), the thirty-two threads fill a 16×16 output tile by assigning two threads to each output column. One thread supplies eight accumulators for the even rows in that column, the other for the odd rows, so the column is complete with no overlap. In the usual block ordering, those two threads sit sixteen positions apart. After the instruction, scatter results in row-major form—each full row of the tile laid out left to right before the next row. The sample *samples/wmma_rdna3_fp16.cpp* and Composable Kernel WMMA tests use the same convention.


### Intrinsic Syntax

Reading left to right, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` means: `f32` is the `C`/`D` (accumulator / result) element type, `16x16x16` is the WMMA tile shape (M×N×K), `f16` is the `A`/`B` input element type, and `_w32` / `_w64` is the wave mode. Integer variants swap the type fields (e.g. `i32_16x16x16_iu8`) and add control bits (`neg_a`, `neg_b`, `clamp`). 

For FP32 accumulator (e.g. `f32_16x16x16_f16`, `f32_16x16x16_bf16`), Clang expects only three arguments:

```c
D_frag = __builtin_amdgcn_wmma_f32_16x16x16_<AB_type>_w<32|64>(
    A_frag, B_frag, C_frag);
```

For FP16 / BF16 accumulator (e.g. `f16_16x16x16_f16`), a fourth `OPSEL` argument selects low vs high 16 bits in packed result VGPRs (`false` / `true`).

```c
D_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w<32|64>(
    A_frag, B_frag, C_frag, OPSEL);
```

Integer WMMA variants take `neg_a` / `neg_b` / `clamp` and packed operands. `i32_16x16x16_iu8` uses `int32x4` (`__builtin_bit_cast` from 16×`int8` ,see the sample code in [Composable Kernel](https://github.com/ROCm/composable_kernel/blob/develop/include/ck/utility/amd_wmma.hpp#L127-L137)). `i32_16x16x16_iu4` uses `int32x2` (16 int4 nibbles in two `int32`s; `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`). CK uses `neg_a` and `neg_b` both `true` for Wave32 iu8; iu4 matches that in [`builtin_wmma_naive_selector`](https://github.com/ROCm/composable_kernel/blob/develop/test/wmma_op/wmma_op_util.hpp#L84-L96) when built with `CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4`. Operand staging follows the [`matmul code`](https://github.com/ROCm/composable_kernel/blob/develop/test/wmma_op/wmma_op_util.hpp#L98-L192) shared-memory swizzle, not FP16-style direct global loads. Composable Kernel’s WMMA op test stores **A row-major M×K** and **B column-major K×N** in global memory; this repo’s **`samples/wmma_rdna3_iu8.cpp`** and **`samples/wmma_rdna3_iu4.cpp`** instead store **A column-major** (`A[m,k]` at `k*16+m`) and **B row-major** (`B[k,n]` at `k*16+n`) like the FP16 sample, and map global loads so the bytes written to shared memory match CK’s staging. Developers can refer to these sample codes in their projects. 

### Example: FP16 Input, FP32 Output (Wave32, RDNA 3)

This example implemented a 16×16×16 GEMM with FP16 inputs and FP32 accumulators using RDNA 3 WMMA intrinsics, plus a host CPU reference to check the GPU result.

The device kernel function loads 16-wide FP16 fragments for A and B, an 8-float piece of C per thread (split across the two 16-lane halves of the wave), then calls wmma intrinsic and stores reslutes to D.

It has neem tested on ROCm 7.2 and an RDNA 3 GPU.

```bash
hipcc --offload-arch=gfx1100 samples/wmma_rdna3_fp16.cpp -o wmma_rdna3_fp16
./wmma_rdna3_fp16
```


### Example: INT8 input, INT32 output (Wave32, RDNA 3)

The INT8 WMMA path does not mirror the FP16-input case’s simple pattern of loading full operands straight from global memory into fragments. In **`samples/wmma_rdna3_iu8.cpp`**, **A** is stored column-major (**M×K**), **B** row-major (**K×N**), and **C**/**D** row-major (**M×N**). Each thread first reads eight adjacent 8-bit values from **A** and **B** using a lane-dependent pattern (two halves of the wave cover complementary strips along **K** for a fixed row of **A** and column of **B**). Those values are written to shared memory and permuted with the same swizzle as Composable Kernel’s [`matmul`](https://github.com/ROCm/composable_kernel/blob/develop/test/wmma_op/wmma_op_util.hpp#L98-L192) helper before the wave issues the RDNA 3 Wave32 iu8 WMMA builtin with both operand-negate flags set, as in [`builtin_wmma_naive_selector` for 8-bit operands](https://github.com/ROCm/composable_kernel/blob/develop/test/wmma_op/wmma_op_util.hpp#L72-L82). Composable Kernel’s WMMA op test still uses **A** row-major and **B** column-major in global memory; this sample’s layout is an equivalent choice: after staging, the wave sees the same operand bytes as in CK’s pipeline. **C** and **D** use the same accumulator layout as the FP16-input / FP32-output example—two threads per output column, splitting even and odd rows. The host reference multiplies and accumulates with wider integer range, then rounds the final tile to 32-bit integers for comparison.

Source file: `samples/wmma_rdna3_iu8.cpp`.

```bash
hipcc --offload-arch=gfx1100 -std=c++20 samples/wmma_rdna3_iu8.cpp -o wmma_rdna3_iu8
./wmma_rdna3_iu8
```

### Example: INT4 input, INT32 output (Wave32, RDNA 3)

The INT4 walkthrough in **samples/wmma_rdna3_iu4.cpp** reuses the same big picture as the INT8 sample: **A** is column-major (**M×K**), **B** row-major (**K×N**), **C** and **D** row-major (**M×N**); operands still pass through the same shared-memory reordering and the same style of integer WMMA with negate flags. What changes is precision: **A** and **B** are held in ordinary byte arrays, but each matrix entry is really a **4-bit signed** value (−8 through 7). Only the **lower four bits** of each byte are used when operands are packed; the upper four bits should be treated as unused padding. Each lane first gathers sixteen such 4-bit values (still represented as sixteen bytes in registers), then the sample **packs those sixteen nibbles into two 32-bit chunks** in a specific order before issuing the RDNA 3 Wave32 INT4 WMMA. If your chip or compiler expects a different nibble order inside those two words, adjust the packing step so it agrees with Clang and the ISA reference.

Source file: `samples/wmma_rdna3_iu4.cpp`.


```bash
hipcc --offload-arch=gfx1100 -std=c++20 samples/wmma_rdna3_iu4.cpp -o wmma_rdna3_iu4
./wmma_rdna3_iu4
```

### Example: FP16 input, FP16 / BF16 accumulators (Wave32, RDNA 3)

The earlier samples keep **C** and **D** as full FP32 tiles. RDNA 3 also offers WMMA variants where the accumulator and result stay in **packed FP16 or BF16**: two half-precision values share each 32-bit register lane. Those builtins add a **fourth selector** after the three operand fragments: one setting uses the **lower 16 bits** of each 32-bit slot for matrix elements, the other uses the **upper 16 bits** ([GPUOpen — WMMA on RDNA 3](https://gpuopen.com/learn/wmma_on_rdna3)).

Loading **A** and **B** follows the same rules as the FP32-accumulator example: **A** is column-major, each lane responsible for one row index; **B** is row-major, each lane for one column index. **C** and **D** still assign **eight matrix positions per lane**, split between even and odd rows within each output column. When using the “lower 16 bits” mode, each of those eight values occupies the **first half** of its paired register slot; results are read back the same way and written to **D** in row-major order with the same row/column mapping as before.


```bash
hipcc --offload-arch=gfx1100 -O2 samples/wmma_rdna3_f16_bf16_acc.cpp -o wmma_rdna3_f16_bf16_acc
./wmma_rdna3_f16_bf16_acc
```

---

## RDNA 4 (GFX12) WMMA

RDNA 4 introduces 3rd-generation Matrix Cores with several improvements:

| Metric | RDNA 3 | RDNA 4 |
|--------|--------|--------|
| FP16 / BF16 FLOPS per clock per CU | 256 | 512 (2×) |
| INT8 FLOPS per clock per CU | 256 | 1024 (4×) |
| FP8 support | No | Yes (E4M3 / E5M2) |
| Structured sparsity | No | Yes (4:2, via SWMMAC) |
| Register duplication | Yes (A/B lanes 16–31) | Eliminated |
| VGPR count per lane (FP16, Wave32) | 16×`half` logical (8 VGPR pairs) | 8×`half` as `half8_t` (4 VGPR pairs); no mirror lanes |

> Breaking Change: RDNA 4 WMMA intrinsics are not backward compatible with RDNA 3 intrinsics. All RDNA 4 intrinsics carry a `_gfx12` suffix and use a different operand and accumulator lane map (see below). LLVM also exposes `_w64_gfx12` variants for Wave64 mode; the examples here use Wave32.

### Register layout: no lane mirroring (RDNA 4)

The main improvement over RDNA 3 is removing redundant A/B replication: on GFX11, lanes 16–31 carried the same A/B operands as lanes 0–15. On GFX12, each lane holds eight unique FP16 values (`half8_t`) with no mirror lanes.

```
RDNA 3 (Wave32) — A/B operands (conceptual):
  Lanes  0–15: primary data
  Lanes 16–31: DUPLICATE of lanes 0–15   ← wasted fetch / register traffic

RDNA 4 (Wave32) — A/B operands:
  Lanes 0–31: eight unique FP16 values per lane (no mirror)
```

Chaining WMMA (e.g. MLP) still needs a layout transform: on gfx12, the layout used for the A and B operands is not the same as the layout of the D accumulator (see formulas below). You normally write D to shared or global memory in row-major form, then reload the next operand with the correct mapping, or use rocWMMA / Composable Kernel. See [ROCm issue #6025](https://github.com/ROCm/ROCm/issues/6025).

### New Data Types in RDNA 4

RDNA 4 adds FP8 and BF8 (brain float 8-bit) WMMA instructions:

| Input (A, B) | Accumulator (C, D) | Intrinsic Suffix (gfx12) |
|--------------|--------------------|--------------------------|
| FP16         | FP16               | `f16_16x16x16_f16_w32_gfx12` |
| BF16         | BF16               | `bf16_16x16x16_bf16_w32_gfx12` |
| FP16         | FP32               | `f32_16x16x16_f16_w32_gfx12` |
| BF16         | FP32               | `f32_16x16x16_bf16_w32_gfx12` |
| FP8 (E4M3)   | FP32               | `f32_16x16x16_fp8_fp8_w32_gfx12` |
| BF8 (E5M2)   | FP32               | `f32_16x16x16_bf8_bf8_w32_gfx12` |
| Mixed FP8/BF8| FP32               | `f32_16x16x16_fp8_bf8_w32_gfx12` |
| INT8         | INT32              | `i32_16x16x16_iu8_w32_gfx12` |
| INT4         | INT32              | `i32_16x16x16_iu4_w32_gfx12` |
| INT4         | INT32              | `i32_16x16x32_iu4_w32_gfx12` |

Additionally, SWMMAC (Sparse Wave Matrix Multiply-Accumulate) instructions exploit 4:2 structured sparsity for a further 2× throughput boost.

### RDNA 4 Intrinsic Syntax

Compared to RDNA 3, gfx12 drops `OPSEL` for packed FP16 and BF16 accumulator results (e.g. `f16_16x16x16_f16_*`, `bf16_16x16x16_bf16_*`): the output is always densely packed. 

For the floating-point WMMA family, the Wave32 pattern is:

```c
D_frag = __builtin_amdgcn_wmma_<CD_type>_16x16x16_<AB_type>_w32_gfx12(
    A_frag,   // vector of A tile elements
    B_frag,   // vector of B tile elements
    C_frag    // vector of C (accumulator) elements
);
```

For the integer type WMMA family, the builtin name always uses int32 data type for the accumulator. A Wave32 call passes 6 arguments in this order: `negate A` (boolean), `packed A`, `negate B` (boolean), `packed B`, C accumulator, `clamp result` (boolean). Those booleans are what Clang/LLVM calls `i1`. Packed here means each lane’s A and B carry several 8- or 4-bit matrix elements inside one or two int32 registers, instead of FP WMMA’s usual pattern (eight half floats per lane) in one vector for A and again for B.

```c
D_frag = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
    neg_a, A_frag,   // i1, then packed UINT8 A (<2 x i32> per lane)
    neg_b, B_frag,   // i1, then packed UINT8 B (<2 x i32> per lane)
    C_frag,          // <8 x i32> accumulator
    clamp            // i1: saturate / clamp result
);

// 16×16×16 INT4: one i32 of packed nibbles per lane for A and B
D_frag = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(
    neg_a, A_i32, neg_b, B_i32, C_frag, clamp);

// 16×16×32 INT4: <2 x i32> packed operands per lane (wider K)
D_frag = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
    neg_a, A_frag, neg_b, B_frag, C_frag, clamp);
```

### Example 1: FP16 Input, FP32 Output (Wave32, RDNA 4)

This example implements a 16×16×16 tile GEMM `D = A×B + C` with FP16 inputs and FP32 accumulators using RDNA 4 (gfx12) WMMA intrinsics, plus a host CPU reference to verify the GPU result—same problem shape as **`samples/wmma_rdna3_fp16.cpp`**, but with the Wave32 register map and builtin names for GFX12.

The device kernel `wmma_gemm_rdna4` launches one block of 32 threads (one wave). Each lane first gathers eight half-precision values into `half8_t` fragments for A and B using the gfx12 lane indexing (the low part of the lane id picks the M row for A and the N column for B; lanes 0-15 cover K=0-7, lanes 16-31 cover K=8-15, and index e chooses one value inside that 8-value K range). It then loads eight floats of C per lane—the accumulator layout matches D, with the eight slots walking M within the tile and the lane id modulo 16 picking N. The kernel calls `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, then writes D with the same scatter pattern as C. There is no `OPSEL` argument and no duplicate A/B lanes: every lane contributes distinct operand data, unlike RDNA 3.

On the host, `cpu_gemm_16x16_ref` recomputes the tile in `long double` for a tight numerical check, compares against the GPU D, prints sample and max absolute error, and returns a non-zero exit code if max error exceeds 1e-2.

Source file: `samples/wmma_rdna4_fp16.cpp`.

It has been tested on ROCm 7.2 and an RDNA 4 GPU (e.g. RX 9070 XT).

```bash
hipcc --offload-arch=gfx1201 samples/wmma_rdna4_fp16.cpp -o wmma_rdna4_fp16
./wmma_rdna4_fp16
```

### Example 2: FP8 Input, FP32 Output (RDNA 4 Only)

This example implements the same 16×16×16 fused multiply-add D = A×B + C with FP32 C/D, but A and B are stored as raw E4M3 FP8 bytes. FP8 WMMA exists only on RDNA 4.

The device kernel `wmma_gemm_rdna4_fp8` uses `pack_fp8_A_col` and `pack_fp8_B_row` to collect eight FP8 bytes per lane from global memory with the same (m, k) and (k, n) indexing as the FP16 sample, then packs them into `int32x2` (a small union of two 32-bit words and eight bytes) for `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`. C load and D store follow the same eight-float-per-lane accumulator map as `wmma_rdna4_fp16.cpp`.

Source file: **`samples/wmma_rdna4_fp8.cpp`**.

It has been tested on ROCm 7.2 and an RDNA 4 GPU (e.g. RX 9070 XT).

```bash
hipcc --offload-arch=gfx1201 samples/wmma_rdna4_fp8.cpp -o wmma_rdna4_fp8
./wmma_rdna4_fp8
```

### Example 3: INT8 Input, INT32 Output (Wave32, RDNA 4)

This example implements D = A×B + C on a 16×16×16 tile with INT8 A/B and INT32 C/D, analogous to `samples/wmma_rdna3_iu8.cpp` but calling the gfx12 six-argument iu8 WMMA instead of the RDNA 3 path.

Unlike FP16 gfx12, iu8 operands are not a simple global gather per lane: the kernel repeats the same shared-memory swizzle as RDNA 3 to build `int8x16` A/B fragments, then splits them for gfx12 — threads 0–15 use bytes `[0..7]`, threads 16–31 use `[8..15]` — and `__builtin_bit_cast` those eight bytes to `int32x2`. C load and D store use the gfx12 accumulator map (same as `wmma_rdna4_fp16.cpp`). The builtin is `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`; the first two `bool` arguments are signedness (`true` = signed int8), not negation, with `clamp` `false`.

Source file: **`samples/wmma_rdna4_iu8.cpp`**. Compile with **C++20** so **`__builtin_bit_cast`** is available.

It has been tested on ROCm 7.2 and an RDNA 4 GPU (e.g. RX 9070 XT).

```bash
hipcc --offload-arch=gfx1201 -std=c++20 samples/wmma_rdna4_iu8.cpp -o wmma_rdna4_iu8
./wmma_rdna4_iu8
```

### Example 4: INT4 Input, INT32 Output (Wave32, RDNA 4)

`samples/wmma_rdna4_iu4.cpp` runs two GPU checks (K = 16 and K = 32) for D = A×B + C: int4 in the low nibble of `int8_t`, int32 C/D, same idea as `wmma_rdna3_iu4.cpp`. `main` returns 0 only if both checks match the CPU.

The K = 16 path packs eight nibbles per lane into one `int32` (`pack_iu4_x8`) and calls `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12`. The K = 32 path uses A as 16×32 and B as 32×16 (reduction length K = 32), packs sixteen nibbles per lane into `int32x2` with `pack_iu4_x16`, and calls `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12`. C/D use the same gfx12 accumulator layout as the FP16 samples; `neg_a` / `neg_b` are true, `clamp` false.

Host reference: `cpu_gemm_i4_i32` with `sx_i4`, same as `wmma_rdna3_iu4.cpp`.

It has been tested on ROCm 7.2 and an RDNA 4 GPU (e.g. RX 9070 XT).

```bash
hipcc --offload-arch=gfx1201 -std=c++20 samples/wmma_rdna4_iu4.cpp -o wmma_rdna4_iu4
./wmma_rdna4_iu4
```

### Example 5: Chained WMMA for MLP Inference (RDNA 4)

On gfx12, the D accumulator layout is not the same as the B operand layout, so you cannot reuse the first WMMA’s register output as the second WMMA’s B fragment without reordering. A small shared-memory staging step (store row-major, reload with the B gather above) is typical. **`samples/mlp_wmma_rdna4.cpp`** implements `output = W2 * relu(W1 * input)` on 16×16 tiles: two `_gfx12` WMMAs, **`relu` in FP32**, hidden tile written to **`__half` shared memory** (matching the second layer’s FP16 B operand). The kernel uses a fresh zero accumulator for the second WMMA (`acc1`), not the variable passed to the first WMMA.

```bash
hipcc --offload-arch=gfx1201 samples/mlp_wmma_rdna4.cpp -o mlp_wmma_rdna4
./mlp_wmma_rdna4
```

## Using the Higher-Level rocWMMA Library

For applications that don't need to hand-tune intrinsic-level code, AMD provides rocWMMA — a C++ template library with an interface modeled after CUDA's `nvcuda::wmma` API. It handles register layout management automatically and supports both RDNA and CDNA architectures.

`samples/rocwmma_example.cpp` is a full HIP program: one 16×16×16 tile with `fragment<matrix_a, …, col_major>`, `fragment<matrix_b, …, row_major>`, FP32 accumulator, `D = A*B + C` with a non-zero **`C`** buffer (same layout as `samples/wmma_rdna4_fp16.cpp`). 

**rocWMMA 7.x note:** loading the accumulator from global memory requires an explicit layout on the `load_matrix_sync` overload, e.g. `load_matrix_sync(c_frag, C, ld, mem_row_major)` — the older three-argument form is not enough when `c_frag` has no static layout in the type.

```bash
hipcc --offload-arch=native samples/rocwmma_example.cpp -o rocwmma_example
./rocwmma_example
```

## Scaling Up: Tiled GEMM Beyond 16×16

Real workloads involve matrices far larger than 16×16. The standard approach is to decompose the problem into a grid of 16×16 tiles and accumulate partial products:

```
for each output tile (bm, bn):
    c_tile = 0
    for each reduction step bk:
        load A_tile = A[bm*16 : (bm+1)*16,  bk*16 : (bk+1)*16]
        load B_tile = B[bk*16 : (bk+1)*16,  bn*16 : (bn+1)*16]
        c_tile += wmma(A_tile, B_tile)
    write D[bm*16 : (bm+1)*16,  bn*16 : (bn+1)*16] = c_tile
```

In HIP, each thread block typically handles one or more output tiles, and each wavefront within the block is assigned exactly one WMMA operation per step.

`samples/tiled_gemm_rdna4.cpp` implements `D = A * B` for `M×K` `__half` A (column-major, stride `lda`), `K×N` B (row-major, `ldb`), `M×N` FP32 D (row-major, `ldd`), with `M, N, K` multiples of 16. Each block covers one 16×16 output tile: `grid(N/16, M/16)`, `block(32)`. Optional `M N K` CLI args; default `32×32×32`.

```bash
hipcc --offload-arch=gfx1201 samples/tiled_gemm_rdna4.cpp -o tiled_gemm_rdna4
./tiled_gemm_rdna4
./tiled_gemm_rdna4 64 48 32
```

---

## Side-by-Side Comparison: RDNA 3 vs RDNA 4

| Feature | RDNA 3 (GFX11) | RDNA 4 (GFX12) |
|---------|---------------|---------------|
| Architecture code | `gfx1100`–`gfx1102` (other GFX11 targets such as `gfx1150`, `gfx1151` also appear in ROCm) | `gfx1200`, `gfx1201` |
| Tile size | **16×16×16** for typical FP / INT8 WMMA | Same for most types; **INT4** on gfx12 also has larger-**K** shapes (e.g. **16×16×32**) — see [RDNA 4 (GFX12) WMMA](#rdna-4-gfx12-wmma) type table / ISA |
| Wavefront mode | Wave32 / Wave64 | Wave32 (examples); `_w64_gfx12` intrinsics also exist |
| FP16/BF16 FLOPS/clock/CU | 256 | 512 (per AMD arch disclosures; verify per SKU) |
| INT8 FLOPS/clock/CU | 256 | 1024 (per AMD arch disclosures; verify per SKU) |
| FP8 support | No | Yes (E4M3, E5M2) |
| Structured sparsity | No | Yes (4:2 SWMMAC) |
| Lane duplication (A/B) | Yes | No |
| Intrinsic suffix | (none) | `_gfx12` |
| `OPSEL` argument | FP16/BF16 accumulator output only (e.g. `f16_16x16x16_f16`); not used for `f32_16x16x16_*` (3-arg) | No |
| Backward compatible | — | No |
| Min ROCm version | ~5.4 (order-of-magnitude; check your ROCm release) | ~6.2+ (gfx12 WMMA / FP8 paths; check release notes) |
| Example intrinsic | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` |

---

## Choosing the Right Approach

| Use Case | Recommendation |
|----------|---------------|
| Maximum portability across RDNA 3 & 4 | rocWMMA library — handles layout differences automatically |
| Maximum performance, full control | Compiler intrinsics with architecture-specific code paths |
| Prototyping / research | Intrinsics with `#ifdef __gfx12__` / `#ifdef __gfx11__` guards |
| CUDA code porting | rocWMMA (API mirrors `nvcuda::wmma`) |

To compile for multiple targets simultaneously:

```bash
# Both RDNA3 and RDNA4 targets in one binary
hipcc --offload-arch=gfx1100 --offload-arch=gfx1201 my_kernel.cpp -o my_kernel
```

To detect the architecture at runtime (useful for choosing the right kernel):

```cpp
hipDeviceProp_t prop;
hipGetDeviceProperties(&prop, 0);
// prop.gcnArchName will be e.g. "gfx1100" or "gfx1201"
bool is_rdna4 = (strncmp(prop.gcnArchName, "gfx12", 5) == 0);
```

---

## Tools and Further Reading

### AMD Matrix Instruction Calculator

The [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) is an official tool that prints detailed WMMA instruction information for any supported architecture:

```bash
# Show FP16 WMMA details for RDNA3 (gfx1100)
python3 matrix_calculator.py --architecture gfx1100 \
    --instruction v_wmma_f32_16x16x16_f16 --detail-instruction

# Show FP8 WMMA details for RDNA4 (gfx1201)
python3 matrix_calculator.py --architecture gfx1201 \
    --instruction v_wmma_f32_16x16x16_fp8_fp8 --detail-instruction
```

Output includes: opcode, VGPR count, element mapping per lane, and peak throughput estimates.

### ISA Reference Guides

- [RDNA3 ISA Reference Guide (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
- [RDNA4 ISA Reference Guide (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)

---

## Summary

WMMA brings hardware matrix acceleration to AMD's consumer GPU line, enabling significant AI and compute performance gains without requiring a datacenter GPU. RDNA 3 introduced the foundation with FP16/BF16/INT8 support; RDNA 4 doubles down with higher throughput, a cleaner operand layout without mirror lanes, and new FP8 data types opening the door to ultra-low-precision inference. 

When writing WMMA code:

1. RDNA 3: use `__builtin_amdgcn_wmma_*_w32` or `*_w64` intrinsics; account for lane duplication in A/B.
2. RDNA 4: use `__builtin_amdgcn_wmma_*_w32_gfx12` (or `*_w64_gfx12`) with `half8_t` / `float8_t` and the m/k/n lane formulas above for loads and stores; add FP8 / SWMMAC where applicable. **iu8** uses six-argument builtins with `int32x2`-packed **A/B** per lane (`samples/wmma_rdna4_iu8.cpp`); **iu4** uses **scalar `int32`** **A/B** for **`…_16x16x16_iu4_…`** and **`int32x2`** for **`…_16x16x32_iu4_…`** (`samples/wmma_rdna4_iu4.cpp`).
3. Both: consider rocWMMA for portable, maintainable code.

---

## Sources

- [How to accelerate AI applications on RDNA 3 using WMMA — AMD GPUOpen](https://gpuopen.com/learn/wmma_on_rdna3/)
- [Using the Matrix Cores of AMD RDNA 4 architecture GPUs — AMD GPUOpen](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/)
- [AMD Matrix Instruction Calculator — GitHub](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [rocWMMA Programming Guide — ROCm Documentation](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/conceptual/programmers-guide.html)
- [RDNA3 ISA Reference Guide — AMD](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
- [RDNA4 ISA Reference Guide — AMD](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)
- [Examining AMD's RDNA 4 Changes in LLVM — Chips & Cheese](https://chipsandcheese.com/p/examining-amds-rdna-4-changes-in-llvm)
- [WMMA Benefits for ML and General Compute — AMD GPUOpen](https://gpuopen.com/news/wmma_benefits_ml_compute/)
- [ROCm issue #6025 — gfx12 WMMA output / lane mapping discussion](https://github.com/ROCm/ROCm/issues/6025)
