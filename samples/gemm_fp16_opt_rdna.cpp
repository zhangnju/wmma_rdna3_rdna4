// gemm_fp16_opt_rdna.cpp
// Compile (repo root):
//   W7900 / RDNA 3:  hipcc --offload-arch=gfx1100 -O3 -std=c++17 -mcumode \
//                      -Rpass-analysis=kernel-resource-usage samples/gemm_fp16_opt_rdna.cpp -o gemm_fp16_opt_rdna
//   R9700 / RDNA 4:  hipcc --offload-arch=gfx1201 -O3 -std=c++17 -mcumode \
//                      -Rpass-analysis=kernel-resource-usage samples/gemm_fp16_opt_rdna.cpp -o gemm_fp16_opt_rdna
//
// Run:  ./gemm_fp16_opt_rdna [M N K] [reps]      (defaults: 4096 4096 4096, 10 reps)
//       ./gemm_fp16_opt_rdna 8192 8192 8192 20   (blog size; note throughput peaks near 4096)
//
// -----------------------------------------------------------------------------
// FP16 GEMM optimization walkthrough for AMD RDNA 3 (gfx11) and RDNA 4 (gfx12),
// following the ROCm blog "FP16 GEMM Optimization on AMD RDNA 3 and RDNA 4".
// A single kernel is improved across eight steps; the driver benchmarks each and
// prints ms / GFLOP-s / correctness so the effect of every change is visible.
//
// Problem:  D = A * B^T   (the WMMA-friendly, transpose-free pairing)
//   A : M x K  row-major  (K contiguous)   -> A[m*K + k]
//   B : N x K  row-major  (K contiguous)   -> B[n*K + k]
//   D : M x N  row-major                   -> D[m*N + n] = sum_k A[m,k] * B[n,k]
// Storing both operands K-contiguous lets each lane walk K with consecutive
// addresses, which is what makes the 128-bit wide loads (step 2+) possible.
//
// The eight steps (each flips one knob on the templated kernel gemm_opt<>):
//   1  baseline      one wave owns a 32x32 tile; scalar LDS copies; barrier / 16 K
//   2  tiled+DB      128x128 tile, 16 waves, 128-bit loads, double-buffered LDS
//   3  +padding      pad LDS rows to break shared-memory bank conflicts
//   4  +OOB-safe     guarded wide loads so M,N need not divide the tile (blog: buffer_rsrc)
//   5  +deep-K       block_k 16 -> 32: more work per barrier, fewer syncs
//   6  +scheduling   grouped block launch so neighbours share A/B in L2
//   7  +prefetch     stage next tile in registers during WMMA; s_setprio around the MMA
//   8  single-buffer drop the 2nd LDS buffer (half the LDS, ~2x occupancy) + 1 barrier
//
// WMMA differs by arch (handled in the helpers below):
//   RDNA 3: 16 fp16 per A/B fragment (K replicated across the wave's two halves)
//   RDNA 4: 8  fp16 per A/B fragment (lane<16 holds K 0-7, lane>=16 holds K 8-15)
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// 128-bit load/store unit: 8 contiguous fp16.
typedef __attribute__((ext_vector_type(8))) _Float16 half8;

// WMMA fragment widths differ between RDNA 3 and RDNA 4.
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1202__)
  #define ARCH_RDNA4 1
  typedef __attribute__((ext_vector_type(8)))  _Float16 AFrag;  // 8 half / lane
  typedef __attribute__((ext_vector_type(8)))  _Float16 BFrag;
#else
  // RDNA 3 (gfx11) and the host-side compile pass.
  typedef __attribute__((ext_vector_type(16))) _Float16 AFrag;  // 16 half / lane
  typedef __attribute__((ext_vector_type(16))) _Float16 BFrag;
#endif
typedef __attribute__((ext_vector_type(8))) float Acc;          // 8 float / lane (both archs)

// ---------------------------------------------------------------------------
// Device helpers: arch-specific WMMA + fragment gather/scatter
// ---------------------------------------------------------------------------
__device__ __forceinline__ half8 zero8() {
    half8 z;
#pragma unroll
    for (int i = 0; i < 8; i++) z[i] = (_Float16)0;
    return z;
}

__device__ __forceinline__ Acc wmma_f16(AFrag a, BFrag b, Acc c) {
#ifdef ARCH_RDNA4
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}

// Gather one 16x16 A fragment from an LDS tile laid out [row][k] with stride `lda`.
// `rowblk` selects the 16-row block, `kblk` the 16-wide K block.
__device__ __forceinline__ AFrag load_a_lds(const _Float16* lds, int lda, int rowblk, int kblk) {
    const int laneW = threadIdx.x & 31;
    const int mn = laneW & 15;
    const _Float16* base = lds + (rowblk * 16 + mn) * lda + kblk * 16;
    AFrag f;
#ifdef ARCH_RDNA4
    const int koff = (laneW >> 4) * 8;  // lane<16 -> K 0-7, lane>=16 -> K 8-15
#pragma unroll
    for (int e = 0; e < 8; e++) f[e] = base[koff + e];
#else
#pragma unroll
    for (int k = 0; k < 16; k++) f[k] = base[k];  // RDNA 3: all 16 K per lane
#endif
    return f;
}

// B tile is laid out [col][k] (B is N x K), so the gather is symmetric with A.
__device__ __forceinline__ BFrag load_b_lds(const _Float16* lds, int lda, int colblk, int kblk) {
    const int laneW = threadIdx.x & 31;
    const int mn = laneW & 15;
    const _Float16* base = lds + (colblk * 16 + mn) * lda + kblk * 16;
    BFrag f;
#ifdef ARCH_RDNA4
    const int koff = (laneW >> 4) * 8;
#pragma unroll
    for (int e = 0; e < 8; e++) f[e] = base[koff + e];
#else
#pragma unroll
    for (int k = 0; k < 16; k++) f[k] = base[k];
#endif
    return f;
}

// Row within a 16x16 output tile that accumulator slot `e` maps to (col = lane%16).
__device__ __forceinline__ int acc_row16(int e, int laneW) {
#ifdef ARCH_RDNA4
    return (laneW >> 4) * 8 + e;       // lane group 0 -> rows 0-7, group 1 -> rows 8-15
#else
    return 2 * e + (laneW >> 4);       // RDNA 3: interleaved even/odd rows
#endif
}

// ---------------------------------------------------------------------------
// Step 1 — baseline: one wave (32 threads) owns a 32x32 output tile
//          (a 2x2 grid of 16x16 WMMA tiles). Scalar LDS copies, block_k = 16,
//          two barriers per 16 K values. Everything else is a refinement of this.
// ---------------------------------------------------------------------------
__global__ void gemm_step1(const __half* __restrict__ A, const __half* __restrict__ B,
                           float* __restrict__ D, int M, int N, int K) {
    __shared__ _Float16 As[32][16];
    __shared__ _Float16 Bs[32][16];
    const int laneW = threadIdx.x;              // 0..31 (one wave)
    const int baseM = blockIdx.y * 32;
    const int baseN = blockIdx.x * 32;

    Acc acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) acc[i][j][e] = 0.f;

    for (int kt = 0; kt < K; kt += 16) {
        // Each lane copies its own row of the 32x16 A and B tiles, one element at a time.
#pragma unroll
        for (int k = 0; k < 16; k++) {
            As[laneW][k] = (baseM + laneW < M) ? (_Float16)A[(long)(baseM + laneW) * K + kt + k] : (_Float16)0;
            Bs[laneW][k] = (baseN + laneW < N) ? (_Float16)B[(long)(baseN + laneW) * K + kt + k] : (_Float16)0;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                acc[i][j] = wmma_f16(load_a_lds(&As[0][0], 16, i, 0),
                                     load_b_lds(&Bs[0][0], 16, j, 0), acc[i][j]);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int row = baseM + i * 16 + acc_row16(e, laneW);
                const int col = baseN + j * 16 + (laneW & 15);
                if (row < M && col < N) D[(long)row * N + col] = acc[i][j][e];
            }
}

// Cooperative 128-bit global->LDS copy of an `nrow x BK` tile (K contiguous).
// GUARD=false assumes the tile is fully in-bounds (steps 2,3); GUARD=true
// zero-fills out-of-range rows so M,N need not divide the tile (step 4+).
template <bool GUARD>
__device__ __forceinline__ void load_tile(_Float16* lds, int lda, const __half* G,
                                          int baseRow, int rowMax, int K, int col0, int nrow, int BK) {
    const int CH = BK / 8;
    const int n = nrow * CH;
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        const int row = idx / CH;
        const int k8 = (idx % CH) * 8;
        half8 v;
        if (!GUARD || (baseRow + row) < rowMax)
            v = *reinterpret_cast<const half8*>(&G[(long)(baseRow + row) * K + col0 + k8]);
        else
            v = zero8();
        *reinterpret_cast<half8*>(&lds[row * lda + k8]) = v;
    }
}

// ---------------------------------------------------------------------------
// Steps 2-8 — one templated kernel; each step flips a template knob:
//   BM,BN,BK  output tile + K-slice depth
//   PAD       LDS row padding (bank-conflict avoidance, step 3)
//   DB        double-buffered LDS (step 2; dropped in step 8)
//   SCHED     grouped block scheduling for L2 reuse (step 6)
//   PF        prefetch next tile into registers + s_setprio around WMMA (step 7)
//   OOB       guarded loads/stores for non-divisible sizes (step 4)
// Wave tiling is fixed at 32x32 per wave (2x2 WMMA tiles); 16 waves per 128x128 block.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int PAD, bool DB, bool SCHED, bool PF, bool OOB>
__global__ __launch_bounds__(BM * BN / 32) void gemm_opt(const __half* __restrict__ A,
                                                         const __half* __restrict__ B,
                                                         float* __restrict__ D, int M, int N, int K) {
    constexpr int LDA = BK + PAD;        // padded LDS row stride (in fp16 elements)
    constexpr int NBUF = DB ? 2 : 1;
    __shared__ _Float16 As[NBUF][BM * LDA];
    __shared__ _Float16 Bs[NBUF][BN * LDA];

    const int tid = threadIdx.x;
    const int laneW = tid & 31;
    const int wid = tid >> 5;
    constexpr int WN = BN / 32;          // waves across a block row
    const int wr = wid / WN;             // wave row / col within the block
    const int wc = wid % WN;

    // Block -> (block-row, block-col). Step 6 remaps 1D launch into 8-row groups
    // so concurrently-running blocks overlap in A/B and stay hot in L2.
    int bm, bn;
    if (SCHED) {
        const int nbn = (N + BN - 1) / BN;
        const int nbm = (M + BM - 1) / BM;
        constexpr int G = 8;
        const int per_group = G * nbn;
        const int grp = blockIdx.x / per_group;
        const int loc = blockIdx.x % per_group;
        const int grp_rows = (nbm - grp * G) < G ? (nbm - grp * G) : G;
        bm = grp * G + loc % grp_rows;
        bn = loc / grp_rows;
    } else {
        bm = blockIdx.y;
        bn = blockIdx.x;
    }
    const int baseM = bm * BM;
    const int baseN = bn * BN;

    Acc acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) acc[i][j][e] = 0.f;

    int cur = 0;

    if (PF) {
        // ---- step 7: register-staged prefetch (one A + one B chunk per thread) ----
        // Requires blockDim == BM*(BK/8); holds for the 128x128, block_k=32 config.
        const int CH = BK / 8;
        const int aRow = tid / CH, aK8 = (tid % CH) * 8;
        const int bRow = tid / CH, bK8 = (tid % CH) * 8;

        half8 ra = (!OOB || (baseM + aRow) < M)
                       ? *reinterpret_cast<const half8*>(&A[(long)(baseM + aRow) * K + aK8]) : zero8();
        half8 rb = (!OOB || (baseN + bRow) < N)
                       ? *reinterpret_cast<const half8*>(&B[(long)(baseN + bRow) * K + bK8]) : zero8();
        *reinterpret_cast<half8*>(&As[cur][aRow * LDA + aK8]) = ra;
        *reinterpret_cast<half8*>(&Bs[cur][bRow * LDA + bK8]) = rb;
        __syncthreads();

        for (int kt = 0; kt < K; kt += BK) {
            const int nxt = kt + BK;
            if (nxt < K) {  // issue next tile's global loads into registers *before* the MMA
                ra = (!OOB || (baseM + aRow) < M)
                         ? *reinterpret_cast<const half8*>(&A[(long)(baseM + aRow) * K + nxt + aK8]) : zero8();
                rb = (!OOB || (baseN + bRow) < N)
                         ? *reinterpret_cast<const half8*>(&B[(long)(baseN + bRow) * K + nxt + bK8]) : zero8();
            }
            __builtin_amdgcn_s_setprio(1);  // bias issue toward the matrix op
#pragma unroll
            for (int ks = 0; ks < BK / 16; ks++)
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        acc[i][j] = wmma_f16(load_a_lds(As[cur], LDA, wr * 2 + i, ks),
                                             load_b_lds(Bs[cur], LDA, wc * 2 + j, ks), acc[i][j]);
            __builtin_amdgcn_s_setprio(0);
            if (nxt < K) {  // commit the staged registers into LDS
                const int nb = DB ? (cur ^ 1) : cur;
                if (!DB) __syncthreads();  // single buffer: wait for all reads of As[cur]
                *reinterpret_cast<half8*>(&As[nb][aRow * LDA + aK8]) = ra;
                *reinterpret_cast<half8*>(&Bs[nb][bRow * LDA + bK8]) = rb;
                __syncthreads();
                cur = nb;
            }
        }
    } else {
        // ---- steps 2-6, 8: cooperative double/single-buffered pipeline ----
        load_tile<OOB>(As[cur], LDA, A, baseM, M, K, 0, BM, BK);
        load_tile<OOB>(Bs[cur], LDA, B, baseN, N, K, 0, BN, BK);
        __syncthreads();

        for (int kt = 0; kt < K; kt += BK) {
            const int nxt = kt + BK;
#pragma unroll
            for (int ks = 0; ks < BK / 16; ks++)
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        acc[i][j] = wmma_f16(load_a_lds(As[cur], LDA, wr * 2 + i, ks),
                                             load_b_lds(Bs[cur], LDA, wc * 2 + j, ks), acc[i][j]);
            if (nxt < K) {
                const int nb = DB ? (cur ^ 1) : cur;
                if (!DB) __syncthreads();  // single buffer: reads must finish before overwrite
                load_tile<OOB>(As[nb], LDA, A, baseM, M, K, nxt, BM, BK);
                load_tile<OOB>(Bs[nb], LDA, B, baseN, N, K, nxt, BN, BK);
                __syncthreads();
                cur = nb;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int row = baseM + (wr * 2 + i) * 16 + acc_row16(e, laneW);
                const int col = baseN + (wc * 2 + j) * 16 + (laneW & 15);
                if (row < M && col < N) D[(long)row * N + col] = acc[i][j][e];
            }
}

// ---------------------------------------------------------------------------
// Host side: step registry + benchmark driver
// ---------------------------------------------------------------------------
#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at line %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

typedef void (*StepFn)(const __half*, const __half*, float*, int, int, int);

static void launch_step1(const __half* A, const __half* B, float* D, int M, int N, int K) {
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    gemm_step1<<<grid, 32>>>(A, B, D, M, N, K);
}
// Steps 2-5, 8 use a 2D grid; steps 6-7 use a 1D grid (block scheduling).
template <int BK, int PAD, bool DB, bool PF, bool OOB>
static void launch_grid2d(const __half* A, const __half* B, float* D, int M, int N, int K) {
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    gemm_opt<128, 128, BK, PAD, DB, /*SCHED=*/false, PF, OOB><<<grid, 512>>>(A, B, D, M, N, K);
}
template <int BK, int PAD, bool DB, bool PF, bool OOB>
static void launch_grid1d(const __half* A, const __half* B, float* D, int M, int N, int K) {
    const int nblk = ((M + 127) / 128) * ((N + 127) / 128);
    gemm_opt<128, 128, BK, PAD, DB, /*SCHED=*/true, PF, OOB><<<nblk, 512>>>(A, B, D, M, N, K);
}

struct Step { const char* name; StepFn fn; };
static const Step kSteps[] = {
    {"1 baseline        ", launch_step1},
    {"2 tiled+dbuf      ", launch_grid2d<16, 0, true,  false, false>},
    {"3 +lds-padding    ", launch_grid2d<16, 8, true,  false, false>},
    {"4 +oob-safe       ", launch_grid2d<16, 8, true,  false, true >},
    {"5 +deep-K(32)     ", launch_grid2d<32, 8, true,  false, true >},
    {"6 +block-sched    ", launch_grid1d<32, 8, true,  false, true >},
    {"7 +prefetch/prio  ", launch_grid1d<32, 8, true,  true,  true >},
    {"8 single-buffer   ", launch_grid1d<32, 8, false, true,  true >},
};
static constexpr int kNumSteps = sizeof(kSteps) / sizeof(kSteps[0]);

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096, reps = 10;
    if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
    if (argc >= 5) reps = std::atoi(argv[4]);
    if (M <= 0 || N <= 0 || K <= 0 || (M % 16) || (N % 16) || (K % 32)) {
        printf("Usage: %s [M N K] [reps]  (M,N multiple of 16; K multiple of 32)\n", argv[0]);
        return 2;
    }

    const size_t szA = (size_t)M * K, szB = (size_t)N * K, szD = (size_t)M * N;
    std::vector<__half> hA(szA), hB(szB);
    std::vector<float> hD(szD), ref(szD);

    // Small, well-conditioned data so the FP16 inputs + FP32 accumulate stay accurate at large K.
    constexpr float S = 1e-2f;
    for (size_t i = 0; i < szA; i++) hA[i] = __float2half(S * (float)(((i * 7 + 3) % 17) - 8) * 0.1f);
    for (size_t i = 0; i < szB; i++) hB[i] = __float2half(S * (float)(((i * 5 + 1) % 13) - 6) * 0.1f);

    __half *dA, *dB; float *dD;
    HIP_CHECK(hipMalloc(&dA, szA * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dB, szB * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dD, szD * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), szA * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), szB * sizeof(__half), hipMemcpyHostToDevice));

    // CPU reference only for modest sizes (the O(M*N*K) long-double loop is slow).
    const bool do_cpu = (double)M * N * K <= 600e6;
    double max_ref = 0.0;
    if (do_cpu) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                long double acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (long double)__half2float(hA[(size_t)m * K + k]) *
                           (long double)__half2float(hB[(size_t)n * K + k]);
                ref[(size_t)m * N + n] = (float)acc;
                if (std::fabs((double)acc) > max_ref) max_ref = std::fabs((double)acc);
            }
    }

    // Baseline (step 1) output is the GPU reference used to gate every other step
    // when the CPU reference is too expensive to compute.
    std::vector<float> baseline(szD);

    printf("GEMM FP16 opt  M=%d N=%d K=%d  reps=%d  (2*M*N*K = %.2f GFLOP)\n",
           M, N, K, reps, 2.0 * M * N * K / 1e9);
    printf("CPU reference: %s\n", do_cpu ? "on (gates step 1)" : "off (steps gated vs step-1 GPU output)");
    printf("%-20s %10s %10s %12s %6s\n", "step", "ms", "GFLOP/s", "max_rel_err", "pass");

    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    int failures = 0;

    for (int s = 0; s < kNumSteps; s++) {
        HIP_CHECK(hipMemset(dD, 0, szD * sizeof(float)));
        kSteps[s].fn(dA, dB, dD, M, N, K);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hD.data(), dD, szD * sizeof(float), hipMemcpyDeviceToHost));

        if (s == 0) baseline = hD;  // trust step 1 (checked vs CPU below when available)

        // Correctness: compare against CPU ref if we have it, else against step-1 output.
        const std::vector<float>& cmp = do_cpu ? ref : baseline;
        double me = 0.0, scale = do_cpu ? max_ref : 0.0;
        for (size_t i = 0; i < szD; i++) {
            me = std::fmax(me, std::fabs((double)hD[i] - (double)cmp[i]));
            if (!do_cpu) scale = std::fmax(scale, std::fabs((double)cmp[i]));
        }
        const double rel = scale > 0 ? me / scale : me;
        const bool pass = rel < 2e-2;
        if (!pass) failures++;

        // Timing.
        kSteps[s].fn(dA, dB, dD, M, N, K);  // warm up
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(t0));
        for (int r = 0; r < reps; r++) kSteps[s].fn(dA, dB, dD, M, N, K);
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipEventSynchronize(t1));
        float ms_total = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms_total, t0, t1));
        const double ms = ms_total / reps;
        const double gflops = 2.0 * M * N * K / (ms * 1e-3) / 1e9;

        printf("%-20s %10.3f %10.1f %12.2e %6s\n", kSteps[s].name, ms, gflops, rel, pass ? "ok" : "FAIL");
    }

    HIP_CHECK(hipEventDestroy(t0));
    HIP_CHECK(hipEventDestroy(t1));
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dD);
    printf("%s\n", failures ? "RESULT: FAIL" : "RESULT: PASS");
    return failures ? 1 : 0;
}
