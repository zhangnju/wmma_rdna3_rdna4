// gemm_fp16_blog_faithful.cpp
// Compile:
//   W7900 / RDNA3:  hipcc --offload-arch=gfx1100 -O3 -std=c++17 -mcumode samples/gemm_fp16_blog_faithful.cpp -o gemm_blog
//   R9700 / RDNA4:  hipcc --offload-arch=gfx1201 -O3 -std=c++17 -mcumode samples/gemm_fp16_blog_faithful.cpp -o gemm_blog
// Run: ./gemm_blog [M N K] [reps]   (defaults 4096^3, 10)
//
// -----------------------------------------------------------------------------
// FAITHFUL reproduction of the ROCm blog "FP16 GEMM Optimization on AMD RDNA 3
// and RDNA 4" -- matching its actual choices, in contrast to gemm_fp16_opt_rdna.cpp
// (which deviated: FP32 accumulate, 128x128/2x2, block_k=32, guarded loads).
//
// Blog choices reproduced here:
//   * FP16 ACCUMULATE  (wmma_f16_...; RDNA3 uses OPSEL=false, acc values in even slots)
//   * output C is FP16, COLUMN-MAJOR (ld = M):  C[m,n] at n*M + m
//   * A row-major (A[m*K+k]), B column-major (B[n*K+k])  -> C = A*B (TN)
//   * per-step tiles:
//       1 baseline     32x32,  2x2 wave,  block_k 16, single buffer
//       2 tiled+dbuf   256x256, 4x4 wave, block_k 16, double buffer, staged commit
//       3 +padding     256x256, 4x4,      block_k 16, double, pad 8
//       4 +buffer_rsrc 256x256, 4x4,      block_k 16, double, pad 8, OOB via make_buffer_rsrc
//       5 +deep-K      128x96,  2x2 (4x3 grid), block_k 64, double, pad 8
//       6 +block-sched 128x96,  2x2,      block_k 64, double, pad, grouped 8-row launch
//       7 +prefetch    128x96,  2x2,      block_k 64, double, fine global->reg prefetch + s_setprio
//       8 single-buf   128x96,  2x2,      block_k 64, single buffer + extra barrier
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef __attribute__((ext_vector_type(8)))  _Float16 half8;
typedef __attribute__((ext_vector_type(16))) _Float16 half16;

// ---- arch-specific fragment + accumulator types ----
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1202__)
  #define ARCH_RDNA4 1
  typedef half8  AFrag;      // 8 fp16 / lane
  typedef half8  AccT;       // dense 8 fp16 accumulator
  #define ACC_STRIDE 1
#else
  typedef half16 AFrag;      // 16 fp16 / lane (RDNA3 + host pass)
  typedef half16 AccT;       // 16-wide; 8 meaningful values in even slots (OPSEL=false)
  #define ACC_STRIDE 2
#endif

__device__ __forceinline__ half8 zero8() { half8 z;
#pragma unroll
    for (int i = 0; i < 8; i++) z[i] = (_Float16)0; return z; }

__device__ __forceinline__ AccT acc_zero() { AccT a;
#pragma unroll
    for (int i = 0; i < (int)(sizeof(AccT) / sizeof(_Float16)); i++) a[i] = (_Float16)0; return a; }

__device__ __forceinline__ AccT wmma_f16(AFrag a, AFrag b, AccT c) {
#ifdef ARCH_RDNA4
    return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, c);
#else
    return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c, false);
#endif
}

// Row within a 16x16 output tile that logical accumulator slot e maps to (col = lane%16).
__device__ __forceinline__ int acc_row16(int e, int laneW) {
#ifdef ARCH_RDNA4
    return (laneW >> 4) * 8 + e;
#else
    return 2 * e + (laneW >> 4);
#endif
}

// Buffer resource descriptor load (blog's step-4 OOB technique): offsets past
// num_records return 0 in hardware, so no explicit bounds guard is needed.
__device__ __forceinline__ half8 bufld(__amdgpu_buffer_rsrc_t r, long elemOff) {
    return __builtin_amdgcn_raw_buffer_load_b128(r, (int)(elemOff * 2), 0, 0);  // byte offset
}
template <bool OOB>
__device__ __forceinline__ half8 loadg(const __half* G, __amdgpu_buffer_rsrc_t r, long elemOff) {
    if constexpr (OOB) return bufld(r, elemOff);
    else return *reinterpret_cast<const half8*>(&G[elemOff]);
}

// Gather a 16x16 A/B fragment from an LDS tile [row][k], stride lda.
__device__ __forceinline__ AFrag load_frag(const _Float16* lds, int lda, int rowblk, int kblk) {
    const int laneW = threadIdx.x & 31, mn = laneW & 15;
    const _Float16* base = lds + (rowblk * 16 + mn) * lda + kblk * 16;
    AFrag f;
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

// ---------------------------------------------------------------------------
// Step 1 baseline: one wave owns 32x32 (2x2 WMMA), scalar LDS copies, block_k 16.
// ---------------------------------------------------------------------------
__global__ void gemm_step1(const __half* __restrict__ A, const __half* __restrict__ B,
                           __half* __restrict__ C, int M, int N, int K) {
    __shared__ _Float16 As[32][16];
    __shared__ _Float16 Bs[32][16];
    const int laneW = threadIdx.x;
    const int baseM = blockIdx.y * 32, baseN = blockIdx.x * 32;
    AccT acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) acc[i][j] = acc_zero();
    for (int kt = 0; kt < K; kt += 16) {
#pragma unroll
        for (int k = 0; k < 16; k++) {
            As[laneW][k] = (baseM + laneW < M) ? (_Float16)A[(long)(baseM + laneW) * K + kt + k] : (_Float16)0;
            Bs[laneW][k] = (baseN + laneW < N) ? (_Float16)B[(long)(baseN + laneW) * K + kt + k] : (_Float16)0;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
            acc[i][j] = wmma_f16(load_frag(&As[0][0], 16, i, 0), load_frag(&Bs[0][0], 16, j, 0), acc[i][j]);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int row = baseM + i * 16 + acc_row16(e, laneW);
            const int col = baseN + j * 16 + (laneW & 15);
            if (row < M && col < N) C[(long)col * M + row] = (__half)acc[i][j][e * ACC_STRIDE];  // column-major
        }
}

// ---------------------------------------------------------------------------
// Steps 2-8: templated kernel. FP16 acc, column-major output, TMxTN wave tile.
//   PF: fine per-WMMA global->register prefetch interleave + s_setprio (step 7+)
//   SCHED: grouped 8-row block launch (step 6+)
//   DB: double buffer (steps 2-7) vs single (step 8)
//   OOB: guarded loads (step 4 flips this on; buffer_rsrc upgrade is a follow-up)
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN, int PAD, bool DB, bool SCHED, bool PF, bool OOB>
__global__ __launch_bounds__(BM * BN / (8 * TM * TN)) void gemm_blog(
    const __half* __restrict__ A, const __half* __restrict__ B, __half* __restrict__ C, int M, int N, int K) {
    constexpr int LDA = BK + PAD, NBUF = DB ? 2 : 1;
    __shared__ _Float16 As[NBUF][BM * LDA];
    __shared__ _Float16 Bs[NBUF][BN * LDA];
    const int tid = threadIdx.x, laneW = tid & 31, wid = tid >> 5;
    constexpr int WN = BN / (16 * TN);
    const int wr = wid / WN, wc = wid % WN;

    int bm, bn;
    if (SCHED) {
        const int nbn = (N + BN - 1) / BN, nbm = (M + BM - 1) / BM;
        constexpr int G = 8;
        const int per = G * nbn, grp = blockIdx.x / per, loc = blockIdx.x % per;
        const int gr = (nbm - grp * G) < G ? (nbm - grp * G) : G;
        bm = grp * G + loc % gr; bn = loc / gr;
    } else { bm = blockIdx.y; bn = blockIdx.x; }
    const int baseM = bm * BM, baseN = bn * BN;

    // Buffer descriptors for OOB-safe global loads (step 4). num_records = full array bytes.
    const __amdgpu_buffer_rsrc_t rA = __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, (int)((long)M * K * 2), 0x31004000u);
    const __amdgpu_buffer_rsrc_t rB = __builtin_amdgcn_make_buffer_rsrc((void*)B, 0, (int)((long)N * K * 2), 0x31004000u);

    AccT acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++) for (int j = 0; j < TN; j++) acc[i][j] = acc_zero();

    constexpr int THREADS = BM * BN / (8 * TM * TN);
    constexpr int CHA = BK / 8, NCA = BM * CHA, NCB = BN * CHA;
    constexpr int CPTA = (NCA + THREADS - 1) / THREADS, CPTB = (NCB + THREADS - 1) / THREADS;
    const int NK = K / BK;

    // prologue: load tile 0 into buffer 0
#pragma unroll
    for (int i = 0; i < CPTA; i++) { int g = tid + i * THREADS; if (g >= NCA) continue; int row = g / CHA, k8 = (g % CHA) * 8;
        *reinterpret_cast<half8*>(&As[0][row * LDA + k8]) = loadg<OOB>(A, rA, (long)(baseM + row) * K + k8); }
#pragma unroll
    for (int i = 0; i < CPTB; i++) { int g = tid + i * THREADS; if (g >= NCB) continue; int row = g / CHA, k8 = (g % CHA) * 8;
        *reinterpret_cast<half8*>(&Bs[0][row * LDA + k8]) = loadg<OOB>(B, rB, (long)(baseN + row) * K + k8); }
    __syncthreads();

    int cur = 0;
    for (int kt = 0; kt < NK; kt++) {
        const int nxt = kt + 1;
        half8 ra[CPTA], rb[CPTB];
        if constexpr (!PF) {
            // coarse staged commit: issue all next-tile global loads before the MMA block
            if (nxt < NK) {
#pragma unroll
                for (int i = 0; i < CPTA; i++) { int g = tid + i * THREADS; if (g >= NCA) continue; int row = g / CHA, k8 = (g % CHA) * 8;
                    ra[i] = loadg<OOB>(A, rA, (long)(baseM + row) * K + nxt * BK + k8); }
#pragma unroll
                for (int i = 0; i < CPTB; i++) { int g = tid + i * THREADS; if (g >= NCB) continue; int row = g / CHA, k8 = (g % CHA) * 8;
                    rb[i] = loadg<OOB>(B, rB, (long)(baseN + row) * K + nxt * BK + k8); }
            }
#pragma unroll
            for (int ks = 0; ks < BK / 16; ks++) {
                AFrag af[TM], bf[TN];
#pragma unroll
                for (int i = 0; i < TM; i++) af[i] = load_frag(As[cur], LDA, wr * TM + i, ks);
#pragma unroll
                for (int j = 0; j < TN; j++) bf[j] = load_frag(Bs[cur], LDA, wc * TN + j, ks);
#pragma unroll
                for (int i = 0; i < TM; i++) for (int j = 0; j < TN; j++) acc[i][j] = wmma_f16(af[i], bf[j], acc[i][j]);
            }
        } else {
            // step 7 fine interleave: one next-tile global chunk issued every 2nd WMMA
            // (blog technique). The flat index w is compile-time in the unrolled loops, so
            // li = w/2 keeps ra[]/rb[] static-indexed. Single s_setprio around the fold.
            static_assert(2 * (CPTA + CPTB) <= (BK / 16) * TM * TN, "need >=2 WMMA per chunk");
            __builtin_amdgcn_s_setprio(1);
#pragma unroll
            for (int ks = 0; ks < BK / 16; ks++) {
                AFrag af[TM], bf[TN];
#pragma unroll
                for (int i = 0; i < TM; i++) af[i] = load_frag(As[cur], LDA, wr * TM + i, ks);
#pragma unroll
                for (int j = 0; j < TN; j++) bf[j] = load_frag(Bs[cur], LDA, wc * TN + j, ks);
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++) {
                        const int w = (ks * TM + i) * TN + j;   // compile-time after unroll
                        if (nxt < NK && (w % 2) == 0) {
                            const int li = w / 2;
                            if (li < CPTA) { int g = tid + li * THREADS; if (g < NCA) { int row = g / CHA, k8 = (g % CHA) * 8;
                                ra[li] = loadg<OOB>(A, rA, (long)(baseM + row) * K + nxt * BK + k8); } }
                            else if (li - CPTA < CPTB) { int bi = li - CPTA; int g = tid + bi * THREADS; if (g < NCB) { int row = g / CHA, k8 = (g % CHA) * 8;
                                rb[bi] = loadg<OOB>(B, rB, (long)(baseN + row) * K + nxt * BK + k8); } }
                        }
                        acc[i][j] = wmma_f16(af[i], bf[j], acc[i][j]);
                    }
            }
            __builtin_amdgcn_s_setprio(0);
        }
        if (nxt < NK) {
            const int nb = DB ? (cur ^ 1) : cur;
            if (!DB) __syncthreads();   // single buffer: all reads of As[cur] must finish
#pragma unroll
            for (int i = 0; i < CPTA; i++) { int g = tid + i * THREADS; if (g >= NCA) continue; int row = g / CHA, k8 = (g % CHA) * 8;
                *reinterpret_cast<half8*>(&As[nb][row * LDA + k8]) = ra[i]; }
#pragma unroll
            for (int i = 0; i < CPTB; i++) { int g = tid + i * THREADS; if (g >= NCB) continue; int row = g / CHA, k8 = (g % CHA) * 8;
                *reinterpret_cast<half8*>(&Bs[nb][row * LDA + k8]) = rb[i]; }
            __syncthreads();
            cur = nb;
        }
    }
#pragma unroll
    for (int i = 0; i < TM; i++) for (int j = 0; j < TN; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int row = baseM + (wr * TM + i) * 16 + acc_row16(e, laneW);
            const int col = baseN + (wc * TN + j) * 16 + (laneW & 15);
            if (row < M && col < N) C[(long)col * M + row] = (__half)acc[i][j][e * ACC_STRIDE];  // column-major
        }
}

// ---------------------------------------------------------------------------
// Host harness
// ---------------------------------------------------------------------------
#define HIP_CHECK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){printf("HIP err %s:%d %s\n",#x,__LINE__,hipGetErrorString(e_));std::exit(1);} } while(0)
typedef void (*StepFn)(const __half*, const __half*, __half*, int, int, int);

static void launch_step1(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    dim3 grid((N + 31) / 32, (M + 31) / 32); gemm_step1<<<grid, 32>>>(A, B, C, M, N, K);
}
// 256x256 / 4x4 wave tile  (steps 2-4): 2D grid
template <int PAD, bool OOB>
static void launch_256(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    dim3 grid((N + 255) / 256, (M + 255) / 256);
    gemm_blog<256, 256, 16, 4, 4, PAD, true, false, false, OOB><<<grid, 512>>>(A, B, C, M, N, K);
}
// 128x96 / 2x2 wave tile (4x3 grid, 384 thr), block_k 64  (steps 5-8)
template <bool SCHED, bool PF, bool DB>
static void launch_128x96(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    constexpr int thr = 128 * 96 / (8 * 2 * 2);   // = 384
    if (SCHED) { const int nb = ((M + 127) / 128) * ((N + 95) / 96);
        gemm_blog<128, 96, 64, 2, 2, 8, DB, true, PF, true><<<nb, thr>>>(A, B, C, M, N, K);
    } else { dim3 grid((N + 95) / 96, (M + 127) / 128);
        gemm_blog<128, 96, 64, 2, 2, 8, DB, false, PF, true><<<grid, thr>>>(A, B, C, M, N, K); }
}

struct Step { const char* name; StepFn fn; };
static const Step kSteps[] = {
    {"1 baseline        ", launch_step1},
    {"2 tiled+dbuf      ", launch_256<0, false>},
    {"3 +lds-padding    ", launch_256<8, false>},
    {"4 +oob-safe       ", launch_256<8, true >},
    {"5 +deep-K(64)     ", launch_128x96<false, false, true>},
    {"6 +block-sched    ", launch_128x96<true,  false, true>},
    {"7 +prefetch/prio  ", launch_128x96<true,  true,  true>},
    {"8 single-buffer   ", launch_128x96<true,  true,  false>},
};
static constexpr int kNumSteps = sizeof(kSteps) / sizeof(kSteps[0]);

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096, reps = 10;
    if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
    if (argc >= 5) reps = std::atoi(argv[4]);
    // Steps 2,3 use a 256 tile without OOB guard, so require M,N multiple of 256.
    // (Steps 1,4-8 guard loads/stores and handle any size; K multiple of 64 for block_k=64.)
    if (M <= 0 || N <= 0 || K <= 0 || (M % 256) || (N % 256) || (K % 64)) {
        printf("Usage: %s [M N K] [reps]  (M,N multiple of 256; K multiple of 64)\n", argv[0]);
        return 2;
    }
    const size_t szA = (size_t)M * K, szB = (size_t)N * K, szC = (size_t)M * N;
    std::vector<__half> hA(szA), hB(szB), hC(szC);
    // Small data so FP16 accumulation stays accurate.
    for (size_t i = 0; i < szA; i++) hA[i] = __float2half(1e-2f * (float)((int)((i * 7 + 3) % 17) - 8) * 0.1f);
    for (size_t i = 0; i < szB; i++) hB[i] = __float2half(1e-2f * (float)((int)((i * 5 + 1) % 13) - 6) * 0.1f);

    // Independent reference (long double) at sample points; FP16-accumulate tolerance.
    constexpr int NS = 2048;
    std::vector<int> sm(NS), sn(NS); std::vector<double> sref(NS); double max_ref = 0;
    { unsigned sd = 0x9e3779b9u; auto rnd = [&]() { sd = sd * 1664525u + 1013904223u; return sd >> 1; };
      for (int s = 0; s < NS; s++) { int m = rnd() % M, n = rnd() % N; sm[s] = m; sn[s] = n; long double a = 0;
        for (int k = 0; k < K; k++) a += (long double)__half2float(hA[(size_t)m * K + k]) * (long double)__half2float(hB[(size_t)n * K + k]);
        sref[s] = (double)a; max_ref = std::fmax(max_ref, std::fabs((double)a)); } }

    __half *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, szA * sizeof(__half))); HIP_CHECK(hipMalloc(&dB, szB * sizeof(__half))); HIP_CHECK(hipMalloc(&dC, szC * sizeof(__half)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), szA * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), szB * sizeof(__half), hipMemcpyHostToDevice));
    hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    printf("GEMM FP16 (faithful blog)  M=%d N=%d K=%d reps=%d  (FP16 accumulate, column-major C)\n", M, N, K, reps);
    printf("correctness: %d random points vs long-double ref (FP16-accum tolerance) + NaN/Inf scan\n", NS);
    printf("%-20s %10s %10s %11s %6s\n", "step", "ms", "GFLOP/s", "rel_err", "pass");
    int failures = 0;
    for (int s = 0; s < kNumSteps; s++) {
        HIP_CHECK(hipMemset(dC, 0, szC * sizeof(__half)));
        kSteps[s].fn(dA, dB, dC, M, N, K);
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hC.data(), dC, szC * sizeof(__half), hipMemcpyDeviceToHost));
        bool finite = true; double ae = 0;
        for (size_t i = 0; i < szC; i++) if (!std::isfinite(__half2float(hC[i]))) { finite = false; break; }
        for (int s2 = 0; s2 < NS; s2++) ae = std::fmax(ae, std::fabs((double)__half2float(hC[(size_t)sn[s2] * M + sm[s2]]) - sref[s2]));  // column-major read
        double rel = max_ref > 0 ? ae / max_ref : ae; bool pass = finite && rel < 6e-2;  // FP16-accum tolerance
        if (!pass) failures++;
        kSteps[s].fn(dA, dB, dC, M, N, K); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(t0)); for (int r = 0; r < reps; r++) kSteps[s].fn(dA, dB, dC, M, N, K);
        HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
        float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1)); ms /= reps;
        printf("%-20s %10.3f %10.1f %11.2e %6s\n", kSteps[s].name, ms, 2.0 * M * N * K / (ms * 1e-3) / 1e9, rel,
               !finite ? "NAN!" : (pass ? "ok" : "FAIL"));
    }
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
    printf("%s\n", failures ? "RESULT: FAIL" : "RESULT: PASS");
    return failures ? 1 : 0;
}
