// hipblaslt_bench.cpp
// Compile (repo root):
//   W7900 / RDNA 3:  hipcc --offload-arch=gfx1100 -O3 -std=c++17 samples/hipblaslt_bench.cpp -o hipblaslt_bench -lhipblaslt
//   R9700 / RDNA 4:  hipcc --offload-arch=gfx1201 -O3 -std=c++17 samples/hipblaslt_bench.cpp -o hipblaslt_bench -lhipblaslt
//
// Run:  ./hipblaslt_bench [M N K] [reps]      (defaults: 4096 4096 4096, 20 reps)
//
// -----------------------------------------------------------------------------
// hipBLASLt FP16 GEMM throughput reference, for comparing against the hand-written
// kernels in gemm_fp16_opt_rdna.cpp under identical conditions:
//   - FP16 inputs, FP32 accumulate (HIPBLAS_COMPUTE_32F), FP32 output  (same as the
//     gemm_fp16_opt_rdna kernels, which use wmma_f32 with an FP32 accumulator/output)
//   - same M,N,K shape, same device
// It queries hipBLASLt's heuristic solutions, benchmarks each, and reports the best
// one's ms / GFLOP-s. This measures what the tuned library achieves; it does not
// verify numerics (hipBLASLt is the trusted reference).
//
// Computes a standard C(MxN) = A(MxK) * B(KxN), column-major. For the square shapes
// used in the blog comparison the FLOP count (2*M*N*K) and achievable throughput are
// representative regardless of the exact transpose mode.
//
// Requires hipBLASLt (ships with ROCm). Set LD_LIBRARY_PATH=/opt/rocm/lib if the
// loader cannot find libhipblaslt at run time.
// -----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s:%d %s\n",#x,__LINE__,hipGetErrorString(e));std::exit(1);} } while(0)
#define LT_CHECK(x)  do { hipblasStatus_t s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){printf("hipBLASLt err %s:%d code=%d\n",#x,__LINE__,(int)s);std::exit(1);} } while(0)

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096, reps = 20;
    if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
    if (argc >= 5) reps = std::atoi(argv[4]);
    if (M <= 0 || N <= 0 || K <= 0) { printf("Usage: %s [M N K] [reps]\n", argv[0]); return 2; }

    const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
    std::vector<__half> hA(szA), hB(szB);
    for (size_t i = 0; i < szA; i++) hA[i] = __float2half(0.01f);
    for (size_t i = 0; i < szB; i++) hB[i] = __float2half(0.01f);

    __half *dA, *dB; float *dC;
    HIP_CHECK(hipMalloc(&dA, szA * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dB, szB * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dC, szC * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), szA * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), szB * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, szC * sizeof(float)));

    hipblasLtHandle_t handle; LT_CHECK(hipblasLtCreate(&handle));
    hipblasLtMatmulDesc_t desc;
    LT_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opN = HIPBLAS_OP_N;
    LT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    hipblasLtMatrixLayout_t lA, lB, lC;
    LT_CHECK(hipblasLtMatrixLayoutCreate(&lA, HIP_R_16F, M, K, M));
    LT_CHECK(hipblasLtMatrixLayoutCreate(&lB, HIP_R_16F, K, N, K));
    LT_CHECK(hipblasLtMatrixLayoutCreate(&lC, HIP_R_32F, M, N, M));

    const size_t wsSize = 256ull * 1024 * 1024;
    void* ws; HIP_CHECK(hipMalloc(&ws, wsSize));
    hipblasLtMatmulPreference_t pref; LT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));

    constexpr int MAXA = 32;
    hipblasLtMatmulHeuristicResult_t heur[MAXA];
    int found = 0;
    LT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, MAXA, heur, &found));
    if (found == 0) { printf("no hipBLASLt algo found for this shape\n"); return 1; }

    const float alpha = 1.f, beta = 0.f;
    hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    double best_ms = 1e30; int best_i = -1;

    for (int a = 0; a < found; a++) {
        // skip solutions that do not apply to this problem
        if (hipblasLtMatmul(handle, desc, &alpha, dA, lA, dB, lB, &beta, dC, lC, dC, lC,
                            &heur[a].algo, ws, wsSize, 0) != HIPBLAS_STATUS_SUCCESS)
            continue;
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(t0));
        for (int r = 0; r < reps; r++)
            hipblasLtMatmul(handle, desc, &alpha, dA, lA, dB, lB, &beta, dC, lC, dC, lC,
                            &heur[a].algo, ws, wsSize, 0);
        HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
        float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        const double per = ms / reps;
        if (per < best_ms) { best_ms = per; best_i = a; }
    }

    const double gflops = 2.0 * M * N * K / (best_ms * 1e-3) / 1e9;
    printf("hipBLASLt FP16  M=%d N=%d K=%d  algos=%d  best_algo=%d  %.3f ms  %.1f GFLOP/s (%.1f TFLOP/s)\n",
           M, N, K, found, best_i, best_ms, gflops, gflops / 1000.0);

    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC); (void)hipFree(ws);
    return 0;
}
