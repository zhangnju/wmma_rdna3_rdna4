// rocwmma_example.cpp
// Compile (repo root, RDNA 3/4, wave32): hipcc --offload-arch=native samples/rocwmma_example.cpp -o rocwmma_example
// CDNA (wave64, e.g. gfx942): hipcc --offload-arch=gfx942 samples/rocwmma_example.cpp -o rocwmma_example
// Block size is chosen at runtime from hipDeviceProp_t::warpSize.
//
// Single 16×16×16 tile: D = A * B + C in FP32, A = column-major __half, B = row-major __half,
// D row-major — same memory layout story as wmma_rdna4_fp16.cpp. CPU reference uses long double.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using namespace rocwmma;

constexpr int M = 16, N = 16, K = 16;

__global__ void rocwmma_gemm_tile(const __half* A, const __half* B, const float* C, float* D) {
    fragment<matrix_a, M, N, K, __half, col_major> a_frag;
    fragment<matrix_b, M, N, K, __half, row_major> b_frag;
    fragment<accumulator, M, N, K, float>           c_frag;
    fragment<accumulator, M, N, K, float>           d_frag;

    load_matrix_sync(a_frag, A, /*ld=*/K);
    load_matrix_sync(b_frag, B, /*ld=*/N);
    // rocWMMA 7.x: accumulator load needs an explicit memory layout (see rocwmma_impl.hpp).
    load_matrix_sync(c_frag, C, /*ld=*/N, mem_row_major);

    mma_sync(d_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(D, d_frag, /*ld=*/N, mem_row_major);
}

static void cpu_gemm_16x16_ref(const std::vector<__half>& A_cm, const std::vector<__half>& B_rm,
                               const std::vector<float>& C_rm, std::vector<float>& D_rm) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            long double acc = C_rm[m * N + n];
            for (int k = 0; k < K; k++)
                acc += (long double)__half2float(A_cm[k * M + m]) *
                       (long double)__half2float(B_rm[k * N + n]);
            D_rm[m * N + n] = (float)acc;
        }
}

int main() {
    constexpr int sz = M * N;

    std::vector<__half> h_A(sz), h_B(sz);
    std::vector<float>  h_C(sz, 0.f), h_D(sz, 0.f);

    for (int i = 0; i < sz; i++) {
        h_A[i] = __float2half(i * 0.01f);
        h_B[i] = __float2half(i * 0.01f);
    }
    h_C[0]  = 0.25f;
    h_C[17] = -0.125f;

    std::vector<float> cpu_D(sz);
    cpu_gemm_16x16_ref(h_A, h_B, h_C, cpu_D);

    __half *d_A, *d_B;
    float *d_C, *d_D;
    hipMalloc(&d_A, sz * sizeof(__half));
    hipMalloc(&d_B, sz * sizeof(__half));
    hipMalloc(&d_C, sz * sizeof(float));
    hipMalloc(&d_D, sz * sizeof(float));
    hipMemcpy(d_A, h_A.data(), sz * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sz * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), sz * sizeof(float), hipMemcpyHostToDevice);

    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, 0);
    const int threads = prop.warpSize;

    rocwmma_gemm_tile<<<1, threads>>>(d_A, d_B, d_C, d_D);
    hipDeviceSynchronize();
    hipMemcpy(h_D.data(), d_D, sz * sizeof(float), hipMemcpyDeviceToHost);

    float max_err = 0.f;
    for (int i = 0; i < sz; i++)
        max_err = std::fmax(max_err, std::fabs(cpu_D[i] - h_D[i]));
    std::cout << "CPU ref D[0][0] = " << cpu_D[0] << ", GPU D[0][0] = " << h_D[0] << "\n";
    std::cout << "Max abs error (CPU vs GPU): " << max_err << " (wave " << threads << ")\n";

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_D);

    return max_err > 1e-2f ? 1 : 0;
}
