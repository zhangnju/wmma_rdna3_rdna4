// wmma_rdna4_fp16.cpp
// Compile (repo root): hipcc --offload-arch=gfx1201 samples/wmma_rdna4_fp16.cpp -o wmma_rdna4_fp16

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <iostream>
#include <vector>

// GFX12 FP16→FP32 WMMA: 8 half / lane for A and B (half8_t), 8 float / lane for C/D
typedef __attribute__((ext_vector_type(8))) _Float16 half8_t;
typedef __attribute__((ext_vector_type(8))) float    float8_t;

__global__ void wmma_gemm_rdna4(
    const __half* __restrict__ A,   // [16×16] column-major: A[k*16+m]
    const __half* __restrict__ B,   // [16×16] row-major:    B[k*16+n]
    const float*  __restrict__ C,   // [16×16] row-major:    C[m*16+n]
          float*  __restrict__ D)
{
    const int lane = threadIdx.x;  // 0–31

    half8_t a_frag, b_frag;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const int mn = lane % 16;  // m for A, n for B (same lane id here)
        const int k = (lane / 16) * 8 + e;
        a_frag[e] = (_Float16)A[k * 16 + mn];
        b_frag[e] = (_Float16)B[k * 16 + mn];
    }

    float8_t c_frag;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        c_frag[e] = C[m * 16 + n];
    }

    float8_t d_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        a_frag, b_frag, c_frag);

    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        D[m * 16 + n] = d_frag[e];
    }
}

// CPU reference (same layouts as kernel). Double accumulation for a tight check vs WMMA.
static void cpu_gemm_16x16_ref(const std::vector<__half>& A_cm, const std::vector<__half>& B_rm,
                               const std::vector<float>& C_rm, std::vector<float>& D_rm) {
    constexpr int N = 16;
    for (int m = 0; m < N; m++)
        for (int n = 0; n < N; n++) {
            long double acc = C_rm[m * N + n];
            for (int k = 0; k < N; k++)
                acc += (long double)__half2float(A_cm[k * N + m]) *
                       (long double)__half2float(B_rm[k * N + n]);
            D_rm[m * N + n] = (float)acc;
        }
}

int main() {
    constexpr int N = 16, sz = N * N;

    std::vector<__half> h_A(sz), h_B(sz);
    std::vector<float>  h_C(sz, 0.f), h_D(sz, 0.f);

    for (int i = 0; i < sz; i++) {
        h_A[i] = __float2half(i * 0.01f);
        h_B[i] = __float2half(i * 0.01f);
    }
    h_C[0] = 0.25f;
    h_C[17] = -0.125f;

    std::vector<float> cpu_D(sz);
    cpu_gemm_16x16_ref(h_A, h_B, h_C, cpu_D);

    __half *d_A, *d_B; float *d_C, *d_D;
    hipMalloc(&d_A, sz * sizeof(__half)); hipMalloc(&d_B, sz * sizeof(__half));
    hipMalloc(&d_C, sz * sizeof(float));  hipMalloc(&d_D, sz * sizeof(float));
    hipMemcpy(d_A, h_A.data(), sz*sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sz*sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), sz*sizeof(float),  hipMemcpyHostToDevice);

    wmma_gemm_rdna4<<<1, 32>>>(d_A, d_B, d_C, d_D);
    hipDeviceSynchronize();
    hipMemcpy(h_D.data(), d_D, sz*sizeof(float), hipMemcpyDeviceToHost);

    float max_err = 0.f;
    for (size_t i = 0; i < sz; i++)
        max_err = std::fmax(max_err, std::fabs(cpu_D[i] - h_D[i]));
    std::cout << "CPU ref D[0][0] = " << cpu_D[0] << ", GPU D[0][0] = " << h_D[0] << "\n";
    std::cout << "Max abs error (CPU vs GPU): " << max_err << "\n";

    hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_D);
    return max_err > 1e-2f ? 1 : 0;
}
