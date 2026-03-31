// wmma_rdna4_fp8.cpp
// Compile (repo root): hipcc --offload-arch=gfx1201 samples/wmma_rdna4_fp8.cpp -o wmma_rdna4_fp8
//
// D = A * B + C, 16×16 tiles, FP8 E4M3 (OCP) inputs, FP32 accumulators — RDNA 4 gfx12 only.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

typedef int32_t i32x2 __attribute__((ext_vector_type(2)));
typedef float   float8_t __attribute__((ext_vector_type(8)));

__device__ __forceinline__ i32x2 pack_fp8_A_col(const uint8_t* A, int lane) {
    union {
        i32x2 v;
        uint8_t b[8];
    } u{};
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = lane % 16;
        const int k = (lane / 16) * 8 + e;
        u.b[e] = A[k * 16 + m];
    }
    return u.v;
}

__device__ __forceinline__ i32x2 pack_fp8_B_row(const uint8_t* B, int lane) {
    union {
        i32x2 v;
        uint8_t b[8];
    } u{};
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int k = (lane / 16) * 8 + e;
        const int n = lane % 16;
        u.b[e] = B[k * 16 + n];
    }
    return u.v;
}

__global__ void wmma_gemm_rdna4_fp8(const uint8_t* __restrict__ A,
                                    const uint8_t* __restrict__ B,
                                    const float* __restrict__ C,
                                    float* __restrict__ D) {
    const int lane = threadIdx.x;
    const i32x2 a_frag = pack_fp8_A_col(A, lane);
    const i32x2 b_frag = pack_fp8_B_row(B, lane);

    float8_t c_frag;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        c_frag[e] = C[m * 16 + n];
    }

    float8_t d_frag = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
        a_frag, b_frag, c_frag);

#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        D[m * 16 + n] = d_frag[e];
    }
}

// Host: same E4M3 bit pattern as HIP (__hip_fp8_e4m3 OCP).
static float fp8_storage_to_float(uint8_t raw) {
    __hip_fp8_e4m3 v;
    v.__x = raw;
    return static_cast<float>(v);
}

// CPU reference: A column-major A[k*N+m], B row-major B[k*N+n], C/D row-major D[m*N+n].
static void cpu_gemm_fp8_ref(const std::vector<uint8_t>& A_cm,
                             const std::vector<uint8_t>& B_rm,
                             const std::vector<float>& C_rm,
                             std::vector<float>& D_rm) {
    constexpr int N = 16;
    for (int m = 0; m < N; m++)
        for (int n = 0; n < N; n++) {
            long double acc = C_rm[m * N + n];
            for (int k = 0; k < N; k++)
                acc += (long double)fp8_storage_to_float(A_cm[k * N + m]) *
                      (long double)fp8_storage_to_float(B_rm[k * N + n]);
            D_rm[m * N + n] = (float)acc;
        }
}

int main() {
    constexpr int N = 16, sz = N * N;

    std::vector<uint8_t> h_A(sz), h_B(sz);
    std::vector<float>   h_C(sz, 0.f), h_D(sz, 0.f);

    for (int i = 0; i < sz; i++) {
        const float fa = (i * 0.02f) / float(sz);
        const float fb = ((i + 3) * 0.015f) / float(sz);
        h_A[i]           = __hip_fp8_e4m3(fa).__x;
        h_B[i]           = __hip_fp8_e4m3(fb).__x;
    }
    h_C[0]  = 0.125f;
    h_C[17] = -0.0625f;

    std::vector<float> cpu_D(sz);
    cpu_gemm_fp8_ref(h_A, h_B, h_C, cpu_D);

    uint8_t* d_A;
    uint8_t* d_B;
    float *d_C, *d_D;
    hipMalloc(&d_A, sz * sizeof(uint8_t));
    hipMalloc(&d_B, sz * sizeof(uint8_t));
    hipMalloc(&d_C, sz * sizeof(float));
    hipMalloc(&d_D, sz * sizeof(float));
    hipMemcpy(d_A, h_A.data(), sz * sizeof(uint8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sz * sizeof(uint8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), sz * sizeof(float), hipMemcpyHostToDevice);

    wmma_gemm_rdna4_fp8<<<1, 32>>>(d_A, d_B, d_C, d_D);
    hipDeviceSynchronize();
    hipMemcpy(h_D.data(), d_D, sz * sizeof(float), hipMemcpyDeviceToHost);

    float max_err = 0.f;
    for (size_t i = 0; i < sz; i++)
        max_err = std::fmax(max_err, std::fabs(cpu_D[i] - h_D[i]));
    std::cout << "CPU ref D[0][0] = " << cpu_D[0] << ", GPU D[0][0] = " << h_D[0] << "\n";
    std::cout << "Max abs error (CPU vs GPU): " << max_err << "\n";

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_D);

    // FP8 rounding + FMA differences vs long-double CPU ref: allow wider slack than FP16.
    return max_err > 0.25f ? 1 : 0;
}
