// wmma_rdna3_fp16.cpp
// Compile (repo root): hipcc --offload-arch=gfx1100 samples/wmma_rdna3_fp16.cpp -o wmma_rdna3_fp16
//
// Computes D = A * B + C for 16x16 FP16 matrices,
// accumulating into FP32. Uses Wave32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <iostream>
#include <vector>

// Vector type aliases for WMMA fragments
// On RDNA3 Wave32: A and B need 8 VGPRs each (16 fp16 values packed 2 per VGPR)
typedef _Float16 fp16x16 __attribute__((ext_vector_type(16)));
typedef float    fp32x8  __attribute__((ext_vector_type(8)));

// -----------------------------------------------------------------------
// WMMA kernel: each block handles one 16x16 tile of the output matrix D.
// Launch with <<<1, 32>>> for a single tile example.
// -----------------------------------------------------------------------
__global__ void wmma_gemm_rdna3(
    const __half* __restrict__ A,   // [16 x 16], column-major
    const __half* __restrict__ B,   // [16 x 16], row-major
    const float*  __restrict__ C,   // [16 x 16], row-major
          float*  __restrict__ D)   // [16 x 16], row-major (output)
{
    const int lIdx  = threadIdx.x;   // 0–31, blockDim.x == 32
    const int lane  = lIdx % 16;     // output col of D; B column index; A row index (see comments above)
    // Same wave2 for A/B duplication (lanes i and i+16 share operands) and for C/D: even vs odd rows in column `lane`.
    const int wave2 = lIdx / 16;     // 0 = threads 0–15, 1 = 16–31

    // ---- Load A fragment -----------------------------------------------
    // A column-major: A[m,k] at k*lda+m (lda=16).  Fix row m = lane, scan k → one row of A.
    fp16x16 a_frag;
    #pragma unroll
    for (int k = 0; k < 16; k++)
        a_frag[k] = (_Float16)A[k * 16 + lane];

    // ---- Load B fragment -----------------------------------------------
    // B row-major: B[k,n] at k*ldb+n (ldb=16).  Fix column n = lane, scan k → one column of B.
    // Same linear index k*16+lane as A: lane is row index m here but column index n above;
    // for dense 16×16 tiles both strides are 16, so the address formula coincides.
    fp16x16 b_frag;
    #pragma unroll
    for (int k = 0; k < 16; k++)
        b_frag[k] = (_Float16)B[k * 16 + lane];

    // ---- Load C fragment (same layout as D store) ----
    // 8 floats per thread cover half the rows in column `lane`: r = i*2+wave2 → even rows if wave2==0, odd if ==1.
    fp32x8 c_frag;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = i * 2 + wave2;
        c_frag[i] = C[r * 16 + lane];
    }

    // ---- Execute WMMA --------------------------------------------------
    fp32x8 d_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
        a_frag, b_frag, c_frag);

    // ---- Store D fragment (row-major D[r*16+lane], same r pattern as C) ----
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = i * 2 + wave2;
        D[r * 16 + lane] = d_frag[i];
    }
}

// -----------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------
void fill_ramp(std::vector<__half>& m, int rows, int cols, float scale = 0.01f) {
    for (int i = 0; i < rows * cols; i++)
        m[i] = __float2half(i * scale);
}

// CPU reference: D = A*B + C with A column-major, B and C row-major (same as kernel).
// Uses double for the reduction to tighten the reference vs FP16 WMMA + FP32 acc.
static void cpu_gemm_16x16_ref(const std::vector<__half>& A_cm, const std::vector<__half>& B_rm,
                             const std::vector<float>& C_rm, std::vector<float>& D_rm) {
    constexpr int N = 16;
    for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
            long double acc = C_rm[m * N + n];
            for (int k = 0; k < N; k++) {
                float a_mk = __half2float(A_cm[k * N + m]); // (m,k)
                float b_kn = __half2float(B_rm[k * N + n]); // (k,n)
                acc += (long double)a_mk * (long double)b_kn;
            }
            D_rm[m * N + n] = (float)acc;
        }
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0.f;
    for (size_t i = 0; i < a.size(); i++)
        e = std::fmax(e, std::fabs(a[i] - b[i]));
    return e;
}

int main() {
    constexpr int N = 16;
    constexpr int sz = N * N;

    std::vector<__half> h_A(sz), h_B(sz);
    std::vector<float>  h_C(sz, 0.f), h_D(sz, 0.f);

    fill_ramp(h_A, N, N);
    fill_ramp(h_B, N, N);
    // Non-zero C to exercise the fused add (optional pattern on a few elements)
    h_C[0] = 0.25f;
    h_C[17] = -0.125f;

    std::vector<float> cpu_D(sz);
    cpu_gemm_16x16_ref(h_A, h_B, h_C, cpu_D);

    __half *d_A, *d_B;
    float  *d_C, *d_D;
    hipMalloc(&d_A, sz * sizeof(__half));
    hipMalloc(&d_B, sz * sizeof(__half));
    hipMalloc(&d_C, sz * sizeof(float));
    hipMalloc(&d_D, sz * sizeof(float));

    hipMemcpy(d_A, h_A.data(), sz * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sz * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), sz * sizeof(float),  hipMemcpyHostToDevice);

    // Launch: 1 block, 32 threads (Wave32)
    wmma_gemm_rdna3<<<1, 32>>>(d_A, d_B, d_C, d_D);
    hipDeviceSynchronize();

    hipMemcpy(h_D.data(), d_D, sz * sizeof(float), hipMemcpyDeviceToHost);

    const float max_err = max_abs_diff(cpu_D, h_D);
    std::cout << "CPU ref D[0][0] = " << cpu_D[0] << ", GPU D[0][0] = " << h_D[0] << "\n";
    std::cout << "Max abs error (CPU vs GPU): " << max_err << "\n";

    hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_D);
    return max_err > 1e-2f ? 1 : 0; // loose threshold; tighten if desired
}