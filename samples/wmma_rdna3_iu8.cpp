// wmma_rdna3_iu8.cpp
// Compile (repo root): hipcc --offload-arch=gfx1100 -std=c++20 samples/wmma_rdna3_iu8.cpp -o wmma_rdna3_iu8
//
// D = A*B + C for 16x16 int8 tiles, int32 accumulators, RDNA 3 Wave32 WMMA.
//
// Operand packing matches Composable Kernel test/wmma_op (matmul kernel): shared-memory
// swizzle into fragments and neg_a/neg_b = true at the builtin. Global layout here is
// A column-major M×K (A[m,k] at k*16+m), B row-major K×N (B[k,n] at k*16+n); loads map
// those layouts onto the same staged bytes as CK’s A row-major / B column-major path.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

typedef int8_t int8x16 __attribute__((ext_vector_type(16)));
typedef int32_t int32x4 __attribute__((ext_vector_type(4)));
typedef int32_t int32x8 __attribute__((ext_vector_type(8)));

__global__ void wmma_gemm_rdna3_iu8(
    const int8_t* __restrict__ A,   // [16 x 16] M×K column-major: A[m,k] at k*16+m
    const int8_t* __restrict__ B,   // [16 x 16] K×N row-major: B[k,n] at k*16+n
    const int32_t* __restrict__ C,  // [16 x 16] row-major C[m,n] at m*16+n
    int32_t* __restrict__ D)        // [16 x 16] row-major
{
    __shared__ int8_t p_shared[16 * 16 * 2];
    const int lIdx    = threadIdx.x;
    const int lane    = lIdx % 16;
    const int lane_lo = lIdx / 2;
    const int lane_hi = lIdx % 2;
    const int wave2   = lIdx / 16;

    int8_t a_temp[8];
    int8_t b_temp[8];
    // Same (m,k) / (k,n) as CK row-major A / col-major B loads, but indexed in
    // column-major A and row-major B: m=lane_lo, k=8*lane_hi+ele, n=lane_lo.
#pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        const int k = 8 * lane_hi + ele;
        a_temp[ele] = A[k * 16 + lane_lo];
        b_temp[ele] = B[k * 16 + lane_lo];
    }

    __syncthreads();

#pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele]                = a_temp[ele];
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele + 16 * 16]      = b_temp[ele];
    }

    __syncthreads();

    int8x16 a_frag{};
    int8x16 b_frag{};
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
        b_frag[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8 + 16 * 16];
        a_frag[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8];
    }

    __syncthreads();

    int32x8 c_frag{};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = i * 2 + wave2;
        c_frag[i] = C[r * 16 + lane];
    }

    // CK intrin_wmma_i32_16x16x16_iu8_w32 uses neg_a=true, neg_b=true by default.
    constexpr bool neg_a = true;
    constexpr bool neg_b = true;
    constexpr bool clamp = false;

    int32x8 d_frag = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        neg_a,
        __builtin_bit_cast(int32x4, a_frag),
        neg_b,
        __builtin_bit_cast(int32x4, b_frag),
        c_frag,
        clamp);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = i * 2 + wave2;
        D[r * 16 + lane] = d_frag[i];
    }
}

// A column-major M×K, B row-major K×N, C/D row-major M×N.
static void cpu_gemm_i8_i32(const std::vector<int8_t>& A_cm,
                            const std::vector<int8_t>& B_rm,
                            const std::vector<int32_t>& C_rm,
                            std::vector<int32_t>& D_rm)
{
    constexpr int N = 16;
    for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
            int64_t acc = C_rm[m * N + n];
            for (int k = 0; k < N; k++) {
                int32_t a = A_cm[k * N + m];
                int32_t b = B_rm[k * N + n];
                acc += (int64_t)a * b;
            }
            D_rm[m * N + n] = (int32_t)acc;
        }
    }
}

static int max_abs_diff_i32(const std::vector<int32_t>& a, const std::vector<int32_t>& b)
{
    int e = 0;
    for (size_t i = 0; i < a.size(); i++)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

int main()
{
    constexpr int N = 16;
    constexpr int sz = N * N;

    std::vector<int8_t> h_A(sz), h_B(sz);
    std::vector<int32_t> h_C(sz, 0), h_D(sz, 0);

    for (int k = 0; k < N; k++) {
        for (int m = 0; m < N; m++)
            h_A[k * N + m] = (int8_t)(((m * N + k) % 7) - 3);
    }
    for (int k = 0; k < N; k++) {
        for (int n = 0; n < N; n++)
            h_B[k * N + n] = (int8_t)(((k * N + n) % 5) - 2);
    }
    h_C[0]  = 10;
    h_C[17] = -7;

    std::vector<int32_t> cpu_D(sz);
    cpu_gemm_i8_i32(h_A, h_B, h_C, cpu_D);

    int8_t *d_A, *d_B;
    int32_t *d_C, *d_D;
    hipMalloc(&d_A, sz * sizeof(int8_t));
    hipMalloc(&d_B, sz * sizeof(int8_t));
    hipMalloc(&d_C, sz * sizeof(int32_t));
    hipMalloc(&d_D, sz * sizeof(int32_t));

    hipMemcpy(d_A, h_A.data(), sz * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sz * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), sz * sizeof(int32_t), hipMemcpyHostToDevice);

    wmma_gemm_rdna3_iu8<<<1, 32>>>(d_A, d_B, d_C, d_D);
    hipDeviceSynchronize();

    hipMemcpy(h_D.data(), d_D, sz * sizeof(int32_t), hipMemcpyDeviceToHost);

    const int err = max_abs_diff_i32(cpu_D, h_D);
    std::cout << "CPU ref D[0] = " << cpu_D[0] << ", GPU D[0] = " << h_D[0] << "\n";
    std::cout << "Max abs diff (CPU vs GPU): " << err << "\n";

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_D);
    return err != 0 ? 1 : 0;
}
