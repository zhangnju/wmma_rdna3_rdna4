// wmma_rdna3_iu4.cpp
// Compile (repo root): hipcc --offload-arch=gfx1100 -std=c++20 samples/wmma_rdna3_iu4.cpp -o wmma_rdna3_iu4
//
// D = A*B + C for 16x16 int4 tiles (stored as int8 in [-8,7]), int32 accumulators,
// RDNA 3 Wave32 WMMA (i32_16x16x16_iu4).
//
// Shared-memory swizzle + neg flags match Composable Kernel wmma_op matmul (same as wmma_rdna3_iu8.cpp).
// Global layout: A column-major M×K, B row-major K×N; loads map to the same staged bytes as CK’s A RM / B CM path.
// Each int4 is in the low 4 bits of an int8 buffer element; pack 16 nibbles into int32x2 for the builtin.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

typedef int8_t int8x16 __attribute__((ext_vector_type(16)));
typedef int32_t int32x2 __attribute__((ext_vector_type(2)));
typedef int32_t int32x8 __attribute__((ext_vector_type(8)));

__device__ int32x2 pack_iu4_x16(int8x16 v)
{
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        lo |= (uint32_t)(v[i] & 15) << (4 * i);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        hi |= (uint32_t)(v[i + 8] & 15) << (4 * i);
    int32x2 r;
    r[0] = (int32_t)lo;
    r[1] = (int32_t)hi;
    return r;
}

__global__ void wmma_gemm_rdna3_iu4(
    const int8_t* __restrict__ A,   // [16 x 16] M×K column-major, int4 in low 4 bits
    const int8_t* __restrict__ B,   // [16 x 16] K×N row-major
    const int32_t* __restrict__ C,  // [16 x 16] row-major
    int32_t* __restrict__ D)
{
    __shared__ int8_t p_shared[16 * 16 * 2];
    const int lIdx    = threadIdx.x;
    const int lane    = lIdx % 16;
    const int lane_lo = lIdx / 2;
    const int lane_hi = lIdx % 2;
    const int wave2   = lIdx / 16;

    int8_t a_temp[8];
    int8_t b_temp[8];
    // Same (m,k)/(k,n) as CK row-major A / col-major B loads; m=n=lane_lo, k=8*lane_hi+ele.
#pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        const int k = 8 * lane_hi + ele;
        a_temp[ele] = A[k * 16 + lane_lo];
        b_temp[ele] = B[k * 16 + lane_lo];
    }

    __syncthreads();

#pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele]           = a_temp[ele];
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele + 16 * 16] = b_temp[ele];
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

    constexpr bool neg_a = true;
    constexpr bool neg_b = true;
    constexpr bool clamp = false;

    const int32x2 a_pk = pack_iu4_x16(a_frag);
    const int32x2 b_pk = pack_iu4_x16(b_frag);

    int32x8 d_frag = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        neg_a, a_pk, neg_b, b_pk, c_frag, clamp);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = i * 2 + wave2;
        D[r * 16 + lane] = d_frag[i];
    }
}

static int32_t sx_i4(int8_t x)
{
    int32_t v = x & 15;
    if (v & 8)
        v |= ~15;
    return v;
}

static void cpu_gemm_i4_i32(const std::vector<int8_t>& A_cm,
                            const std::vector<int8_t>& B_rm,
                            const std::vector<int32_t>& C_rm,
                            std::vector<int32_t>& D_rm)
{
    constexpr int N = 16;
    for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
            int64_t acc = C_rm[m * N + n];
            for (int k = 0; k < N; k++) {
                int32_t a = sx_i4(A_cm[k * N + m]);
                int32_t b = sx_i4(B_rm[k * N + n]);
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

    auto clamp_i4 = [](int v) -> int8_t {
        if (v > 7)
            v = 7;
        if (v < -8)
            v = -8;
        return (int8_t)v;
    };

    for (int k = 0; k < N; k++) {
        for (int m = 0; m < N; m++)
            h_A[k * N + m] = clamp_i4((m * N + k) % 13 - 6);
    }
    for (int k = 0; k < N; k++) {
        for (int n = 0; n < N; n++)
            h_B[k * N + n] = clamp_i4((k * N + n) % 11 - 5);
    }
    h_C[0]  = 3;
    h_C[17] = -2;

    std::vector<int32_t> cpu_D(sz);
    cpu_gemm_i4_i32(h_A, h_B, h_C, cpu_D);

    int8_t *d_A, *d_B;
    int32_t *d_C, *d_D;
    hipMalloc(&d_A, sz * sizeof(int8_t));
    hipMalloc(&d_B, sz * sizeof(int8_t));
    hipMalloc(&d_C, sz * sizeof(int32_t));
    hipMalloc(&d_D, sz * sizeof(int32_t));

    hipMemcpy(d_A, h_A.data(), sz * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sz * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), sz * sizeof(int32_t), hipMemcpyHostToDevice);

    wmma_gemm_rdna3_iu4<<<1, 32>>>(d_A, d_B, d_C, d_D);
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
