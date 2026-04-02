// wmma_rdna4_iu4.cpp
// Compile (repo root): hipcc --offload-arch=gfx1201 -std=c++20 samples/wmma_rdna4_iu4.cpp -o wmma_rdna4_iu4
//
// D = A*B + C: int4 operands (low nibble of int8), int32 accumulators — RDNA 4 gfx12 Wave32.
//   - 16×16×16 tile: __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12 (scalar int32 A/B per lane)
//   - 16×16×32 tile: __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12 (int32x2 A/B per lane)
//
// Global: A column-major M×K (index k*M+m), B row-major K×N (index k*N+n), C/D row-major M×N.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

typedef int8_t int8x8 __attribute__((ext_vector_type(8)));
typedef int8_t int8x16 __attribute__((ext_vector_type(16)));
typedef int32_t int32x2 __attribute__((ext_vector_type(2)));
typedef int32_t int32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ int32_t pack_iu4_x8(int8x8 v)
{
    uint32_t w = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        w |= (uint32_t)(v[i] & 15) << (4 * i);
    return (int32_t)w;
}

// Sixteen nibbles → two i32 words (same bit order as wmma_rdna3_iu4.cpp pack_iu4_x16).
__device__ __forceinline__ int32x2 pack_iu4_x16(int8x16 v)
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

// M=N=K=16
__global__ void wmma_gemm_rdna4_iu4_16x16x16(const int8_t* __restrict__ A,
                                             const int8_t* __restrict__ B,
                                             const int32_t* __restrict__ C,
                                             int32_t* __restrict__ D)
{
    const int lane = threadIdx.x;

    int8x8 a8{}, b8{};
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int mn = lane % 16;
        const int k = (lane / 16) * 8 + e;
        a8[e] = A[k * 16 + mn];
        b8[e] = B[k * 16 + mn];
    }

    const int32_t a_pk = pack_iu4_x8(a8);
    const int32_t b_pk = pack_iu4_x8(b8);

    int32x8 c_frag{};
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        c_frag[e] = C[m * 16 + n];
    }

    constexpr bool neg_a = true;
    constexpr bool neg_b = true;
    constexpr bool clamp = false;

    const int32x8 d_frag = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(
        neg_a, a_pk, neg_b, b_pk, c_frag, clamp);

#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        D[m * 16 + n] = d_frag[e];
    }
}

// M=N=16, K=32 — same lane split as 16×16×16 but 16 K indices per half-wave.
__global__ void wmma_gemm_rdna4_iu4_16x16x32(const int8_t* __restrict__ A,
                                             const int8_t* __restrict__ B,
                                             const int32_t* __restrict__ C,
                                             int32_t* __restrict__ D)
{
    constexpr int M = 16;
    constexpr int N = 16;

    const int lane = threadIdx.x;

    int8x16 a16{}, b16{};
#pragma unroll
    for (int e = 0; e < 16; e++) {
        const int mn = lane % 16;
        const int k = (lane / 16) * 16 + e;
        a16[e] = A[k * M + mn];
        b16[e] = B[k * N + mn];
    }

    const int32x2 a_pk = pack_iu4_x16(a16);
    const int32x2 b_pk = pack_iu4_x16(b16);

    int32x8 c_frag{};
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        c_frag[e] = C[m * N + n];
    }

    constexpr bool neg_a = true;
    constexpr bool neg_b = true;
    constexpr bool clamp = false;

    const int32x8 d_frag = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
        neg_a, a_pk, neg_b, b_pk, c_frag, clamp);

#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        D[m * N + n] = d_frag[e];
    }
}

static int32_t sx_i4(int8_t x)
{
    int32_t v = x & 15;
    if (v & 8)
        v |= ~15;
    return v;
}

static void cpu_gemm_i4_i32(int M, int N, int Kdim,
                            const std::vector<int8_t>& A_cm,
                            const std::vector<int8_t>& B_rm,
                            const std::vector<int32_t>& C_rm,
                            std::vector<int32_t>& D_rm)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int64_t acc = C_rm[m * N + n];
            for (int k = 0; k < Kdim; k++) {
                const int32_t a = sx_i4(A_cm[k * M + m]);
                const int32_t b = sx_i4(B_rm[k * N + n]);
                acc += (int64_t)a * (int64_t)b;
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

static int clamp_i4(int v)
{
    if (v > 7)
        v = 7;
    if (v < -8)
        v = -8;
    return v;
}

static bool run_case_16x16x16()
{
    constexpr int M = 16, N = 16, Kdim = 16;
    constexpr int szAB = M * Kdim;
    constexpr int szCD = M * N;

    std::vector<int8_t> h_A(szAB), h_B(szAB);
    std::vector<int32_t> h_C(szCD, 0), h_D(szCD, 0);

    for (int k = 0; k < Kdim; k++) {
        for (int m = 0; m < M; m++)
            h_A[k * M + m] = (int8_t)clamp_i4((m * Kdim + k) % 13 - 6);
    }
    for (int k = 0; k < Kdim; k++) {
        for (int n = 0; n < N; n++)
            h_B[k * N + n] = (int8_t)clamp_i4((k * N + n) % 11 - 5);
    }
    h_C[0] = 3;
    h_C[17] = -2;

    std::vector<int32_t> cpu_D(szCD);
    cpu_gemm_i4_i32(M, N, Kdim, h_A, h_B, h_C, cpu_D);

    int8_t *d_A, *d_B;
    int32_t *d_C, *d_D;
    hipMalloc(&d_A, szAB * sizeof(int8_t));
    hipMalloc(&d_B, szAB * sizeof(int8_t));
    hipMalloc(&d_C, szCD * sizeof(int32_t));
    hipMalloc(&d_D, szCD * sizeof(int32_t));
    hipMemcpy(d_A, h_A.data(), szAB * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), szAB * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), szCD * sizeof(int32_t), hipMemcpyHostToDevice);

    wmma_gemm_rdna4_iu4_16x16x16<<<1, 32>>>(d_A, d_B, d_C, d_D);
    hipDeviceSynchronize();
    hipMemcpy(h_D.data(), d_D, szCD * sizeof(int32_t), hipMemcpyDeviceToHost);

    const int err = max_abs_diff_i32(cpu_D, h_D);
    std::cout << "[16x16x16 iu4] CPU D[0] = " << cpu_D[0] << ", GPU D[0] = " << h_D[0] << "\n";
    std::cout << "[16x16x16 iu4] Max abs diff: " << err << "\n";

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_D);
    return err == 0;
}

static bool run_case_16x16x32()
{
    constexpr int M = 16, N = 16, Kdim = 32;
    constexpr int szA = M * Kdim;
    constexpr int szB = Kdim * N;
    constexpr int szCD = M * N;

    std::vector<int8_t> h_A(szA), h_B(szB);
    std::vector<int32_t> h_C(szCD, 0), h_D(szCD, 0);

    for (int k = 0; k < Kdim; k++) {
        for (int m = 0; m < M; m++)
            h_A[k * M + m] = (int8_t)clamp_i4((m * Kdim + k) % 13 - 6);
    }
    for (int k = 0; k < Kdim; k++) {
        for (int n = 0; n < N; n++)
            h_B[k * N + n] = (int8_t)clamp_i4((k * N + n) % 11 - 5);
    }
    h_C[0] = 3;
    h_C[17] = -2;

    std::vector<int32_t> cpu_D(szCD);
    cpu_gemm_i4_i32(M, N, Kdim, h_A, h_B, h_C, cpu_D);

    int8_t *d_A, *d_B;
    int32_t *d_C, *d_D;
    hipMalloc(&d_A, szA * sizeof(int8_t));
    hipMalloc(&d_B, szB * sizeof(int8_t));
    hipMalloc(&d_C, szCD * sizeof(int32_t));
    hipMalloc(&d_D, szCD * sizeof(int32_t));
    hipMemcpy(d_A, h_A.data(), szA * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), szB * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), szCD * sizeof(int32_t), hipMemcpyHostToDevice);

    wmma_gemm_rdna4_iu4_16x16x32<<<1, 32>>>(d_A, d_B, d_C, d_D);
    hipDeviceSynchronize();
    hipMemcpy(h_D.data(), d_D, szCD * sizeof(int32_t), hipMemcpyDeviceToHost);

    const int err = max_abs_diff_i32(cpu_D, h_D);
    std::cout << "[16x16x32 iu4] CPU D[0] = " << cpu_D[0] << ", GPU D[0] = " << h_D[0] << "\n";
    std::cout << "[16x16x32 iu4] Max abs diff: " << err << "\n";

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_D);
    return err == 0;
}

int main()
{
    const bool ok16 = run_case_16x16x16();
    const bool ok32 = run_case_16x16x32();
    return (ok16 && ok32) ? 0 : 1;
}
