// wmma_rdna4_iu8.cpp
// Compile (repo root): hipcc --offload-arch=gfx1201 -std=c++20 samples/wmma_rdna4_iu8.cpp -o wmma_rdna4_iu8
//
// D = A*B + C for 16×16 int8 tiles, int32 accumulators — RDNA 4 gfx12 Wave32 WMMA only.
//
// INT8 iu8 operands must match the same byte staging as Composable Kernel / RDNA3 iu8
// (samples/wmma_rdna3_iu8.cpp): global A column-major, B row-major, then shared swizzle
// into int8x16. Each hardware lane (threadIdx.x 0..31) then takes eight bytes:
//   lanes 0–15 → a16[0..7],   b16[0..7]
//   lanes 16–31 → a16[8..15], b16[8..15]
// and bit_cast to int32x2 for the gfx12 builtin. C/D use the gfx12 accumulator map
// (same as wmma_rdna4_fp16.cpp).
//
// The first two builtin arguments are signedness flags (true = signed int8), not negation;
// see Clang gfx12 WMMA tests / gemm_rocm MULTITYPE_EXTENSION.md.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

typedef int8_t int8x8 __attribute__((ext_vector_type(8)));
typedef int8_t int8x16 __attribute__((ext_vector_type(16)));
typedef int32_t int32x2 __attribute__((ext_vector_type(2)));
typedef int32_t int32x8 __attribute__((ext_vector_type(8)));

__global__ void wmma_gemm_rdna4_iu8(const int8_t* __restrict__ A,
                                    const int8_t* __restrict__ B,
                                    const int32_t* __restrict__ C,
                                    int32_t* __restrict__ D)
{
    __shared__ int8_t p_shared[16 * 16 * 2];
    const int lIdx    = threadIdx.x;
    const int lane    = lIdx % 16;
    const int lane_lo = lIdx / 2;
    const int lane_hi = lIdx % 2;

    int8_t a_temp[8];
    int8_t b_temp[8];
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

    int8x16 a16{}, b16{};
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
        b16[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8 + 16 * 16];
        a16[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8];
    }

    __syncthreads();

    const int k_half = lIdx / 16;
    int8x8 a8{}, b8{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a8[i] = a16[i + k_half * 8];
        b8[i] = b16[i + k_half * 8];
    }

    const int32x2 a_pk = __builtin_bit_cast(int32x2, a8);
    const int32x2 b_pk = __builtin_bit_cast(int32x2, b8);

    const int gl = threadIdx.x;
    int32x8 c_frag{};
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (gl / 16) * 8 + e;
        const int n = gl % 16;
        c_frag[e] = C[m * 16 + n];
    }

    constexpr bool signed_a = true;
    constexpr bool signed_b = true;
    constexpr bool clamp = false;

    const int32x8 d_frag = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        signed_a, a_pk, signed_b, b_pk, c_frag, clamp);

#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (gl / 16) * 8 + e;
        const int n = gl % 16;
        D[m * 16 + n] = d_frag[e];
    }
}

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
                const int32_t a = A_cm[k * N + m];
                const int32_t b = B_rm[k * N + n];
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

int main()
{
    int n_dev = 0;
    const hipError_t cnt_st = hipGetDeviceCount(&n_dev);
    if (cnt_st != hipSuccess || n_dev < 1) {
        std::cerr << "No ROCm-capable GPU is available (hipGetDeviceCount: ";
        if (cnt_st != hipSuccess)
            std::cerr << hipGetErrorString(cnt_st);
        else
            std::cerr << "0 devices";
        std::cerr << ").\n";
        std::cerr << "Run this binary on a machine with an AMD GPU and a working ROCm stack; "
                     "check `rocminfo` and driver install.\n";
        return 2;
    }

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
    h_C[0] = 10;
    h_C[17] = -7;

    std::vector<int32_t> cpu_D(sz);
    cpu_gemm_i8_i32(h_A, h_B, h_C, cpu_D);

    int8_t* d_A;
    int8_t* d_B;
    int32_t *d_C, *d_D;
    hipMalloc(&d_A, sz * sizeof(int8_t));
    hipMalloc(&d_B, sz * sizeof(int8_t));
    hipMalloc(&d_C, sz * sizeof(int32_t));
    hipMalloc(&d_D, sz * sizeof(int32_t));

    hipMemcpy(d_A, h_A.data(), sz * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sz * sizeof(int8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), sz * sizeof(int32_t), hipMemcpyHostToDevice);

    wmma_gemm_rdna4_iu8<<<1, 32>>>(d_A, d_B, d_C, d_D);
    hipDeviceSynchronize();
    const hipError_t st = hipGetLastError();
    if (st != hipSuccess) {
        std::cerr << "HIP error after kernel/sync: " << hipGetErrorString(st) << "\n";
        if (st == hipErrorNoDevice)
            std::cerr << "(No ROCm GPU — same as missing device at startup; use a real AMD GPU.)\n";
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        hipFree(d_D);
        return 1;
    }

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
