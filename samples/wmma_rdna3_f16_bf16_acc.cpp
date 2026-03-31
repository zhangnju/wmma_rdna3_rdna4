// wmma_rdna3_f16_bf16_acc.cpp
// Build (repo root): hipcc --offload-arch=gfx1100 -O2 samples/wmma_rdna3_f16_bf16_acc.cpp -o wmma_rdna3_f16_bf16_acc
//
// Two RDNA 3 Wave32 kernels: D = A*B + C with FP16 or BF16 accumulators.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

typedef _Float16 fp16x16 __attribute__((ext_vector_type(16)));
typedef __attribute__((ext_vector_type(16))) __bf16 bf16x16;

__global__ void wmma_gemm_rdna3_f16acc(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ C,
          __half* __restrict__ D)
{
    const int lIdx  = threadIdx.x;
    const int lane  = lIdx % 16;
    const int wave2 = lIdx / 16;

    fp16x16 a_frag{};
    fp16x16 b_frag{};
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        a_frag[k] = (_Float16)A[k * 16 + lane];
        b_frag[k] = (_Float16)B[k * 16 + lane];
    }

    fp16x16 c_frag{};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = i * 2 + wave2;
        c_frag[i * 2] = (_Float16)C[r * 16 + lane];
    }

    fp16x16 d_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = i * 2 + wave2;
        D[r * 16 + lane] = (__half)d_frag[i * 2];
    }
}

__global__ void wmma_gemm_rdna3_bf16acc(
    const __bf16* __restrict__ A,
    const __bf16* __restrict__ B,
    const __bf16* __restrict__ C,
          __bf16* __restrict__ D)
{
    const int lIdx  = threadIdx.x;
    const int lane  = lIdx % 16;
    const int wave2 = lIdx / 16;

    bf16x16 a_frag{};
    bf16x16 b_frag{};
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        a_frag[k] = A[k * 16 + lane];
        b_frag[k] = B[k * 16 + lane];
    }

    bf16x16 c_frag{};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = i * 2 + wave2;
        c_frag[i * 2] = C[r * 16 + lane];
    }

    bf16x16 d_frag =
        __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(a_frag, b_frag, c_frag, false);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = i * 2 + wave2;
        D[r * 16 + lane] = d_frag[i * 2];
    }
}

static void fill_ramp_half(std::vector<__half>& m, float scale = 0.01f) {
    for (int i = 0; i < (int)m.size(); i++)
        m[i] = __float2half(i * scale);
}

static void cpu_gemm_f16_acc(const std::vector<__half>& A_cm,
                             const std::vector<__half>& B_rm,
                             const std::vector<__half>& C_rm,
                             std::vector<__half>& D_rm)
{
    constexpr int N = 16;
    for (int m = 0; m < N; m++)
        for (int n = 0; n < N; n++) {
            __half acc = C_rm[m * N + n];
            for (int k = 0; k < N; k++) {
                float p = __half2float(A_cm[k * N + m]) * __half2float(B_rm[k * N + n]);
                acc     = __float2half(__half2float(acc) + p);
            }
            D_rm[m * N + n] = acc;
        }
}

static float max_abs_diff_half(const std::vector<__half>& a, const std::vector<__half>& b) {
    float e = 0.f;
    for (size_t i = 0; i < a.size(); i++)
        e = std::fmax(e, std::fabs(__half2float(a[i]) - __half2float(b[i])));
    return e;
}

static float bf16_to_f32_u16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}
static uint16_t f32_to_bf16_u16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u = (u + (uint32_t)(0x7FFFU + ((u >> 16) & 1U))) & 0xFFFF0000U;
    return (uint16_t)(u >> 16);
}

static void cpu_gemm_bf16_acc_bf(const std::vector<uint16_t>& A_cm,
                                 const std::vector<uint16_t>& B_rm,
                                 const std::vector<uint16_t>& C_rm,
                                 std::vector<uint16_t>& D_rm)
{
    constexpr int N = 16;
    for (int m = 0; m < N; m++)
        for (int n = 0; n < N; n++) {
            float acc = bf16_to_f32_u16(C_rm[m * N + n]);
            for (int k = 0; k < N; k++)
                acc += bf16_to_f32_u16(A_cm[k * N + m]) * bf16_to_f32_u16(B_rm[k * N + n]);
            D_rm[m * N + n] = f32_to_bf16_u16(acc);
        }
}

static float max_abs_diff_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    float e = 0.f;
    for (size_t i = 0; i < a.size(); i++)
        e = std::fmax(e, std::fabs(bf16_to_f32_u16(a[i]) - bf16_to_f32_u16(b[i])));
    return e;
}

int main()
{
    constexpr int N = 16;
    constexpr int sz = N * N;

    {
        std::vector<__half> h_A(sz), h_B(sz), h_C(sz), h_D(sz), cpu_D(sz);
        fill_ramp_half(h_A);
        fill_ramp_half(h_B);
        for (int i = 0; i < sz; i++)
            h_C[i] = __float2half(0.f);
        h_C[0]  = __float2half(0.25f);
        h_C[17] = __float2half(-0.125f);

        cpu_gemm_f16_acc(h_A, h_B, h_C, cpu_D);

        __half *d_A, *d_B, *d_C, *d_D;
        hipMalloc((void**)&d_A, sz * sizeof(__half));
        hipMalloc((void**)&d_B, sz * sizeof(__half));
        hipMalloc((void**)&d_C, sz * sizeof(__half));
        hipMalloc((void**)&d_D, sz * sizeof(__half));
        hipMemcpy(d_A, h_A.data(), sz * sizeof(__half), hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), sz * sizeof(__half), hipMemcpyHostToDevice);
        hipMemcpy(d_C, h_C.data(), sz * sizeof(__half), hipMemcpyHostToDevice);

        wmma_gemm_rdna3_f16acc<<<1, 32>>>(d_A, d_B, d_C, d_D);
        hipDeviceSynchronize();
        hipMemcpy(h_D.data(), d_D, sz * sizeof(__half), hipMemcpyDeviceToHost);

        float err = max_abs_diff_half(cpu_D, h_D);
        std::cout << "[FP16 acc] CPU D[0][0] = " << __half2float(cpu_D[0])
                  << "  GPU = " << __half2float(h_D[0])
                  << "  max abs err = " << err << "\n";

        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        hipFree(d_D);
    }

    {
        std::vector<uint16_t> h_A(sz), h_B(sz), h_C(sz), h_D(sz), cpu_D(sz);
        for (int i = 0; i < sz; i++) {
            h_A[i] = f32_to_bf16_u16(i * 0.01f);
            h_B[i] = f32_to_bf16_u16(i * 0.01f);
        }
        h_C.assign(sz, 0);
        h_C[0]  = f32_to_bf16_u16(0.25f);
        h_C[17] = f32_to_bf16_u16(-0.125f);

        cpu_gemm_bf16_acc_bf(h_A, h_B, h_C, cpu_D);

        void *d_A, *d_B, *d_C, *d_D;
        hipMalloc(&d_A, sz * sizeof(uint16_t));
        hipMalloc(&d_B, sz * sizeof(uint16_t));
        hipMalloc(&d_C, sz * sizeof(uint16_t));
        hipMalloc(&d_D, sz * sizeof(uint16_t));
        hipMemcpy(d_A, h_A.data(), sz * sizeof(uint16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), sz * sizeof(uint16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_C, h_C.data(), sz * sizeof(uint16_t), hipMemcpyHostToDevice);

        wmma_gemm_rdna3_bf16acc<<<1, 32>>>(reinterpret_cast<const __bf16*>(d_A),
                                            reinterpret_cast<const __bf16*>(d_B),
                                            reinterpret_cast<const __bf16*>(d_C),
                                            reinterpret_cast<__bf16*>(d_D));
        hipDeviceSynchronize();
        hipMemcpy(h_D.data(), d_D, sz * sizeof(uint16_t), hipMemcpyDeviceToHost);

        float err = max_abs_diff_bf16(cpu_D, h_D);
        std::cout << "[BF16 acc] CPU D[0][0] f32 = " << bf16_to_f32_u16(cpu_D[0])
                  << "  GPU f32 = " << bf16_to_f32_u16(h_D[0])
                  << "  max abs err (in float) = " << err << "\n";

        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        hipFree(d_D);
    }

    return 0;
}
