// mlp_wmma_rdna4.cpp
// Compile (repo root): hipcc --offload-arch=gfx1201 samples/mlp_wmma_rdna4.cpp -o mlp_wmma_rdna4
//
// Two-layer MLP tile: output = W2 * relu(W1 * input), 16×16, FP16 weights/activations,
// FP32 WMMA accum (gfx12). Shared memory stages relu(hidden) for the second WMMA B operand.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <iostream>
#include <vector>

typedef __attribute__((ext_vector_type(8))) _Float16 half8_t;
typedef __attribute__((ext_vector_type(8))) float    float8_t;

__device__ float8_t relu8(float8_t x) {
#pragma unroll
    for (int i = 0; i < 8; i++)
        x[i] = x[i] > 0.f ? x[i] : 0.f;
    return x;
}

__global__ void mlp_two_layer(const __half* __restrict__ W1,   // [16×16] col-major W1[k*16+m]
                              const __half* __restrict__ W2,   // [16×16] col-major
                              const __half* __restrict__ input, // [16×16] row-major input[k*16+n]
                              float* __restrict__ output)      // [16×16] row-major
{
    const int lane = threadIdx.x;
    __shared__ __half smem_h[16 * 16];

    half8_t w1_frag, in_frag;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = lane % 16;
        const int k = (lane / 16) * 8 + e;
        const int n = lane % 16;
        w1_frag[e]  = (_Float16)W1[k * 16 + m];
        in_frag[e] = (_Float16)input[k * 16 + n];
    }

    float8_t acc0{};
    float8_t h = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(w1_frag, in_frag, acc0);

    h = relu8(h);

#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int row = (lane / 16) * 8 + e;
        const int col = lane % 16;
        smem_h[row * 16 + col] = __float2half(h[e]);
    }
    __syncthreads();

    half8_t w2_frag, hid_frag;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = lane % 16;
        const int k = (lane / 16) * 8 + e;
        const int n = lane % 16;
        w2_frag[e]  = (_Float16)W2[k * 16 + m];
        hid_frag[e] = (_Float16)smem_h[k * 16 + n];
    }

    float8_t acc1{};
    float8_t result = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(w2_frag, hid_frag, acc1);

#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        output[m * 16 + n] = result[e];
    }
}

// CPU: D[m,n] = sum_k A_cm[k,m]*B_rm[k,n], same layouts as single-tile GEMM in wmma_rdna4_fp16.cpp
static void cpu_gemm_f16_col_row(const std::vector<__half>& A_cm, const std::vector<__half>& B_rm,
                                 std::vector<float>& D_rm) {
    constexpr int N = 16;
    for (int m = 0; m < N; m++)
        for (int n = 0; n < N; n++) {
            long double acc = 0.L;
            for (int k = 0; k < N; k++)
                acc += (long double)__half2float(A_cm[k * N + m]) *
                       (long double)__half2float(B_rm[k * N + n]);
            D_rm[m * N + n] = (float)acc;
        }
}

static void cpu_mlp_ref(const std::vector<__half>& W1_cm, const std::vector<__half>& W2_cm,
                        const std::vector<__half>& in_rm, std::vector<float>& out_rm) {
    constexpr int N = 16;
    const int       sz = N * N;
    std::vector<float> z_f(sz);
    cpu_gemm_f16_col_row(W1_cm, in_rm, z_f);
    std::vector<__half> z_h(sz);
    for (int i = 0; i < sz; i++) {
        const float v = z_f[i] > 0.f ? z_f[i] : 0.f;
        z_h[i]        = __float2half(v);
    }
    cpu_gemm_f16_col_row(W2_cm, z_h, out_rm);
}

int main() {
    constexpr int N = 16, sz = N * N;

    std::vector<__half> h_W1(sz), h_W2(sz), h_in(sz);
    std::vector<float>  h_out(sz, 0.f);

    for (int i = 0; i < sz; i++) {
        h_W1[i] = __float2half(0.02f * float(i % 31) - 0.25f);
        h_W2[i] = __float2half(0.03f * float((i + 5) % 29) - 0.2f);
        h_in[i] = __float2half(0.015f * float(i) / float(sz));
    }

    std::vector<float> cpu_out(sz);
    cpu_mlp_ref(h_W1, h_W2, h_in, cpu_out);

    __half *d_W1, *d_W2, *d_in;
    float*  d_out;
    hipMalloc(&d_W1, sz * sizeof(__half));
    hipMalloc(&d_W2, sz * sizeof(__half));
    hipMalloc(&d_in, sz * sizeof(__half));
    hipMalloc(&d_out, sz * sizeof(float));
    hipMemcpy(d_W1, h_W1.data(), sz * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), sz * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_in, h_in.data(), sz * sizeof(__half), hipMemcpyHostToDevice);

    mlp_two_layer<<<1, 32>>>(d_W1, d_W2, d_in, d_out);
    hipDeviceSynchronize();
    hipMemcpy(h_out.data(), d_out, sz * sizeof(float), hipMemcpyDeviceToHost);

    float max_err = 0.f;
    for (int i = 0; i < sz; i++)
        max_err = std::fmax(max_err, std::fabs(cpu_out[i] - h_out[i]));
    std::cout << "CPU ref out[0] = " << cpu_out[0] << ", GPU out[0] = " << h_out[0] << "\n";
    std::cout << "Max abs error (CPU vs GPU): " << max_err << "\n";

    hipFree(d_W1);
    hipFree(d_W2);
    hipFree(d_in);
    hipFree(d_out);

    return max_err > 2e-2f ? 1 : 0;
}
