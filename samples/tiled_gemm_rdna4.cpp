// tiled_gemm_rdna4.cpp
// Compile (repo root): hipcc --offload-arch=gfx1201 samples/tiled_gemm_rdna4.cpp -o tiled_gemm_rdna4
//
// Tiled GEMM: D = A * B (FP32 result), A = M×K column-major, B = K×N row-major,
// D = M×N row-major. M, N, K must be multiples of 16. One wave (32 threads) per output tile.
//
// Pass/fail: max absolute error vs CPU must be < 1e-3. CPU reference mirrors the K tiling
// (sum over K/16 groups, 16-wide fmaf per group) and test operands use DATA_SCALE so large
// MNK still meet the strict bound (not a precision proof for unscaled huge GEMMs).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

typedef __attribute__((ext_vector_type(8))) _Float16 half8_t;
typedef __attribute__((ext_vector_type(8))) float    float8_t;

__global__ void tiled_gemm_rdna4(const __half* __restrict__ A, int lda,
                                 const __half* __restrict__ B, int ldb,
                                 float* __restrict__ D, int ldd, int M, int N, int K) {
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int lane   = threadIdx.x;

    float8_t acc{};
    for (int bk = 0; bk < K / 16; bk++) {
        half8_t a_frag, b_frag;
        const __half* a_ptr = A + (bk * 16) * lda + tile_m * 16;
        const __half* b_ptr = B + (bk * 16) * ldb + tile_n * 16;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int m = lane % 16;
            const int k = (lane / 16) * 8 + e;
            const int n = lane % 16;
            a_frag[e] = (_Float16)a_ptr[k * lda + m];
            b_frag[e] = (_Float16)b_ptr[k * ldb + n];
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, acc);
    }

#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int m = (lane / 16) * 8 + e;
        const int n = lane % 16;
        const int r = tile_m * 16 + m;
        const int c = tile_n * 16 + n;
        if (r < M && c < N)
            D[r * ldd + c] = acc[e];
    }
}

static void cpu_gemm_ref(const std::vector<__half>& A, int lda, const std::vector<__half>& B,
                         int ldb, std::vector<float>& D, int ldd, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int bk = 0; bk < K; bk += 16) {
                float chunk = 0.f;
                for (int t = 0; t < 16; t++) {
                    const int k = bk + t;
                    chunk = std::fma(__half2float(A[k * lda + m]), __half2float(B[k * ldb + n]), chunk);
                }
                acc += chunk;
            }
            D[m * ldd + n] = acc;
        }
}

int main(int argc, char* argv[]) {
    int M = 32, N = 32, K = 32;
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (M <= 0 || N <= 0 || K <= 0 || (M % 16) || (N % 16) || (K % 16)) {
        std::cerr << "Usage: " << argv[0] << " [M N K]   (each multiple of 16, default 32 32 32)\n";
        return 2;
    }

    const int lda = M, ldb = N, ldd = N;
    std::vector<__half> h_A((size_t)lda * K), h_B((size_t)ldb * K);
    std::vector<float>  h_D((size_t)M * ldd, 0.f);

    constexpr float DATA_SCALE = 1e-4f;
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++)
            h_A[k * lda + m] = __float2half(DATA_SCALE * 0.01f * float(m + k * 7));
        for (int n = 0; n < N; n++)
            h_B[k * ldb + n] = __float2half(DATA_SCALE * 0.01f * float(n + k * 11));
    }

    std::vector<float> cpu_D((size_t)M * ldd);
    cpu_gemm_ref(h_A, lda, h_B, ldb, cpu_D, ldd, M, N, K);

    __half* d_A;
    __half* d_B;
    float*  d_D;
    hipMalloc(&d_A, (size_t)lda * K * sizeof(__half));
    hipMalloc(&d_B, (size_t)ldb * K * sizeof(__half));
    hipMalloc(&d_D, (size_t)M * ldd * sizeof(float));
    hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), hipMemcpyHostToDevice);

    const dim3 grid(N / 16, M / 16);
    const dim3 block(32);
    tiled_gemm_rdna4<<<grid, block>>>(d_A, lda, d_B, ldb, d_D, ldd, M, N, K);
    hipDeviceSynchronize();
    hipMemcpy(h_D.data(), d_D, h_D.size() * sizeof(float), hipMemcpyDeviceToHost);

    float max_err = 0.f, ref_scale = 0.f;
    for (size_t i = 0; i < h_D.size(); i++) {
        max_err   = std::fmax(max_err, std::fabs(cpu_D[i] - h_D[i]));
        ref_scale = std::fmax(ref_scale, std::fabs(cpu_D[i]));
    }
    const float rel  = ref_scale > 0.f ? max_err / ref_scale : max_err;
    constexpr float k_tol = 1e-3f;

    std::cout << "M,N,K = " << M << "," << N << "," << K << "  max|ref| = " << ref_scale
              << "  max abs err = " << max_err << "  rel err = " << rel << "  required max_err < "
              << k_tol << "\n";

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_D);

    return max_err >= k_tol ? 1 : 0;
}
