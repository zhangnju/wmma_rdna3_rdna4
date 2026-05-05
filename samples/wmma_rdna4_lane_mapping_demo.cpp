// wmma_rdna4_lane_mapping_demo.cpp
// Demonstrates the gfx12 accumulator lane mapping:
//   VGPR[lane][j] = D[(lane/16)*8 + j][lane%16]
//
// Compile: hipcc --offload-arch=gfx1201 wmma_rdna4_lane_mapping_demo.cpp -o wmma_rdna4_lane_mapping_demo
// Tested:  ROCm 7.2, RDNA 4 (RX 9070 XT / gfx1201)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>

static const int M = 16, N = 16, K = 16;

// ---------------------------------------------------------------------------
// Device kernel: one wave (32 threads) computes D = A * B + C
//   A: 16x16 FP16, column-major (lda=16)
//   B: 16x16 FP16, row-major    (ldb=16)
//   C, D: 16x16 FP32, row-major (ldc=ldd=16)
// ---------------------------------------------------------------------------
__global__ void wmma_lane_mapping_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const float*  __restrict__ C,
          float*               D)
{
    int lane = threadIdx.x % 32;

    // ------------------------------------------------------------------
    // A operand: lane selects m-row and k-range
    //   mn = lane % 16          (A row index)
    //   k  = (lane / 16)*8 + e  (K offset, e=0..7)
    //   A is column-major: A[k][m] stored at A[k*16 + m]
    // ------------------------------------------------------------------
    using half8  = __attribute__((__vector_size__(8  * sizeof(__half))))  __half;
    using float8 = __attribute__((__vector_size__(8  * sizeof(float))))   float;

    int mn       = lane % 16;
    int k_base   = (lane / 16) * 8;

    half8 a_frag, b_frag;
    for (int e = 0; e < 8; e++) {
        int k = k_base + e;
        // A column-major: element (row=mn, col=k) → A[k * 16 + mn]
        a_frag[e] = A[k * 16 + mn];
        // B row-major:    element (row=k,  col=mn) → B[k * 16 + mn]
        b_frag[e] = B[k * 16 + mn];
    }

    // ------------------------------------------------------------------
    // C/D accumulator layout:
    //   col  = lane % 16
    //   row0 = (lane / 16) * 8
    //   slot j → matrix element at [row0+j][col]
    //   C row-major: C[row * 16 + col]
    // ------------------------------------------------------------------
    int out_col  = lane % 16;
    int out_row0 = (lane / 16) * 8;

    float8 c_frag;
    for (int j = 0; j < 8; j++) {
        int r = out_row0 + j;
        c_frag[j] = C[r * 16 + out_col];
    }

    // ------------------------------------------------------------------
    // WMMA intrinsic (gfx12, Wave32, FP16 → FP32)
    // ------------------------------------------------------------------
    float8 d_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        a_frag, b_frag, c_frag);

    // ------------------------------------------------------------------
    // Write D: same scatter pattern as C load
    //   D[row0+j][out_col] = d_frag[j]
    // ------------------------------------------------------------------
    for (int j = 0; j < 8; j++) {
        int r = out_row0 + j;
        D[r * 16 + out_col] = d_frag[j];
    }
}

// ---------------------------------------------------------------------------
// Host reference: D = A * B + C  (long double for accuracy)
//   A column-major, B row-major, C/D row-major
// ---------------------------------------------------------------------------
static void cpu_ref(const __half* A, const __half* B, const float* C, float* D)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            long double acc = (long double)C[m * N + n];
            for (int k = 0; k < K; k++) {
                // A column-major: A[k][m] → A[k*M + m]
                // B row-major:    B[k][n] → B[k*N + n]
                acc += (long double)(float)A[k * M + m]
                     * (long double)(float)B[k * N + n];
            }
            D[m * N + n] = (float)acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void check(hipError_t e, const char* msg)
{
    if (e != hipSuccess) {
        fprintf(stderr, "HIP error (%s): %s\n", msg, hipGetErrorString(e));
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // Host buffers
    __half h_A[M * K], h_B[K * N];
    float  h_C[M * N], h_D_gpu[M * N], h_D_ref[M * N];

    // Fill A and B with simple values; C with non-zero to test accumulation
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)(i % 8) * 0.125f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half((float)(i % 5) * 0.25f);
    for (int i = 0; i < M * N; i++) h_C[i] = (float)(i % 3) * 0.5f;

    // GPU buffers
    __half *d_A, *d_B;
    float  *d_C, *d_D;
    check(hipMalloc(&d_A, sizeof(h_A)), "malloc A");
    check(hipMalloc(&d_B, sizeof(h_B)), "malloc B");
    check(hipMalloc(&d_C, sizeof(h_C)), "malloc C");
    check(hipMalloc(&d_D, sizeof(h_D_gpu)), "malloc D");

    check(hipMemcpy(d_A, h_A, sizeof(h_A), hipMemcpyHostToDevice), "copy A");
    check(hipMemcpy(d_B, h_B, sizeof(h_B), hipMemcpyHostToDevice), "copy B");
    check(hipMemcpy(d_C, h_C, sizeof(h_C), hipMemcpyHostToDevice), "copy C");

    // Launch: 1 block × 32 threads (one Wave32)
    wmma_lane_mapping_kernel<<<1, 32>>>(d_A, d_B, d_C, d_D);
    check(hipGetLastError(), "kernel launch");
    check(hipDeviceSynchronize(), "sync");

    check(hipMemcpy(h_D_gpu, d_D, sizeof(h_D_gpu), hipMemcpyDeviceToHost), "copy D");

    // CPU reference
    cpu_ref(h_A, h_B, h_C, h_D_ref);

    // Compare
    float max_err = 0.f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_D_gpu[i] - h_D_ref[i]);
        if (err > max_err) max_err = err;
    }

    // Print a few elements to show correct [row][col] placement
    printf("Lane mapping demo — D = A*B + C on gfx1201\n");
    printf("Sample GPU results vs CPU reference (row-major):\n");
    printf("  D[0][0]  GPU=%.4f  CPU=%.4f\n", h_D_gpu[0],  h_D_ref[0]);
    printf("  D[0][1]  GPU=%.4f  CPU=%.4f\n", h_D_gpu[1],  h_D_ref[1]);
    printf("  D[1][0]  GPU=%.4f  CPU=%.4f\n", h_D_gpu[16], h_D_ref[16]);
    printf("  D[8][0]  GPU=%.4f  CPU=%.4f\n", h_D_gpu[8*16], h_D_ref[8*16]);
    printf("  D[15][15] GPU=%.4f  CPU=%.4f\n", h_D_gpu[15*16+15], h_D_ref[15*16+15]);
    printf("Max absolute error: %.2e\n", max_err);

    if (max_err > 1e-2f) {
        printf("FAIL: error too large\n");
        hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_D);
        return 1;
    }
    printf("PASS\n");

    hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_D);
    return 0;
}
