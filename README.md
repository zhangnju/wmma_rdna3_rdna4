# AMD RDNA 3 & RDNA 4 WMMA 编程 / WMMA Programming on AMD RDNA 3 & RDNA 4

本仓库提供在 AMD RDNA 3（GFX11）和 RDNA 4（GFX12）消费级 GPU 上使用 **WMMA（Wave Matrix Multiply-Accumulate，波前矩阵乘累加）** 指令进行 HIP 编程的完整技术文档与示例代码。

This repository provides comprehensive technical documentation and sample code for HIP programming using **WMMA (Wave Matrix Multiply-Accumulate)** instructions on AMD RDNA 3 (GFX11) and RDNA 4 (GFX12) consumer GPUs.

---

## 文档 / Documentation

| 文件 | 说明 |
|------|------|
| [wmma_rdna3_rdna4.md](wmma_rdna3_rdna4.md) | 英文技术文章 / English technical article |
| [wmma_rdna3_rdna4_zh.md](wmma_rdna3_rdna4_zh.md) | 中文技术文章 / Chinese technical article |

文档内容涵盖：

- WMMA 原理：波前级矩阵乘累加 `D = A × B + C`，GEMM M×N×K 约定
- RDNA 3（GFX11）寄存器布局、lane 镜像机制、编译器 intrinsic 语法
- RDNA 4（GFX12）架构改进：2× FP16/BF16 吞吐、4× INT8 吞吐、新增 FP8/BF8 数据类型、消除 lane 重复、结构化稀疏（SWMMAC）
- RDNA 3 → RDNA 4 移植要点（breaking changes）
- Tiled GEMM 扩展方案（超过 16×16 的矩阵）
- rocWMMA 高层库使用

---

## 示例代码 / Sample Code

所有示例位于 [`samples/`](samples/) 目录，每个程序均含 GPU kernel、CPU 参考实现和正确性验证，可直接编译运行。

All samples are in the [`samples/`](samples/) directory. Each program includes a GPU kernel, a CPU reference implementation, and a correctness check.

### RDNA 3（GFX11）示例

| 文件 | 数据类型 | 说明 |
|------|----------|------|
| [wmma_rdna3_fp16.cpp](samples/wmma_rdna3_fp16.cpp) | FP16 输入 → FP32 累加 | 基础 FP16 WMMA，展示 lane 镜像与寄存器布局 |
| [wmma_rdna3_iu8.cpp](samples/wmma_rdna3_iu8.cpp) | INT8 输入 → INT32 累加 | 共享内存 swizzle 暂存操作数，C++20 位转换 |
| [wmma_rdna3_iu4.cpp](samples/wmma_rdna3_iu4.cpp) | INT4 输入 → INT32 累加 | nibble 打包，16 个 INT4 压入 int32×2 |
| [wmma_rdna3_f16_bf16_acc.cpp](samples/wmma_rdna3_f16_bf16_acc.cpp) | FP16/BF16 输入 → FP16/BF16 累加 | FP16 与 BF16 双路累加器示例 |

### RDNA 4（GFX12）示例

| 文件 | 数据类型 | 说明 |
|------|----------|------|
| [wmma_rdna4_fp16.cpp](samples/wmma_rdna4_fp16.cpp) | FP16 输入 → FP32 累加 | GFX12 新 intrinsic，无 lane 重复；含完整 lane mapping 注释（`VGPR[lane][j] = D[(lane/16)*8+j][lane%16]`） |
| [wmma_rdna4_fp8.cpp](samples/wmma_rdna4_fp8.cpp) | FP8(E4M3) 输入 → FP32 累加 | RDNA 4 专有 FP8 支持 |
| [wmma_rdna4_iu8.cpp](samples/wmma_rdna4_iu8.cpp) | INT8 输入 → INT32 累加 | GFX12 符号位标志语义变化 |
| [wmma_rdna4_iu4.cpp](samples/wmma_rdna4_iu4.cpp) | INT4 输入 → INT32 累加 | K=16 与 K=32 两种 tile 宽度 |

### 高级示例 / Advanced Samples

| 文件 | 说明 |
|------|------|
| [mlp_wmma_rdna4.cpp](samples/mlp_wmma_rdna4.cpp) | 两层 MLP 推理（`output = W2 * relu(W1 * input)`），演示链式 WMMA 与共享内存中转 |
| [tiled_gemm_rdna4.cpp](samples/tiled_gemm_rdna4.cpp) | 任意尺寸 M×N×K 矩阵的 Tiled GEMM（16 的倍数），支持命令行参数 |
| [gemm_fp16_opt_rdna.cpp](samples/gemm_fp16_opt_rdna.cpp) | **FP16 GEMM 八步优化教程**：从 naive 单波前逐步优化到接近 hipBLASLt 的内核，同一文件内含逐步 benchmark（ms / GFLOP-s / 正确性）。跨 RDNA 3 与 RDNA 4，已在 **W7900（gfx1100）** 与 **R9700（gfx1201）** 上验证 |
| [hipblaslt_bench.cpp](samples/hipblaslt_bench.cpp) | hipBLASLt FP16 GEMM 吞吐基准（best-of-heuristics），与上面的内核**同条件对比**（FP16 输入 / FP32 累加 / FP32 输出 / 同 shape / 同卡）。需链接 `-lhipblaslt` |
| [rocwmma_example.cpp](samples/rocwmma_example.cpp) | 使用高层 rocWMMA 模板库，架构可移植（RDNA 3/4 及 CDNA） |

#### FP16 GEMM 优化步骤（`gemm_fp16_opt_rdna.cpp`）

参照 ROCm 博客 [*FP16 GEMM Optimization on AMD RDNA 3 and RDNA 4*](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html)，将同一个 GEMM 内核分八步迭代优化，每一步只改动一个“旋钮”：

| 步骤 | 技术 | 说明 |
|------|------|------|
| 1 | baseline | 单波前负责 32×32 输出瓦片（2×2 个 16×16 WMMA），标量 LDS 拷贝 |
| 2 | tiling + 双缓冲 | 128×128 瓦片、16 波前、128-bit 宽加载、双 LDS 缓冲区软流水 |
| 3 | LDS padding | 行填充消除 shared memory bank conflict |
| 4 | OOB-safe | 带边界保护的宽加载，使 M、N 不必整除瓦片（博客中用 buffer resource descriptor） |
| 5 | deep-K | block_k 16→32，减少 barrier、每次同步做更多运算 |
| 6 | block scheduling | 分组启动 block，使相邻 block 复用 L2 中的 A/B |
| 7 | prefetch + s_setprio | WMMA 执行时将下一瓦片预取进寄存器，`s_setprio` 偏置矩阵指令发射 |
| 8 | single buffer | 去掉第二个 LDS 缓冲（LDS 减半、占用率翻倍）+ 一次额外 barrier |

**实测（8192³，20 次平均）**：

| GPU | baseline | 最终 | 加速比 | 峰值 |
|-----|----------|------|--------|------|
| W7900 (gfx1100) | 738 ms | 13.5 ms（step 8） | ~55× | 81.7 TFLOP/s |
| R9700 (gfx1201) | 733 ms | 12.6 ms（step 7） | ~58× | 87.0 TFLOP/s |

**与 hipBLASLt 对比（同一 `rocm/dev-ubuntu-24.04:10.0.0-full` 容器内，ROCm 10.0 / hipBLASLt 1.4.1，8192³）**：

| GPU | hipBLASLt | 本例最佳 | 相对 hipBLASLt |
|-----|-----------|----------|----------------|
| W7900 (gfx1100) | 14.69 ms（74.8 TFLOP/s） | **13.29 ms（step 7，82.8 TFLOP/s）** | **~110%，超过 hipBLASLt** |
| R9700 (gfx1201) | **10.61 ms（103.6 TFLOP/s）** | 12.64 ms（step 7，87.0 TFLOP/s） | ~84%，未超过 |

> hipBLASLt 数字由 [`samples/hipblaslt_bench.cpp`](samples/hipblaslt_bench.cpp) 在同一张卡、同一容器、同 shape、同精度（FP16 输入 / FP32 累加 / FP32 输出）下测得,可复现。

> 注 1：step 8（单缓冲）的收益与架构/尺寸相关。W7900 上 step 8 与 step 7 相当（均超过 hipBLASLt）；但 R9700 在 8192³ 下双缓冲的 step 7 明显更优，去掉双缓冲损失的预取/计算重叠超过了占用率提升的收益。
>
> 注 2：与博客结论一致——自定义内核在 RDNA 3（gfx1100）上可超过 hipBLASLt（博客 ~125%，本例 ~110%），但在 RDNA 4（gfx1201）上 hipBLASLt 调优更强，自定义内核约为其 84%~92%。

---

## 编译 / Build

所有示例使用 `hipcc` 直接编译，无需额外构建系统。

### RDNA 3（gfx1100）

```bash
# FP16
hipcc --offload-arch=gfx1100 samples/wmma_rdna3_fp16.cpp -o wmma_rdna3_fp16

# INT8 / INT4（需要 C++20）
hipcc --offload-arch=gfx1100 -std=c++20 samples/wmma_rdna3_iu8.cpp -o wmma_rdna3_iu8
hipcc --offload-arch=gfx1100 -std=c++20 samples/wmma_rdna3_iu4.cpp -o wmma_rdna3_iu4

# FP16/BF16 累加器
hipcc --offload-arch=gfx1100 samples/wmma_rdna3_f16_bf16_acc.cpp -o wmma_rdna3_f16_bf16_acc

# FP16 GEMM 八步优化（W7900 / gfx1100）
hipcc --offload-arch=gfx1100 -O3 -std=c++17 -mcumode \
  -Rpass-analysis=kernel-resource-usage \
  samples/gemm_fp16_opt_rdna.cpp -o gemm_fp16_opt_rdna
./gemm_fp16_opt_rdna 8192 8192 8192 20    # [M N K] [reps]，默认 4096 4096 4096 10

# hipBLASLt 吞吐基准（对比用，W7900）
hipcc --offload-arch=gfx1100 -O3 -std=c++17 samples/hipblaslt_bench.cpp -o hipblaslt_bench -lhipblaslt
./hipblaslt_bench 8192 8192 8192 20
```

### RDNA 4（gfx1201）

```bash
# FP16
hipcc --offload-arch=gfx1201 samples/wmma_rdna4_fp16.cpp -o wmma_rdna4_fp16

# FP8
hipcc --offload-arch=gfx1201 samples/wmma_rdna4_fp8.cpp -o wmma_rdna4_fp8

# INT8 / INT4（需要 C++20）
hipcc --offload-arch=gfx1201 -std=c++20 samples/wmma_rdna4_iu8.cpp -o wmma_rdna4_iu8
hipcc --offload-arch=gfx1201 -std=c++20 samples/wmma_rdna4_iu4.cpp -o wmma_rdna4_iu4

# MLP 与 Tiled GEMM
hipcc --offload-arch=gfx1201 samples/mlp_wmma_rdna4.cpp -o mlp_wmma_rdna4
hipcc --offload-arch=gfx1201 samples/tiled_gemm_rdna4.cpp -o tiled_gemm_rdna4

# FP16 GEMM 八步优化（R9700 / gfx1201）
hipcc --offload-arch=gfx1201 -O3 -std=c++17 -mcumode \
  -Rpass-analysis=kernel-resource-usage \
  samples/gemm_fp16_opt_rdna.cpp -o gemm_fp16_opt_rdna
./gemm_fp16_opt_rdna 8192 8192 8192 20

# hipBLASLt 吞吐基准（对比用，R9700）
hipcc --offload-arch=gfx1201 -O3 -std=c++17 samples/hipblaslt_bench.cpp -o hipblaslt_bench -lhipblaslt
./hipblaslt_bench 8192 8192 8192 20

# rocWMMA（需安装 rocWMMA 7.x+）
hipcc --offload-arch=native samples/rocwmma_example.cpp -o rocwmma_example
```

> `-std=c++20` 为 INT8/INT4 示例所必需（使用了 `__builtin_bit_cast`）。

---

## RDNA 3 vs RDNA 4 关键差异 / Key Differences

| 特性 | RDNA 3 (GFX11) | RDNA 4 (GFX12) |
|------|----------------|----------------|
| Intrinsic 后缀 | `_w32` | `_w32_gfx12` |
| A/B Lane 重复 | lane i 与 i+16 重复 | 无重复，每 lane 持有唯一数据 |
| FP16/BF16 吞吐 | 基准 | 2× |
| INT8 吞吐 | 基准 | 4× |
| FP8 支持 | 无 | E4M3 / E5M2 |
| INT4 Tile | 16×16×16 | 16×16×16 及 16×16×32 |
| 结构化稀疏 | 无 | SWMMAC（2:4 稀疏） |
| INT 布尔参数语义 | `neg_a / neg_b`（取反） | `a_sign / b_sign`（有符号标志） |

---

## 环境要求 / Requirements

- **ROCm 7.2+**
- **hipcc** 编译器
- RDNA 3 示例：RX 7000 系列 GPU（gfx1100/gfx1101/gfx1102）
- RDNA 4 示例：RX 9000 系列 GPU（gfx1200/gfx1201，如 RX 9070 XT）
- rocWMMA 示例：rocWMMA 7.x 库

---

## 参考资料 / References

- [AMD GPU ISA Reference Guide — GFX11](https://gpuopen.com/amd-gpu-isa-documentation/)
- [AMD GPU ISA Reference Guide — GFX12](https://gpuopen.com/amd-gpu-isa-documentation/)
- [AMD Matrix Instruction Calculator](https://gpuopen.com/matrix-instruction-calculator/)
- [rocWMMA Library](https://github.com/ROCm/rocWMMA)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [Matrix Core Programming on AMD CDNA3 and CDNA4](https://gpuopen.com/)
