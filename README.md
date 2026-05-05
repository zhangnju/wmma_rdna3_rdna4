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
| [wmma_rdna4_fp16.cpp](samples/wmma_rdna4_fp16.cpp) | FP16 输入 → FP32 累加 | GFX12 新 intrinsic，无 lane 重复 |
| [wmma_rdna4_fp8.cpp](samples/wmma_rdna4_fp8.cpp) | FP8(E4M3) 输入 → FP32 累加 | RDNA 4 专有 FP8 支持 |
| [wmma_rdna4_iu8.cpp](samples/wmma_rdna4_iu8.cpp) | INT8 输入 → INT32 累加 | GFX12 符号位标志语义变化 |
| [wmma_rdna4_iu4.cpp](samples/wmma_rdna4_iu4.cpp) | INT4 输入 → INT32 累加 | K=16 与 K=32 两种 tile 宽度 |

### 高级示例 / Advanced Samples

| 文件 | 说明 |
|------|------|
| [mlp_wmma_rdna4.cpp](samples/mlp_wmma_rdna4.cpp) | 两层 MLP 推理（`output = W2 * relu(W1 * input)`），演示链式 WMMA 与共享内存中转 |
| [tiled_gemm_rdna4.cpp](samples/tiled_gemm_rdna4.cpp) | 任意尺寸 M×N×K 矩阵的 Tiled GEMM（16 的倍数），支持命令行参数 |
| [rocwmma_example.cpp](samples/rocwmma_example.cpp) | 使用高层 rocWMMA 模板库，架构可移植（RDNA 3/4 及 CDNA） |

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
