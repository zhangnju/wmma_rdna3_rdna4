# AMD RDNA 3 与 RDNA 4 GPU 上的 WMMA（波前矩阵乘累加）编程

## 概述

AMD RDNA 3（GFX11）架构引入了 Wave Matrix Multiply-Accumulate（WMMA，波前矩阵乘累加）——一套面向 AI 与高性能计算的硬件加速矩阵乘法指令集。RDNA 4（GFX12）在此基础上将吞吐量翻倍、简化寄存器布局，并新增 FP8 等数据类型及结构化稀疏支持。

本文介绍 WMMA 的工作原理、如何用 WMMA 编译器 intrinsic 编写 HIP kernel，以及从 RDNA 3 迁到 RDNA 4 时需要改动的要点。

> 说明：WMMA 是消费级 GPU 上对应数据中心 CDNA（MI 系列）MFMA 的类似能力。二者都做 D = A × B + C 的矩阵融合乘加，但 intrinsic 名称、分块尺寸与寄存器约定在两代架构间差异很大。

---

## 背景：WMMA 是什么？

WMMA 即 Wave Matrix Multiply-Accumulate（波前矩阵乘累加），计算：

```
D = A × B + C
```

其中 A、B、C、D 均为小型 2D 矩阵分块，由波前中所有 lane 协同完成。与常规 GPU 算术（每线程独立处理标量/向量）不同，WMMA 是*波前级操作*——整个波前协作产生一块矩阵输出。

该模式便于硬件优化数据复用、将取数与计算重叠调度，从而接近峰值算力，思路与 NVIDIA Tensor Core 相近。

### GEMM 记法

分块运算遵循 GEMM 的 M×N×K 约定：

| 矩阵 | 形状 | 作用 |
|------|------|------|
| A | M × K | 左操作数 |
| B | K × N | 右操作数 |
| C / D | M × N | 累加器输入 / 输出 |

在 RDNA 3 与 RDNA 4 上，WMMA 仅支持 16×16 分块，即 M = N = K = 16。更大矩阵需拆成 16×16 子块。

### 内存布局约定

| 矩阵 | 布局 |
|------|------|
| A | 列主序（Column-major） |
| B | 行主序（Row-major） |
| C / D | 行主序（Row-major） |

---

## RDNA 3 (GFX11) WMMA

### 支持的数据类型

RDNA 3 支持的 WMMA 类型组合如下：

| 输入（A、B） | 累加器（C、D） | Intrinsic 后缀 |
|-------------|---------------|----------------|
| FP16 | FP16 | `f16_16x16x16_f16` |
| FP16 | FP32 | `f32_16x16x16_f16` |
| BF16 | FP32 | `f32_16x16x16_bf16` |
| BF16 | BF16 | `bf16_16x16x16_bf16` |
| INT8 | INT32 | `i32_16x16x16_iu8` |
| INT4 | INT32 | `i32_16x16x16_iu4` |

### 波前模式：Wave32 与 Wave64

**Wave32** 与 **Wave64** 是 **波前宽度（wavefront size）**：多少个线程（lane）成组以锁步执行同一条指令，与 CUDA 的 warp 类似。Wave32 每波 32 条 lane，Wave64 为 64。RDNA 3 的 WMMA 可通过 `*_w32` 与 `*_w64` 编译器 intrinsic 在两种模式下发出。Wave32 是 RDNA 3 示例代码与文档中的常见默认。

通过 Wave32 或 Wave64 编写 WMMA 时，需考虑下列因素：

- **占用率与 launch 形状：** 更宽的波会改变在寄存器与 SIMD 资源固定时能同时驻留的波数，因此峰值占用率并非某一种宽度自动更好——取决于具体 kernel。建议将线程块大小取为 32 或 64 的倍数，以免最后一波大量空转。

- **控制流与 wave 级操作：** 波内分支发散时，硬件仍对两条路径做掩码执行；波越宽，单次发散区域牵连的 lane 越多。定义在「当前波」上的 intrinsic 与惯用法（如 ballot、shuffle 一类）也随波宽而变：Wave64 kernel 里参与的一组是 64 条 lane，而非 32。

- **一致性：** 同一 kernel 内波前模式须一致，请勿在同一入口的同一编译单元中混用 `*_w32` 与 `*_w64` 的 WMMA builtin。HIP/Clang 的选项与属性可选定或提示波前模式。

- **分片（fragment）与 VGPR：** 分片（`*_frag`）是每条 lane 在一次 WMMA 中置于 VGPR 的 A/B/C/D 寄存器映像的一部分；rocWMMA 与 CUDA WMMA 使用相同术语。Wave32 与 Wave64 下 A、B 的 VGPR 数量相同，C/D 则不同。对 FP16/BF16 输入，每条 lane 的 `A_frag` 与 `B_frag` 各含 16 个半精度元素并打包在 VGPR 中，即 Wave32 与 Wave64 下每条 lane 各需 8 个 VGPR。相应地，Int8 输入时每条 lane 的 `A_frag`/`B_frag` 需 4 个 VGPR，Int4 为 2 个。`C_frag`/`D_frag` 每条 lane 需要多少个 32 位 VGPR 取决于累加器元素类型与波前模式：FP32/INT32 累加在 Wave32 下每条 lane 需 8 个，Wave64 下 4 个；打包 FP16/BF16 累加在 Wave32 下每条 lane 需 4 个，Wave64 下 2 个。

### RDNA 3 寄存器布局

**寄存器布局** 在此指 WMMA 分块中每个矩阵元素如何映射到具体的 lane 与 VGPR，包括 A/B 在 lane 间的复制，以及 C/D 的行、列如何与 `thread`/`lane` 绑定——换句话说，在硬件寄存器层面上，调用 intrinsic 前哪个线程装入哪一项内存元素、返回后哪个线程写回哪一项。

下面以 Wave32、FP16 输入 / FP32 累加器为例介绍 RDNA 3 WMMA 布局。

在 Wave32 FP16 WMMA 中有 32 个线程，但 A、B 只需 16 种不同的操作数输入；每种输入在 lane *i* 与 *i*+16 上重复（两侧寄存器相同）。将这 16 种配置编号为 t = 0…15：t 是 A 分块的一行下标，也是 B 分块的一列下标，且 A、B 使用同一个 t。这一 A/B 复制特性在 RDNA 4 上被消除——这是下文要谈的关键改进之一。

在 A 为列主序（`A[m,k]` 位于 `k*16+m`）、B 为行主序（`B[k,n]` 位于 `k*16+n`）时，AMD GPUOpen 与 Composable Kernel 测试采用：

- `a_frag[k]` = A[t,k]，k = 0…15 —— A 分块的第 t 行。
- `b_frag[k]` = B[k,t]，k = 0…15 —— B 分块的第 t 列。

对 FP32 的 C 与 D（累加器），32 条线程通过「每个输出列两条线程」填满 16×16 输出分块：一条线程提供该列偶数行的 8 个累加器，另一条提供奇数行的 8 个，列完整且无重叠。在常见的块内编号下，这两条线程相距 16 个位置。指令执行后，结果按行主序散布写回——先完整一行从左到右，再下一行。示例 *samples/wmma_rdna3_fp16.cpp* 与 Composable Kernel 的 WMMA 测试采用同一约定。

### Intrinsic 语法

以 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` 为例，可从左到右理解为：`f32` 表示 C/D（累加器/结果）元素类型，`16x16x16` 表示 WMMA 分块形状（M×N×K），`f16` 表示 A/B 输入元素类型，`_w32` / `_w64` 表示波前模式。整数变体会替换类型字段（如 `i32_16x16x16_iu8`），并带额外控制位（`neg_a`、`neg_b`、`clamp`）。在 RDNA 4 上，命名规则相同但会追加 `_gfx12`。

FP32 累加器（如 `f32_16x16x16_f16`、`f32_16x16x16_bf16`）：Clang 为三参数形式——无 `OPSEL`：

```c
D_frag = __builtin_amdgcn_wmma_f32_16x16x16_<AB_type>_w<32|64>(
    A_frag, B_frag, C_frag);
```

FP16 / BF16 累加器（如 `f16_16x16x16_f16`）：第四个参数 `OPSEL` 选择打包结果 VGPR 中低 16 位还是高 16 位（`false` / `true`）。

```c
D_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w<32|64>(
    A_frag, B_frag, C_frag, OPSEL);
```

整数 WMMA 带有 `neg_a` / `neg_b` / `clamp` 及打包操作数。`i32_16x16x16_iu8` 使用 `int32x4`（由 16×`int8` 经 `__builtin_bit_cast`；示例代码见 [Composable Kernel](https://github.com/ROCm/composable_kernel/blob/develop/include/ck/utility/amd_wmma.hpp#L127-L137)）。`i32_16x16x16_iu4` 使用 `int32x2`（16 个 int4 半字节装在两个 `int32` 中；`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`）。CK 在 Wave32 iu8 上令 `neg_a`、`neg_b` 均为 `true`；在启用 `CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4` 时，iu4 与 [`builtin_wmma_naive_selector`](https://github.com/ROCm/composable_kernel/blob/develop/test/wmma_op/wmma_op_util.hpp#L84-L96) 中的约定一致。操作数装片遵循 [`matmul` 代码](https://github.com/ROCm/composable_kernel/blob/develop/test/wmma_op/wmma_op_util.hpp#L98-L192) 中的共享内存 swizzle，而非 FP16 式的全局直接加载。Composable Kernel 的 WMMA op 测试在全局内存中使用 **A 行主序 M×K**、**B 列主序 K×N**；本仓库 **`samples/wmma_rdna3_iu8.cpp`** 与 **`samples/wmma_rdna3_iu4.cpp`** 则采用与 FP16 示例一致的 **A 列主序**（`A[m,k]` 在 `k*16+m`）、**B 行主序**（`B[k,n]` 在 `k*16+n`），并映射全局加载，使写入共享内存的字节与 CK 的 staging 一致。开发者可在项目中参考这些示例代码。

### 示例：FP16 输入、FP32 输出（Wave32，RDNA 3）

本示例用 RDNA 3 WMMA intrinsic 实现 16×16×16 GEMM（FP16 输入、FP32 累加器），并在 Host 侧用 CPU 参考结果校验 GPU 输出。

设备端 kernel 为 A、B 装入 16 宽的 FP16 分片、为 C 每线程装入 8 个 float（分布在波的两个 16-lane 半波上），再调用 WMMA intrinsic，将结果写入 D。

已在 ROCm 7.2 与 RDNA 3 GPU 上测试。

```bash
hipcc --offload-arch=gfx1100 samples/wmma_rdna3_fp16.cpp -o wmma_rdna3_fp16
./wmma_rdna3_fp16
```

### 示例：INT8 输入、INT32 输出（Wave32，RDNA 3）

INT8 的 WMMA 路径不会像 FP16 输入那样，把操作数用一次简单的全局加载直接装进分片。在 **`samples/wmma_rdna3_iu8.cpp`** 中，**A** 为列主序 **M×K**，**B** 为行主序 **K×N**，**C**/**D** 为行主序 **M×N**（与上文 FP16 输入、FP32 累加的约定一致）。每条线程先从 **A**、**B** 各读出八个 8 位元素，下标随 lane 变化；波的前半与后半分别覆盖沿 **K** 的互补条带，对应 **A** 的固定行与 **B** 的固定列。数据写入共享内存后，经与 Composable Kernel [`matmul`](https://github.com/ROCm/composable_kernel/blob/develop/test/wmma_op/wmma_op_util.hpp#L98-L192) 相同的重排，再发出 RDNA 3 Wave32 的 iu8 WMMA，两个操作数取反标志均置位，与 [`builtin_wmma_naive_selector` 的 8 位路径](https://github.com/ROCm/composable_kernel/blob/develop/test/wmma_op/wmma_op_util.hpp#L72-L82) 一致。CK 的 WMMA op 测试在全局内存仍采用 **A 行主序、B 列主序**；本示例的布局与之不同，但 staging 之后送入硬件的操作数字节与 CK 管线一致。**C**/**D** 的累加器分布与前面 FP16→FP32 示例相同：每个输出列由两条线程分工，分别负责偶数行与奇数行上的八个累加位置。Host 侧用更宽的整数做乘加，再把整块结果收束到 32 位整数后与 GPU 对比。

源文件：`samples/wmma_rdna3_iu8.cpp`。

```bash
hipcc --offload-arch=gfx1100 -std=c++20 samples/wmma_rdna3_iu8.cpp -o wmma_rdna3_iu8
./wmma_rdna3_iu8
```

### 示例：INT4 输入、INT32 输出（Wave32，RDNA 3）

**samples/wmma_rdna3_iu4.cpp** 的讲解与 INT8 示例同属一套套路：**A** 列主序 **M×K**，**B** 行主序 **K×N**，**C**/**D** 行主序 **M×N**；操作数仍先经共享内存重排，再发带取反标志的整数 WMMA。差别在精度：**A**、**B** 虽用字节数组存放，但每个元素在数学上是 **4 位有符号**（−8～7）。打包进指令时 **只认每个字节的低 4 位**，高 4 位视为不参与运算。每条 lane 先凑齐十六个这样的 4 位数（在寄存器里仍以十六个字节形式出现），再按固定顺序 **把十六个半字节压进两个 32 位字**，然后才发出 RDNA 3 Wave32 的 INT4 WMMA。若你的芯片或工具链对这两个字内部半字节顺序有不同约定，应改打包逻辑，使其与 Clang 与 ISA 说明一致。

源文件：`samples/wmma_rdna3_iu4.cpp`。


```bash
hipcc --offload-arch=gfx1100 -std=c++20 samples/wmma_rdna3_iu4.cpp -o wmma_rdna3_iu4
./wmma_rdna3_iu4
```


### 示例：FP16 输入，FP16 / BF16 累加器（Wave32，RDNA 3）

前文示例里 **C**、**D** 为完整 FP32 分块。RDNA 3 另有 WMMA 变体，使累加器与结果仍为 **打包的 FP16 或 BF16**：每个 32 位寄存器槽里放两个半精度元素。这类 builtin 在三个操作数分片之后还有 **第四个选择项**：一种用法让每个 32 位槽的 **低 16 位**表示矩阵元素，另一种用 **高 16 位**（见 [GPUOpen — RDNA 3 上的 WMMA](https://gpuopen.com/learn/wmma_on_rdna3)）。

**A**、**B** 的装入规则与 FP32 累加示例相同：**A** 列主序，每条 lane 对应一行下标；**B** 行主序，每条 lane 对应一列下标。**C**、**D** 仍为 **每条 lane 八个输出位置**，在同一输出列内按偶数行与奇数行分给不同线程。选用“低 16 位”模式时，这八个值各占成对寄存器槽里的 **前半段**；读回结果时同样只取该半段，再按与先前示例一致的行、列含义写回 **D** 的行主序缓冲。

Host 校验时，FP16 路径在规约过程中按半精度写回做舍入，比纯 FP32 仿真更贴近设备。BF16 路径在 CPU 侧用 **无符号 16 位**保存位型，内核侧使用工具链提供的 **BF16 类型**（需较新的 ROCm HIP-Clang）。硬件对 BF16 的融合与顺序可能与 CPU 上简单的标量 float 循环不一致，在浮点域对比时可预期 **比 FP16 稍宽**的误差。

```bash
hipcc --offload-arch=gfx1100 -O2 samples/wmma_rdna3_f16_bf16_acc.cpp -o wmma_rdna3_f16_bf16_acc
./wmma_rdna3_f16_bf16_acc
```

---

## RDNA 4 (GFX12) WMMA

RDNA 4 引入第三代 Matrix Core，主要变化包括：

| 指标 | RDNA 3 | RDNA 4 |
|------|--------|--------|
| 每 CU 每时钟 FP16/BF16 FLOPS | 256 | 512（约 2×，按 AMD 架构公开信息；以具体 SKU 为准） |
| 每 CU 每时钟 INT8 FLOPS | 256 | 1024（约 4×，按 AMD 架构公开信息；以具体 SKU 为准） |
| FP8 支持 | 无 | 有（E4M3 / E5M2） |
| 结构化稀疏 | 无 | 有（4:2，经 SWMMAC） |
| 寄存器复制 | 有（A/B 在 lane 16–31） | 已取消 |
| 每 lane VGPR（FP16，Wave32） | 16×`half` 逻辑（8 对 VGPR） | 8×`half` 作为 `half8_t`（4 对 VGPR）；无镜像 lane |

> 破坏性变更：RDNA 4 的 WMMA intrinsic 与 RDNA 3 不兼容。所有 RDNA 4 intrinsic 带 `_gfx12` 后缀，且操作数与累加器的 lane 映射不同（见下文）。LLVM 还提供 Wave64 的 `_w64_gfx12` 变体；本文示例使用 Wave32。

### 寄存器布局：无 lane 镜像（RDNA 4）

相对 RDNA 3 的主要改进是去掉冗余的 A/B 复制：在 GFX11 上，lane 16–31 与 0–15 携带相同 A/B 操作数。在 GFX12 上，每条 lane 持有 8 个互不相同的 FP16（`half8_t`），无镜像 lane。

```
RDNA 3（Wave32）—— A/B 操作数（概念）：
  Lane  0–15：主数据
  Lane 16–31：与 0–15 重复   ← 浪费的取数/寄存器流量

RDNA 4（Wave32）—— A/B 操作数：
  Lane 0–31：每 lane 8 个独立 FP16（无镜像）
```

链式 WMMA（如 MLP）仍需要布局变换：在 gfx12 上，A/B 操作数布局 ≠ D 累加器布局（见下文公式）。通常将 D 以行主序写入共享或全局内存，再按正确映射加载下一层操作数，或使用 rocWMMA / Composable Kernel。参见 [ROCm issue #6025](https://github.com/ROCm/ROCm/issues/6025)。

### RDNA 4 的新数据类型

RDNA 4 增加 FP8 与 BF8（brain float 8-bit）WMMA：

| 输入（A、B） | 累加器（C、D） | Intrinsic 后缀（gfx12） |
|-------------|---------------|--------------------------|
| FP16 | FP16 | `f16_16x16x16_f16_w32_gfx12` |
| BF16 | BF16 | `bf16_16x16x16_bf16_w32_gfx12` |
| FP16 | FP32 | `f32_16x16x16_f16_w32_gfx12` |
| BF16 | FP32 | `f32_16x16x16_bf16_w32_gfx12` |
| FP8（E4M3） | FP32 | `f32_16x16x16_fp8_fp8_w32_gfx12` |
| BF8（E5M2） | FP32 | `f32_16x16x16_bf8_bf8_w32_gfx12` |
| 混合 FP8/BF8 | FP32 | `f32_16x16x16_fp8_bf8_w32_gfx12` |
| INT8 | INT32 | `i32_16x16x16_iu8_w32_gfx12` |
| INT4 | INT32 | `i32_16x16x16_iu4_w32_gfx12` |
| INT4 | INT32 | `i32_16x16x32_iu4_w32_gfx12`（更大 K）；操作数布局见 ISA PDF / [Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) |

此外，SWMMAC（Sparse Wave Matrix Multiply-Accumulate）利用 4:2 结构化稀疏可再获约 2× 吞吐。

### RDNA 4 Intrinsic 语法

与 RDNA 3 相比，gfx12 对打包 FP16、BF16 累加器结果（如 `f16_16x16x16_f16_*`、`bf16_16x16x16_bf16_*`）去掉 `OPSEL`：输出始终稠密打包。

对浮点 WMMA 族，Wave32 形式为：

```c
D_frag = __builtin_amdgcn_wmma_<CD_type>_16x16x16_<AB_type>_w32_gfx12(
    A_frag,   // A 分块元素向量
    B_frag,   // B 分块元素向量
    C_frag    // C（累加器）元素向量
);
```

整数 WMMA 族中，builtin 名对累加器总用 int32；Wave32 调用按此顺序传 6 个参数：否定 A（布尔）、打包 A、否定 B（布尔）、打包 B、C 累加器、是否饱和结果（布尔）。这些布尔量在 Clang/LLVM 中为 `i1`。打包指每条 lane 的 A/B 在一个或两个 int32 寄存器内携带多个 8 位或 4 位矩阵元素，而非 FP WMMA 常见的每 lane 一个向量各装八个 half。

```c
D_frag = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
    neg_a, A_frag,   // i1，然后打包 UINT8 A（每 lane <2 x i32>）
    neg_b, B_frag,   // i1，然后打包 UINT8 B（每 lane <2 x i32>）
    C_frag,          // <8 x i32> 累加器
    clamp            // i1：饱和 / 钳位结果
);

// 16×16×16 INT4：每 lane 一个 i32 打包的 nibble 作为 A 与 B
D_frag = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(
    neg_a, A_i32, neg_b, B_i32, C_frag, clamp);

// 16×16×32 INT4：每 lane <2 x i32> 打包操作数（更长 K）
D_frag = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
    neg_a, A_frag, neg_b, B_frag, C_frag, clamp);
```

**易错点：** 对结果分块而言，lane 下标对 16 取余对应的是 N 列而非 M 行，从 CUDA 移植时容易混淆（[ROCm #6025](https://github.com/ROCm/ROCm/issues/6025)）。

### 示例 1：FP16 输入、FP32 输出（Wave32，RDNA 4）

本示例在 RDNA 4（gfx12）上用 WMMA intrinsic 实现 16×16×16 分块 GEMM `D = A×B + C`，FP16 输入、FP32 累加，并在 Host 用 CPU 参考验证 GPU 结果——问题形状与 `samples/wmma_rdna3_fp16.cpp` 相同，但使用 GFX12 的 Wave32 寄存器映射与 builtin 名。

设备核 `wmma_gemm_rdna4` 启动含 32 个线程的一个块（一波）。每条 lane 先按 gfx12 下标将八个半精度值收集到 A、B 的 `half8_t` 分片中（lane 低位选 A 的 M 行与 B 的 N 列；lane 0–15 覆盖 K=0–7，16–31 覆盖 K=8–15；索引 e 在该 8 元 K 范围内再细分）。再按 lane 装入 C 的八个 float——累加器布局与 D 一致，八个槽沿分块内 M 行走，lane 对 16 取余选 N。随后调用 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`，再按与 C 相同的 scatter 模式写 D。无 `OPSEL`，A/B 也无重复 lane：每条 lane 提供不同操作数数据，与 RDNA 3 不同。

Host 上 `cpu_gemm_16x16_ref` 用 `long double` 重算该分块，与 GPU 的 D 比较，打印采样与最大绝对误差；若最大误差超过 1e-2 则返回非零退出码。

源文件：`samples/wmma_rdna4_fp16.cpp`。

已在 ROCm 7.2 与 RDNA 4 GPU（如 RX 9070 XT）上测试。

```bash
hipcc --offload-arch=gfx1201 samples/wmma_rdna4_fp16.cpp -o wmma_rdna4_fp16
./wmma_rdna4_fp16
```

### 示例 2：FP8 输入、FP32 输出（仅 RDNA 4）

与上例相同的 16×16×16 融合乘加 `D = A×B + C`，C/D 为 FP32，但 A、B 以原始 E4M3 FP8 字节存储。FP8 WMMA 仅 RDNA 4 支持。

设备核 `wmma_gemm_rdna4_fp8` 使用 `pack_fp8_A_col` 与 `pack_fp8_B_row`，按与 FP16 示例相同的 (m,k)、(k,n) 下标从全局内存为每 lane 收集八个 FP8 字节，再打包为 `int32x2`（两个小端 32 位字与八个字节的联合体），供 `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` 使用。C 的加载与 D 的写回遵循与 `wmma_rdna4_fp16.cpp` 相同的每 lane 八个 float 累加器映射。

源文件：`samples/wmma_rdna4_fp8.cpp`。

已在 ROCm 7.2 与 RDNA 4 GPU（如 RX 9070 XT）上测试。

```bash
hipcc --offload-arch=gfx1201 samples/wmma_rdna4_fp8.cpp -o wmma_rdna4_fp8
./wmma_rdna4_fp8
```

### 示例 3：INT8 输入、INT32 输出（Wave32，RDNA 4）

在 16×16×16 分块上实现 `D = A×B + C`，INT8 的 A/B、INT32 的 C/D，与 `samples/wmma_rdna3_iu8.cpp` 类似，但调用 gfx12 六参数 iu8 WMMA 而非 RDNA 3 路径。

与 FP16 gfx12 不同，iu8 操作数不能仅靠每 lane 简单全局 gather：核内复用与 RDNA 3 相同的共享内存 swizzle 构造 `int8x16` A/B 分片，再为 gfx12 拆分——线程 0–15 使用字节 `[0..7]`，16–31 使用 `[8..15]`——并用 `__builtin_bit_cast` 将八个字节转为 `int32x2`。C 的加载与 D 的写回使用 gfx12 累加器映射（与 `wmma_rdna4_fp16.cpp` 相同）。builtin 为 `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`；前两个 `bool` 为符号标志（`true` 表示有符号 int8），并非取反；`clamp` 为 `false`。

源文件：`samples/wmma_rdna4_iu8.cpp`。需用 C++20 以使用 `__builtin_bit_cast`。

已在 ROCm 7.2 与 RDNA 4 GPU（如 RX 9070 XT）上测试。

```bash
hipcc --offload-arch=gfx1201 -std=c++20 samples/wmma_rdna4_iu8.cpp -o wmma_rdna4_iu8
./wmma_rdna4_iu8
```

### 示例 4：INT4 输入、INT32 输出（Wave32，RDNA 4）

`samples/wmma_rdna4_iu4.cpp` 在 GPU 上跑两组检查（K=16 与 K=32），求 `D = A×B + C`：int4 放在 `int8_t` 的低半字节，int32 的 C/D，思路与 `wmma_rdna3_iu4.cpp` 相同。仅当两项均与 CPU 一致时 `main` 返回 0。

K=16 路径将每 lane 八个 nibble 打包进一个 `int32`（`pack_iu4_x8`），并调用 `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12`。K=32 路径将 A 视为 16×32、B 为 32×16（约化维 K=32），每 lane 用 `pack_iu4_x16` 将十六个 nibble 打包为 `int32x2`，并调用 `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12`。C/D 使用与 FP16 示例相同的 gfx12 累加器布局；`neg_a` / `neg_b` 为 true，`clamp` 为 false。

Host 参考：`cpu_gemm_i4_i32` 与 `sx_i4`，与 `wmma_rdna3_iu4.cpp` 相同。

已在 ROCm 7.2 与 RDNA 4 GPU（如 RX 9070 XT）上测试。

```bash
hipcc --offload-arch=gfx1201 -std=c++20 samples/wmma_rdna4_iu4.cpp -o wmma_rdna4_iu4
./wmma_rdna4_iu4
```

### 示例 5：链式 WMMA 用于 MLP 推理（RDNA 4）

在 gfx12 上，D 的累加器布局与 B 的操作数布局不同，因此不能不经重排就把第一次 WMMA 的寄存器输出直接当作第二次 WMMA 的 B 分片。典型做法是一小段共享内存：按行主序写入，再按上文 B 的 gather 方式加载。`samples/mlp_wmma_rdna4.cpp` 在 16×16 分块上实现 `output = W2 * relu(W1 * input)`：两次 `_gfx12` WMMA，在 FP32 上做 `relu`，隐藏分块写入 `__half` 共享内存（与第二层的 FP16 B 操作数一致）。第二次 WMMA 使用新的零累加器 `acc1`，而不是传入第一次 WMMA 的那个变量。

```bash
hipcc --offload-arch=gfx1201 samples/mlp_wmma_rdna4.cpp -o mlp_wmma_rdna4
./mlp_wmma_rdna4
```

---

## 使用高层 rocWMMA 库

若无需在 intrinsic 层手调寄存器，可使用 AMD 的 rocWMMA——C++ 模板库，接口贴近 CUDA 的 `nvcuda::wmma`，可自动处理寄存器布局，并支持 RDNA 与 CDNA。

`samples/rocwmma_example.cpp` 为完整 HIP 程序：单个 16×16×16 分块，`fragment<matrix_a, …, col_major>`、`fragment<matrix_b, …, row_major>`、FP32 累加器，`D = A*B + C` 且 `C` 为非零缓冲（布局与 `samples/wmma_rdna4_fp16.cpp` 一致）。

**rocWMMA 7.x 说明：** 从全局内存加载累加器时，`load_matrix_sync` 的重载需显式指定布局，例如 `load_matrix_sync(c_frag, C, ld, mem_row_major)`——当 `c_frag` 类型没有静态布局时，旧的三参数形式不够用。

```bash
hipcc --offload-arch=native samples/rocwmma_example.cpp -o rocwmma_example
./rocwmma_example
```

---

## 扩展：大于 16×16 的分块 GEMM

实际矩阵远大于 16×16。标准做法是将问题分解为 16×16 分块网格，并对部分积累加：

```
for each output tile (bm, bn):
    c_tile = 0
    for each reduction step bk:
        load A_tile = A[bm*16 : (bm+1)*16,  bk*16 : (bk+1)*16]
        load B_tile = B[bk*16 : (bk+1)*16,  bn*16 : (bn+1)*16]
        c_tile += wmma(A_tile, B_tile)
    write D[bm*16 : (bm+1)*16,  bn*16 : (bn+1)*16] = c_tile
```

在 HIP 中，每个线程块通常负责一个或多个输出分块；块内每个波前在每步恰好执行一次 WMMA。

`samples/tiled_gemm_rdna4.cpp` 实现 `D = A * B`：`M×K` 的 `__half` 矩阵 A（列主序，步长 `lda`）、`K×N` 的 B（行主序，`ldb`）、`M×N` 的 FP32 D（行主序，`ldd`），且 `M, N, K` 均为 16 的倍数。每个线程块负责一个 16×16 输出分块：`grid(N/16, M/16)`，`block(32)`。可选命令行参数 `M N K`；默认 `32×32×32`。

```bash
hipcc --offload-arch=gfx1201 samples/tiled_gemm_rdna4.cpp -o tiled_gemm_rdna4
./tiled_gemm_rdna4
./tiled_gemm_rdna4 64 48 32
```

---

## 对照：RDNA 3 与 RDNA 4

| 特性 | RDNA 3（GFX11） | RDNA 4（GFX12） |
|------|----------------|----------------|
| 架构代号 | `gfx1100`–`gfx1102`（ROCm 中亦见 `gfx1150`、`gfx1151` 等 GFX11 目标） | `gfx1200`、`gfx1201` |
| 分块尺寸 | 典型 FP / INT8 WMMA 为 **16×16×16** | 多数类型相同；gfx12 上 **INT4** 另有更大 **K** 形状（如 **16×16×32**）——见 [RDNA 4 (GFX12) WMMA](#rdna-4-gfx12-wmma) 类型表 / ISA |
| 波前模式 | Wave32 / Wave64 | Wave32（示例）；亦有 `_w64_gfx12` intrinsic |
| FP16/BF16 FLOPS/时钟/CU | 256 | 512（按 AMD 架构公开信息；以 SKU 为准） |
| INT8 FLOPS/时钟/CU | 256 | 1024（按 AMD 架构公开信息；以 SKU 为准） |
| FP8 | 无 | 有（E4M3、E5M2） |
| 结构化稀疏 | 无 | 有（4:2 SWMMAC） |
| Lane 复制（A/B） | 有 | 无 |
| Intrinsic 后缀 | 无 | `_gfx12` |
| `OPSEL` 参数 | 仅 FP16/BF16 累加器输出（如 `f16_16x16x16_f16`）；`f32_16x16x16_*` 为三参数、无 `OPSEL` | 无 |
| 向后兼容 | — | 否 |
| 最低 ROCm 版本 | 约 5.4（量级；以发行说明为准） | 约 6.2+（gfx12 WMMA / FP8 等；以发行说明为准） |
| 示例 intrinsic | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` |

---

## 如何选择实现路径

| 场景 | 建议 |
|------|------|
| RDNA 3 与 4 间最大可移植性 | rocWMMA —— 自动处理布局差异 |
| 极致性能、完全掌控 | 编译器 intrinsic + 按架构分路径 |
| 原型 / 研究 | intrinsic + `#ifdef __gfx12__` / `#ifdef __gfx11__` |
| 从 CUDA 移植 | rocWMMA（API 贴近 `nvcuda::wmma`） |

同时为多个目标编译：

```bash
# 同一二进制内包含 RDNA3 与 RDNA4 目标
hipcc --offload-arch=gfx1100 --offload-arch=gfx1201 my_kernel.cpp -o my_kernel
```

运行时检测架构（便于选择正确 kernel）：

```cpp
hipDeviceProp_t prop;
hipGetDeviceProperties(&prop, 0);
// prop.gcnArchName 例如 "gfx1100" 或 "gfx1201"
bool is_rdna4 = (strncmp(prop.gcnArchName, "gfx12", 5) == 0);
```

---

## 工具与延伸阅读

### AMD Matrix Instruction Calculator

[AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) 是官方工具，可打印各支持架构上 WMMA 指令的详细信息：

```bash
# RDNA3（gfx1100）上 FP16 WMMA 详情
python3 matrix_calculator.py --architecture gfx1100 \
    --instruction v_wmma_f32_16x16x16_f16 --detail-instruction

# RDNA4（gfx1201）上 FP8 WMMA 详情
python3 matrix_calculator.py --architecture gfx1201 \
    --instruction v_wmma_f32_16x16x16_fp8_fp8 --detail-instruction
```

输出含：操作码、VGPR 数量、每 lane 元素映射、峰值吞吐估计等。

### ISA 参考手册

- [RDNA3 ISA Reference Guide（PDF）](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
- [RDNA4 ISA Reference Guide（PDF）](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)

---

## 小结

WMMA 将硬件矩阵加速带入 AMD 消费级 GPU，在无需数据中心 GPU 的情况下显著提升 AI 与通用计算性能。RDNA 3 奠定了 FP16/BF16/INT8 基础；RDNA 4 在更高吞吐、取消镜像 lane 带来的更清晰操作数布局，以及 FP8 等新数据类型所支撑的超低精度推理等方面进一步发力。

编写 WMMA 代码时请牢记：

1. RDNA 3：使用 `__builtin_amdgcn_wmma_*_w32` 或 `*_w64`；为 A/B 的 lane 复制建模。
2. RDNA 4：使用 `__builtin_amdgcn_wmma_*_w32_gfx12`（或 `*_w64_gfx12`），配合 `half8_t` / `float8_t` 与上文 m/k/n lane 公式做加载与存储；按需使用 FP8 / SWMMAC。**iu8** 使用六参数 builtin，每 lane 用 `int32x2` 打包 A/B（`samples/wmma_rdna4_iu8.cpp`）；**iu4** 在 `…_16x16x16_iu4_…` 上每 lane 用标量 `int32` A/B，在 `…_16x16x32_iu4_…` 上用 `int32x2`（`samples/wmma_rdna4_iu4.cpp`）。
3. 两代均可考虑 rocWMMA，以获得可移植、易维护的代码。

---

## 参考来源

- [How to accelerate AI applications on RDNA 3 using WMMA — AMD GPUOpen](https://gpuopen.com/learn/wmma_on_rdna3/)
- [Using the Matrix Cores of AMD RDNA 4 architecture GPUs — AMD GPUOpen](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/)
- [Matrix Core Programming on AMD CDNA3 and CDNA4 — AMD ROCm Blog](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [AMD Matrix Instruction Calculator — GitHub](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [rocWMMA Programming Guide — ROCm Documentation](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/conceptual/programmers-guide.html)
- [RDNA3 ISA Reference Guide — AMD](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
- [RDNA4 ISA Reference Guide — AMD](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)
- [Examining AMD's RDNA 4 Changes in LLVM — Chips & Cheese](https://chipsandcheese.com/p/examining-amds-rdna-4-changes-in-llvm)
- [WMMA Benefits for ML and General Compute — AMD GPUOpen](https://gpuopen.com/news/wmma_benefits_ml_compute/)
- [ROCm issue #6025 — gfx12 WMMA 输出 / lane 映射讨论](https://github.com/ROCm/ROCm/issues/6025)
