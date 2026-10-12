---
type: concept
title: Da Vinci 架构
created: 2026-10-12
updated: 2026-10-12
sources: ["raw/sources/昇腾硬件架构调研-2026-10.md", "raw/sources/AscendC算子开发调研-2026-10.md"]
tags: [ascend, 硬件层, da-vinci, ai-core]
---

# Da Vinci 架构

> 所属层次：[[purpose|硬件层]]。昇腾 [[Ascend NPU]] 的计算核心架构（DSA），决定了 [[Ascend C]] 算子的写法。横向对照见 [[Ascend 与 NVIDIA GPU 对照]]。

## AI Core：三单元三流水线

每个 AI Core 含三种计算单元，由控制单元统一调度，形成三条独立执行流水线：

| 单元 | 职责 | 要点 |
|---|---|---|
| **Cube Unit** | 矩阵乘加：一条指令完成 FP16 16×16×16（4096 MAC）或 INT8 16×32×16（8192 MAC） | 占整芯片算力 ~99%，FLOPS 主要来源 |
| **Vector Unit** | 向量/逐元素运算（activation、norm、reduce、gather），支持倒数/开方等 | 2048-bit 宽 SIMD（初代）；操作数必须在 UB，32B 对齐 |
| **Scalar Unit** | 标量运算 + 流程控制（循环/分支/地址计算） | 相当于核内小 CPU |

**910B 起的 AIC/AIV 拆分**：一个物理 AI Core 拆为 1 个 **AIC（AI Cube core）** + 2 个 **AIV（AI Vector core）**（比例因型号而异），各自含独立 Scalar Unit——编程模型上向量核数量翻倍，[[Ascend C]] 用 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 宏隔离两类核的代码。

## 显式管理的内存层级（与 GPU cache 哲学的根本分歧）

昇腾不靠硬件 cache 一致性，而是**程序员显式管理的 buffer 层级**（容量为初代 910 典型值）：

```
GM (HBM, 片外)                    ←→ MTE2/MTE3 (DMA)
  └─ L2 (芯片级共享, 910B 实测 192MB)
       └─ L1 Buffer (1MB)         ← AI Core 内最大中转区，Img2Col 源数据必须在 L1
            ├─ L0A / L0B (各 64KB) ← Cube 指令输入（A/B 矩阵）
            ├─ L0C (256KB)         ← Cube 输出/累加器（FP32 累加）
            └─ UB Unified Buffer (256KB) ← Vector/Scalar 的输入输出
```

- 新型号容量用 `get_soc_spec` 查询；triton-ascend 实践中 A2 可用 UB 为 192KB、950 为 248KB（[[triton-ascend]] 的 UB Overflow 坑即源于此）
- L0A/L0B/L0C 的角色 ≈ Tensor Core operand register 的**显式化**——GPU 上由硬件/MMA 指令隐式管理，昇腾上程序员可见可调度

## MTE：与计算并行的专用搬运引擎

- **MTE2**：GM → L1/L0A/L0B/UB（加载）；**MTE1**：L1 → L0A/L0B/UB（片内搬运 + 格式转换 padding/transpose/Img2Col）；**MTE3**：UB → GM（写回）
- 对照 NVIDIA `cp.async` / TMA：异步搬运是 double buffering 等流水优化的硬件基础

## 多队列乱序执行

指令顺序译码后分派到 **S / V / M / MTE1 / MTE2 / MTE3 六个硬件队列**：队列内顺序、队列间满足依赖即可并行，跨队列同步用 `set_flag/wait_flag` + Barrier。这正是 [[Ascend C]] 中 `SetFlag<HardEvent::X_Y>` / `WaitFlag` 和 `TQue` 隐式同步语义的硬件来源。

## 编程含义（串起算子层）

- 算子写法 = 显式编排"DMA 搬入 → Vector/Cube 计算 → DMA 写回"的流水线，见 [[Ascend C]] 的 CopyIn/Compute/CopyOut 范式
- 执行模型是 SPMD + 核内 SIMD，与 GPU SIMT 的本质差异见 [[SIMT 与 SIMD 与 SPMD]]
- 官方已宣布 950 起转向 SIMD+SIMT 混合（访存粒度 512B→128B），见 [[Ascend NPU]] roadmap

## 参见

- [[Ascend NPU]]、[[Ascend C]]、[[Ascend 与 NVIDIA GPU 对照]]、[[SIMT 与 SIMD 与 SPMD]]
- 来源摘要：[[昇腾硬件架构调研-2026-10]]
