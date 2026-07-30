---
type: concept
title: Grouped GEMM（分组矩阵乘）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html"]
tags: [moe, gemm, kernel, 算子层]
---

# Grouped GEMM（分组矩阵乘）

> 所属层次：[[purpose|算子层]]。[[MoE]] 层的核心计算形态。

## 定义

MoE 层中，同一专家的 token 共享一份权重矩阵 `W_e`。N 个专家 = N 组独立的 GEMM，这就是 Grouped GEMM。每个线程块（block）处理某个专家的一个 M 方向分块。

## 为什么需要布局对齐

GEMM kernel 以固定块长 `BLOCK_M`（通常 16/32/64）为粒度工作，因此要求：

- 同一专家的 token **连续存放**（块内不能混专家，否则权重指针要逐行切换）
- 每专家的 token 数**补齐到 BLOCK_M 整数倍**（grid 按块数启动，不能处理半块）

这项工作由 [[moe_align_block_size]] 完成。对齐后：

- 每个块通过 `expert_ids[pid_m]` **只查一次专家号**
- 整块复用同一权重指针 `b_ptr + e * stride_be`
- PAD 槽位（值为 numel 的越界下标）被 mask 屏蔽，结果丢弃
- `expert_ids` 为 -1 的块（非本卡专家）整块跳过 → 见 [[专家并行 EP]]

## 消费示意（[[Triton]]）

见 [[moe_align_block_size-交互式详解#消费方式（Triton 伪代码）|来源页的消费伪代码]]。

## 参见

- [[MoE]]、[[moe_align_block_size]]、[[vLLM]]
