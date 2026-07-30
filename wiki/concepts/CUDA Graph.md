---
type: concept
title: CUDA Graph
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html"]
tags: [cuda, 性能, 框架层, 硬件层]
---

# CUDA Graph

> 所属层次：[[purpose|框架层 ↔ 硬件层]]之间。GPU kernel 启动开销的消除技术。

## 关键约束（本来源涉及的）

CUDA Graph 捕获时要求**所有张量形状固定**。

这与 [[MoE]] 推理天然冲突：每步、每个专家收到的 token 数完全随机，张量形状步步变化。

## 对 MoE 推理的影响

[[vLLM]] 的应对策略（见 [[数据并行 DP]]）：

- Dispatch 写入**形状恒定**的缓冲 `(E × max_tokens_per_batch, K)`
- 用 `expert_num_tokens` 数组记录有效数，而非改变张量形状
- 由 `batched_moe_align_block_size` 在定长缓冲内做补齐打包

> 启示：**"形状恒定化"** 是让动态负载兼容 CUDA Graph 的通用思路——把变化量从 tensor shape 转移到数据内容（有效计数数组 + padding）。

## 参见

- [[数据并行 DP]]、[[专家并行 EP]]、[[vLLM]]
