---
type: concept
title: moe_align_block_size（MoE 布局整理算子）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html"]
tags: [vllm, moe, kernel, 算子层]
---

# moe_align_block_size

> 所属层次：[[purpose|算子层]]。[[vLLM]] fused MoE 路径上的预处理 kernel。

## 作用

把路由器输出的散乱 `topk_ids` 整理成 **「按专家连续存放、每专家槽位数补齐到 block_size 整数倍」** 的布局，使 [[Grouped GEMM]] 的每个线程块：

- 只查一次专家号（`expert_ids[pid_m]`）
- 整块复用同一个权重矩阵 `W_e`
- PAD 槽位通过 mask 丢弃

## 三个输出

| 输出 | 含义 |
|---|---|
| `sorted_token_ids` | 对齐后槽位下标；槽位 i ↔ 激活第 i//top_k 行；补位填 numel（越界值，被 mask） |
| `expert_ids` | 第 i 个块用哪个专家；-1 表示整块跳过 |
| `num_tokens_post_padded` | 补齐后总槽位数，决定 GEMM grid 大小 |

## 实现要点

- **三段式**：原子加计数 → 带 pad 前缀和 → scatter 写回
- 输出缓冲按上界 `numel + num_experts×(block_size−1)` 预分配（每专家最多补 block_size−1 个槽）
- **开销可忽略**：仅处理 num_tokens×top_k 个 int32 下标
- 已知问题：官方 main 分支 docstring 中 batched 示例的 `expert_ids` 有笔误（`[0,1,3,3,4,5,5]`），正确结果为 `[0,1,3,3,4,4]`

## 变体

- **EP 模式**：配合 `expert_map`，默认模式把非本卡块标 -1；过滤模式（`ignore_invalid_experts=True`）计数阶段直接忽略 → 见 [[专家并行 EP]]
- **batched 变体**（`batched_moe_align_block_size`）：token 已被通信层按专家分段，无需排序，只做补齐打包 → 见 [[数据并行 DP]]

## 与上层/下层的关系

- 上层：解决 [[MoE]] 路由输出散乱的问题
- 下层：为 [[Triton]] 编写的 `fused_moe_kernel`（[[Grouped GEMM]]）提供可直接消费的定长布局

## 参见

- 来源摘要：[[moe_align_block_size-交互式详解]]
