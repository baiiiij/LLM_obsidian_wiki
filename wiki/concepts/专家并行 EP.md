---
type: concept
title: 专家并行（EP, Expert Parallelism）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html"]
tags: [moe, 并行, ep, 框架层]
---

# 专家并行（EP, Expert Parallelism）

> 所属层次：[[purpose|框架层]]。[[MoE]] 部署的并行策略之一。

## 定义

把 N 个专家**切分到多张 GPU** 上，每卡只持有一部分专家的权重。token 通过通信（All-to-All）被发送到持有其目标专家的卡上计算。

## 对算子的影响（以 [[moe_align_block_size]] 为例）

EP 下该 kernel 的用法变化：

- `num_experts` 传**全局**专家数
- 传入 `expert_map`：全局专家号 → 本卡局部号，非本卡专家记为 **-1**
- 两种工作模式：

| 模式 | 行为 | 特点 |
|---|---|---|
| 默认（`ignore_invalid_experts=False`） | 全部全局专家参与计数/排序/补齐，返回前映射，非本卡块标 -1 | 布局全局一致；MoE GEMM 遇 -1 整块跳过 |
| 过滤（`ignore_invalid_experts=True`） | 计数阶段直接忽略非本卡专家 | 缓冲更小，`expert_ids` 无 -1 |

## 与 DP 的组合

EP 常与 [[数据并行 DP]] 组合部署（典型：[[DeepSeek]] 系列），此时需要 `batched_moe_align_block_size` 变体配合定长通信缓冲，并受 [[CUDA Graph]] 定形约束。

## 参见

- [[MoE]]、[[数据并行 DP]]、[[moe_align_block_size]]、[[vLLM]]
