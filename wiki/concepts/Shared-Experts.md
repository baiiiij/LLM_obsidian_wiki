---
type: concept
title: Shared Experts（共享专家）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size 路径.md"]
tags: [moe, 模型层]
---

# Shared Experts（共享专家）

> 所属层次：[[purpose|模型层]]。某些 [[MoE]] 架构的标配组件。

## 定义

在 routed sparse 专家（每个 token 只激活 top-k 个）之外，再配一个**所有 token 都必须经过的密集专家**（dense FFN）。相当于在 MoE 层里塞了一个标准 MLP。

## 典型模型

- **Mixtral-8x7B**：8 个 routed 专家 + 1 个 shared expert
- **Qwen1.5-MoE**：有 shared expert
- **Qwen3-30B-A3B**：`shared_expert_intermediate_size = 0` → **无 shared expert**

## 设计动机

1. **通用知识基座**：把高频、通用的表示放到 always-on 的 dense 路径，避免 routed 专家被迫学太泛
2. **梯度稳定性**：dense 路径为梯度提供稳定通道，防止稀疏路由导致梯度消失/爆炸
3. **负载均衡**：shared expert 分担一部分计算，减轻 routed 专家的负载压力

## vLLM 中的实现

- `Qwen3MoeSparseMoeBlock.__init__` 检查 `shared_expert_intermediate_size`（`qwen3_moe.py` L187）
- 若 `> 0`，创建 `self.shared_expert = Qwen3MoeMLP(...)`，并通过 `FusedMoE(shared_experts=...)` 传入
- `MoERunner` 的 `_maybe_apply_shared_experts` 负责调度：可以**串行**在 routed 专家前后执行，也可以在**独立 CUDA stream** 上与其并行重叠（`MULTI_STREAM_OVERLAPPED`）

## 参见

- [[MoE]]、[[vLLM]]
