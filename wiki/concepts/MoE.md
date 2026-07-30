---
type: concept
title: MoE（混合专家模型）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html", "raw/sources/moe_align_block_size 路径.md"]
tags: [moe, 模型层]
---

# MoE（Mixture of Experts，混合专家）

> 所属层次：[[purpose|模型层]]。本页随更多来源持续扩充。

## 核心机制

- 每个 MoE 层包含 N 个"专家"（通常是独立的 FFN 权重矩阵）
- **路由器（router）** 为每个 token 选出 top-k 个专家（输出 `topk_ids` 和路由权重）
- 只有被选中的专家参与计算 → 参数量与计算量解耦，稀疏激活

## 对系统/算子层的决定性影响

MoE 的结构直接塑造了底层算子的形态：

1. **计算按专家分组**：同一专家的 token 共享同一份权重矩阵 → 自然对应 [[Grouped GEMM]]
2. **路由输出是散乱的**：`topk_ids` 中同一专家的 token 分散在各处，必须先做布局整理 → 这正是 [[moe_align_block_size]] 存在的意义
3. **专家成为并行的天然切分单位** → [[专家并行 EP]]；token 成为另一个切分单位 → [[数据并行 DP]]
4. **每步每专家的 token 数动态变化** → 与 [[CUDA Graph]] 的定形要求冲突，需要定长缓冲 + 补齐方案

## 扩展结构（来自来源 2）

### Shared Experts（共享专家）

在 routed sparse 专家外再配一个所有 token 都经过的 dense 专家。典型：Mixtral-8x7B、Qwen1.5-MoE。Qwen3-30B-A3B 无此结构。
→ 详见 [[Shared Experts]]

### Latent MoE（潜空间 MoE）

路由前先把 hidden_states 投影到更低维的 latent space，算完再升维。减少 gate + expert 的计算量与内存。
→ 详见 [[Latent MoE]]

### 序列并行（Sequence Parallel MoE）

TP 下 attention 后 hidden_states 是各卡完全相同的副本。序列并行在 router 前把 token 均切成 tp_size 份，每卡只路由 + GEMM 自己的 chunk，最后 all-gather 拼回。节省路由冗余计算和 all-reduce 通信量。
→ 与 [[moe_align_block_size]] 联动：切分后的 chunk token 数可能小到触发 naive 捷径

## 典型代表

- [[DeepSeek]] 系列：MoE 架构 + DP/EP 混合部署的典型代表

## 参见

- [[moe_align_block_size]]、[[Grouped GEMM]]、[[专家并行 EP]]、[[数据并行 DP]]、[[Shared Experts]]、[[Latent MoE]]
