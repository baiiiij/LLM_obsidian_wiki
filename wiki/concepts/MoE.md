---
type: concept
title: MoE（混合专家模型）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html"]
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

## 典型代表

- [[DeepSeek]] 系列：MoE 架构 + DP/EP 混合部署的典型代表

## 参见

- [[moe_align_block_size]]、[[Grouped GEMM]]、[[专家并行 EP]]、[[数据并行 DP]]
