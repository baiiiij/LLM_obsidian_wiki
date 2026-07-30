---
type: entity
title: vLLM
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html"]
tags: [推理框架, 框架层]
---

# vLLM

> 所属层次：[[purpose|框架层]]。主流 LLM 推理框架。

## 与本知识库相关的要点

### fused MoE 算子路径（来自 [[moe_align_block_size-交互式详解|来源 1]]）

- MoE 推理使用 `fused_moe_kernel`（[[Triton]] 编写的 [[Grouped GEMM]]）
- 前置预处理 kernel：[[moe_align_block_size]]，把散乱的路由输出整理成定长对齐布局
- 支持 [[专家并行 EP|EP]]（`expert_map`、两种过滤模式）和 [[数据并行 DP|DP]]（`batched_moe_align_block_size` 变体）

### 应对动态形状的设计哲学

- 为兼容 [[CUDA Graph]] 的定形要求，采用**定长缓冲 + 有效计数数组**（`expert_num_tokens`）代替动态分配
- 预处理 kernel 只处理 int32 下标，开销相对 GEMM 可忽略——"用廉价的整理换 GEMM 的高效"

## 待补充方向

- PagedAttention、Continuous Batching（[[purpose]] 中的关注点，待相关来源 ingest）
- 调度器与内存管理实现

## 参见

- [[MoE]]、[[Grouped GEMM]]、[[moe_align_block_size]]
