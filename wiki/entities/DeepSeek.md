---
type: entity
title: DeepSeek
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html"]
tags: [模型, moe, 模型层]
---

# DeepSeek

> 所属层次：[[purpose|模型层]]。以 [[MoE]] 架构著称的大模型系列。

## 与本知识库相关的要点

- DeepSeek 系列是 **DP（[[数据并行 DP|数据并行]]）+ EP（[[专家并行 EP|专家并行]]）混合部署**的典型代表
- 正是这种部署形态催生了 [[vLLM]] 中 `batched_moe_align_block_size` 变体的设计（见 [[moe_align_block_size]]）

## 待补充方向

- 模型结构细节（MLA、细粒度专家等，待相关来源 ingest）

## 参见

- [[MoE]]、[[数据并行 DP]]、[[专家并行 EP]]
