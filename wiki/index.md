---
type: index
title: Wiki 内容目录
updated: 2026-07-30
---

# Index — 内容目录

> 每次 ingest 后更新本文件。查询时先读这里定位页面。
> 格式：页面链接 + 一句话摘要。

## Entities（实体）

- [[vLLM]] — 主流 LLM 推理框架；fused MoE 路径与定长缓冲设计哲学
- [[DeepSeek]] — MoE 架构大模型系列，DP+EP 混合部署的典型代表
- [[Triton]] — Python 风格 GPU 算子 DSL，vLLM MoE kernel 的实现语言

## Concepts（概念）

- [[MoE]] — 混合专家模型：路由 + top-k 稀疏激活，决定底层算子形态
- [[moe_align_block_size]] — vLLM MoE 布局整理算子：散乱路由输出 → 定长对齐布局
- [[Grouped GEMM]] — MoE 的核心计算形态：同专家 token 共享权重，按块复用
- [[专家并行 EP]] — 专家切分到多卡；expert_map 与 -1 跳过机制
- [[数据并行 DP]] — token 切分到多卡；All-to-All Dispatch/Combine 流水线
- [[CUDA Graph]] — 定形约束与"形状恒定化"应对思路

## Sources（来源摘要）

- [[moe_align_block_size-交互式详解]] — vLLM `moe_align_block_size` 交互式讲解（基础版 / EP / DP 三场景）

## Queries（问答沉淀）

_（暂无）_

## Synthesis（综合分析）

_（暂无）_

## Comparisons（对比）

_（暂无）_
