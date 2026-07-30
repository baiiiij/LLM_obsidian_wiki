---
type: index
title: Wiki 内容目录
updated: 2026-07-30
---

# Index — 内容目录

> 每次 ingest 后更新本文件。查询时先读这里定位页面。
> 格式：页面链接 + 一句话摘要。

## Entities（实体）

- [[vLLM]] — 主流 LLM 推理框架；v0.20.2 MoERunner + 模块化 kernel 架构、TP/SP/EP/DP 通信节奏
- [[DeepSeek]] — MoE 架构大模型系列，DP+EP 混合部署的典型代表
- [[Triton]] — Python 风格 GPU 算子 DSL，vLLM MoE kernel 的实现语言

## Concepts（概念）

- [[MoE]] — 混合专家模型：路由 + top-k 稀疏激活，决定底层算子形态；含 Shared/Latent/序列并行扩展
- [[moe_align_block_size]] — vLLM MoE 布局整理算子：三段式实现、naive 捷径分支、与 SP 联动
- [[Grouped GEMM]] — MoE 的核心计算形态：同专家 token 共享权重，按块复用
- [[专家并行 EP]] — 专家切分到多卡；expert_map 与 -1 跳过机制；naive dispatch 通信
- [[数据并行 DP]] — token 切分到多卡；All-to-All Dispatch/Combine 流水线；PCP 关联
- [[CUDA Graph]] — 定形约束与"形状恒定化"应对思路
- [[Shared Experts]] — 模型架构级 dense 专家，分担通用知识与梯度稳定性
- [[Latent MoE]] — 潜空间降维-升维投影，减少 gate + expert 计算量

## Sources（来源摘要）

- [[moe_align_block_size-交互式详解]] — vLLM `moe_align_block_size` 交互式讲解（基础版 / EP / DP 三场景）
- [[moe_align_block_size-代码路径]] — v0.20.2 从 Qwen3MoeForCausalLM 到 kernel 的完整代码分支路径 + 5 轮问答

## Queries（问答沉淀）

_（暂无）_

## Synthesis（综合分析）

_（暂无）_

## Comparisons（对比）

_（暂无）_
