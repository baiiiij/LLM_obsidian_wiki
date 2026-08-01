---
type: hub
title: vLLM 推理框架（大类粘合页）
created: 2026-07-31
updated: 2026-07-31
sources: []
tags: [vllm, hub, 导航, moe]
---

# vLLM 推理框架

> 本大类的主线：**一个 MoE 模型的 token，在 vLLM 里从进来到出去，经过哪些模块、哪些算子、哪些卡？** 以 [[vLLM]]（v0.20.2）为框架底座，以 [[moe_align_block_size]] → [[Grouped GEMM]] 为已深挖的算子路径。

## 推荐阅读顺序（第一次读）

| 顺序 | 页面 | 为什么在这个位置 |
|---|---|---|
| 1 | [[LLM 前向计算]] | **背景**：token/hidden states、FFN = 两次 GEMM、计算与访存、权重复用——后续所有算子优化的判据都来自这里 |
| 2 | [[MoE]] | **模型结构**：路由 + top-k 稀疏激活，决定了底层算子为什么长这样 |
| 3 | [[vLLM]] | **框架总览**：MoERunner + 模块化 kernel 架构、TP/SP/EP/DP/PCP 通信节奏——知道算子在框架里的位置 |
| 4 | [[moe_align_block_size-代码路径]] | **调用链**：从 Qwen3MoeForCausalLM 到 kernel 的完整代码分支路径（来源摘要页，可当地图用） |
| 5 | [[moe_align_block_size]] | **算子本体（前置）**：MoE 布局整理，朴素缺点 → 三段式前置计算 → naive/native 捷径 |
| 6 | [[Grouped GEMM]] | **算子本体（核心计算）**：同专家 token 共享权重按块复用——moe_align 整理布局正是为它服务 |
| 7 | [[moe_align_block_size-交互式详解]] | **来源讲解**：基础版 / EP / DP 三场景的交互式讲解（来源摘要页，复习用） |

## 并行策略网（按主题查）

vLLM 多卡部署涉及的概念页，按需查阅：

| 我想…… | 去这页 |
|---|---|
| 专家怎么切到多卡、expert_map 与 -1 跳过 | [[专家并行 EP]] |
| token 怎么切到多卡、Dispatch/Combine 流水线 | [[数据并行 DP]] |
| TP 下 gate 冗余怎么消 | [[序列并行 SP]] |
| 超长序列怎么切段 | [[Context Parallel]] |
| 专家负载不均怎么办（冗余专家/双层 ID） | [[EPLB]] |
| 集合通信原语对比 | [[All-to-All]] |
| 动态形状与 CUDA Graph 的矛盾 | [[CUDA Graph]] |

## 延伸（本大类与其他大类的接口）

- [[Triton]] — vLLM MoE kernel 的实现语言（算子语言）
- [[PyTorch-ATen-Dispatcher]] — vLLM 自定义算子（`TORCH_LIBRARY` 注册）依赖的机制（PyTorch 大类）
- [[FlagGems]] — 算子替换的另一条路线（覆盖既有 aten 算子 vs vLLM 的自研 kernel 注册）
- [[DeepSeek]] — MoE 架构模型，DP+EP 混合部署的典型负载

## 成员页清单

- 实体：[[vLLM]]
- 概念：[[moe_align_block_size]]、[[Grouped GEMM]]、[[专家并行 EP]]、[[数据并行 DP]]、[[序列并行 SP]]、[[Context Parallel]]、[[EPLB]]、[[All-to-All]]、[[CUDA Graph]]
- 来源摘要：[[moe_align_block_size-交互式详解]]、[[moe_align_block_size-代码路径]]
