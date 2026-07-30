---
type: source
title: vLLM moe_align_block_size 代码分支路径（v0.20.2 + Qwen3-30B-A3B）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size 路径.md"]
tags: [vllm, moe, kernel, ep, dp, triton, 代码分析]
---

# vLLM moe_align_block_size 代码分支路径

> 来源摘要页。原始文件：`raw/sources/moe_align_block_size 路径.md`（经多轮问答打磨后的定稿，五章 + 12 项分支决策表）。

## 一句话概括

以 Qwen3-30B-A3B 为例，完整追踪从 `Qwen3MoeForCausalLM.forward` 到 `moe_align_block_size` CUDA kernel 的**外部（模型层）→ runner → method → modular kernel → TritonExperts → align kernel** 全链路分支，并覆盖并行策略（TP/SP/EP/DP/PCP）、负载均衡（EPLB）、通信原语等周边知识。

## 文档结构（五章）

1. **外部路径（模型定义层）**：MoE/Dense 分支；TP 背景（attention all-reduce、ReplicatedLinear）；序列并行分支；router 内外分支
2. **内部路径（框架层）**：FusedMoE→MoERunner 委托；Latent MoE 与 padding；shared experts；monolithic vs modular；`_maybe_dispatch`/`_maybe_combine` 通信前置；后端 oracle；EPLB 映射
3. **模块化 kernel 内部**：ModularImpl 调用链 6 分支（inplace/async/DBO/空批次/finalize/shared 重叠）；TritonExperts.apply；naive 捷径；EP 语义；batched 变体；align kernel 本体
4. **完整调用链一图流**
5. **分支决策总表（12 项）**

## 核心知识点

- **v0.20.2 大重构**：`FusedMoE` 变壳 → `MoERunner`；无量化走模块化 kernel（`FusedMoEKernel = prepare_finalize + experts`），后端由 oracle 选择（CUDA 默认 TritonExperts）
- **monolithic vs modular 的本质**：输入是 `router_logits`（kernel 内路由）还是 `topk_ids/weights`（外部先路由）；method 层 monolithic 仅 CPU，MK 层还有 FlashInfer 类
- **`supports_internal_mk`**：通信从"runner 外挂"迁移到"kernel 内置"的过渡开关
- **naive_block_assignment 捷径**：无 EP 且 `tokens×top_k×4 ≤ E` 时跳过对齐；Qwen3-30B-A3B（E=128, top-8）仅 ≤4 token 触发
- **序列并行联动**：TP=8 时 32 token → 每卡 4 个 → 恰好触发捷径
- **EPLB**：逻辑/物理专家双层 ID；路由后映射 + 负载记录的融合 Triton kernel；后台 async rebalance
- **通信节奏**：`_maybe_dispatch`（naive DP/EP all2all；PCP all-gather）→ kernel → `_maybe_combine`
- **ReplicatedLinear 的必要性**：gate 必须每卡完全复制，否则各卡路由分叉

## 关联页面

- 算子：[[moe_align_block_size]]、[[Grouped GEMM]]
- 概念：[[MoE]]、[[专家并行 EP]]、[[数据并行 DP]]、[[CUDA Graph]]、[[Shared Experts]]、[[Latent MoE]]、[[序列并行 SP]]、[[EPLB]]、[[All-to-All]]、[[Context Parallel]]
- 实体：[[vLLM]]、[[DeepSeek]]、[[Triton]]
