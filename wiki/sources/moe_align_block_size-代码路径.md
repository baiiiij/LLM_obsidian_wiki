---
type: source
title: vLLM moe_align_block_size 代码分支路径（v0.20.2 + Qwen3-30B-A3B）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size 路径.md"]
tags: [vllm, moe, kernel, ep, dp, triton, 代码分析]
---

# vLLM moe_align_block_size 代码分支路径

> 来源摘要页。原始文件：`raw/sources/moe_align_block_size 路径.md`（含主文档 + 5 轮问答记录）。

## 一句话概括

以 Qwen3-30B-A3B 为例，完整追踪从 `Qwen3MoeForCausalLM.forward` 到 `moe_align_block_size` CUDA kernel 的**外部（模型层）+ 内部（框架层）代码分支**，并分析每个分叉点的触发条件与对算子的影响。

## 七层调用链（GPU / 单卡无量化 / ≥5 token）

```
Qwen3MoeForCausalLM.forward (qwen3_moe.py:764)
└─ Qwen3MoeModel.forward
   └─ Qwen3MoeDecoderLayer.forward (L416) ×48
      └─ Qwen3MoeSparseMoeBlock.forward (L226)
         ├─ [分支①] MoE vs Dense（num_experts>0 & decoder_sparse_step）→ MoE
         ├─ [分支②] router 内外（is_internal_router=True）→ 内部
         └─ FusedMoE.forward (layer.py:1545)
            └─ MoERunner.forward (moe_runner.py:567)
               └─ custom op _moe_forward → _forward_impl (L717)
                  ├─ [分支③] monolithic?（仅 CPU）→ False
                  ├─ router.select_experts → topk_weights/topk_ids
                  └─ UnquantizedFusedMoEMethod.apply → moe_kernel.apply
                     └─ TritonExperts.apply (fused_moe.py:1985)
                        └─ _prepare_expert_assignment (L1554)
                           ├─ [分支⑤] naive 捷径?（tokens≤4 且无 EP）→ 跳过对齐
                           └─ moe_align_block_size (moe_align_block_size.py:11)
                              └─ ops.moe_align_block_size (CUDA)
```

## 关键发现

- **0.20.2 大重构**：FusedMoE 本体变壳，逻辑全部委托给 `MoERunner`；无量化后端走**模块化 kernel**（`FusedMoEKernel` = prepare_finalize + TritonExperts），由 `oracle/unquantized.py` 自动选后端
- **naive_block_assignment 捷径**（v0.20.2 新增）：当 `expert_map is None` 且 `num_tokens × top_k × 4 ≤ global_num_experts` 时，跳过 `moe_align_block_size`，走 kernel 内低延迟路径。对 Qwen3-30B-A3B（E=128, top-8）→ **仅 ≤4 token 时触发**
- **序列并行联动**：`sequence_parallel_chunk` 在 router 前切分 token → `moe_align_block_size` 看到的 `num_tokens` 是每卡 chunk。TP=8 时 32 token → 每卡 4 个 → 恰好触发 naive 捷径
- **shared_experts / latent MoE**：模型架构级概念（Qwen3-30B-A3B 无 shared expert、无 latent 投影），vLLM 通过 `FusedMoE` / `MoERunner` 统一支持
- **通信节奏**：`_maybe_dispatch`（前置）→ kernel → `_maybe_combine`（后置）；单卡场景均为恒等映射

## 关联页面

- 算子：[[moe_align_block_size]]、[[Grouped GEMM]]
- 概念：[[MoE]]、[[专家并行 EP]]、[[数据并行 DP]]、[[CUDA Graph]]、[[Shared Experts]]、[[Latent MoE]]
- 实体：[[vLLM]]、[[DeepSeek]]、[[Triton]]
