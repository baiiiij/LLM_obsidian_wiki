---
type: concept
title: 序列并行 MoE（Sequence Parallel MoE）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size 路径.md"]
tags: [moe, 并行, tp, 框架层]
---

# 序列并行 MoE（Sequence Parallel MoE）

> 所属层次：[[purpose|框架层]]。消除 TP 下 MoE 冗余计算的并行优化。vLLM 中由 `ParallelConfig.use_sequence_parallel_moe` 开启（配 TP>1）。

## 要解决的问题

TP 场景的两个冗余：

1. attention 的 `o_proj`（RowParallel）做完 all-reduce 后，**各卡 hidden_states 完全相同**
2. gate 是 ReplicatedLinear（每卡完整复制）→ **每卡对全部 token 重复算相同的路由**

## 机制

`sequence_parallel_chunk`（`models/utils.py` L815）：沿 token 维把 hidden_states 均切成 tp_size 份（不能整除先 padding），每卡只取自己那 1/tp：

```python
chunk = y.shape[0] // tp_size
return torch.narrow(y, 0, tp_rank * chunk, chunk)
```

算完专家后 `tensor_model_parallel_all_gather(dim=0)` 拼回完整序列，再 `[:num_tokens]` 裁掉 padding。

## 收益

| | 不用 SP | 用 SP |
|---|---|---|
| 每卡路由计算 | 全部 N token（冗余） | N/tp |
| 每卡 expert GEMM | M = N | M = N/tp |
| 通信 | all-reduce 完整张量 ~2S | all-reduce(chunk) + all-gather ≈ 2S/tp + S |

## 实现细节

包成 custom op（`torch.ops.vllm.sequence_parallel_chunk_impl`）：小 batch 输出长度可能为 0，torch.compile/[[CUDA Graph]] 下会出问题，故注册 custom op 并提供 fake impl。

## 与下游算子的联动

切分发生在 router 之前 → [[moe_align_block_size]] 看到的 `num_tokens` 是**每卡切分后**的值：

- 例：TP=8、32 token → 每卡 4 个
- 对 Qwen3-30B-A3B（E=128, top-8）：`4 × 8 × 4 = 128 ≤ 128` → **恰好触发 naive 捷径**，跳过对齐 kernel
- 启示：序列并行不仅省计算/通信，还可能**改变 kernel 路径选择**

## 参见

- [[MoE]]、[[moe_align_block_size]]、[[All-to-All]]、[[vLLM]]
