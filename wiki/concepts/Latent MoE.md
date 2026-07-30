---
type: concept
title: Latent MoE（潜空间 MoE）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size 路径.md"]
tags: [moe, 模型层, 框架层]
---

# Latent MoE（潜空间 MoE）

> 所属层次：[[purpose|模型层 ↔ 框架层]]。部分 MoE 架构使用的降维-升维技巧。

## 核心思想

在路由（gate）和专家 GEMM 之前，先把 hidden_states **投影到一个更低维度的潜空间**（latent space），计算完后再**升维**回原始尺寸。

```
hidden_states [S, hidden_dim]
  ↓ routed_input_transform（降维投影）
latent [S, latent_dim]          # latent_dim < hidden_dim
  ↓ gate + expert GEMM
expert_output [S, latent_dim]
  ↓ routed_output_transform（升维投影）
output [S, hidden_dim]
```

## 动机

- **减少 gate + expert 的计算量**：gate 的 GEMM 和 expert 的输入维度都从 `hidden_dim` 降到 `latent_dim`
- **减少激活内存**：workspace 和中间缓冲更小
- **类似 AutoEncoder bottleneck**：用压缩表示做核心计算，再解压缩

## vLLM 中的体现

- `apply_routed_input_transform`（`moe_runner.py` L284）：若配置了 `routed_input_transform`，返回降维后的张量给 routed 专家，同时保留原始张量给 [[Shared Experts]]
- `_maybe_pad_hidden_states`（L428）：降维后 latent_dim ≠ hidden_dim，kernel 要求统一尺寸 → python 层补零，算完再裁掉
- 标准模型（如 Qwen3-30B-A3B）没有降维配置，此分支为**恒等映射**

## 参见

- [[MoE]]、[[vLLM]]、[[Shared Experts]]
