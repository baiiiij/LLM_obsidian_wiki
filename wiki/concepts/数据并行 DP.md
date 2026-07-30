---
type: concept
title: 数据并行（DP, Data Parallelism）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html", "raw/sources/moe_align_block_size 路径.md"]
tags: [moe, 并行, dp, 框架层]
---

# 数据并行（DP, Data Parallelism）

> 所属层次：[[purpose|框架层]]。与 [[专家并行 EP]] 组合是 [[MoE]] 大模型（如 [[DeepSeek]]）的典型部署方式。

## MoE 层在 DP+EP 下的完整流水线

1. 每张 DP 卡各拿一批 token，各自跑路由器，确定每个 token 去哪些专家
2. **All-to-All Dispatch**：把 token 发送到「持有它目标专家的那张卡」
3. 每张卡只对自己持有的专家做 GEMM
4. **All-to-All Combine**：结果发回原卡，按路由权重加权求和

## 核心矛盾：动态形状 vs CUDA Graph

第 ② → ③ 步之间，每步、每个专家收到的 token 数**完全随机**：

- 按实际数量现分内存 → 张量形状步步变化
- [[CUDA Graph]] 要求所有张量形状固定 → 无法捕获
- 通信和计算 kernel 也没法做定长优化

## vLLM 的对策：定长缓冲 + batched align

- Dispatch 直接把 token 写入形状恒定的缓冲 `(E × max_tokens_per_batch, K)`：每专家固定一排，来几个写几个，空着就空着
- 配一个 `expert_num_tokens` 数组记录每排实际有效数
- `batched_moe_align_block_size`：token 已被通信层按专家分好段，**无需排序**，只负责"打包"（向上补齐到 block_size 整数倍），输出语义与基础版 [[moe_align_block_size]] 相同

类比：每个专家有一排固定车位，token 是车，GEMM 按 block_size 辆一卡车拉走。

## PCP（Prefill Context Parallel）的关联

v0.20.2 的 `_maybe_dispatch` 里还有一条分支：`pcp_size > 1` 时通过 `get_pcp_group().all_gather(dim=0)` 把按序列长度切分的片段拼成完整序列，让 MoE 层能看到全部 token 做路由。算完后 `reduce_scatter` 再切开。这是**与 DP 不同维度**的并行策略。
→ 详见 [[Context Parallel]]

## 参见

- [[MoE]]、[[专家并行 EP]]、[[CUDA Graph]]、[[moe_align_block_size]]、[[All-to-All]]、[[Context Parallel]]
