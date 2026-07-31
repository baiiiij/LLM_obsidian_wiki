---
type: concept
title: LLM 前向计算（Transformer 基础流程）
created: 2026-07-31
updated: 2026-07-31
sources: ["raw/sources/moe_align_block_size交互讲解.html", "raw/sources/moe_align_block_size 路径.md"]
tags: [llm, transformer, 模型层]
---

# LLM 前向计算（Transformer 基础流程）

> 所属层次：[[purpose|模型层]]。从零背景页：假设读者**从没见过大模型计算流程**。
>
> 本页是 [[moe_align_block_size]] 等算子页的共同前置背景——讲清三件事：一次前向计算里发生了什么；FFN 为什么是每层最重的计算；GEMM 的**计算操作与访存操作**如何决定算子快慢。

## 一、最小单元：token 与 hidden states

- 输入一句话，模型先把它切成 **token**（词元，可以理解为半个词到几个词）
- 每个 token 经词嵌入变成一个**向量**（比如 4096 个浮点数），叫 hidden state
- 一批 T 个 token 就是一张二维表 `hidden_states`，形状 `[T, D]`（T = token 数，D = 向量维度）
- 整个模型干的事：让这 T 个向量**逐层变换**，越变越"懂"上下文，最后一层输出对下一个词的预测

## 二、一层 Transformer：attention + FFN

Transformer 的每一层做两件事：

1. **Attention**（注意力）：让 token 之间互相"看"、交换信息
2. **FFN**（前馈网络）：对每个 token **独立地**做非线性变换——这是每层最重的一步计算

FFN 的形态（D = 隐藏维度，中间维度一般 4D 左右）：

```
h_out = W_down( activation( h_in @ W_up ) )
```

即**两次矩阵乘（GEMM）+ 中间一次激活函数**：

| 步骤 | 计算 | 形状 |
|---|---|---|
| 第一段 | `X @ W_up` | `[T, D] × [D, 4D] → [T, 4D]` |
| 激活 | SiLU（逐元素） | `[T, 4D]` |
| 第二段 | `... @ W_down` | `[T, 4D] × [4D, D] → [T, D]` |

## 三、GEMM 里的两类操作：计算与访存

矩阵乘 `Y = X @ W` 在 GPU 上同时做两件事：

- **计算（compute）**：M×N×K 次乘加。GPU 算力单位是 TFLOP/s
- **访存（memory）**：从显存读 X（M×K 个元素）、读 W（K×N 个元素）、写回 Y（M×N 个元素）。GPU 访存单位是 GB/s（带宽）

一个算子跑多快取决于哪个先到极限——数据搬不动，算力再高也白等（这就是 roofline 模型的核心思想）。

**关键认知**：FFN 的权重 W_up/W_down 是大矩阵（动辄几十 GB），而每个 token 的输入只是一行向量。**"读权重"花的带宽往往比"算它"花的算力更值钱**。所以"一份权重数据被多少个 token 重复使用"（权重复用次数）是 FFN 快慢的核心指标。

## 四、向后看：这些事实如何决定下层形态

- FFN 是每层最重的计算 → 模型层用 [[MoE]] 把 FFN 换成"路由器 + 一群专家"：参数量 ×E，但每 token 只过 top-k 个专家（稀疏激活）
- 权重复用是核心指标 → 算子层的布局优化（[[moe_align_block_size]] 前置整理 + [[Grouped GEMM]] 按块复用）都围绕"让一次权重加载服务更多 token"
- 权重巨大、读多算少 → 小 batch 推理是 memory-bound，省 kernel launch 比省算力值钱（naive 捷径的 roofline 依据，见 [[moe_align_block_size]] §8.4）
- attention 一侧的算子优化（FlashAttention 等）——待补充

## 参见

- [[MoE]]、[[moe_align_block_size]]、[[Grouped GEMM]]、[[vLLM]]
