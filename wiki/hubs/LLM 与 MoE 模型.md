---
type: hub
title: LLM 与 MoE 模型（大类粘合页）
created: 2026-07-31
updated: 2026-07-31
sources: []
tags: [llm, moe, hub, 导航]
---

# LLM 与 MoE 模型

> 本大类覆盖 **LLM 与 MoE 模型这个领域的一切**：模型结构、算法、训练与推理行为、代表模型。
> **当前内容集中在"模型基础与结构"子线**——模型结构本身是什么，以及它如何决定下层系统/算子的形态。这是 [[vLLM 推理框架]] 和 [[FlagGems]] 两个大类共同的上游——不理解 MoE 的结构，就无法理解 moe_align / Grouped GEMM / EP 为什么长那样。

## 子线：模型基础与结构（当前唯一子线）

## 推荐阅读顺序（第一次读）

| 顺序 | 页面 | 为什么在这个位置 |
|---|---|---|
| 1 | [[LLM 前向计算]] | **地基**：token/hidden states、FFN = 两次 GEMM、计算与访存、权重复用——所有算子优化的判据（什么时候访存 bound、什么时候计算 bound）都来自这里 |
| 2 | [[MoE]] | **核心结构**：路由 + top-k 稀疏激活 → 参数量与计算量解耦；"对系统/算子层的决定性影响"一节是通往下层大类的桥 |
| 3 | [[Shared Experts]] | **扩展结构①**：所有 token 都经过的 dense 专家（Mixtral/Qwen1.5-MoE 有，Qwen3 无） |
| 4 | [[Latent MoE]] | **扩展结构②**：路由前先降维到 latent space，省 gate + expert 的计算与内存 |
| 5 | [[DeepSeek]] | **典型代表**：MoE 架构 + DP/EP 混合部署，把前面所有结构落到真实模型上 |

## 成员之间的逻辑关系

```
LLM 前向计算（稠密模型的计算/访存常识）
   │ 结构变体：FFN 换成 N 个专家 + 路由器
   ▼
MoE（路由 + top-k 稀疏激活）
   ├── 扩展：Shared Experts（+dense 专家）
   ├── 扩展：Latent MoE（路由前降维）
   └── 代表：DeepSeek（DP+EP 混合部署的典型负载）
```

## 延伸（本大类与其他大类的接口）

- **MoE 结构 → 下层算子**：同专家 token 共享权重 → [[Grouped GEMM]]；路由输出散乱需布局整理 → [[moe_align_block_size]]；专家成为切分单位 → [[专家并行 EP]]（均属 [[vLLM 推理框架]] 大类）
- [[Triton]] — 算子实现语言：vLLM MoE kernel 与 [[FlagGems]] 的共同实现技术，是模型层通往算子实现层的桥梁
- [[PyTorch-ATen-Dispatcher]] — 框架层机制（[[PyTorch]] 大类），模型代码（`nn.Module`）最终经由它落到 kernel

## 成员页清单

- 概念：[[LLM 前向计算]]、[[MoE]]、[[Shared Experts]]、[[Latent MoE]]
- 实体：[[DeepSeek]]、[[Triton]]
