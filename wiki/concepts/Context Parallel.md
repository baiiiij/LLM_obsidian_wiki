---
type: concept
title: Context Parallel（上下文并行：PCP / DCP）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size 路径.md"]
tags: [并行, 长上下文, 框架层]
---

# Context Parallel（上下文并行）

> 所属层次：[[purpose|框架层]]。面向超长序列的并行策略。

## 定义

把**超长序列沿 token 维切成段**，每卡只处理一段，分摊 attention 的 KV 内存与计算。用于 100K+ 长上下文场景。

## vLLM 的两个变体

| | PCP（Prefill CP） | DCP（Decode CP） |
|---|---|---|
| 阶段 | 预填充（一次性吃整段 prompt） | 解码（逐 token 生成） |
| 特征 | 计算密集、序列长 | 访存密集、序列短 |
| 通信组 | `get_pcp_group()`（`parallel_state.py` L1281） | 独立 DCP 组 |

之所以拆分，是因为 prefill 与 decode 的计算/访存特征差异大，适合不同的并行配置。

## MoE 层与 PCP 的交互（v0.20.2）

token 被切段后，vLLM 的 MoE 实现希望看到完整序列，于是选了最简单的 **AgRsAll2All** 方案（`moe_runner.py` L683）：

```python
# kernel 前：all_gather 拼回完整序列
hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
# kernel 后：reduce_scatter 归约并切回各段
hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)
```

源码注释明确说这是**临时简单实现**（"For simplicity, AgRsAll2All was added separately for PCP here"），未来计划重构进 All2AllManager 抽象。

## 参见

- [[All-to-All]]、[[数据并行 DP]]、[[vLLM]]
