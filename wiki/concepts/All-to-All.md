---
type: concept
title: 集合通信原语（All-Reduce / All-Gather / Reduce-Scatter / All-to-All）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size 路径.md"]
tags: [并行, 通信, 框架层, 硬件层]
---

# 集合通信原语

> 所属层次：[[purpose|框架层 ↔ 硬件层]]。分布式并行策略的通信基石。

## 四种核心原语对比

| 原语 | 语义 | 数据量变化 | MoE/LLM 中的典型用途 |
|---|---|---|---|
| **All-Reduce** | 每卡出一份，大家收**归约后的结果** | 不变 | TP 的 o_proj/down_proj partial sum 求和 |
| **All-Gather** | 每卡出一份，大家收**全量拼接** | 每卡 ×N | 序列并行算完拼回完整 token 序列 |
| **Reduce-Scatter** | 归约 + 切分，每卡收**一段结果** | 每卡 ÷N | PCP 收尾；ZeRO 类优化 |
| **All-to-All** | 每卡把数据切成 N 份，第 i 份发给第 i 卡；同时从每卡各收一份 | 不变（重排） | **EP 的 dispatch/combine** |

## All-to-All 与 MoE 的天然对应

[[专家并行 EP|EP]] 场景下，每张 DP 卡手里有一批 token，各自要去不同的专家（分布在不同卡上）：

- 每卡**发出** N 份（按目标卡分包）
- 每卡**收回** N 份（从各卡收属于自己的 token）

这正是 all-to-all 的语义。它是 MoE EP 部署的**主要通信开销**，因此 vLLM 提供多种专用后端优化它（见 `prepare_finalize/` 目录）：

- `naive_dp_ep.py`：基础实现
- `deepep_ht.py` / `deepep_ll.py`：DeepSeek 开源的 DeepEP 高吞吐 / 低延迟 kernel
- `flashinfer_nvlink_one_sided.py` / `two_sided.py`：NVLink 优化
- `nixl_ep.py`、`mori.py`：其他传输层

## 组合模式

- **AgRsAll2All**：All-Gather + Reduce-Scatter 组合替代 All-to-All——vLLM 在 [[Context Parallel|PCP]] 场景用的临时方案
- **通信/计算重叠**：DBO（Dual-Batch Overlap）、shared expert 塞进 combine 等待窗口（见 [[moe_align_block_size-代码路径|代码路径文档]] §3.1）

## 参见

- [[专家并行 EP]]、[[数据并行 DP]]、[[Context Parallel]]、[[vLLM]]
