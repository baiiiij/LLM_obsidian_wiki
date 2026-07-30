---
type: overview
title: 全局摘要
updated: 2026-07-30
---

# Overview — 知识库全局摘要

> 每次 ingest 后重新生成本页，反映 wiki 的最新状态。

## 当前状态

- 目标与范围：见 [[purpose]]（LLM 算子开发，五层结构）
- 来源数量：1
- 页面数量：10（3 实体 + 6 概念 + 1 来源摘要）

## 主题地图

当前知识集中在一条纵向链路上（模型层 → 框架层 → 算子层）：

```
MoE（模型结构）
  └─ 路由输出散乱 → moe_align_block_size（算子：布局整理）
       └─ Grouped GEMM（算子：核心计算，Triton 实现）
  └─ 部署：EP（专家切卡）+ DP（token 切卡 + All-to-All）
       └─ 动态形状 ↔ CUDA Graph 定形约束 → 定长缓冲方案
```

核心实体：[[vLLM]]（框架）、[[DeepSeek]]（模型）、[[Triton]]（算子语言）。

## 已覆盖 vs 空白

| 层次 | 状态 |
|---|---|
| 1. 模型层 | 有 [[MoE]] 入口页，缺具体模型结构细节 |
| 2. 框架层 | vLLM MoE 路径已有基础；PagedAttention/调度器空白 |
| 3. 算子层 | 1 个具体算子 + Grouped GEMM + Triton 入口 |
| 4. 硬件层 | 空白 |
| 5. 底层抽象层 | 空白（SIMT/SIMD/SPMD 尚无页面） |

## 建议的深挖方向

- SGLang 的 RadixAttention（框架层横向对比）
- FlashAttention 类算子（算子层最经典案例）
- SIMT vs SIMD vs SPMD 辨析（底层抽象层打地基）
