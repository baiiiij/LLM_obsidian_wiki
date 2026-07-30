---
type: overview
title: 全局摘要
updated: 2026-07-30
---

# Overview — 知识库全局摘要

> 每次 ingest 后重新生成本页，反映 wiki 的最新状态。

## 当前状态

- 目标与范围：见 [[purpose]]（LLM 算子开发，五层结构）
- 来源数量：2
- 页面数量：18（3 实体 + 12 概念 + 2 来源摘要 + 1 索引 + 1 本页）

## 主题地图

### 纵向链路（模型层 → 框架层 → 算子层）

```
MoE（模型结构：路由 + top-k 稀疏激活）
  ├─ Shared Experts（dense 辅助专家）
  ├─ Latent MoE（降维-升维投影）
  ├─ 序列并行 SP（TP 下 token 切分）
  ├─ EPLB（EP 负载均衡：逻辑/物理双层 ID）
  └─ moe_align_block_size（算子：布局整理）
       ├─ naive 捷径分支（tokens≤4 时跳过）
       └─ Grouped GEMM（算子：核心计算，Triton 实现）
  └─ 并行策略网：EP（专家切卡）+ DP（token 切卡）+ PCP（序列切段）
       ├─ 通信原语：All-Reduce / All-Gather / Reduce-Scatter / All-to-All
       └─ CUDA Graph 定形约束 → 定长缓冲方案
```

### 已覆盖 vs 空白

| 层次 | 状态 |
|---|---|
| 1. 模型层 | MoE 入口页，含 Shared/Latent/SP 扩展；缺具体模型结构细节（如 MLA、GQA） |
| 2. 框架层 | vLLM MoE 路径已有深度（v0.20.2 runner + 模块化 kernel）；PagedAttention/调度器空白 |
| 3. 算子层 | 2 个具体算子（moe_align + Grouped GEMM）+ Triton 入口；FlashAttention 空白 |
| 4. 硬件层 | 空白 |
| 5. 底层抽象层 | 空白（SIMT/SIMD/SPMD 尚无页面） |

## 建议的深挖方向

1. **FlashAttention / FlashDecoding**：算子层最经典的 kernel 案例，与 moe_align_block_size 形成对比
2. **SGLang RadixAttention**：框架层横向对比 vLLM 的调度思想
3. **SIMT vs SIMD vs SPMD**：底层抽象层打地基，影响 Triton block 编程模型
4. **Triton 编译模型与 TileLang 对比**：算子语言层的核心关注点
