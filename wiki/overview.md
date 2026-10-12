---
type: overview
title: 全局摘要
updated: 2026-10-12
---

# Overview — 知识库全局摘要

> 每次 ingest 后重新生成本页，反映 wiki 的最新状态。

## 当前状态

- 目标与范围：见 [[purpose]]（LLM 算子开发，五层结构）
- 来源数量：6
- 页面数量：45（6 实体 + 20 概念 + 7 来源摘要 + 5 问答 + 4 粘合页 + 1 对比 + 1 索引 + 1 本页）

## 大类导航

- **PyTorch** — 粘合页 [[PyTorch]]（当前为源码机制子线）：代码生成管线 / ATen Dispatcher / 源码分析工具箱 / 定位方法论 / flatten 实例（定位版 + 全流程逐层详解 + 白话分享版）/ VSCode 联合调试 / debug vs release 差异
- **vLLM 推理框架** — 粘合页 [[vLLM 推理框架]]：vLLM / moe_align_block_size / Grouped GEMM / 并行策略网 / 来源页；新增 vllm-ascend 插件段落
- **LLM 与 MoE 模型** — 粘合页 [[LLM 与 MoE 模型]]（当前为模型基础与结构子线）：LLM 前向计算 / MoE / Shared Experts / Latent MoE / DeepSeek / Triton（新增 triton-ascend 链接）
- **FlagGems** — 入口 [[FlagGems]]（目前单页）：新增 NPU 后端（GEMS_VENDOR=ascend）
- **华为昇腾 Ascend** — 粘合页 [[华为昇腾 Ascend]]（2026-10-12 新建，硬件层首个大类）：Ascend 与 NVIDIA GPU 对照 / Da Vinci 架构 / Ascend NPU / CANN / Ascend C / triton-ascend + 4 份调研来源
- **并行执行模型** — 入口 [[SIMT 与 SIMD 与 SPMD]]（单页，成员 ≥3 时补 hub）

## 主题地图

### 纵向链路（模型层 → 框架层 → 算子层）

```
LLM 前向计算（基础背景：token/FFN/GEMM 计算与访存）
  └─ MoE（模型结构：路由 + top-k 稀疏激活）
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

### 横向链路（昇腾全栈，与 CUDA 对照）

```
Ascend NPU（310P / 910 / 910B / 910C / 950 roadmap；CloudMatrix 384）
  └─ Da Vinci 架构（AI Core = Cube + Vector + Scalar；L1/L0/UB 显式层级；MTE DMA）
       └─ 执行模型：SPMD + 核内 SIMD（vs GPU SIMT，见 SIMT 与 SIMD 与 SPMD）
            └─ CANN 软件栈（AscendCL / GE / HCCL / aclnn / 毕昇；≈ CUDA 全家桶）
                 ├─ Ascend C（原生算子语言：Pipe/Que 流水线、Tiling 一等公民）
                 ├─ triton-ascend（Triton 后端：TTIR→Linalg→BiSheng）
                 └─ 框架：torch_npu（PrivateUse1）/ vllm-ascend / SGLang / FlagGems NPU
```

### 已覆盖 vs 空白

| 层次 | 状态 |
|---|---|
| 1. 模型层 | LLM 前向计算背景页 + MoE 入口页（含 Shared/Latent/SP 扩展）；缺具体模型结构细节（如 MLA、GQA） |
| 2. 框架层 | vLLM MoE 路径已有深度（v0.20.2 runner + 模块化 kernel + vllm-ascend）；PyTorch 大类（当前为源码机制子线：Dispatcher / 代码生成管线 / 源码分析工具箱）+ FlagGems 覆盖机制（含 NPU 后端）；PagedAttention/调度器空白 |
| 3. 算子层 | 2 个具体算子（moe_align + Grouped GEMM）+ Triton 入口 + FlagGems（Triton 算子库）+ **昇腾双线（Ascend C 原生 + triton-ascend 移植）**；FlashAttention 空白 |
| 4. 硬件层 | **已开局（昇腾）**：Da Vinci 架构 + Ascend NPU 产品谱系 + 与 NVIDIA 对照；NVIDIA 自身硬件页、AMD、其他国产芯片空白 |
| 5. 底层抽象层 | **已开局（单页）**：SIMT/SIMD/SPMD 辨析（昇腾对照视角）；指令集、内存层级理论等空白 |

## 建议的深挖方向

1. **FlashAttention / FlashDecoding**：算子层最经典的 kernel 案例，可做成 GPU（Triton/CUDA）vs 昇腾（Ascend C）的双实现对比
2. **SGLang RadixAttention**：框架层横向对比 vLLM 的调度思想（SGLang 已有昇腾后端，可联动）
3. **昇腾实战**：跑通一个 Ascend C 自定义算子（msOpGen → torch_npu 注册）或把库里的 Triton kernel 经 triton-ascend 跑起来，沉淀实操 query 页
4. **NVIDIA 硬件页补全**：SM 架构/Tensor Core 代际，与 Da Vinci 对照页形成双向引用
5. **Triton 编译模型与 TileLang 对比**：算子语言层的核心关注点
