---
type: hub
title: 华为昇腾 Ascend（大类粘合页）
created: 2026-10-12
updated: 2026-10-12
sources: ["raw/sources/昇腾硬件架构调研-2026-10.md", "raw/sources/昇腾CANN与编程模型调研-2026-10.md", "raw/sources/AscendC算子开发调研-2026-10.md", "raw/sources/Triton-on-Ascend调研-2026-10.md"]
tags: [ascend, hub, 导航, 硬件层, 算子层]
---

# 华为昇腾 Ascend

> 本大类的主线：**从硬件（Da Vinci 架构）→ 软件栈（CANN）→ 算子开发（Ascend C / Triton），建立昇腾全栈与 CUDA 生态的对照理解**。这是知识库 [[purpose|五层结构]]中"硬件层"的第一个大类，同时覆盖算子层（语言/库）与底层抽象层（执行模型）的相应内容。

## 推荐阅读顺序（第一次读）

| 顺序 | 页面 | 为什么在这个位置 |
|---|---|---|
| 1 | [[Ascend 与 NVIDIA GPU 对照]] | **总览地图**：先建立"昇腾的 X ≈ CUDA 的 Y"的整体坐标系，再深入任何细节 |
| 2 | [[Da Vinci 架构]] | **硬件底座**：AI Core 三单元（Cube/Vector/Scalar）、显式 buffer 层级、MTE——算子写法的物理约束来源 |
| 3 | [[SIMT 与 SIMD 与 SPMD]] | **执行模型**：GPU 靠海量线程隐藏延迟 vs 昇腾靠显式流水隐藏延迟——范式鸿沟所在 |
| 4 | [[CANN]] | **软件栈**：AscendCL/GE/HCCL/aclnn 分层、Host 编程模型、版本配套锁定 |
| 5 | [[Ascend C]] | **原生算子开发**：SPMD + Pipe/Que 流水线、Tiling 一等公民、msOpGen 工程化全流程 |
| 6 | [[triton-ascend]] | **Triton 路线**：kernel 零改动迁移、编译链差异、坑位清单、生态接通现状 |
| 7 | [[Ascend NPU]] | **产品谱系**：310P/910/910B/910C 规格、950/960 roadmap、CloudMatrix 384 超节点 |

## 按主题查

| 我想…… | 去这页 |
|---|---|
| 快速建立昇腾 ↔ CUDA 对照 | [[Ascend 与 NVIDIA GPU 对照]] |
| 理解 AI Core 内部（Cube/Vector/UB/MTE） | [[Da Vinci 架构]] |
| 选芯片 / 看规格与 roadmap | [[Ascend NPU]] |
| 搭环境 / 理解 CANN 分层与版本配套 | [[CANN]] |
| 写第一个昇腾算子（C++） | [[Ascend C]] |
| 把已有 Triton kernel 迁到昇腾 | [[triton-ascend]] |
| 理解为什么昇腾 kernel 没有 thread 概念 | [[SIMT 与 SIMD 与 SPMD]] |

## 延伸（本大类与其他大类的接口）

- [[Triton]] / [[FlagGems]] — 算子语言与算子库（FlagGems 已有正式 NPU 后端）
- [[vLLM]] — vllm-ascend 插件：昇腾上的 MoE 推理路径
- [[PyTorch-ATen-Dispatcher]] — torch_npu 与自定义算子注册依赖的 PrivateUse1 / torch.library 机制
- [[Grouped GEMM]] — triton-ascend 官方宣称与手写 Ascend C 有竞争力的代表算子

## 成员页清单

- 实体：[[Ascend NPU]]、[[CANN]]
- 概念：[[Da Vinci 架构]]、[[Ascend C]]、[[triton-ascend]]
- 对比：[[Ascend 与 NVIDIA GPU 对照]]
- 来源摘要：[[昇腾硬件架构调研-2026-10]]、[[昇腾CANN与编程模型调研-2026-10]]、[[AscendC算子开发调研-2026-10]]、[[Triton-on-Ascend调研-2026-10]]
