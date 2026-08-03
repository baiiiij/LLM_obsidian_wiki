---
type: index
title: Wiki 内容目录
updated: 2026-08-03
---

# Index — 内容目录

> 每次 ingest 后更新本文件。查询时先读这里定位页面。
> 组织方式：**按知识大类分区**——每个大类有一个粘合页（hub，含阅读顺序）+ 成员清单。

## PyTorch — 粘合页：[[PyTorch]]

PyTorch 领域的一切（机制/实现/算法/API）。当前内容集中在源码机制子线：算子从 Python 到 device kernel 的全过程。

- [[PyTorch-ATen-Dispatcher]] — 机制：双层模型（算子 vs kernel）、过表 vs 直连、注册表结构与运行时覆盖
- [[PyTorch-代码生成管线]] — 文件地图：yaml → 三个生成器 → 产物归属；新算子绑定判定；Register 三件套与分片
- [[PyTorch-源码分析工具箱]] — 工具：profiler（表格读法/activities）/ dispatch trace（NDEBUG 坑）/ rg / gdb
- [[aten算子调用链定位方法论]] — 问答：不凭先验经验的七步 SOP + 完整分析链路（含 ④ 断点走法）
- [[pytorch-flatten-调用链路定位]] — 问答：flatten 实例（2.12 行号），computeStride 判定 view/copy
- [[vscode-python-cpp-联合调试pytorch]] — 问答：VSCode 双调试器（gdb + debugpy）单步进 C++，含 launch 配置与坑位

## vLLM 推理框架 — 粘合页：[[vLLM 推理框架]]

MoE token 在 vLLM 里经过哪些模块、算子、卡。

- [[vLLM]] — 实体：v0.20.2 MoERunner + 模块化 kernel 架构、TP/SP/EP/DP/PCP 通信节奏
- [[moe_align_block_size]] — 概念：MoE 布局整理算子（前置计算三段式 → naive/native 捷径）
- [[Grouped GEMM]] — 概念：MoE 核心计算形态，同专家 token 共享权重按块复用
- 并行策略：[[专家并行 EP]] / [[数据并行 DP]] / [[序列并行 SP]] / [[Context Parallel]] / [[EPLB]] / [[All-to-All]] / [[CUDA Graph]]
- 来源摘要：[[moe_align_block_size-代码路径]] / [[moe_align_block_size-交互式详解]]

## FlagGems（Triton 算子库）— 入口：[[FlagGems]]

- [[FlagGems]] — 实体：智源 FlagOS 的 Triton 算子库；torch.library 运行时覆盖 aten kernel（改表不截断 dispatch）；覆盖的三条边界

## LLM 与 MoE 模型 — 粘合页：[[LLM 与 MoE 模型]]

模型领域的一切（结构/算法/训练与推理行为/代表模型）。当前内容集中在模型基础与结构子线：结构如何决定下层系统/算子的形态。

- [[LLM 前向计算]] — 概念：从零背景页（token/FFN/GEMM/访存/权重复用），各算子页共同前置
- [[MoE]] — 概念：混合专家模型入口（路由 + top-k；含 Shared/Latent/SP/EPLB 扩展）
- [[Shared Experts]] / [[Latent MoE]] — 概念：MoE 架构扩展
- [[DeepSeek]] — 实体：MoE 架构大模型系列，DP+EP 混合部署典型负载
- [[Triton]] — 实体：Python 风格 GPU 算子 DSL（vLLM MoE kernel 与 FlagGems 的实现语言）

## 元页面

- [[overview]] — 全局摘要（主题地图、已覆盖 vs 空白、深挖方向）
- [[log]] — 操作日志（append-only）
