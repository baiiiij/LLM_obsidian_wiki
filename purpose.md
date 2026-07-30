# Purpose — 这个 Wiki 为什么存在

> 本文件定义 wiki 的**方向与意图**（schema.md 定义结构规则）。
> LLM 在每次 ingest 和 query 时都应阅读本文件以获得上下文。

## 目标

建立一个关于 **LLM 算子（Kernel）开发** 的个人知识库，主线是：

> **从模型结构 → 推理框架 → 算子语言与库 → 硬件与编程模型，逐层向下打通。**

通过持续收录论文、源码解析、官方文档、技术文章，构建一套结构化、互相链接的知识网络，支撑系统性的学习与深入的研究，最终形成自己对"高性能 LLM 算子如何设计与实现"的完整理解。

## 知识地图（五个层次）

### 1. 模型层 —— LLM 热门大模型结构
- 主流架构：Transformer 及其变体（GQA、MLA、MoE、Sliding Window 等）
- 代表模型：Llama、Qwen、DeepSeek、Mistral 等的结构设计与取舍
- 关注点：**哪些计算模式决定了算子的形态**（attention、GEMM、norm、RoPE、sampling……）

### 2. 框架层 —— LLM 推理/训练框架的思想与实现
- 推理框架：vLLM（PagedAttention、Continuous Batching）、SGLang（RadixAttention、结构化生成）等
- 关注点：调度思想、内存管理、算子融合策略，以及**框架如何调用和组合底层算子**

### 3. 算子层 —— 算子开发语言与算子库
- **DSL / 算子语言**：Triton、TileLang（以及 CUDA C++ 作为对照）
- **算子库**：FlagGems 及各类基于这些语言实现的开源算子集合
- **传统 C/C++ 算子**：手写 kernel 的经典实现与优化技巧（融合、tiling、vectorization）

### 4. 硬件层 —— 硬件架构与编程模型
- 不同厂商的硬件知识：NVIDIA（SM/Tensor Core）、AMD、国产芯片（昇腾等）
- 各平台的**编程模型**：CUDA、HIP、CANN 等
- RISC-V 相关知识（指令集、向量扩展 RVV、在 AI 芯片中的应用）

### 5. 底层抽象层 —— 并行执行模型与指令集
- SIMT / SIMD / SPMD 等并行执行模型的本质区别与联系
- 指令集层面的知识：向量指令、矩阵指令、内存层级（寄存器/共享内存/HBM）

## 关键问题（Key Questions）

- 一个大模型从"结构定义"到"在硬件上跑起来"，每一层分别做了什么？
- vLLM / SGLang 的核心加速思想是什么？在算子层面如何落地？
- Triton / TileLang 这类 DSL 如何抽象硬件？相比手写 CUDA/C 的取舍是什么？
- 同一个算子（如 attention）在不同硬件、不同语言下的实现有何异同？
- SIMT/SIMD/SPMD 模型如何决定算子的写法？指令集如何成为性能天花板？

## 范围

- ✅ 收录：上述五层相关的论文、源码分析、官方文档、博客、课程、芯片资料
- ✅ 收录：算子开发实战、性能调优案例、benchmark 分析
- ❌ 不收：纯应用层的 LLM 使用技巧、与算子/系统无关的算法理论

## 演进的论点（Evolving Thesis）

> 随着来源增多，在这里维护你对这个领域不断更新的核心判断。
> 初始为空，由 LLM 在 ingest 过程中提出建议、由你确认后更新。

_（待填充）_
