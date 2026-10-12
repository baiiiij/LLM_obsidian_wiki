---
type: source
title: Triton on Ascend 调研（2026-10）
created: 2026-10-12
updated: 2026-10-12
sources: ["raw/sources/Triton-on-Ascend调研-2026-10.md"]
tags: [ascend, triton, 算子层, flaggems]
---

# Triton on Ascend 调研（来源摘要）

> 2026-10-12 网络调研（17 个来源，以 GitHub 一手仓库/官方文档站为主）。原始笔记：[[Triton-on-Ascend调研-2026-10|raw 文件]]。

## 核心结论

- **Triton kernel 可以跑在昇腾上**：`triton-ascend` 已迁入 **triton-lang 官方组织**（成为官方后端之一），华为昇腾团队实际维护；当前正式版 3.2.2（配 CANN 9.1.0）
- **编译路径不同**：不走 TritonGPU→LLVM→PTX，而是 **TTIR → Linalg IR → BiSheng Compiler（AscendNPU IR: HFusion/HIVM）→ NPU 二进制**；950 系列新增 `compile_mode` SIMT 路径
- 迁移成本低：`@triton.jit` kernel 零改动，只改 host 侧 `.cuda()`→`.npu()`
- **主要坑**：与社区 triton 包同名冲突；CANN/torch_npu/triton 版本强耦合锁定；`tl.dot` 无 fp8（950 前）、tf32 映射为 HF32、`max_num_imprecise_acc` 静默忽略；`num_warps` 语义不同（仅编译提示）；grid 物理绑核；UB Overflow 高频；i64 标量化性能杀手
- 生态已接通：[[FlagGems]] 正式 NPU 后端（`GEMS_VENDOR=ascend`）、vLLM-Ascend pip 依赖 triton-ascend、**torch.compile Inductor 在 torch_npu 26.1.0 起默认生成 Triton 算子**、SGLang sgl-kernel-npu 双轨（Ascend C + Triton）
- 第二条编译链路：智源 FlagTree 的 Triton-TLE → FLIR → AscendNPU IR（2026-04 官宣）
- 性能：官方称 GroupGEMM 等场景与手写 Ascend C "有竞争力"，但缺权威对比；旁证显示手写 PTO-ISA 仍优于 Triton 生成码（sgl-kernel-npu PR #462：2–15%）

## 沉淀的 wiki 页

- [[triton-ascend]]（概念：项目、编译链、能力矩阵、坑位清单）
- 更新 [[Triton]]、[[FlagGems]]、[[vLLM]]

## 主要来源

- github.com/triton-lang/triton-ascend（README/架构文档/RFC #1193）、triton-ascend.readthedocs.io（FAQ/迁移指南）
- flagos-ai.github.io/FlagGems Ascend 调优页（最实操的坑位参考）
- vllm-ascend RFC #8010（Inductor 适配）、TorchNPU 26.1.0 Inductor 文档

⚠️ 未确认项（详见 raw 文件）：迁入 triton-lang 的精确日期；CSDN 二手性能数据。
