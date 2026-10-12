---
type: source
title: Ascend C 算子开发调研（2026-10）
created: 2026-10-12
updated: 2026-10-12
sources: ["raw/sources/AscendC算子开发调研-2026-10.md"]
tags: [ascend, ascend-c, 算子开发, 算子层]
---

# Ascend C 算子开发调研（来源摘要）

> 2026-10-12 网络调研（12 个来源，主线为 hiascend.com 官方文档与 gitee.com/ascend/samples 样例库）。原始笔记：[[AscendC算子开发调研-2026-10|raw 文件]]。

## 核心结论

- **Ascend C = 标准 C++ + 类库 API + 少量扩展**（`__aicore__` / `__gm__` 限定符），2023-05 发布，编译器为 ccec；核函数 `extern "C" __global__ __aicore__ void k(GM_ADDR...)`
- 编程范式 **SPMD + 显式流水线**：`GetBlockIdx()` 核级分片；核内 CopyIn → Compute → CopyOut 三段流水，用 `TPipe`（片上内存管理）+ `TQue`（带同步语义的队列，EnQue/DeQue 隐式插入 MTE2→V→MTE3 依赖）表达
- 核心抽象：`GlobalTensor`/`LocalTensor`、`TPosition`（GM/VECIN/VECOUT/A1/A2/B1/B2/CO1/CO2）、`DataCopy`（≈ DMA）、`SetFlag/WaitFlag`（跨指令队列同步）
- **Matmul 高层 API**：`Matmul<MatmulType...>` 五步（REGIST_MATMUL_OBJ → SetTensorA/B → IterateAll → End），内部自动多级搬运 + double buffer；纯 Cube 场景用 `ASCEND_IS_AIC/AIV` 宏隔离核型
- **Tiling 是一等公民**：host 侧 tiling 函数算核间+核内切分，字节流传 kernel，`GET_TILING_DATA` 反序列化；double buffer 只需队列深度=2
- 开发全流程：原型 json → **msOpGen** 生成工程 → kernel + host tiling → CPU 孪生调试（`__CCE_KT_TEST__`）→ msOpST 上板 → 打包 custom_opp.run → torch.library（PrivateUse1）/ GE / aclnn 接入
- 官方明确 **TBE DSL/TIK 不再演进**，Ascend C 是新算子标准路径
- 心智模型：CUDA 是"很多线程各自算一点"；Ascend C 是"每个核跑一条 DMA→Vector/Cube→DMA 流水线"——更像手工版 [[Triton]] + 显式 DMA

## 沉淀的 wiki 页

- [[Ascend C]]（概念：语言、范式、开发全流程）
- [[Ascend 与 NVIDIA GPU 对照]]（含 CUDA C++ ↔ Ascend C 逐条对照表）

## 主要来源

- hiascend.com《昇腾Ascend C编程入门教程》、Matmul Instructions API 文档、msOpGen/创建算子工程/PyTorch 框架适配文档
- gitee.com/ascend/samples（官方样例）、gitee.com/ascend/cann-ops-adv（高阶算子源码）

⚠️ 未确认项（详见 raw 文件）：ccec 基于 Clang/LLVM 的说法来自社区；CrossCoreSetFlag 精确语义。
