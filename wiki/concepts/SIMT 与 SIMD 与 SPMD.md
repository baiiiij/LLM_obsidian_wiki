---
type: concept
title: SIMT 与 SIMD 与 SPMD
created: 2026-10-12
updated: 2026-10-12
sources: ["raw/sources/昇腾硬件架构调研-2026-10.md", "raw/sources/昇腾CANN与编程模型调研-2026-10.md"]
tags: [执行模型, 底层抽象层, simt, simd, spmd]
---

# SIMT 与 SIMD 与 SPMD

> 所属层次：[[purpose|底层抽象层]]。三种并行执行模型的辨析——这是理解 "CUDA kernel ↔ [[Ascend C]] ↔ [[Triton]]" 写法差异的地基，也回答了 [[purpose]] 的关键问题："执行模型如何决定算子的写法"。

## 三个模型的定义

| 模型 | 全称 | 本质 | 代表 |
|---|---|---|---|
| **SIMD** | Single Instruction Multiple Data | 一条指令对一整块向量数据操作；向量宽度由硬件定 | x86 AVX、ARM NEON/SVE、昇腾 Vector Unit（2048-bit） |
| **SIMT** | Single Instruction Multiple Threads | 程序员写"单线程"标量代码，硬件以 warp（32 线程）为单位锁步执行；每线程独立寄存器与（逻辑上独立的）控制流 | NVIDIA GPU（CUDA core + warp 调度） |
| **SPMD** | Single Program Multiple Data | 一份程序跑在多个独立处理单元上，各单元靠编号区分数据分片；**是编程模型而非硬件执行模型**，核内可以是 SIMD 也可以是 SIMT | MPI 多进程、[[Triton]] 的 program、[[Ascend C]] 的 `GetBlockIdx()` |

关键区分：**SIMD/SIMT 是"一条指令怎么执行"的硬件问题；SPMD 是"程序怎么复制到多个单元"的软件问题**——两者正交。Triton 和 Ascend C 都是 SPMD 编程模型，但底层一个落在 SIMT 硬件（GPU）、一个落在 SIMD 硬件（昇腾 Vector Unit）。

## GPU SIMT vs 昇腾 SIMD 的本质差异

- **CUDA SIMT**：海量线程隐藏延迟；并行度在 threadIdx/blockIdx 的细粒度索引；分支以 warp 为单位分化（divergence）
- **昇腾 SIMD + SPMD**：并行度直接对应**物理 AI Core**，核内一条向量指令处理 UB 中 32B 对齐的整块数据，分支控制归 Scalar Unit；靠**显式流水（DMA↔计算重叠）**而非线程切换隐藏延迟（见 [[Da Vinci 架构]] 六队列乱序执行）

**取舍**：
- 昇腾纯 SIMD 在规则、密集的张量负载上面积/能效更优
- 对不规则、离散访存（gather/scatter、稀疏）灵活性弱于 SIMT
- **佐证：华为官方宣布 950 起转向 SIMD+SIMT 混合架构**（访存粒度 512B→128B，见 [[Ascend NPU]] roadmap），[[triton-ascend]] 也相应新增 `compile_mode` SIMT 路径——等于承认纯 SIMD 的局限并向 GPU 编程模型靠拢

## 对算子写法的具体影响

| 写法差异 | SIMT（CUDA） | SPMD+SIMD（Ascend C） |
|---|---|---|
| 并行表达 | `threadIdx.x` 定位到元素 | `GetBlockIdx()` 定位到数据分片，核内 tile 循环 |
| 延迟隐藏 | 超发线程（occupancy） | 显式 double buffer（队列深度=2） |
| 同步 | `__syncthreads()` | 核内无线程同步，靠 EnQue/DeQue 流水依赖 + `SetFlag/WaitFlag` |
| Triton 的映射 | `tl.program_id` → block → 线程束 | `tl.program_id` → 物理核（grid 物理绑核，`num_warps` 退化为编译提示，见 [[triton-ascend]]） |

## 参见

- [[Da Vinci 架构]]、[[Ascend C]]、[[triton-ascend]]、[[Ascend 与 NVIDIA GPU 对照]]
- 关联：[[CUDA Graph]]（SIMT 世界的定形机制）
- 来源摘要：[[昇腾硬件架构调研-2026-10]]、[[昇腾CANN与编程模型调研-2026-10]]
