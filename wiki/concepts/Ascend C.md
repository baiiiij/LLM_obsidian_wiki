---
type: concept
title: Ascend C
created: 2026-10-12
updated: 2026-10-12
sources: ["raw/sources/AscendC算子开发调研-2026-10.md", "raw/sources/昇腾CANN与编程模型调研-2026-10.md"]
tags: [ascend, ascend-c, 算子层, 算子语言]
---

# Ascend C

> 所属层次：[[purpose|算子层]]。昇腾原生算子编程语言（2023-05 发布）——**标准 C++ + 类库 API + 少量扩展**（`__aicore__`/`__gm__` 限定符），不是新语言。运行在 [[Da Vinci 架构|AI Core]] 上，由 [[CANN]] 的 ccec 编译。官方已明确 TBE DSL/TIK 不再演进，Ascend C 是新算子标准路径。

## 核函数与下发

```cpp
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z);
// GM_ADDR = __gm__ uint8_t* __restrict__；入参只支持指针和内置类型

// Host 侧（AscendCL）：
add_custom<<<blockDim, nullptr, stream>>>(x, y, z);  // blockDim = AI Core 核数！
```

语法借用了 CUDA 三尖括号，但**语义完全不同**：`blockDim` 指定用几个核，无 block 内 thread 概念。

## 编程范式：SPMD + 显式流水线

**核间 SPMD**：所有核跑同一份代码，`GetBlockIdx()` 区分身份，各核处理自己分片（`block_idx * BLOCK_LENGTH` 偏移）。并行度直接对应物理核（见 [[SIMT 与 SIMD 与 SPMD]]）。

**核内流水线**（与 CUDA 最大的范式差异）：Vector 算子固定三段 **CopyIn → Compute → CopyOut**，用队列做任务间同步：

```cpp
// CopyIn
LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
DataCopy(xLocal, xGm[progress * tileLength], tileLength);  // GM→UB (MTE2)
inQueueX.EnQue(xLocal);       // 隐式插入 MTE2→V 依赖
// Compute
xLocal = inQueueX.DeQue<half>();
LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
LeakyRelu(yLocal, xLocal, scalar, tileLength);             // Vector SIMD
outQueueY.EnQue(yLocal);      // 隐式插入 V→MTE3 依赖
inQueueX.FreeTensor(xLocal);
// CopyOut
yLocal = outQueueY.DeQue<half>();
DataCopy(yGm[progress * tileLength], yLocal, tileLength);  // UB→GM (MTE3)
```

### 核心抽象（`namespace AscendC`）

| 类 | 作用 |
|---|---|
| `TPipe` | 片上内存管理者；`InitBuffer(que, bufferNum, bytes)`，`bufferNum=2` 即开 double buffer |
| `TQue<TPosition, depth>` | 带同步语义的队列（AllocTensor/EnQue/DeQue/FreeTensor 全带 wait/signal） |
| `TBuf<TPosition>` | 只管内存、无同步语义的 buffer |
| `GlobalTensor` / `LocalTensor` | GM / 片上内存张量抽象 |
| `TPosition` | 逻辑位置：`GM`、`VECIN/VECCALC/VECOUT`（Vector 侧）、`A1/A2/B1/B2/CO1/CO2`（Cube 侧，对应 L1/L0A/L0B/L0C/UB） |

**同步**：EnQue/DeQue 隐式流水同步（对应硬件的六队列乱序执行，见 [[Da Vinci 架构]]）；显式用 `SetFlag<HardEvent::X_Y>/WaitFlag`、`PipeBarrier<PIPE_*>`；核间 `SyncAll()`；AIC↔AIV 用 `CrossCoreSetFlag/WaitFlag`。

## Matmul 高层 API（Cube 编程的正路）

```cpp
AscendC::Matmul<aType, bType, cType, biasType> mm;   // MatmulType<TPosition, CubeFormat, dtype...>
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a); mm.SetTensorB(gm_b);
mm.IterateAll(gm_c);   // 内部自动 GM→A1→A2→Cube→CO1→CO2→GM 多级搬运 + double buffer
mm.End();
```

支持 half/fp32/bf16/int8/int4 组合；tiling 参数（baseM/baseK/baseN/stepM…）由 Matmul Tiling API 生成。纯 Cube 场景定义 `ASCENDC_CUBE_ONLY`，用 `ASCEND_IS_AIC`/`ASCEND_IS_AIV` 宏隔离核型代码。

## 性能手段

1. **Tiling（一等公民）**：host 侧 tiling 函数算核间切分（blockDim、blockLength）+ 核内切分（tileNum、tileLength），字节流传 kernel，`GET_TILING_DATA` 反序列化；结构体用 `BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF` 宏定义
2. **Double buffer**：队列深度=2 → CopyIn(i+1) 的 MTE2 与 Compute(i) 的 Vector 重叠——流水优化第一手段
3. **搬运/向量化**：连续大块、32B 对齐、减少冗余 DataCopy

## 开发全流程（工程化）

```
原型 json → msOpGen 生成工程 → op_kernel/(kernel) + op_host/(tiling/原型注册)
  → CPU 孪生调试（__CCE_KT_TEST__，可 gdb）→ msOpST 上板测试
  → 打包 custom_opp_<os>_<arch>.run → 安装到 OPP vendors
  → 框架接入：torch.library(PrivateUse1) / Pybind / GE 图 / aclnn 单算子 API
```

接入 torch_npu 的 kernel 直调方式：`TORCH_LIBRARY` 定义 schema + 注册到 **PrivateUse1** DispatchKey（机制同 [[PyTorch-ATen-Dispatcher]]），内部 `add_custom<<<blockDim, nullptr, c10_npu::getCurrentNPUStream()>>>`。

## 心智模型

> **CUDA 是"很多线程各自算一点"；Ascend C 是"每个核跑一条 DMA→Vector/Cube→DMA 的流水线，数据按 tile 流过片上内存"**——更像手工版 [[Triton]] + 显式 DMA，或 DSP/GPGPU 混合风格。

逐条对照表见 [[Ascend 与 NVIDIA GPU 对照]] §算子语言对照。

## 参见

- [[Da Vinci 架构]]、[[CANN]]、[[Ascend NPU]]、[[triton-ascend]]、[[Triton]]
- 来源摘要：[[AscendC算子开发调研-2026-10]]
