# 调研笔记：华为昇腾 Ascend C（C/C++）算子开发

> 调研日期：2026-10-12
> 调研方式：LLM agent 网络调研（websearch/webfetch 多来源综合），主线来源为昇腾社区官方文档（hiascend.com，CANN 7.0/8.x 文档）、官方样例库 gitee.com/ascend/samples
> 状态：已交叉核对，个别项标注"⚠️ 未确认"
> 面向读者：熟悉 CUDA C++ kernel 与 Triton 的算子工程师

---

# 华为昇腾 Ascend C（C/C++）算子开发调研笔记

## 1. Ascend C 语言本质

- **是什么**：Ascend C 是华为 CANN 软件栈中面向算子开发场景的编程语言，2023 年 5 月昇腾 AI 开发者峰会正式发布。它**原生支持 C/C++ 语法**，本质上是"标准 C++ + 一组类库 API + 少量语法扩展（函数类型限定符、GM 地址空间限定符）"，不是新语言。
- **编译与目标**：kernel 代码由 CANN 自带的 CCE 编译器（`ccec`，基于 Clang/LLVM 扩展，⚠️ 该细节官方文档表述为"通过编译器编译和运行时调度"，ccec 基于 LLVM 的说法常见于社区文章）编译为 AI Core 指令。算子运行在昇腾 NPU 的 **AI Core** 上；AI Core 内部有 Scalar / Vector / Cube 三种计算单元 + DMA（MTE）搬运单元，共享片外 Global Memory（GM，HBM/DDR），片上有高带宽小容量 Local Memory（UB、L1、L0A/L0B/L0C 等）。
- **核函数定义**（官方教程原文）：

```cpp
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z);
// GM_ADDR = __gm__ uint8_t* __restrict__
```

`extern "C"` 按 C 规约编译链接；`__global__` 标识核函数；`__aicore__` 标识在 AI Core 上执行；`__gm__` 表示指针指向 Global Memory。入参只支持指针和 C/C++ 内置类型。调用方式借用了 CUDA 风格的三尖括号（语义不同）：

```cpp
kernel_name<<<blockDim, l2ctrl, stream>>>(args);  // blockDim=核数, l2ctrl=nullptr保留, stream为acl流
```

Host 侧用 AscendCL（aclInit / aclrtMalloc / aclrtMemcpy / aclrtCreateStream / aclrtSynchronizeStream）管理设备与数据，与 CUDA Runtime API 角色对应。

## 2. 编程范式：SPMD + 流水线（Pipe/Queue）

### 2.1 SPMD 多核并行

Ascend C 是 **SPMD（Single-Program Multiple-Data）**：所有 AI Core 执行同一份代码，靠内置核号区分数据分片。核号通过 `GetBlockIdx()`（或 `get_block_idx()`）获取，核总数用 `GetBlockNum()`；每个核对 GM 地址加 `block_idx * BLOCK_LENGTH` 偏移处理自己的分片。`blockDim` 在 `<<<>>>` 中指定，等价于 CUDA 的 gridDim，但**没有 block 内 thread 概念**——一个核就是一个执行单元。

### 2.2 流水线编程范式（Pipe / Que）

这是与 CUDA 最大的范式差异。官方把核内程序拆成流水任务，用**队列（Queue）做任务间通信和同步**，用 **TPipe 统一管理片上内存**。Vector 算子固定三阶段：**CopyIn → Compute → CopyOut**：

- **核心类**（`namespace AscendC`）：
  - `TPipe`：片上内存管理者，`pipe.InitBuffer(que, bufferNum, byteSize)` 为 TQue/TBuf 分配内存；进程内全局管道可用 `GetTPipePtr()`。
  - `TQue<TPosition, bufferDepth>`：带同步语义的队列，关键 API：`AllocTensor<T>()`、`EnQue()`、`DeQue<T>()`、`FreeTensor()`。EnQue/DeQue 隐式插入指令队列间的同步（MTE2→V→MTE3），甚至 AllocTensor/FreeTensor 也带 wait/signal 语义。
  - `TBuf<TPosition>`：只管理内存、无队列同步语义的 buffer。
  - `GlobalTensor<T>` / `LocalTensor<T>`：GM / 片上内存张量抽象；`SetGlobalBuffer()` 绑定 GM 地址，LocalTensor 只能从 TQue/TBuf 分配。
  - `TPosition`：逻辑位置枚举。矢量相关：`GM`、`VECIN`、`VECCALC`、`VECOUT`；Cube 相关：`A1`/`A2`（L1/L0A）、`B1`/`B2`、`CO1`（L0C）、`CO2`（UB 中 fixpipe 区）、`TSCM`。
- **典型流程**（LeakyRelu 官例）：

```cpp
// CopyIn
LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
DataCopy(xLocal, xGm[progress * tileLength], tileLength);  // GM→UB
inQueueX.EnQue(xLocal);
// Compute
LocalTensor<half> xLocal = inQueueX.DeQue<half>();
LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
LeakyRelu(yLocal, xLocal, scalar, tileLength);
outQueueY.EnQue(yLocal);
inQueueX.FreeTensor(xLocal);
// CopyOut
LocalTensor<half> yLocal = outQueueY.DeQue<half>();
DataCopy(yGm[progress * tileLength], yLocal, tileLength);  // UB→GM
```

### 2.3 内存搬运与同步

- **DataCopy**：GM↔UB 及片上各 buffer 间搬运的主力 API（底层是 MTE2/MTE3 DMA），按元素个数/对齐块搬运；矩阵搬运有 `LoadData`、结果搬出有 `Fixpipe`（CO1→GM/UB）。
- **显式同步**：`SetFlag<HardEvent::X_Y>(eventID)` / `WaitFlag<HardEvent::X_Y>(eventID)` 控制不同指令队列（V/MTE2/MTE3/M 等）间依赖，eventID 由 `GetTPipePtr()->FetchEventID(event)` 获取；`PipeBarrier<PIPE_*>` 做流水屏障；`SyncAll()` 做多核全局同步（如 Matmul 写 workspace 场景）；跨核/跨 AIC-AIV 还有 `CrossCoreSetFlag/CrossCoreWaitFlag`（⚠️ 细节未逐一核对官方页面，见 cann-ops-adv 源码用例）。

## 3. Vector 与 Cube 编程

- **Vector API**：`Add / Muls / Sub / Div / Max / Exp / ReduceSum / Gather / Transpose` 等，全部以 `LocalTensor` 为操作数、按 tile（一次一个 tileLength 的数据块）做 SIMD 计算，一条指令处理一整块数据。另有封装好的高阶 API（如 lib/ 下的 SoftMax）。
- **Matmul 高层 API**（`#include "lib/matmul/matmul_intf.h"`，官方文档 "Matmul Instructions"）：五步——建对象、初始化、设输入、执行、结束：

```cpp
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
// ... bType/cType/biasType 同理
AscendC::Matmul<aType, bType, cType, biasType> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);  // 初始化
mm.SetTensorA(gm_a); mm.SetTensorB(gm_b); mm.SetBias(gm_bias);
mm.IterateAll(gm_c);          // 或 while (mm.Iterate()) { mm.GetTensorC(gm_c); }
mm.End();
```

  模板参数 `MatmulType<TPosition, CubeFormat, TYPE, ISTRANS, LAYOUT, IBSHARE>` 指定位置/格式（ND/NZ）/dtype/转置/batch 布局；支持 half/fp32/bf16/int8/int4 组合。内部自动完成 GM→A1→A2→Cube→CO1→CO2→GM 的多级搬运与 double buffer（tiling 参数 baseM/baseK/baseN、stepM 等由 Matmul Tiling API 或手写 tiling 给出）。结果也可输出到 CO1 后用 `Fixpipe` 自行搬出。纯 Cube 场景定义 `ASCENDC_CUBE_ONLY` 宏，并用 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 宏隔离 AIC（Cube 核）与 AIV（Vector 核）代码——Atlas A2/A3 架构中 Cube 与 Vector 是分离的核。
- **底层 Cube 编程**：不使用高阶 API 时，需手工管 A1/A2/B1/B2/CO1 队列、`LoadData`、`Mmad` 等指令级接口，样例库 2_features 目录有对应示例。

## 4. 性能手段

- **Tiling**：双层。Host 侧 tiling 函数根据 shape/dtype/平台信息计算：核间切分（blockDim、每核 blockLength）+ 核内切分（tileNum、tileLength），结果序列化为字节流随 `<<<>>>` 传给 kernel，kernel 侧用 `GET_TILING_DATA(tilingData, tiling)` 宏反序列化。工程化开发中 tiling 结构体用宏定义：`BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF / TILING_DATA_FIELD_DEF_ARR / END_TILING_DATA_DEF`，配合 `REGISTER_TILING_DATA_CLASS` 注册（官方"内部关联接口"文档）。Matmul 场景另有 **Matmul Tiling API** 自动生成切分参数。
- **Double buffer（多级流水）**：`pipe.InitBuffer(inQueueX, BUFFER_NUM, bytes)` 中 `BUFFER_NUM=2` 即开启——队列深度为 2 时，CopyIn(i+1) 的 MTE2 搬运可与 Compute(i) 的 Vector 计算重叠，这是流水优化的第一手段；外部数据需能分成偶数块。Matmul 内部对 L1/L0 也自动做 db。
- **Vectorization / 搬运优化**：一次 DataCopy/Add 尽量处理连续大块（对齐 32B）；官方"性能优化实用技巧"系列（流水优化、搬运优化、内存优化、API 优化、Tiling 优化）系统讲解，如减少冗余 DataCopy、用 `Copy`/`Brcb` 替代展开搬运、Matmul 开 NBuffer33 模板等。

## 5. 算子开发全流程（工程化方式）

标准链路（官方"创建算子工程"文档 + samples 仓库）：

1. **写原型定义 json**（算子名、输入输出 dtype/format/shape）→ 用 **msOpGen** 工具生成算子工程骨架：`msopgen gen -i *.json -c ai_core-<soc> -lan cpp -out ./AddCustom`。
2. 工程结构：`op_kernel/`（Ascend C kernel + `GET_TILING_DATA`）、`op_host/`（tiling 函数、InferShape/InferDataType、原型注册 `REGISTER_OP`）。
3. **kernel 实现**：Init / Process（CopyIn-Compute-CopyOut）。
4. **host 侧 tiling 实现**：在 `TilingContext` 里填 TilingData、`context->SetBlockDim(...)`、SetTilingKey 区分 tiling 模板。
5. **测试**：CPU 孪生调试（`__CCE_KT_TEST__` 宏 + `AscendC::GmAlloc` + `ICPU_RUN_KF` 在 CPU 上跑 kernel 做 UT，可 gdb）；**msOpST** 生成上板 ST 用例，与标杆数据比对。
6. **编译打包**：工程 `build.sh` 调 ccec 编 kernel（.o）、编 host 侧 so，打包成 `custom_opp_<os>_<arch>.run` 算子包，安装到 OPP 的 vendors 目录（含 op_proto、op_tiling、kernel binary、算子信息库）。也可提供 **aclnn 单算子 API**（如 aclnnAddCustom）供直调。
7. **接入框架**：
   - **PyTorch / torch_npu**：三种方式——(a) Kernel 直调：用 `torch.library`（`TORCH_LIBRARY` 定义 schema + `TORCH_LIBimpl(..., PrivateUse1, ...)` 注册到 NPU 的 PrivateUse1 DispatchKey，支持 torch.compile）或 Pybind11 扩展，内部 `add_custom<<<blockDim, nullptr, c10_npu::getCurrentNPUStream()>>>` 启动 kernel；(b) 单算子 API 调用（torch_npu OpPlugin 适配）；(c) 图模式经 TorchAir 自定义算子入图。
   - **GE 图调用**：算子包装进 OPP 后，GE（Graph Engine）按原型/tile 库加载，图编译时调用 tiling 函数并下发 kernel。
   - **MindSpore**：custom 算子注册调用。
- **交付件清单**：op_kernel/*.cpp（kernel）、op_host/*（tiling + 原型 + infershape）、算子包 .run（内含 kernel .o、op_proto.so、op_tiling.so、op info 配置）、可选 aclnn 接口库、UT/ST 用例与 golden 数据、（框架接入时）torch.library/Pybind 适配层动态库。

## 6. 与 CUDA C++ 逐条对照

| 维度 | CUDA C++ | Ascend C |
|---|---|---|
| 并行模型 | SIMT：grid×block×thread，每线程算几个元素 | SPMD：一个 AI Core 一个执行实例，`GetBlockIdx()` 分片，核内无线程概念 |
| 核函数 | `__global__ void k(...)`，`<<<grid, block, smem, stream>>>` | `extern "C" __global__ __aicore__ void k(GM_ADDR...)`，`<<<blockDim, nullptr, stream>>>`（blockDim=核数） |
| 数据并行 | threadIdx/blockIdx 算元素偏移 | block_idx 算核级分片偏移；核内按 tile 循环 |
| 片上内存 | shared memory（`__shared__`/动态 smem） | UB/L1 等 Local Memory，经 `TPipe::InitBuffer` + `TQue/TBuf` 显式管理，`LocalTensor` 操作 |
| 全局内存 | `float*` 直接寻址 | `GlobalTensor` + `SetGlobalBuffer`，`__gm__` 指针 |
| 数据搬运 | 指针拷贝 / cp.async / TMA | `DataCopy`（MTE DMA，GM↔UB），矩阵用 `LoadData`/`Fixpipe` |
| 计算粒度 | 标量线程代码（编译器向量化）或 warp 内显式 simd | 天然 SIMD：`Add(z, x, y, tileLen)` 一条调用算整个 tile；Cube 一条调算一块矩阵 |
| 同步 | `__syncthreads()`、warp 原语、barrier | 无核内线程同步；靠 EnQue/DeQue 隐式流水同步 + `SetFlag/WaitFlag`、`PipeBarrier`；核间 `SyncAll()` |
| 双缓冲 | 手工 smem 双 buffer + cp.async 流水 | 队列深度 BUFFER_NUM=2 即自动流水（CopyIn 与 Compute 重叠） |
| Tiling | kernel 内模板参数/常量，或 launcher 计算 | Host 侧 tiling 函数计算 → 字节流传 kernel，`GET_TILING_DATA` 解析；tiling 是一等公民 |
| 高层类比 | cuBLAS/cuDNN 调用；Triton 的 tl.dot | Matmul 高层 API（SetTensorA/IterateAll）；Triton 的 program_id 对应 block_idx，但 Triton 的自动流水在 Ascend C 里要手工用 Que 表达 |

一句话心智模型：**CUDA 是"很多线程各自算一点"；Ascend C 是"每个核跑一条 DMA→Vector/Cube→DMA 的流水线，数据按 tile 流过片上内存"**——更像手工版的 Triton + 显式 DMA，或 DSP/GPGPU 混合风格。

## 7. 老技术栈：TBE DSL / TIK 现状

- **演进线**：TBE DSL（Python DSL，自动切分调度，表达力有限）→ TBE TIK（Python，手工控制搬运/计算，门槛高）→ TIK C++（TIK2，C++ 化）→ **Ascend C**（2023，C/C++，现任主力）。
- **官方结论**：《TBE & AI CPU 算子开发指南》（CANN 社区版文档）明确写明 **"TBE 算子开发后续版本不再演进，推荐参考《Ascend C 算子开发指南》完成自定义算子开发"**。samples 仓库 issue 中华为官方答复亦为新自定义算子推荐 Ascend C。即：**TBE DSL/TIK 处于维护存量状态，已被 Ascend C 取代为新算子开发标准路径**；存量内置算子（ops-nn 等）仍有大量 TBE 实现，新硬件代际上也在向 Ascend C（及更新的 PyPTO 等 Python DSL）迁移。

## 来源清单

1. https://www.hiascend.com/developer/techArticles/20230830-1 — 官方《昇腾Ascend C编程入门教程》：SPMD、Pipe/Que 范式、核函数定义、LeakyRelu 全流程、孪生调试。
2. https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0614.html — 官方 Matmul Instructions：MatmulType、REGIST_MATMUL_OBJ、SetTensorA/SetTensorB/Iterate/IterateAll/End、CO1+Fixpipe 模式、数据类型表。
3. https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/devaids/optool/atlasopdev_16_0018.html — msOpGen 工具概述（工程创建/实现/编译部署/仿真流水图解析）。
4. https://www.hiascend.com/doc_center/source/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0060.html — 官方"创建算子工程"：原型 json → msOpGen 生成工程。
5. https://www.hiascend.com/doc_center/source/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0057.html — 官方"AI框架算子适配 > PyTorch框架"：torch.library（PrivateUse1）/Pybind 注册 + `<<<>>>` 直调 + c10_npu::getCurrentNPUStream。
6. https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/API/ascendcopapi/atlasascendc_api_07_1205.html — Tiling 定义宏 BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF / REGISTER_TILING_DATA_CLASS 等内部关联接口。
7. https://gitee.com/ascend/samples — 官方样例库（operator/ascendc 目录：0_introduction kernel直调/框架调用、2_features matmul/IBShare/double buffer 等）；Matmul IBShare 样例见 2_features/13_matmul_api_ibshare。
8. https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/others/tbeaicpudevg/ — 《TBE & AI CPU 算子开发指南》："TBE 算子开发后续版本不再演进，推荐 Ascend C"（TBE/TIK 现状的官方出处）。
9. https://www.hiascend.com/developer/blog/details/02175216197946795349 — 官方博客：EnQue/DeQue/AllocTensor/FreeTensor 的隐式同步语义。
10. https://gitee.com/ascend/cann-ops-adv — 高阶算子库源码：SetFlag/WaitFlag/FetchEventID 实际用例、Matmul tiling 实践（FFN 算子设计文档）。
11. https://www.hiascend.com/developer/techArticles/20241107-1 及"流水优化"系列 — 官方性能优化技巧：TPipe/TQue 内存管理、流水/Tiling/搬运优化。
12. https://github.com/Ascend/torchair/blob/master/docs/zh/custom_op_graph/in_place_op_cases.md — 图模式（TorchAir）自定义算子入图指引。

⚠️ 未确认/需注意的点：
- "ccec 编译器基于 Clang/LLVM" 出自社区文章，官方文档未显式表述（官方仅说"通过编译器编译和运行时调度"）。
- TIK → TIK C++ → Ascend C 的完整换代时间线部分来自社区，官方仅明确了 TBE 整体不再演进。
- CrossCoreSetFlag/SyncAll 的精确语义建议以 CANN 8.x API 参考（ascendcopapi）为准，本文从 cann-ops-adv 源码用例归纳。
