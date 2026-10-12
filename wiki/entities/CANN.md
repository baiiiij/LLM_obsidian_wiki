---
type: entity
title: CANN
created: 2026-10-12
updated: 2026-10-12
sources: ["raw/sources/昇腾CANN与编程模型调研-2026-10.md", "raw/sources/AscendC算子开发调研-2026-10.md"]
tags: [ascend, cann, 软件栈, 编程模型]
---

# CANN（Compute Architecture for Neural Networks）

> 所属层次：[[purpose|框架层 / 算子层之间]]。昇腾 [[Ascend NPU]] 的异构计算软件栈——**定位相当于 CUDA 软件栈的整体打包**（cudart + 驱动 + cuDNN/cuBLAS + NCCL + 编译器）。2025-08 起编译器与虚拟指令集开放接口、其余全开源（gitee.com/ascend）。

## 架构分层（对照 CUDA 生态）

| CANN 组件 | 作用 | CUDA 对照 |
|---|---|---|
| **AscendCL** | 统一 C/C++ + Python 编程接口：Device/Context/Stream/Event、内存、模型加载、单算子执行 | CUDA Runtime + Driver API |
| **GE（Graph Engine）** | 图编译/融合/切分/内存复用 | TensorRT 图优化 + CUDA Graph 部分职能 |
| **AOL / aclnn** | 内置高性能算子库（NN/Math/Transformer），`aclnn*` 单算子 API | cuBLAS / cuDNN / ATen native |
| **HCCL** | 集合通信（AllReduce/AllGather/ReduceScatter/AllToAll），走 HCCS/RoCE/PCIe | NCCL（+ NVLink/IB） |
| **Runtime（aclrt）** | 硬件资源管理、算子/模型执行 | cudart |
| **毕昇编译器** | Host-Device 异构编译（LLVM 体系） | nvcc / ptxas |
| **[[Ascend C]]** | 原生算子语言（C++ 超集） | CUDA C++ |
| 上层套件 | torch_npu、MindSpeed、MindIE、MindStudio | PyTorch CUDA / apex / TRT-LLM / Nsight |

## Host/Device 编程模型（与 CUDA 几乎一一对应）

```c
aclInit(nullptr);               // ≈ 驱动初始化
aclrtSetDevice(0);              // ≈ cudaSetDevice
aclrtCreateContext(&ctx, 0);    // ≈ cuCtxCreate（CUDA 中隐式，这里显式）
aclrtCreateStream(&stream);     // ≈ cudaStreamCreate
aclrtMalloc(&dev, size, ACL_MEM_MALLOC_HUGE_FIRST);  // ≈ cudaMalloc
aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE);         // ≈ cudaMemcpy
aclrtSynchronizeStream(stream); // ≈ cudaStreamSynchronize
```

关键差异：
- **Context 显式创建**（可多 Context 绑定一个 Device）
- **Kernel 下发**：`kernel<<<blockDim, l2ctrl, stream>>>(args)`——语法像 CUDA，但 **`blockDim` 是 AI Core 核数**，无 block×thread 二维结构（SPMD 模型，见 [[SIMT 与 SIMD 与 SPMD]]）
- **执行模式**：单算子模式（aclnn 逐算子下发，≈ eager）vs 图模式（GE 整图编译；torch_npu 的 ACLGraph ≈ [[CUDA Graph]] 捕获）
- 推理另有 `.om` 离线模型路径（ATC 编译 ONNX → om，`aclmdlLoadFromFile` 加载，≈ TensorRT engine）

## 版本演进与配套锁定

6.x（2022-23）→ 7.x（2023-24）→ 8.0（2024-11）→ 8.2（2025-07）→ 8.5（2026 初）→ **9.1.x（2026-07 起，当前主线）**。

⚠️ **版本强耦合是工程常态**：CANN ↔ torch_npu ↔ [[triton-ascend]] 三者锁定配套（如 CANN 9.1.0 + torch_npu 2.7.1.post8 + triton-ascend 3.2.2），升级必须整组推进。

## 框架对接

- **torch_npu**：PyTorch **PrivateUse1** 后端（`torch.npu.*`、`dist.init_process_group(backend="hccl")`）；`transfer_to_npu` 自动拦截 `torch.cuda` 调用迁移；26.1.0 起 **torch.compile Inductor 默认生成 Triton 算子**（依赖 triton-ascend 3.2.2）
- **vLLM-Ascend**：官方插件（vllm-project/vllm-ascend），见 [[vLLM]]
- **SGLang**：2025-07 起官方支持（`--attention-backend ascend`），基于 torch_npu + triton-ascend

## 工具链对照

`npu-smi`（≈ nvidia-smi）/ `msprof` + `msprof-analyze`（≈ nsys，timeline 可导入 chrome tracing）/ MindStudio Insight（GUI）/ `msopgen`（算子工程生成器，CUDA 无对应）/ ATC（≈ trtexec）

## 参见

- [[Ascend NPU]]、[[Da Vinci 架构]]、[[Ascend C]]、[[triton-ascend]]、[[Ascend 与 NVIDIA GPU 对照]]
- 来源摘要：[[昇腾CANN与编程模型调研-2026-10]]
