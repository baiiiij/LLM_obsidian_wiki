---
type: entity
title: vLLM
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html", "raw/sources/moe_align_block_size 路径.md"]
tags: [推理框架, 框架层]
---

# vLLM

> 所属层次：[[purpose|框架层]]。主流 LLM 推理框架。

## 与本知识库相关的要点

### fused MoE 算子路径（来自 [[moe_align_block_size-交互式详解|来源 1]] + [[moe_align_block_size-代码路径|来源 2]]）

#### v0.20.2 架构重构（关键变化）

- `FusedMoE.forward`（`layer.py` L1545）本体变壳，逻辑全部委托给 **`MoERunner`**（`runner/moe_runner.py`）
- 无量化 GPU 路径走**模块化 kernel**（`FusedMoEKernel` = prepare_finalize + experts），后端由 `oracle/unquantized.py` 自动选择：CUDA 默认 **TritonExperts**
- 完整调用链（单卡 GPU）：`MoERunner.forward` → custom op `_moe_forward` → `_forward_impl` → `_apply_quant_method` → `UnquantizedFusedMoEMethod.apply` → `moe_kernel.apply` → `TritonExperts.apply` → `_prepare_expert_assignment` → [[moe_align_block_size]]

#### 关键设计决策

- **应对动态形状**：为兼容 [[CUDA Graph]] 定形要求，采用**定长缓冲 + 有效计数数组**（`expert_num_tokens`）
- **预处理 kernel 定位**：`moe_align_block_size` 只处理 int32 下标，开销相对 GEMM 可忽略——"用廉价的整理换 GEMM 的高效"
- **naive_block_assignment 捷径**（v0.20.2 新增）：无 EP 且 `num_tokens × top_k × 4 ≤ E` 时跳过对齐，走 kernel 内低延迟路径

#### Modular Kernel（MK）架构细节

- `FusedMoEKernel = prepare_finalize（通信+量化预处理+归并）+ fused_experts（纯 GEMM）`，两种 impl：
  - `FusedMoEKernelModularImpl`：常规路径，apply 链含 6 个分支点（inplace / prepare 同步异步 / DBO / 空批次短路 / finalize 同步异步 / shared expert 与 combine 重叠）——详见 [[moe_align_block_size-代码路径|代码路径文档]] §3.1
  - `FusedMoEKernelMonolithicImpl`：吃 router_logits 的融合 kernel（FlashInfer TRTLLM 类）
- **monolithic vs modular 的本质区别在输入**：monolithic 吃 `router_logits`（kernel 内路由），modular 吃 `topk_ids/weights`（外部先路由）；method 层 monolithic 仅 CPU（CPUFusedMOE）
- **`supports_internal_mk`**：`moe_kernel is not None` 即为 True，表示 dispatch/combine 已内置；否则 runner 外层手动做——通信从"runner 外挂"迁移到"kernel 内置"的过渡开关
- **通信后端可插拔**：`prepare_finalize/` 目录 9 种实现（naive/DeepEP/NVLink/NIXL/mori…），见 [[All-to-All]]
- **EPLB 支持**：路由后逻辑 ID → 物理 ID 映射 + 负载记录，见 [[EPLB]]

### TP 与序列并行（来自来源 2）

- TP 下 attention 的 `o_proj`（RowParallelLinear）做完 all-reduce 后，hidden_states 是各卡**完全相同的副本**
- gate 必须使用 **ReplicatedLinear**（`linear.py` L289）：每卡完整复制权重，保证各卡路由决策一致，无需通信达成共识
- **[[序列并行 SP|序列并行 MoE]]**（`sequence_parallel_chunk`，`models/utils.py` L815）：router 前把 token 均切成 tp_size 份，每卡只处理 1/tp；算完 `tensor_model_parallel_all_gather(dim=0)` 拼回
- **与 naive 捷径的联动**：TP=8 时 32 token → 每卡 4 个 → 恰好满足 `num_tokens × top_k × 4 ≤ 128` → 触发跳过对齐 kernel

### 通信节奏（`_maybe_dispatch` / `_maybe_combine`）

- **naive DP/EP dispatch**：`dp_size>1` 且量化后端未内封装时，走 `get_ep_group().dispatch_router_logits` 做 [[All-to-All]]
- **[[Context Parallel|PCP（Prefill Context Parallel）]]**：`pcp_size>1` 时 all-gather 拼序列 → reduce_scatter 切开（AgRsAll2All 临时方案）
- 单卡场景：前后通信均为恒等映射

### 支持的 MoE 变体

- [[专家并行 EP]]（`expert_map`、两种过滤模式）+ [[EPLB]] 动态负载均衡
- [[数据并行 DP]]（`batched_moe_align_block_size` 变体）
- [[Shared Experts]]（模型架构级 dense 专家，可开独立 CUDA stream 重叠）
- [[Latent MoE]]（降维-升维投影，kernel 前需 `_maybe_pad_hidden_states` 补零对齐）
- [[序列并行 SP]]（消除 TP 下 gate 冗余）

## 待补充方向

- PagedAttention、Continuous Batching（[[purpose]] 中的关注点，待相关来源 ingest）
- 调度器与内存管理实现

## 参见

- [[MoE]]、[[Grouped GEMM]]、[[moe_align_block_size]]、[[Triton]]
