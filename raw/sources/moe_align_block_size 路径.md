# vLLM 0.20.2：Qwen3-30B-A3B 走到 moe_align_block_size 的代码分支路径

> 模型：Qwen3-30B-A3B（128 专家，top-8，48 层全部为 MoE 层，无共享专家）
> 版本：vllm-project/vllm @ v0.20.2
> 说明："外部" = 模型定义层（`model_executor/models/`），"内部" = fused_moe 框架层（`model_executor/layers/fused_moe/`）

## 一、外部路径：模型定义层

文件：`vllm/model_executor/models/qwen3_moe.py`

```
1. Qwen3MoeForCausalLM.forward (L764)
   → self.model(input_ids, positions, ...)                # Qwen3MoeModel

2. Qwen3MoeModel.forward (L478)
   → embed_tokens → 逐层 Qwen3MoeDecoderLayer.forward

3. Qwen3MoeDecoderLayer.forward (L416)
   → input_layernorm → self_attn → post_attention_layernorm
   → hidden_states = self.mlp(hidden_states)
```

### 分支 ①：mlp 是 MoE 还是 Dense？（DecoderLayer.__init__, L398-412）

```python
if (layer_idx not in mlp_only_layers) and \
   (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0):
    self.mlp = Qwen3MoeSparseMoeBlock(...)   # ← Qwen3-30B-A3B 走这里（128 专家，每步稀疏）
else:
    self.mlp = Qwen3MoeMLP(...)              # Dense FFN
```

Qwen3-30B-A3B：`num_experts=128`、`decoder_sparse_step=1`、`mlp_only_layers=[]` → **所有 48 层都走 `Qwen3MoeSparseMoeBlock`**。

### 分支 ②：router 在哪算？（Qwen3MoeSparseMoeBlock.forward, L226-258）

```python
if self.experts.is_internal_router:
    # gate 在构造时传给了 FusedMoE，路由在 FusedMoE 内部完成
    final_hidden_states = self.experts(hidden_states=hs, router_logits=hs)  # ← 实际走这里
else:
    router_logits, _ = self.gate(hidden_states)   # 注释里说是 dead code，保留备用
    final_hidden_states = self.experts(hidden_states=hs, router_logits=router_logits)
```

Qwen3MoeSparseMoeBlock 构造时 `FusedMoE(gate=self.gate, ...)` → `is_internal_router=True` → **走内部分支**。

## 二、内部路径：fused_moe 框架层

### 4. FusedMoE.forward（`layer.py` L1545）

```python
def forward(self, hidden_states, router_logits, input_ids=None):
    return self.runner.forward(hidden_states, router_logits, input_ids)
```

0.20.2 起 `FusedMoE` 本体只是壳，逻辑委托给 `MoERunner`（`runner/moe_runner.py`）。

### 5. MoERunner.forward（`moe_runner.py` L567）

```
forward
  → apply_routed_input_transform   # 恒等（除非 latent MoE）
  → _maybe_pad_hidden_states       # 恒等（普通情况）
  → _forward_entry                 # custom op：_moe_forward / _moe_forward_shared
      → _forward_impl
          → _apply_quant_method    # ← 关键分叉
  → shared/routed 输出合并 + all-reduce（TP 时）
```

`_forward_entry` 走 custom op 包装（`_moe_forward`），目的是让 torch.compile / CUDA Graph 把整段当成一个算子捕获。

### 分支 ③：monolithic 还是模块化？（`_apply_quant_method`, L472-526）

```python
if self.quant_method.is_monolithic:
    fused_out = self.quant_method.apply_monolithic(...)   # 仅 CPU 后端
else:
    topk_weights, topk_ids = self.router.select_experts(  # ← GPU 走这里
        hidden_states=hidden_states,
        router_logits=router_logits,   # 内部 router 在这里真正算 gate
    )
    fused_out = self.quant_method.apply(
        layer=layer, x=hidden_states,
        topk_weights=topk_weights, topk_ids=topk_ids, ...)
```

- **路由发生在这一步**：`select_experts` 内部 softmax(gate(x)) → top-8，产出 `topk_weights`/`topk_ids`
- GPU + 无量化 → `UnquantizedFusedMoEMethod`，非 monolithic → 走 `apply`

### 6. UnquantizedFusedMoEMethod.apply（`unquantized_fused_moe_method.py` L253）

```
apply → CustomOp.forward 按平台分发
  → forward_cuda (L291) → forward_native (L269)
      → self.moe_kernel.apply(...)     # 模块化 kernel
```

### 分支 ④：后端选择（初始化期，`oracle/unquantized.py`）

`select_unquantized_moe_backend` 决定 `experts_cls`：

| 后端 | 触发条件 | experts 类 |
|---|---|---|
| TRITON | **CUDA 默认** | `TritonExperts` |
| BATCHED_TRITON | DP+EP（BatchedExperts 格式） | `BatchedTritonExperts` |
| FLASHINFER_TRTLLM / CUTLASS | 装了 FlashInfer 且条件满足 | 对应类 |
| AITER | ROCm | 对应类 |
| CPU / TPU / XPU / OOT | 对应平台 | 各自类 |

CUDA 裸跑默认 → **TRITON**，经 `make_unquantized_moe_kernel` 组装成
`FusedMoEKernel(prepare_finalize + TritonExperts)`（`modular_kernel.py`）。

### 7. TritonExperts.apply（`fused_moe.py` L1985）

```
apply
  → moe_problem_size、try_get_optimal_moe_config（选 BLOCK_SIZE_M 等 tuning 配置）
  → 分配 intermediate_cache1/2/3（workspace）
  → _prepare_expert_assignment(...)          # ← moe_align_block_size 的调用点！
  → invoke_fused_moe_triton_kernel (w13 GEMM)
  → activation (SiLU)
  → _prepare_expert_assignment 的结果复用 → w2 GEMM
```

### 分支 ⑤：naive 捷径 vs 正式对齐（`_prepare_expert_assignment`, L1554-1602）⭐

```python
naive_block_assignment = (
    expert_map is None                       # 无 EP
    and num_tokens * top_k_num * 4 <= global_num_experts
    and not (wna16 量化且 block_shape[1] > 0)
)

if naive_block_assignment:
    # 跳过 moe_align_block_size！
    # sorted_token_ids=None → 走 fused_moe_kernel 的 "sorted_token_ids is None" 低延迟路径
    return (None, topk_ids.view(-1),
            full((1,), topk_ids.numel() * BLOCK_SIZE_M))

return moe_align_block_size(
    topk_ids, config["BLOCK_SIZE_M"], global_num_experts,
    expert_map, ignore_invalid_experts=ignore_invalid_experts,
)
```

**对 Qwen3-30B-A3B（E=128, top-8）的推论**：

- 无 EP 时，捷径条件 = `num_tokens × 8 × 4 ≤ 128` → **`num_tokens ≤ 4`**
- 即：只有极小 batch（≤4 token）才跳过 `moe_align_block_size`
- 只要 ≥5 个 token，或开了 EP → **必然调用 `moe_align_block_size`**

### 分支 ⑥：EP 与否（`moe_align_block_size` 的两个语义）

- `expert_map` 由 `determine_expert_map`（`layer.py` L71）在 EP>1 时生成：全局→本卡映射，非本卡为 -1
- `fused_experts_impl`（旧 monolithic 路径，L1806）调用时传 `ignore_invalid_experts=True`（过滤模式）
- `TritonExperts.apply`（L2062）用默认值 `False`（-1 标记模式，GEMM 遇 -1 整块跳过）

### 分支 ⑦：DP+EP 的 batched 变体

- 若部署为 DP+EP（DeepSeek 式），`prepare_finalize` 为 BatchedExperts 格式
- 走 `BatchedTritonExperts` + `batched_moe_align_block_size`（`moe_align_block_size.py` L106）
- token 已被 All-to-All 按专家分段，只补齐不排序

### 8. moe_align_block_size（`moe_align_block_size.py` L11-103）

Python 包装层：按上界 `numel + num_experts×(block_size-1)` 预分配三个输出缓冲
→ 调 `ops.moe_align_block_size`（**C++/CUDA kernel**，三段式：原子计数 → 带 pad 前缀和 → scatter）

## 三、完整调用链一图流（GPU / 单卡无 EP / 无量化 / ≥5 token）

```
Qwen3MoeForCausalLM.forward
└─ Qwen3MoeModel.forward
   └─ Qwen3MoeDecoderLayer.forward          ×48
      └─ Qwen3MoeSparseMoeBlock.forward     [分支①②: MoE + 内部router]
         └─ FusedMoE.forward (layer.py:1545)
            └─ MoERunner.forward
               └─ custom op _moe_forward → _forward_impl
                  └─ _apply_quant_method    [分支③: 非monolithic]
                     ├─ router.select_experts → topk_weights/topk_ids
                     └─ UnquantizedFusedMoEMethod.apply
                        └─ forward_cuda → forward_native
                           └─ FusedMoEKernel.apply (modular)
                              └─ TritonExperts.apply (fused_moe.py:1985)
                                 └─ _prepare_expert_assignment (L1554)  [分支⑤]
                                    ├─ tokens≤4 且无EP → naive 捷径（跳过）
                                    └─ 否则 → moe_align_block_size (L11)
                                               └─ ops.moe_align_block_size (CUDA)
                                 └─ invoke_fused_moe_triton_kernel (w13) → SiLU → (w2)
```

## 四、分支决策表

| # | 分叉点 | 条件 | Qwen3-30B-A3B 典型走向 |
|---|---|---|---|
| ① | MoE vs Dense | `num_experts>0` 且稀疏步整除 | MoE（全部 48 层） |
| ② | router 内外 | `is_internal_router`（构造时传了 gate） | 内部 |
| ③ | monolithic | 仅 CPU 为 True | False → apply |
| ④ | 后端 oracle | 平台/依赖/配置 | CUDA → TritonExperts |
| ⑤ | naive 捷径 | `tokens×top_k×4 ≤ E` 且无 EP | tokens≥5 时走对齐 |
| ⑥ | EP 过滤模式 | `ignore_invalid_experts` | 新模块化路径默认 False |
| ⑦ | batched 变体 | DP+EP 部署 | 单卡不走 |

---

# 附：问答记录（ingest 时一并整理）

## Q1: `Qwen3MoeSparseMoeBlock.forward` 里的 `is_sequence_parallel` 分支是干什么的？

**答**：这是 MoE 序列并行（Sequence Parallel MoE），由 `ParallelConfig.use_sequence_parallel_moe` 开启（配 TP>1）。

- **作用**：`sequence_parallel_chunk`（`models/utils.py` L815）沿 token 维把 hidden_states 均切成 tp_size 份（不能整除先 padding），每卡只处理自己那 1/tp；算完专家后 `tensor_model_parallel_all_gather(dim 0)` 拼回完整序列，再 `[:num_tokens]` 裁掉 padding
- **动机**：TP 下 attention 后 hidden_states 是各卡完全相同的副本，gate 又是 ReplicatedLinear → 不切分则每卡对全部 token 重复算路由；SP 让每卡只路由 + 只 GEMM 自己的 chunk，同时 w2 的 all-reduce 也只在 chunk 上进行，通信量从 ~2S 降到 ~2S/tp + S
- **实现细节**：包成 custom op（`torch.ops.vllm.sequence_parallel_chunk_impl`）因为小 batch 输出长度可能为 0，torch.compile/CUDA Graph 下会出问题，故提供 fake impl
- **与分支⑤的联动**：切分发生在 router 之前 → moe_align_block_size 看到的 num_tokens 是**每卡切分后**的值；如 TP=8、32 token → 每卡 4 个，恰好触发 naive 捷径跳过对齐 kernel

## Q2: ReplicatedLinear（每卡一份完整复制）是什么概念？

**答**：vLLM 里的"复制型线性层"（`linear.py` L289），每卡持有完整权重副本（非切分）。

- **为什么 gate 必须是 ReplicatedLinear**：路由决策（`topk_ids`）必须全局一致，若 gate 是 ColumnParallel，每卡只看部分特征 → logits 不同 → token 被发往不同专家 → 结果错乱。ReplicatedLinear 用计算冗余换一致性，无需通信即可达成共识
- 代价：每卡重复跑完整 gate GEMM；序列并行（SP-MoE）的意义之一就是砍掉这份冗余

## Q3: `apply_routed_input_transform` 注释里的 latent MoE 是什么？

**答**：**Latent MoE = 路由前先投影到更低维的潜空间**。

- `apply_routed_input_transform`（`moe_runner.py` L284）若配置了 `routed_input_transform`（降维线性层），会把 hidden_states 投影到 latent_size < hidden_dim → gate + expert GEMM 的计算量/内存减少 → 算完后再 `apply_routed_output_transform` 升维回原始尺寸
- 类似 AutoEncoder bottleneck 思想；Qwen3-30B-A3B 没有降维配置，此分支为恒等

## Q4: `_maybe_pad_hidden_states` 的 padding 条件是什么？shared_experts 又是什么概念？

**答**：
- **padding 作用**：`transformed_hidden_dim`（经过 latent transform 后）可能小于 `moe_config.hidden_dim`，而 fused kernel 的 workspace/weight 按 hidden_dim 对齐。条件 `skip_forward_padding == False and 两维不等` 时补零，kernel 跑完后再裁掉
- **shared_experts**：模型架构自带概念（Mixtral、Qwen1.5-MoE 等），在 routed sparse 专家外再配一个"所有 token 都跑"的密集专家（类似标准 FFN），分担通用知识。vLLM 的 `FusedMoE` / `MoERunner` 通过 `shared_experts_input` 接入，可串行也可开独立 CUDA stream 并行重叠
- **Qwen3-30B-A3B**：`shared_expert_intermediate_size = 0`（`qwen3_moe.py` L187-209）→ **无 shared expert**

## Q5: `_forward_impl` 里的 `_maybe_dispatch` 有哪些条件分支？

**答**：`_maybe_dispatch`（`moe_runner.py` L661）在 kernel 前做通信前置，两条分支：

| 分支 | 条件 | 行为 | Qwen3-30B-A3B |
|---|---|---|---|
| A. naive DP/EP dispatch | `dp_size>1 and not supports_internal_mk` | `get_ep_group().dispatch_router_logits(...)` 做 All-to-All 发 token | 单卡 `dp_size=1` → **不走** |
| B. Prefill Context Parallel (PCP) | `pcp_size>1` | `get_pcp_group().all_gather(dim=0)` 拼序列长度 | 单卡 `pcp_size=1` → **不走** |

完整通信节奏：`_maybe_dispatch`（前置）→ `_apply_quant_method`（计算）→ `_maybe_combine`（后置收结果）。对单卡无量化场景，前后通信都是恒等映射。

## Q6: `supports_internal_mk`、量化后端、combine、"封装进 modular kernel" 分别是什么概念？

**答**：
- **量化后端（quant_method）**：每种量化方式（FP8/INT8/GPTQ/无量化…）实现为一个 `FusedMoEMethodBase` 子类，负责建权重 + 执行 MoE 计算
- **dispatch / combine**：DP+EP 下 kernel 前把 token 发给专家所在卡（dispatch），kernel 后收回结果（combine），是一对通信操作
- **Modular Kernel（MK）**：0.20.x 新架构，`FusedMoEKernel = prepare_finalize（通信+量化预处理+归并）+ fused_experts（纯 GEMM）`；不同通信后端 = 不同 prepare_finalize 实现（`no_dp_ep`/`naive_dp_ep`/`deepep_ht`/`deepep_ll`/`flashinfer_nvlink_*`/`nixl_ep`/`mori`）
- **`supports_internal_mk`**（`method_base.py` L34）：`self.moe_kernel is not None` 即 True，表示 dispatch/combine 已由 prepare_finalize 在 kernel 内部完成；为 False 时 runner 只能在外层手动做（"naive"方式）。这是通信从"runner 外挂"迁移到"kernel 内置"的过渡期开关

## Q7: All-to-All 是什么概念？

**答**：集合通信原语：组内每卡把数据切成 N 份，第 i 份发给第 i 张卡，同时从每卡各收一份。对比：All-Reduce（收归约结果，TP 用）、All-Gather（收全量拼接，SP 用）、Reduce-Scatter（归约+切分）。EP 的 dispatch/combine 天然是 all-to-all（token 按目标专家路由到对应卡），是 MoE EP 部署的主要通信开销——这就是 vLLM 有众多 all2all 后端（DeepEP/NVLink/NIXL…）专门优化它的原因。

## Q8: PCP（Prefill Context Parallel）详解 —— 含勘误

**答**：⚠️ 勘误：PCP = **Prefill Context Parallel**（非 Pipeline），`parallel_state.py` L1281 明确。

- **Context Parallel**：超长序列沿 token 维切段，每卡处理一段，分摊 attention 的 KV 内存与计算
- vLLM 区分 **PCP（prefill 阶段）** 与 **DCP（decode 阶段）**，因两阶段计算/访存特征不同
- MoE 层遇 PCP：vLLM 选择最简单的 **AgRsAll2All** 方案——kernel 前 `all_gather(dim=0)` 拼回完整序列，kernel 后 `reduce_scatter` 切回各段。源码注释明确说是临时简单实现，未来将重构进 All2AllManager

## Q9: `apply_monolithic` 的概念、原理、内部算子、为何仅 CPU？

**答**：
- **monolithic vs modular 的本质区别在输入**：monolithic 吃 `router_logits`（路由在 kernel 内部做），路由+专家+归并揉成一团；modular 吃 `topk_ids/topk_weights`（外部先路由），prepare→experts→finalize 三段拆
- **内部算子**（CPU 路径）：`cpu_fused_moe.CPUFusedMOE`，一个类内完成 softmax→topk→逐专家 GEMM→加权求和（x86 可选 SGL kernel）
- **为何仅 CPU**：GPU 后端已全部迁移模块化；CPU 无跨卡 all-to-all（单节点），不需要 prepare/finalize 拆分，旧一体实现更简单高效（`unquantized_fused_moe_method.py` L53-57 注释 "Escape hatch for CPU"）
- **易混淆点**：MK 层还有 `FusedMoEKernelMonolithicImpl`（`modular_kernel.py` L1420），用于 FlashInfer TRTLLM 这类"路由+专家融合"的 GPU kernel。完整说法：**method 层 monolithic 仅 CPU；MK 层 monolithic 还有 FlashInfer 类 GPU 融合 kernel**

## Q10: `_apply_eplb_mapping` 分支（EPLB）是干什么的？

**答**：**EPLB = Expert Parallelism Load Balancer**（专家并行负载均衡）。

- **问题**：EP 下专家静态分卡，路由却是动态的——热点专家挤爆某卡，冷门闲置
- **思路**：冗余专家（物理专家数 = 逻辑数 + 冗余数），热点逻辑专家在多卡摆副本 → 引入两层 ID：逻辑 ID（router 输出）↔ 物理 ID（实际调度）
- **该分支做的事**：`enable_eplb` 时把 topk_ids 从逻辑 ID 翻译成物理 ID，同时**记录每个物理专家的 token 负载**（`expert_load_view`，供后台重平衡决策用）；不开则恒等
- **实现**：融合 Triton kernel（`router/base_router.py` L111），一次 pass 完成映射+计数；`should_record_tensor` 控制是否记录（避免预热污染统计）；后台 `eplb/async_worker.py` 定期按负载重摆副本（`rebalance_execute.py`）

## Q11: `FusedMoEKernelModularImpl` 调用链的完整分支

**答**（`modular_kernel.py` L1016）：

```
apply (L1332)
├─ [A] inplace? → 是：output 复用 hidden_states（要求无 shared_experts）
├─ _prepare (L1120)
│   ├─ [B] supports_async()? 否（no_dp_ep）→ 同步 prepare
│   │                        是（all2all 后端）→ prepare_async
│   │   └─ [C] DBO 双批次重叠? → hook 挂 ubatch 上下文
├─ _fused_experts (L1196)
│   ├─ [D] M_full==0? → 空批次短路（EP 下某卡可能收不到 token）
│   ├─ _allocate_buffers（workspace13 与 output 共享内存）
│   └─ fused_experts.apply → TritonExperts.apply → _prepare_expert_assignment
│      → moe_align_block_size（最初入口）
└─ _finalize (L1267)
    ├─ [E] supports_async()? 否 → 同步 finalize（加权+reduce）
    │                        是 → finalize_async
    └─ [F] shared_experts 与 combine 通信重叠（MK_INTERNAL_OVERLAPPED）
```

六个分支点：A 省显存；B 取决于通信后端能力；C DBO 计算/通信互遮；D EP 空批次保护；E 归并是否与通信重叠；F 把 dense 专家塞进 combine 等待窗口。

---
*本文档由用户开题、Claudian 对照 v0.20.2 源码（commit bc150f5）补充整理，待用户确认后 ingest。*
