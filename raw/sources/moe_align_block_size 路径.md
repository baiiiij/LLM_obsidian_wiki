# vLLM 0.20.2：Qwen3-30B-A3B 走到 moe_align_block_size 的代码分支路径

> 模型：Qwen3-30B-A3B（128 专家，top-8，48 层全部为 MoE 层，无共享专家）
> 版本：vllm-project/vllm @ v0.20.2（commit bc150f5）
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

### 1.1 分支①：mlp 是 MoE 还是 Dense（DecoderLayer.__init__, L398-412）

```python
if (layer_idx not in mlp_only_layers) and \
   (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0):
    self.mlp = Qwen3MoeSparseMoeBlock(...)   # ← Qwen3-30B-A3B 走这里
else:
    self.mlp = Qwen3MoeMLP(...)              # Dense FFN
```

Qwen3-30B-A3B：`num_experts=128`、`decoder_sparse_step=1`、`mlp_only_layers=[]` → **所有 48 层都走 `Qwen3MoeSparseMoeBlock`**。

### 1.2 背景：TP 下进入 MoE 时的张量状态

理解 MoE 层各分支前，需要明确 TP（Tensor Parallel）场景下 hidden_states 到达 MoE 层时的状态。

**attention 的 all-reduce**。`Qwen3MoeAttention` 的 `o_proj` 是 `RowParallelLinear`（L310）：权重沿输出维切分，每卡只算出部分结果，必须经 all-reduce 求和才得到完整输出。因此 `self_attn` 返回时，**各卡上的 hidden_states 是完全相同的副本**。

**gate 为什么必须是 ReplicatedLinear**。vLLM 的线性层按切分方式分三类（`linear.py`）：

| 类型 | 每卡权重 | 输出 | 通信 |
|---|---|---|---|
| ColumnParallel | 输入维切分 | 输出维也切分 | 无 |
| RowParallel | 输出维切分 | partial sum | all-reduce |
| **ReplicatedLinear**（L289） | **完整复制** | 完整 | 无 |

路由决策（`topk_ids`）必须全局一致：若 gate 用 ColumnParallel，每卡只能看到部分特征，算出的 logits 不同，同一 token 会在不同卡被路由到不同专家，结果错乱。ReplicatedLinear 以**每卡重复计算 gate** 为代价换取一致性，无需通信即可达成共识。

### 1.3 序列并行分支（Qwen3MoeSparseMoeBlock.forward, L234-235）

```python
if self.is_sequence_parallel:
    hidden_states = sequence_parallel_chunk(hidden_states)
```

由 `ParallelConfig.use_sequence_parallel_moe` 开启（配 TP>1），是消除上述 gate 冗余计算的手段。

**机制**：`sequence_parallel_chunk`（`models/utils.py` L815）沿 token 维把 hidden_states 均切成 tp_size 份（不能整除先 padding），每卡只取自己那 1/tp；专家算完后 `tensor_model_parallel_all_gather(dim=0)` 拼回完整序列，再 `[:num_tokens]` 裁掉 padding。

**收益**：每卡只对 1/tp 的 token 做路由和 expert GEMM（消除 gate 冗余），w2 的 all-reduce 也只在 chunk 上进行，通信量从 ~2S 降到 ~2S/tp + S。

**实现细节**：该操作包成 custom op（`torch.ops.vllm.sequence_parallel_chunk_impl`）——小 batch 时输出长度可能为 0，即使显式 padding 也会在 torch.compile/CUDA Graph 下出问题，故注册 custom op 并提供 fake impl。

**与下游的联动**：切分发生在 router 之前，`moe_align_block_size` 看到的 `num_tokens` 是**每卡切分后**的值。例如 TP=8、32 token → 每卡 4 个，恰好触发后文分支⑤的 naive 捷径。

### 1.4 分支②：router 在哪算（Qwen3MoeSparseMoeBlock.forward, L226-258）

```python
if self.experts.is_internal_router:
    # gate 在构造时传给了 FusedMoE，路由在 FusedMoE 内部完成
    final_hidden_states = self.experts(hidden_states=hs, router_logits=hs)  # ← 实际走这里
else:
    router_logits, _ = self.gate(hidden_states)   # dead code，保留备用
    final_hidden_states = self.experts(hidden_states=hs, router_logits=router_logits)
```

构造时 `FusedMoE(gate=self.gate, ...)` → `is_internal_router=True` → **路由在 FusedMoE 内部完成**。

## 二、内部路径：fused_moe 框架层

### 2.1 FusedMoE.forward → MoERunner（`layer.py` L1545）

```python
def forward(self, hidden_states, router_logits, input_ids=None):
    return self.runner.forward(hidden_states, router_logits, input_ids)
```

0.20.2 起 `FusedMoE` 本体只是壳，逻辑委托给 `MoERunner`（`runner/moe_runner.py`）。

### 2.2 MoERunner.forward 的准备步骤（`moe_runner.py` L567）

```
forward
  → apply_routed_input_transform   # Latent MoE 的降维投影（无配置时恒等）
  → _maybe_pad_hidden_states       # 维度补齐（常规情况恒等）
  → _forward_entry                 # custom op：_moe_forward / _moe_forward_shared
      → _forward_impl              # 见 2.4
  → shared/routed 输出合并 + all-reduce（TP 时）
```

`_forward_entry` 走 custom op 包装，让 torch.compile / CUDA Graph 把整段当成一个算子捕获。

**Latent MoE（潜空间 MoE）**。`apply_routed_input_transform`（L284）支持在路由前把 hidden_states 投影到更低维的潜空间（`latent_dim < hidden_dim`）：gate 和 expert GEMM 的计算量与激活内存随之减小，算完再由 `apply_routed_output_transform` 升维还原，类似 AutoEncoder 的 bottleneck。此时原始全维张量被保留给 shared experts 使用。Qwen3-30B-A3B 无此配置，该步为恒等。

**`_maybe_pad_hidden_states`（L428）**。Latent 降维后 `transformed_hidden_dim ≠ moe_config.hidden_dim`，而 fused kernel 的 workspace 与权重形状按 `hidden_dim` 建立，要求输入末维一致，故在进入 kernel 前补零对齐，算完由 `_maybe_reduce_final_output` 裁掉。条件 `not quant_method.skip_forward_padding` 是因为部分量化后端自行处理 shape 对齐，无需 Python 层 padding。

**shared experts（共享专家）**。部分 MoE 架构（Mixtral-8x7B、Qwen1.5-MoE 等）在 routed sparse 专家之外还配一个**所有 token 都经过的密集专家**，承载通用知识、稳定梯度、分担负载。vLLM 中由 `Qwen3MoeSparseMoeBlock.__init__` 检查 `shared_expert_intermediate_size`（L187）创建，`MoERunner` 通过 `_maybe_apply_shared_experts` 调度——可串行执行，也可在独立 CUDA stream 上与 routed 专家重叠。Qwen3-30B-A3B 该配置为 0，**无 shared expert**。

### 2.3 分支③：monolithic 还是模块化（`_apply_quant_method`, L472-526）

```python
if self.quant_method.is_monolithic:
    fused_out = self.quant_method.apply_monolithic(...)   # method 层仅 CPU
else:
    topk_weights, topk_ids = self.router.select_experts(  # ← GPU 走这里
        hidden_states=hidden_states,
        router_logits=router_logits,
    )
    fused_out = self.quant_method.apply(
        layer=layer, x=hidden_states,
        topk_weights=topk_weights, topk_ids=topk_ids, ...)
```

**monolithic 与 modular 的本质区别在输入与路由位置**：

| | 输入 | 路由位置 | 结构 |
|---|---|---|---|
| monolithic | `router_logits` | kernel 内部做 softmax/topk | 路由+专家+归并揉为一体 |
| modular | `topk_ids` + `topk_weights` | 外部 `select_experts` 先做 | prepare → experts → finalize 三段拆分 |

- **路由发生在本步**：`select_experts` 内部 softmax(gate(x)) → top-8，产出 `topk_weights`/`topk_ids`
- **CPU 单体的内部算子**：`cpu_fused_moe.CPUFusedMOE`，一个类内完成 softmax → top-k → 逐专家 GEMM → 加权求和（x86 可选 SGL kernel 加速）
- **为何 method 层 monolithic 仅 CPU**：GPU 后端已全部迁移模块化；CPU 无跨卡 all-to-all（通常单节点），不需要 prepare/finalize 拆分，旧一体实现更简单高效（`unquantized_fused_moe_method.py` L53-57，注释 "Escape hatch for CPU"）
- **易混淆点**：MK 层另有 `FusedMoEKernelMonolithicImpl`（`modular_kernel.py` L1420），服务于 FlashInfer TRTLLM 这类"路由+专家融合"的 GPU kernel。完整结论：**method 层 monolithic 仅 CPU；MK 层 monolithic 还有 FlashInfer 类 GPU 融合 kernel**

### 2.4 通信前置：`_maybe_dispatch` / `_maybe_combine`（`_forward_impl` 内，L661）

`_forward_impl` 的执行顺序为：`_maybe_dispatch`（通信前置）→ `_apply_quant_method`（计算）→ `_maybe_combine`（通信后置）。

**dispatch / combine 概念**：DP+EP 部署下，kernel 前需把 token 发送给目标专家所在的卡（dispatch），kernel 后把结果收回（combine），是一对通信操作。

**集合通信原语对比**：

| 原语 | 语义 | MoE 中的用途 |
|---|---|---|
| All-Reduce | 每卡出一份，大家收归约结果 | TP 的 o_proj/down_proj 求和 |
| All-Gather | 每卡出一份，大家收全量拼接 | 序列并行算完拼回 token |
| Reduce-Scatter | 归约 + 切分，每卡收一段 | PCP 收尾 |
| **All-to-All** | 每卡把数据切成 N 份，第 i 份发给第 i 张卡；同时从每卡各收一份 | **EP dispatch/combine：token 按目标专家路由到对应卡** |

All-to-All 是 MoE EP 部署的主要通信开销，vLLM 因此提供众多 all2all 后端（DeepEP、NVLink、NIXL 等）专门优化。

**分支 A：naive DP/EP dispatch**

```python
# do_naive_dispatch_combine (L656)
return self.moe_config.dp_size > 1 and not self.quant_method.supports_internal_mk
```

满足时调 `get_ep_group().dispatch_router_logits(...)` 做 All-to-All。其中涉及两个关键概念：

- **量化后端（quant_method）**：每种量化方式（FP8/INT8/GPTQ/无量化…）实现为一个 `FusedMoEMethodBase` 子类，负责建权重与执行 MoE 计算。
- **`supports_internal_mk`**（`method_base.py` L34）：`self.moe_kernel is not None` 即为 True，表示 dispatch/combine 已由 kernel 内部的 prepare_finalize 组件完成；为 False 时 runner 只能在外层手动做（"naive"方式）。这是 vLLM 把通信从"runner 外挂"迁移到"kernel 内置"过程中的**过渡期开关**（源码注释亦标注 temporary）。

**分支 B：PCP（Prefill Context Parallel）**

```python
if self.moe_config.pcp_size > 1:
    hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
    router_logits = get_pcp_group().all_gather(router_logits, dim=0)
```

Context Parallel（上下文并行）把超长序列沿 token 维切段、每卡处理一段，以分摊 attention 的 KV 内存与计算。vLLM 区分 **PCP（prefill 阶段）** 与 **DCP（decode 阶段）**，因两阶段计算/访存特征不同（`parallel_state.py` L1281）。

MoE 层遇到 PCP 时，vLLM 选择最简单的 **AgRsAll2All** 方案：kernel 前 `all_gather(dim=0)` 把各段拼回完整序列，kernel 后 `reduce_scatter` 切回。源码注释明确这是临时简单实现，未来将重构进 All2AllManager。

**单卡场景**：`dp_size=1` 且 `pcp_size=1` → 两条分支都不走，前后通信均为恒等映射。

### 2.5 后端选择（初始化期，`oracle/unquantized.py`）

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

**Modular Kernel（MK）架构**：0.20.x 将 MoE 拆为两个可插拔组件——`prepare_finalize`（通信 + 量化预处理 + 结果归并）+ `fused_experts`（纯 GEMM）。不同通信后端即不同的 prepare_finalize 实现，见 `prepare_finalize/` 目录：`no_dp_ep.py`、`naive_dp_ep.py`、`deepep_ht.py`、`deepep_ll.py`、`flashinfer_nvlink_one_sided.py`、`flashinfer_nvlink_two_sided.py`、`nixl_ep.py`、`mori.py`。

### 2.6 路由内部：EPLB 负载均衡分支（`_apply_eplb_mapping`）

`select_experts` 产出逻辑视角的 `topk_ids` 后，若开启 EPLB 还需一步映射。

**EPLB（Expert Parallelism Load Balancer）要解决的问题**：EP 下专家静态分卡，路由却是动态的——热点专家挤爆某张卡，冷门专家闲置。

**机制**：
- **冗余专家**：物理专家数 = 逻辑专家数 + 冗余数，热点逻辑专家可在多张卡上摆副本
- 由此引入两层 ID：**逻辑 ID**（模型权重视角，router 输出）↔ **物理 ID**（部署视角，实际调度与 GEMM 使用）

**该分支的行为**：`enable_eplb` 时调用 `eplb_map_to_physical_and_record`，把 topk_ids 从逻辑 ID 翻译成物理 ID，同时**记录每个物理专家收到的 token 负载**（`expert_load_view`），作为后台重平衡决策的输入；未开启则为恒等。

**实现**：融合 Triton kernel（`router/base_router.py` L111），一次 pass 完成映射 + 计数；`should_record_tensor` 控制本步是否记录（避免预热阶段污染统计）。后台另有 `eplb/async_worker.py` 定期按负载统计重新摆放专家副本（`rebalance_execute.py`）。

## 三、模块化 kernel 内部（`modular_kernel.py`）

### 3.1 FusedMoEKernelModularImpl 调用链（L1016, apply 在 L1332）

```
apply (L1332)
├─ [A] inplace? → 是：output 复用 hidden_states（要求无 shared_experts）
│                 否：新分配 output
├─ _prepare (L1120)
│   ├─ [B] prepare_finalize.supports_async()?
│   │      否（no_dp_ep 等无通信后端）→ 同步 prepare()，assert not DBO
│   │      是（all2all 后端）→ prepare_async()
│   │      └─ [C] DBO 双批次重叠? → hook 挂到 ubatch 上下文，否则直接执行
│   └─ 产出：a1q（可能已量化）、a1q_scale、expert_tokens_meta、topk_ids/weights
├─ _fused_experts (L1196)
│   ├─ [D] M_full == 0? → 空批次短路（EP 下某卡可能一个 token 都收不到）
│   ├─ _allocate_buffers（workspace13 与 output 共享内存，cache1 用完即弃）
│   └─ fused_experts.apply → TritonExperts.apply（见 3.2）
└─ _finalize (L1267)
    ├─ [E] supports_async()?
    │      否 → 同步 finalize()：topk_weights 加权 + 跨 rank reduce
    │      是 → finalize_async
    └─ [F] shared_experts 与 combine 通信重叠（MK_INTERNAL_OVERLAPPED：
           把 dense 专家的计算塞进 combine 的等待窗口）
```

六个分支点的意义：A 省显存；B 取决于通信后端能力；C 以 DBO 让计算/通信互遮；D EP 空批次保护；E 归并是否与通信重叠；F shared expert 利用通信等待时间。

### 3.2 TritonExperts.apply（`fused_moe.py` L1985）

```
apply
  → moe_problem_size、try_get_optimal_moe_config（选 BLOCK_SIZE_M 等 tuning 配置）
  → 分配 intermediate_cache1/2/3（workspace）
  → _prepare_expert_assignment(...)          # ← moe_align_block_size 的调用点
  → invoke_fused_moe_triton_kernel (w13 GEMM)
  → activation (SiLU)
  → 复用对齐结果 → w2 GEMM
```

### 3.3 分支⑤：naive 捷径 vs 正式对齐（`_prepare_expert_assignment`, L1554-1602）⭐

```python
naive_block_assignment = (
    expert_map is None                       # 无 EP
    and num_tokens * top_k_num * 4 <= global_num_experts
    and not (wna16 量化且 block_shape[1] > 0)
)

if naive_block_assignment:
    # 跳过 moe_align_block_size
    # sorted_token_ids=None → 走 fused_moe_kernel 的低延迟路径
    return (None, topk_ids.view(-1),
            full((1,), topk_ids.numel() * BLOCK_SIZE_M))

return moe_align_block_size(
    topk_ids, config["BLOCK_SIZE_M"], global_num_experts,
    expert_map, ignore_invalid_experts=ignore_invalid_experts,
)
```

**对 Qwen3-30B-A3B（E=128, top-8）的推论**：

- 无 EP 时，捷径条件 = `num_tokens × 8 × 4 ≤ 128` → **`num_tokens ≤ 4`**
- 只有极小 batch（≤4 token）才跳过 `moe_align_block_size`
- 只要 ≥5 个 token，或开了 EP → **必然调用 `moe_align_block_size`**
- 结合 1.3 的序列并行：TP=8 时 32 token 被切成每卡 4 个，恰好触发捷径

### 3.4 分支⑥：EP 与否（`moe_align_block_size` 的两个语义）

- `expert_map` 由 `determine_expert_map`（`layer.py` L71）在 EP>1 时生成：全局→本卡映射，非本卡为 -1
- `fused_experts_impl`（旧路径，L1806）调用时传 `ignore_invalid_experts=True`（过滤模式）
- `TritonExperts.apply`（L2062）用默认值 `False`（-1 标记模式，GEMM 遇 -1 整块跳过）

### 3.5 分支⑦：DP+EP 的 batched 变体

- 若部署为 DP+EP（DeepSeek 式），`prepare_finalize` 为 BatchedExperts 格式
- 走 `BatchedTritonExperts` + `batched_moe_align_block_size`（`moe_align_block_size.py` L106）
- token 已被 All-to-All 按专家分段，只补齐不排序

### 3.6 moe_align_block_size 本体（`moe_align_block_size.py` L11-103）

Python 包装层：按上界 `numel + num_experts×(block_size-1)` 预分配三个输出缓冲
→ 调 `ops.moe_align_block_size`（**C++/CUDA kernel**，三段式：原子计数 → 带 pad 前缀和 → scatter）。

## 四、完整调用链一图流（GPU / 单卡无 EP / 无量化 / ≥5 token）

```
Qwen3MoeForCausalLM.forward
└─ Qwen3MoeModel.forward
   └─ Qwen3MoeDecoderLayer.forward          ×48
      └─ Qwen3MoeSparseMoeBlock.forward     [① MoE；序列并行可选]
         └─ FusedMoE.forward (layer.py:1545)
            └─ MoERunner.forward
               ├─ apply_routed_input_transform   [Latent MoE 时降维]
               ├─ _maybe_pad_hidden_states       [维度补齐]
               └─ custom op _moe_forward → _forward_impl
                  ├─ _maybe_dispatch             [naive DP/EP all2all；PCP all-gather]
                  └─ _apply_quant_method         [③ 非 monolithic]
                     ├─ router.select_experts → topk_weights/topk_ids
                     │   └─ [EPLB] 逻辑 ID → 物理 ID + 负载记录
                     └─ UnquantizedFusedMoEMethod.apply
                        └─ forward_cuda → forward_native
                           └─ FusedMoEKernel.apply
                              └─ ModularImpl.apply
                                 ├─ _prepare      [同步/异步；DBO]
                                 ├─ _fused_experts
                                 │   └─ TritonExperts.apply (fused_moe.py:1985)
                                 │      └─ _prepare_expert_assignment (L1554)  [⑤]
                                 │         ├─ tokens≤4 且无EP → naive 捷径（跳过）
                                 │         └─ 否则 → moe_align_block_size (L11)
                                 │                    └─ ops.moe_align_block_size (CUDA)
                                 │      └─ w13 GEMM → SiLU → w2 GEMM
                                 └─ _finalize    [加权+reduce；shared expert 重叠]
                  └─ _maybe_combine              [DP/EP 收结果；PCP reduce-scatter]
```

## 五、分支决策总表

| # | 分叉点 | 条件 | Qwen3-30B-A3B 典型走向 |
|---|---|---|---|
| ① | MoE vs Dense | `num_experts>0` 且稀疏步整除 | MoE（全部 48 层） |
| ② | router 内外 | `is_internal_router`（构造时传了 gate） | 内部 |
| ③ | monolithic | method 层仅 CPU 为 True | False → modular apply |
| ④ | 后端 oracle | 平台/依赖/配置 | CUDA → TritonExperts |
| ⑤ | naive 捷径 | `tokens×top_k×4 ≤ E` 且无 EP | tokens≥5 时走对齐 |
| ⑥ | EP 过滤模式 | `ignore_invalid_experts` | 新模块化路径默认 False |
| ⑦ | batched 变体 | DP+EP 部署 | 单卡不走 |
| ⑧ | 序列并行 | `use_sequence_parallel_moe`（TP>1） | 视部署而定 |
| ⑨ | naive dispatch | `dp_size>1 且 not supports_internal_mk` | 单卡不走 |
| ⑩ | PCP | `pcp_size>1` | 单卡不走 |
| ⑪ | EPLB | `enable_eplb` | 默认关闭 |
| ⑫ | Latent MoE / shared experts | 模型配置 | Qwen3-30B-A3B 均无 |

---
*本文档由用户开题，Claudian 对照 v0.20.2 源码（commit bc150f5）整理，经多轮问答勘误后定稿。*
