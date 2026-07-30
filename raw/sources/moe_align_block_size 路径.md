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
*本文档由用户开题、Claudian 对照 v0.20.2 源码（commit bc150f5）补充整理，待用户确认后 ingest。*
