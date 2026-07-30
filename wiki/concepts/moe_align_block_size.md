---
type: concept
title: moe_align_block_size（MoE 布局整理算子）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html", "raw/sources/moe_align_block_size 路径.md"]
tags: [vllm, moe, kernel, 算子层]
---

# moe_align_block_size

> 所属层次：[[purpose|算子层]]。[[vLLM]] fused MoE 路径上的预处理 kernel。

## 作用

把路由器输出的散乱 `topk_ids` 整理成 **「按专家连续存放、每专家槽位数补齐到 block_size 整数倍」** 的布局，使 [[Grouped GEMM]] 的每个线程块：

- 只查一次专家号（`expert_ids[pid_m]`）
- 整块复用同一个权重矩阵 `W_e`
- PAD 槽位通过 mask 丢弃

## 在完整调用链中的位置（v0.20.2）

```
Qwen3MoeSparseMoeBlock.forward
└─ FusedMoE.forward → MoERunner.forward
   └─ _forward_impl
      └─ _apply_quant_method
         └─ TritonExperts.apply (fused_moe.py:1985)
            └─ _prepare_expert_assignment (L1554)  ← 调用点
               ├─ naive 捷径 → 跳过本 kernel
               └─ moe_align_block_size (moe_align_block_size.py:11)
                  └─ ops.moe_align_block_size (CUDA C++ kernel)
```

## 三个输出

| 输出 | 含义 |
|---|---|
| `sorted_token_ids` | 对齐后槽位下标；槽位 i ↔ 激活第 i//top_k 行；补位填 numel（越界值，被 mask） |
| `expert_ids` | 第 i 个块用哪个专家；-1 表示整块跳过 |
| `num_tokens_post_padded` | 补齐后总槽位数，决定 GEMM grid 大小 |

## 实现要点

- **三段式**：原子加计数 → 带 pad 前缀和 → scatter 写回
- 输出缓冲按上界 `numel + num_experts×(block_size−1)` 预分配（每专家最多补 block_size−1 个槽）
- **开销可忽略**：仅处理 num_tokens×top_k 个 int32 下标
- 已知问题：官方 main 分支 docstring 中 batched 示例的 `expert_ids` 有笔误（`[0,1,3,3,4,5,5]`），正确结果为 `[0,1,3,3,4,4]`

## naive_block_assignment 捷径分支（v0.20.2 新增）

`_prepare_expert_assignment`（`fused_moe.py` L1554）在调用本 kernel 前先做判断：

```python
naive_block_assignment = (
    expert_map is None          # 无 EP
    and num_tokens * top_k * 4 <= global_num_experts
    and not (wna16 量化且 block_shape[1] > 0)
)
```

若满足：跳过本 kernel，走 `fused_moe_kernel` 的**低延迟特化路径**。

**对 Qwen3-30B-A3B（E=128, top-8）的推论**：仅当 `num_tokens ≤ 4` 且无 EP 时触发捷径。只要 ≥5 token 或开了 EP，**必然调用本 kernel**。

### naive 返回值的"两种方言"

naive 三元组与本 kernel 的输出形似而语义迥异，是同一份"生产者-消费者契约"的两种方言：

| 返回值 | 对齐模式语义 | naive 模式语义 |
|---|---|---|
| `sorted_token_ids` | 槽位 → 对索引的间接查表 | `None`：对索引即 block 号，无需间接层 |
| `expert_ids` | 每 BLOCK_M 槽位块一个专家（已排序） | 直接复用 `topk_ids.view(-1)`：每对（=每块）一个专家 |
| `num_tokens_post_padded` | 有效槽位总数 | `numel × BLOCK_M`，编码"numel 个活跃块" |

### 下游 kernel 特化（`fused_moe.py` L419-434, L769-782）

- grid：naive 时 `EM = numel × BLOCK_M` → `grid_m = numel`，**一个 block 对应一个 (token, expert) 对**
- kernel 以 constexpr `naive_block_assignment=True` 特化：
  ```python
  offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)  # 块号即对索引，其余 lane 越界被 mask
  off_experts = tl.load(expert_ids_ptr + pid_m)              # 展平 topk_ids 的第 pid_m 项
  ```
- 两种模式共享消费代码：`expert_ids[pid_m]` 查本块专家、`offs_token // top_k` 得 token 行号（展平 topk_ids 为 token-major）
- **设计精髓**：naive 模式把"对索引 = block 索引"做成恒等映射，整个砍掉 `sorted_token_ids` 间接表

### 收益与代价

- **收益**：省掉 align kernel 的 launch + 设备同步；省掉排序与补位缓冲
- **代价**：每个 BLOCK_M 宽 tile 只有 1 行有效，浪费 `(BLOCK_M-1)/BLOCK_M` 算力
- **阈值的意义**：`tokens×top_k×4 ≤ E` 保证平均每专家 ≤ 1/4 对——此时对齐也无法形成多样本块，浪费相同，跳过预处理是纯收益

## 与序列并行的联动

[[序列并行 SP]] 的 `sequence_parallel_chunk` 在 router 前把 token 切成 tp_size 份 → 每卡 chunk 的 token 数变小。

- 例：TP=8、32 token → 每卡 4 个 → 恰好满足 `4 × 8 × 4 = 128 ≤ 128` → **触发 naive 捷径**
- 启示：序列并行不仅节省计算/通信，还可能改变 kernel 路径选择

## 变体

- **EP 模式**：配合 `expert_map`，默认模式把非本卡块标 -1；过滤模式（`ignore_invalid_experts=True`）计数阶段直接忽略 → 见 [[专家并行 EP]]
- **batched 变体**（`batched_moe_align_block_size`）：token 已被通信层按专家分段，无需排序，只做补齐打包 → 见 [[数据并行 DP]]

## 与上层/下层的关系

- 上层：解决 [[MoE]] 路由输出散乱的问题；受序列并行 token 切分影响
- 下层：为 [[Triton]] 编写的 `fused_moe_kernel`（[[Grouped GEMM]]）提供可直接消费的定长布局

## 参见

- 来源：[[moe_align_block_size-交互式详解]]、[[moe_align_block_size-代码路径]]
