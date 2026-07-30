---
type: source
title: vLLM moe_align_block_size 交互式详解（含 EP / DP）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html"]
tags: [vllm, moe, kernel, ep, dp, triton]
---

# vLLM `moe_align_block_size` 交互式详解

> 来源摘要页。原始文件：`raw/sources/moe_align_block_size交互讲解.html`（交互式网页，含可步进的图示动画）。

## 一句话概括

`moe_align_block_size` 是 [[vLLM]] [[MoE]] 推理路径上的**布局整理 kernel**：把路由器输出的散乱 `topk_ids` 整理成「按专家连续存放、每个专家的槽位数补齐到 block_size 整数倍」的布局，让 [[Grouped GEMM]] 的每个线程块只查一次专家号、整块复用同一个权重矩阵。

## 三个场景

### 1. 基础版
- 示例：8 token、top-2、4 专家、block_size=4
- 详见 [[moe_align_block_size]] 概念页

### 2. [[专家并行 EP|专家并行（EP）]]
- N 个专家切到多张卡，`num_experts` 传**全局**专家数，传入 `expert_map`（全局号 → 本卡局部号，非本卡为 -1）
- 两种模式：
  - **默认模式**（`ignore_invalid_experts=False`）：全部全局专家参与计数/排序/补齐，返回前经 `expert_map` 映射，非本卡专家的块标 -1，MoE GEMM kernel 遇到 -1 整块跳过
  - **过滤模式**（`ignore_invalid_experts=True`）：计数阶段直接忽略非本卡专家，缓冲更小，`expert_ids` 无 -1

### 3. [[数据并行 DP|数据并行（DP）]]—— `batched_moe_align_block_size`
- 为 DP+EP 部署（典型：[[DeepSeek]] 系列）量身定做
- MoE 层流水线：各 DP 卡跑路由器 → **All-to-All Dispatch**（token 发往持有目标专家的卡）→ 各卡只对持有专家做 GEMM → **All-to-All Combine**（结果发回原卡加权求和）
- 痛点：每步每专家收到的 token 数随机，张量形状步步变化 → [[CUDA Graph]] 无法捕获（要求形状固定）
- vLLM 对策：Dispatch 直接写入形状恒定的缓冲 `(E × max_tokens_per_batch, K)`，每专家固定一排，配 `expert_num_tokens` 记录有效数
- batched 变体因此**无需排序**，只做"打包"（向上补齐到 block_size 整数倍），输出语义与基础版相同

## 输入输出速查

| 参数 / 返回 | 含义 |
|---|---|
| `topk_ids` | [num_tokens, top_k]，每个 token 选中的专家编号（EP 时为全局号） |
| `block_size` | GEMM 块长，即 kernel 的 BLOCK_M（通常 16/32/64） |
| `num_experts` | 专家总数；EP 场景必须是全局专家数 |
| `expert_map` | 全局 → 本卡局部专家号映射；不在本卡为 -1 |
| `sorted_token_ids`（出） | 对齐后槽位下标；槽位 i ↔ 激活第 i//top_k 行；补位填 numel（被 mask） |
| `expert_ids`（出） | 第 i 个块用哪个专家；-1 整块跳过 |
| `num_tokens_post_padded`（出） | 补齐后总槽位数，决定 GEMM grid 大小 |

## 值得注意的细节

- **实现三段式**：原子加计数 → 带 pad 前缀和 → scatter 写回；输出缓冲按上界 `numel + num_experts×(block_size−1)` 预分配
- **开销可忽略**：只处理 num_tokens×top_k 个 int32 下标，相对后续 GEMM 微不足道
- **官方 docstring 笔误**：main 分支 batched 示例的 `expert_ids` 写作 `[0,1,3,3,4,5,5]`，golden 测试的正确结果是 `[0,1,3,3,4,4]`

## 消费方式（[[Triton]] 伪代码）

```python
pid_m = tl.program_id(0)
e     = tl.load(expert_ids_ptr + pid_m)              # 整块只查一次专家号；-1 整块跳过
offs  = tl.load(sorted_token_ids_ptr + pid_m*BLOCK_M + arange(BLOCK_M))
mask  = offs < numel                                  # PAD 槽位被屏蔽
a_ptrs = a_ptr + (offs // top_k) * stride_am         # 槽位号 → 激活行号
b_ptrs = b_ptr + e * stride_be                       # 整块复用权重 W_e
```

## 关联页面

- 算子本体：[[moe_align_block_size]]
- 上层概念：[[MoE]]、[[Grouped GEMM]]
- 并行策略：[[专家并行 EP]]、[[数据并行 DP]]、[[CUDA Graph]]
- 实体：[[vLLM]]、[[DeepSeek]]、[[Triton]]
