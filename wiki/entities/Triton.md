---
type: entity
title: Triton
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size交互讲解.html"]
tags: [算子语言, dsl, 算子层]
---

# Triton

> 所属层次：[[purpose|算子层]]。Python 风格的 GPU 算子开发 DSL（[[purpose]] 的核心关注语言之一）。

## 与本知识库相关的要点

[[vLLM]] 的 MoE GEMM kernel（`fused_moe_kernel`）用 Triton 编写，消费 [[moe_align_block_size]] 产出的布局：

```python
pid_m = tl.program_id(0)
e     = tl.load(expert_ids_ptr + pid_m)              # 整块只查一次专家号
offs  = tl.load(sorted_token_ids_ptr + pid_m*BLOCK_M + arange(BLOCK_M))
mask  = offs < numel                                  # PAD 槽位屏蔽
a_ptrs = a_ptr + (offs // top_k) * stride_am
b_ptrs = b_ptr + e * stride_be                       # 整块复用权重 W_e
```

体现的 Triton 编程特征：

- 以 **block（BLOCK_M）为粒度**编程，而非逐线程——SPMD 风格（见 [[purpose]] 底层抽象层）
- 显式的指针算术与 mask 机制
- `tl.program_id` 映射到 GPU grid

## 待补充方向

- Triton 编译模型、与 CUDA C++ 的取舍、TileLang 对比（待相关来源 ingest）

## 参见

- [[Grouped GEMM]]、[[moe_align_block_size]]、[[vLLM]]
