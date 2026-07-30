---
type: concept
title: EPLB（专家并行负载均衡）
created: 2026-07-30
updated: 2026-07-30
sources: ["raw/sources/moe_align_block_size 路径.md"]
tags: [moe, ep, 负载均衡, 框架层]
---

# EPLB（Expert Parallelism Load Balancer）

> 所属层次：[[purpose|框架层]]。[[专家并行 EP|EP]] 部署下的动态负载均衡机制。

## 要解决的问题

EP 下专家**静态**分配到卡，但路由是**动态**的——热点专家挤爆某张卡，冷门专家闲置，集群负载严重不均。

## 核心机制

- **冗余专家（redundant experts）**：物理专家数 = 逻辑专家数 + 冗余数。热点逻辑专家可以在**多张卡上摆副本**
- 由此引入两层 ID：
  - **逻辑 ID**：模型权重视角，router 输出的 `topk_ids`
  - **物理 ID**：部署视角，实际调度与 GEMM 使用

## vLLM 中的实现（v0.20.2）

路由产出逻辑 `topk_ids` 后，`_apply_eplb_mapping` 分支：

```python
if self.enable_eplb:
    return eplb_map_to_physical_and_record(...)   # 逻辑→物理 + 负载记录
return topk_ids                                    # 未开启：恒等
```

1. **翻译**：topk_ids 从逻辑 ID 映射到物理 ID（之后的 align/GEMM 全用物理视角）
2. **记录**：统计每个物理专家收到的 token 数（`expert_load_view`），作为重平衡决策输入
3. **实现**：融合 [[Triton]] kernel（`router/base_router.py` L111），一次 pass 完成映射+计数；`should_record_tensor` 控制本步是否记录（避免预热污染统计）
4. **后台重平衡**：`eplb/async_worker.py` 周期性根据负载统计重新摆放专家副本（`rebalance_execute.py`）

## 与下游算子的关系

- 映射发生在路由之后、[[moe_align_block_size]] 之前 → align kernel 看到的是**物理 ID**
- 与 `expert_map` 协同：expert_map 处理"全局 vs 本卡"，EPLB 映射处理"逻辑 vs 物理"

## 参见

- [[专家并行 EP]]、[[MoE]]、[[vLLM]]、[[DeepSeek]]
