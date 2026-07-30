# Log — 操作日志

> Append-only。每条以 `## [日期] 操作 | 标题` 开头。
> 解析示例：`grep "^## \[" wiki/log.md | tail -5`

## [2026-07-30] init | 初始化 LLM Wiki 结构

按照 [Karpathy llm-wiki 模式](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)（参考 nashsu/llm_wiki 的目录约定）初始化 vault：

- 创建三层结构：`raw/`（sources + assets）、`wiki/`（entities/concepts/sources/queries/synthesis/comparisons）、schema 层
- 编写 [[purpose]]、[[schema]]、[[AGENTS]]
- 创建 [[index]]、[[overview]]、本日志
- 配置 Obsidian 附件目录为 `raw/assets`
- 配置 git 远程：GitHub `baiiiij/LLM_obsidian_wiki`

## [2026-07-30] update | 明确 purpose：LLM 算子开发知识库

根据用户输入重写 [[purpose]]，将研究方向规整为五层结构：

1. 模型层：热门大模型结构（Transformer 变体、GQA/MLA/MoE 等）
2. 框架层：vLLM、SGLang 等推理框架的思想与实现
3. 算子层：Triton / TileLang 等 DSL、FlagGems 等算子库、传统 C/C++ 算子
4. 硬件层：各厂商硬件、编程模型（CUDA/HIP/CANN）、RISC-V
5. 底层抽象层：SIMT / SIMD / SPMD 执行模型与指令集

主线：从模型结构到硬件指令集，逐层向下打通。

## [2026-07-30] ingest | vLLM moe_align_block_size 交互式详解

首个来源。HTML 交互式讲解页，覆盖 `moe_align_block_size` 三场景（基础 / EP / DP batched）。

**新建页面**：
- 来源摘要：[[moe_align_block_size-交互式详解]]
- 概念 ×6：[[MoE]]、[[moe_align_block_size]]、[[Grouped GEMM]]、[[专家并行 EP]]、[[数据并行 DP]]、[[CUDA Graph]]
- 实体 ×3：[[vLLM]]、[[DeepSeek]]、[[Triton]]

**关键知识点**：布局整理算子的作用与三段式实现；EP 的 expert_map 两种模式；DP 定长缓冲 ↔ CUDA Graph 定形约束；官方 docstring 笔误（expert_ids 正确结果为 [0,1,3,3,4,4]）。

**值得深挖**：SGLang RadixAttention；FlashAttention；SIMT/SIMD/SPMD 辨析（底层抽象层尚空白）。

## [2026-07-30] ingest | vLLM moe_align_block_size 代码路径（含 5 轮问答）

第二个来源。用户自建文档，由 Claudian 对照 v0.20.2 源码补充整理，含 5 轮问答。

**新建页面**：
- 来源摘要：[[moe_align_block_size-代码路径]]
- 概念 ×2：[[Shared Experts]]、[[Latent MoE]]

**更新页面**（大幅扩充）：
- [[vLLM]]：补充 v0.20.2 runner 架构、TP/SP/EP/DP 通信节奏、naive 捷径与序列并行联动
- [[MoE]]：补充 Shared/Latent/序列并行扩展
- [[moe_align_block_size]]：补充完整调用链位置、naive_block_assignment 分支、与 SP 联动
- [[专家并行 EP]]：补充 naive dispatch 前置通信
- [[数据并行 DP]]：补充 PCP 关联

**关键知识点**：
- v0.20.2 架构重构：FusedMoE → MoERunner → 模块化 kernel（oracle 选后端）
- naive 捷径：`num_tokens × top_k × 4 ≤ E` 时跳过对齐；Qwen3-30B-A3B 仅 ≤4 token 触发
- 序列并行联动：TP=8 时 32 token → 每卡 4 个 → 恰好触发捷径
- ReplicatedLinear 的必要性：gate 必须每卡完全复制，否则路由分叉
- shared_experts / latent MoE 的模型架构来源与 vLLM 支持方式
- `_maybe_dispatch` / `_maybe_combine` 的通信节奏（DP/EP All-to-All + PCP all-gather/reduce-scatter）
