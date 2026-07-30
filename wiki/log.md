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

## [2026-07-30] ingest | 代码路径文档定稿（问答融入章节）+ 新增概念页

用户要求将 raw 文档的 11 轮问答书面化融入章节，形成五章定稿（外部路径 / 框架层 / MK 内部 / 一图流 / 12 项决策表）。

**新建概念页 ×4**：
- [[EPLB]] — 专家并行负载均衡：冗余专家、逻辑/物理双层 ID、映射+负载记录融合 kernel、后台 rebalance
- [[All-to-All]] — 四种集合通信原语对比；EP dispatch/combine 的本质；all2all 后端生态
- [[Context Parallel]] — PCP/DCP 之分；MoE 层的 AgRsAll2All 临时方案
- [[序列并行 SP]] — 消除 TP 下 gate 冗余；与 naive 捷径的联动

**更新页面**：
- [[vLLM]]：MK 两种 impl、monolithic vs modular 本质、supports_internal_mk 语义、EPLB 支持
- [[moe_align_block_size-代码路径]]（来源摘要）：同步定稿结构与核心知识点
- [[专家并行 EP]]：EPLB 链接、All-to-All 链接
- [[数据并行 DP]]：PCP 改为 Prefill Context Parallel（勘误），链接 [[Context Parallel]]
- [[MoE]]：EPLB 扩展、SP 链接

**勘误**：PCP = Prefill Context Parallel（此前误写为 Pipeline）。

## [2026-07-30] ingest | naive 路径下游执行语义（增量）

用户追问 naive_block_assignment 分支的下游走向与返回值差异，对照 `fused_moe_kernel` 源码补全。

**更新页面**：
- [[moe_align_block_size]]：naive 分支扩充——返回值"两种方言"对照表、kernel constexpr 特化（块=对恒等映射）、grid 启动差异、收益/代价与阈值意义
- [[moe_align_block_size-代码路径]]（来源摘要）：同步 §3.3 增量
- raw 文档 §3.3：补入下游执行分析（书面化）

**关键知识点**：naive 模式 grid_m=numel（一块一对）；`offs_token=tl.where(offs==0, pid_m, num_valid)` 构造恒等映射砍掉 sorted_token_ids 间接表；阈值 tokens×top_k×4≤E 保证平均每专家≤1/4 对，对齐无收益。

## [2026-07-30] ingest | naive 路径详解版（含微型例子与 roofline 分析）

用户要求将 naive 路径的详细讲解（生动版）沉淀进 wiki。

**更新页面**：
- [[moe_align_block_size]]：naive 章节扩充——"对在下标里不在值里"的解读、2token/top2/E16 微型例子完整对照图、权重复用丧失的三层缓冲（L2/启发式/tile 浪费相同）、memory-bound vs compute-bound 的 roofline 分析
- raw 文档 §3.3：同步扩充四个小节（微型例子 / 对的本质 / 共享代码 / roofline 分析）

**关键知识点**：展平 topk_ids 的"值定位权重、下标定位 token（i//top_k）"；小 batch decode 是访存瓶颈，浪费的 tile 算力免费、省下的 kernel launch 才是收益；naive 仍是 grouped GEMM，只是每块有效行退化为 1。

## [2026-07-30] ingest | naive 适用范围勘误（≠ decode 默认路径）

**更新页面**：[[moe_align_block_size]]、raw 文档 §3.3

**关键知识点**：naive 判断依据是本 forward 的 token 数而非推理阶段——decode 单人/小流量（batch≤4）才触发；生产高并发 decode 与 prefill 走对齐模式；EP 永不 naive；投机解码加速阈值突破；分支逐 step 动态切换、结果等价。
