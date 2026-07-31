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

## [2026-07-31] query | torch.flatten 调用链路定位（view vs copy）

用户问：IDE 点进 `Tensor.flatten` 只见 `torch/_C/__init__.pyi` 类型存根，如何定位源码调用链、dispatch 机制、复制与只改 metadata 的判断条件。

**新建页面**：
- 问答沉淀：[[pytorch-flatten-调用链路定位]]
- 概念：[[PyTorch-ATen-Dispatcher]]

**关键知识点**：.pyi 由 `tools/pyi/gen_pyi.py` 生成，是死路；定位原点是 `aten/src/ATen/native/native_functions.yaml`；flatten 是 `structured_delegate: reshape`，非独立 kernel；reshape（CompositeImplicitAutograd，TensorShape.cpp）内 `computeStride` 贪心匹配连续段——成功走 `_reshape_alias/as_strided`（view，0 拷贝），失败走 `clone/contiguous`（拷贝）；实证工具：`TORCH_SHOW_DISPATCH_TRACE=1`、profiler 看 `aten::clone/copy_`、`._base is t`。

**待验证**：具体行号与 mkldnn/nested tensor 的 dispatch 分支以用户本地编译版本为准（未对照其实际源码树）。

## [2026-07-31] update | flatten 链路对照本地源码实测定稿

用户反馈初版调用链解释不清晰，且要求标注 build 生成 vs 源码自带。找到本机源码树 `/home/admin/code/LLM/source_code/pytorch`（version.txt=2.14.0a0，用户称 2.12，结构一致），逐文件核实后重写。

**勘误（重要）**：2.12+ 的 `flatten.using_ints` 在 yaml 中**没有 `structured_delegate: reshape`**（旧版本才有），而是无 dispatch 段 = 默认 CompositeImplicitAutograd，直接实现于 TensorShape.cpp:4121，且**直接 C++ 调用** `native::reshape_symint`（不再走 dispatcher）。已同步修正 [[PyTorch-ATen-Dispatcher]]。

**实测要点**：
- flatten 本体 TensorShape.cpp:4121（0 维→reshape({1})；start==end→return self；拼 shape→reshape_symint）
- reshape_symint :1976 三分支：contiguous 快路径 view_symint；computeStride 成功→_reshape_alias→alias_with_sizes_and_strides(:1945，共享 Storage + set_sizes_and_strides)；失败→clone(Contiguous)+_unsafe_view
- computeStride 模板实现 TensorUtils.cpp:325（双指针连续块匹配，填不满返回 nullopt）
- build 产物落点：写回源码树（torch/_C/__init__.pyi、torch/csrc/autograd/generated/）vs build 目录（build/aten/src/ATen/，cmake/Codegen.cmake:213 --install_dir）

## [2026-07-31] query | 不凭先验经验定位调用链的方法论

用户追问：直接给文件/函数名是先验经验；预期流程是"Python 函数体里有分支、逐层点进去"，PyTorch 为什么不是？如何自己一步步推？

**新建页面**：[[aten算子调用链定位方法论]]

**实测实验**（本机 torch 2.6.0 + 2.14 源码树）：
- `inspect.getsource(Tensor.flatten)` 报 TypeError method_descriptor → C 扩展方法的判别信号；对照 `Tensor.unflatten` 能拿到源码（torch/_tensor.py 有 Python 包装）→ "两种流程都有，逐算子判别"
- docstring 自带 schema `flatten(start_dim=0, end_dim=-1) -> Tensor`（从 yaml 生成时写入）
- profiler 实测：连续张量 flatten → aten::flatten + aten::view（_base is t）；transpose 后 → aten::flatten + clone/copy_/_unsafe_view（_base is None）；trace 中无 aten::reshape，证实 composite 直接调用
- yaml 4 个 flatten overload 按参数类型匹配：int → using_ints；Dimname 三个属 named tensor 特性

**核心原则**：能观测的不猜测（profiler/trace 拿 ground truth），能匹配的不背诵（参数类型对 overload），yaml 是唯一地图。

## [2026-07-31] update | 方法论页重写：10 个追问逐条展开 + 版本基准切到 2.12

用户提出 10 个追问（profiler 原理、activities 含义、为何能看到 dispatch、dispatch vs 直接调用、yaml 名字是否总一致、TORCH_SHOW_DISPATCH_TRACE、rg 是什么、胶水生成细节、版本用 2.12）。整体重写 [[aten算子调用链定位方法论]]。

**2.12 在线核对**（GitHub release/2.12）：flatten.using_ints 在 yaml:2702（无 dispatch 段）；flatten impl TensorShape.cpp:4178；reshape_symint :2058；alias_with_sizes_and_strides :1992；_reshape_alias :2168；computeStride TensorUtils.cpp:327。[[pytorch-flatten-调用链路定位]] 行号已全部切换为 2.12。

**新实测/新发现**：
- **TORCH_SHOW_DISPATCH_TRACE 在官方 release 版被编译掉**：打印代码包在 `#if defined(HAS_TORCH_SHOW_DISPATCH_TRACE) || !defined(NDEBUG)`（Dispatcher.h），pip 版实测无输出；仅 debug 构建可用
- profiler 可见性原理：RecordFunction 钩子埋在 Dispatcher.h（callWithDispatchKeySlowPath），故 native:: 直接调用对 profiler 不可见——profiler 事件流 ≈ 过 dispatcher 的算子清单
- 无 GPU 机器请求 CUDA activity：仅 warning "CUDA is not available, disabling CUDA profiling"，不报错
- 名字不一致实例（实测）：a+a→aten::add；torch.concat→aten::concat→aten::cat；a@a→aten::matmul→aten::mm
- 胶水链四文件闭环：templates/python_variable_methods.cpp（${py_methods} 占位）→ gen_python_functions.py → generated/python_variable_methods.cpp → python_variable.cpp:3887/3913 挂载到 torch._C.TensorBase

## [2026-07-31] update | profiler 表格读法 + GPU 实验正确开法（用户实测反馈）

用户在 GPU 机（vllm 环境）实测：CPU-only 输出正常（aten::flatten→aten::view）；只开 CUDA activity 后 aten 算子全消失，只剩 cudaDeviceSynchronize + Activity Buffer Request。

**更新页面**：[[aten算子调用链定位方法论]] 新增"profiler 输出表格怎么读"小节。

**关键知识点**：Name 列即 dispatch 算子名，事件树状嵌套；Self CPU=自身耗时（不含子调用），CPU total=含子调用（flatten 141.077us = self 64.712 + view 76.365）；GPU 实验必须 CPU+CUDA 双开，Self CUDA 列全 0 = view 零拷贝眼见为实，Memcpy DtoD = 真拷贝；Activity Buffer Request/USDT 日志/cycle warning 均为 profiler 自身噪音。

## [2026-07-31] update | "Tensor flatten" 搜索词的来源（去先验化）

用户追问：`rg "Tensor flatten"` 里的 `Tensor` 前缀是否也是先验经验。

**更新页面**：[[aten算子调用链定位方法论]] 第 5 步新增小节。

**关键知识点**：搜索词 = yaml 返回类型（`-> Tensor(a)`）+ 算子名 + `(`，非背诵；实测 `rg "flatten"` 在 TensorShape.cpp 有 38 命中 → 加返回类型收敛到 5；参数类型（int64_t vs DimnameList）可反向确认 overload；零先验终极路线：build 后 rg op 名于 RegisterCompositeImplicitAutograd*.cpp，生成的注册表直接给出 op→实现函数绑定；搜索是迭代收窄（命中多加特征、命中少减特征）。

## [2026-07-31] update | moe_align_block_size 概念页重写：从零讲解的全链路文档

用户反馈：前置/后置计算要说明清楚，避免过段时间忘上下文再重新理解。按"假设读者从没见过大模型计算流程"的要求整体重写 [[moe_align_block_size]]。

**新文档结构**（九章，从零背景到两条下游路径）：
- §1 背景：token/hidden states、Transformer 层 FFN = 两次 GEMM、GEMM 的**计算操作与访存操作**、权重复用是 FFN 快慢核心指标
- §2 MoE 正常算法：为什么用专家（稀疏激活）、四步流程（路由打分→top-k→逐对 FFN→加权求和）、计算/访存清单、直白优缺点
- §3 朴素实现四宗罪：动态形状（CUDA Graph 无法捕获）、权重读多算少（访存/计算比）、block lane 浪费、数据散乱 gather——"散 + 动 vs 规整 + 静态"的核心矛盾
- §4 理想解法 Grouped GEMM：前置计算（布局整理：按专家分组+补齐 BLOCK_M）→ 后置计算（每 block 查一次专家号、加载一次权重、算 BLOCK_M 行）——"跳着取 token"即 sorted_token_ids 间接寻址
- §5 vLLM 整体优化：流水线（select_experts → _prepare_expert_assignment → fused_moe_kernel → finalize）、融合点清单（路由合并、w13+SiLU+w2 一次 launch）、条件分支表、调用链位置
- §6 本体三段式：数人数（原子计数）→ 算位置（带 pad 前缀和）→ 写回（scatter），每段直白解释
- §7 后置计算：fused_moe_kernel 消费伪代码逐行注释 + 访存特征
- §8 naive/native 捷径：触发条件直白含义（平均每专家 ≤1/4 对 → 对齐无收益）、**两条路宏观对照表**（加载一次专家算 BLOCK_M 行跳着取 token vs 一块一对）、下游两种方言、roofline 收益代价、微型例子、适用范围辨析
- §9 周边链接

**同步更新**：[[index]] 该页摘要；页面 frontmatter updated → 2026-07-31。

**依据**：与本地 vllm 源码树（v0.22.1）对照了 `moe_align_block_size.py` 包装层与 `csrc/moe/moe_align_sum_kernels.cu`（atomicAdd 计数 / BlockScan 前缀和 / scatter 写回），三段式描述与源码一致；正文仍以 v0.20.2 调用链为准。

## [2026-07-31] update | 解耦背景：LLM 前向计算独立成页

用户要求把 [[moe_align_block_size]] 中的 LLM 通用背景（原 §1）解耦为独立文档，算子页只留结论 + 链接。

**新建页面**：
- [[LLM 前向计算]] — 从零背景页：token/hidden states、Transformer 层（attention + FFN = 两次 GEMM + 激活）、GEMM 的计算与访存、权重复用核心指标；末尾"向后看"一节说明这些事实如何决定 MoE / 布局优化 / memory-bound 等下层形态

**更新页面**：
- [[moe_align_block_size]]：§1 收缩为"继承来的三个结论"（FFN 最重 / roofline / 权重复用是共同判据），指向新页；阅读地图、§9 链接同步
- [[MoE]]：页首加前置背景链接
- [[index]]、[[overview]]：新页入账（概念 13→14）

**结构意图**：背景页作为各算子页（未来的 FlashAttention 等）共同的可复用前置，算子页聚焦本体。

## [2026-07-31] fix | dispatch 示意图 at::clone(t) 的"调用方"注释歧义

用户指出：示意图里 `at::clone(t) ← 你写的/生成的包装代码`，但"我没有写"。

**修正**：[[aten算子调用链定位方法论]] 示意图注释改为"任何 C++ 代码（你写的扩展 / 生成的 TensorBody 包装 / PyTorch 自己的 kernel）"并加注：flatten 链路中 clone 的调用方是 PyTorch kernel 作者（`reshape_symint` 内的 `self.clone(...)`，2.12 TensorShape.cpp:2104）；判别"谁调了它"靠 profiler 事件树父子关系，不是靠调用形式。

## [2026-07-31] update | 双层模型澄清：aten::reshape 从 Python 进入时照样 dispatch

用户三连问：①reshape 没有 dispatch 的话，从 Python 到 C++ 的链路是什么、怎么自己分析？②SOP 找到的 at::native::xxx 本身没有 dispatch，这是合理链路吗？③是不是解释时省略了一步？要完整分析链路。

**核心澄清（双层模型）**：算子（aten::reshape，dispatcher 表里的项）与 kernel（at::native::reshape_symint，查表终点）是两层。dispatch 发生在到达 kernel 之前；之前"reshape 不过 dispatcher"的表述仅指 flatten→reshape 的 kernel 内部直连。

**实测对照**（pip torch 2.12 CPU）：`t.reshape(6,4)` → profiler 有 `aten::reshape → aten::view`（dispatch 了！）；`t.flatten(0,1)` → `aten::flatten → aten::view`（无 aten::reshape，内部直连）。一个实验同时证明两层。

**SOP 省略的一环**：注册表项 `m.impl("aten::reshape", TORCH_FN(at::native::reshape_symint))`（build 后 `rg "aten::reshape" build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp`）是"算子名→kernel"绑定的直接证据；SOP 靠 yaml 规则推断这步，看它则每环有实证。

**更新页面**：[[aten算子调用链定位方法论]] 第二节新增"双层模型""注册表项""完整分析链路（每层附验证手段）"三小节；修正两处易误读表述；reshape 在 2.12 yaml:5141（Composite）、view yaml:8422（有 dispatch 段）、_reshape_alias yaml:5157（有 dispatch 段）。

## [2026-07-31] update | 为什么会有 dispatcher：设计动机补全

用户反馈"回答片面"：只讲了是什么（clone 过表/reshape 不过），没讲为什么——为什么 clone 会 dispatch、dispatch 与内部调用的本质区别、为什么有 dispatch 这个概念。

**更新页面**：[[aten算子调用链定位方法论]] 第二节扩为四小节。

**关键知识点**：
- 机械层面：dispatch = "你调的函数体里有没有查表代码"。`at::_ops::clone::call` 的整个函数体就是 `Dispatcher::singleton().findSchemaOrThrow` + `op.call`（torchgen gen.py:667 生成器原文为证）；`native::reshape_symint` 是手写普通函数，编译期定跳转。kernel 作者对每个调用点二选一。
- 设计动机：一算子名→N 实现，运行时才能选。if/switch 方案死于规模与插件后端（编译期不存在）；虚函数只有一根多态轴，PyTorch 有多根正交轴（设备/autograd/tracing/functionalization）同时生效；注册表方案加后端零调用点改动。
- 杀手级收益：横切机制（profiler/compile）一次插桩覆盖全算子（不过表则全看不见）；Autograd key 高优先级自动套 backward。
- 判断准则：实现随 tensor 运行时属性而变 → dispatch；不变 → 直连（过表有开销：算 key/哈希查找/间接跳转）。
- 顺带勘误：reshape 内 clone 调用点在 2.12 TensorShape.cpp:2104（此前误写 2068，已全局修正）。
