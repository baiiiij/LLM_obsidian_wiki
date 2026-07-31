---
type: concept
title: moe_align_block_size（MoE 布局整理算子）
created: 2026-07-30
updated: 2026-07-31
sources: ["raw/sources/moe_align_block_size交互讲解.html", "raw/sources/moe_align_block_size 路径.md"]
tags: [vllm, moe, kernel, 算子层]
---

# moe_align_block_size（MoE 布局整理算子）

> 所属层次：[[purpose|算子层]]。[[vLLM]] MoE 推理路径上的**预处理（布局整理）kernel**。
>
> 一句话：它把路由器输出的散乱 `topk_ids` 整理成「按专家连续存放、每个专家的槽位数补齐到 block_size 整数倍」的布局，让后面的 [[Grouped GEMM]] 每个线程块只查一次专家号、加载一次权重、算一整块 token。

## 阅读地图：本文在讲什么

本文的目标是让**从没看过大模型计算流程的人**也能读懂这个 kernel。所需背景已抽到 [[LLM 前向计算]]，只需带走 §1 的三个结论即可继续。按这个顺序读：

1. **§1 背景**：LLM 前向计算留给本文的三个结论（细节已独立成页：[[LLM 前向计算]]）
2. **§2 MoE**：MoE（专家）是什么；正常算法逻辑里有哪些**计算操作和访存操作**；它的直白优缺点
3. **§3 朴素实现的缺点**：为什么"正常算法逻辑"直接搬上 GPU 很慢
4. **§4 理想解法**：Grouped GEMM —— "先整理布局（前置计算），再按块算（后置计算）"
5. **§5 vLLM 的整体优化**：把哪些步骤合并了、什么条件下走哪条算法、`moe_align_block_size` 在整条流程中的哪一环节
6. **§6 本体**：`moe_align_block_size` 的三段式算法（前置计算细节）
7. **§7 后置计算**：`fused_moe_kernel` 怎么消费整理好的布局
8. **§8 naive 捷径**：不走对齐的另一条路（naive / 常被记成 "native"），两条路下游流程的差别

---

## 一、背景：从 LLM 前向计算继承来的三个结论

> 完整背景已独立成页：[[LLM 前向计算]]（token / hidden states / Transformer 层 = attention + FFN / GEMM 的计算与访存）。本文只带走与本算子直接相关的三个结论：

1. **FFN 是每层最重的一步计算**：两次 GEMM（`W_up`、`W_down`）+ 中间一次激活
2. **GEMM 快慢取决于"计算"与"访存"谁先触顶**（roofline）；FFN 权重巨大，**读权重的带宽往往比算力更值钱**
3. 因此 **"一份权重被多少个 token 复用"是 FFN 快慢的核心指标**——这是 §3 朴素实现之慢、§4 布局整理之快、§8 naive 捷径敢于浪费算力的共同判据

---

## 二、MoE：把 FFN 换成"路由器 + 一群专家"

### 2.1 为什么要用专家

模型的"知识"存在权重里。想更聪明通常要加大模型——但每层 FFN 变大，**每个 token 都要过全部计算**，成本跟着翻倍。

MoE（Mixture of Experts，混合专家）的思路：**准备 E 个专家 FFN（参数量 ×E），但每个 token 只让其中 top-k 个专家算**。

- 参数量（容量）暴涨 → 记得住更多知识
- 每个 token 的计算量只涨 k 倍（k 通常 1~8），而不是 E 倍（E 通常 64~256）
- 这就是"稀疏激活"：模型很大，但每次只用一小部分

### 2.2 正常算法逻辑（一步步直白描述）

一个 MoE 层接收 `hidden_states`（`[T, D]`），分四步：

```
① 路由打分：   scores = hidden_states @ W_gate               [T, E]
② 选专家：     topk_ids [T, k]、topk_weights [T, k]
               （对每个 token：取 scores 最大的 k 个专家 + 归一化权重）
③ 专家计算：   对每个被选中的 (token t, 专家 e) 对：
               y = FFN_e(x_t) = SiLU(x_t @ W_up_e) @ W_down_e   [D]
④ 加权求和：   output_t = Σ_j topk_weights[t][j] × y_j         [D]
```

- 第①②步叫**路由（routing）**：决定"谁去哪"
- 第③步是**重头戏**：总共有 T×k 个 (token, 专家) 对，每个对都是一次小 FFN（两次 GEMM）
- 第④步把同一个 token 的 k 个结果按权重合并

### 2.3 里面的计算操作与访存操作

| 步骤 | 计算操作 | 访存操作 |
|---|---|---|
| ① 路由 | gate GEMM：T×E×D 次乘加 | 读 X（T×D）、读 W_gate（D×E） |
| ② top-k | 每行 softmax + 取最大 k 个（逐元素/比较） | 读 scores |
| ③ 专家 FFN | 每对两次 GEMM：T×k × (D×4D + 4D×D) 次乘加 | 读专家权重 W_up_e / W_down_e |
| ④ 加权求和 | 逐元素乘加（k 个 [D] 向量加权累加） | 读 k 个输出向量 |

### 2.4 直白优缺点

**优点（算法层面）**：
- 同样的算力预算，模型容量大得多——把 FFN 的"知识"分而治之，每个专家专精一类输入
- 是 LLM 把参数量做大、而单 token 算力不涨的关键技术（[[DeepSeek]] 等都在用）

**缺点（对硬件/系统不友好）**——这正是后面所有优化的动机：
- 路由是**动态的**：每步每个专家分到多少 token 完全随机，形状步步在变
- token 找专家的顺序是**散乱的**：同一专家的 token 在内存里东一个西一个
- 专家权重巨大，如果一次只算一两个 token，"读权重"的带宽就浪费了

---

## 三、朴素实现为什么慢："散"与"动"

### 3.1 最朴素的写法

不玩任何技巧，直接照 §2.2 的逻辑写：

```python
for e in range(num_experts):            # 对每个专家
    mask = (topk_ids == e)              # 找出选了我的 token
    x_e = hidden_states[mask]           # gather：把这些 token 收集起来
    y_e = ffn_e(x_e)                    # 一次 GEMM（可能只有 1 行）
    output[mask] += topk_weights[mask] * y_e
```

这个写法在 CPU/小规模上完全正确，在 GPU 上却慢得离谱，原因有四：

### 3.2 问题 1：形状动态变化（"动"）

`x_e` 的行数（= 选专家 e 的 token 数）**每步都在变**：上一步 E7 分到 5 个 token，下一步可能 500 个，再下一步 0 个。

- GPU kernel 编译时不知道形状，没法按固定形状深度优化
- [[CUDA Graph]] 要求所有张量形状固定，这种动态形状的流程根本没法捕获

### 3.3 问题 2：权重"读多算少"（访存/计算比爆炸）

GEMM 快慢取决于**权重被复用多少次**。理想情况：读一次 W_e，尽量多算几行 token。

朴素写法下，专家 e 若只分到 1 个 token：读 8D² 个权重元素只做了 1 行计算——权重利用率只有 ~1/BLOCK_M（BLOCK_M 是 GPU 一次处理的合理行数，16~64）。E 个专家 × 48 层 × 每步，带宽全耗在读权重上。

### 3.4 问题 3：block 内 lane 浪费

GPU 以线程块（block）为单位工作，一个 block 固定处理 BLOCK_M 行。专家 e 只有 1~5 个 token 时，一个 block 里 BLOCK_M−1 条 lane 空转。

### 3.5 问题 4：数据散乱（"散"）

`hidden_states[mask]` 是 gather 操作：按 mask 把散在各处的行拷到一起，多一次访存、多一次同步。

**核心矛盾一句话**：路由的输出是"散乱 + 动态"的，而 GPU 喜欢"规整 + 静态"。

---

## 四、理想解法：Grouped GEMM —— "先整理，再按块算"

要又快又正确，思路是把问题拆成两步：**前置计算负责把数据摆好，后置计算负责高效地算**。

### 4.1 前置计算：布局整理（把"散乱"变成"规整"）

在 GEMM 之前，花一点点代价把路由结果整理好：

1. **按专家分组**：把 T×k 个 (token, 专家) 对按专家号排好——同一专家的对连续存放
2. **补齐**：每个专家的对数量向上凑成 BLOCK_M 的整数倍（不够的用"空气"占位，计算时 mask 掉）

这就是 `moe_align_block_size` 干的事（§6 详述）。整理完产出三个东西：

- `sorted_token_ids`：整理后的槽位表——"第 i 个槽位放的是原来第几个对"（相当于一张跳表）
- `expert_ids`：每个 block 用哪个专家
- `num_tokens_post_padded`：总共多少个有效槽位（决定开几个 block）

### 4.2 后置计算：按块算（每个 block 加载一次权重，算一整块）

GEMM kernel 拿到整理好的布局，开 `grid = num_tokens_post_padded / BLOCK_M` 个 block，每个 block：

```
block i：
  1. 查一次 expert_ids[i]              → 知道本块用专家 e（不用再逐个 token 判断）
  2. 加载专家 e 的权重 W_up_e / W_down_e → 一整块连续读
  3. 取 BLOCK_M 个 token 行（按 sorted_token_ids 跳着取，见下）
  4. 一次性算 BLOCK_M 行 × W_e 的 GEMM
  5. 补位行（"空气"）被 mask 掉，结果丢弃
```

**"跳着取 token"是什么意思**：`sorted_token_ids` 是一张**间接寻址表**。block 内第 j 行要算的 token 行号是 `sorted_token_ids[block×BLOCK_M + j] // top_k`——这些 token 原本在 X 里东一个西一个，所以每次都要"查表跳过去"取。行与行之间不连续，但每一行内部是一整段连续 D 个浮点数，GPU 的 gather 加载可以处理。

### 4.3 收益（对比朴素实现）

| | 朴素实现 | 布局整理 + Grouped GEMM |
|---|---|---|
| 每专家 token 数 | 动态，kernel 无法定形 | 补齐后固定（对 CUDA Graph 友好） |
| 权重复用 | 每个对读一次权重（利用率 ~1/BLOCK_M） | 每个 block 读一次，复用 BLOCK_M 行 |
| kernel launch | 每专家至少 1~2 次（共 2E 次量级） | 一次 launch，所有专家一起算 |
| token 取用 | 先 gather 拷贝再算 | 查表 gather 直接加载（省拷贝） |

**核心等价性**：两种方式算的是同一个东西——每个 (token, 专家) 对做一次 FFN，再按权重合并。只是数据摆放和调度方式不同，结果完全一致。

---

## 五、vLLM 整体是怎么优化这条流程的

### 5.1 一条流水线

vLLM（v0.20.2，CUDA 默认后端）把整个 MoE 层做成一条流水线：

```
① select_experts（路由）        gate GEMM + softmax + top-k     → topk_ids / topk_weights
② _prepare_expert_assignment    布局整理：moe_align_block_size，或 naive 捷径（§8）
③ fused_moe_kernel（后置计算）  w13 GEMM + SiLU + w2 GEMM（一个 kernel 内完成）
④ finalize（归并）              按 topk_weights 加权求和（+ 并行策略的通信）
```

### 5.2 合并了哪些步骤

- **路由合并**：`select_experts` 把 gate GEMM + softmax + top-k 合成一个算子（[[Triton]] 实现）
- **两段 GEMM 融合**：朴素做法是每个专家分开 launch；vLLM 把 **w13 GEMM（第一段权重，up/gate 融合）+ SiLU 激活 + w2 GEMM（第二段 down）全部融进一次 kernel launch**（`fused_moe_kernel`），这就是 "fused" 名字的由来
- **布局整理独立成前置 kernel**：用"廉价的整理"（只处理 int32 下标，不做浮点）换"GEMM 的高效"（定形 + 权重复用）
- **归并融合**：finalize 里做加权求和，可与 EP/DP 的 combine 通信重叠

### 5.3 条件分支：什么条件下走什么算法

| 条件 | 走哪条路 |
|---|---|
| 默认（CUDA、无 EP、token 足够多） | `moe_align_block_size` 对齐 + grouped GEMM |
| 无 EP 且 `tokens × top_k × 4 ≤ E`（极小 batch） | **naive 捷径**：跳过对齐 kernel（§8） |
| EP 开启 | 传入 `expert_map`：非本卡专家的 block 标 -1 跳过（或过滤模式直接不排） |
| DP+EP（[[DeepSeek]] 式） | `batched_moe_align_block_size`：token 已被通信层按专家分好段，只补齐不排序 |
| 序列并行 SP | router 前把 token 切小 → 可能把小 chunk 推入 naive 捷径的条件区间 |

### 5.4 moe_align_block_size 在这条流程中的哪一环节

```
Qwen3MoeSparseMoeBlock.forward
└─ FusedMoE.forward → MoERunner.forward
   └─ _forward_impl
      └─ _apply_quant_method
         ├─ router.select_experts                  ① 路由：topk_ids / topk_weights
         └─ TritonExperts.apply (fused_moe.py:1985)
            └─ _prepare_expert_assignment (L1554)  ← 调用点
               ├─ naive 捷径 → 跳过本 kernel
               └─ moe_align_block_size (moe_align_block_size.py:11)   ② 布局整理（前置计算）
                  └─ ops.moe_align_block_size (CUDA C++ kernel)
               └─ invoke_fused_moe_triton_kernel   ③ 后置计算：grouped GEMM（§7）
```

即：`moe_align_block_size` 是**算 GEMM 之前的"排版工人"**——它不做任何矩阵乘，只重排下标，让下游 GEMM 一次 launch 把活干完。

---

## 六、moe_align_block_size 本体：前置计算的三段式

### 6.1 输入与输出

**输入**：
- `topk_ids`：`[num_tokens, top_k]`，路由器的输出——每个 token 选中了哪些专家
- `block_size`：GEMM 的块长（BLOCK_M，通常 16/32/64）
- `num_experts`：专家总数（EP 时必须传**全局**数）
- `expert_map`：EP 时"全局专家号 → 本卡局部号"映射（§6.3）

**输出**（三个）：

| 输出 | 直白解释 |
|---|---|
| `sorted_token_ids` | 对齐后的槽位表：槽位 i 里存的是"原来第几个 (token, 专家) 对"的**下标**。按专家分段排好、每段补齐到 block_size 的倍数；补位槽填 numel（越界值，下游见它就 mask） |
| `expert_ids` | 第 i 个 block 用哪个专家（按块记，不是按 token 记）；-1 表示整块跳过 |
| `num_tokens_post_padded` | 补齐后的总槽位数 = GEMM 的 grid 大小（开多少个 block） |

### 6.2 三段式算法（直白描述）

这个 kernel 只用 int32 下标做算术，不碰任何浮点——所以它的开销相对后面的 GEMM 可忽略。分三段：

**第一段：数人数（原子加计数）**
遍历 T×k 个对，每遇到一个"选专家 e"的对，就给 e 的计数器 +1（多线程同时加，用原子操作）。结束后 `count[e]` = 专家 e 这步分到几个 token。

**第二段：算位置（带 pad 前缀和）**
按专家顺序做前缀和：`start[e] = Σ_{e'<e} round_up(count[e'], block_size)`。
即"每个专家段从输出缓冲的第几个槽位开始"，且每段起点天然对齐到 block_size 的整数倍。顺带得到总槽位数 `num_tokens_post_padded`。

**第三段：写回（scatter）**
再遍历一遍 T×k 个对：第 i 个对（属于专家 e）写到 `sorted_token_ids[start[e] + 该专家段内序号] = i`；同时给每个 block 写专家号 `expert_ids[block] = e`，段内的补位槽填越界值 numel。

一句话总结：**先统计每个专家有几个 token，再算好每段的起始位置，最后把每个对"扔"进自己的槽位**。顺序被打乱没关系——反正下游是查表跳着取。

### 6.3 变体

- **EP（专家并行）**：专家切到多张卡。默认模式先对**全部全局专家**做三段，返回前经 `expert_map` 把非本卡专家的块标成 -1，GEMM 见 -1 整块跳过；过滤模式（`ignore_invalid_experts=True`）第一段计数时就忽略非本卡专家，缓冲更小、没有 -1。→ 详见 [[专家并行 EP]]
- **DP+EP（batched 变体）**：token 已被 All-to-All 按专家分段送上门，`batched_moe_align_block_size` 不需要排序，只做"补齐打包"。→ 详见 [[数据并行 DP]]
- 已知问题：官方 main 分支 docstring 里 batched 示例的 `expert_ids` 有笔误（写作 `[0,1,3,3,4,5,5]`），golden 测试的正确结果是 `[0,1,3,3,4,4]`

---

## 七、后置计算：fused_moe_kernel 怎么消费布局（对齐模式）

### 7.1 消费伪代码（[[Triton]]，逐行注释）

```python
pid_m = tl.program_id(0)                          # 本 block 的编号
e     = tl.load(expert_ids_ptr + pid_m)           # ① 只查一次专家号；-1 则整块跳过
offs  = tl.load(sorted_token_ids_ptr + pid_m*BLOCK_M
                + tl.arange(0, BLOCK_M))          # ② 本块 BLOCK_M 个槽位（查跳表）
mask  = offs < numel                              # ③ 补位槽（值为 numel）被屏蔽
row   = offs // top_k                             # ④ 槽位 → token 行号（跳着取）
a_ptrs = a_ptr + row * stride_am                  #    A 矩阵：gather 加载散乱行
b_ptrs = b_ptr + e * stride_be                    # ⑤ B 矩阵：整块连续加载专家权重 W_e
# ... 一次 GEMM 算 BLOCK_M 行 × W_e，mask 掉无效行
```

### 7.2 计算与访存特征（对齐模式）

- **访存**：B（专家权重）**每个 block 读一次**，连续读、复用 BLOCK_M 行；A（token 行）按跳表 gather——行间散、行内连续
- **计算**：一个 kernel 内完成 w13 GEMM → SiLU → w2 GEMM，E 个专家**共享一次 launch**，而不是 E 次
- **省了什么**：形状固定（CUDA Graph 可捕获）、权重复用高、无逐专家的 kernel 启动开销

---

## 八、naive 捷径：不走对齐的另一条路（naive / "native"）

> 注：代码里这个分支叫 `naive_block_assignment`（naive = 朴素），口语/笔记里常被记成 "native"。两者是同一个东西。

### 8.1 触发条件

`_prepare_expert_assignment`（`fused_moe.py` L1554）在调用 `moe_align_block_size` **之前**先判断：

```python
naive_block_assignment = (
    expert_map is None                     # ① 没开 EP
    and num_tokens * top_k * 4 <= global_num_experts   # ② 对总数极少
    and not (wna16 量化且 block_shape[1] > 0)          # ③ 无形状约束
)
```

**条件②的直白含义**：T×k 个 (token, 专家) 对，平均每个专家分不到 1/4 个对。这么稀疏时，即使做对齐，每个专家段大概率也只有 1 个有效对 + 一堆补位——**对齐合并不出"多样本块"，布局整理的收益为零**，纯属多花一次 kernel 的钱。于是干脆跳过，按"一块一对"直接算。

对 Qwen3-30B-A3B（E=128，top-8）的推论：无 EP 时条件② = `tokens×8×4 ≤ 128` → **tokens ≤ 4** 才触发。只要 ≥5 个 token 或开了 EP，必然走对齐模式。

### 8.2 两条路的宏观意义差别（核心对照）

| | 走 `moe_align_block_size`（非 naive） | 走 naive 捷径 |
|---|---|---|
| **一句话** | 先排版再按块算——**加载一次专家权重，在一个 block 里算 BLOCK_M 行，跳着取 token** | 不排版，**一个 block 只算一个 (token, 专家) 对** |
| 前置计算 | 三段式整理（launch + 原子计数 + 前缀和 + scatter + 同步） | 无（连 kernel 都不用开） |
| 布局信息 | `sorted_token_ids` 间接表：槽位 → 原对索引，跳着取 | `None`：块号即对索引，恒等映射 |
| 专家号 | `expert_ids[pid_m]`（每块一个） | `topk_ids.view(-1)[pid_m]`（展平后每对一格） |
| 权重加载 | 每 block 一次，被 BLOCK_M 行复用 | 每 block 一次，**只被 1 行用**（靠 L2 缓存兜底） |
| token 取用 | 查表跳着取（gather） | 直接从块号推：`row = pid_m // top_k` |
| grid 大小 | `num_tokens_post_padded / BLOCK_M` 个块 | `T×k` 个块（每块有效 1 行） |
| 适用场景 | token 多、专家共享多 | 极小 batch（decode 低并发），memory-bound |

**为什么 naive 敢于"浪费"**：小 batch decode 时瓶颈是**读权重带宽**而不是算力（memory-bound）。block 里只有 1 行有效 = 算力浪费，但 SM 反正也在等内存，浪费的 lane 几乎免费；真正花钱的是 kernel launch 开销，naive 把它全省了。等 batch 变大进入 compute-bound，权重复用开始值钱，条件②自动不满足，切回对齐模式。

### 8.3 naive 的下游流程（后置计算）

naive 的返回三元组与对齐模式"形似而语义迥异"，是同一份"生产者-消费者契约"的两种方言：

| 返回值 | 对齐模式语义 | naive 模式语义 |
|---|---|---|
| `sorted_token_ids` | 槽位 → 对索引的间接查表 | `None`：对索引即 block 号，无需间接层 |
| `expert_ids` | 每 BLOCK_M 槽位块一个专家（已排序） | 直接复用 `topk_ids.view(-1)`：每对（=每块）一个专家 |
| `num_tokens_post_padded` | 有效槽位总数 | `numel × BLOCK_M`，编码"numel 个活跃块" |

下游 `fused_moe_kernel` 以 constexpr `naive_block_assignment=True` 特化（L419-434）：

```python
offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)  # 块号即对索引，其余 lane 越界被 mask
off_experts = tl.load(expert_ids_ptr + pid_m)              # 展平 topk_ids 的第 pid_m 项
```

- grid：`grid_m = numel`（T×k 个 block），**一个 block 对应一个 (token, expert) 对**
- 块内 BLOCK_M 条 lane 只有第 0 条干活，其余填越界值被 mask
- **设计精髓**：把"对索引 = block 索引"做成恒等映射，整个砍掉 `sorted_token_ids` 间接表——用"每块只算一行"换"省掉前置 kernel + 省掉查表"

**"对"在下标里，不在值里**：`topk_ids` 形状 `[num_tokens, top_k]`，行优先展开后——**值**回答"找哪个专家"（定位权重），**下标**回答"是哪个 token"（`i // top_k` 得行号）。naive 不打乱路由器原始顺序，位置本身就是对索引，所以不需要映射表。

### 8.4 收益与代价（roofline 视角）

- **代价**：放弃 block 级权重复用。三层缓冲：① L2 cache 兜底（kernel 的 `GROUP_SIZE_M` 分组排序就是为促进 L2 复用）；② 条件②保证撞专家本就稀少；③ 无共享时两种模式的 tile 浪费相同（对齐块里也只有 1 有效行 + 补位）
- **收益**：省掉 align kernel（launch + 原子计数 + 前缀和 + scatter + 同步）× 48 层 × 每 decode step
- **本质**：小 batch decode 是 **memory-bound**——浪费的计算 lane 几乎免费（SM 反正在等内存），省下的 launch 开销是真金白银；batch 增大进入 compute-bound 后自动切回对齐
- **阈值意义**：`tokens×top_k×4 ≤ E` 保证平均每专家 ≤ 1/4 对，稀疏到对齐也合并不出多样本块，跳过预处理是纯收益

### 8.5 微型例子（2 token, top-2, E=16, BLOCK_M=4）

```
topk_ids = [[3, 7], [7, 12]]    ← t0→E3,E7；t1→E7,E12（共享 E7）
```

**对齐模式**（3 blocks）：E3 段 `[p0,✕,✕,✕]`、E7 段 `[p1,p2,✕,✕]`、E12 段 `[p3,✕,✕,✕]`——**E7 权重加载一次算两个 token**。

**naive 模式**（4 blocks）：block 0-3 分别算 (t0,E3)、(t0,E7)、(t1,E7)、(t1,E12)，每块仅第 0 条 lane 有效——E7 被两个 block 各读一次。

### 8.6 适用范围辨析：naive ≠ decode 默认路径

naive 判断的是**本 forward 的实际 token 数**而非推理阶段。decode 阶段每请求每步 1 token，`num_tokens` = 并发请求数：

| 场景 | num_tokens | 路径 |
|---|---|---|
| 单人调试（batch=1） | 1 | naive |
| 小流量 decode（batch ≤ 4） | ≤4 | naive |
| **生产高并发 decode（batch 几十~几百）** | ≥5 | **对齐模式** |
| prefill（整段 prompt） | 几百~几千 | 对齐模式 |

- **EP 开启 → 永不 naive**（`expert_map is None` 是硬条件）
- **投机解码**（MTP/EAGLE）：`num_tokens = batch × (1+k)`，阈值更快被突破
- 分支**每 step 动态判断**：batch 随流量波动，路径可实时切换（两路径结果等价，纯性能启发式）

---

## 九、与周边概念的联系

- 背景：[[LLM 前向计算]]（token / hidden states / FFN = 两次 GEMM / 计算与访存）
- 上层：解决 [[MoE]] 路由输出散乱的问题；受 [[序列并行 SP]] token 切分影响（TP=8 时 32 token → 每卡 4 个 → 恰好满足 naive 条件，触发捷径）
- 下层：为 [[Triton]] 编写的 `fused_moe_kernel`（[[Grouped GEMM]]）提供可直接消费的定长布局
- 并行策略：[[专家并行 EP]]（expert_map / -1 跳过 / 过滤模式）、[[数据并行 DP]]（batched 变体）、[[CUDA Graph]]（定形约束）、[[All-to-All]]（dispatch/combine 通信）
- 来源：[[moe_align_block_size-交互式详解]]、[[moe_align_block_size-代码路径]]
