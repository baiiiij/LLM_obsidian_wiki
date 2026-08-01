---
type: concept
title: PyTorch ATen Dispatcher（算子路由机制）
created: 2026-07-31
updated: 2026-07-31
sources: []
tags: [pytorch, aten, dispatcher, view, autograd]
---

# PyTorch ATen Dispatcher

PyTorch 算子从 Python 调用到 device kernel 的**路由系统**。理解它是阅读一切 aten 算子源码的前提。

> 配套页面：[[PyTorch-代码生成管线]]（dispatcher 的表与各层代码是谁生成的）；[[PyTorch-源码分析工具箱]]（profiler/trace 等观测手段）；[[FlagGems]]（利用注册表可变性做算子覆盖的典型）；[[pytorch-flatten-调用链路定位]]（实例）。

## 双层模型：算子 vs kernel

必须区分同名的两个东西：

| 层 | 例子 | 是什么 |
|---|---|---|
| **算子（op）** | `aten::reshape` | dispatcher 表里的一项，有 schema、有注册表项。**从 Python / `at::` 包装进入时永远过表** |
| **kernel** | `at::native::reshape_symint` | 注册在表里的那个函数，查表的**终点**，逻辑本体 |

"dispatch" 发生在**到达 kernel 之前**：包装 → 查表 → kernel。kernel 自己"不 dispatch"——它是被查表找到的目的地。kernel 内部再调别人时，作者二选一：过表（`self.view(...)`/`at::clone(...)`）或直连（`native::yyy(...)`）。

实测对照（pip torch 2.12，CPU profiler）：

```
t.reshape(6, 4)   → aten::reshape → aten::view   ← reshape 作为算子，dispatch 了
t.flatten(0, 1)   → aten::flatten → aten::view   ← 没有 aten::reshape：
                                                     flatten 的 kernel 直连 reshape 的 kernel
```

## 过 dispatcher vs 直接函数调用

**"过 dispatcher"的精确定义**：调用经过 `c10::Dispatcher` 的算子注册表查表：

```
at::clone(t)  ← 调用方：任何 C++ 代码（你写的扩展 / 生成的 TensorBody 包装 / PyTorch 自己的 kernel）
  → 从 t 算出 DispatchKeySet（CPU/CUDA？requires_grad？……）
  → 查表：先命中 Autograd key 的 kernel（记录 backward 所需信息）
       → 内部 redispatch：摘掉 Autograd key 再查一次
  → 命中 backend kernel（CPU/CUDA 真正的实现）
  → 沿途触发：RecordFunction（profiler 靠它）、dispatch trace、torch.compile 的拦截点
```

**触发 dispatch 的入口只有三种**：① Python 绑定（生成的 `THPVariable_xxx`）；② C++ 里调 `at::foo(...)` 或 `tensor.foo()`（生成的包装）；③ 显式 `at::_ops::foo::call(...)`。

**"不过 dispatcher"**：`native::reshape_symint(self, shape)` 是普通 C++ 函数调用——编译器直接跳转，没有查表、没有 key、没有 RecordFunction，profiler 和 torch.compile 都看不见它。

**机械层面**：dispatch 不是算子的"属性"，而是**你调用的那个函数，函数体里有没有查表代码**。`at::_ops::clone::call` 的整个函数体就是查表+跳转（生成器源码 torchgen/gen.py:667 白纸黑字）：

```cpp
static auto op = create_clone_typed_handle();  // = Dispatcher::singleton().findSchemaOrThrow(...)
return op.call(self, memory_format);           // ← 进 dispatcher 查表跳转
```

而 `at::native::reshape_symint` 是手写普通函数，编译/链接期定跳转。**kernel 作者对每个调用点二选一。**

**选择的判断准则**：

> 实现是否随 tensor 的运行时属性（设备 / 梯度 / 追踪）而变？变 → 必须 dispatch；不变 → 直接调用。

- clone：拷贝实现按设备路由 → 过表；
- reshape 的 shape 逻辑：纯整数运算，CPU/CUDA 一字不差 → 直连（过表有开销：算 key、哈希查找、间接跳转；且 flatten 已过 autograd 层，再过表会重复记录 view）。

## 为什么会有 dispatcher（设计动机）

要解决的问题：**一个算子名对应 N 个实现，选哪个在运行时才知道**。而写算子编排逻辑的文件（如 TensorShape.cpp）是设备无关、只编译一次的；甚至新后端可以是运行时加载的插件（PrivateUse1），编译期不存在。

| 方案 | 为什么不行 |
|---|---|
| 调用点写 if/switch | 2000+ 算子 × N 后端，每加一个后端改几千个调用点；插件后端编译期不存在 |
| 虚函数 | vtable 只支持**一根轴**的多态（对象类型）。PyTorch 有多根正交轴要同时生效：设备、autograd（`requires_grad` 是运行时标志不是类型）、tracing、functionalization |
| **注册表（dispatcher）** | 算子名 → 按优先级排列的 (DispatchKey → kernel)。加后端 = 运行时注册新 kernel，调用点零改动 |

杀手级收益：

1. **横切机制的统一拦截点**：profiler 的 RecordFunction、dispatch trace、torch.compile 的拦截全挂在"查表"这一步——一次插桩覆盖全部 2000+ 算子（这就是 profiler 看不见 kernel 内部直连的原因）；
2. **autograd 不用逐算子手写**：Autograd key 优先级高于后端 key，注册一份包装就自动给所有算子套上 backward 记录。

## 注册表：两个时刻与真实结构

```
【加载期：import torch 时，main 之前】
Register{Key}.cpp 里的静态初始化代码执行 m.impl(...)，把 kernel 函数指针塞进全局表
【运行期：调用发生时】
create_{op}_typed_handle() 拿算子句柄 → op.call(...) 算 DispatchKeySet → 查表 → 跳函数指针
```

**表的真实结构**（`aten/src/ATen/core/dispatch/OperatorEntry.h:242-259`）：每个算子一个 `OperatorEntry`，`kernels_` 按 key 存注册列表，`dispatchTable_[key]` 取**列表头部 = 最后注册者生效**；`registerKernel`/`deregisterKernel_`（OperatorEntry.cpp:137/237）运行时增删并重算表。Register*.cpp 里的 `m.impl` 只是加载期的"种子代码"，执行一次写完即完成使命。

## 注册表运行时可变：覆盖机制

`m.impl` 不是 build 期一次性写死的——第三方库可在运行时**覆盖**既有算子的 kernel 而不改动 PyTorch 任何文件：

- **API**：C++ `TORCH_LIBRARY_IMPL`；Python `torch.library.Library("aten", "IMPL")` + `lib.impl("mm", fn, "CUDA")`；
- **两种形态**：① 同 key 替换（`("aten::mm", CUDA)` 格从原版 kernel 换成新实现）；② 对 composite 算子放更具体的 key（composite 注册项是所有后端的 fallback，具体 backend key 查表优先——"遮蔽"而非"替换同一格"）；
- **三条边界**：只拦过表调用（kernel 内部直连感知不到）；按 key 粒度（CUDA 注册不影响 CPU）；autograd 层不受影响（Autograd key 先命中，backward 需单独注册 `_backward` 变体接管）。

典型应用详见 [[FlagGems]]（Triton 算子库的覆盖实践，含实证方法）；同机制用户还有 PrivateUse1 后端厂商、[[vLLM]]/torch.compile 的自定义算子。

## 算子的三种实现形态（yaml 决定注册到哪个 key）

| yaml 形态 | 注册位置 | kernel |
|---|---|---|
| `dispatch: CPU: foo_cpu / CUDA: foo_cuda` | RegisterCPU/RegisterCUDA | per-backend kernel |
| 无 `dispatch:` 段 | RegisterCompositeImplicitAutograd | 一份实现服务全后端（如 flatten、reshape） |
| `structured_delegate: bar` | 委托算子 | meta 算 shape 后转调 bar（旧版 flatten 曾用） |

完整字段表见 [[PyTorch-代码生成管线]]。注意：composite 算子之间经常**直接 C++ 调用**（flatten 直接调 `native::reshape_symint`）——读代码时不要想当然地以为每次调用都过表。

## View vs Copy：元数据算子的分水岭

reshape/view/flatten/permute 这类**不改数据的算子**，分水岭在 `computeStride`（`aten/src/ATen/TensorUtils.cpp`）：

- 目标 shape 能由现有 (sizes, strides) 的连续段拼出 → **view**：`_reshape_alias` → `alias_with_sizes_and_strides` 新建 TensorImpl、**共享同一 Storage**、只写元数据，0 拷贝；
- 拼不出 → **copy**：`clone(Contiguous)` dispatch 到 device copy kernel 搬数据，再 `_unsafe_view`。

view 的 autograd 通过 `ADInplaceOrView` key 上的生成包装记录 ViewInfo，backward 时按逆变换回传梯度；因此 view 之后的 in-place 修改会触发 autograd 报错，而 copy 不会。逐行源码见 [[pytorch-flatten-调用链路定位]]。

## 与知识库主线的关联

框架层（[[vLLM]] 等）调 aten 算子时同样经过此管线；自定义算子（`TORCH_LIBRARY`）也是向 Dispatcher 注册 key。性能分析时，[[Grouped GEMM]] 前后的 layout 整理（如 [[moe_align_block_size]]）是否产生隐式 copy，可用 profiler 中 `aten::clone/copy_` 是否出现来判定。
