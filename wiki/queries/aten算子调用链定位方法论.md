---
type: query
title: 如何不凭先验经验，自己一步步推出 aten 算子的调用链（方法论）
created: 2026-07-31
updated: 2026-07-31
sources: []
tags: [pytorch, aten, dispatcher, codegen, 方法论]
---

# 问题

看 PyTorch 源码的"正常思路"应该是：点开 Python 的 flatten → 看到函数体 → if/else 分支 → 调不同底层算子 → 顺着名字找 C++。但点进去只有 .pyi 存根。**PyTorch 真实的调用流程到底是什么样的？不凭先验经验，如何自己一步步推出来？**

> **版本基准：本文所有源码引用以 GitHub `pytorch/pytorch` 的 `release/2.12` 分支为准**（已逐文件在线核对）。本机实验只跑 CPU 上的 Python 侧观测（观测结论与有无 GPU 无关），GPU 相关行为以源码为准并注明。

**前置阅读**（本页只讲操作流程，概念与工具详解已拆出）：

- [[PyTorch-ATen-Dispatcher]] — 核心概念：双层模型（算子 vs kernel）、过表 vs 直连、注册表结构
- [[PyTorch-代码生成管线]] — 文件地图：每个文件由谁生成、生成到哪、新算子找哪个绑定文件
- [[PyTorch-源码分析工具箱]] — profiler / dispatch trace / rg / gdb 的用法与坑

# 〇、先纠正心智模型：分支存在，但不在 Python 层

你预期的流程（Python 函数体里写分支）对**少数**算子成立，但对绝大多数 aten 算子不成立。原因：PyTorch 采用 **schema 驱动代码生成**（详见 [[PyTorch-代码生成管线]]），Python 层只是生成的薄壳（解析参数 → 扔进 dispatcher），没有业务逻辑。

你期待的"条件分支"搬到了两个地方：

| 分支位置 | 判断什么 | 例子 |
|---|---|---|
| ① Dispatcher（查表，自动） | tensor 的设备/requires_grad 等属性 | CPU 还是 CUDA？要不要包 autograd？ |
| ② C++ kernel 内部的 if/else | 数据布局等运行时条件 | `computeStride` 成功 → view；失败 → clone |

# 一、七步 SOP（不凭先验经验的推导流程）

## 第 1 步：判别"这个 op 有没有 Python 包装层"

```python
import torch, inspect
torch.Tensor.flatten                       # <method 'flatten' of 'torch._C.TensorBase' objects>
inspect.getsource(torch.Tensor.flatten)    # TypeError: ... got method_descriptor ← C 扩展方法
inspect.getsource(torch.Tensor.unflatten)  # 成功！源码在 torch/_tensor.py ← 有 Python 包装
```

**判别规则**：`inspect.getsource` 成功 → 有 Python 层，按常规思路读；报 `TypeError ... method_descriptor` → C 扩展的方法描述符，Python 层不存在，进第 2 步。两种流程 PyTorch 都有，逐算子判别，不需要背。

## 第 2 步：docstring 里白捡 schema

```python
torch.Tensor.flatten.__doc__   # 'flatten(start_dim=0, end_dim=-1) -> Tensor ...'
```

这是生成绑定时从 yaml 顺手写进去的，是你去 yaml 搜索的钥匙。

## 第 3 步：profiler 拿"实际 dispatch 了哪个算子"的 ground truth

"万一点进去的 Python flatten 调的不是 aten::flatten 呢？"——不猜，直接观测（原理：profiler 钩子埋在 dispatcher 里，见 [[PyTorch-源码分析工具箱]]）：

```python
with profile(activities=[ProfilerActivity.CPU]) as prof:
    t.flatten(0, 1)          # 事件流：aten::flatten → aten::view
    t.transpose(0,1).flatten(0,1)  # 事件流：aten::flatten → aten::clone → aten::copy_ → aten::_unsafe_view
```

### 关于"torch.xxx 的名字和 yaml 里一定一样吗？"——不一定，但这套流程不依赖它

**多数对得上**（flatten、sum、permute……），但有几类对不上（本机实测）：

| 情况 | 例子 | profiler 里看到的 |
|---|---|---|
| 运算符语法，没有 torch.xxx 调用点 | `a + a` | `aten::add` |
| 一个调用产生一串事件（内部 redispatch） | `torch.concat([...])` | `aten::concat → aten::cat` |
| 同上 | `a @ a` | `aten::matmul → aten::mm` |
| 根本不是算子 | `torch.save`、`torch.compile`、`torch.no_grad` | 没有任何 aten 事件 |
| Python 组合层 | 部分 `torch.nn.functional`/`torch.linalg` 函数 | 多个 aten 事件的组合 |

所以 SOP 的顺序是**先 profiler 拿真实的 aten 名，再拿这个名字去 yaml 搜**——不依赖同名假设。另外注意：yaml 覆盖所有 aten:: 算子（dispatcher 的注册表就是从 yaml 生成的），唯一例外是第三方扩展用 `TORCH_LIBRARY` 注册的自定义算子——那种 profiler 也能看到名字，但要去扩展自己的源码里搜。

## 第 4 步：yaml 里多个同名算子，按参数类型匹配

`rg "func: flatten" aten/src/ATen/native/native_functions.yaml`（2.12 在第 2702 行起）出来 4 个：

```yaml
- func: flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
- func: flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)
- func: flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)
- func: flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)
```

**匹配看参数类型，不看名字**：你传的是两个 int → 只可能命中第一个；后三个参数是 `Dimname`（维度名），属于 named tensor 特性（`t.flatten('N','C',out_dim='NC')` 这种给维度起名的写法）——.pyi 里带 `str` 的 overload 就对应它们。点号后的 `using_ints` 只是 codegen 的内部标识；C++ 侧由生成的绑定代码（PythonArgParser）按签名逐个尝试匹配。

交叉验证：第 2 步 docstring 的 `flatten(start_dim=0, end_dim=-1)` 与 `using_ints` 签名（含默认值）完全一致；第 3 步 trace 里打出的全名也正是 `aten::flatten.using_ints`。

## 第 5 步：读 yaml 条目，决定实现去哪找

| yaml 长什么样 | 含义 | 去哪找实现 |
|---|---|---|
| 有 `dispatch: CPU: foo_cpu / CUDA: foo_cuda` | 每后端独立 kernel | `rg "foo_cpu" aten/` |
| **没有 `dispatch:` 段**（flatten 就是这种） | 默认 CompositeImplicitAutograd：一份 C++ 实现服务所有后端 | `rg "Tensor flatten" aten/src/ATen/native/` → `at::native::flatten` |
| `CompositeImplicitAutograd: bar` | 同上，实现函数名是 bar | rg bar |
| `structured_delegate: bar` | meta 算 shape 后委托给 bar | 看 bar（旧版 flatten 曾是这种，2.12 已改为直接实现） |

### 搜索词 `Tensor flatten` 是怎么推出来的（不是经验）

直接 `rg "flatten"` 在 TensorShape.cpp 一个文件就有 38 个命中（注释/调用点/同名函数），信噪比太低，需要收窄到"函数定义行"。收窄依据全部来自 yaml 本身：

1. yaml 那行的 `-> Tensor(a)` 给出**返回类型**；torchgen 约定 C++ 实现签名 = 返回类型 + 算子名 + schema 参数（`Tensor(a) self`→`const Tensor& self`，`int`→`int64_t`）；
2. 于是搜索词 = **返回类型 + 算子名 + `(`**，即 `rg "Tensor flatten"` → 从 38 收敛到 5，命中 `Tensor flatten(const Tensor& self, int64_t start_dim, ...)`，参数类型还能反向确认 overload（带 `DimnameList` 的是 named tensor 变体）；
3. "无 dispatch 段 → 函数名 = 算子名"这条规则也不用背：build 后 `rg "aten::flatten" build/aten/src/ATen/RegisterCompositeImplicitAutograd*.cpp`，生成的注册表（dispatcher 的电话簿）里白纸黑字写着 op 名 → 实现函数的绑定，这是零先验路线。

通用技巧：**命中太多加特征（返回类型/左括号/参数类型），命中太少减特征；搜索是迭代收窄，不是一击即中。**

## 第 6 步：读 C++，分辨"过不过 dispatcher"

2.12 定位到 `aten/src/ATen/native/TensorShape.cpp:4178` 的 `at::native::flatten`：算完目标 shape 后 `return native::reshape_symint(self, shape);`——直接调用，不过 dispatcher（与第 3 步 trace 里没有 `aten::reshape` 互相印证）。顺到 `reshape_symint`（:2058）就能看到 view/copy 分水岭 `computeStride`（详见 [[pytorch-flatten-调用链路定位]]）。

**通用判别**：`self.xxx()` / `at::xxx()` → 过 dispatcher（profiler 可见）；`native::xxx(...)` → 不过（profiler 不可见）。原理见 [[PyTorch-ATen-Dispatcher]]"过 dispatcher vs 直接函数调用"。

## 第 7 步：看懂 Python↔C++ 的胶水是怎么生成的

到第 6 步链路已闭环，这一步是补全"Python 的 `t.flatten` 到底怎么接到 C++"的最后一块拼图。涉及 4 个文件，前 3 个源码自带：

**① 模板（源码自带）`tools/autograd/templates/python_variable_methods.cpp`**

打开它你会发现它"不完整"——这正是模板的特征：

```cpp
// ${generated_comment}              ← 第 2 行：生成时会被替换成"本文件由 xx 生成，勿手改"
...
// generated methods start here     ← 第 1079 行
${py_methods}                        ← 占位符：所有 THPVariable_xxx 函数体注入到这里
...
PyMethodDef variable_methods[] = {   ← 文件尾部：方法表
  {"new", ...}, {"numpy", ...},      ← 手写的特殊方法
  ${py_method_defs}                  ← 占位符：生成的 {"flatten", THPVariable_flatten, ...} 条目注入这里
  {nullptr}
};
```

**② 生成器（源码自带）`tools/autograd/gen_python_functions.py`**

build 时运行：读 `native_functions.yaml`，为每个带 `variants: method` 的算子生成一个 `THPVariable_{name}` 函数——内部用 `PythonArgParser` 按 schema 签名解析 Python 参数（这就是为什么多个 overload 能共存：ArgParser 逐个签名试匹配），然后调用 C++ 接口。生成的 `THPVariable_flatten` 大致是：

```cpp
// （示意，build 后可 rg 到原文）
static PyObject* THPVariable_flatten(PyObject* self_, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flatten(int64_t start_dim=0, int64_t end_dim=-1)",
    // ...Dimname 变体的签名也在这里
  });
  auto& self = THPVariable_Unpack(self_);
  // 解析参数 → 调用 dispatch_flatten(self, start_dim, end_dim) → 包装回 PyObject
  END_HANDLE_TH_ERRORS
}
```

**③ 输出（build 生成）`torch/csrc/autograd/generated/python_variable_methods.cpp`**

模板 + 生成器的产物，编译进 `torch._C` 扩展。build 之后 `rg THPVariable_flatten torch/csrc/autograd/generated/` 就能看到原文。

**④ 挂载点（源码自带）`torch/csrc/autograd/python_variable.cpp`**

```cpp
// :3887  extern PyMethodDef variable_methods[];        ← 引用生成的那张表
// :3913  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
//         ← 把表挂到 THPVariable 类型上，即你在 Python 看到的 torch._C.TensorBase
```

于是 `t.flatten` 的完整路径闭环了：**CPython 属性查找 → variable_methods 表找到 "flatten" → THPVariable_flatten → PythonArgParser → _ops::flatten_using_ints::call → dispatcher → at::native::flatten**。

**终极验证手段——gdb**：如果你的编译版带符号，`gdb --args python your_script.py`，然后 `break at::native::flatten`、`run`、`bt`，就能看到从 `THPVariable_flatten` 一路到 `computeStride` 的完整真实调用栈——不依赖任何"我以为"，栈帧就是真相。

# 二、完整分析链路（任一算子通用）

**链路图只画"运行时发生了什么"**（每行一个环节 + 一句话角色）：

```
① t.op(...) / torch.op(...)              Python 调用点
   ↓
② THPVariable_op（或 torch.op 的绑定函数）  解析 Python 参数 → 调 C++
   ↓
③ at::_ops::op::call()                    C++ 包装，body = 查表入口
   ↓
④ c10::Dispatcher 查表                     按 tensor 的 DispatchKeySet 路由：
   ├─ Autograd/ADInplaceOrView 包装 kernel   先命中：记 backward / view 关系
   └─ 注册表项 → kernel 函数指针             再命中：算子名绑定的实现
   ↓
⑤ at::native::xxx()                        kernel 本体，逻辑所在
   ↓
⑥ kernel 内部：直连别的 kernel，或再次过表调别的算子
```

**每层的归属与验证**（生成关系的完整参考见 [[PyTorch-代码生成管线]]）：

| 环节 | 产物文件 | 谁生成的 | 验证（无需 build） | 验证（需要 build） |
|---|---|---|---|---|
| ② Python 绑定 | `torch/csrc/autograd/generated/python_variable_methods.cpp`（`torch.op` 则是 `python_torch_functions.cpp`） | `tools/autograd/gen_python_functions.py` ← 模板 `tools/autograd/templates/python_variable_methods.cpp` | 读生成器源码（:233-237 的归属判定规则）与模板 | `rg THPVariable_op torch/csrc/autograd/generated/` |
| ③ C++ 包装 | `build/aten/src/ATen/ops/{op}_ops.h` + `{op}.h` | `torchgen/gen.py` ← 模板 `aten/src/ATen/templates/Operator.h`、`Function.h` | **pip 版自带**：`torch/include/ATen/ops/{op}.h` 里能直接看到 `at::{op}()` 的整个 body 就是一行 `return at::_ops::{op}_{overload}::call(...)` | 读 build 产物 |
| ④a Autograd 包装 | `torch/csrc/autograd/generated/VariableType*.cpp` | `tools/autograd/gen_autograd.py` ← 模板 `VariableType.cpp` | profiler 事件树的父子层级 | 读产物 |
| ④b 注册表项 | `build/aten/src/ATen/Register{Key}.cpp` 里的 `m.impl("aten::op", TORCH_FN(kernel))` | `torchgen/gen.py` | **pip 版旁证**：`torch/include/ATen/ops/{op}_{key}_dispatch.h`（如 `flatten_compositeimplicitautograd_dispatch.h`）证明该 op 注册在哪个 key 下 | `rg "aten::op" build/aten/src/ATen/Register*.cpp` |
| ④c 查表机制本身 | 源码自带：`aten/src/ATen/core/dispatch/Dispatcher.h` | 手写 | 直接读源码 | `TORCH_SHOW_DISPATCH_TRACE=1`（需 debug build）/ gdb |
| ⑤ kernel 本体 | 源码自带：`aten/src/ATen/native/*.cpp` | 手写 | `rg "Tensor xxx" aten/src/ATen/native/` | — |
| ⑥ 内部调用 | — | — | profiler 事件树：出现 `aten::zzz` = 过表；没出现 = 直连 | — |

**两个通用判别动作**（比背这张表更重要）：

1. **生成的文件自己会报出生**：每个生成文件的头注释都写着 `@generated by {生成器} from {模板}`——拿到任何一个文件，先看头注释就知道它在地图里的位置。pip 版 torch 的 `torch/include/ATen/ops/*.h` 全部可验。
2. **验证分两类**：无需 build 就能做的（pip 自带头文件、生成器源码、模板、native kernel、profiler）；需要 build 才能做的（rg 生成的 .cpp、注册表项原文、dispatch trace）。分析时先用前一类走完全程，后一类只作为补充实证。

## ④ 详解：控制流转数据流的断点（从 op.call 到 kernel 的正确走法）

**这是整个链路唯一需要换工具的地方。** `op.call()` 再往里（`Dispatcher::call`，Dispatcher.h:179）是 2000+ 算子**共用的通用路由机器**——只有"算 key、查表、跳函数指针"，没有任何算子专属代码。顺着调用点往里读，永远读不到具体算子。算子与 kernel 的绑定是**加载期写进表里的数据**，要用搜索找"注册"，而不是找"调用"。

**数据流的两个时刻**：

```
【加载期：import torch 时，main 之前】
Register{Key}.cpp 里的静态初始化代码执行 m.impl(...)，把 kernel 函数指针塞进全局表
【运行期：调用发生时】
create_{op}_typed_handle() 拿算子句柄 → op.call(...) 算 DispatchKeySet → 查表 → 跳函数指针
```

**具体走法（有 build 目录时）**：

```bash
# ① 找注册项 + 通往 kernel 的最后一跳（同一文件内）
rg -n "flatten" build/aten/src/ATen/RegisterCompositeImplicitAutograd*.cpp
#   会看到：
#   a) 包装定义：compositeimplicitautograd::flatten_using_ints(...) { return at::native::flatten(...); }
#   b) 注册语句：m.impl("aten::flatten.using_ints", TORCH_FN(compositeimplicitautograd::flatten_using_ints));

# ② 找 Autograd/view 包装（view 算子走 ADInplaceOrView key）
rg -n "flatten" torch/csrc/autograd/generated/ADInplaceOrViewType*.cpp \
                 torch/csrc/autograd/generated/VariableType*.cpp
```

选哪个 `Register{Key}.cpp`：按 yaml 推断（无 dispatch 段 → CompositeImplicitAutograd；有 dispatch 段 → 对应后端 key）。懒得推断就全搜 `rg "aten::op.overload" build/aten/src/ATen/Register*.cpp`，让文件自己命中。注册语句的固定格式、三件套结构与分片机制详见 [[PyTorch-代码生成管线]]"Register 文件"一节。

**注册表运行时可被第三方覆盖**（[[FlagGems]] 等）：理解 ④ 的数据流后，"为什么覆盖只拦过表调用、拦不到 kernel 内部直连"可以直接推出（机制详见 [[PyTorch-ATen-Dispatcher]]"注册表运行时可变"）。

读完注册项里 `at::native::flatten` 这一跳，就回到"读普通 C++ 函数"的世界——kernel 本体及之后的内部调用（⑤⑥）不再有任何表查询的间接性。

# 三、SOP 一页纸总结

```
1. inspect.getsource 失败？ ──否──→ 读 Python 包装层（如 torch/_tensor.py）
        │是（method_descriptor）
2. 读 __doc__ → 白捡 schema 签名
3. profiler(CPU activity) → 拿到真实 aten 算子名（不过 dispatcher 的步骤这里看不见）
4. rg "func: {op}" native_functions.yaml → 按【参数类型】匹配 overload
5. 看 dispatch 段决定实现形态（无段 = Composite → at::native::{op名}）
6. rg 定位 C++：at::xxx/self.xxx() 过 dispatcher；native::xxx 不过
7.（可选）胶水链：templates → gen_python_functions.py → generated/ → python_variable.cpp 挂载
   终极验证：gdb break at::native::{op} + bt 看真实调用栈
```

核心原则：**能观测的不猜测**（profiler / dispatch trace / gdb 拿 ground truth），**能匹配的不背诵**（参数类型对 overload），**yaml 是唯一地图**。

## 相关页面

- [[PyTorch-ATen-Dispatcher]] — 概念：双层模型、过表 vs 直连、设计动机、注册表与覆盖机制
- [[PyTorch-代码生成管线]] — 文件地图：生成器/产物/绑定归属判定/Register 三件套
- [[PyTorch-源码分析工具箱]] — profiler / dispatch trace / rg / gdb 详解
- [[pytorch-flatten-调用链路定位]] — 用本 SOP 走完 flatten 全程的实例（2.12 行号）
- [[FlagGems]] — 注册表覆盖机制的典型第三方应用
