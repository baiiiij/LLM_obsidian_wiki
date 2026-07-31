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

# 〇、先纠正心智模型：分支存在，但不在 Python 层

你预期的流程（Python 函数体里写分支）对**少数**算子成立，但对绝大多数 aten 算子不成立。原因：PyTorch 有 2000+ 算子，每个都要配 Python 绑定、C++ 方法、autograd、各后端 kernel、类型存根——全手写不可维护，于是采用 **schema 驱动代码生成**：`native_functions.yaml` 是唯一事实来源，Python 层只是生成的薄壳（解析参数 → 扔进 dispatcher），没有业务逻辑。

你期待的"条件分支"搬到了两个地方：

| 分支位置 | 判断什么 | 例子 |
|---|---|---|
| ① Dispatcher（查表，自动） | tensor 的设备/requires_grad 等属性 | CPU 还是 CUDA？要不要包 autograd？ |
| ② C++ kernel 内部的 if/else | 数据布局等运行时条件 | `computeStride` 成功 → view；失败 → clone |

# 一、工具箱详解（先把 SOP 里用到的每个工具讲透）

## 1.1 torch.profiler 是什么

`torch.profiler` 是 PyTorch 自带的**插桩式性能分析器**。`profile` 是上下文管理器，`ProfilerActivity` 是指定"采集哪类事件"的枚举：

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:   # 在 with 块内开启采集
    t.flatten(0, 1)                                        # 块内所有算子调用被记录
print(prof.key_averages().table())                         # 汇总表
# prof.events() 可以拿到逐条事件流
```

它的数据来源（重点）：**dispatcher 的 C++ 代码里埋了观测钩子**。每次有算子经过 dispatcher，调用前后会触发一对 `RecordFunction` 回调（2.12 源码证据：`aten/src/ATen/core/dispatch/Dispatcher.h` 中 `callWithDispatchKeySlowPath` / `step_callbacks` 逻辑）。profiler 启动时注册一个全局回调，把每条事件的**算子名（schema 名，如 aten::flatten）**和耗时收集起来。

## 1.2 为什么是 `ProfilerActivity.CPU`，填别的会怎样

`ProfilerActivity` 的取值（2.12）：`CPU`、`CUDA`、`XPU`、`MTIA`、`HPU`、`PrivateUse1`（第三方后端自定义）。两类完全不同的东西：

| activity | 记录什么               | 事件名字长什么样                                                | 需要什么            |
| -------- | ------------------ | ------------------------------------------------------- | --------------- |
| `CPU`    | **主机侧**所有算子调用      | `aten::flatten`、`aten::clone`                           | 什么都不要，永远可用      |
| `CUDA` 等 | **设备侧** kernel 时间线 | CUDA kernel 的函数名（如 `Memcpy DtoD`、各种 elementwise kernel） | 对应硬件 + kineto 库 |

关键认知：**`aten::xxx` 这个名字只存在于 CPU 侧**（dispatcher 在 CPU 上运行）。即使 tensor 在 GPU 上，每次调用在 CPU 侧也会留下事件（kernel launch），所以**看"dispatch 了哪个算子"永远用 `CPU`**；`CUDA` 是给你看 GPU 上实际跑了哪些 kernel、耗时多少的，两者通常一起开来对齐时间线。

填了本机不支持的 activity 会怎样（本机无 GPU 实测）：

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    ...
# 不报错，只产生 warning：CUDA is not available, disabling CUDA profiling
# 结果里就只有 CPU 事件
```

查本机支持哪些：`torch.autograd.profiler._supported_activities()`（本机返回 `{ProfilerActivity.CPU}`）。

## 1.3 为什么 profiler 能看出 dispatch 了哪个算子

因为钩子（RecordFunction）**就埋在 dispatcher 里**——每一次"过 dispatcher"的调用都会被记录。反过来也成立：**不过 dispatcher 的调用，profiler 完全看不见**。

这就是为什么连续张量 `flatten` 的事件流是 `aten::flatten → aten::view`，**没有 `aten::reshape`**：flatten 内部是普通 C++ 直接调用 `native::reshape_symint`（见 §二），没经过 dispatcher，所以不产生事件。profiler 事件流 ≈ "本次执行过 dispatcher 的算子清单"，这本身就是判断"哪一步过 dispatch"的实验手段。

## 1.4 `TORCH_SHOW_DISPATCH_TRACE=1` 详解

- **作用**：dispatcher 每次查表前打印一行日志，格式（2.12 `Dispatcher.cpp` 的 `_print_dispatch_trace`）：

  ```
  [call] op=[aten::flatten.using_ints], key=[AutogradCPU]
   [redispatch] op=[aten::flatten.using_ints], key=[CPU]
  ```

  行首缩进表示调用嵌套层级。`key=[...]` 显示的是本次查表用的 dispatch key——**这是它和 profiler 的分工**：profiler 告诉你"跑了哪些算子"，trace 告诉你"每个算子按什么 key 逐级路由的"（能看到 Autograd → CPU 的逐级 redispatch，回答"dispatch 到底干了什么"）。

- **一个大坑（实测）**：打印代码被编译宏包着（2.12 `Dispatcher.h`）：

  ```cpp
  #if defined(HAS_TORCH_SHOW_DISPATCH_TRACE) || !defined(NDEBUG)
    if (show_dispatch_trace()) { detail::_print_dispatch_trace(...); }
  #endif
  ```

  **官方 pip release 版定义了 `NDEBUG`，这段代码被编译掉了**——本机 pip 版实测设了环境变量也毫无输出。它只在 **debug 编译**（`DEBUG=1 python setup.py develop`）或显式加了 `HAS_TORCH_SHOW_DISPATCH_TRACE` 宏的构建里有效。你有自己编译的源码，能否用取决于编译类型（`torch.__config__.show()` 可看 build 配置）。用不了也不亏：profiler 已覆盖 90% 的需求。

## 1.5 rg 是什么

`rg` = **ripgrep**，一个为"在超大代码库里搜文本"而生的命令行工具（Rust 实现）。为什么源码阅读都用它而不是传统 `grep`：

- **快**：pytorch 源码树几百万行，`grep -rn` 要几分钟，rg 通常秒出；
- **聪明**：自动跳过 `.gitignore` 里的目录（不会搜进 build 产物和 third_party 垃圾）、自动跳过二进制文件；
- 行号、颜色默认开启。

常用法（源码导航三板斧）：

```bash
rg "func: flatten" aten/src/ATen/native/native_functions.yaml   # 精确定位算子定义
rg "Tensor flatten" aten/src/ATen/native/                        # 找 C++ 实现
rg "computeStride" aten/ -g "*.cpp"                              # 只搜 .cpp 文件
```

没有 rg 的替代方案：`grep -rn "pattern" path`（慢但到处都有）；VSCode 全局搜索（Ctrl+Shift+F）；没有本地代码时用 GitHub 网页版 code search（`github.com/search` 里限定 repo:pytorch/pytorch）。安装：`sudo apt install ripgrep` 或 `conda install -c conda-forge ripgrep`。

# 二、dispatch vs 直接函数调用（核心概念）

这是理解整个链路最关键的一对概念。

**"过 dispatcher"的精确定义**：调用经过 `c10::Dispatcher` 的**算子注册表查表**。一次完整旅程：

```
at::clone(t)  ← 你写的/生成的包装代码
  → 从 t 算出 DispatchKeySet（CPU/CUDA？requires_grad？……）
  → 查表：先命中 Autograd key 的 kernel（记录 backward 所需信息）
       → 内部 redispatch：摘掉 Autograd key 再查一次
  → 命中 backend kernel（CPU/CUDA 真正的实现）
  → 沿途触发：RecordFunction（profiler 靠它）、dispatch trace、torch.compile 的拦截点
```

**触发 dispatch 的入口只有三种**：① Python 绑定（生成的 THPVariable_xxx）；② C++ 里调 `at::foo(...)` 或 `tensor.foo()`（生成的包装方法）；③ 显式 `at::_ops::foo::call(...)`。

**"不过 dispatcher"**：`native::reshape_symint(self, shape)` 就是一个**普通 C++ 函数调用**——编译器直接跳转，没有查表、没有 key、没有 RecordFunction、profiler 和 torch.compile 都看不见它。

**为什么 flatten 里两者混用**（2.12 TensorShape.cpp:4178 的 flatten → :2058 的 reshape_symint）：

| 调用 | 方式 | 原因 |
|---|---|---|
| `native::reshape_symint(...)` | 直接调用 | reshape 的逻辑（拼 shape + computeStride）所有后端共用同一份代码，没有"按设备路由"的需求；且 flatten 自己已经过了 autograd 包装，再调 `at::reshape` 会**重复记录** view 信息 |
| `self.clone(...)` | 过 dispatcher | 拷贝需要 per-device kernel（CPU memcpy / CUDA copy kernel），必须查表选后端 |

一句话：**dispatch = 按 tensor 属性动态路由到正确实现 + 触发 autograd/profiler/compile 等所有横切机制；直接调用 = 静态确定的普通函数跳转，什么都不触发。**

# 三、七步 SOP（不凭先验经验的推导流程）

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

"万一点进去的 Python flatten 调的不是 aten::flatten 呢？"——不猜，直接观测（原理见 §1.3）：

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

## 第 6 步：读 C++，用 §二的概念分辨"过不过 dispatcher"

2.12 定位到 `aten/src/ATen/native/TensorShape.cpp:4178` 的 `at::native::flatten`：算完目标 shape 后 `return native::reshape_symint(self, shape);`——直接调用，不过 dispatcher（与第 3 步 trace 里没有 `aten::reshape` 互相印证）。顺到 `reshape_symint`（:2058）就能看到 view/copy 分水岭 `computeStride`（详见 [[pytorch-flatten-调用链路定位]]）。

**通用判别**：`self.xxx()` / `at::xxx()` → 过 dispatcher（profiler 可见）；`native::xxx(...)` → 不过（profiler 不可见）。

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

# 四、SOP 一页纸总结

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

- [[pytorch-flatten-调用链路定位]] — 用本方法论走完 flatten 全程的实例（2.12 行号）
- [[PyTorch-ATen-Dispatcher]] — codegen 管线与 dispatch 机制总览
