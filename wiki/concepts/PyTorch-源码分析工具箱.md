---
type: concept
title: PyTorch 源码分析工具箱（profiler / dispatch trace / rg / gdb）
created: 2026-07-31
updated: 2026-08-03
sources: []
tags: [pytorch, profiler, ripgrep, 工具]
---

# PyTorch 源码分析工具箱

分析 aten 算子调用链时反复用到的四件工具，每件讲透"是什么、为什么有效、怎么用、有什么坑"。使用场景见 [[aten算子调用链定位方法论]] 的七步 SOP。

> 核心原则：**能观测的不猜测**（profiler / trace / gdb 拿 ground truth），**能匹配的不背诵**（rg 按特征迭代收窄）。

## 1. torch.profiler

### 是什么

PyTorch 自带的**插桩式性能分析器**。`profile` 是上下文管理器，`ProfilerActivity` 指定"采集哪类事件"：

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:   # with 块内开启采集
    t.flatten(0, 1)                                        # 块内算子调用被记录
print(prof.key_averages().table())                         # 汇总表
# prof.events() 拿逐条事件流
```

**数据来源（为什么它能看出 dispatch 了哪个算子）**：观测钩子（`RecordFunction` 回调）**就埋在 dispatcher 的 C++ 代码里**（2.12 `aten/src/ATen/core/dispatch/Dispatcher.h` 的 `callWithDispatchKeySlowPath` / `step_callbacks`）。每一次"过 dispatcher"的调用都被记录；反过来，**不过 dispatcher 的调用（kernel 内部直连），profiler 完全看不见**。所以 profiler 事件流 ≈ "本次执行过 dispatcher 的算子清单"，这本身就是判断"哪一步过 dispatch"的实验手段。

### activities 怎么选

`ProfilerActivity` 取值（2.12）：`CPU`、`CUDA`、`XPU`、`MTIA`、`HPU`、`PrivateUse1`。两类完全不同的东西：

| activity | 记录什么 | 事件名长什么样 | 需要什么 |
|---|---|---|---|
| `CPU` | **主机侧**所有算子调用 | `aten::flatten`、`aten::clone` | 什么都不要，永远可用 |
| `CUDA` 等 | **设备侧** kernel 时间线 | CUDA kernel 函数名（`Memcpy DtoD`、elementwise kernel） | 对应硬件 + kineto |

关键认知：**`aten::xxx` 这个名字只存在于 CPU 侧**（dispatcher 在 CPU 上运行）。即使 tensor 在 GPU 上，每次调用在 CPU 侧也会留下事件（kernel launch），所以**看"dispatch 了哪个算子"永远用 `CPU`**；`CUDA` 用来看 GPU 上实际跑了哪些 kernel。

填了本机不支持的 activity：**不报错**，只 warning（实测：`CUDA is not available, disabling CUDA profiling`）。查本机支持：`torch.autograd.profiler._supported_activities()`。

### 输出表格怎么读

```
     Name      Self CPU %   Self CPU   CPU total %   CPU total   CPU time avg   # of Calls
aten::flatten     45.87%     64.712us      100.00%    141.077us     141.077us       1
aten::view        54.13%     76.365us       54.13%     76.365us      76.365us       1
```

| 列 | 含义 |
|---|---|
| `Name` | 事件名——**调用关系树状嵌套**（flatten 是父，view 是子） |
| `Self CPU` | 算子**自己函数体**的耗时，**不含子调用** |
| `CPU total` | 算子**连同所有子调用**的总耗时。flatten：141.077us = 自己 64.712us + 子事件 view 76.365us |

**GPU 实验的正确开法**：必须 `CPU + CUDA` 都开。只开 CUDA 时表里**一个 aten 算子都没有**，只剩 `cudaDeviceSynchronize`（runtime API 事件）和 `Activity Buffer Request`（kineto 内部开销）这类非算子噪音。双开后多出 `Self CUDA` 列：

- 连续张量 flatten：CPU 列 `aten::flatten → aten::view`，**Self CUDA 全为 0**——GPU 零 kernel，"view 零拷贝"眼见为实；
- 非连续 flatten：CPU 列多出 `aten::clone/copy_/_unsafe_view`，**Self CUDA 出现 `Memcpy DtoD`**——显存里真的搬了数据。

**噪音识别**：`Profiler clears events at the end of each cycle` warning（多周期采集默认只保留当前周期，想累积设 `acc_events=True`）与 `USDT:...` 日志（kineto 探针）都是 profiler 基础设施输出，与代码无关。

## 2. TORCH_SHOW_DISPATCH_TRACE=1

- **作用**：dispatcher 每次查表前打印一行日志（2.12 `Dispatcher.cpp` 的 `_print_dispatch_trace`）：

  ```
  [call] op=[aten::flatten.using_ints], key=[AutogradCPU]
   [redispatch] op=[aten::flatten.using_ints], key=[CPU]
  ```

  行首缩进表示调用嵌套层级。**与 profiler 的分工**：profiler 告诉你"跑了哪些算子"，trace 告诉你"每个算子按什么 key 逐级路由"（能看到 Autograd → CPU 的逐级 redispatch）。

- **大坑（实测）**：打印代码被编译宏包着（2.12 `Dispatcher.h`）：

  ```cpp
  #if defined(HAS_TORCH_SHOW_DISPATCH_TRACE) || !defined(NDEBUG)
  ```

  **官方 pip release 版定义了 `NDEBUG`，这段代码被编译掉了**——pip 版设了环境变量也毫无输出。只在 **debug 编译**（`DEBUG=1`）或显式加宏的构建里有效（`torch.__config__.show()` 看 build 配置）。用不了也不亏：profiler 已覆盖 90% 需求。

## 3. rg（ripgrep）

为"在超大代码库里搜文本"而生的命令行工具（Rust 实现）。比 `grep -rn` 快几个数量级、自动跳过 `.gitignore` 和二进制、默认带行号。

源码导航三板斧：

```bash
rg "func: flatten" aten/src/ATen/native/native_functions.yaml   # 精确定位算子定义
rg "Tensor flatten" aten/src/ATen/native/                        # 找 C++ 实现
rg "computeStride" aten/ -g "*.cpp"                              # 只搜 .cpp
```

**搜索词是迭代收窄出来的，不是背的**：命中太多加特征（返回类型/左括号/参数类型），命中太少减特征。例：直接 `rg "flatten"` 在 TensorShape.cpp 有 38 个命中；收窄到 `rg "Tensor flatten"`（返回类型 `Tensor` 来自 yaml 的 `-> Tensor(a)`，函数名来自"无 dispatch 段 → 实现名 = 算子名"规则）→ 5 个命中，参数类型还能反向确认 overload。

替代方案：`grep -rn`（慢但到处都有）；VSCode 全局搜索；无本地代码时 GitHub 网页 code search（限定 `repo:pytorch/pytorch`）。安装：`sudo apt install ripgrep` 或 `conda install -c conda-forge ripgrep`。

## 4. gdb（终极验证）

前面所有手段都是"推断 + 间接观测"，gdb 是**直接看真相**：

```bash
gdb --args python your_script.py
(gdb) break at::native::flatten
(gdb) run
(gdb) bt        # 从 THPVariable_flatten 一路到 computeStride 的完整真实调用栈
```

需要带符号的自编译版本。栈帧就是真相，不依赖任何"我以为"。

> VSCode 里 Python+C++ 双调试器联合调试（一个进程挂 gdb + debugpy）的完整配置与坑位见 [[vscode-python-cpp-联合调试pytorch]]。

## 验证手段的分层

| 无需 build 就能做 | 需要 build / debug 构建 |
|---|---|
| pip 自带头文件（`torch/include/ATen/ops/*.h`）、生成器源码、模板、native kernel、profiler、rg | rg 生成的 .cpp、注册表项原文、dispatch trace、gdb |

分析时先用左列走完全程，右列只作补充实证。

## 相关页面

- [[aten算子调用链定位方法论]] — 这些工具在七步 SOP 中的使用时机
- [[PyTorch-ATen-Dispatcher]] — profiler/trace 能观测 dispatch 的原理（RecordFunction 钩子在查表路径上）
- [[PyTorch-代码生成管线]] — rg 搜索目标的文件地图
