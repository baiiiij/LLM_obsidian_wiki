---
type: query
title: VSCode Python+C++ 联合调试自编译 PyTorch（从 flatten 单步进 C++）
created: 2026-08-03
updated: 2026-08-03
sources: []
tags: [pytorch, vscode, gdb, debugpy, 调试]
---

# 问题

远程机器上有自编译的 PyTorch 源码，如何在 VSCode 里 Python 和 C++ 联合调试？例如：在 Python 里写 `xxx.flatten()`，单步调试进 C++ 实现。

# 〇、核心思路

**一个 python 进程，同时挂两个调试器**：

- **C++ 调试器**（cppdbg / gdb）：管原生层（libtorch、ATen、Dispatcher）
- **Python 调试器**（debugpy）：管 Python 层

两个 session 并存，VSCode 侧栏顶部切换：Python session 看 Python 调用栈和变量，C++ session 看原生调用栈——正好对照 [[pytorch-flatten-调用链路定位]] 的五层链路逐层验证。

# 一、前提：构建必须带符号

```bash
# 二选一
DEBUG=1 python setup.py develop              # -O0，单步丝滑，跑得慢，编译更久
REL_WITH_DEB_INFO=1 python setup.py develop  # 带符号但有优化：变量常 <optimized out>、单步乱跳
```

验证：`python -c "import torch; print(torch.__config__.show())"`。

> **pip 官方版 stripped + NDEBUG，断点永远 pending**，必须自编译。看控制流逻辑用 DEBUG=1；RelWithDebInfo 只适合"大概看栈"。debug 版看到的流程和 release 是否一致？一致——见 [[debug与release构建的行为差异]]。

# 二、推荐工作流：C++ 启动 python + Python attach

`launch.json` 两个配置：

```jsonc
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "C++: python+debugpy",
      "type": "cppdbg",
      "request": "launch",
      "program": "/path/to/venv/bin/python",   // ← python 解释器本身就是 C++ 调试目标
      "args": ["-m", "debugpy", "--listen", "5678", "--wait-for-client", "${file}"],
      "cwd": "${workspaceFolder}",
      "MIMode": "gdb",
      "stopAtEntry": false,
      "setupCommands": [
        { "text": "set breakpoint pending on" },  // ← 关键：动态库未加载时断点先生效
        { "text": "set pagination off" }
      ]
    },
    {
      "name": "Python: attach",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "127.0.0.1", "port": 5678 },
      "justMyCode": false
    }
  ]
}
```

**操作顺序**（以 flatten 为例）：

1. 启动 **C++: python+debugpy** → gdb 拉起 python 进程，停在等待 debugpy 客户端；
2. 启动 **Python: attach** → debugpy 连上，脚本开始跑；
3. 脚本跑过 `import torch` → `libtorch.so` 等动态库加载完成；
4. 在 `aten/src/ATen/native/TensorShape.cpp` 的 `at::native::flatten` 打断点；
5. 继续执行到 `t.flatten(0, 1)` → **命中 C++ 断点**。

C++ session 里 `bt` 能看到完整真实调用栈：`THPVariable_flatten` → Dispatcher → `VariableType`（autograd/view 包装）→ `at::native::flatten` → `reshape_symint`。

# 三、备选：Python 正向 launch + C++ attach

debugpy 正常 launch 脚本，在 `import torch` 之后用 Python 断点暂停，然后 C++ 配置 attach：

```jsonc
{
  "name": "C++: attach python",
  "type": "cppdbg",
  "request": "attach",
  "processId": "${command:pickProcess}",
  "MIMode": "gdb"
}
```

适合想精确控制 attach 时机的场景。嫌两个配置麻烦可装第三方扩展 **Python C++ Debugger**（benjamin-simmonds.pythoncpp-debug），合成一个配置。

# 四、关键坑（按踩坑概率排序）

| 坑 | 现象 | 解法 |
|---|---|---|
| 无符号/NDEBUG | 断点永远 pending、栈帧全是 `??` | 必须 debug 构建，`torch.__config__.show()` 确认 |
| **pending breakpoint** | `import torch` 前打的断点灰色"模块未加载" | `set breakpoint pending on`；或先跑过 import 再设断点 |
| 优化干扰 | 单步乱跳、变量 `<optimized out>` | RelWithDebInfo 的通病，看逻辑换 DEBUG=1 |
| 断点打错层 | 以为 flatten 有 per-device kernel | flatten 是 CompositeImplicitAutograd（无 dispatch 段），全后端共用 `at::native::flatten`；绑定层断点是 `THPVariable_flatten`（生成的 `python_variable_methods.cpp`，写回源码树） |
| 远程机器 | 本地 VSCode 挂不到远程 gdb | **Remote-SSH** 连编译机，workspace 直接开 pytorch 源码树，体验与本地一致；路径不一致才需 `sourceFileMap` |
| C++ 跳转/补全 | VSCode 里跳不动 | 构建加 `CMAKE_EXPORT_COMPILE_COMMANDS=ON`，配 clangd 或 `C_Cpp.default.compileCommands` |
| CUDA kernel | gdb 断不进 device 代码 | 需要 cuda-gdb；flatten 是 CPU metadata 算子，普通 gdb 足够 |

# 五、配合技巧

- C++ session 的 **DEBUG CONSOLE** 直接发 gdb 命令：`-exec bt`、`-exec thread apply all bt`（多线程全栈）、`-exec info sharedlibrary`（确认 libtorch 已加载）。
- 看 Tensor 内部：手翻 `impl_.sizes_and_strides_` / `storage_`（c10 类型没有 pretty printer）。
- 联合调试是"重武器"，轻量验证先用 [[PyTorch-源码分析工具箱]] 的 profiler / dispatch trace（注意 trace 同样需要 debug 构建，NDEBUG 会编译掉打印代码）。

# 六、一句话总结

> debug 构建是门票；`cppdbg` 以 python 解释器为 program 启动、脚本里套 `debugpy --listen --wait-for-client`，再用 debugpy attach——一个进程挂两个调试器，断点记得 `set breakpoint pending on`。

## 相关页面

- [[pytorch-flatten-调用链路定位]] — 本文调试目标的五层调用链（断点该打在哪几层）
- [[debug与release构建的行为差异]] — debug 版看到的流程和 release 一样吗？（一样，NDEBUG 只删诊断代码）
- [[PyTorch-源码分析工具箱]] — gdb 的轻量替代：profiler / dispatch trace
- [[aten算子调用链定位方法论]] — 断点走法对应的七步 SOP
- [[PyTorch-ATen-Dispatcher]] — 调用栈中 VariableType / Dispatcher 各帧的机制背景
