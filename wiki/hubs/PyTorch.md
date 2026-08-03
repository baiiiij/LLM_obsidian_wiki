---
type: hub
title: PyTorch（大类粘合页）
created: 2026-07-31
updated: 2026-07-31
sources: []
tags: [pytorch, hub, 导航]
---

# PyTorch

> 本大类覆盖 **PyTorch 这个领域的一切**：源码机制、算子实现、算法、API 用法、生态。
> **当前内容集中在"源码机制"子线**——一个算子调用从 Python 到 device kernel 中间发生了什么：代码生成、dispatcher 路由、源码分析方法论，以及第三方库（[[FlagGems]]）如何利用这套机制。
> 以后涉及 PyTorch 的实现/算法/API 等新内容，直接归入本大类，在下方新增子线即可。

## 子线：源码机制（当前唯一子线）

主线逻辑——算子的一生，四个阶段：

```
① 代码生成（build 前）      yaml → 生成器 → 各层代码         → [[PyTorch-代码生成管线]]
② 加载注册（import 时）     种子代码 m.impl 把 kernel 写进表  → [[PyTorch-ATen-Dispatcher]]
③ 运行查表（每次调用）       包装 → 查表 → kernel             → [[PyTorch-ATen-Dispatcher]]
④ 运行时改表（第三方覆盖）   FlagGems 等改表项的值            → [[FlagGems]]
```

## 推荐阅读顺序（第一次读）

| 顺序 | 页面 | 为什么在这个位置 |
|---|---|---|
| 1 | [[pytorch-flatten-调用链路定位]] | **先建立直观**：拿 flatten 这个最简单的算子走完全程，看到"点进 .pyi 是死路 → yaml 反查 → kernel 源码"的完整实例；读完想要逐行细节再看 [[pytorch-flatten-全流程逐层详解]] |
| 2 | [[PyTorch-ATen-Dispatcher]] | **再理解机制**：双层模型（算子 vs kernel）、过表 vs 直连、为什么需要 dispatcher、注册表结构——实例里每一步的"为什么"都在这里 |
| 3 | [[PyTorch-代码生成管线]] | **然后建立地图**：每个文件由谁生成、生成到哪；拿到新算子该找哪个文件 |
| 4 | [[PyTorch-源码分析工具箱]] | **备好工具**：profiler / dispatch trace / rg / gdb 的用法与坑 |
| 5 | [[aten算子调用链定位方法论]] | **最后自己动手**：七步 SOP + 完整分析链路，用任意新算子实操 |
| 6 | [[FlagGems]] | **看机制的应用**：第三方库如何不改 PyTorch 一行代码就换掉 700+ 算子的 kernel |

## 页面速查（按需求查）

| 我想…… | 去这页 |
|---|---|
| 理解 dispatch 和内部调用的区别、为什么会有 dispatcher | [[PyTorch-ATen-Dispatcher]] |
| 知道某个文件是谁生成的、build 产物在哪 | [[PyTorch-代码生成管线]] |
| 知道 `t.op` / `torch.op` / `F.op` 分别找哪个绑定文件 | [[PyTorch-代码生成管线]]"判定规则" |
| profiler 表格怎么读、activities 怎么选 | [[PyTorch-源码分析工具箱]] |
| 自己分析一个新算子的完整调用链 | [[aten算子调用链定位方法论]] |
| flatten 什么时候 view 什么时候 copy | [[pytorch-flatten-调用链路定位]] |
| flatten 每个函数内部干什么、内存在哪动、CPU 还是 device | [[pytorch-flatten-全流程逐层详解]] |
| 理解 FlagGems / vLLM 自定义算子怎么接管算子 | [[FlagGems]]、[[PyTorch-ATen-Dispatcher]]"注册表运行时可变" |

## 成员页清单

- 概念：[[PyTorch-ATen-Dispatcher]]、[[PyTorch-代码生成管线]]、[[PyTorch-源码分析工具箱]]
- 问答沉淀：[[aten算子调用链定位方法论]]、[[pytorch-flatten-调用链路定位]]、[[pytorch-flatten-全流程逐层详解]]、[[vscode-python-cpp-联合调试pytorch]]、[[debug与release构建的行为差异]]
- 外部应用：[[FlagGems]]（entity，FlagGems 大类入口）
