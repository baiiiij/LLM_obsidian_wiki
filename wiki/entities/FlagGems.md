---
type: entity
title: FlagGems
created: 2026-07-31
updated: 2026-07-31
sources: []
tags: [pytorch, triton, 算子库, dispatcher, 覆盖机制]
---

# FlagGems

> 所属层次：算子层。智源 FlagOS 生态的 **Triton 算子库**（[github.com/flagos-ai/FlagGems](https://github.com/flagos-ai/FlagGems)）：用 [[Triton]] 实现 700+ 个 PyTorch 兼容算子，目标是"一次开发，多芯运行"——通过**运行时覆盖 aten 算子的 kernel**，让模型代码不改一行就切换到 Triton 实现。

## 核心机制：运行时改表，不截断 dispatch

FlagGems **不修改 PyTorch 的任何文件/生成代码**，而是利用 dispatcher 注册表运行时可变的特性（机制详见 [[PyTorch-ATen-Dispatcher]]"注册表是运行时可变的"）：

```python
aten_lib = torch.library.Library("aten", "IMPL")   # 拿到 aten 命名空间的注册句柄

class use_gems:                                     # 用户入口：with flag_gems.use_gems():
    def __enter__(self): ... enable(lib=self.lib, ...)   # 批量注册 Triton kernel
    def __exit__(self):  ... self.lib._destroy()         # 注销，恢复原 kernel
```

`enable()` 让 Registrar 遍历 `_FULL_CONFIG`——**700+ 条 `(overload名 → Triton 实现)` 对照表**（如 `("mm", mm)`、`("sum.dim_IntList", sum_dim)`、`("_softmax", softmax)`），逐项注册到**当前设备的 backend key**（这也是它支持 10+ 后端的原理：按设备 key 注册）。

**"不截断 dispatch"的精确含义**：查表照常运行，Autograd 包装、profiler 钩子全部照常触发——只是表里 `(aten::mm, CUDA)` 这一项的**值**从原版 CUDA kernel 换成了 Triton kernel。退出 `use_gems()` 时 `_destroy()` 注销注册、重算表，恢复原状。

## 覆盖的三条边界（双层模型的直接推论）

1. **只拦过表的调用**：kernel 内部直连（如 flatten→`native::reshape_symint`）感知不到覆盖 → FlagGems 只替换 mm/softmax/layernorm 等总被上层直接 dispatch 的"叶子算子"；
2. **按 key 粒度**：注册在 CUDA key 只影响 CUDA tensor，同一份代码跑 CPU 仍走原版 kernel；
3. **autograd 层不受影响**：Autograd key 优先级高于 backend key 先命中；backward 由单独注册的 `_backward` 变体（如 `silu_backward`）接管。

## 一个佐证细节：注册名单本身就是教科书

`_FULL_CONFIG` 里**没有** flatten/view/reshape——纯 view 算子不搬数据，没有可加速的东西；但注册了 `_unsafe_view`、`view_copy`、`alias_copy`、`as_strided_copy` 这些**真正搬数据**的变体。这正好印证 [[PyTorch-ATen-Dispatcher]] 的 view vs copy 分水岭：flatten 拷贝路径里的 `at::_unsafe_view` 是过表的，恰好可被 FlagGems 拦到。

## 实证方法

```python
import torch, flag_gems
from torch.profiler import profile, ProfilerActivity

a = torch.randn(512, 512, device='cuda')
# 覆盖前：aten::mm 下 Self CUDA 是 cuBLAS kernel 名
with flag_gems.use_gems():
    # 覆盖后：aten::mm 仍在（算子名不变！），但 Self CUDA 变成 Triton kernel 名
    ...
```

关键观察点：**profiler 里的 `aten::mm` 事件名不变**（算子层未动），变化的是 CUDA 列的 kernel 名（kernel 层被换）——这就是"双层模型"下覆盖机制的直接可视化。

## 与知识库主线的关联

- 机制基础：[[PyTorch-ATen-Dispatcher]]（注册表结构、覆盖的两种形态：同 key 替换 vs 更具体 key 遮蔽 composite fallback）
- 同类机制用户：后端厂商（PrivateUse1 注册整个 backend）、[[vLLM]] / torch.compile 的自定义算子（`TORCH_LIBRARY` 新 namespace）
- 与本库 [[moe_align_block_size]] 等 vLLM MoE 路径对照：vLLM 走"自研 CUDA/Triton kernel + 自定义算子注册"，FlagGems 走"覆盖既有 aten 算子"，是算子替换的两种路线
