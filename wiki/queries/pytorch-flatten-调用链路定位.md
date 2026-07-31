---
type: query
title: 如何定位 torch.Tensor.flatten 的源码调用链路（view vs copy 判断）
created: 2026-07-31
updated: 2026-07-31
sources: []
tags: [pytorch, aten, dispatcher, view, codegen]
---

# 问题

在 IDE 里对 `tensor.flatten()` 点跳转到 `torch/_C/__init__.pyi`，只能看到 `@overload` 类型存根，看不到实现。如何从 PyTorch 源码找到完整的调用链路？底层怎么 dispatch？什么情况下复制数据、什么情况下只改 metadata？

# 回答

## 为什么 .pyi 里看不到实现

`torch/_C/__init__.pyi` 是 `tools/pyi/gen_pyi.py` **自动生成的类型存根**，只服务于类型检查/IDE 补全，不包含任何实现。真正的 Python 绑定、C++ 前端包装、kernel 注册都是**构建期由 torchgen 代码生成**的 C++。所以"点进去"这条路对 aten 算子天然走不通，要从 YAML 反向定位。

## 通用定位法：native_functions.yaml 是一切的原点

```bash
# 1. 找到算子定义
rg "func: flatten" aten/src/ATen/native/native_functions.yaml
```

会看到（以本地版本为准）：

```yaml
- func: flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
  structured_delegate: reshape
  variants: function, method
```

三个关键字段决定下游走向：

| 字段 | 含义 |
|---|---|
| `structured_delegate: reshape` | flatten **不是独立 kernel**，codegen 生成桥接代码：meta 函数算出目标 shape，整体委托给 reshape |
| `CompositeImplicitAutograd: xxx` | 一份 C++ 实现服务所有后端（autograd 由 codegen 自动包） |
| `dispatch: CPU: ... / CUDA: ...` | 有独立 per-backend kernel 时才会出现 |

`.pyi` 里带字符串维度的 overload 来自 Dimname 变体（`flatten.using_names` 等 named tensor 路径）。

## flatten 的完整调用链

```
t.flatten(0, 1)                                    # Python
  → THPVariable_flatten                            # torch/csrc/autograd/generated/python_variable_methods.cpp
    （tools/autograd/gen_python_functions.py 生成，PythonArgParser 解析参数）
  → at::_ops::flatten::call()                      # ATen/ops/flatten.h（生成的内联包装）
  → Dispatcher: "aten::flatten.using_ints"         # aten/src/ATen/core/dispatch/Dispatcher.h
    （按 DispatchKeySet 查表：Autograd/ADInplaceOrView → backend）
  → structured_delegate 桥接：算目标 shape → 调 reshape
  → aten::reshape（CompositeImplicitAutograd）
  → at::native::reshape                            # aten/src/ATen/native/TensorShape.cpp
      ├─ computeStride(sizes, strides, new_shape) 成功
      │    → _reshape_alias / as_strided           # ★ view：共享 storage，0 拷贝
      └─ 失败
           → clone/contiguous → view               # ★ 拷贝：dispatch 到 device 的 copy kernel
```

注意：view 路径上的 autograd 处理不是单独 kernel，而是 `ADInplaceOrView` key 上的生成包装（记录 ViewInfo 供 backward 用）。

## 复制 vs 只改 metadata：computeStride 的判断条件

reshape 的核心是一段**贪心双指针匹配**（`computeStride`，rg 定位具体文件，版本间位置略有移动）：

- 目标 shape 的每个维度，必须能由旧 (sizes, strides) 中**内存连续的一段**构成：即合并相邻维度要求 `stride[i] == stride[i+1] * size[i+1]`；
- `size == 1` 的维度 stride 任意，自由通过；
- 全部匹配成功 → 返回新 strides → view（`_reshape_alias`/`as_strided`，只改 sizes/strides/offset，storage 共享）；
- 任何一段匹配不上 → 返回 nullopt → `clone/contiguous` 拷贝成连续布局再 view。

**flatten 的特例**：它只合并 `[start_dim, end_dim]` 一段维度，所以 view 的充要条件直观来说就是——**这段维度在内存里连续**（或其中含 size 0/1 的维）。contiguous tensor 必然 view；transpose 过的 tensor 合并被交换的维时必然拷贝。

## 快速实证工具

```python
t = torch.randn(2, 3, 4)
t.flatten(0, 1)._base is t            # True → view
t.transpose(0, 1).flatten(0, 1)._base # None → 发生了拷贝
```

```bash
# 看 dispatcher 实际经过哪些 op / dispatch key
TORCH_SHOW_DISPATCH_TRACE=1 python -c "import torch; torch.randn(2,3,4).flatten(0,1)"
```

```python
# profiler：出现 aten::clone / aten::copy_ 说明拷贝；只有 aten::view/as_strided 说明是 view
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:
    t.transpose(0, 1).flatten(0, 1)
print(prof.key_averages().table())
```

## 相关页面

- [[PyTorch-ATen-Dispatcher]] — codegen 管线与 dispatch 机制总览
