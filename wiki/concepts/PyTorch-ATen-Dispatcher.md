---
type: concept
title: PyTorch ATen Dispatcher 与代码生成管线
created: 2026-07-31
updated: 2026-07-31
sources: []
tags: [pytorch, aten, dispatcher, codegen, view]
---

# PyTorch ATen Dispatcher

PyTorch 算子从 Python 调用到 device kernel 的**路由系统**。理解它是阅读一切 aten 算子源码的前提（如 [[pytorch-flatten-调用链路定位]]）。

## 代码生成管线（build-time）

```
aten/src/ATen/native/native_functions.yaml     # ★ 唯一事实来源：每个算子的 schema + dispatch 表
  ├─ tools/pyi/gen_pyi.py                      # → torch/_C/__init__.pyi（纯类型存根，无实现）
  ├─ tools/autograd/gen_python_functions.py    # → torch/csrc/autograd/generated/python_variable_methods.cpp
  │                                            #    （THPVariable_xxx：Python 参数解析 → C++ 调用）
  ├─ torchgen                                  # → ATen/ops/{op}.h（内联包装）、Register{Backend}.cpp
  └─ autograd codegen                          # → Autograd 键上的生成包装（view tracking / backward）
```

**IDE 点跳转到 .pyi 是死路**；定位任何算子都从 native_functions.yaml 反向出发。

## Dispatcher（run-time）

- 每个算子注册为 `aten::op.overload`，存在 `c10::Dispatcher` 的表中；
- 按 **DispatchKeySet**（CPU / CUDA / Autograd / ADInplaceOrView / Functionalize …）从高优先级到低依次查表；
- 调试：`TORCH_SHOW_DISPATCH_TRACE=1` 打印每次 dispatch 经过的 op 与 key。

## native_functions.yaml 的三种实现形态

| 形态 | 含义 | 例 |
|---|---|---|
| `dispatch: CPU: foo_cpu / CUDA: foo_cuda` | 独立 per-backend kernel | add、matmul |
| `CompositeImplicitAutograd: foo` | 一份 C++ 实现服务全部后端，autograd 自动包 | reshape |
| `structured_delegate: bar` | meta 算 shape 后整体委托另一算子 | flatten → reshape |

## View vs Copy：元数据算子的分水岭

reshape/view/flatten/permute 这类**不改数据的算子**，分水岭在 `computeStride`：

- 目标 shape 能由现有 (sizes, strides) 的连续段拼出（`stride[i] == stride[i+1]*size[i+1]`；size==1 自由）→ **view**：`as_strided`/`_reshape_alias` 只改 TensorImpl 的 sizes/strides/storage_offset，共享 storage，0 拷贝；
- 拼不出 → **copy**：`clone/contiguous` dispatch 到 device copy kernel，再对拷贝结果 view。

view 的 autograd 通过 `ADInplaceOrView` key 上的生成包装记录 ViewInfo，backward 时按逆变换回传梯度；因此 view 之后的 in-place 修改会触发 autograd 报错，而 copy 不会。

## 与知识库主线的关联

框架层（[[vLLM]] 等）调 aten 算子时同样经过此管线；自定义算子（`TORCH_LIBRARY`）也是向 Dispatcher 注册 key。性能分析时，[[Grouped GEMM]] 前后的 layout 整理（如 [[moe_align_block_size]]）是否产生隐式 copy，可用 profiler 中 `aten::clone/copy_` 是否出现来判定。
