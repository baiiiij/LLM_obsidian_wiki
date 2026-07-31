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
  ├─ tools/pyi/gen_pyi.py                      # → torch/_C/__init__.pyi（build 生成·写回源码树；纯类型存根，无实现）
  ├─ tools/autograd/gen_python_functions.py    # → torch/csrc/autograd/generated/python_variable_methods.cpp
  │   （模板 tools/autograd/templates/*.cpp）   #    （build 生成·写回源码树；THPVariable_xxx：参数解析 → C++ 调用）
  ├─ tools/autograd（autograd codegen）         # → torch/csrc/autograd/generated/VariableType*.cpp、ADInplaceOrViewType*.cpp
  └─ torchgen                                  # → build/aten/src/ATen/：core/TensorBody.h、ops/{op}.h、
                                               #    Register{Backend}.cpp、RegisterCompositeImplicitAutograd.cpp
```

**IDE 点跳转到 .pyi 是死路**；定位任何算子都从 native_functions.yaml 反向出发。生成产物的两类落点：写回源码树（`torch/_C/`、`torch/csrc/autograd/generated/`）与只存在于 build 目录（`build/aten/src/ATen/`，由 `cmake/Codegen.cmake` 的 `--install_dir` 决定）。

## Dispatcher（run-time）

- 每个算子注册为 `aten::op.overload`，存在 `c10::Dispatcher` 的表中；
- 按 **DispatchKeySet**（CPU / CUDA / Autograd / ADInplaceOrView / Functionalize …）从高优先级到低依次查表；
- 调试：`TORCH_SHOW_DISPATCH_TRACE=1` 打印每次 dispatch 经过的 op 与 key。

## native_functions.yaml 的三种实现形态

| 形态 | 含义 | 例 |
|---|---|---|
| `dispatch: CPU: foo_cpu / CUDA: foo_cuda` | 独立 per-backend kernel | add、matmul |
| 无 `dispatch:` 段 / `CompositeImplicitAutograd: foo` | 一份 C++ 实现服务全部后端，autograd 自动包 | flatten（2.12+）、reshape |
| `structured_delegate: bar` | meta 算 shape 后整体委托另一算子 | 旧版本的 flatten（2.0 前后曾委托 reshape，后改为直接实现） |

> 注意版本漂移：同名算子在不同版本的 yaml 形态可能不同，以本地源码树的 `native_functions.yaml` 为准。composite 算子之间经常**直接 C++ 函数调用**（如 flatten 直接调 `native::reshape_symint`），不再走 dispatcher——读代码时不要想当然地以为每次调用都过 dispatch 表。

## View vs Copy：元数据算子的分水岭

reshape/view/flatten/permute 这类**不改数据的算子**，分水岭在 `computeStride`（`aten/src/ATen/TensorUtils.cpp`）：

- 目标 shape 能由现有 (sizes, strides) 的连续段拼出（`stride[i] == stride[i+1]*size[i+1]`；size==1 自由）→ **view**：`_reshape_alias` → `alias_with_sizes_and_strides`（TensorShape.cpp）新建一个 TensorImpl、**共享同一 Storage**、只 `set_sizes_and_strides` 写元数据，0 拷贝；
- 拼不出 → **copy**：`clone(Contiguous)` dispatch 到 device copy kernel 真正搬数据，再对拷贝结果 `_unsafe_view`。

view 的 autograd 通过 `ADInplaceOrView` key 上的生成包装记录 ViewInfo，backward 时按逆变换回传梯度；因此 view 之后的 in-place 修改会触发 autograd 报错，而 copy 不会。

## 与知识库主线的关联

框架层（[[vLLM]] 等）调 aten 算子时同样经过此管线；自定义算子（`TORCH_LIBRARY`）也是向 Dispatcher 注册 key。性能分析时，[[Grouped GEMM]] 前后的 layout 整理（如 [[moe_align_block_size]]）是否产生隐式 copy，可用 profiler 中 `aten::clone/copy_` 是否出现来判定。
