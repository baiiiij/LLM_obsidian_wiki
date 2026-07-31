---
type: query
title: torch.Tensor.flatten 调用链路全解（view vs copy 判断，对照 2.14 源码实测）
created: 2026-07-31
updated: 2026-07-31
sources: []
tags: [pytorch, aten, dispatcher, view, codegen]
---

# 问题

在 IDE 里对 `tensor.flatten()` 点跳转到 `torch/_C/__init__.pyi`，只能看到 `@overload` 类型存根，看不到实现。如何找到完整调用链路？底层怎么 dispatch？什么情况下复制数据、什么情况下只改 metadata？

> **版本基准：本文行号对照 GitHub `pytorch/pytorch` 的 `release/2.12` 分支在线核对**。与本地 2.14.0a0 树结构一致，仅行号略有偏移。

# 〇、先建立地图：PyTorch 代码的三个世界

看 PyTorch 源码最大的困惑是"文件到底在哪"。其实只有三类：

| 世界                   | 说明                | 例子                                                       |
| -------------------- | ----------------- | -------------------------------------------------------- |
| ① 源码自带               | git clone 就有，直接可读 | `aten/src/ATen/native/TensorShape.cpp`                   |
| ② build 生成·写回源码树     | 编译时生成，但落在源码目录里    | `torch/_C/__init__.pyi`、`torch/csrc/autograd/generated/` |
| ③ build 生成·落在 build/ | 编译时生成，只在 build 目录 | `build/aten/src/ATen/ops/flatten.h`                      |

**你点进 `.pyi` 看不到实现，是因为它属于世界②**——`tools/pyi/gen_pyi.py`（世界①）在编译时生成的纯类型存根，只服务 IDE 补全，从来不含实现。

**一切算子的原点只有一个文件**（世界①）：

```
aten/src/ATen/native/native_functions.yaml
```

第 2702 行：

```yaml
- func: flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
  variants: function, method
```

注意：它**没有 `dispatch:` 段**。在 torchgen 的规则里，没有 dispatch 段 = 默认 `CompositeImplicitAutograd`，即"一份 C++ 实现服务所有后端，autograd 由代码生成自动包一层"，实现函数按名字约定就是 `at::native::flatten`。（旧版本这里曾写 `structured_delegate: reshape`，2.12+ 已改为直接实现。）

# 一、完整调用链（五层，自上而下）

```
t.flatten(0, 1)                                    ← Python
│
├─ ① THPVariable_flatten                          [build 生成·源码树内]
│     torch/csrc/autograd/generated/python_variable_methods.cpp
│     （由 tools/autograd/gen_python_functions.py 读取 yaml 生成；
│       用 PythonArgParser 解析参数，然后调 C++ 接口）
│
├─ ② Tensor::flatten / at::flatten                [build 生成·build/ 目录]
│     build/aten/src/ATen/core/TensorBody.h       （模板：aten/src/ATen/templates/TensorBody.h）
│     build/aten/src/ATen/ops/flatten.h           （内联包装：转调 _ops::flatten_using_ints::call）
│
├─ ③ Dispatcher 查表                              [源码自带]
│     aten/src/ATen/core/dispatch/Dispatcher.h
│     按 tensor 的 DispatchKeySet 查 "aten::flatten.using_ints" 注册的 kernel：
│     先过生成的 autograd/view 包装（VariableType / ADInplaceOrViewType，
│     位于 torch/csrc/autograd/generated/，记录 view 关系供 backward），
│     再落到 CompositeImplicitAutograd kernel —— at::native::flatten
│     （注册表本身也是生成的：build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp）
│
├─ ④ at::native::flatten                          [源码自带] ← 真正的逻辑从这里开始
│     aten/src/ATen/native/TensorShape.cpp:4178
│     拼出目标 shape 后【直接函数调用】native::reshape_symint（同文件:2058）
│     ⚠️ 这一步是 C++ 直接调用，不再走 dispatcher！
│
└─ ⑤ view / copy 分水岭：reshape_symint           [源码自带]
      aten/src/ATen/native/TensorShape.cpp:2058
```

# 二、第四层：flatten 本体（TensorShape.cpp:4178，源码自带）

```cpp
Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim) {
  start_dim = maybe_wrap_dim(start_dim, self.dim());  // 负数维度转正（-1 → dim-1）
  end_dim = maybe_wrap_dim(end_dim, self.dim());
  TORCH_CHECK(start_dim <= end_dim, ...);

  if (self.dim() == 0)  return self.reshape({1});   // 0 维张量 → shape [1]
  if (start_dim == end_dim) return self;            // 合并单个维 = 啥也没干，直接返回自己

  // 把 [start_dim, end_dim] 一段维度的乘积算出来，拼成新 shape
  auto slice_numel = c10::multiply_integers(
      self.sym_sizes().slice(start_dim, end_dim - start_dim + 1));
  // ... 前段 dims + slice_numel + 后段 dims ...
  return native::reshape_symint(self, shape);       // ★ 直接调用，不经过 dispatcher
}
```

所以 flatten 自己**没有任何复制/view 判断**，它只是"算目标 shape"，决策全部交给 reshape。这也解释了为什么 yaml 里没有 dispatch 段——它根本不需要 per-device kernel。

# 三、第五层：view 还是 copy？（TensorShape.cpp:2058，源码自带）

```cpp
Tensor reshape_symint(const Tensor& self, c10::SymIntArrayRef proposed_shape) {
  if (self.is_contiguous_or_false() && !self.is_mkldnn()) {
    return self.view_symint(proposed_shape);        // 快路径：连续 ⇒ 必然可以 view
  }
  c10::SymDimVector shape = infer_size_dv(proposed_shape, self.sym_numel());  // 处理 -1
  ...
  auto stride = at::detail::computeStride(sym_sizes, sym_strides, shape);     // ★ 分水岭

  if (stride.has_value()) {                          // 情况 A：可以 view
    return self._reshape_alias_symint(shape, stride.value());   // → 只改 metadata
  }
  return at::_unsafe_view_symint(                    // 情况 B：必须复制
      self.clone(at::MemoryFormat::Contiguous), shape);         // ← clone 才拷数据
}
```

三个分支的归宿：

## A-1. 快路径 `view_symint` / A-2. `_reshape_alias` —— 只改 metadata

`_reshape_alias`（TensorShape.cpp:2168）调 `alias_with_sizes_and_strides`（**TensorShape.cpp:1992**），它的函数体就是"只改 metadata"这句话的字面实现：

```cpp
Tensor self_ = at::detail::make_tensor<TensorImpl>(
    c10::TensorImpl::VIEW,
    Storage(self.storage()),      // ★ 共享同一块 storage，零拷贝
    self.key_set(), self.dtype());
self_tmp_->set_sizes_and_strides(sizes, strides, self.storage_offset());  // ★ 只写元数据
return self_;
```

新建一个 TensorImpl，指向**同一个 Storage**，只写入新的 sizes/strides/offset。没有任何数据移动。

## B. `clone(Contiguous)` —— 真正复制

`self.clone(...)` 是一次新的 dispatcher 调用（`aten::clone`），落到 device 专属 copy kernel（CPU memcpy / CUDA copy kernel）。克隆出一块连续的新内存后，再对它 `_unsafe_view`——连续张量必然可以 view。

# 四、computeStride 的判断条件（TensorUtils.cpp:327 起，源码自带）

`at::detail::computeStride`（三个重载在 TensorUtils.cpp:409/417/425）回答一个问题：**"新 shape 能不能用旧的 stride 体系表达出来？"** 算法是从最后一维往前走的双指针：

1. 把旧维度从后往前切成若干"**连续块**"：块内要求 `stride[d-1] == (块内累计 numel) × base_stride`（即相邻维在内存里无缝衔接）；`size == 1` 的维随便跳过；
2. 每切出一块，尝试用新 shape 末尾的若干维恰好填满它（`view_numel == tensor_numel`）；
3. **填不满 → 返回 `std::nullopt` → 走 clone 拷贝**；全部填满 → 返回算好的新 strides → view。

**翻译到 flatten 的场景**：flatten 只是把 `[start_dim, end_dim]` 合并成一维，所以结论是——

> **被合并的这段维度在内存中连续（`stride[i] == stride[i+1] × size[i+1]`），或其中含 size 为 0/1 的维 ⇒ view；否则 ⇒ copy。**

contiguous 张量必然 view；`transpose` 之后再合并被交换的那两维，必然 copy。

# 五、实证工具（30 秒验证，不用读代码）

```python
t = torch.randn(2, 3, 4)
t.flatten(0, 1)._base is t            # True  → view（_base 指向原张量）
t.transpose(0, 1).flatten(0, 1)._base # None  → 发生了拷贝
```

```bash
# 看 dispatcher 实际经过哪些 op / dispatch key
TORCH_SHOW_DISPATCH_TRACE=1 python -c "import torch; torch.randn(2,3,4).flatten(0,1)"
```

```python
# profiler：出现 aten::clone / aten::copy_ 就是拷贝路径；只有 view/as_strided 就是 view
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:
    t.transpose(0, 1).flatten(0, 1)
print(prof.key_averages().table())
```

# 六、文件速查表（含"什么时候才存在"）

| 文件（相对 pytorch 源码树根） | 存在时机 | 角色 |
|---|---|---|
| `aten/src/ATen/native/native_functions.yaml` | 源码自带 | 算子 schema 唯一原点（flatten 在 :2702） |
| `aten/src/ATen/native/TensorShape.cpp` | 源码自带 | flatten :4178 / reshape_symint :2058 / _reshape_alias :2168 / alias_with_sizes_and_strides :1992 |
| `aten/src/ATen/TensorUtils.cpp` | 源码自带 | computeStride_impl :327，重载 :409/417/425 |
| `aten/src/ATen/core/dispatch/Dispatcher.h` | 源码自带 | dispatcher 本体 |
| `tools/pyi/gen_pyi.py` | 源码自带 | 生成你点进去看到的 .pyi |
| `tools/autograd/gen_python_functions.py` | 源码自带 | 生成 Python 绑定的脚本 |
| `tools/autograd/templates/python_variable_methods.cpp` | 源码自带（模板） | 绑定代码的模板 |
| `aten/src/ATen/templates/TensorBody.h` | 源码自带（模板） | Tensor 方法的模板 |
| `torch/_C/__init__.pyi` | **build 生成**（写回源码树） | 类型存根（无实现，你点进去的那个） |
| `torch/csrc/autograd/generated/python_variable_methods.cpp` | **build 生成**（写回源码树） | THPVariable_flatten 等 Python 绑定 |
| `torch/csrc/autograd/generated/VariableType*.cpp`、`ADInplaceOrViewType*.cpp` | **build 生成**（写回源码树） | autograd / view 追踪包装 |
| `build/aten/src/ATen/core/TensorBody.h` | **build 生成**（build 目录） | Tensor::flatten 方法定义 |
| `build/aten/src/ATen/ops/flatten.h` | **build 生成**（build 目录） | `at::flatten` 内联包装 + `_ops::flatten_using_ints` |
| `build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp` | **build 生成**（build 目录） | 把 `at::native::flatten` 注册进 dispatcher |

> build 产物的安装目录由 `cmake/Codegen.cmake:213` 的 `--install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen` 决定；Python 绑定的输出目录约定见 `tools/autograd/gen_autograd.py` 头部注释（`torch/csrc/autograd/generated/`）。

# 七、一句话总结

> flatten 本身只是"算 shape"（TensorShape.cpp:4178），决策在 reshape（:2058）：`computeStride` 能算出新 strides 就 `_reshape_alias` 共享 storage 只改 metadata；算不出就 `clone` 连续化再 view，clone 才是唯一真正搬数据的地方。

## 相关页面

- [[aten算子调用链定位方法论]] — 本文的"上游"：不凭先验经验推出这条链的七步 SOP
- [[PyTorch-ATen-Dispatcher]] — codegen 管线与 dispatch 机制总览
