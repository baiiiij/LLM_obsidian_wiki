---
type: query
title: x.flatten() 从 Python 到 C++/device 的全链路逐层详解（含每个函数内部）
created: 2026-08-03
updated: 2026-08-03
sources: []
tags: [pytorch, aten, dispatcher, view, codegen, 调用链]
---

# 问题

把 `x.flatten()` 从 Python 端到 C++ 再到 device 的**全部**调用流程讲清楚：每一步走到哪个分支、这个函数内部干什么、有什么内存操作、在 CPU 还是 device 上执行——相当于把整条链路的代码整体详细解释一遍。

> **版本基准：GitHub `pytorch/pytorch` `release/2.12`，本文所有行号与代码片段均对照该分支在线核对。** 前置阅读：[[pytorch-flatten-调用链路定位]]（同一条链的"定位版"，含文件归属地图）；机制背景见 [[PyTorch-ATen-Dispatcher]] 与 [[PyTorch-代码生成管线]]。

# 全景图

```
【L0 Python】t.flatten(0, 1)
  │ CPython 属性查找：type(t) = THPVariableType → tp_methods = variable_methods[]
  ▼
【L1 Python 绑定（生成代码）】THPVariable_flatten
  torch/csrc/autograd/generated/python_variable_methods.cpp
  │ 解包 → PythonArgParser 解析参数 → __torch_function__ 检查 → 释放 GIL → self.flatten() → wrap 回 PyObject
  ▼
【L2 C++ 方法包装（生成）】Tensor::flatten        build/aten/src/ATen/core/TensorBody.h
  │ 一行转发：at::_ops::flatten_using_ints::call(*this, start_dim, end_dim)
  ▼
【L3 算子句柄（生成）】_ops::flatten_using_ints::call   build/aten/src/ATen/ops/flatten_ops.h
  │ 静态 TypedOperatorHandle（findSchemaOrThrow("aten::flatten","using_ints")）→ Dispatcher::call
  ▼
【L4 Dispatcher 查表】aten/src/ATen/core/dispatch/Dispatcher.h:778
  │ 算 DispatchKeySet → dispatchTable_[highestPriorityKey] → 命中链（4 跳）：
  │   AutogradCPU（VariableType 包装）→ ADInplaceOrView（view 专属包装）
  │   → BackendSelect（设备检查/DeviceGuard）→ CPU 槽 → alias fallback
  │   → CompositeImplicitAutograd wrapper → at::native::flatten
  ▼
【L5 kernel 本体（手写）】at::native::flatten     TensorShape.cpp:4178
  │ 纯整数运算拼目标 shape；直连 native::reshape_symint（★不过表）
  ▼
【L6 reshape_symint】TensorShape.cpp:2058  ← view/copy 分水岭
  ├─ 连续 → view_symint（过表）→ at::native::view → view_impl → alias_with_sizes_and_strides 【只改 metadata】
  ├─ 非连续但 computeStride 成功 → _reshape_alias_symint（过表）→ 同上 【只改 metadata】
  └─ computeStride 失败 → clone(Contiguous)（过表，★唯一搬数据处）→ _unsafe_view_symint（过表）→ alias
```

# L0 Python 侧：方法从哪来

`t.flatten` 的属性查找发生在 CPython 层，没有任何 Python 级包装（不像 `F.flatten` 会经过 `torch/nn/functional.py`）：

- `torch.Tensor` 的 C 类型是 `THPVariableType`（`torch/csrc/autograd/python_variable.cpp:3671`），在模块初始化时注册为 `torch._C.TensorBase`；
- `THPVariable_initModule`（python_variable.cpp:3957 起）把 `tp_methods` 赋值为 `variable_methods[] + extra_methods`；
- `variable_methods[]` 定义在生成的 `python_variable_methods.cpp`——yaml 里 `variants` 含 `method` 的每个算子一项：`{"flatten", castPyCFunctionWithKeywords(THPVariable_flatten), METH_VARARGS|METH_KEYWORDS, ...}`；
- 所以 `t.flatten(0,1)` = CPython 在类型方法表里找到这个 PyCFunction 并直接调用。

**内存/设备**：纯 host 的 PyObject 查找，无任何数据接触。

# L1 THPVariable_flatten（生成的 Python 绑定）

由 `tools/autograd/gen_python_functions.py` 的 `PY_VARIABLE_METHOD_VARARGS(_SINGLETON)` 模板生成。函数体逐步：

1. **`HANDLE_TH_ERRORS`**——try/catch 宏，把整个函数体包起来，C++ 异常翻译成 Python 异常（`END_HANDLE_TH_ERRORS` 收尾）；
2. **`const Tensor& self = THPVariable_Unpack(self_);`**——读 THPVariable 结构体的 `cdata` 字段拿到 `at::Tensor`，零开销，不拷贝 TensorImpl；
3. **`static PythonArgParser parser({"flatten(int64_t start_dim=0, int64_t end_dim=-1)"}, /*traceable=*/true);`**——static 局部变量，签名表只构建一次，之后每次调用复用；
4. **`parser.parse(self_, args, kwargs, parsed_args)`**——按 schema 匹配位置/关键字参数，做类型转换（PyLong→int64_t）、填默认值；多 overload 时返回 `r.idx` 选分支，`flatten` 的 method 形态只有 `using_ints` 一个签名，走单分支模板；
5. **`check_has_torch_function`**——分支点①：若 tensor 是带 `__torch_function__` 的子类，转 `handle_torch_function` 走 Python 覆盖逻辑，本链路终止于此；普通张量跳过；
6. 生成的 dispatch lambda：

```cpp
auto dispatch_flatten = [](const at::Tensor& self, int64_t start_dim, int64_t end_dim) -> at::Tensor {
  pybind11::gil_scoped_release no_gil;        // ★ 释放 GIL：整个 C++ 阶段其他 Python 线程可并行
  return self.flatten(start_dim, end_dim);    // → L2
};
return wrap(dispatch_flatten(self, _r.int64_t(0), _r.int64_t(1)));
```

7. **`wrap(...)` → `THPVariable_Wrap`（python_variable.cpp:473）**：先查返回 Tensor 的 TensorImpl 上的 `pyobj_slot`——若已有对应 PyObject 就直接 `Py_NewRef` 复用（这保证了 `t.flatten(0,0) is t` 这类同一性）；否则 `tp_alloc` 一个新 THPVariable，placement-new 把 `at::Tensor` 存进 `cdata`，并登记进 pyobj_slot。

**内存/设备**：全 host。操作对象是 PyObject 与引用计数；释放 GIL 是这里唯一影响全局状态的动作。

# L2 Tensor::flatten（生成的 C++ 方法）

`build/aten/src/ATen/core/TensorBody.h`（模板 `aten/src/ATen/templates/TensorBody.h`，生成器 `torchgen/gen.py` 的 `ComputeTensorMethod`）：

```cpp
inline at::Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
  return at::_ops::flatten_using_ints::call(*this, start_dim, end_dim);
}
```

纯转发。**这一行就是"过 dispatcher"的入口**（三种入口之一：`tensor.method()` / `at::op()` / `_ops::op::call`，本质都汇聚到 L3）。

# L3 _ops::flatten_using_ints（生成的算子句柄）

`build/aten/src/ATen/ops/flatten_ops.h`（模板 `aten/src/ATen/templates/Operator.h`）：

```cpp
struct flatten_using_ints {
  using schema = at::Tensor(const at::Tensor&, int64_t, int64_t);
  static C10_ALWAYS_INLINE at::Tensor call(const at::Tensor& self, int64_t start_dim, int64_t end_dim) {
    static auto op = create_flatten_using_ints_typed_handle();
    // = c10::Dispatcher::singleton().findSchemaOrThrow("aten::flatten", "using_ints").typed<schema>()
    return op.call(self, start_dim, end_dim);    // → L4
  }
};
```

- `findSchemaOrThrow`：在 Dispatcher 的全局算子表里按 (名字, overload) 找到 `aten::flatten.using_ints` 这一项，拿到类型化句柄 `TypedOperatorHandle`。static 局部 ⇒ 按名查找只做一次，之后缓存句柄直接复用；
- `op.call` → `TypedOperatorHandle::call`（Dispatcher.h:615）→ `c10::Dispatcher::singleton().call(...)`。

**内存/设备**：host。首次调用初始化静态句柄（查名）；之后零额外状态。

# L4 Dispatcher::call 内部（Dispatcher.h:778）

```cpp
Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  // ① 计算 DispatchKeySet
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
      .template getDispatchKeySetUnboxed<Args...>(args...);
  // (debug 构建里：TORCH_SHOW_DISPATCH_TRACE=1 时打印 [call] aten::flatten key=...)
  // ② 查表
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
  // ③ 若有 RecordFunction 回调（profiler 等）走慢路径 callWithDispatchKeySlowPath
  // ④ 跳转
  return kernel.template call<Return, Args...>(op, dispatchKeySet, args...);
}
```

## ① DispatchKeySet 怎么算出来的

两步（`c10/core/DispatchKeyExtractor.h` + `c10/core/DispatchKeySet.h`）：

1. **`multi_dispatch_key_set(args...)`**：遍历全部参数，把每个 tensor 的 `TensorImpl::key_set_` 按位 OR。一个普通 dense CPU float 张量的 key_set = {CPU（运行时 key）+ Dense/CPUBit（backend 组件）+ **AutogradCPU**}。注意 autograd key 是 TensorImpl 构造时无条件加进去的——**即使 requires_grad=False，AutogradCPU 也在集合里**；
2. **`impl::computeDispatchKeySet(ks, nonFallthroughKeys)`**（DispatchKeySet.h:24）：

```cpp
return (((ks | local.included_) - local.excluded_) & key_mask);
```

- `local.included_` 默认 = `{BackendSelect, ADInplaceOrView}`（`default_included_set`，TLS 初始化时注入）——**ADInplaceOrView 不来自张量，而是每次调用由 TLS 默认注入**；
- `local.excluded_` 默认 = 各 Autocast key；此外当前线程栈上的 `AutoDispatchBelowXXX` guard 会往 excluded_ 里加 key——这正是 redispatch 链"一层层往下走"的机械原理；
- `key_mask`：抠掉本算子注册为 fallthrough 的 key。

最终（CPU 张量、无任何 guard）：ks = {CPU, AutogradCPU, ADInplaceOrView, BackendSelect} + backend bits。

## ② 查表：优先级与 alias fallback

DispatchKeySet 是 64 位位集，key 枚举值大小即优先级。本链路相关优先级从高到低：

```
AutogradCPU  >  ADInplaceOrView  >  BackendSelect  >  CPU
```

`OperatorEntry::lookup(ks)` = `dispatchTable_[ks.highestPriorityKey()]`。`dispatchTable_` 在注册期（`import torch` 时各 `Register*.cpp` 静态初始化执行 `m.impl`）已填好：每个运行时 key 先看有无直接注册；没有则按 alias 顺序 fallback：`CompositeExplicitAutograd` → `CompositeImplicitAutograd`。

flatten 的注册情况（对照 yaml 与生成器规则）：

| key | 注册项 | 谁生成的 |
|---|---|---|
| CompositeImplicitAutograd | `wrapper_CompositeImplicitAutograd_using_ints_flatten` | torchgen（yaml 无 dispatch 段 → 默认 composite） |
| Autograd{CPU,CUDA,...} | VariableType 包装 | `gen_autograd.py`（**所有算子都有**） |
| ADInplaceOrView | ADInplaceOrViewType 包装 | `gen_inplace_or_view_type.py`（**view/inplace 算子才有**；flatten 返回 `Tensor(a)` ⇒ view） |
| BackendSelect | 设备检查/DeviceGuard 包装 | torchgen（带 tensor 参数的算子都有） |
| CPU/CUDA | 无直接注册 ⇒ 槽位 = alias fallback 到 composite wrapper | — |

## 命中链逐跳（dense 张量、requires_grad=False、未开 profiler）

**第 1 跳：AutogradCPU → VariableType 生成的 kernel**（`torch/csrc/autograd/generated/VariableType_*.cpp`）

所有算子必经（`DispatchKey.h` 的 Note [Dream: skip VariableType kernel when requires_grad=false] 明说了目前不跳过）。对 flatten（view 算子）：

- unpack self；
- `compute_requires_grad(self)`：本例 false ⇒ **不创建 grad_fn、不记录图**；
- **`at::AutoDispatchBelowAutograd guard;`**——view/inplace 算子用这个（`gen_variable_type.py:1745-1748`：view 或修改参数的算子用 `AutoDispatchBelowAutograd`，其余算子用更强的 `AutoDispatchBelowADInplaceOrView`）。guard 把 autograd keys 加进 TLS excluded_；
- `at::redispatch::flatten(ks & c10::after_autograd_keyset, self, ...)`：`after_autograd_keyset` = 全部 autograd key 之下的 key 全集 ⇒ 剩余 ks 最高 = **ADInplaceOrView**；
- 为什么 view 算子只排除 autograd 而不连 ADInplaceOrView 一起排除：**view 关系元数据要在 ADInplaceOrView 层记录，不能跳过它**。

**第 2 跳：ADInplaceOrView → `gen_inplace_or_view_type.py` 生成的 kernel**（`ADInplaceOrViewType_*.cpp`）

view/inplace 算子专属层（`DispatchKey.h` 的 Note [ADInplaceOrView key]：view 算子在此做 `as_view` 设置、inplace 算子在此做 version bump）。对 flatten：

- `at::AutoDispatchBelowADInplaceOrView guard;`（ADInplaceOrView 及以上全部排除）；
- redispatch 到 `ks & after_ADInplaceOrView_keyset` ⇒ 剩余最高 = **BackendSelect**；
- 拿到下层结果后做 **as_view 设置**：若 base requires_grad 且 GradMode 开启，给输出 TensorImpl 挂 `DifferentiableViewMeta`（记录 view_func/rev_view_func/creation_meta），backward 时用它把梯度逆变换回 base 的 shape。本例 requires_grad=False ⇒ creation_meta=NO_GRAD_MODE，不建 meta，纯透传。

**第 3 跳：BackendSelect → 生成的设备检查 kernel**

- 从参数推出设备（CPU），做设备一致性检查；需要时 `OptionalDeviceGuard` 切换当前设备（对 CUDA 算子这是 `cudaSetDevice` 的发生点；CPU 无操作）；
- redispatch 排除 BackendSelect ⇒ 剩余最高 = **CPU**。

**第 4 跳：CPU 槽 → alias fallback → CompositeImplicitAutograd wrapper**（`build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp`）

```cpp
at::Tensor wrapper_CompositeImplicitAutograd_using_ints_flatten(
    const at::Tensor& self, int64_t start_dim, int64_t end_dim) {
  return at::native::flatten(self, start_dim, end_dim);   // → L5
}
```

四跳完成"过表"。此后不再有查表，直到 kernel 内部自己发起新的过表调用。

> 若开着 profiler：每次 `call` 前还会过 `RecordFunction` 慢路径——profiler 靠这个钩子看到 `aten::flatten`；而 kernel 内部的**直连**调用没有 RecordFunction，所以 profiler 看不见（这就是 [[PyTorch-ATen-Dispatcher]] 里 flatten→view 而不见 reshape 的原因）。

# L5 at::native::flatten（TensorShape.cpp:4178，手写源码）

```cpp
Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim) {
  start_dim = maybe_wrap_dim(start_dim, self.dim());   // 分支②：负维度转正（-1 → dim-1）；越界 → TORCH_CHECK 报错
  end_dim = maybe_wrap_dim(end_dim, self.dim());
  TORCH_CHECK(start_dim <= end_dim, ...);              // 分支③：start > end → 抛异常

  if (self.dim() == 0)  return self.reshape({1});      // 分支④：0 维 → reshape({1})（self.reshape 是方法调用 ⇒ 过表）
  if (start_dim == end_dim) return self;               // 分支⑤：合并单维 = 恒等 ⇒ 直接返回自己（refcount++，零拷贝）

  auto slice_numel = c10::multiply_integers(
      self.sym_sizes().slice(start_dim, end_dim - start_dim + 1));   // 合并区间维度连乘
  std::vector<c10::SymInt> shape;                                     // 拼 [前段..., slice_numel, 后段...]
  shape.reserve(self.dim() - end_dim + start_dim);
  for (i < start_dim)              shape.push_back(sym_sizes()[i]);
  shape.push_back(slice_numel);
  for (i in end_dim+1 .. dim)      shape.push_back(sym_sizes()[i]);

  return native::reshape_symint(self, shape);          // ★ 直接 C++ 调用，不过表
}
```

- **执行位置**：全部在 host（发起调用的 CPU 线程），纯整数运算，不读张量数据，与 tensor 在 CPU 还是 GPU 完全无关——这正是 flatten 在 yaml 里没有 per-backend dispatch 的原因；
- **内存操作**：一个 shape vector（几十字节）；
- **为什么直连 `native::reshape_symint` 而不是 `self.reshape`（过表）**：① shape 推导逻辑与设备/梯度无关，过表是纯开销；② flatten 已经过了 Autograd/ADInplaceOrView 层，再过表会让 view 关系被重复记录。

# L6 reshape_symint（TensorShape.cpp:2058）——view/copy 分水岭

```cpp
Tensor reshape_symint(const Tensor& self, c10::SymIntArrayRef proposed_shape) {
  if (self.is_sparse()) TORCH_CHECK(false, ...);                 // 分支⑥：sparse 报错

  if (self.is_contiguous_or_false() && !self.is_mkldnn()) {
    return self.view_symint(proposed_shape);                     // 分支⑦（快路径）：连续 ⇒ 必然可 view
  }

  auto sym_numel = self.sym_numel();
  c10::SymDimVector shape = infer_size_dv(proposed_shape, sym_numel);   // 推断 -1
  if (self.is_mkldnn()) return at::_mkldnn_reshape(...);         // 分支⑧：mkldnn 专用

  auto stride = at::detail::computeStride(sym_sizes, sym_strides, shape);   // ★ 判定
  if (stride.has_value()) {
    if (!self.is_xla() && !self.is_lazy() && !self.is_ipu() && !at::isTensorSubclassLike(self)) {
      return self._reshape_alias_symint(shape, stride.value());  // 分支⑨：可 view（过表 _reshape_alias）
    } else {
      return self.view_symint(shape);                            // 分支⑩：特殊后端/子类退回 view
    }
  }
  return at::_unsafe_view_symint(                                // 分支⑪：必须 copy
      self.clone(at::MemoryFormat::Contiguous), shape);
}
```

对 flatten 的具体语义：**被合并的那段维度在内存中连续（`stride[i] == stride[i+1] × size[i+1]`），或其中含 size 为 0/1 的维 ⇒ view；否则 ⇒ copy。** contiguous 张量必 view；transpose 后合并被交换的两维必 copy。

## computeStride 算法（TensorUtils.cpp:327，`computeStride_impl`）

回答一个问题：**"新 shape 能不能用旧的 stride 体系表达出来？"** 逐段：

1. 旧 shape 为空：直接返回全 1 的 newstride；
2. `numel == 0` 特判：shape 相同 → 复制旧 stride；不同 → 按连续规则造 stride（此时 stride 本就无意义，对齐 NumPy 行为）；
3. 主算法——双指针从最后一维往前走：
   - `tensor_d` 从后往前扫旧 shape，累积 `tensor_numel`，把旧维度切成若干"**连续块**"：块结束条件 = 到第 0 维，或 `oldshape[d-1] != 1` 且 `oldstride[d-1] != tensor_numel × chunk_base_stride`（即上一维与本块在内存里不是无缝衔接；size==1 的维随便归并）；
   - 每切出一块，内层 while 用新 shape 末尾的若干维去填它：`newstride[view_d] = view_numel × chunk_base_stride`，累积 `view_numel`，直到 `view_numel == tensor_numel`（新 shape 里 size==1 的维可以白送）；
   - **填不满（`view_numel != tensor_numel`）→ return nullopt → 走 clone 拷贝**；全部填完且 `view_d` 恰好走完 → 返回 newstride → view；
4. `TORCH_GUARD_OR_TRUE/FALSE`：符号 shape（unbacked SymInt）下的保守处理——guard 失败宁可返回 nullopt 走 clone，也不会给出错误结果。

## 分支⑦/⑩：view_symint 路径（过表）

`self.view_symint(...)` 是生成的 TensorBody 方法 → `_ops::view::call` → 新一轮的 `Dispatcher::call`：

- 此时 TLS 的 excluded_ 里已有 Autograd + ADInplaceOrView（L4 两个 guard 还活着）⇒ **不会再过 autograd 包装，view 关系不会被重复记录**；
- 命中链：BackendSelect → CPU 槽 → `wrapper_CPU_view`（yaml 里 view 注册在 `CPU, CUDA, Meta...: view`；schema 是 SymInt，torchgen 在 wrapper 里做 SymInt→IntArrayRef 的参数翻译，见 `torchgen/api/cpp.py:84` 的 `_symint` 命名与 `register_dispatch_key.py` 的 `translate`）→ `at::native::view`（TensorShape.cpp:4563）→ `view_impl`（:4093）：

```cpp
static inline Tensor view_impl(const Tensor& self, IntArrayRef size) {
  auto inferred_size = at::infer_size_dv(size, self.numel());            // 推断 -1
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(stride.has_value(),
              "view size is not compatible ... Use .reshape(...) instead.");
  return alias_with_sizes_and_strides(self, inferred_size, *stride);
}
```

view 与 reshape 的区别就在这一行：computeStride 失败时 **view 报错、reshape 走 clone**。flatten 走快路径时输入已连续，必然成功。

## 分支⑨：_reshape_alias_symint 路径（过表）

`_ops::_reshape_alias::call` → wrapper → `at::native::_reshape_alias`（:2168）→ `alias_with_sizes_and_strides`。它存在的意义（源码注释）：reshape 已经做过 `infer_size_dv` 和 `computeStride`，直接复用结果，省掉 view 里的重复计算；同时不调 `as_strided` 是因为其 backward 不如 view 高效。

## alias_with_sizes_and_strides（:2028）——"只改 metadata"的字面实现

```cpp
Tensor self_ = at::detail::make_tensor<TensorImpl>(
    c10::TensorImpl::VIEW,
    Storage(self.storage()),        // ★ 共享同一 Storage：intrusive_ptr 原子 refcount++，零数据拷贝
    self.key_set(), self.dtype());
self_.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides, self.sym_storage_offset());  // ★ 只写元数据
namedinference::propagate_names(self_, self);
return self_;
```

内存操作清单（全部 host）：

- new 一个 TensorImpl（host 堆，几百字节：sizes/strides/offset/key_set/dtype/storage 指针）；
- Storage 引用计数 +1（原子操作）；
- 写 sizes/strides/offset 数组、传播命名；
- **数据区一个字节都没动。对 CUDA 张量同样如此——整条 view 分支没有任何一次 CUDA API 调用，GPU 完全无感。**

## 分支⑪：copy 路径（全链路唯一真正搬数据的分支）

`at::_unsafe_view_symint(self.clone(at::MemoryFormat::Contiguous), shape)`，两小步：

### ⑪a. self.clone(Contiguous)（过表）

命中链（guard 已排除 autograd/ADIV）：BackendSelect → `CompositeExplicitAutograd: clone`（yaml:7147）→ `at::native::clone`（**TensorFactories.cpp:2270**）：

```cpp
Tensor clone(const Tensor& src, std::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);
  Tensor self;
  if (memory_format == MemoryFormat::Preserve) {
    if (src.is_non_overlapping_and_dense()) {
      self = at::empty_strided_symint(src.sym_sizes(), src.sym_strides(), src.options());
    } else {
      self = at::empty_like(src);
    }
  } else {
    self = at::empty_like(src, src.options(), memory_format);   // ← flatten 走这里：Contiguous
  }
  if (src._is_zerotensor()) { self.zero_(); } else { self.copy_(src); }   // ★ 数据搬运
  return self;
}
```

**分配：`empty_like → empty → at::detail::_empty_generic`（EmptyTensor.cpp:176）**：

```cpp
auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());
auto storage_impl = c10::make_intrusive<StorageImpl>(
    use_byte_size_t, size_bytes, allocator, /*resizeable=*/true);    // ★ 分配数据内存
auto tensor = detail::make_tensor_base<TensorImpl>(std::move(storage_impl), ks, dtype);
tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);    // 连续 strides
```

allocator 按设备分：

- **CPU 张量**：`DefaultCPUAllocator::allocate` → posix_memalign/malloc，分配 host RAM，立即发生；
- **CUDA 张量**：`CUDACachingAllocator::allocate` → 先从显存缓存池找空闲块复用；池里没有才 `cudaMalloc`。分配的是 device HBM，但分配动作本身是 host 侧记账，不启动任何 kernel。

**搬运：`self.copy_(src)`（过表）**：`aten::copy_` → `CompositeExplicitAutograd: copy_`（Copy.cpp:353）→ `copy_impl`（:140）：

1. fbgemm 特判（CPU 上 FP32↔FP16 连续互转走 fbgemm SIMD）；
2. meta / 量化检查；
3. 构建 **TensorIterator**（`add_output(dst).add_input(src)`）：维度合并/重排、广播计算——PyTorch 所有 elementwise 操作的标准前奏；
4. `numel == 0` → 直接返回；
5. 选 device_type：dst/src 皆 CPU → CPU；任一是 CUDA → CUDA（flatten 场景同设备）；
6. `copy_stub(device_type, iter, non_blocking)`——按设备分派的函数指针表：

**CPU → `copy_kernel`（cpu/CopyKernel.cpp:284）**：

- 同 dtype（clone 场景必然同）→ `copy_same_dtype` → `direct_copy_kernel`（:220）→ `cpu_kernel_vec`：按 dtype 实例化的向量化拷贝循环；连续块做整块 vector load/store（等效 memcpy 速度），src 非连续（transpose 过）按 stride gather；大 tensor 由 TensorIterator 的 `parallel_for` 按 GRAIN_SIZE 切块多线程；
- 不同 dtype → `at::vec::convert` / `cpu_kernel` 逐元素转换。

**CUDA → `copy_kernel_cuda`（cuda/Copy.cu:381）**：

1. `copy_requires_temporaries`：同卡永远 false；跨卡看 P2P；CPU↔GPU 且（非连续或转 dtype）→ 先做连续化临时缓冲（递归 copy_）；
2. 同卡 → `copy_device_to_device`（:264）：
   - `memcpy_eligible = 同 dtype && same_conj && same_neg && iter.is_contiguous()`；
   - **flatten 场景：dst 连续、src 非连续 ⇒ false ⇒ `direct_copy_kernel_cuda` ⇒ `gpu_kernel` 启动一个 elementwise CUDA kernel**——每个线程按 iterator 的 stride 计算 src 地址、读一个元素写到 dst。★ 这是整个 flatten 链路里**唯一一次 GPU kernel launch**（contiguous flatten 则一次都没有）；
   - 若两侧都连续 → `CUDACachingAllocator::memcpyAsync`（同卡即 `cudaMemcpyAsync` D2D，在 src 设备当前 stream 上）；跨卡时还要用 event pool 做双向流同步；
3. CPU↔GPU：`cudaMemcpyAsync`（non_blocking，要求 pinned memory；CUDA Graph 捕获期间强制要求 pinned）或 `memcpy_and_sync`（阻塞同步）。

产物：一个连续的新张量——新 storage、数据已复制。

### ⑪b. at::_unsafe_view_symint(cloned, shape)（过表）

`CompositeExplicitAutograd: _unsafe_view`（yaml:6653）→ `at::native::_unsafe_view`（TensorShape.cpp:4105）→ `view_impl` → 连续张量 computeStride 必然成功 → `alias_with_sizes_and_strides`。

为什么用 `_unsafe_view` 而不是 `view`：NOTE [ Unsafe View ]——克隆结果是临时张量，不希望它被 autograd 当作"用户的 view"来记录和做 inplace 检查（那些检查有额外开销）；backward 走 clone 自己的 backward（恒等回传）。

# 返回路径（逐层回去）

1. `at::native::flatten` 的返回值沿 redispatch 链返回：ADInplaceOrView kernel 收到结果 → 需要时 `as_view` 挂 DifferentiableViewMeta（本例不需要）；
2. VariableType kernel 收到 → requires_grad=True 时设置输出 autograd meta（view 的 backward 由 ADIV 层的 ViewInfo 支撑）；本例透传；
3. 回到绑定层 lambda → `gil_scoped_release` 析构重新持有 GIL → `wrap` → `THPVariable_Wrap`：新 TensorImpl 尚无 PyObject → tp_alloc 新 THPVariable → 交还 Python。

# 总表 A：每一步在哪执行、动什么内存

| 步骤 | 执行位置 | 内存/设备行为 |
|---|---|---|
| L0 属性查找 | host（CPython） | 无 |
| L1 绑定：解包/参数解析 | host | PyObject 操作；释放 GIL |
| L2/L3 方法 → 算子句柄 | host | 无（static 句柄首次查名一次） |
| L4 dispatcher 查表 ×4 跳 | host | 不碰数据；requires_grad 时 VariableType/ADIV 建 autograd meta（host 堆） |
| L5 flatten 拼 shape | host | 一个 shape vector（几十字节） |
| L6 computeStride | host | 一个 strides vector |
| view 分支 alias | host | new TensorImpl + Storage refcount++；**零数据拷贝，设备无感** |
| copy 分支 empty_like | host 记账 | CPU：malloc host RAM；CUDA：caching allocator 取显存块（可能 cudaMalloc） |
| copy 分支 copy_ | **设备**（CPU 向量化循环 / GPU elementwise kernel 或 cudaMemcpyAsync） | **全链路唯一搬数据的地方** |
| _unsafe_view | host | new TensorImpl 共享 clone 的 storage |
| 返回 wrap | host（重新持 GIL） | 新 PyObject |

# 总表 B：分支决策一览

| # | 分支点 | 条件 | 去向 |
|---|---|---|---|
| ① | 绑定层 | `__torch_function__` 存在 | 走 Python 覆盖 |
| ② | flatten | 维度越界 | TORCH_CHECK 异常 |
| ③ | flatten | start_dim > end_dim | 异常 |
| ④ | flatten | dim()==0 | `self.reshape({1})`（过表） |
| ⑤ | flatten | start_dim==end_dim | 直接返回 self（零拷贝） |
| ⑥ | reshape | sparse | 报错 |
| ⑦ | reshape | contiguous 且非 mkldnn | 快路径 view_symint（view） |
| ⑧ | reshape | mkldnn | `_mkldnn_reshape` |
| ⑨ | reshape | computeStride 成功且非 xla/lazy/ipu/子类 | `_reshape_alias`（view） |
| ⑩ | reshape | computeStride 成功但特殊后端/子类 | view_symint（view） |
| ⑪ | reshape | computeStride 失败 | clone + _unsafe_view（**copy**） |
| ⑫ | copy_ | CPU↔GPU 且非连续/转 dtype | 先连续化临时缓冲再 memcpyAsync |
| ⑬ | copy_device_to_device | 同 dtype 且连续 | cudaMemcpyAsync；否则 elementwise kernel |

# 实证验证

```bash
# dispatch trace：能看到 [call] aten::flatten → 各层 [redispatch] → 内部过表的 aten::view / aten::clone / aten::copy_
# 注意：release（pip）版 NDEBUG 把 trace 编译掉了，需 debug 构建，见 [[debug与release构建的行为差异]]
TORCH_SHOW_DISPATCH_TRACE=1 python -c "import torch; torch.randn(2,3,4).flatten(0,1)"
```

```python
# view/copy 实证
t.flatten(0,1)._base is t                     # True  → view（_base 指向原张量）
t.transpose(0,1).flatten(0,1)._base is None   # True  → 走了 clone

# profiler：copy 路径会出现 aten::clone / aten::copy_；view 路径只有 aten::view
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:
    t.transpose(0, 1).flatten(0, 1)
print(prof.key_averages().table())
```

# 一句话总结

> flatten 的旅程：**Python 绑定（解包+放 GIL）→ 生成的两层包装 → dispatcher 四跳（Autograd → ADInplaceOrView → BackendSelect → backend 槽 alias 到 composite）→ `at::native::flatten` 拼 shape → `reshape_symint` 判定**：能 view 就 `alias_with_sizes_and_strides` 共享 storage 只写元数据（全程 host，GPU 无感）；不能 view 才 `clone`——`empty_like` 分配新内存、`copy_` 在设备上真正搬数据（CPU 向量化循环 / GPU elementwise kernel 或 cudaMemcpyAsync），最后 `_unsafe_view` 给克隆结果换 shape。整条链路里唯一碰数据、唯一可能在 device 上执行的，只有 copy 分支的 `copy_`。

## 相关页面

- [[pytorch-flatten-调用链路定位]] — 同一条链的"定位版"（怎么找到这些文件），本文是"逐行详解版"
- [[PyTorch-ATen-Dispatcher]] — L4 的机制详解（双层模型、注册表、过表 vs 直连）
- [[PyTorch-代码生成管线]] — L1/L2/L3/L4 各层代码是谁生成的、生成到哪
- [[aten算子调用链定位方法论]] — 把这套走查方法推广到任意算子的 SOP
- [[debug与release构建的行为差异]] — 为什么 debug 单步看到的这条链与 release 逐行一致
- [[PyTorch-源码分析工具箱]] — 实证工具（profiler / dispatch trace / gdb）详解
