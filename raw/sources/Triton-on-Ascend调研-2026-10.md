# 调研笔记：Triton on Ascend（昇腾 NPU）现状

> 调研日期：2026-10-12
> 调研方式：LLM agent 网络调研（websearch/webfetch 多来源综合），优先一手来源（GitHub 仓库 README/issues、官方文档站）
> 状态：已交叉核对，个别项标注"⚠️ 未确认"
> 面向读者：用 Triton 写过 GPU kernel、了解 FlagGems 和 vLLM 的算子工程师

---

# Triton on Ascend（昇腾 NPU）调研笔记

**核心结论先行**：**Triton kernel 可以跑在昇腾上**，官方项目 `triton-ascend` 已进入 `triton-lang` 组织、发布到 3.2.2；vLLM/SGLang/FlagGems 均已有真实落地；但与 CUDA 后端存在语义差异和版本强耦合，性能上离手写 Ascend C / PTO-ISA 仍有差距。

## 1. triton-ascend 项目概况

### 1.1 仓库与维护方

- **当前主仓库**：[github.com/triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend)——已挂到 **Triton 官方组织 `triton-lang` 之下**（即上游社区正式接纳为官方后端之一，与 sgl-project 无关）。旧仓 [Ascend/triton-ascend](https://github.com/Ascend/triton-ascend) README 明确写着 "Triton Ascend has moved to triton-lang/triton-ascend. This repository will be archived."（迁入具体日期 ⚠️ 未确认，约在 2025 年底—2026 年初）。
- **实际开发与维护方仍是华为昇腾团队**（最早 2025-05-20 在 GitCode 的 Ascend 组织下开源）。
- 活跃度：约 **185 stars / 251 forks / 7000+ commits**（fork 数多于 star，典型厂商共建模式），有公开的社区例会和 `GOVERNANCE.md`/`MAINTAINERS.md`。
- License: MIT。文档站：[triton-ascend.readthedocs.io](https://triton-ascend.readthedocs.io)。

### 1.2 架构原理（编译路径）

它是 Triton 的**完整 fork + `third_party/ascend` 后端**（与官方 `third_party/nvidia`、`third_party/amd` 同构），而非上游的一个 runtime 插件。**关键差异：它不走 Triton 传统的 TritonGPU → LLVM → PTX 路径**，而是：

```
Triton IR (TTIR)
  → [Triton-Ascend MLIR passes: triton-to-structured / triton-to-linalg / triton-to-annotation ...]
  → Linalg IR
  → [BiSheng Compiler: bishengir-compile]
  → AscendNPU IR（HFusion / HIVM dialect，即 MLIR 方言栈）
  → hivmc → LLVM IR → NPU 可执行二进制 kernel*.o
  → driver.py 通过 CANN 运行时加载执行
```

- 前半段（TTIR → Linalg）在 triton-ascend 仓内（`third_party/ascend/lib/TritonToLinalg` 等）；后半段交给华为 **BiSheng Compiler / AscendNPU-IR**（[Ascend/AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR)，基于 MLIR 生态，llvm-project 用昇腾维护的 `llvmorg-19.1.7` 分支）。
- 代码原则：target-independent 的改动尽量回上游/以 patch 形式隔离（`third_party/ascend/patch/`），target-affinitive 的放 `third_party/ascend/`。
- **950 系列新增 SIMT 路径**：`compile_mode` 可选 `simd`（纯 SIMD，结构化访问走 DMA、非结构化展开为标量循环）、`simd_simt_template`（默认，混合）、`simt_only`（TTIR 直接进 AscendNPU IR，更接近 GPU warp 语义，接社区 TritonGPU dialect → HIVM）。
- 语言扩展：`triton.language.extra.cann`（libdevice 适配层），以及更底层的 `triton.backends.ascend.language.cann.extension.core`（`ascend_address_space`、`copy`、`fixpipe`、`sync_block_set/wait` 等 Cube/Vector 手工控制原语）。

### 1.3 支持型号与版本配套

- **支持 Atlas A2（910B 系列）、Atlas A3（910C/910_93 系列）、Ascend 950PR/950DT**。昇腾推理卡（310P 等 Atlas inference 产品）**不支持** triton / triton-ascend（vllm-ascend 文档明确要求卸载）。
- 版本配套（README/release notes）：

| triton-ascend | 发布日期 | 配套 CANN | 配套 torch_npu |
|---|---|---|---|
| 3.2.0（首个正式版，基于社区 Triton 3.2.0） | 2026-01-20 | 8.5.0 | 2.7.1 系 |
| 3.2.1 | 2026-04-30 | 9.0.0 | 2.7.1 系 |
| 3.2.2 | 2026-07-31 | 9.1.0（推荐） | 2.7.1.post8 |

  `main` 分支已跟踪上游 Triton 3.6.0。Python 3.9–3.13。
- **安装**：`pip install triton-ascend --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi`，也有 nightly 通道和官方 Docker 镜像（`quay.io/ascend/triton:3.2.2-cann9.1.0-...-910b/a3/950`）。
- ⚠️ **大坑：与社区 triton 同名冲突**。triton-ascend 安装后覆盖 `triton` 包目录；3.2.0 及之前与社区 triton 不能共存，3.2.1 起通过把 triton 声明为依赖缓解。任何后续 `pip install` 拉升级版 triton 都会损坏它。sglang 社区也反馈 triton-ascend 与 triton 3.5/3.7 不二进制兼容。
- **使用方式**：kernel 代码（`@triton.jit`、launch grid）基本零改动；只需把 host 侧 `device='cuda'`→`'npu'`、`.cuda()`→`.npu()`、`torch.cuda.*`→`torch.npu.*`。官方提供 GPU→NPU 迁移指南。

## 2. 能力矩阵与限制

**已支持**：约 85% Triton Python API（2025-06 起，覆盖连续访存基本场景），2025-08 完善 Atomic 类 API，2025-09 完善 Scan/Sort 与非连续访存，2025-11 加 FP8 `tt.fp_to_fp` 转换、`scatter_ub_to_out`。Autotune 可用，且 Ascend 后端扩展了**自动 tiling 候选生成**（`configs=[]` 即可），但 `num_warps`/`num_stages` 等非 tiling 参数不在自动生成范围。

**tl.dot（重点）**（[RFC #1193](https://github.com/triton-lang/triton-ascend/issues/1193) + API 文档）：
- A2/A3 上 `tl.dot` 支持 int8/fp16/fp32/bf16，**不支持 fp8**；950 系列增加 fp8 e4m3/e5m2（910_95 上 FP8 仅限 `tl.dot_scaled` 路径）。不支持 fp64/int16/int32 等。
- 昇腾没有真 TF32，`input_precision="tf32"` 会被映射到 **HF32**（19-bit 尾数）；HF32 仅对 fp32×fp32 生效，其他 dtype 回落 `ieee`；默认 `ieee`；`tf32x3` 不支持；`max_num_imprecise_acc` **静默忽略**（写进 IR 但 lowering 不消费，无 warning）。

**与 CUDA 后端的行为差异 / 已知坑**：
- **num_warps 语义不同**：SIMD 路径下它只是编译器并行/布局选择提示，不对应 GPU warp 调度；版本会对 `num_warps/num_ctas/num_stages` 等打出 "DO NOT tune" 警告。
- **grid 物理绑核**：Ascend 是物理核绑定模型，无 `auto-blockify` 时 grid ≤ 65535 且实际受 AI core 数限制；`SIMTAutoBlockify` pass 可把逻辑 grid 折叠到物理核，但要求逻辑 block 间顺序无关（跨 block 同步语义的 kernel 要重写）。
- **指针别名不支持**：编译器假定所有指针参数指向不同内存，同一 tensor 传两个指针参数会导致优化失效或结果错误。
- **UB Overflow 是最高频问题**（官方 FAQ 单列一节；A2 的 UB 仅 192KB，950 为 248KB 可用，跨型号迁移易触发）。
- **32B 对齐**要求；i64 向量算术/部分整数比较会降级为标量（`aiv_scalar_ratio` 飙升，性能杀手）；`tl.swizzle2d` 在 Ascend 路径不可用（FlagGems 文档）；`tl.device_print` 可用。
- CANN `libdevice` 缺部分符号（如 `j0`/`j1` Bessel 函数），FlagGems 曾因此 import 即挂（issue #5328，后由 FlagGems 加纯 Triton fallback 修复）。
- `torch.compile` 图缓存与 Triton kernel 的 `kernel_side_table` 不兼容：含 triton kernel 的图不能跨进程序列化缓存（vllm-ascend `compiler_interface.py` 中专门处理）。

## 3. FlagGems / FlagOS 与昇腾

- **FlagGems 已有正式 NPU 后端**：`device_name="npu"`、`GEMS_VENDOR=ascend`，`flaggems-setup ascend-cann900` 一键装环境；C++ 封装算子有 `cpp-npu` 变体（dispatch key 用 `PrivateUse1`），底层走多后端 `libtriton_jit`（已支持 CUDA/MUSA/NPU/IX）。triton-ascend 官方 2025-08 宣布"完成 FlagGems 开源仓重点 Triton 算子适配"。FlagGems 文档站有专门的 [Ascend 调优页](https://flagos-ai.github.io/FlagGems/zh-cn/optimization/ascend/)，详述 Cube/Vector 双路径、L1/L0A/L0B/L0C/UB 容量、`compile_mode`/`multibuffer`/`enable_mixed_cv` 等昇腾专属选项——是目前最实操的一份能力/调优参考。
- **FlagOS（智源）与昇腾的关系**：FlagOS 的 **FlagTree** 多芯片编译器提出了 **Triton-TLE** 扩展语言，通过 FLIR 中间层与华为 **AscendNPU IR** 对接（2026-04 双方官宣"深度适配"），即：**在昇腾上跑 Triton 系 DSL 存在两条编译链路——华为 triton-ascend（TTIR→Linalg→BiSheng）与 FlagTree（TLE→FLIR→AscendNPU IR）**。FlagGems 安装时默认优先选用 FlagTree（若后端提供），否则回退到厂商 Triton（昇腾上即 triton-ascend）。

## 4. 上层框架路径（vLLM / torch.compile）

- **vLLM-Ascend**：pip 依赖里直接装 `triton-ascend`，vLLM 里的手写 Triton kernel 可经它在 NPU 上执行；triton-ascend 官方 2025-09 宣布"完成 vLLM、sglang 开源仓重点 Triton 算子适配"。
- **torch.compile 的 Inductor 路径是 2025–2026 的关键变化**：
  - 历史上 `torch_npu` **不支持 Inductor 的 Triton codegen**，vLLM-Ascend 只能走 AOTAutograd 后的 FX 图 + torchair/npugraph_ex 后端，融合靠自研 graph rewriter / pattern matcher。
  - 2026-04 的 RFC #8010 正式提出 vLLM-Ascend 适配 Inductor（`use_inductor` / `use_triton` 开关）。
  - **TorchNPU 26.1.0 官方文档已把 Inductor 的 Triton 模式列为默认**：`torch.compile(backend="inductor")` 在昇腾上生成 Triton 融合算子（要求 triton-ascend 3.2.2），另可选 `npu_backend="mlir"`（Torch-MLIR）或 `"dvm"`。即：**torch.compile → Triton 在昇腾上已打通，处于早期可用阶段**。
- **SGLang**：[sgl-project/sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) 同时含 Ascend C 算子和一批 **Triton 算子**（attention/rmsnorm/swiglu/fla/mamba/moe/投机采样），跑在 triton-ascend 上；但社区反馈版本耦合脆弱（CANN 9.0.0 + triton-ascend 3.2.1 + MindSpore 依赖冲突、bishengir-compile SIGSEGV 等）。

## 5. 性能数据（公开的不多）

- 官方口径：triton-ascend README 称在 **GroupGEMM** 等场景与手写 **Ascend C** 实现"性能有竞争力"，但未给出系统对比表。⚠️ 未见权威第三方的 Triton-Ascend vs Ascend C vs CUDA 完整 benchmark。
- [TritonAscendBench](https://github.com/Gxj230958/TritonAscendBench)（2026，研究预览）：从 vLLM/SGLang 抽取 101 个生产级 GPU Triton kernel，考察迁移到 910B 的正确性（83 verified + 18 ceiling-accepted）与 Roofline 效率；性能部分尚未端到端打通。它本身证明了"生产 kernel 可迁移"，但不提供性能结论。
- 旁证（Triton 非最优）：sgl-kernel-npu 用华为 PTO-ISA 手写 mega-kernel 替换 Triton 版 GDN kernel，prefill 提速 2–8%、serving 场景最高 15%（PR #462）——说明手写底层 ISA 仍明显优于 Triton 生成代码。
- 二手博客数据（⚠️ 未确认，CSDN 专栏）：某 fused kernel 上 Triton-Ascend 对比 PyTorch 原生算子吞吐 3.22x、带宽利用率 85%。可信度有限，仅作参考。

## 6. 2025–2026 大事记

- 2025-05-20：Triton-Ascend 在 GitCode 开源。
- 2025-06：85% Python API 覆盖；2025-08：Atomic API + FlagGems 适配；2025-09：Scan/Sort、非连续访存、vLLM/SGLang 重点算子适配、nightly 包；2025-11：3.2.0rc4（FP8 转换、scatter_ub_to_out）。
- 2025 底–2026 初：**迁入 `triton-lang` 组织**，成为官方后端仓。
- 2026-01-20：**3.2.0 首个正式版**（基于 Triton 3.2.0，CANN 8.5.0）。
- 2026-04：FlagTree/Triton-TLE × AscendNPU IR 官宣对接；vLLM-Ascend 发起 Inductor RFC。
- 2026-04-30 / 07-31：3.2.1（CANN 9.0.0）、**3.2.2（CANN 9.1.0）**；TorchNPU 26.1.0 将 Inductor-Triton 设为默认 codegen。
- 2026 年内：950PR/950DT 的 **SIMT/SIMD 混合编译**落地；`main` 跟踪上游 Triton 3.6.0；官方预置 A2/A3/950 全系 Docker 镜像。

**总体判断**：Triton-on-Ascend 已从"能跑"进入"官方化 + 框架集成"阶段，迁移成本低（改 host 侧几行）、生态（FlagGems/vLLM/SGLang/torch.compile）已接通；主要风险是版本强耦合（CANN/torch_npu/triton 三者锁定）、与社区 triton 的包冲突、`tl.dot` 精度和 dtype 限制、以及性能仍需昇腾专属调优（UB/对齐/CV 混合）才能接近手写 Ascend C。

## 来源清单

1. https://github.com/triton-lang/triton-ascend — 主仓库 README：版本新闻、支持型号（A2/A3/950）、CANN 9.1.0/torch_npu 2.7.1.post8 配套、安装方式。
2. https://github.com/Ascend/triton-ascend — 旧仓，声明确认迁入 triton-lang 组织；含版本↔CANN 配套表。
3. https://github.com/triton-lang/triton-ascend/blob/main/docs/en/architecture_design_and_core_features.md — 架构：TTIR→Linalg→AscendNPU IR 编译链、BiSheng 集成、950 SIMT `compile_mode`、代码目录结构。
4. https://ascendnpu-ir.gitcode.com/en/introduction/architecture.html — AscendNPU IR（HFusion/HIVM/bishengir-compile/hivmc）架构与 Triton 集成层。
5. https://triton-ascend.readthedocs.io/en/latest/ — 安装冲突 FAQ、UB Overflow、指针别名限制、auto-blockify、自动 tiling、dot dtype 支持矩阵。
6. https://github.com/triton-lang/triton-ascend/issues/1193 — tl.dot 精度（tf32→hf32 映射、ieee 默认、max_num_imprecise_acc 忽略）设计文档。
7. https://flagos-ai.github.io/FlagGems/zh-cn/optimization/ascend/ — FlagGems 昇腾调优页：Cube/Vector/UB 规格、num_warps 语义差异、i64 标量化、swizzle2d 不可用等实操坑。
8. https://flagos-ai.github.io/FlagGems/zh-cn/getting-started/install/ — FlagGems NPU 后端、flaggems-setup ascend、cpp-npu、libtriton_jit 四后端。
9. https://github.com/flagos-ai/FlagGems/issues/5328 — CANN libdevice 缺 j0 导致 FlagGems import 失败的实例。
10. https://segmentfault.com/a/1190000047702113 — FlagOS Triton-TLE/FlagTree × AscendNPU IR 对接官宣（2026-04）。
11. https://github.com/vllm-project/vllm-ascend/issues/2386 与 /issues/4239 与 /issues/8010 — vLLM-Ascend 无 Inductor 的历史原因、graph-rewriter 方案、2026 Inductor 适配 RFC。
12. https://github.com/vllm-project/vllm-ascend — vllm-ascend 装 triton-ascend 3.2.1、Atlas 推理卡不支持 triton。
13. https://www.hiascend.com/document/detail/en/Pytorch/2610/userguide/torchcompile/docs/en/torch_compile/pytorch_inductor_desc.md — TorchNPU 26.1.0：Inductor Triton 默认模式、mlir/dvm 选项、依赖 triton-ascend 3.2.2。
14. https://docs.sglang.io/docs/hardware-platforms/ascend-npus/development/operator_development — sgl-kernel-npu 的 Triton 算子体系。
15. https://github.com/sgl-project/sgl-kernel-npu/pull/462 — PTO-ISA mega-kernel vs Triton 的 2–8%/15% 提速数据。
16. https://github.com/Gxj230958/TritonAscendBench — vLLM/SGLang kernel 迁移 benchmark（101 任务，正确性 83+18）。
17. https://ascendai.csdn.net/69d4c85272111d255bf7cab3.html — 二手性能案例（⚠️ 未确认）。
