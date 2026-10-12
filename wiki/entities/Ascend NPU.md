---
type: entity
title: Ascend NPU（昇腾处理器）
created: 2026-10-12
updated: 2026-10-12
sources: ["raw/sources/昇腾硬件架构调研-2026-10.md"]
tags: [ascend, 硬件层, npu, 华为]
---

# Ascend NPU（昇腾处理器）

> 所属层次：[[purpose|硬件层]]。华为自研 AI 处理器系列，计算核心架构为 [[Da Vinci 架构]]，软件栈为 [[CANN]]，算子语言为 [[Ascend C]] / [[triton-ascend]]。

## 产品系列

| 规格 | 310P（推理） | 910（2019） | 910B（2023–24） | 910C（2025） |
|---|---|---|---|---|
| FP16 | 70 TFLOPS | 256–320 TFLOPS | ~320 TFLOPS | **~800 TFLOPS**（BF16 780） |
| INT8 | 140 TOPS | 512–640 TOPS | ~640 TOPS | 1600 TOPS |
| FP8 | ❌ | ❌ | ❌ | ❌（950 起） |
| 显存 | LPDDR4X 24GB @ 204.8GB/s | 32GB HBM2 @ ~1.2TB/s | 64GB HBM2e @ ~1.6TB/s | **128GB HBM2e @ 3.2TB/s** |
| AI Core | 8–10 | 32 | 20 AIC + 40 AIV | 每 die 25 AIC + 50 AIV，双 die |
| 片间互联 | PCIe 4.0 | HCCS | HCCS 392GB/s 总带宽（P2P，单对 56GB/s） | UB 2800Gb/s scale-up + 400Gb/s RDMA |
| 制程/功耗 | 整卡 72W | 7nm, 310W | SMIC 7nm, ~400W | SMIC 7nm, ~600W |

- **910C = 两颗 910B 级 die 封装在同一基板**（dual-die），算力/显存大致翻倍，非新架构；910B 有 B1–B4 变体（早期 B1 为 32GB）
- 310P 等 Atlas 推理卡**不支持** Triton（[[triton-ascend]] 仅支持训练系 A2/A3/950）

## Roadmap（Huawei Connect 2025 官方 keynote）

- **950PR**（2026 Q1，prefill/推荐优化，自研 HBM HiBL 1.0）/ **950DT**（2026 Q4，decode/训练优化，HiZQ 2.0 HBM 144GB @ 4TB/s）：**1 PFLOPS FP8 / 2 PFLOPS FP4**，新增 FP8/MXFP8/MXFP4 与私有 HiF8 格式；**转向 SIMD+SIMT 混合架构**（见 [[SIMT 与 SIMD 与 SPMD]]）
- **960**（2027 Q4）：2× 算力、288GB @ 9.6TB/s；**970**（2028 Q4）：4 PFLOPS FP8
- 超节点：Atlas 950 SuperPoD（8192×950DT，8 EFLOPS FP8）；互联协议 **UnifiedBus（灵衢）2.0** 已开放规范
- ⚠️ 2026 年报道：960PR 提前、Hi-ONE NPO 近封装光学超节点（二手来源，待官方确认）

## 系统级：CloudMatrix 384

Atlas 900 A3 SuperPoD 的云服务形态：**384×910C + 192×Kunpeng CPU**，UB 全互联 all-to-all，~300 PFLOPS BF16、48+TB HBM，功耗 ~559kW。SemiAnalysis 评价：单芯片落后 NVIDIA 一代，但 scale-up 系统设计领先（对 GB200 NVL72：1.7× 算力、3.6× 显存、~4× 功耗）。

## 互联：HCCS 与 NVLink 的拓扑差异

HCCS 是 **P2P 拓扑（无 NVSwitch 式交换芯片）**：910B 8 卡总带宽 392GB/s 看似接近 A800 的 400GB/s NVLink，但**单对设备间仅 56GB/s**——通信模式设计（如 [[All-to-All]] 拓扑感知）在昇腾上更重要。950 起转向 UB（UnifiedBus）统一互联。

## 参见

- [[Da Vinci 架构]]、[[CANN]]、[[Ascend C]]、[[Ascend 与 NVIDIA GPU 对照]]
- 来源摘要：[[昇腾硬件架构调研-2026-10]]
