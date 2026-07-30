# Schema — Wiki 结构与维护规则

> 本文件是 LLM（wiki 维护者）的**操作手册**。每次 ingest / query / lint 前必读。
> 方向性问题看 [[purpose]]，结构规则看本文件。

## 三层架构

```
vault/
├── purpose.md          # 目标、关键问题、范围（人类主导，LLM 可建议更新）
├── schema.md           # 本文件：结构与规则
├── raw/                # 原始来源（不可变，LLM 只读不写）
│   ├── sources/        # 导入的文档：文章、论文、剪藏
│   └── assets/         # 图片等附件（Obsidian 附件目录）
└── wiki/               # LLM 生成与维护的 wiki（LLM 全权负责）
    ├── index.md        # 内容目录（每次 ingest 必更新）
    ├── log.md          # 操作日志（append-only）
    ├── overview.md     # 全局摘要（每次 ingest 后刷新）
    ├── entities/       # 实体页：人物、组织、模型、产品
    ├── concepts/       # 概念页：理论、方法、技术
    ├── sources/        # 来源摘要页：每个 raw 来源对应一页
    ├── queries/        # 有价值的问答沉淀
    ├── synthesis/      # 跨来源综合分析
    └── comparisons/    # 横向对比
```

## 页面类型与 frontmatter

每个 wiki 页面必须有 YAML frontmatter：

```yaml
---
type: entity | concept | source | query | synthesis | comparison
title: 页面标题
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: ["raw/sources/原始文件名.md"]   # 贡献该页内容的来源
tags: [相关标签]
---
```

- **entity**：具体的人、组织、模型、产品（如 GPT-4、Andrej Karpathy、OpenAI）
- **concept**：抽象的理论、方法、技术（如 Transformer、RLHF、思维链）
- **source**：对某个 raw 来源的摘要页，文件名与来源对应
- **query**：有沉淀价值的问答，来自 query 操作
- **synthesis**：跨多个来源的综合分析
- **comparison**：两个以上实体/概念的横向对比，建议用表格

## 核心操作

### Ingest（收录新来源）

用户把来源放入 `raw/sources/` 后：

1. **分析**：通读来源，提取关键实体、概念、论点；与现有 wiki 对照，找出关联、矛盾与张力
2. **生成/更新**：
   - 在 `wiki/sources/` 创建来源摘要页
   - 创建或更新相关 entity / concept 页，用 `[[wikilink]]` 互相链接
   - 在相关页面标注与已有知识的矛盾（如有）
3. **记账**：更新 `wiki/index.md`、刷新 `wiki/overview.md`、追加 `wiki/log.md`
4. **建议**：提出值得 deep research 的问题、建议更新的 purpose.md 内容

一次 ingest 可能触及 10+ 个页面，这是正常的。

### Query（查询）

1. 先读 `wiki/index.md` 定位相关页面，再深入阅读
2. 综合回答，用 `[[wikilink]]` 引用页面
3. **有价值的回答要沉淀**：存到 `wiki/queries/` 并触发一次小型 ingest（提取实体/概念进入知识网络）

### Lint（健康检查）

用户要求时执行，检查：
- 页面之间的矛盾、被新来源推翻的旧论断
- 孤儿页（无入链）、缺失的重要概念页
- 缺失的交叉引用、可以补的数据缺口
- 输出检查报告 + 修复建议，经用户确认后修复

## 约定

- **语言**：wiki 内容用中文；专有名词保留英文原文
- **链接**：页面间一律用 `[[wikilink]]` 互相引用，这是知识图谱的边
- **raw 不可变**：绝不修改 `raw/` 下的文件
- **来源可追溯**：每个 wiki 页的 frontmatter `sources[]` 必须记录贡献来源
- **人类策展，LLM 维护**：用户负责选来源、提问题、定方向；LLM 负责所有写作、交叉引用和簿记
- **log 格式**：每条日志以 `## [YYYY-MM-DD] 操作类型 | 标题` 开头，可用 `grep "^## \[" wiki/log.md` 解析

## 版本管理

- 本 vault 是 git 仓库（远程：GitHub `baiiiij/LLM_obsidian_wiki`）
- 完成一轮 ingest 后 commit 并 push，commit message 简要说明变更
