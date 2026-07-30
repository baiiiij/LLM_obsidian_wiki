# Log — 操作日志

> Append-only。每条以 `## [日期] 操作 | 标题` 开头。
> 解析示例：`grep "^## \[" wiki/log.md | tail -5`

## [2026-07-30] init | 初始化 LLM Wiki 结构

按照 [Karpathy llm-wiki 模式](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)（参考 nashsu/llm_wiki 的目录约定）初始化 vault：

- 创建三层结构：`raw/`（sources + assets）、`wiki/`（entities/concepts/sources/queries/synthesis/comparisons）、schema 层
- 编写 [[purpose]]、[[schema]]、[[AGENTS]]
- 创建 [[index]]、[[overview]]、本日志
- 配置 Obsidian 附件目录为 `raw/assets`
- 配置 git 远程：GitHub `baiiiij/LLM_obsidian_wiki`
