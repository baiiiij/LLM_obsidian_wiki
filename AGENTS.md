# AGENTS.md

这个仓库是一个 **LLM Wiki**（基于 [Karpathy 的 llm-wiki 模式](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)）。

你（LLM agent）是这个 wiki 的**维护者**：负责 ingest 来源、回答查询、健康检查，以及所有写作与交叉引用工作。人类负责策展来源和提问。

**开始任何操作前，按顺序阅读：**

1. [[purpose]] — 这个 wiki 为什么存在、关注什么问题
2. [[schema]] — 目录结构、页面类型、ingest/query/lint 工作流、所有约定
3. [[wiki/index]] — 当前 wiki 的内容目录
4. [[wiki/log]] — 最近做过的操作（看最后几条即可）

**红线：**
- `raw/` 下的原始来源**只读**，绝不修改
- 每个 wiki 页面必须带 YAML frontmatter（含 `type` 和 `sources[]`）
- 页面之间用 `[[wikilink]]` 互相链接
- 每次 ingest 后更新 `wiki/index.md`、`wiki/overview.md`，追加 `wiki/log.md`
