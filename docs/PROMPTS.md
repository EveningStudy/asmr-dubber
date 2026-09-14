# 运行时 Prompt 的维护边界

`src/asmr_dubber/prompts/*.md` 虽然使用 Markdown，但会直接进入模型请求，是运行时代码资产，不是普通用户教程。文字整理不能把它们当作不影响行为的文档改写。

| 文件 | 用途 | 必须保持的合同 |
|---|---|---|
| `translation.md` | 翻译规则 | 每个输入 ID 恰好返回一次，保留顺序，正文或空中文 |
| `translation-structure.md` | 请求结构模板 | `translations` 数组及 `id`、`zh` 字段；占位符由程序填入 |
| `script-reconciliation.md` | 台本文字与识别区间匹配 | `corrections`、不重叠且顺序一致的 `script_spans` 与原有时间区间；兼容旧 `script_ids` |

多模型音频复核不使用以上 Prompt 裁决。台本重新定时使用 Python 字符索引范围 `[start,end)` 引用原文；同一台本可跨识别区间，但字符范围不得重叠、倒序或越界。程序提取原文字串，不信任模型重写的 text，不按字符比例调整音频时间。无匹配时返回空范围；原文台本模式不混入 ASR 台词，未匹配内容写入报告供人工检查。全空匹配会拒绝应用。重试附带具体冲突；失败报告保存在项目 `imports/script-reconciliation-*-error.json` 或 `imports/script-reconciliation-error.json`，其中包含私人台本文字，请勿未经检查公开。

维护时验证占位符完整、JSON 字段、ID 顺序、空值、重复台本、取消恢复及不同源语言。没有人工参考文本时，格式测试通过不代表翻译或纠错质量提高。
