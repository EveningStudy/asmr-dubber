中文 | [English](en/PROMPTS.md)

[文档索引](INDEX.md) · [README](../README.md)

# 运行时 Prompt 的维护边界

`src/asmr_dubber/prompts/*.md` 虽然使用 Markdown，但会直接进入模型请求，是运行时代码资产，不是普通用户教程。文字整理不能把它们当作不影响行为的文档改写。

| 文件 | 用途 | 必须保持的合同 |
|---|---|---|
| `translation.md` | 翻译规则 | 每个输入 ID 恰好返回一次，保留顺序，正文或空中文 |
| `translation-structure.md` | 请求结构模板 | `translations` 数组及 `id`、`zh` 字段；占位符由程序填入 |
| `script-reconciliation.md` | 台本文字与识别区间匹配 | `corrections`、逐字 `script_quotes` 与原有时间区间；兼容旧 `script_spans` / `script_ids` |

多模型音频复核不使用以上 Prompt 裁决。台本请求使用 `script_quotes` 中的 `id`、`quote` 和可选 `skip_before`，由 Python 计算字符范围；跳过文字必须逐字匹配并写入报告。同一台本可跨识别区间，不得重叠、倒序或越界，不按字符比例猜时间。批次验证失败后逐句重试；仍失败的句子保留原识别/译文并标记待人工校对，不计为成功匹配。正常空匹配与失败降级是两种状态。网络、鉴权和输出长度错误仍明确失败。诊断保存在项目 `imports/script-reconciliation*-error.json`，其中含私人台本与模型响应，分享前脱敏。运行时 Prompt 文件不添加文档语言导航。

维护时验证占位符完整、JSON 字段、ID 顺序、空值、重复台本、取消恢复及不同源语言。没有人工参考文本时，格式测试通过不代表翻译或纠错质量提高。
