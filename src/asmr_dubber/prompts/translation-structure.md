请翻译下面的“目标项”，并且只输出合法 JSON。不要输出 Markdown 代码块、解释、注释或其它文字。

输出结构必须遵守以下规则：

1. 顶层对象只能有 `translations` 一个字段。
2. `translations` 必须是数组；数组项数量必须与目标项相同。
3. 每个数组项只能有 `id`、`zh`、`kind` 三个字段，都必须是字符串。`kind` 只能是 "speech" 或 "sound"：
   有实义台词时为 "speech"，`zh` 不能为空；整句只有拟声、音效或无实义短音时为 "sound"。{{SOUND_POLICY_RULE}}
4. 必须原样保留结构模板里的全部 `id` 及其顺序，只填写每项的 `zh` 和 `kind`。
5. 没有实义的项目也不能省略，必须输出该项，`kind` 写 "sound"。
6. 不要输出输入中的 `source` 字段，也不要增加任何其它字段。

{{RETRY_NOTE}}

本批次 JSON 结构模板：

{{OUTPUT_SCHEMA}}

目标项：

{{TARGET_JSON}}
