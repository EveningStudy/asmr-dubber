"""Regenerate bilingual parameter documentation from the application catalog."""

import json
from pathlib import Path

from asmr_dubber.localization import catalog as locale_catalog
from asmr_dubber.services.parameters import catalog as parameter_catalog

root = Path(__file__).resolve().parents[1]
loc = locale_catalog()
params = parameter_catalog()


def cell(v):
    if isinstance(v, (dict, list, bool)) or v is None:
        v = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return (
        str(v)
        .replace("|", "&#124;")
        .replace("\n", "<br>")
        .replace("`", "&#96;")
        .replace(str(root), "<program>")
        .replace(root.as_posix(), "<program>")
    )


def default(p, en):
    v = p["default"]
    if isinstance(v, str) and len(v) > 160:
        return (
            "Built-in template; see [PROMPTS](PROMPTS.md)"
            if en
            else "内置模板，见 [Prompt](PROMPTS.md)"
        )
    return "`" + cell(v) + "`" if v != "" else ("Empty" if en else "空")


def conditions(p):
    d = p["visible_when"]
    a = p.get("visible_any", [])
    s = " AND ".join(f"{k} in {cell(v)}" for k, v in d.items()) or "Always"
    if a:
        s += (
            " AND ("
            + " OR ".join(" AND ".join(f"{k} in {cell(v)}" for k, v in x.items()) for x in a)
            + ")"
        )
    return s


panels = {
    "sep": ("人声分离", "Separation"),
    "asr": ("识别与复核", "Recognition and review"),
    "tr": ("翻译", "Translation"),
    "tts": ("配音", "Synthesis"),
    "mix": ("混音与空间跟随", "Mixing and spatial follow"),
    "sub": ("字幕", "Subtitles"),
    "general": ("通用", "General"),
    "batch": ("批量", "Batch"),
}
for en in (False, True):
    lines = [
        "English | [中文](../PARAMETERS.md)" if en else "中文 | [English](en/PARAMETERS.md)",
        "",
        "[Documentation](INDEX.md) · [Configuration](CONFIGURATION.md)"
        if en
        else "[文档索引](INDEX.md) · [配置讲解](CONFIGURATION.md)",
        "",
        "# Complete parameter reference" if en else "# 完整参数参考",
        "",
        (
            "Generated from `services/parameters.py`: **{count} fields**. "
            "Schema defaults differ from saved values. Backend/language choices can set "
            "effective models, prompts, devices and voices; see the feature chapters."
        )
        if en
        else (
            "按当前 `services/parameters.py` 清单生成，**{count} 个字段全部列出**。"
            "这是 schema 默认值，不是已保存值。后端/语言可能带入不同模型、Prompt、设备或音色；"
            "操作讲解见对应功能章节。"
        ),
        "",
        (
            "Project fields appear in project panels/new defaults; global fields affect "
            "future projects/queue entries. Always means no static visibility condition; "
            "expand advanced groups. Options can depend on backend. Validation also checks "
            "cross-field dependencies, language and JSON contracts."
        )
        if en
        else (
            "项目字段出现在项目/新默认中；全局字段作用于全局/新项目/新队列。"
            "Always 表示无静态显示条件，高级分组仍需展开。选项可能随后端变化；"
            "校验还包括跨字段依赖、语言能力与 JSON 合同。"
        ),
        "",
        (
            "Paths use `<program>` instead of this checkout. Empty prompts use built-in "
            "language templates. Legacy fields are listed for completeness; do not edit "
            "migration counters or obsolete review prompts for ordinary operations."
        )
        if en
        else (
            "路径用 `<program>` 代替本机路径。空 Prompt 使用内置语言模板。"
            "迁移/旧复核字段也列出，但普通操作不应修改兼容计数器或旧复核 Prompt。"
        ),
        "",
    ]
    for panel, (zh, eng) in panels.items():
        lines += [
            "## " + (eng if en else zh),
            "",
            "| Key / label | Type / scope | Default | Range / options | Display condition |"
            if en
            else "| 键 / 界面名称 | 类型 / 范围归属 | 默认值 | 范围 / 可选值 | 显示条件 |",
            "|---|---|---|---|---|",
        ]
        for p in params:
            if p["panel"] != panel:
                continue
            ranges = []
            for k, sign in [
                ("minimum", "≥"),
                ("maximum", "≤"),
                ("exclusiveMinimum", ">"),
                ("exclusiveMaximum", "<"),
                ("minItems", "min items "),
                ("maxItems", "max items "),
                ("minLength", "min length "),
                ("maxLength", "max length "),
            ]:
                if k in p:
                    ranges.append(sign + str(p[k]))
            if "pattern" in p:
                ranges.append(p["pattern"])
            if p.get("options"):
                ranges.append(", ".join(str(x[1]) for x in p["options"]))
            elif p.get("options_source"):
                ranges.append("Backend-dependent" if en else "随所选后端")
            elif p.get("enum"):
                ranges.append(", ".join(map(str, p["enum"])))
            label = loc.get(p["label"], p["label"]) if en else p["label"]
            if isinstance(label, dict):
                label = label.get("en", p["label"])
            lines.append(
                f"| `{p['key']}`<br>{cell(label)} | {p.get('type', 'string')}"
                f"{' / nullable' if p.get('nullable') else ''}<br>{p['scope']} "
                f"| {default(p, en)} | {cell('; '.join(ranges) or '—')} | {conditions(p)} |"
            )
        lines += [""]
    path = root / "docs" / ("en/PARAMETERS.md" if en else "PARAMETERS.md")
    path.write_text(
        ("\n".join(lines) + "\n").replace("{count}", str(len(params))), encoding="utf-8"
    )
    print(path, len(lines))
