"""Anonymous blind-listening (MUSHRA / ABX) HTML generator for ASMR Dubber evaluation."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path


def write_blind_listening_page(
    root: Path,
    candidate_takes: list[str],
    clips: list[tuple[str, Path]],
) -> Path:
    """Write ``<eval-root>/blind.html`` with deterministic per-clip shuffled labels."""
    chosen = [t for t in candidate_takes if (root / "takes" / t).is_dir()]
    mapping: dict[str, dict[str, str]] = {}
    sections: list[str] = []

    for clip, source in clips:
        available = [
            (take, (root / "takes" / take / f"{clip}.mix.wav").relative_to(root).as_posix())
            for take in chosen
            if (root / "takes" / take / f"{clip}.mix.wav").is_file()
        ]
        if len(available) < 2:
            continue
        # Deterministic shuffle per clip so reload preserves labels until revealed.
        available.sort(
            key=lambda item: hashlib.sha256(f"blind-3.0:{clip}:{item[0]}".encode()).hexdigest()
        )
        labels = [f"匿名方案 {chr(65 + idx)}" for idx in range(len(available))]
        mapping[clip] = {label: take for label, (take, _) in zip(labels, available, strict=True)}

        src_rel = html.escape(source.relative_to(root).as_posix())
        buttons = ['<button data-i="0" class="ref">日语原声（基准）</button>']
        audios = [f'<audio preload="auto" src="{src_rel}"></audio>']
        score_rows = []
        for idx, (label, (_take, rel_path)) in enumerate(
            zip(labels, available, strict=True), start=1
        ):
            esc_lbl = html.escape(label)
            buttons.append(f'<button data-i="{idx}" data-label="{esc_lbl}">{esc_lbl}</button>')
            audios.append(f'<audio preload="auto" src="{html.escape(rel_path)}"></audio>')
            score_rows.append(
                f'<tr data-clip="{clip}" data-opt="{esc_lbl}">'
                f"<td><strong>{esc_lbl}</strong> "
                f"<span class='real-name' hidden></span></td>"
                f'<td><input type="checkbox" data-k="position" checked> 位置对</td>'
                f'<td><input type="checkbox" data-k="distance" checked> 距离对</td>'
                f'<td><input type="checkbox" data-k="whisper" checked> 真耳语</td>'
                f'<td><input type="radio" name="pref-{clip}" value="{esc_lbl}"> 最喜欢</td>'
                f'<td><input type="text" data-k="note" placeholder="备注（可选）"></td>'
                f"</tr>"
            )

        sections.append(
            f'<section class="clip" data-clip="{clip}">'
            f"<h2>{clip}</h2>"
            f'<div class="bar">{"".join(buttons)}<button class="stop">停止</button></div>'
            f'<input type="range" min="0" max="1000" value="0">'
            f"{''.join(audios)}"
            f'<table class="scores">{"".join(score_rows)}</table>'
            f"</section>"
        )

    mapping_json = json.dumps(mapping, ensure_ascii=False)
    page = f"""<!doctype html><meta charset="utf-8"><title>ASMR Dubber 3.0 双盲试听台</title>
<style>
body{{font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;max-width:980px;margin:24px auto;
padding:0 16px;background:#13161b;color:#eaeef5}}
h1{{font-size:21px;margin-bottom:4px}}h2{{font-size:16px;margin:0 0 8px}}
.toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}
.clip{{background:#1d222b;border:1px solid #2d3545;border-radius:12px;padding:16px;margin:16px 0}}
button{{font:inherit;margin:0 6px 6px 0;padding:6px 14px;border-radius:8px;border:1px solid #3c4659;
background:#262d3a;color:#eaeef5;cursor:pointer}}
button.ref{{border-color:#64748b}}
button.on{{background:#2563eb;border-color:#3b82f6}}
input[type=range]{{width:100%;margin:8px 0}}
table.scores{{width:100%;border-collapse:collapse;margin-top:8px;font-size:13.5px}}
table.scores td{{padding:6px 8px;border-top:1px solid #2b3240;text-align:left}}
input[type=text]{{background:#13161b;color:#eaeef5;border:1px solid #384152;
border-radius:6px;padding:3px 8px;width:180px}}
.real-name{{color:#fbbf24;margin-left:6px;font-size:12px}}
p.sub{{color:#9aa5b8;margin:4px 0 12px}}
</style>
<h1>ASMR Dubber 3.0.0 匿名双盲验收台</h1>
<p class="sub">每段素材的候选顺序已独立打乱。戴上耳机在同一时间轴秒切对比，
勾选听感达标项并选出每段最佳方案；完成后可一键导出 JSON 或揭晓真实模型/参数名称。</p>
<div class="toolbar">
  <button id="exportBtn">导出盲听打分 JSON</button>
  <button id="revealBtn">揭晓真实版本名称</button>
  <a href="index.html" style="color:#60a5fa;align-self:center;margin-left:8px">
    打开全量指标试听页 →
  </a>
</div>
{"".join(sections)}
<script>
const KEY_MAP = {mapping_json};
const STORAGE_KEY = "asmr-dubber-3.0-blind-scores";
document.querySelectorAll('.clip').forEach(sec=>{{
  const audios=[...sec.querySelectorAll('audio')];
  const buttons=[...sec.querySelectorAll('button[data-i]')];
  const range=sec.querySelector('input[type=range]');let current=-1;
  function play(i){{
    const t=current>=0?audios[current].currentTime:range.value/1000*(audios[i].duration||0);
    document.querySelectorAll('audio').forEach(a=>a.pause());
    document.querySelectorAll('button.on').forEach(b=>b.classList.remove('on'));
    current=i;audios[i].currentTime=t;audios[i].play();buttons[i].classList.add('on');
  }}
  buttons.forEach((b,i)=>b.onclick=()=>play(i));
  sec.querySelector('.stop').onclick=()=>{{
    audios.forEach(a=>a.pause());buttons.forEach(b=>b.classList.remove('on'));
  }};
  audios.forEach((a,i)=>a.ontimeupdate=()=>{{
    if(i===current&&a.duration)range.value=a.currentTime/a.duration*1000;
  }});
  range.oninput=()=>{{
    const i=Math.max(current,0);
    audios[i].currentTime=range.value/1000*(audios[i].duration||0);
  }};
}});
function collectScores(){{
  const clips={{}};
  document.querySelectorAll('tr[data-clip]').forEach(row=>{{
    const clip=row.dataset.clip,opt=row.dataset.opt,real=(KEY_MAP[clip]||{{}})[opt]||opt;
    clips[clip]=clips[clip]||{{}};
    const pref=document.querySelector(`input[name="pref-${{clip}}"]:checked`);
    clips[clip][real]={{
      anonymous_label:opt,
      position:row.querySelector('[data-k="position"]').checked,
      distance:row.querySelector('[data-k="distance"]').checked,
      whisper:row.querySelector('[data-k="whisper"]').checked,
      preferred:pref?pref.value===opt:false,
      note:row.querySelector('[data-k="note"]').value
    }};
  }});
  return {{version:"3.0.0",timestamp:new Date().toISOString(),clips}};
}}
document.addEventListener('change',()=>
  localStorage.setItem(STORAGE_KEY,JSON.stringify(collectScores()))
);
document.getElementById('exportBtn').onclick=()=>{{
  const blob=new Blob([JSON.stringify(collectScores(),null,2)],{{type:'application/json'}});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='blind-listening-3.0.json';a.click();
}};
document.getElementById('revealBtn').onclick=()=>{{
  document.querySelectorAll('tr[data-clip]').forEach(row=>{{
    const real=(KEY_MAP[row.dataset.clip]||{{}})[row.dataset.opt];
    const span=row.querySelector('.real-name');
    if(real&&span){{span.textContent=`(${{real}})`;span.hidden=false;}}
  }});
}};
</script>
"""
    out = root / "blind.html"
    out.write_text(page, encoding="utf-8")
    return out
