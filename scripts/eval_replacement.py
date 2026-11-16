"""Benchmark for the full-dub ("replacement") output.

    python scripts/eval_replacement.py render <eval-root> <take> [--set '{"k": v}'] [--no-tts]
    python scripts/eval_replacement.py report <eval-root>

``<eval-root>/projects/*`` are ordinary ASMR Dubber projects that already have
sentences and translations.  ``render`` applies setting overrides, synthesizes
what is missing, mixes the replacement output and stores it as a named take in
``<eval-root>/takes/<take>/``.  ``report`` measures every take against the
source and writes ``report.json`` plus a listening page ``index.html`` with
synchronized A/B playback.  Nothing here calls a paid service.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BANDS = [(100, 400), (400, 1500), (1500, 4000), (4000, 8000), (8000, 12000), (12000, 20000)]


# ----------------------------------------------------------------- rendering
def render(root: Path, take: str, overrides: dict, synthesize: bool) -> None:
    from asmr_dubber.filtering import has_speakable_text
    from asmr_dubber.models import load_project, save_project
    from asmr_dubber.pipeline import mix_project, synthesize_project

    destination = root / "takes" / take
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "settings.json").write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for directory in sorted((root / "projects").iterdir()):
        if not (directory / "project.json").is_file():
            continue
        clip = directory.name.split("_")[0]
        project, project_dir = load_project(directory)
        values = {
            "separation_enabled": True,
            "separation_mix_mode": "replace",
            "mix_output_mode": "both",
            **overrides,
        }
        project.settings = type(project.settings).model_validate(
            {**project.settings.model_dump(), **values}
        )
        save_project(project, project_dir)
        dubbed = [s for s in project.sentences if s.enabled and has_speakable_text(s.zh_text)]
        if not dubbed:
            print(f"{clip}: no dubbed lines, skipped")
            continue
        if synthesize:
            synthesize_project(project, project_dir)
            project, project_dir = load_project(directory)
        mix_project(project, project_dir, output_variant="replace")
        project, project_dir = load_project(directory)
        shutil.copyfile(project_dir / project.output_file, destination / f"{clip}.mix.wav")
        shutil.copyfile(project_dir / project.chinese_stem_file, destination / f"{clip}.stem.wav")
        events = _events(project, project_dir)
        report_file = project_dir / "mix" / "experimental-report.json"
        if report_file.is_file():
            windows = json.loads(report_file.read_text(encoding="utf-8")).get("windows")
            if windows is not None:
                (destination / f"{clip}.windows.json").write_text(
                    json.dumps(windows), encoding="utf-8"
                )
        (destination / f"{clip}.events.json").write_text(json.dumps(events), encoding="utf-8")
        print(f"{clip}: rendered {len(events)} lines")


def _events(project, project_dir: Path) -> list[dict]:
    from asmr_dubber.audio import sentence_events

    settings = project.settings
    replacing = settings.separation_mix_mode == "replace"
    events = sentence_events(
        project_dir,
        project.sentences,
        getattr(settings, "replacement_dubbing_offset_ms", 0)
        if replacing
        else settings.chinese_dubbing_offset_ms,
        getattr(settings, "replacement_max_auto_speed", 1.2)
        if replacing
        else settings.chinese_max_auto_speed,
        getattr(settings, "replacement_timing_mode", settings.chinese_dubbing_timing_mode)
        if replacing
        else settings.chinese_dubbing_timing_mode,
        trim_silence=replacing,
    )
    return [
        {
            "id": e.sentence_id,
            "start": e.start_seconds,
            "duration": e.effective_duration_seconds,
            "source_start": e.source_start_seconds,
            "source_end": e.source_end_seconds,
            "speed": e.speed_factor,
        }
        for e in events
    ]


# ------------------------------------------------------------------- metrics
def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if x.size else 0.0


def _read(path: Path, rate: int, frames: int | None = None) -> np.ndarray:
    data, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if source_rate != rate:
        data = soxr.resample(data, source_rate, rate, quality="HQ")
    if frames is not None:
        data = np.pad(data, ((0, max(0, frames - len(data))), (0, 0)))[:frames]
    return data


def _band_powers(x: np.ndarray, rate: int) -> np.ndarray:
    """Mean power per analysis band of a mono signal (Welch, 2048-point)."""
    size = 2048
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    hop = size // 2
    count = 1 + (len(x) - size) // hop
    window = np.hanning(size)
    frames = np.stack([x[i * hop : i * hop + size] * window for i in range(count)])
    power = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(size, 1 / rate)
    return np.array([power[(freqs >= lo) & (freqs < hi)].mean() + 1e-20 for lo, hi in BANDS])


def _voiced_ratio(x: np.ndarray, rate: int) -> float:
    """Share of active frames with a clear pitch.  A whisper scores near zero."""
    mono = soxr.resample(x.astype(np.float32), rate, 16000) if rate != 16000 else x
    size, hop = 640, 160
    if len(mono) < size:
        return 0.0
    count = 1 + (len(mono) - size) // hop
    frames = np.stack([mono[i * hop : i * hop + size] for i in range(count)]).astype(np.float64)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    active = energy > max(energy.max() * 0.1, 1e-6)
    if not active.any():
        return 0.0
    frames = frames[active] - frames[active].mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(frames * np.hanning(size), 2048, axis=1)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)
    correlation /= np.maximum(correlation[:, :1], 1e-20)
    peak = correlation[:, 40:230].max(axis=1)  # 70-400 Hz
    return float(np.mean(peak > 0.45))


def _itd_ms(stereo: np.ndarray, rate: int) -> float:
    if len(stereo) < 256:
        return 0.0
    size = 1 << (2 * len(stereo)).bit_length()
    left, right = np.fft.rfft(stereo[:, 0], size), np.fft.rfft(stereo[:, 1], size)
    freqs = np.fft.rfftfreq(size, 1 / rate)
    cross = left * np.conj(right)
    cross = np.where((freqs > 150) & (freqs < 1600), cross / np.maximum(np.abs(cross), 1e-12), 0)
    correlation = np.fft.irfft(cross, size)
    limit = round(rate * 0.001)
    lags = np.concatenate((np.arange(0, limit + 1), np.arange(-limit, 0)))
    return float(lags[np.argmax(correlation[lags])] / rate * 1000)


def _ild_db(stereo: np.ndarray) -> float:
    return _db(_rms(stereo[:, 0])) - _db(_rms(stereo[:, 1]))


def _slice(x: np.ndarray, rate: int, start: float, end: float) -> np.ndarray:
    return x[max(0, round(start * rate)) : max(0, round(end * rate))]


def measure_clip(source: Path, vocals: Path, take: Path, clip: str) -> dict | None:
    mix_path, stem_path = take / f"{clip}.mix.wav", take / f"{clip}.stem.wav"
    if not mix_path.is_file():
        return None
    events = json.loads((take / f"{clip}.events.json").read_text(encoding="utf-8"))
    rate = sf.info(source).samplerate
    original = _read(source, rate)
    frames = len(original)
    mix, stem, voice = (_read(p, rate, frames) for p in (mix_path, stem_path, vocals))
    # What the mix did to the original, with the dub itself taken out.
    change = mix - stem - original
    inside = np.zeros(frames, dtype=bool)
    windows_file = take / f"{clip}.windows.json"
    if windows_file.is_file():
        # The real replaced windows (plus their 30 ms fades), as the mixer reported them.
        for lo, hi in json.loads(windows_file.read_text(encoding="utf-8")):
            inside[max(0, round((lo - 0.02) * rate)) : round((hi + 0.02) * rate)] = True
    for e in events:
        lo = max(0, round((e["source_start"] - 0.35) * rate))
        inside[lo : round((e["source_end"] + 0.45) * rate)] = True
    outside = ~inside
    rows = []
    for e in events:
        src = _slice(voice, rate, e["source_start"], e["source_end"])
        dub = _slice(stem, rate, e["start"], e["start"] + e["duration"])
        if len(src) < rate // 20 or len(dub) < rate // 20 or _rms(dub) < 1e-7:
            continue
        src_mono, dub_mono = src.mean(axis=1), dub.mean(axis=1)
        src_bands, dub_bands = _band_powers(src_mono, rate), _band_powers(dub_mono, rate)
        difference = 10 * np.log10(dub_bands / src_bands)
        shape = difference - np.mean(difference[:4])  # level-independent spectral shape
        envelope = np.abs(dub_mono)
        onset = int(np.argmax(envelope > envelope.max() * 0.05)) / rate
        kept = _slice(mix - stem, rate, e["source_start"], e["source_end"])
        rows.append(
            {
                "id": e["id"],
                "speed": e["speed"],
                "onset_ms": (e["start"] + onset - e["source_start"]) * 1000,
                "level_db": _db(_rms(dub)) - _db(_rms(src)),
                "ild_error_db": abs(_ild_db(dub) - _ild_db(src)),
                "itd_error_ms": abs(_itd_ms(dub, rate) - _itd_ms(src, rate)),
                "spectrum_error_db": float(np.mean(np.abs(shape[:5]))),
                "air_db": float(shape[4]),
                "top_db": float(shape[5]),
                "voiced_source": _voiced_ratio(src_mono, rate),
                "voiced_dub": _voiced_ratio(dub_mono, rate),
                "original_left_db": _db(_rms(kept))
                - _db(_rms(_slice(original, rate, e["source_start"], e["source_end"]))),
            }
        )

    def median(key: str, absolute: bool = False) -> float | None:
        values = [abs(r[key]) if absolute else r[key] for r in rows]
        return round(float(np.median(values)), 2) if values else None

    return {
        "lines": len(rows),
        "outside_change_dbfs": round(_db(_rms(change[outside])), 1) if outside.any() else None,
        "outside_seconds": round(float(outside.sum() / rate), 1),
        "original_left_in_lines_db": median("original_left_db"),
        "onset_ms": median("onset_ms"),
        "speed_over_1_15": round(float(np.mean([r["speed"] > 1.15 for r in rows])), 2)
        if rows
        else None,
        "level_error_db": median("level_db", True),
        "ild_error_db": median("ild_error_db"),
        "itd_error_ms": median("itd_error_ms"),
        "spectrum_error_db": median("spectrum_error_db"),
        "air_8_12k_db": median("air_db"),
        "top_12_20k_db": median("top_db"),
        "voiced_gap": round(
            float(np.mean([abs(r["voiced_dub"] - r["voiced_source"]) for r in rows])), 2
        )
        if rows
        else None,
        "rows": rows,
    }


LABELS = {
    "outside_change_dbfs": "配音句以外被改动的量（dBFS，越低越好，< −90 为未改动）",
    "original_left_in_lines_db": "配音句内保留的原声（相对原声 dB，越低越干净）",
    "onset_ms": "配音起点相对原句起点（ms，接近 0 为好）",
    "speed_over_1_15": "加速超过 1.15 倍的句子比例",
    "level_error_db": "配音与原人声的响度差（dB）",
    "ild_error_db": "左右电平差误差（dB）",
    "itd_error_ms": "左右时间差误差（ms）",
    "spectrum_error_db": "频谱形状误差 100 Hz–12 kHz（dB）",
    "air_8_12k_db": "8–12 kHz 相对原人声（dB，负数表示气声不足）",
    "top_12_20k_db": "12–20 kHz 相对原人声（dB）",
    "voiced_gap": "发声方式差距（0 = 耳语/实声比例与原声一致）",
}


def report(root: Path) -> None:
    takes = sorted(p for p in (root / "takes").iterdir() if p.is_dir())
    takes.sort(key=lambda p: p.stat().st_ctime)
    results: dict[str, dict] = {}
    clips: list[tuple[str, Path]] = []
    for directory in sorted((root / "projects").iterdir()):
        if not (directory / "project.json").is_file():
            continue
        clip = directory.name.split("_")[0]
        source = next(directory.glob("source.*"))
        vocals = sorted(directory.glob("analysis/separation/*/vocals.wav"))
        if not vocals:
            continue
        clips.append((clip, source))
        for take in takes:
            measured = measure_clip(source, vocals[-1], take, clip)
            if measured:
                results.setdefault(take.name, {})[clip] = measured
    summary = {}
    for take, per_clip in results.items():
        summary[take] = {}
        for key in LABELS:
            values = [c[key] for c in per_clip.values() if c.get(key) is not None]
            summary[take][key] = round(float(np.mean(values)), 2) if values else None
    (root / "report.json").write_text(
        json.dumps({"summary": summary, "clips": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_page(root, [t.name for t in takes], clips, summary)
    header = f"{'':44}" + "".join(f"{t.name[:14]:>16}" for t in takes)
    print(header)
    for key in LABELS:
        cells = "".join(f"{summary.get(t.name, {}).get(key)!s:>16}" for t in takes)
        print(f"{key:44}{cells}")
    print(f"\n试听页：{root / 'index.html'}")


def _write_page(root: Path, takes: list[str], clips: list[tuple[str, Path]], summary: dict):
    rows = "".join(
        "<tr><td>{}</td>{}</tr>".format(
            html.escape(label),
            "".join(f"<td>{summary.get(t, {}).get(key, '')}</td>" for t in takes),
        )
        for key, label in LABELS.items()
    )
    sections = []
    for clip, source in clips:
        tracks = [("原声", source.relative_to(root).as_posix())]
        for take in takes:
            path = root / "takes" / take / f"{clip}.mix.wav"
            if path.is_file():
                tracks.append((take, path.relative_to(root).as_posix()))
        if len(tracks) < 2:
            continue
        buttons = "".join(
            f'<button data-i="{i}">{html.escape(name)}</button>'
            for i, (name, _) in enumerate(tracks)
        )
        audios = "".join(
            f'<audio preload="auto" src="{html.escape(p)}"></audio>' for _, p in tracks
        )
        sections.append(
            f'<section class="clip"><h2>{clip}</h2><div class="bar">{buttons}'
            f'<button class="stop">停止</button></div>'
            f'<input type="range" min="0" max="1000" value="0">{audios}</section>'
        )
    page = f"""<!doctype html><meta charset="utf-8"><title>完全配音版对比试听</title>
<style>
body{{font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;max-width:980px;margin:24px auto;
padding:0 16px;background:#16181c;color:#e8e8e8}}
h1{{font-size:20px}}h2{{font-size:16px;margin:0 0 8px}}
.clip{{background:#20242b;border-radius:10px;padding:14px;margin:14px 0}}
button{{font:inherit;margin:0 6px 6px 0;padding:6px 14px;border-radius:8px;border:1px solid #444;
background:#2b3038;color:#e8e8e8;cursor:pointer}}
button.on{{background:#3b82f6;border-color:#3b82f6}}
input[type=range]{{width:100%}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border-bottom:1px solid #333;padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}
p{{color:#aaa}}
</style>
<h1>完全配音版对比试听</h1>
<p>点按钮在同一时间点切换版本，建议戴耳机。数字是自动指标，最终以听感为准。</p>
{"".join(sections)}
<h2>指标（各素材平均）</h2>
<table><tr><th></th>{"".join(f"<th>{html.escape(t)}</th>" for t in takes)}</tr>{rows}</table>
<script>
document.querySelectorAll('.clip').forEach(sec=>{{
  const audios=[...sec.querySelectorAll('audio')];
  const buttons=[...sec.querySelectorAll('button[data-i]')];
  const range=sec.querySelector('input');let current=-1;
  function play(i){{
    const t=current>=0?audios[current].currentTime:range.value/1000*(audios[i].duration||0);
    document.querySelectorAll('audio').forEach(a=>a.pause());
    document.querySelectorAll('button.on').forEach(b=>b.classList.remove('on'));
    current=i;audios[i].currentTime=t;audios[i].play();buttons[i].classList.add('on');
  }}
  buttons.forEach((b,i)=>b.onclick=()=>play(i));
  sec.querySelector('.stop').onclick=()=>{{audios.forEach(a=>a.pause());
    buttons.forEach(b=>b.classList.remove('on'));}};
  audios.forEach((a,i)=>a.ontimeupdate=()=>{{if(i===current&&a.duration)
    range.value=a.currentTime/a.duration*1000;}});
  range.oninput=()=>{{const i=Math.max(current,0);
    audios[i].currentTime=range.value/1000*(audios[i].duration||0);}};
}});
</script>
"""
    (root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("render")
    r.add_argument("root", type=Path)
    r.add_argument("take")
    r.add_argument("--set", default="{}", help="JSON object of project setting overrides")
    r.add_argument("--set-file", type=Path, help="JSON file of project setting overrides")
    r.add_argument("--no-tts", action="store_true", help="reuse existing dubs, only remix")
    p = commands.add_parser("report")
    p.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "render":
        overrides = json.loads(args.set)
        if args.set_file:
            overrides.update(json.loads(args.set_file.read_text(encoding="utf-8")))
        render(args.root.resolve(), args.take, overrides, not args.no_tts)
    else:
        report(args.root.resolve())


if __name__ == "__main__":
    main()
