"""Checks run before a batch starts, so it does not fail hours in.

Each check answers one question in plain Chinese: is there enough disk, enough
video memory, a working translation key and every model the queue needs?  A
failed check blocks the start; a warning is shown but does not.

Every message is a fixed template filled with ``{name}`` values.  The templates
are also keys of the English catalog, so the page can translate a message it
receives; ``TEMPLATES`` lists them for the test that keeps both in step.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

from ..app_logging import redact_sensitive
from ..platforms import portable_home

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
TRANSLATION_PROBE_TTL_SECONDS = 300.0
_TRANSLATION_PROBE_CACHE: dict[tuple[Any, ...], float] = {}
_TRANSLATION_PROBE_LOCK = threading.Lock()


def clear_translation_probe_cache() -> None:
    """Drop cached translation API probe successes (used by tests and settings resets)."""
    with _TRANSLATION_PROBE_LOCK:
        _TRANSLATION_PROBE_CACHE.clear()


GIB = 1024**3
# Working files of one dubbed version: float32 stems, separated vocals, caches and the
# delivered audio.  A rough upper bound per second of source audio; plans differ.
BYTES_PER_SECOND_DUBBED = 1_200_000
BYTES_PER_SECOND_SEPARATED = 1_000_000
BYTES_PER_SECOND_ORIGINAL_ONLY = 250_000
UNKNOWN_DURATION_BYTES_PER_SECOND = 16_000  # compressed audio, when its length cannot be read
SAFETY_MARGIN = GIB

TITLES = ("磁盘空间", "显存", "翻译服务", "识别模型", "配音模型", "人声分离模型", "所需模型")
DISK_SUBTITLES = "这次只处理字幕，占用很小。"
DISK_FAIL = (
    "{drive} 剩余 {free}，预计需要约 {need}（{count} 个作品、约 {hours} 小时音频）。"
    "请清理磁盘，或把输出换到空间更大的位置。"
)
DISK_TIGHT = (
    "{drive} 剩余 {free}，预计需要约 {need}（{count} 个作品、约 {hours} 小时音频），空间比较紧张。"
)
DISK_OK = "够用（{count} 个作品、约 {hours} 小时音频，预计约 {need}）。"
VRAM_NONE = "这次不需要 GPU。"
VRAM_ONLINE = "所选后端是在线服务或不需要独立显存。"
VRAM_CPU = "没有检测到可用的 NVIDIA 显卡，{backend} 将用 CPU 运行，会非常慢。"
VRAM_NO_GPU = "{backend} 需要 NVIDIA 显卡（至少 {minimum} GB 显存），但没有检测到可用的显卡。"
VRAM_LOW = (
    "{backend} 至少需要 {minimum} GB 显存，当前显卡 {gpu} 只有 {have} GB。"
    "请换用更轻的后端或在线服务。"
)
VRAM_TIGHT = (
    "{backend} 建议 {recommended} GB 显存，当前 {have} GB，可能偏慢，建议关闭其他占用显存的程序。"
)
VRAM_OK = "{gpu} {have} GB，够用。"
TRANSLATION_NONE = "这次不需要翻译。"
TRANSLATION_NO_KEY = "{provider} 尚未保存 API Key；请前往设置 → 翻译服务。"
TRANSLATION_KEY_ONLY = "{provider} 的 Key 已设置（未联网测试）。"
TRANSLATION_FAIL = "{provider} 的最小测试请求失败：{reason}。请检查 Key、额度和网络后再开始。"
TRANSLATION_OK = "{provider} 测试请求通过。"
MODELS_NONE = "这次不需要识别或配音模型。"
MODELS_UNKNOWN = "未知的后端：{backend}。"
MODELS_MISSING = "{backend}：{state}{detail}。请先到“设备与模型”页安装或修复。"
MODELS_PARTIAL = "{backend}：{state}{detail}。"
MODELS_OK = "{backend}：{state}。"
SEPARATION_EXTERNAL = "使用外部分离服务。"
SEPARATION_ERROR = "无法检查分离模型：{reason}"
SEPARATION_OK = "{model} 已就绪。"
SEPARATION_MISSING = (
    "完全配音版需要人声分离，但分离环境或模型还没有安装。请先到“设备与模型”页安装，或改选“双语版”。"
)
CHECK_BROKEN = "这项检查没能完成：{reason}"
TEMPLATES = (
    *(DISK_SUBTITLES, DISK_FAIL, DISK_TIGHT, DISK_OK),
    *(VRAM_NONE, VRAM_ONLINE, VRAM_CPU, VRAM_NO_GPU, VRAM_LOW, VRAM_TIGHT, VRAM_OK),
    *(TRANSLATION_NONE, TRANSLATION_NO_KEY, TRANSLATION_KEY_ONLY, TRANSLATION_FAIL, TRANSLATION_OK),
    *(MODELS_NONE, MODELS_UNKNOWN, MODELS_MISSING, MODELS_PARTIAL, MODELS_OK),
    *(SEPARATION_EXTERNAL, SEPARATION_ERROR, SEPARATION_OK, SEPARATION_MISSING, CHECK_BROKEN),
)


@dataclass
class Check:
    id: str
    status: str
    title: str
    message: str


def _check(id: str, status: str, title: str, template: str, **values: Any) -> Check:
    return Check(id, status, title, template.format(**values))


@dataclass
class PreflightReport:
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(check.status != FAIL for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == FAIL]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}

    def summary(self) -> str:
        lines = [f"- {check.title}：{check.message}" for check in self.failures]
        return "开始前检查未通过，此次没有处理任何文件：\n" + "\n".join(lines)


def _gb(value: float) -> str:
    return f"{value / GIB:.1f} GB"


def _disk_free(path: Path) -> tuple[int, int]:
    """Free bytes and a drive key for the nearest existing parent of ``path``."""

    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free, os.stat(probe).st_dev


def plan_seconds(plan: Any) -> float:
    from ..autoflow import engine

    total = 0.0
    for source in plan.sources:
        try:
            total += (
                engine.audio_duration_samples(cast(Any, None), source.path) / engine.SAMPLE_RATE
            )
        except Exception:
            size = max(int(getattr(source, "size", 0) or 0), 0)
            total += size / UNKNOWN_DURATION_BYTES_PER_SECOND
    return total


def estimate_bytes(plan: Any, seconds: float) -> int:
    """Disk a plan needs: working files plus delivered files."""
    if getattr(plan, "source_subtitles_only", False):
        return int(seconds * BYTES_PER_SECOND_ORIGINAL_ONLY)
    policy = plan.edition.get("output_policy") or {}
    layout_factor = 2.0 if getattr(plan, "layout", "") == "both" else 1.0
    harmonized = 200_000 if getattr(plan, "mode", "") == "video_harmonized" else 0
    if getattr(plan, "subtitles_only", False):
        return int(seconds * (BYTES_PER_SECOND_ORIGINAL_ONLY + harmonized) * layout_factor)
    variants = max(1, len(policy.get("variants") or []))
    separated = bool((policy.get("settings") or {}).get("separation_enabled")) or (
        "replace" in (policy.get("variants") or [])
    )
    per_second = (BYTES_PER_SECOND_DUBBED * variants + harmonized) * layout_factor + (
        BYTES_PER_SECOND_SEPARATED if separated else 0
    )
    return int(seconds * per_second)


def check_disk(plans: list[Any], user: Any | None = None) -> list[Check]:
    title, needs, home = "磁盘空间", {}, portable_home()
    if user is None:
        from ..user_settings import load_user_settings

        try:
            user = load_user_settings()
        except Exception:
            user = None
    raw_root = str(getattr(user, "projects_root", "") or "").strip() if user else ""
    projects_home = Path(raw_root).expanduser() if raw_root else (home / "projects")
    total_seconds, total_need = 0.0, 0
    for plan in plans:
        seconds = plan_seconds(plan)
        need = estimate_bytes(plan, seconds)
        total_seconds, total_need = total_seconds + seconds, total_need + need
        targets = ((projects_home, 0.5), (home, 0.1), (Path(plan.output_root), 0.4)) if need else ()
        for path, share in targets:
            free, drive = _disk_free(path)
            needs.setdefault(drive, [free, 0, path])[1] += int(need * share)
    if not total_need:
        return [_check("disk", SKIP, title, DISK_SUBTITLES)]
    scale = {"count": len(plans), "hours": f"{total_seconds / 3600:.1f}"}
    checks = []
    for free, need, path in needs.values():
        required, anchor = need + SAFETY_MARGIN, Path(path).anchor or str(path)
        values = {"drive": anchor, "free": _gb(free), "need": _gb(required), **scale}
        if free < required:
            checks.append(_check("disk", FAIL, title, DISK_FAIL, **values))
        elif free < required * 1.3:
            checks.append(_check("disk", WARN, title, DISK_TIGHT, **values))
    if checks:
        return checks
    if all(getattr(plan, "source_subtitles_only", False) for plan in plans):
        return [_check("disk", SKIP, title, DISK_SUBTITLES)]
    return [_check("disk", OK, title, DISK_OK, need=_gb(total_need + SAFETY_MARGIN), **scale)]


def _plan_policy(plan: Any) -> dict[str, Any]:
    edition = getattr(plan, "edition", None)
    return (edition.get("output_policy") or {}) if isinstance(edition, dict) else {}


def _plan_val(plan: Any, user: Any, key: str, default: Any = None) -> Any:
    settings = _plan_policy(plan).get("settings") or {}
    if key in settings and settings[key] not in (None, ""):
        return settings[key]
    val = getattr(user, key, default)
    return default if val is None else val


def _plan_uses_separation(plan: Any, user: Any) -> bool:
    policy = _plan_policy(plan)
    enabled = (policy.get("settings") or {}).get(
        "separation_enabled", getattr(user, "separation_enabled", False)
    )
    return bool(enabled) or ("replace" in (policy.get("variants") or []))


def _policy_settings(plans: list[Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for plan in plans:
        merged.update(_plan_policy(plan).get("settings") or {})
    return merged


def _needs_asr(plans: list[Any]) -> bool:
    return any(
        not (
            getattr(s, "transcript_path", None)
            and getattr(s, "transcript_timed", False)
            and getattr(s, "transcript_mode", "direct") == "direct"
        )
        for p in plans
        if not getattr(p, "source_subtitles_only", False)
        for s in getattr(p, "sources", ())
    )


def _is_tts_plan(plan: Any) -> bool:
    return not (
        getattr(plan, "source_subtitles_only", False) or getattr(plan, "subtitles_only", False)
    )


def check_models(plans: list[Any], user: Any, policy: dict[str, Any] | None = None) -> list[Check]:
    from .. import runtime_manager
    from ..model_registry import ASR_BACKENDS, TTS_BACKENDS

    if all(getattr(plan, "source_subtitles_only", False) for plan in plans):
        return [_check("models", SKIP, "所需模型", MODELS_NONE)]
    wanted: list[tuple[str, Any, str]] = []
    if _needs_asr(plans):
        asr_ids = list(
            dict.fromkeys(str(_plan_val(p, user, "asr_backend")) for p in plans if _needs_asr([p]))
        ) or [str(user.asr_backend)]
        wanted.extend(("识别模型", ASR_BACKENDS.get(bid), bid) for bid in asr_ids)
    tts_plans = [p for p in plans if _is_tts_plan(p)]
    if tts_plans:
        tts_ids = list(dict.fromkeys(str(_plan_val(p, user, "tts_backend")) for p in tts_plans))
        wanted.extend(("配音模型", TTS_BACKENDS.get(bid), bid) for bid in tts_ids)
    checks: list[Check] = []
    for title, spec, backend_id in wanted:
        if spec is None:
            checks.append(_check("models", FAIL, title, MODELS_UNKNOWN, backend=backend_id))
            continue
        status = runtime_manager.backend_status(spec, settings=user)
        detail = f"（{status.detail}）" if status.detail else ""
        values = {"backend": spec.label, "state": status.label, "detail": detail}
        state_map = {"missing": FAIL, "broken": FAIL, "incompatible": FAIL, "partial": WARN}
        level = state_map.get(status.state, OK)
        tpl = {FAIL: MODELS_MISSING, WARN: MODELS_PARTIAL, OK: MODELS_OK}[level]
        checks.append(_check("models", level, title, tpl, **values))
    seen_sep: set[tuple[str, str]] = set()
    for plan in tts_plans:
        if not _plan_uses_separation(plan, user):
            continue
        backend = str(_plan_val(plan, user, "separation_backend", "local"))
        model = str(_plan_val(plan, user, "separation_model", "vocals_mel_band_roformer.ckpt"))
        if (backend, model) not in seen_sep:
            seen_sep.add((backend, model))
            checks.append(
                _check_separation({"separation_backend": backend, "separation_model": model})
            )
    return checks


def _check_separation(policy: dict[str, Any]) -> Check:
    from .. import separation

    title = "人声分离模型"
    if str(policy.get("separation_backend") or "local") != "local":
        return _check("separation", OK, title, SEPARATION_EXTERNAL)
    model = str(policy.get("separation_model") or "vocals_mel_band_roformer.ckpt")
    try:
        ready = (
            separation.runtime_python().is_file()
            and (separation.model_directory() / separation.local_model_name(model)).is_file()
        )
    except Exception as exc:
        return _check("separation", FAIL, title, SEPARATION_ERROR, reason=redact_sensitive(exc))
    tpl = SEPARATION_OK if ready else SEPARATION_MISSING
    return _check("separation", OK if ready else FAIL, title, tpl, model=model)


class _StageVram(NamedTuple):
    label: str
    min_gb: float
    rec_gb: float
    devices: tuple[str, ...] = ("cpu", "cuda")


def _separation_stage_vram(model: str) -> _StageVram:
    lowered = model.casefold()
    if "roformer" in lowered:
        return _StageVram("人声分离 · Mel-Band RoFormer", 3.0, 4.0)
    if "demucs_ft" in lowered:
        return _StageVram("人声分离 · Demucs FT", 2.5, 4.0)
    return _StageVram("人声分离 · Demucs" if "demucs" in lowered else "人声分离模型", 2.5, 4.0)


def _backend_stage(spec: Any) -> _StageVram | None:
    if not spec or spec.runtime == "http" or not spec.minimum_vram_gb:
        return None
    min_gb = float(spec.minimum_vram_gb)
    rec_gb = float(spec.recommended_vram_gb or min_gb)
    return _StageVram(spec.label, min_gb, rec_gb, tuple(spec.devices))


def check_vram(plans: list[Any], user: Any, policy: dict[str, Any] | None = None) -> Check:
    from .. import runtime_manager
    from ..model_registry import ASR_BACKENDS, TTS_BACKENDS

    title = "显存"
    if all(getattr(plan, "source_subtitles_only", False) for plan in plans):
        return _check("vram", SKIP, title, VRAM_NONE)
    local: list[_StageVram] = []
    for plan in plans:
        if getattr(plan, "source_subtitles_only", False):
            continue
        if _needs_asr([plan]):
            asr_id = str(_plan_val(plan, user, "asr_backend"))
            if stage := _backend_stage(ASR_BACKENDS.get(asr_id)):
                local.append(stage)
            if asr_id in {"anime_whisper", "qwen3_asr"} or bool(
                _plan_val(plan, user, "asr_forced_alignment_enabled", False)
            ):
                local.append(_StageVram("Qwen3 时间对齐", 2.5, 4.0))
        if _is_tts_plan(plan):
            if stage := _backend_stage(TTS_BACKENDS.get(str(_plan_val(plan, user, "tts_backend")))):
                local.append(stage)
            if (
                _plan_uses_separation(plan, user)
                and str(_plan_val(plan, user, "separation_backend", "local")) == "local"
                and str(_plan_val(plan, user, "separation_device", "cuda")) != "cpu"
            ):
                sep = str(
                    _plan_val(plan, user, "separation_model", "vocals_mel_band_roformer.ckpt")
                )
                local.append(_separation_stage_vram(sep))
    if not local:
        return _check("vram", SKIP, title, VRAM_ONLINE)
    hardest_min = max(local, key=lambda s: (s.min_gb, s.rec_gb))
    hardest_rec = max(local, key=lambda s: (s.rec_gb, s.min_gb))
    hardware = runtime_manager.detect_hardware()
    if not hardware.cuda_available or hardware.vram_gb is None:
        gpu_only = [s for s in local if "cpu" not in s.devices]
        if gpu_only:
            strict = max(gpu_only, key=lambda s: (s.min_gb, s.rec_gb))
            kw = {"backend": strict.label, "minimum": f"{strict.min_gb:g}"}
            return _check("vram", FAIL, title, VRAM_NO_GPU, **kw)
        return _check("vram", WARN, title, VRAM_CPU, backend=hardest_min.label)
    have, gpu = float(hardware.vram_gb), hardware.gpu or "显卡"
    if have < hardest_min.min_gb:
        kw = {
            "backend": hardest_min.label,
            "minimum": f"{hardest_min.min_gb:g}",
            "gpu": gpu,
            "have": f"{have:.1f}",
        }
        return _check("vram", FAIL, title, VRAM_LOW, **kw)
    if have < hardest_rec.rec_gb:
        kw = {
            "backend": hardest_rec.label,
            "recommended": f"{hardest_rec.rec_gb:g}",
            "have": f"{have:.1f}",
        }
        return _check("vram", WARN, title, VRAM_TIGHT, **kw)
    return _check("vram", OK, title, VRAM_OK, gpu=gpu, have=f"{have:.1f}")


def _source_skips_translation(source: Any, user: Any, target_language: str) -> bool:
    path = getattr(source, "transcript_path", None)
    lang = str(getattr(source, "transcript_language", "") or "").casefold()
    mode = str(getattr(source, "transcript_mode", "direct") or "direct")
    if (
        path
        and Path(str(path)).suffix.casefold() != ".pdf"
        and bool(getattr(source, "transcript_timed", False))
        and mode == "direct"
        and lang == target_language
    ):
        return True
    src_lang = str(
        getattr(source, "source_language", None) or getattr(user, "default_source_language", "ja")
    ).casefold()
    return src_lang == target_language and (not path or lang == target_language)


def _plan_needs_translation(plan: Any, user: Any) -> bool:
    if getattr(plan, "source_subtitles_only", False):
        return False
    policy = _plan_policy(plan)
    subtitle = str(policy.get("subtitle") or getattr(plan, "subtitle_language", ""))
    if getattr(plan, "subtitles_only", False) and subtitle == "source":
        return False
    target = str(_plan_val(plan, user, "tts_target_language", "zh")).casefold()
    sources = getattr(plan, "sources", None) or ()
    return not bool(sources) or not all(
        _source_skips_translation(source, user, target) for source in sources
    )


def check_translation(plans: list[Any], user: Any, network: bool) -> Check:
    from .. import backend_diagnostics
    from ..errors import AsmrDubberError
    from ..user_settings import PROVIDER_PRESETS, resolve_api_key

    title = "翻译服务"
    if not any(_plan_needs_translation(plan, user) for plan in plans):
        return _check("translation", SKIP, title, TRANSLATION_NONE)
    provider = user.translation_provider
    preset = PROVIDER_PRESETS.get(provider, {})
    label = str(preset.get("label", provider))
    try:
        key = resolve_api_key(provider)
    except AsmrDubberError:
        return _check("translation", FAIL, title, TRANSLATION_NO_KEY, provider=label)
    if not network:
        return _check("translation", OK, title, TRANSLATION_KEY_ONLY, provider=label)
    probe = backend_diagnostics.test_translation_api
    base_url = user.translation_base_url.strip() or str(preset.get("base_url", ""))
    model = user.translation_model.strip() or str(preset.get("default_model", ""))
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    cache_key = (provider, model, base_url, key_hash, str(portable_home()), probe)
    now = time.monotonic()
    with _TRANSLATION_PROBE_LOCK:
        cached_at = _TRANSLATION_PROBE_CACHE.get(cache_key)
        if cached_at is not None and now - cached_at < TRANSLATION_PROBE_TTL_SECONDS:
            return _check("translation", OK, title, TRANSLATION_OK, provider=label)
    try:
        probe(user, key)
    except Exception as exc:
        reason = redact_sensitive(exc)[:300]
        return _check("translation", FAIL, title, TRANSLATION_FAIL, provider=label, reason=reason)
    finished = time.monotonic()
    with _TRANSLATION_PROBE_LOCK:
        for k, ts in list(_TRANSLATION_PROBE_CACHE.items()):
            if finished - ts >= TRANSLATION_PROBE_TTL_SECONDS:
                _TRANSLATION_PROBE_CACHE.pop(k, None)
        _TRANSLATION_PROBE_CACHE[cache_key] = finished
    return _check("translation", OK, title, TRANSLATION_OK, provider=label)


def run_preflight(
    plans: list[Any], *, settings: Any | None = None, network: bool = True
) -> PreflightReport:
    """Run every check; a check that itself breaks is reported, not raised."""
    from ..user_settings import load_user_settings

    user = settings or load_user_settings()
    policy, checks = _policy_settings(plans), []
    steps = (
        ("disk", lambda: check_disk(plans, user)),
        ("vram", lambda: [check_vram(plans, user, policy)]),
        ("models", lambda: check_models(plans, user, policy)),
        ("translation", lambda: [check_translation(plans, user, network)]),
    )
    for name, step in steps:
        try:
            checks.extend(step())
        except Exception as exc:
            reason = redact_sensitive(exc)[:200]
            title = {"disk": TITLES[0], "vram": TITLES[1], "models": TITLES[6]}.get(name, TITLES[2])
            checks.append(_check(name, WARN, title, CHECK_BROKEN, reason=reason))
    return PreflightReport(checks)
