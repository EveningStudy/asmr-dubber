"""Checks run before a batch starts, so it does not fail hours in.

Each check answers one question in plain Chinese: is there enough disk, enough
video memory, a working translation key and every model the queue needs?  A
failed check blocks the start; a warning is shown but does not.

Every message is a fixed template filled with ``{name}`` values.  The templates
are also keys of the English catalog, so the page can translate a message it
receives; ``TEMPLATES`` lists them for the test that keeps both in step.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from ..app_logging import redact_sensitive
from ..platforms import portable_home

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
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
    DISK_SUBTITLES,
    DISK_FAIL,
    DISK_TIGHT,
    DISK_OK,
    VRAM_NONE,
    VRAM_ONLINE,
    VRAM_CPU,
    VRAM_NO_GPU,
    VRAM_LOW,
    VRAM_TIGHT,
    VRAM_OK,
    TRANSLATION_NONE,
    TRANSLATION_NO_KEY,
    TRANSLATION_KEY_ONLY,
    TRANSLATION_FAIL,
    TRANSLATION_OK,
    MODELS_NONE,
    MODELS_UNKNOWN,
    MODELS_MISSING,
    MODELS_PARTIAL,
    MODELS_OK,
    SEPARATION_EXTERNAL,
    SEPARATION_ERROR,
    SEPARATION_OK,
    SEPARATION_MISSING,
    CHECK_BROKEN,
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

    if plan.source_subtitles_only:
        return 0
    policy = plan.edition.get("output_policy") or {}
    if plan.subtitles_only:
        return int(seconds * BYTES_PER_SECOND_ORIGINAL_ONLY)
    variants = max(1, len(policy.get("variants") or []))
    separated = bool((policy.get("settings") or {}).get("separation_enabled")) or (
        "replace" in (policy.get("variants") or [])
    )
    per_second = BYTES_PER_SECOND_DUBBED * variants + (
        BYTES_PER_SECOND_SEPARATED if separated else 0
    )
    return int(seconds * per_second)


def check_disk(plans: list[Any]) -> list[Check]:
    title = "磁盘空间"
    needs: dict[int, list[Any]] = {}
    total_seconds = 0.0
    total_need = 0
    work_home = portable_home()
    for plan in plans:
        seconds = plan_seconds(plan)
        need = estimate_bytes(plan, seconds)
        total_seconds += seconds
        total_need += need
        if not need:
            continue
        # Working files live under the program folder; delivered files in the output.
        for path, share in ((work_home, 0.6), (Path(plan.output_root), 0.4)):
            free, drive = _disk_free(path)
            entry = needs.setdefault(drive, [free, 0, path])
            entry[1] += int(need * share)
    if not total_need:
        return [_check("disk", SKIP, title, DISK_SUBTITLES)]
    scale = {"count": len(plans), "hours": f"{total_seconds / 3600:.1f}"}
    checks = []
    for free, need, path in needs.values():
        required = need + SAFETY_MARGIN
        values = {
            "drive": Path(path).anchor or str(path),
            "free": _gb(free),
            "need": _gb(required),
            **scale,
        }
        if free < required:
            checks.append(_check("disk", FAIL, title, DISK_FAIL, **values))
        elif free < required * 1.3:
            checks.append(_check("disk", WARN, title, DISK_TIGHT, **values))
    return checks or [
        _check("disk", OK, title, DISK_OK, need=_gb(total_need + SAFETY_MARGIN), **scale)
    ]


def _policy_settings(plans: list[Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for plan in plans:
        merged.update((plan.edition.get("output_policy") or {}).get("settings") or {})
    return merged


def _needs_asr(plans: list[Any]) -> bool:
    for plan in plans:
        if plan.source_subtitles_only:
            continue
        for source in plan.sources:
            timed = source.transcript_path and source.transcript_timed
            if not (timed and source.transcript_mode == "direct"):
                return True
    return False


def check_models(plans: list[Any], user: Any, policy: dict[str, Any]) -> list[Check]:
    from .. import runtime_manager
    from ..model_registry import ASR_BACKENDS, TTS_BACKENDS

    if all(plan.source_subtitles_only for plan in plans):
        return [_check("models", SKIP, "所需模型", MODELS_NONE)]
    wanted = []
    if _needs_asr(plans):
        wanted.append(("识别模型", ASR_BACKENDS.get(user.asr_backend), user.asr_backend))
    if any(not plan.subtitles_only for plan in plans):
        backend = str(policy.get("tts_backend") or user.tts_backend)
        wanted.append(("配音模型", TTS_BACKENDS.get(backend), backend))
    checks: list[Check] = []
    for title, spec, backend_id in wanted:
        if spec is None:
            checks.append(_check("models", FAIL, title, MODELS_UNKNOWN, backend=backend_id))
            continue
        status = runtime_manager.backend_status(spec, settings=user)
        values = {
            "backend": spec.label,
            "state": status.label,
            "detail": f"（{status.detail}）" if status.detail else "",
        }
        if status.state in {"missing", "broken", "incompatible"}:
            checks.append(_check("models", FAIL, title, MODELS_MISSING, **values))
        elif status.state == "partial":
            checks.append(_check("models", WARN, title, MODELS_PARTIAL, **values))
        else:
            checks.append(_check("models", OK, title, MODELS_OK, **values))
    variants = ((p.edition.get("output_policy") or {}).get("variants") or [] for p in plans)
    if any("replace" in item for item in variants):
        checks.append(_check_separation(policy))
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
    if ready:
        return _check("separation", OK, title, SEPARATION_OK, model=model)
    return _check("separation", FAIL, title, SEPARATION_MISSING)


def check_vram(plans: list[Any], user: Any, policy: dict[str, Any]) -> Check:
    from .. import runtime_manager
    from ..model_registry import ASR_BACKENDS, TTS_BACKENDS

    title = "显存"
    if all(plan.source_subtitles_only for plan in plans):
        return _check("vram", SKIP, title, VRAM_NONE)
    needed: list[Any] = []
    if _needs_asr(plans):
        needed.append(ASR_BACKENDS.get(user.asr_backend))
    if any(not plan.subtitles_only for plan in plans):
        needed.append(TTS_BACKENDS.get(str(policy.get("tts_backend") or user.tts_backend)))
    local = [spec for spec in needed if spec and spec.runtime != "http" and spec.minimum_vram_gb]
    if not local:
        return _check("vram", SKIP, title, VRAM_ONLINE)
    # Stages run one after another, so the largest single requirement counts.
    hardest = max(local, key=lambda spec: spec.minimum_vram_gb or 0)
    minimum = float(hardest.minimum_vram_gb or 0)
    recommended = float(hardest.recommended_vram_gb or minimum)
    hardware = runtime_manager.detect_hardware()
    if not hardware.cuda_available or hardware.vram_gb is None:
        if "cpu" in hardest.devices:
            return _check("vram", WARN, title, VRAM_CPU, backend=hardest.label)
        return _check(
            "vram", FAIL, title, VRAM_NO_GPU, backend=hardest.label, minimum=f"{minimum:g}"
        )
    have = float(hardware.vram_gb)
    gpu = hardware.gpu or "显卡"
    if have < minimum:
        return _check(
            "vram",
            FAIL,
            title,
            VRAM_LOW,
            backend=hardest.label,
            minimum=f"{minimum:g}",
            gpu=gpu,
            have=f"{have:.1f}",
        )
    if have < recommended:
        return _check(
            "vram",
            WARN,
            title,
            VRAM_TIGHT,
            backend=hardest.label,
            recommended=f"{recommended:g}",
            have=f"{have:.1f}",
        )
    return _check("vram", OK, title, VRAM_OK, gpu=gpu, have=f"{have:.1f}")


def check_translation(plans: list[Any], user: Any, network: bool) -> Check:
    from .. import backend_diagnostics
    from ..errors import AsmrDubberError
    from ..user_settings import PROVIDER_PRESETS, resolve_api_key

    title = "翻译服务"
    if all(plan.source_subtitles_only for plan in plans):
        return _check("translation", SKIP, title, TRANSLATION_NONE)
    provider = user.translation_provider
    label = str(PROVIDER_PRESETS.get(provider, {}).get("label", provider))
    try:
        key = resolve_api_key(provider)
    except AsmrDubberError:
        return _check("translation", FAIL, title, TRANSLATION_NO_KEY, provider=label)
    if not network:
        return _check("translation", OK, title, TRANSLATION_KEY_ONLY, provider=label)
    try:
        backend_diagnostics.test_translation_api(user, key)
    except Exception as exc:
        reason = redact_sensitive(exc)[:300]
        return _check("translation", FAIL, title, TRANSLATION_FAIL, provider=label, reason=reason)
    return _check("translation", OK, title, TRANSLATION_OK, provider=label)


def run_preflight(
    plans: list[Any], *, settings: Any | None = None, network: bool = True
) -> PreflightReport:
    """Run every check; a check that itself breaks is reported, not raised."""

    from ..user_settings import load_user_settings

    user = settings or load_user_settings()
    policy = _policy_settings(plans)
    checks: list[Check] = []
    steps = (
        ("disk", lambda: check_disk(plans)),
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
