"""One-click diagnostic bundle: logs, settings, failed sentences and the doctor report.

The bundle is meant to be attached to a bug report, so nothing secret goes in:
``secrets.json`` is never read, every text is run through the log redaction, and
any API key the program has saved is also removed by value.  Sentence text is left
out; only ids, times and the failure reason are included.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..app_logging import application_log_path, redact_sensitive
from ..platforms import portable_home
from ..stage_result import collect_failures

TAIL_BYTES = 2 * 1024 * 1024
SECRET_NAME = re.compile(r"(?i)(key|secret|token|password|authorization|credential)")
HIDDEN = "[已隐藏]"


def known_secret_values() -> list[str]:
    """Every API key the program stores, so a stray copy in a log can be removed."""

    from ..user_settings import PROVIDER_PRESETS, saved_api_key, saved_service_key
    from .settings import key_statuses

    values: list[str] = []
    try:
        values += [saved_api_key(provider) for provider in PROVIDER_PRESETS]
        values += [saved_service_key(name) for name in key_statuses()["services"]]
    except Exception:
        pass
    for preset in PROVIDER_PRESETS.values():
        env_name = str(preset.get("env") or "")
        if env_name:
            values.append(os.environ.get(env_name, "").strip())
    for extra in ("REPLICATE_API_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        values.append(os.environ.get(extra, "").strip())
    return sorted({value for value in values if value and len(value) >= 6}, key=len, reverse=True)


def scrub_text(text: str, secrets: list[str]) -> str:
    text = redact_sensitive(text)
    for secret in secrets:
        text = text.replace(secret, HIDDEN)
    return text


def scrub_value(value: Any, secrets: list[str]) -> Any:
    """Copy a settings structure with every key-like field and secret value hidden."""

    if isinstance(value, dict):
        return {
            name: (
                HIDDEN
                if SECRET_NAME.search(str(name))
                and not str(name).endswith("tokens")
                and isinstance(value[name], str)
                and value[name]
                else scrub_value(value[name], secrets)
            )
            for name in value
        }
    if isinstance(value, list):
        return [scrub_value(item, secrets) for item in value]
    if isinstance(value, str):
        return scrub_text(value, secrets)
    return value


def _tail(path: Path, limit: int = TAIL_BYTES) -> str:
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"无法读取 {path.name}：{exc}"


def run_doctor() -> str:
    """The ``doctor`` command output; a slow or broken doctor is reported, not raised."""

    try:
        result = subprocess.run(
            [sys.executable, "-m", "asmr_dubber.cli", "doctor", "--no-network"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
            env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb", **_environment()},
        )
        return f"退出码：{result.returncode}\n{result.stdout}{result.stderr}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"doctor 没有运行完：{exc}"


def _environment() -> dict[str, str]:
    import os

    return {key: value for key, value in os.environ.items() if not SECRET_NAME.search(key)}


def failed_sentences(plan: Any) -> list[dict[str, Any]]:
    """Sentences each job of ``plan`` is missing a translation or a dub for."""

    from ..autoflow import engine
    from ..models import load_project

    rows: list[dict[str, Any]] = []
    for job in engine.load_plan_metadata(str(plan.plan_id)).get("jobs") or []:
        manifest = Path(str(job.get("project_json") or ""))
        if not manifest.is_file():
            continue
        try:
            project, _ = load_project(manifest)
        except Exception as exc:
            rows.append({"job": job.get("job_id"), "error": f"项目无法读取：{exc}"})
            continue
        for stage in ("translate", "synthesize"):
            total, failed = collect_failures(stage, project)
            sentences = {s.id: s for s in project.sentences}
            for item in failed:
                sentence = sentences[item["id"]]
                rows.append(
                    {
                        "work": Path(str(plan.folder)).name,
                        "job": job.get("job_id"),
                        "stage": stage,
                        "sentence": item["id"],
                        "start": round(sentence.start_seconds, 2),
                        "end": round(sentence.end_seconds, 2),
                        "reason": item["reason"],
                        "stage_total": total,
                    }
                )
    return rows


def _manifest_partial(plan: Any) -> list[Any]:
    path = Path(str(plan.output_root)) / "处理清单.json"
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("partial") or [])
    except (OSError, ValueError, AttributeError):
        return []


def build_bundle(
    plans: list[Any],
    settings_values: dict[str, Any],
    *,
    destination: Path | None = None,
    doctor: Callable[[], str] | None = None,
    outcomes: dict[str, Any] | None = None,
) -> tuple[Path, list[str]]:
    """Write the zip and return ``(path, names of the files inside)``."""

    from ..autoflow import engine, stages

    secrets = known_secret_values()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = destination or portable_home() / "temp" / f"ASMR-Dubber-diagnostics-{stamp}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    files["README.txt"] = (
        f"ASMR Dubber {__version__} 诊断包\n生成时间：{datetime.now().astimezone().isoformat()}\n"
        f"系统：{platform.platform()}\nPython：{platform.python_version()}\n\n"
        "内容：logs/（应用日志和每个作品自己的日志）、settings.json（不含任何 Key）、\n"
        "failed-sentences.json（失败句的编号、时间和原因，不含句子文本）、doctor.txt。\n"
        "日志已隐藏 API Key；secrets.json 从不会被打包。\n"
    )
    files["settings.json"] = json.dumps(
        scrub_value(settings_values, secrets), ensure_ascii=False, indent=2
    )
    app_log = application_log_path()
    files["logs/application.log"] = scrub_text(_tail(app_log), secrets)
    app_log_backup = app_log.with_suffix(".log.1")
    if app_log_backup.is_file():
        files["logs/application.log.1"] = scrub_text(_tail(app_log_backup), secrets)
    if engine.LOG_FILE.is_file():
        files["logs/autoflow.log"] = scrub_text(_tail(engine.LOG_FILE), secrets)
    autoflow_backup = engine.LOG_FILE.with_suffix(".log.1")
    if autoflow_backup.is_file():
        files["logs/autoflow.log.1"] = scrub_text(_tail(autoflow_backup), secrets)
    failures: list[dict[str, Any]] = []
    plan_notes: list[dict[str, Any]] = []
    for plan in plans:
        work = Path(str(plan.folder)).name
        log = stages.work_log_path(engine.STATE_ROOT, str(plan.plan_id), work)
        if log.is_file():
            files[f"logs/work-{log.stem}.log"] = scrub_text(_tail(log), secrets)
        work_backup = log.with_suffix(".log.1")
        if work_backup.is_file():
            files[f"logs/work-{log.stem}.log.1"] = scrub_text(_tail(work_backup), secrets)
        failures += failed_sentences(plan)
        plan_notes.append(
            {
                "work": work,
                "plan_id": str(plan.plan_id),
                "partial": _manifest_partial(plan),
                "outcome": (outcomes or {}).get(str(plan.plan_id)),
            }
        )
    files["failed-sentences.json"] = json.dumps(
        scrub_value({"works": plan_notes, "sentences": failures}, secrets),
        ensure_ascii=False,
        indent=2,
    )
    files["doctor.txt"] = scrub_text((doctor or run_doctor)(), secrets)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, text in files.items():
            bundle.writestr(name, text)
    return destination, list(files)
