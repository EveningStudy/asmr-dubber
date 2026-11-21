"""Application operations composed from the project, model and batch services."""

import os
from pathlib import Path

from .. import __version__, app_logging, backend_diagnostics, cache_cleanup, runtime_health
from ..localization import catalog
from ..models import load_project
from ..platforms import portable_home
from ..user_settings import resolve_api_key, saved_service_key
from . import models, onboarding, parameters, presets, settings, updates
from .batch import Batch
from .media import MediaStore
from .project_audio import preview_edge_tts_voice
from .projects import Projects
from .tasks import ACTIVE, Tasks


class Application:
    def __init__(self, task_path=None):
        self.media = MediaStore()
        self.projects = Projects(self.media)
        self.batch = Batch(self.media)
        self.tasks = Tasks(self.execute, task_path)

    def bootstrap(self):
        saved = settings.current()
        keys = settings.key_statuses()
        onboarding_needed = onboarding.required(saved, keys)
        return {
            "version": __version__,
            "parameters": parameters.catalog(),
            "choices": parameters.choices(),
            "settings": saved.model_dump(mode="json"),
            "keys": keys,
            "onboarding_required": onboarding_needed,
            "asr_upgrade_prompt": onboarding.asr_upgrade_prompt(
                saved, onboarding_needed=onboarding_needed
            ),
            "presets": presets.catalog(),
            "tasks": self.tasks.list(),
            "queue": self.batch.list(),
            "locales": catalog(),
            "labels": catalog("zh"),
            "health_backends": runtime_health.BACKENDS,
        }

    def start(self, request):
        kind = request["kind"]
        if kind == "batch":
            self._require_batch_queue(request)
        if request.get("project"):
            project, directory = load_project(request["project"])
            if request.get("revision") != project.revision:
                raise ValueError("项目已在别处改动，请先重新载入项目再开始任务。")
            resource = str((directory / "project.json").resolve())
        else:
            resource = (
                "batch"
                if kind == "batch"
                else "models"
                if kind in {"download", "import_models", "repair", "update"}
                else kind
            )
        with self.tasks.lock:
            self._assert_ready(kind, resource)
            return self.tasks.start(request, resource)

    def _assert_ready(self, kind, resource):
        active = [task for task in self.tasks.list() if task["status"] in ACTIVE]
        independent = {"preview_reference", "preview_edge", "health", "diagnostic"}
        if kind not in independent and any(task["kind"] not in independent for task in active):
            raise ValueError("请等待当前操作结束。")
        if (kind == "batch" and active) or (
            any(task["kind"] == "batch" for task in active) and kind != "preview_reference"
        ):
            raise ValueError("请等待当前批量操作结束。")
        if (resource == "models" and active) or any(
            task["resource"] == "models" for task in active
        ):
            raise ValueError("请等待当前运行环境操作结束。")

    def resume(self, identifier):
        with self.tasks.lock:
            task = self.tasks.get(identifier)
            if task["kind"] == "batch":
                self._require_batch_queue(task["request"])
            self._assert_ready(task["kind"], task["resource"])
            return self.tasks.resume(identifier)

    def _require_batch_queue(self, request):
        queue = self.batch.list()
        if request.get("plan"):
            queue = [p for p in queue if p["plan_id"] == request["plan"]]
        if not queue:
            raise ValueError("队列没有作品：请先添加作品，再开始处理。此次未启动批量任务。")

    def update_settings(self, changes, project=None, revision=None):
        if project:
            self.tasks.assert_idle(str(Path(project).resolve()))
        return settings.update(changes, project, revision)

    def save_table(self, project, rows, revision):
        self.tasks.assert_idle(str(Path(project).resolve()))
        return self.projects.table(project, rows, revision)

    def save_sentence(self, project, row, revision):
        self.tasks.assert_idle(str(Path(project).resolve()))
        return self.projects.sentence(project, row, revision)

    def resolve_proposals(self, project, revision, accept, reject):
        self.tasks.assert_idle(str(Path(project).resolve()))
        return self.projects.proposals(project, accept, reject, revision)

    def choose_take(self, project, revision, sentence, attempt):
        self.tasks.assert_idle(str(Path(project).resolve()))
        return self.projects.take(project, sentence, attempt, revision)

    def set_speaker_reference(
        self,
        project,
        speaker_id,
        mode="auto",
        sentence="",
        external="",
        external_name="",
        text="",
        language="auto",
        revision=None,
    ):
        self.tasks.assert_idle(str(Path(project).resolve()))
        return self.projects.speaker_reference(
            project,
            speaker_id=speaker_id,
            mode=mode,
            sentence=sentence,
            external=external,
            external_name=external_name,
            text=text,
            language=language,
            revision=revision,
        )

    def remove_model(self, model):
        if any(task["status"] in ACTIVE for task in self.tasks.list()):
            raise ValueError("请等待当前任务结束后再删除模型。")
        return models.remove(model)

    def execute(self, request, report, token, reference):
        kind = request["kind"]
        if kind == "download":
            return models.download(request, report, token)
        if kind == "import_models":
            return models.import_packs(request, report, token)
        if kind == "update":
            return updates.install(request, report, token)
        if kind == "batch":
            return self.batch.run(request, report, token, reference)
        if kind == "preview_edge":
            return {"url": self.media.register(preview_edge_tts_voice(request["voice"]))}
        if kind == "health":
            return {"message": runtime_health.check_runtime_health(request["backend"])}
        if kind == "repair":
            env_name = "ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS"
            previous = os.environ.get(env_name)
            os.environ[env_name] = "1" if settings.current().download_source == "original" else "0"
            try:
                return {
                    "message": runtime_health.repair_runtime_health(
                        request["backend"], request["action"], True
                    )
                }
            finally:
                if previous is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = previous
        if kind == "diagnostic":
            active = settings.current()
            if request.get("project"):
                active = settings.Settings.model_validate(
                    load_project(request["project"])[0].settings.model_dump()
                )
            service = request["service"]
            if service == "translation":
                result = backend_diagnostics.test_translation_api(
                    active, resolve_api_key(active.translation_provider)
                )
            elif service == "asr":
                result = backend_diagnostics.test_asr_api(
                    active, saved_service_key("asr:generic_asr_api")
                )
            else:
                result = backend_diagnostics.test_tts_api(
                    request.get("project", ""),
                    active,
                    saved_service_key(f"tts:{active.tts_backend}"),
                )
            return {"message": result}
        result = self.projects.action(request, report, token)
        # Task results are persisted and polled; the browser reloads the project itself.
        if "manifest" in result:
            return {key: result[key] for key in ("manifest", "message") if key in result}
        return result

    def storage(self):
        from ..project_origin import project_inventory
        from . import storage_report

        result = storage_report.report(settings.current())
        projects_root = Path(result["projects_root"])
        return {
            "sizes": {item["label"]: item["bytes"] for item in result["locations"]},
            "locations": result["locations"],
            "root": str(projects_root),
            "projects": project_inventory(projects_root),
        }

    def open_storage(self, key):
        from . import storage_report

        return storage_report.open_folder(key, settings.current().projects_root)

    def scan_storage(self, root, categories):
        return cache_cleanup.scan_caches(root, categories, app_home=portable_home())

    def clean_storage(self, plan, selected, confirmed):
        if any(task["status"] in ACTIVE for task in self.tasks.list()):
            raise ValueError("请等待当前任务结束后再清理缓存。")
        removed, skipped = cache_cleanup.clean_caches(plan, selected, confirmed)
        return {"removed": removed, "skipped": skipped}

    def diagnostics(self, identifier=""):
        """Bundle for a bug report; ``identifier`` limits it to one queued work."""

        outcomes = {}
        for task in sorted(
            (item for item in self.tasks.list() if item["kind"] == "batch"),
            key=lambda item: item.get("started_at", 0),
            reverse=True,
        ):
            plans = task.get("plans") or (task.get("result") or {}).get("outcomes") or {}
            for plan_id, entry in plans.items():
                if identifier and plan_id != identifier:
                    continue
                if plan_id not in outcomes:
                    outcomes[plan_id] = entry
        return self.batch.diagnostics(identifier, outcomes)

    def logs(self):
        from ..storage import atomic_write_text

        target = portable_home() / "temp" / "ASMR-Dubber.log.txt"
        source = app_logging.application_log_path()
        contents = source.read_text(encoding="utf-8", errors="replace") if source.is_file() else ""
        atomic_write_text(target, app_logging.redact_sensitive(contents))
        return {
            "text": app_logging.redact_sensitive(app_logging.recent_log_text()),
            "url": self.media.register(target),
        }
