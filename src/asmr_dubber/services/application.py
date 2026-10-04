"""Application operations composed from the project, model and batch services."""

import json
from importlib.resources import files
from pathlib import Path

from .. import __version__, app_logging, backend_diagnostics, cache_cleanup, runtime_health
from ..models import load_project
from ..platforms import portable_home
from ..ui_services import preview_edge_tts_voice
from ..user_settings import resolve_api_key, saved_service_key
from . import models, parameters, settings
from .batch import Batch
from .media import MediaStore
from .projects import Projects
from .tasks import ACTIVE, Tasks


class Application:
    def __init__(self, task_path=None):
        self.media = MediaStore()
        self.projects = Projects(self.media)
        self.batch = Batch(self.media)
        self.tasks = Tasks(self.execute, task_path)

    def bootstrap(self):
        return {
            "version": __version__,
            "parameters": parameters.catalog(),
            "choices": parameters.choices(),
            "settings": settings.current().model_dump(mode="json"),
            "keys": settings.key_statuses(),
            "tasks": self.tasks.list(),
            "queue": self.batch.list(),
            "locales": json.loads(
                files("asmr_dubber").joinpath("locales/en.json").read_text(encoding="utf-8")
            ),
            "health_backends": runtime_health.BACKENDS,
        }

    def start(self, request):
        kind = request["kind"]
        if request.get("project"):
            project, directory = load_project(request["project"])
            if request.get("revision") != project.revision:
                raise ValueError("Project changed; reload before starting a task")
            resource = str((directory / "project.json").resolve())
        else:
            resource = (
                "batch"
                if kind == "batch"
                else "models"
                if kind in {"download", "import_models", "repair"}
                else kind
            )
        active = [task for task in self.tasks.list() if task["status"] in ACTIVE]
        if (resource == "models" and active) or any(
            task["resource"] == "models" for task in active
        ):
            raise ValueError("Wait for the active runtime operation to finish")
        return self.tasks.start(request, resource)

    def update_settings(self, changes, project=None, revision=None):
        if project:
            self.tasks.assert_idle(str(Path(project).resolve()))
        return settings.update(changes, project, revision)

    def save_table(self, project, rows, revision):
        self.tasks.assert_idle(str(Path(project).resolve()))
        return self.projects.table(project, rows, revision)

    def remove_model(self, model):
        if any(task["status"] in ACTIVE for task in self.tasks.list()):
            raise ValueError("Wait for active tasks before deleting a model")
        return models.remove(model)

    def execute(self, request, report, token, reference):
        kind = request["kind"]
        if kind == "download":
            return models.download(request, report, token)
        if kind == "import_models":
            return models.import_packs(request, report, token)
        if kind == "batch":
            return self.batch.run(request, report, token, reference)
        if kind == "preview_edge":
            return {"url": self.media.register(preview_edge_tts_voice(request["voice"]))}
        if kind == "health":
            return {"message": runtime_health.check_runtime_health(request["backend"])}
        if kind == "repair":
            return {
                "message": runtime_health.repair_runtime_health(
                    request["backend"], request["action"], True
                )
            }
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
        return self.projects.action(request, report, token)

    def storage(self):
        home = portable_home()
        roots = {
            "模型": home / "models",
            "项目": Path(settings.current().projects_root or home / "projects"),
            "缓存": home / "cache",
        }
        return {
            "sizes": {label: models.directory_bytes(path) for label, path in roots.items()},
            "root": str(roots["项目"]),
        }

    def scan_storage(self, root, categories):
        return cache_cleanup.scan_caches(root, categories)

    def clean_storage(self, plan, selected, confirmed):
        if any(task["status"] in ACTIVE for task in self.tasks.list()):
            raise ValueError("Wait for active tasks before clearing caches")
        removed, skipped = cache_cleanup.clean_caches(plan, selected, confirmed)
        return {"removed": removed, "skipped": skipped}

    def logs(self):
        return {
            "text": app_logging.recent_log_text(),
            "url": self.media.register(app_logging.application_log_path()),
        }
