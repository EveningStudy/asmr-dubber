"""Validated route bindings to application services."""

from . import api_contract as contract
from .services import files, models, settings


class API:
    def __init__(self, application):
        self.application = application
        self.routes = {
            "bootstrap": (contract.Empty, application.bootstrap),
            "projects/list": (contract.Empty, application.projects.list),
            "projects/get": (contract.ProjectRequest, application.projects.get),
            "projects/table": (contract.TableRequest, application.save_table),
            "projects/sentence": (contract.SentenceRequest, application.save_sentence),
            "projects/open": (contract.OpenRequest, application.projects.open),
            "projects/reference": (contract.ReferenceRequest, application.projects.reference),
            "review/get": (contract.ReviewRequest, application.projects.review),
            "settings/get": (contract.Empty, lambda: settings.current().model_dump(mode="json")),
            "settings/update": (contract.SettingsRequest, application.update_settings),
            "settings/key": (contract.KeyRequest, settings.key),
            "settings/keys": (contract.Empty, settings.key_statuses),
            "settings/reference-upload": (contract.FilePath, files.reference_upload),
            "files/folder": (contract.Empty, files.folder),
            "batch/open": (contract.FilePath, files.open_output),
            "models/list": (contract.Empty, models.catalog),
            "models/remove": (contract.ModelRequest, application.remove_model),
            "tasks/start": (contract.TaskRequest, self._start),
            "tasks/list": (contract.Empty, application.tasks.list),
            "tasks/get": (contract.TaskID, application.tasks.get),
            "tasks/cancel": (contract.TaskID, application.tasks.cancel),
            "tasks/resume": (contract.TaskID, application.resume),
            "batch/list": (contract.Empty, application.batch.list),
            "batch/scan": (contract.ScanRequest, application.batch.scan),
            "batch/edition": (contract.EditionRequest, application.batch.edition),
            "batch/tracks": (contract.TracksRequest, application.batch.tracks),
            "batch/subtitle": (contract.SubtitleRequest, application.batch.subtitle),
            "batch/save": (contract.PlanRequest, application.batch.save),
            "batch/edit": (contract.TaskID, application.batch.edit),
            "batch/remove": (contract.TaskID, application.batch.remove),
            "batch/restart": (contract.TaskID, application.batch.restart),
            "batch/reorder": (contract.OrderRequest, application.batch.reorder),
            "storage/get": (contract.Empty, application.storage),
            "storage/scan": (contract.StorageRequest, application.scan_storage),
            "storage/clean": (contract.CleanRequest, application.clean_storage),
            "logs/get": (contract.Empty, application.logs),
        }

    def _start(self, **request):
        return self.application.start(request)

    def call(self, route, payload):
        if route not in self.routes:
            raise KeyError("Unknown API route")
        schema, service = self.routes[route]
        request = schema.model_validate(payload).model_dump()
        return service(**request)
