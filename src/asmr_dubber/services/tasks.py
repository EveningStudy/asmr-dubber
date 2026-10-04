"""Persistent cancellable jobs; workers reuse the existing core checkpoints."""

import copy
import json
import logging
import threading
import time
import uuid

from ..app_logging import redact_sensitive
from ..errors import InstallPausedError, OperationCancelledError
from ..platforms import portable_home
from ..storage import atomic_write_text
from ..task_control import CancellationToken, cancellation_scope, check_cancelled

ACTIVE = {"queued", "running", "cancelling"}
HISTORY = 50
logger = logging.getLogger(__name__)


class Tasks:
    def __init__(self, execute, path=None):
        self.execute = execute
        self.path = path or portable_home() / "runtime" / "tasks.json"
        self.lock = threading.RLock()
        self.tokens = {}
        self.items = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        )
        for task in self.items.values():
            if task["status"] in ACTIVE:
                task.update(status="interrupted", message="Interrupted; saved results are retained")
        if self.items:
            self._prune()
            self._save()

    def _prune(self):
        finished = sorted(
            (task for task in self.items.values() if task["status"] not in ACTIVE),
            key=lambda task: task.get("updated_at", 0),
        )
        for task in finished[:-HISTORY]:
            del self.items[task["id"]]

    def _save(self):
        atomic_write_text(self.path, json.dumps(self.items, ensure_ascii=False))

    def list(self):
        with self.lock:
            return copy.deepcopy(list(self.items.values()))

    def get(self, identifier):
        with self.lock:
            return copy.deepcopy(self.items[identifier])

    def assert_idle(self, resource):
        with self.lock:
            if any(
                t["resource"] == resource and t["status"] in ACTIVE for t in self.items.values()
            ):
                raise ValueError("A task is already running for this resource")

    def start(self, request, resource, identifier=None):
        with self.lock:
            self.assert_idle(resource)
            # Secrets are resolved in services, never put into task snapshots.
            if any("key" in name.lower() or "secret" in name.lower() for name in request):
                raise ValueError("Secrets must not be stored in tasks")
            identifier = identifier or uuid.uuid4().hex
            self.items[identifier] = {
                "id": identifier,
                "kind": request["kind"],
                "resource": resource,
                "status": "queued",
                "current": 0,
                "total": 0,
                "message": "",
                "logs": [],
                "result": None,
                "error": None,
                "request": copy.deepcopy(request),
                "started_at": time.time(),
                "updated_at": time.time(),
                "reference": None,
            }
            token = self.tokens[identifier] = CancellationToken()
            self._prune()
            self._save()
            threading.Thread(
                target=self._run,
                args=(identifier, token),
                daemon=True,
                name=f"asmr-task-{identifier[:8]}",
            ).start()
            return self.get(identifier)

    def _run(self, identifier, token):
        last_save = 0.0

        def report(message="", current=0, total=0, **kwargs):
            nonlocal last_save
            if "desc" in kwargs:
                if isinstance(message, tuple):
                    current, total = message
                elif isinstance(message, (int, float)):
                    current, total = message, 1
                message = kwargs["desc"]
            with self.lock:
                task = self.items[identifier]
                message = redact_sensitive(message)
                task.update(message=message, current=current, total=total, updated_at=time.time())
                if message and (not task["logs"] or task["logs"][-1] != str(message)):
                    task["logs"] = [*task["logs"][-199:], str(message)]
                if time.monotonic() - last_save > 0.3:
                    self._save()
                    last_save = time.monotonic()
            check_cancelled(token)

        def reference(payload):
            with self.lock:
                self.items[identifier]["reference"] = (
                    dict(payload) if payload.get("kind") == "ready" else None
                )
                self._save()

        try:
            with self.lock:
                self.items[identifier]["status"] = "running"
                self._save()
                request = copy.deepcopy(self.items[identifier]["request"])
            with cancellation_scope(token):
                check_cancelled(token)
                result = self.execute(request, report, token, reference)
                check_cancelled(token)
            with self.lock:
                failure = result.get("error") if isinstance(result, dict) else None
                self.items[identifier].update(
                    status="failed" if failure else "completed",
                    error=redact_sensitive(failure) if failure else None,
                    result=result,
                    reference=None,
                )
                if not failure:
                    task = self.items[identifier]
                    task["current"] = task["total"] = task["total"] or 1
        except (OperationCancelledError, InstallPausedError) as exc:
            with self.lock:
                self.items[identifier].update(status="cancelled", message=str(exc), reference=None)
        except Exception as exc:
            logger.exception("Task %s failed", identifier)
            with self.lock:
                self.items[identifier].update(
                    status="failed", error=redact_sensitive(exc), reference=None
                )
        finally:
            with self.lock:
                self.items[identifier]["updated_at"] = time.time()
                self.tokens.pop(identifier, None)
                self._save()

    def cancel(self, identifier):
        with self.lock:
            token = self.tokens.get(identifier)
            if token is None or self.items[identifier]["status"] not in ACTIVE:
                return self.get(identifier)
            self.items[identifier]["status"] = "cancelling"
            self._save()
        token.set()
        return self.get(identifier)

    def resume(self, identifier):
        task = self.get(identifier)
        if task["status"] not in {"cancelled", "interrupted", "failed"}:
            raise ValueError("This task cannot be resumed")
        request = task["request"]
        if request.get("project"):
            from ..models import load_project

            project, _ = load_project(request["project"])
            request["revision"] = project.revision
        return self.start(request, task["resource"], identifier)

    def close(self):
        for identifier in tuple(self.tokens):
            self.cancel(identifier)
