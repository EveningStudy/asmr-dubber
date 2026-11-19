"""Persistent batch queue editing and the core reference selection handshake."""

import inspect
import json
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from ..platforms import portable_home
from ..storage import atomic_write_text
from . import batch_catalog, batch_plan, batch_queue
from .settings import current, reference_upload


class Batch:
    def __init__(self, media):
        self.media = media
        self.path = portable_home() / "config" / "batch-queue.json"
        self.lock = threading.RLock()
        self.reference_path = self.path.with_name("batch-reference.json")
        self.finished = set()
        self.running = False

    def reference(self, external=None, text="", language="auto", clear=False):
        with self.lock:
            if clear:
                self.reference_path.unlink(missing_ok=True)
            elif external is not None:
                stored = reference_upload(external)
                value = {
                    "audio": stored,
                    "text": text,
                    "language": language,
                    "plans": [
                        p["plan_id"]
                        for p in self.list()
                        if not self.running or p["plan_id"] not in self.finished
                    ],
                }
                atomic_write_text(self.reference_path, json.dumps(value, ensure_ascii=False))
            value = (
                json.loads(self.reference_path.read_text(encoding="utf-8"))
                if self.reference_path.is_file()
                else {}
            )
        return {**value, "url": self.media.register(value.get("audio"))}

    def background(self, folder, background):
        return {
            "preview": self.media.register(
                batch_catalog.background_preview_path(folder, background)
            )
        }

    def list(self):
        with self.lock:
            from .batch_compat import normalize_subtitle_plan

            old = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else []
            queue = [normalize_subtitle_plan(p) for p in old]
            if queue != old:
                self._save(queue)
            return queue

    def _save(self, queue):
        atomic_write_text(self.path, json.dumps(queue, ensure_ascii=False))
        if self.reference_path.is_file():
            value = json.loads(self.reference_path.read_text(encoding="utf-8"))
            value["plans"] = [
                p["plan_id"] for p in queue if not self.running or p["plan_id"] not in self.finished
            ]
            atomic_write_text(self.reference_path, json.dumps(value, ensure_ascii=False))
        return queue

    def scan(self, folder, include_bonus=None):
        result = asdict(batch_catalog.scan_for_ui(folder, include_bonus, settings=current()))
        result["preview"] = self.media.register(result.get("selected_background_preview"))
        return result

    def edition(self, folder, edition, include_bonus=False):
        return asdict(
            batch_catalog.preview_edition_for_ui(folder, edition, include_bonus, settings=current())
        )

    def tracks(self, folder, sources, order):
        return asdict(
            batch_catalog.reorder_tracks_for_ui(folder, sources, order, settings=current())
        )

    def subtitle(self, folder, sources, track, file, language, mode="direct"):
        return asdict(
            batch_catalog.set_track_subtitle_for_ui(
                folder, sources, track, file, language, mode, settings=current()
            )
        )

    def save(
        self,
        folder,
        edition,
        sources,
        mode,
        layout,
        background="black",
        embed_subtitles=False,
        rebuild=False,
        content=None,
        subtitle_language=None,
        editing="",
    ):
        defaults = current()
        content = content or defaults.autoflow_default_content
        subtitle_language = subtitle_language or defaults.autoflow_subtitle_language
        plan = batch_plan.build_plan_for_ui(
            folder,
            edition,
            sources,
            mode,
            layout,
            background,
            embed_subtitles,
            rebuild,
            subtitles_only=content in {"subtitles", "source_subtitles"},
            source_subtitles_only=content == "subtitles",
            subtitle_language=subtitle_language,
            task_content=content,
            settings=defaults,
        )
        with self.lock:
            queue = self.list()
            queue = (
                batch_queue.replace_plan_in_queue(queue, editing, plan)
                if editing
                else batch_queue.add_plan_to_queue(queue, plan)
            )
            return self._save(queue)

    def edit(self, identifier):
        result = asdict(batch_plan.edit_plan_for_ui(self.list(), identifier, settings=current()))
        result["preview"] = self.media.register(result.get("selected_background_preview"))
        return result

    def remove(self, identifier):
        with self.lock:
            return self._save(batch_queue.remove_plan_from_queue(self.list(), identifier))

    def reorder(self, order):
        with self.lock:
            return self._save(batch_queue.reorder_queue_for_ui(self.list(), order))

    def restart(self, identifier):
        with self.lock:
            return self._save(batch_queue.toggle_plan_rebuild(self.list(), identifier))

    def preflight(self):
        """The checks a batch start runs, on demand: disk, video memory, key and models."""

        from .batch_plan import deserialize_plan
        from .batch_preflight import run_preflight

        queue = self.list()
        if not queue:
            raise ValueError("队列没有作品：请先添加作品，再做开始前检查。")
        return run_preflight([deserialize_plan(item) for item in queue]).as_dict()

    def diagnostics(self, identifier="", outcomes=None):
        """Zip the logs, settings (no keys), failed sentences and doctor output."""

        from ..autoflow import engine
        from . import diagnostics
        from .batch_plan import deserialize_plan
        from .settings import current as current_settings

        queue = [item for item in self.list() if not identifier or item["plan_id"] == identifier]
        plans = []
        seen_ids: set[str] = set()
        for item in queue:
            try:
                plan = deserialize_plan(item)
                plans.append(plan)
                seen_ids.add(plan.plan_id)
            except Exception:
                continue
        fallback_ids = list((outcomes or {}).keys())
        if identifier and identifier not in fallback_ids:
            fallback_ids.append(identifier)
        for plan_id in fallback_ids:
            if not plan_id or (identifier and plan_id != identifier) or plan_id in seen_ids:
                continue
            meta = engine.load_plan_metadata(plan_id)
            entry = (outcomes or {}).get(plan_id) or {}
            folder = meta.get("source_folder") or entry.get("work")
            output = meta.get("output_folder") or entry.get("output") or ""
            if folder or meta.get("jobs") or entry:
                plans.append(
                    SimpleNamespace(
                        plan_id=plan_id,
                        folder=Path(str(folder or plan_id)),
                        output_root=Path(str(output)) if output else Path("."),
                    )
                )
                seen_ids.add(plan_id)
        path, names = diagnostics.build_bundle(
            plans, current_settings().model_dump(mode="json"), outcomes=outcomes
        )
        return {"path": str(path), "url": self.media.register(path), "files": names}

    def _clear_rebuild_flag(self, plan_id: str) -> None:
        with self.lock:
            queue = self.list()
            changed = False
            for item in queue:
                if item.get("plan_id") == plan_id and (item.get("rebuild") or item.get("force")):
                    item["rebuild"] = False
                    item["force"] = False
                    changed = True
            if changed:
                self._save(queue)

    def run(self, request, report, token, reference):
        queue = self.list()
        with self.lock:
            self.finished = set()
            self.running = True
        if request.get("plan"):
            queue = [item for item in queue if item["plan_id"] == request["plan"]]
        if not queue:
            self.running = False
            raise ValueError("队列没有作品：请先添加作品，再开始处理。此次未处理任何文件。")
        outcomes = {}
        reference(
            {
                "kind": "queue",
                "plans": [
                    {
                        "plan_id": p["plan_id"],
                        "work": Path(p["folder"]).name,
                        "output": p["output_root"],
                    }
                    for p in queue
                ],
            }
        )
        report("开始处理队列。", 0, len(queue))

        current_plan = {"id": ""}

        def progress_event(message):
            if message.strip():
                report(message.strip(), len(self.finished), len(queue))

        def stage_event(stage):
            if current_plan["id"]:
                reference({"kind": "plan_stage", "plan_id": current_plan["id"], "stage": stage})

        def reference_event(payload):
            kind = payload.get("kind")
            if kind == "plan_started":
                current_plan["id"] = payload["plan_id"]
            if kind == "plan_rebuild_started":
                if payload.get("plan_id"):
                    self._clear_rebuild_flag(payload["plan_id"])
            elif kind == "plan_finished":
                current_plan["id"] = ""
                with self.lock:
                    self.finished.add(payload["plan_id"])
                    outcomes[payload["plan_id"]] = payload
                if payload.get("status") in {"completed", "partial"}:
                    self._clear_rebuild_flag(payload["plan_id"])
                reference(payload)
            elif kind == "prepare":
                chosen = self.reference()
                if payload.get("plan_id") in chosen.get("plans", []):
                    from ..autoflow.engine import apply_shared_reference

                    apply_shared_reference(Path(payload["project_json"]), chosen)
            else:
                reference(payload)

        run_kwargs = {
            "cancel_event": token,
            "reference_event_callback": reference_event,
            "progress_callback": progress_event,
            "stage_callback": stage_event,
        }
        parameters = inspect.signature(batch_queue.run_queue).parameters
        if not any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
            run_kwargs = {key: value for key, value in run_kwargs.items() if key in parameters}
        try:
            result, outputs = batch_queue.run_queue(queue, **run_kwargs)
        finally:
            with self.lock:
                self.running = False
        report("队列处理完成。", len(queue), len(queue))
        result_payload = {
            "exit_code": result,
            "plans": [plan["plan_id"] for plan in queue],
            "outputs": outputs,
            "outcomes": outcomes,
            "subtitles": batch_queue.subtitle_output_rows(outputs),
        }
        if result:
            failed_entries = [
                entry for entry in outcomes.values() if entry.get("status") == "failed"
            ]
            first_entry = next(
                (entry for entry in failed_entries if entry.get("error")),
                failed_entries[0] if failed_entries else {},
            )
            first_error = str(first_entry.get("error") or "").strip()
            if len(failed_entries) == 1 and first_error:
                work_name = str(first_entry.get("work") or "").strip()
                result_payload["error"] = (
                    f"{work_name}：{first_error}" if work_name else first_error
                )
            elif first_error:
                result_payload["error"] = (
                    f"队列中有 {len(failed_entries)} 个作品失败，首个错误：{first_error}"
                )
            else:
                result_payload["error"] = "队列中有任务失败，请检查日志后重试。"
            if first_entry.get("error_code"):
                result_payload["error_code"] = first_entry["error_code"]
                if first_entry.get("error_hint"):
                    result_payload["error_hint"] = first_entry["error_hint"]
            elif first_error:
                from ..errors import classify_error

                code, hint = classify_error(RuntimeError(first_error))
                result_payload["error_code"] = code
                result_payload["error_hint"] = hint
        return result_payload
