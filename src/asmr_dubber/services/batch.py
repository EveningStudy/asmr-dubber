"""Persistent batch queue editing and the core reference selection handshake."""

import io
import json
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict

from ..platforms import portable_home
from ..storage import atomic_write_text
from . import batch_catalog, batch_plan, batch_queue
from .settings import current


class Batch:
    def __init__(self, media):
        self.media = media
        self.path = portable_home() / "config" / "batch-queue.json"
        self.lock = threading.RLock()

    def list(self):
        with self.lock:
            return json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else []

    def _save(self, queue):
        atomic_write_text(self.path, json.dumps(queue, ensure_ascii=False))
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
        content="dubbing",
        subtitle_language="bilingual",
        editing="",
    ):
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
            source_subtitles_only=content == "source_subtitles",
            subtitle_language=subtitle_language,
            task_content=content,
            settings=current(),
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
        return asdict(batch_plan.edit_plan_for_ui(self.list(), identifier, settings=current()))

    def remove(self, identifier):
        with self.lock:
            return self._save(batch_queue.remove_plan_from_queue(self.list(), identifier))

    def reorder(self, order):
        with self.lock:
            return self._save(batch_queue.reorder_queue_for_ui(self.list(), order))

    def restart(self, identifier):
        with self.lock:
            return self._save(batch_queue.toggle_plan_rebuild(self.list(), identifier))

    def run(self, request, report, token, reference):
        queue = self.list()
        if request.get("plan"):
            queue = [item for item in queue if item["plan_id"] == request["plan"]]
        report("开始处理队列。", 0, len(queue))

        class LogStream(io.TextIOBase):
            def write(self, value):
                if value.strip():
                    report(value.strip(), 0, len(queue))
                return len(value)

        with redirect_stdout(LogStream()), redirect_stderr(LogStream()):
            result, outputs = batch_queue.run_queue(
                queue, cancel_event=token, reference_event_callback=reference
            )
        report("队列处理完成。", len(queue), len(queue))
        result_payload = {
            "exit_code": result,
            "plans": [plan["plan_id"] for plan in queue],
            "outputs": outputs,
            "subtitles": batch_queue.subtitle_output_rows(outputs),
        }
        if result:
            result_payload["error"] = "队列中有任务失败，请检查日志后重试。"
        return result_payload
