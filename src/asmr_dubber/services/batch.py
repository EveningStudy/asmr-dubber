"""Persistent batch queue editing and the core reference selection handshake."""

import io
import json
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict

from ..autoflow import ui_services as flow
from ..platforms import portable_home
from ..storage import atomic_write_text
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
        result = asdict(flow.scan_for_ui(folder, include_bonus, settings=current()))
        result["preview"] = self.media.register(result.get("selected_background_preview"))
        return result

    def edition(self, folder, edition, include_bonus=False):
        return asdict(
            flow.preview_edition_for_ui(folder, edition, include_bonus, settings=current())
        )

    def tracks(self, folder, sources, order):
        return asdict(flow.reorder_tracks_for_ui(folder, sources, order, settings=current()))

    def subtitle(self, folder, sources, track, file, language, mode="direct"):
        return asdict(
            flow.set_track_subtitle_for_ui(
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
        plan = flow.build_plan_for_ui(
            folder,
            edition,
            sources,
            mode,
            layout,
            background,
            embed_subtitles,
            rebuild,
            subtitle_language=subtitle_language,
            task_content=content,
            settings=current(),
        )
        with self.lock:
            queue = self.list()
            queue = (
                flow.replace_plan_in_queue(queue, editing, plan)
                if editing
                else flow.add_plan_to_queue(queue, plan)
            )
            return self._save(queue)

    def edit(self, identifier):
        return asdict(flow.edit_plan_for_ui(self.list(), identifier, settings=current()))

    def remove(self, identifier):
        with self.lock:
            return self._save(flow.remove_plan_from_queue(self.list(), identifier))

    def reorder(self, order):
        with self.lock:
            return self._save(flow.reorder_queue_for_ui(self.list(), order))

    def restart(self, identifier):
        with self.lock:
            return self._save(flow.toggle_plan_rebuild(self.list(), identifier))

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
            result, outputs = flow.run_queue(
                queue, cancel_event=token, reference_event_callback=reference
            )
        report("队列处理完成。", len(queue), len(queue))
        return {
            "exit_code": result,
            "plans": [plan["plan_id"] for plan in queue],
            "outputs": outputs,
            "subtitles": flow.subtitle_output_rows(outputs),
        }
