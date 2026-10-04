"""Project snapshots and revision checked editing operations."""

from pathlib import Path

from .. import pipeline
from ..audio import verify_source
from ..lifecycle import browser_revision_scope
from ..models import load_project
from . import project_audio, project_operations, project_records, review
from .settings import current


class Projects:
    def __init__(self, media):
        self.media = media

    def get(self, project):
        active, directory = load_project(project)
        result = active.model_dump(mode="json")
        result.update(
            manifest=str(directory / "project.json"),
            title=pipeline._PROJECT_STAMP.sub("", directory.name),
            rows=project_records.project_rows(active),
        )
        result["source_url"] = self.media.register(verify_source(directory, active.source))
        external_reference = Path(active.settings.tts_external_reference_audio)
        result["external_reference_url"] = (
            self.media.register(external_reference) if external_reference.is_file() else None
        )
        result["reference_external"] = active.settings.tts_reference_source == "external" or (
            active.settings.tts_backend in {"indextts2", "indextts2_5"}
            and active.settings.tts_index_speaker_source == "external"
        )
        outputs = {}
        for name in (
            "output_file",
            "chinese_stem_file",
            "output_video_file",
            "subtitle_srt_file",
            "subtitle_lrc_file",
            "subtitle_video_file",
        ):
            stored = getattr(active, name)
            if stored:
                candidate = (directory / stored).resolve()
                candidate.relative_to(directory.resolve())
                outputs[name] = {"name": candidate.name, "url": self.media.register(candidate)}
        result["outputs"] = outputs
        for sentence in result["sentences"]:
            stored = sentence.get("tts_file")
            sentence["tts_url"] = self.media.register(directory / stored) if stored else None
        result["diagnostics"] = project_records.diagnostics(active)
        choices, selected, _ = project_audio.reference_picker(project, include_preview=False)
        result["reference_choices"], result["reference_selected"] = choices, selected
        return result

    def list(self):
        result = []
        for label, manifest in project_records.recent_projects(current().projects_root or None):
            active, _ = load_project(manifest)
            result.append(
                {
                    "label": label,
                    "manifest": manifest,
                    "source_language": active.source_language,
                    "duration": active.source.duration_seconds,
                    "sentences": len(active.sentences),
                    "translated": sum(bool(s.zh_text) for s in active.sentences),
                    "synthesized": sum(bool(s.tts_file) for s in active.sentences),
                    "exported": bool(active.output_file or active.chinese_stem_file),
                    "updated_at": str(active.updated_at),
                }
            )
        return result

    def table(self, project, rows, revision):
        with browser_revision_scope(project, revision):
            project_operations.save_table(project, rows)
        return self.get(project)

    def sentence(self, project, row, revision):
        with browser_revision_scope(project, revision):
            active, _ = load_project(project)
            rows = project_records.project_rows(active)
            matched = next((index for index, item in enumerate(rows) if item[0] == row[0]), None)
            if matched is None:
                raise ValueError("Sentence no longer exists")
            rows[matched] = row
            project_operations.save_table(project, rows)
        return self.get(project)

    def open(self, project, output=False):
        function = (
            project_records.open_project_output_directory
            if output
            else project_records.open_project_directory
        )
        return {"message": function(project)}

    def reference(
        self,
        project,
        sentence="",
        external="",
        text="",
        language="auto",
        start=None,
        end=None,
        revision=None,
    ):
        with browser_revision_scope(project, revision):
            if external:
                message, preview = project_audio.select_autoflow_external_reference(
                    project, external, text=text, language=language
                )
            elif start is not None or end is not None:
                message, preview = project_audio.select_autoflow_project_reference(
                    project, sentence, start, end, text
                )
            else:
                message, preview = project_audio.select_reference(project, sentence)
        return {
            "message": message,
            "url": self.media.register(preview),
            "project": self.get(project),
        }

    def review(self, project, window="", candidate=""):
        rows, choices, status = review.review_overview(project)
        result = {"rows": rows, "choices": choices, "status": status}
        if window:
            candidates, diff, audio = review.candidate_details(project, window, candidate)
            result.update(candidates=candidates, diff=diff, audio=self.media.register(audio))
        return result

    def action(self, request, report, token):
        kind = request["kind"]
        if kind == "create":
            result = project_operations.create_project(
                request["source"], request.get("source_language", "ja"), report, token
            )
            manifest = result.manifest
            if request.get("target_language"):
                from .settings import update

                active, _ = load_project(manifest)
                update(
                    {"tts_target_language": request["target_language"]}, manifest, active.revision
                )
            return self.get(manifest)
        manifest = request["project"]
        active, directory = load_project(manifest)
        rows = project_records.project_rows(active)
        with browser_revision_scope(manifest, request.get("revision")):
            actions = {
                "analyze": project_operations.analyze,
                "translate": project_operations.translate,
                "synthesize": project_operations.synthesize,
                "mix": project_operations.mix,
            }
            if kind in actions:
                if kind == "translate" and request.get("force"):
                    from ..user_settings import resolve_api_key

                    pipeline.translate_project(
                        active,
                        directory,
                        api_key=resolve_api_key(active.settings.translation_provider),
                        force=True,
                        progress=report,
                        cancel_event=token,
                    )
                else:
                    actions[kind](manifest, rows, report, token)
            elif kind == "import":
                project_operations.import_transcript_data(
                    manifest,
                    request.get("file"),
                    request.get("text", ""),
                    request.get("timing", "estimate"),
                    request.get("script_kind", "source"),
                    report,
                    token,
                )
            elif kind == "subtitles":
                project_operations.subtitles(
                    manifest, rows, request.get("language", "bilingual"), report, token
                )
            elif kind == "export":
                if request.get("audio", True):
                    project_operations.mix(manifest, rows, report, token)
                if request.get("language", "bilingual") != "none":
                    latest, _ = load_project(manifest)
                    project_operations.subtitles(
                        manifest,
                        project_records.project_rows(latest),
                        request.get("language", "bilingual"),
                        report,
                        token,
                    )
            elif kind in {"accept_review", "keep_review"}:
                review.apply_review(
                    manifest, request["window"], request.get("candidate", ""), kind == "keep_review"
                )
            elif kind == "undo_review":
                review.undo_review(manifest)
            elif kind == "unlock_review":
                review.unlock_review(manifest)
            elif kind == "retry_review":
                review.retry_review(manifest, rows, report, token)
            elif kind == "align_review":
                review.align_review(manifest, rows, report, token)
            elif kind == "preview_reference":
                return {
                    "url": self.media.register(
                        project_audio.preview_reference(manifest, request.get("sentence"))
                    )
                }
            else:
                raise ValueError("Unknown project action")
        return self.get(manifest)
