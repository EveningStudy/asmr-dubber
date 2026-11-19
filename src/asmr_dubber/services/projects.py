"""Project snapshots and revision checked editing operations."""

import logging
from pathlib import Path

from .. import pipeline
from ..audio import probe_audio, verify_source
from ..lifecycle import browser_revision_scope
from ..models import load_project
from . import project_audio, project_operations, project_records, project_review, review
from .media import discard_upload
from .settings import current

logger = logging.getLogger(__name__)


class Projects:
    def __init__(self, media):
        self.media = media

    def get(self, project):
        active, directory = load_project(project)
        return self._serialize(active, directory, project)

    def _register_project_file(self, root: Path, stored: str | None) -> str | None:
        if not stored:
            return None
        candidate = (root / stored).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            logger.warning("Skipping path outside project directory: %s", stored)
            return None
        return self.media.register(candidate) if candidate.is_file() else None

    def _collect_outputs(self, active, root: Path) -> dict:
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
                candidate = (root / stored).resolve()
                candidate.relative_to(root)
                outputs[name] = {"name": candidate.name, "url": self.media.register(candidate)}
        return outputs

    def _serialize(self, active, directory, project_ref=None):
        result = active.model_dump(mode="json")
        manifest_str = str(directory / "project.json")
        project_root = directory.resolve()
        result.update(
            manifest=manifest_str,
            title=pipeline._PROJECT_STAMP.sub("", directory.name),
            rows=project_records.project_rows(active),
        )
        result["source_url"] = self.media.register(verify_source(directory, active.source))
        external_reference_url = None
        raw_external = (active.settings.tts_external_reference_audio or "").strip()
        if raw_external:
            external_reference = Path(raw_external).expanduser().resolve()
            if external_reference.is_file():
                try:
                    probe_audio(external_reference)
                except Exception:
                    external_reference_url = None
                else:
                    external_reference_url = self.media.register(external_reference)
        result["external_reference_url"] = external_reference_url
        result["reference_external"] = active.settings.tts_reference_source == "external" or (
            active.settings.tts_backend in {"indextts2", "indextts2_5"}
            and active.settings.tts_index_speaker_source == "external"
        )
        result["outputs"] = self._collect_outputs(active, project_root)
        result["replacement_report"] = project_review.replacement_flags(active, directory)
        result["compare_tracks"] = project_review.compare_tracks(
            active, directory, result["source_url"], self.media.register
        )
        for sentence in result["sentences"]:
            sentence.setdefault("zh_text", sentence.get("target_text", ""))
            sentence["tts_url"] = self._register_project_file(
                project_root, sentence.get("tts_file")
            )
            for take in sentence.get("tts_candidates") or []:
                take["url"] = self._register_project_file(project_root, take.get("file"))
        result["diagnostics"] = project_records.diagnostics(active)
        from ..translation import empty_translation_is_expected
        from ..translation_backup import backup_summary

        result["translation_backup"] = backup_summary(directory)

        result["recoverable_translations"] = [
            s.id
            for s in active.sentences
            if s.status == "skipped_filler"
            and not s.enabled
            and not s.zh_text
            and not empty_translation_is_expected(s)
        ]
        choices, selected, _ = project_audio.reference_picker(
            project_ref or manifest_str, include_preview=False
        )
        result["reference_choices"], result["reference_selected"] = choices, selected
        return result

    def list(self):
        result = []
        for label, manifest in project_records.recent_projects(current().projects_root or None):
            try:
                active, _ = load_project(manifest)
            except Exception as exc:
                # One unreadable manifest must not hide every other project.
                logger.warning("Skipping unreadable project %s: %s", manifest, exc)
                result.append({"label": label, "manifest": manifest, "error": str(exc)})
                continue
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
        from ..models import save_project

        target_id = str(row[0] or "").strip() if row else ""
        with browser_revision_scope(project, revision):
            active, directory = load_project(project)
            before_count = len(active.sentences)
            changed = project_records.apply_single_row(active, row)
            deleted = len(active.sentences) < before_count
            save_project(active, directory)
            if changed:
                pipeline.export_transcript(active, directory)
        project_root = directory.resolve()
        sentences_payload = active.model_dump(mode="json")["sentences"]
        for sentence in sentences_payload:
            sentence.setdefault("zh_text", sentence.get("target_text", ""))
            sentence["tts_url"] = self._register_project_file(
                project_root, sentence.get("tts_file")
            )
            for take in sentence.get("tts_candidates") or []:
                take["url"] = self._register_project_file(project_root, take.get("file"))
        updated_row = (
            None
            if deleted
            else next((item for item in sentences_payload if item["id"] == target_id), None)
        )
        return {
            "manifest": str(directory / "project.json"),
            "revision": active.revision,
            "settings": active.settings.model_dump(mode="json"),
            "outputs": self._collect_outputs(active, project_root),
            "deleted": deleted,
            "row": updated_row,
            "rows": project_records.project_rows(active),
            "sentences": sentences_payload,
        }

    def waveform(self, project, start, end, bins=160):
        from ..audio import extract_waveform_peaks

        active, directory = load_project(project)
        source_file = verify_source(directory, active.source)
        t0 = max(0.0, float(start))
        t1 = min(active.source.duration_seconds, max(t0 + 0.05, float(end)))
        return {
            "start": t0,
            "end": t1,
            "duration": active.source.duration_seconds,
            "peaks": extract_waveform_peaks(source_file, t0, t1, bins=bins),
        }

    def proposals(self, project, accept, reject, revision):
        """Accept or reject pending translation proposals and save the result."""

        from ..models import save_project
        from ..translation_revision import resolve_proposals

        with browser_revision_scope(project, revision):
            active, directory = load_project(project)
            counts = resolve_proposals(active, accept, reject)
            if counts["accepted"] or counts["rejected"] or counts["stale"]:
                save_project(active, directory)
                pipeline.export_transcript(active, directory)
        return {**self.get(project), "proposal_result": counts}

    def take(self, project, sentence, attempt, revision):
        """Make one of a sentence's candidate dubs the active one (F1)."""

        from ..models import save_project

        with browser_revision_scope(project, revision):
            active, directory = load_project(project)
            project_operations.choose_dub_candidate(active, directory, sentence, attempt)
            save_project(active, directory)
            pipeline.export_transcript(active, directory)
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
                discard_upload(external)
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
            discard_upload(request["source"])
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
            if kind == "synthesize" and request.get("candidates") and request.get("sentence_ids"):
                project_operations.generate_dub_candidates(
                    manifest,
                    rows,
                    request["sentence_ids"],
                    int(request["candidates"]),
                    report,
                    token,
                )
            elif kind == "synthesize" and request.get("sentence_ids"):
                project_operations.regenerate_dubs(
                    manifest, rows, request["sentence_ids"], report, token
                )
            elif kind == "translate" and request.get("compress"):
                project_operations.compress_translations(
                    manifest, rows, request.get("sentence_ids") or None, report, token
                )
            elif kind == "translate" and request.get("sentence_ids"):
                project_operations.retranslate(
                    manifest, rows, request["sentence_ids"], report, token
                )
            elif kind in actions:
                if kind == "translate" and request.get("force"):
                    from ..user_settings import resolve_api_key

                    pipeline.translate_project(
                        active,
                        directory,
                        api_key=""
                        if active.source_language == active.settings.tts_target_language
                        else resolve_api_key(active.settings.translation_provider),
                        force=True,
                        progress=report,
                        cancel_event=token,
                        overwrite_edited=bool(request.get("overwrite_edited")),
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
                discard_upload(request.get("file"))
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
            elif kind == "restore_translations":
                from ..translation_backup import restore_project_translations

                restore_project_translations(manifest)
                return {**self.get(manifest), "message": "已恢复上一版译文。"}
            elif kind == "revise_translations":
                from ..translation_revision import revise_project_translations

                outcome = revise_project_translations(
                    active,
                    directory,
                    request.get("sentence_ids", []),
                    request.get("instruction", ""),
                    include_edited=bool(request.get("include_edited")),
                    progress=report,
                    cancel_event=token,
                )
                return {**self.get(manifest), "message": outcome.summary()}
            elif kind == "preview_reference":
                return {
                    "url": self.media.register(
                        project_audio.preview_reference(manifest, request.get("sentence"))
                    )
                }
            else:
                raise ValueError("未知的项目操作。")
        return self.get(manifest)
