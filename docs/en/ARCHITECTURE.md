English | [中文](../ARCHITECTURE.md)

[Documentation](INDEX.md) · [README](../../README.en.md)

# Native interface and service architecture

The interface is native HTML/CSS/JS, without frontend build tooling or Gradio. User operations are in the [guide](USER_GUIDE.md).

```text
EXE/scripts → ui.py → http_server.py → API contracts/routes
            → services → pipeline/ASR/translation/TTS/audio/autoflow
            → project and portable state
frontend JSON → API → services → core
CLI → core workflows
```

API validates/routes requests; interface-independent services compose core behavior/persistence. Browser uses token-protected JSON and registered media capabilities. Revision/locks protect concurrent changes.

| Modules | Responsibility |
|---|---|
| frontend index/styles | Shell/layout |
| projects/project_dialogs | Four steps, imports/references/review |
| forms | Catalog-driven controls/autosave |
| session/app | Requests/save queue/polling/navigation/localization |
| models/batch/settings | Model/queue/settings/cleanup/diagnostics UI |
| api_contract/api_contracts/api | Request types/validation/routes |
| services parameters/layout | Single catalog composed from domain types/defaults and layout/conditions |
| services settings/projects/project_* | Scoped configuration/project views/edits/actions/references |
| services review/models/model_status | Review operations, runtime/model integrity and management |
| services batch* / tasks | Plans/queue/checkpoints, start/progress/cancel/resume/history |
| models/lifecycle/storage | Schema/revision/invalidation/locks/atomic persistence |
| registry/runtime manager | Capabilities/install contracts |

Defaults copy at creation. Keys are separate, not in task snapshots; project task results store manifest and UI reloads. Consumed uploads are removed; abandoned uploads expire. Media registration indexes canonical paths. History keeps 50 finished tasks.

Long-operation exclusion protects runtime changes; token/process cancellation preserves completed core checkpoints. Restart marks active tasks interrupted. Cache signatures include relevant text/time/reference/engine/parameters. Cleanup fingerprints cross JSON as strings to preserve precision and are rechecked with locks. Complete recorded model files determine readiness/removal.

Loopback Host allow-list; remote Basic authentication; write token/Origin checks; registered media paths; request/upload limits. It is not an arbitrary file/public anonymous server.

Validate with pytest/Ruff/Pyright and optional real browser regressions native_viewport.cjs/native_cleanup.cjs. Contracts do not prove inference/listening quality. Runtime prompts remain code assets; [PROMPTS](PROMPTS.md).
