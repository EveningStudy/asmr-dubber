[中文](../PROMPTS.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# Runtime prompt contracts

`src/asmr_dubber/prompts/*.md` are runtime assets, not ordinary documentation. Editing them changes model requests. Do not add documentation language-navigation links to these templates.

| File | Contract |
|---|---|
| `translation.md` | Every input ID once, original order, Chinese body or empty text |
| `translation-structure.md` | `translations` array with `id` and `zh`; required placeholders |
| `script-reconciliation.md` | `corrections` ordered by recognition ID; literal `script_quotes`, unchanged audio intervals |

Script quotes contain `id`, exact `quote` and optional literal `skip_before`. Python computes offsets; the model does not count characters. Skips are recorded. Overlap, backward allocation and invented text are rejected. Legacy `script_spans`/`script_ids` responses remain supported under their strict validators.

Malformed batches retry, then retry sentence by sentence. Unresolved items preserve prior recognition/translation and receive a persistent manual-review note. Network/auth/output-limit failures are not relabeled as successful correction. Normal empty matches and failed-validation fallback are distinct.

Reports under `imports/script-reconciliation*-error.json` contain private script text and model responses. Review before sharing. Multi-model audio review does not use these prompts.

Test placeholders, ID order, empty responses, repeated text, skipped directions, cancellation and both source languages. Schema tests establish format safety, not translation accuracy.
