[中文](../AUDIO_REVIEW_TUTORIAL.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# Multi-model audio review

**Experimental; may perform worse than one model.** Review helps inspect disagreements; it is not a guaranteed accuracy upgrade. Ordinary single-model ASR does not need the proposals panel.

Flow: primary transcript → reviewers hear common audio regions → proposals → listen and decide → translation/TTS.

## Enable

Settings → ASR → Multi-model audio review:

1. Enable review. If unavailable, install an eligible local ASR model in Devices & models.
2. Select auxiliary models. More models increase runtime; the primary model also reviews clips.
3. Keep suggestion-only mode initially.
4. Keep clip/context durations at their defaults.
5. Save to Current project for an existing project, or Defaults for future projects.

![Review settings](../../assets/tutorials/review-settings.png)

These are isolated UI-demo screenshots, not accuracy measurements. The demo has no ASR models installed, so its enable control is disabled.

## Run

| Situation | Action |
|---|---|
| No transcript yet | Run ASR |
| Existing transcript; retry or add review | Retry review only |
| Review already finished | Refresh review results |

Review completion means proposals exist; suggestion-only mode does not change the sentence table. Auxiliary failures do not discard the saved primary transcript.

## Compare

Open the review proposals panel in Single project. Select an audio region, then a candidate.

![Proposals](../../assets/tutorials/review-proposals.png)

Identical text is not proof of correctness. Listen to disagreements; keep the original when evidence is insufficient. Different sentence splits are not inherently errors: comparison uses common audio regions. Unsafe boundary changes are not applied through character-based timing guesses.

![Comparison and listening](../../assets/tutorials/review-compare.png)

- Accept when the candidate better matches the recording.
- Keep and confirm when the primary transcript is better.
- Skip uncertain regions, or edit the relevant sentence manually.
- Undo reverses the last acceptance/confirmation.

Manual edits/confirmation lock sentences against stale proposals. Unlocking removes protection, not text. Accepted source edits invalidate related translation, TTS and final results.

## Continue

If timing is wrong, use Qwen3 realignment after confirming the text. It adjusts boundaries, not dialogue. Then translate, save edits, synthesize and mix normally.

Conservative automatic correction applies only a restricted set of proposals. It is experimental and not calibrated against your corpus.

Trusted timed Chinese subtitles need neither ASR review nor body translation. Use the [existing-subtitle route](SUBTITLE_WORKFLOW.md).
