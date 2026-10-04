English | [中文](../PARAMETERS.md)

[Documentation](INDEX.md) · [Configuration](CONFIGURATION.md)

# Complete parameter reference

Generated from `services/parameters.py`: **201 fields**. Schema defaults differ from saved values. Backend/language choices can set effective models, prompts, devices and voices; see the feature chapters.

Project fields appear in project panels/new defaults; global fields affect future projects/queue entries. Always means no static visibility condition; expand advanced groups. Options can depend on backend. Validation also checks cross-field dependencies, language and JSON contracts.

Paths use `<program>` instead of this checkout. Empty prompts use built-in language templates. Legacy fields are listed for completeness; do not edit migration counters or obsolete review prompts for ordinary operations.

## Separation

| Key / label | Type / scope | Default | Range / options | Display condition |
|---|---|---|---|---|
| `separation_enabled`<br>Vocal separation | boolean<br>project | `false` | — | Always |
| `separation_backend`<br>Separation backend | string<br>project | `local` | local, replicate, http | Always |
| `separation_model`<br>Models | string<br>project | `vocals_mel_band_roformer.ckpt` | — | Always |
| `separation_device`<br>Device | string<br>project | `cuda` | cuda, cpu | Always |
| `separation_chunk_seconds`<br>Outer chunk duration (s) | number<br>project | `30.0` | ≥5; ≤300 | Always |
| `separation_overlap_seconds`<br>Retention edge padding (ms) | number<br>project | `1.0` | ≥0; ≤3 | Always |
| `separation_timeout_seconds`<br>Timeout per chunk | number<br>project | `1800.0` | ≥30; ≤14400 | Always |
| `separation_use_autocast`<br>Automatic mixed precision | boolean<br>project | `true` | — | Always |
| `separation_normalization`<br>Separation normalization peak | number<br>project | `1.0` | ≤1; >0 | Always |
| `separation_vocal_stem`<br>Output stem used as vocals | string<br>project | `Vocals` | — | Always |
| `separation_common_params`<br>Common advanced parameters (JSON) | string<br>project | `{"invert_using_spec":false,"use_torch_compile":false,"use_native_fp16":false,"amplification_threshold":0.0}` | — | Always |
| `separation_mdx_params`<br>MDX parameters | string<br>project | `{"hop_length":1024,"segment_size":256,"overlap":0.25,"batch_size":1,"enable_denoise":false}` | — | Always |
| `separation_vr_params`<br>VR parameters | string<br>project | `{"batch_size":1,"window_size":512,"aggression":5,"enable_tta":false,"enable_post_process":false,"post_process_threshold":0.2,"high_end_process":false}` | — | Always |
| `separation_demucs_params`<br>Demucs parameters | string<br>project | `{"segment_size":"Default","shifts":2,"overlap":0.25,"segments_enabled":true}` | — | Always |
| `separation_mdxc_params`<br>Mel/BS-RoFormer & MDX23C parameters | string<br>project | `{"segment_size":256,"override_model_segment_size":true,"batch_size":1,"overlap":4,"pitch_shift":0}` | — | Always |
| `separation_api_url`<br>TTS API base URL | string<br>project | Empty | — | Always |
| `separation_api_model`<br>Cloud model / pinned version ID | string<br>project | Empty | — | Always |
| `separation_api_audio_field`<br>Input audio field name | string<br>project | `audio` | — | Always |
| `separation_api_output_field`<br>Vocal URL JSON field path | string<br>project | `vocals` | — | Always |
| `separation_api_params`<br>TTS API extra parameters (JSON, optional) | string<br>project | `{}` | — | Always |
| `separation_cloud_consent`<br>Consent to audio upload to the selected separation service (experimental; not recommended) | boolean<br>project | `false` | — | Always |

## Recognition and review

| Key / label | Type / scope | Default | Range / options | Display condition |
|---|---|---|---|---|
| `asr_backend`<br>ASR backend | string<br>project | `parakeet_nemo` | parakeet_nemo, kotoba_whisper, faster_whisper, generic_asr_api | Always |
| `asr_model`<br>Version | string<br>project | `grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo` | Backend-dependent | Always |
| `asr_batch_size`<br>Batch size | integer<br>project | `1` | ≥1; ≤32 | Always |
| `asr_device`<br>Device | string<br>project | `cuda` | cuda, cpu | Always |
| `asr_compute_type`<br>Compute precision | string<br>project | `float16` | — | asr_backend in ["faster_whisper","kotoba_whisper"] |
| `asr_beam_size`<br>Beam size | integer<br>project | `5` | ≥1; ≤100 | asr_backend in ["faster_whisper"] |
| `asr_vad_filter`<br>Use native VAD | boolean<br>project | `false` | — | Always |
| `asr_vad_mode`<br>Mode | string<br>project | `off` | off, backend, asmr | Always |
| `asr_vad_min_silence_ms`<br>VAD minimum silence (ms) | integer<br>project | `500` | ≥50; ≤10000 | asr_vad_mode in ["backend"] |
| `asr_asmr_vad_threshold`<br>ASMR VAD speech threshold | number<br>project | `0.5` | ≥0.05; ≤0.95 | asr_vad_mode in ["asmr"] |
| `asr_asmr_vad_min_speech_ms`<br>ASMR VAD minimum speech (ms) | integer<br>project | `250` | ≥20; ≤10000 | asr_vad_mode in ["asmr"] |
| `asr_asmr_vad_min_silence_ms`<br>ASMR VAD minimum silence (ms) | integer<br>project | `100` | ≥20; ≤10000 | asr_vad_mode in ["asmr"] |
| `asr_asmr_vad_speech_pad_ms`<br>ASMR VAD boundary padding (ms) | integer<br>project | `200` | ≥0; ≤5000 | asr_vad_mode in ["asmr"] |
| `aligner_model`<br>Alignment model | string<br>project | `Qwen/Qwen3-ForcedAligner-0.6B` | — | Always |
| `asr_forced_alignment_enabled`<br>Qwen3 alignment | boolean<br>project | `false` | — | Always |
| `asr_condition_on_previous_text`<br>Condition recognition on previous text | boolean<br>project | `true` | — | asr_backend in ["faster_whisper"] |
| `asr_initial_prompt`<br>Prompts | string<br>project | Empty | — | asr_backend in ["faster_whisper"] |
| `asr_timeout_seconds`<br>Parakeet idle timeout (s) | number<br>project | `600.0` | ≥10.0; ≤7200.0 | asr_backend in ["parakeet_nemo"] |
| `asr_api_base_url`<br>ASR API base URL | string<br>project | `http://127.0.0.1:8000/v1` | — | asr_backend in ["generic_asr_api"] |
| `asr_api_extra_body`<br>ASR API extra parameters (JSON) | string<br>project | `{}` | — | asr_backend in ["generic_asr_api"] |
| `asr_parakeet_decoder`<br>Parakeet decoder head | string<br>project | `tdt` | tdt, ctc | asr_backend in ["parakeet_nemo"] |
| `asr_chunk_seconds`<br>Kotoba-Whisper chunk duration (5–120 s) | number<br>project | `120.0` | ≥15.0; ≤600.0 | asr_backend in ["parakeet_nemo"] |
| `asr_kotoba_chunk_seconds`<br>Kotoba-Whisper chunk duration (5–120 s) | number<br>project | `30.0` | ≥5.0; ≤120.0 | asr_backend in ["kotoba_whisper"] |
| `asr_review_enabled`<br>Multi-model review | boolean<br>project | `false` | — | Always |
| `asr_review_mode`<br>Review handling | string<br>project | `suggest` | suggest, conservative | asr_review_enabled in [true] |
| `asr_review_window_seconds`<br>Common review clip target (s) | number<br>project | `30.0` | ≥10.0; ≤90.0 | asr_review_enabled in [true] |
| `asr_review_context_seconds`<br>Review context (s) | number<br>project | `0.5` | ≥0.0; ≤3.0 | asr_review_enabled in [true] |
| `asr_review_models`<br>Review models | array<br>project | `["parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo","kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.2"]` | max items 6; parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo, parakeet_nemo&#124;nvidia/parakeet-tdt_ctc-0.6b-ja, kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.2, kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.1, kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.0, faster_whisper&#124;large-v2, faster_whisper&#124;kotoba-tech/kotoba-whisper-v2.0-faster, faster_whisper&#124;distil-large-v2, faster_whisper&#124;large-v3, faster_whisper&#124;large-v3-turbo, faster_whisper&#124;medium, faster_whisper&#124;small | asr_review_enabled in [true] |
| `asr_review_text_priority_model`<br>Primary transcript source | string<br>project | `parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo` | — | asr_review_enabled in [true] |
| `asr_review_timestamp_priority_model`<br>Final timestamp source | string<br>project | `qwen_forced_aligner&#124;Qwen/Qwen3-ForcedAligner-0.6B` | — | asr_review_enabled in [true] |
| `asr_review_background`<br>Work, characters & scene context | string<br>project | Empty | — | asr_review_enabled in [true] |
| `asr_review_prompt`<br>ASR review prompt | string<br>project | Built-in template; see [PROMPTS](PROMPTS.md) | — | asr_review_enabled in [true] |
| `asr_review_max_drift_seconds`<br>Allowed timestamp drift (s) | number<br>project | `1.5` | ≥0.1; ≤10.0 | asr_review_enabled in [true] |
| `pause_split_seconds`<br>Pause ends a sentence after | number<br>project | `0.55` | ≥0.1; ≤5.0 | Always |
| `max_sentence_seconds`<br>Maximum sentence length | number<br>project | `15.0` | ≥2.0; ≤60.0 | Always |
| `default_source_language`<br>Audio language | string<br>global | `ja` | ja, en, zh | Always |

## Translation

| Key / label | Type / scope | Default | Range / options | Display condition |
|---|---|---|---|---|
| `translation_provider`<br>Translation provider | string<br>project | `deepseek` | deepseek, bailian, doubao, openai, anthropic, gemini, openai_compatible, sensenova, deepl, google_translate, microsoft_translate | Always |
| `translation_model`<br>Models | string<br>project | `deepseek-v4-flash` | Backend-dependent | Always |
| `translation_base_url`<br>TTS API base URL | string<br>project | Empty | — | Always |
| `translation_prompt`<br>Translation prompt | string<br>project | Empty | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_temperature`<br>Temperature | number<br>project | `0.1` | ≥0.0; ≤2.0 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_top_p`<br>Top P | number<br>project | `1.0` | ≤1.0; >0.0 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_max_output_tokens`<br>Maximum output tokens | integer<br>project | `16384` | ≥1024; ≤131072 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_send_context`<br>Send neighboring sentence context | boolean<br>project | `true` | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_context_sentences`<br>Context sentences | integer<br>project | `24` | ≥0; ≤200 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_memory_sentences`<br>Translation memory sentences | integer<br>project | `50` | ≥0; ≤500 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_deepl_formality`<br>DeepL formality | string<br>project | `default` | — | translation_provider in ["deepl"] |
| `translation_microsoft_region`<br>Azure Translator region | string<br>project | Empty | — | translation_provider in ["microsoft_translate"] |
| `translation_extra_body`<br>TTS API extra parameters (JSON, optional) | string<br>project | `{}` | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_prompt_ja`<br>translation_prompt_ja | string<br>global | Empty | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_prompt_en`<br>translation_prompt_en | string<br>global | Empty | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_prompt_zh`<br>translation_prompt_zh | string<br>global | Empty | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |

## Synthesis

| Key / label | Type / scope | Default | Range / options | Display condition |
|---|---|---|---|---|
| `tts_target_language`<br>Dub into | string<br>project | `zh` | zh, en | Always |
| `tts_backend`<br>TTS backend | string<br>project | `indextts2` | indextts2, indextts2_5, edge_tts, indextts2_api, gpt_sovits, cosyvoice, mimo_tts, minimax, fish_speech, generic_tts_api | Always |
| `tts_model`<br>Models | string<br>project | `IndexTTS2` | Backend-dependent | Always |
| `tts_device`<br>Device | string<br>project | `cuda` | cuda, cpu | tts_backend in ["indextts2","indextts2_5"] |
| `tts_reference_source`<br>Reference source | string<br>project | `project_sentence` | project_sentence, external | Always |
| `tts_external_reference_audio`<br>External reference audio | string<br>project | Empty | — | Always AND (tts_reference_source in ["external"] AND tts_backend in ["gpt_sovits","cosyvoice","fish_speech","mimo_tts","generic_tts_api"] OR tts_index_speaker_source in ["external"] AND tts_backend in ["indextts2","indextts2_api","indextts2_5"]) |
| `tts_external_reference_text`<br>Reference text | string<br>project | Empty | — | tts_backend in ["gpt_sovits","fish_speech"] AND tts_reference_source in ["external"] |
| `tts_external_reference_language`<br>Reference audio language | string<br>project | `auto` | auto, ja, en, zh | tts_backend in ["gpt_sovits"] AND tts_reference_source in ["external"] |
| `tts_api_base_url`<br>TTS API base URL | string<br>project | `http://127.0.0.1:9880` | — | tts_backend in ["indextts2_api","gpt_sovits","cosyvoice","mimo_tts","minimax","fish_speech"] |
| `tts_api_extra_body`<br>TTS API extra parameters (JSON, optional) | string<br>project | `{}` | — | tts_backend in ["indextts2_api","gpt_sovits","cosyvoice","mimo_tts","minimax","fish_speech"] |
| `tts_timeout_seconds`<br>Per-sentence timeout (s) | number<br>project | `600.0` | ≥10.0; ≤7200.0 | Always |
| `tts_request_concurrency`<br>External API concurrency | integer<br>project | `2` | ≥1; ≤8 | tts_backend in ["indextts2_api","gpt_sovits","cosyvoice","mimo_tts","minimax","fish_speech"] |
| `tts_model_path`<br>Model weights directory (checkpoints) | string<br>project | `<program>\.asmr-dubber\runtimes\index-tts\checkpoints` | — | tts_backend in ["indextts2"] |
| `tts_config_path`<br>Configuration file (config.yaml) | string<br>project | `<program>\.asmr-dubber\runtimes\index-tts\checkpoints\config.yaml` | — | tts_backend in ["indextts2"] |
| `tts_executable`<br>tts_executable | string<br>project | Empty | — | tts_backend in ["indextts2","indextts2_5"] |
| `tts_speed`<br>Speech rate | number<br>project | `1.0` | ≥0.25; ≤4.0 | Always |
| `tts_voice`<br>Voice ID | string<br>project | Empty | Backend-dependent | tts_backend in ["edge_tts","mimo_tts","minimax","fish_speech","generic_tts_api"] |
| `tts_volume`<br>Volume | number<br>project | `1.0` | ≥0.1; ≤10.0 | tts_backend in ["edge_tts","minimax"] |
| `tts_pitch`<br>Pitch | integer<br>project | `0` | ≥-12; ≤12 | tts_backend in ["minimax"] |
| `tts_emotion`<br>Emotion | string<br>project | `auto` | — | tts_backend in ["minimax"] |
| `tts_style_prompt`<br>MiMo delivery/style instructions (optional) | string<br>project | Empty | — | tts_backend in ["mimo_tts"] |
| `tts_temperature`<br>Temperature | number<br>project | `0.8` | ≥0.0; ≤2.0 | Always |
| `tts_top_p`<br>Top P | number<br>project | `0.9` | ≤1.0; >0.0 | Always |
| `tts_index_use_fp16`<br>Use FP16 | boolean<br>project | `true` | — | tts_backend in ["indextts2","indextts2_api","indextts2_5"] |
| `tts_index_emo_alpha`<br>Emotion weight | number<br>project | `0.5` | ≥0.0; ≤1.0 | tts_backend in ["indextts2","indextts2_api","indextts2_5"] |
| `tts_index_speaker_source`<br>Speaker reference source | string<br>project | `project_reference` | project_reference, sentence_reference, external | tts_backend in ["indextts2","indextts2_api","indextts2_5"] |
| `tts_index_emotion_source`<br>Emotion reference source | string<br>project | `sentence_reference` | sentence_reference, project_reference, speaker_reference, external, text, vector | tts_backend in ["indextts2","indextts2_api","indextts2_5"] |
| `tts_index_external_emotion_audio`<br>Emotion reference audio | string<br>project | Empty | — | tts_backend in ["indextts2","indextts2_api","indextts2_5"] AND tts_index_emotion_source in ["external"] |
| `tts_index_emo_text`<br>Emotion description | string<br>project | Empty | — | tts_backend in ["indextts2","indextts2_api","indextts2_5"] AND tts_index_emotion_source in ["text"] |
| `tts_index25_model_path`<br>Model weights directory (checkpoints) | string<br>project | `<program>\.asmr-dubber\runtimes\index-tts-2.5\checkpoints` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_config_path`<br>Configuration file (config.yaml) | string<br>project | `<program>\.asmr-dubber\runtimes\index-tts-2.5\checkpoints\config.yaml` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_language`<br>Synthesis language | string<br>project | `zh` | zh, en, ja, es, ar | tts_backend in ["indextts2_5"] |
| `tts_index25_use_bf16`<br>Use BF16 | boolean<br>project | `true` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_use_cuda_kernel`<br>BigVGAN CUDA kernel | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_use_deepspeed`<br>DeepSpeed | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_use_accel`<br>GPT acceleration engine | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_use_torch_compile`<br>torch.compile | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_emotion_vector`<br>Emotion vector | array<br>project | `[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]` | min items 8; max items 8 | tts_backend in ["indextts2_5"] |
| `tts_index25_use_random`<br>Randomize emotion / speaker conditions | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_interval_silence_ms`<br>Internal segment gap (ms) | integer<br>project | `200` | ≥0; ≤2000 | tts_backend in ["indextts2_5"] |
| `tts_index25_max_text_tokens`<br>Maximum text tokens per segment | integer<br>project | `120` | ≥20; ≤600 | tts_backend in ["indextts2_5"] |
| `tts_index25_duration_factor`<br>Duration factor | number<br>project | `1.0` | ≥0.5; ≤2.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_text_normalization`<br>Text normalization | boolean<br>project | `true` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_do_sample`<br>Enable sampling | boolean<br>project | `true` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_temperature`<br>Temperature | number<br>project | `0.8` | ≤2.0; >0.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_top_p`<br>Top P | number<br>project | `0.8` | ≤1.0; >0.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_top_k`<br>Top K | integer<br>project | `30` | ≥0; ≤100 | tts_backend in ["indextts2_5"] |
| `tts_index25_num_beams`<br>Beams | integer<br>project | `3` | ≥1; ≤10 | tts_backend in ["indextts2_5"] |
| `tts_index25_repetition_penalty`<br>Repetition penalty | number<br>project | `10.0` | ≥0.1; ≤20.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_length_penalty`<br>Length penalty | number<br>project | `0.0` | ≥-2.0; ≤2.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_max_mel_tokens`<br>Maximum acoustic tokens | integer<br>project | `1500` | ≥100; ≤1815 | tts_backend in ["indextts2_5"] |
| `tts_gpt_top_k`<br>Top K | integer<br>project | `15` | ≥1; ≤100 | tts_backend in ["gpt_sovits"] |
| `tts_gpt_text_split_method`<br>Text splitting method | string<br>project | `cut5` | — | tts_backend in ["gpt_sovits"] |
| `tts_gpt_sample_steps`<br>Sampling steps | integer<br>project | `32` | ≥1; ≤64 | tts_backend in ["gpt_sovits"] |
| `tts_cosyvoice_mode`<br>CosyVoice mode | string<br>project | `zero_shot` | zero_shot, cross_lingual | tts_backend in ["cosyvoice"] |
| `tts_clone_mode`<br>Reference strategy | string<br>project | `stable_reference` | stable_reference, reference_only | Always |
| `tts_reference_sentence_id`<br>Shared reference sentence ID | string / nullable<br>project | `null` | — | Always |
| `skip_japanese_fillers`<br>Skip nonsemantic Japanese fillers | boolean<br>project | `true` | — | Always |
| `reference_padding_seconds`<br>Reference edge padding (s) | number<br>project | `0.0` | ≥0.0; ≤2.0 | Always |
| `random_seed`<br>Random seed | integer<br>project | `20260722` | ≥0 | Always |
| `tts_device_selection_version`<br>tts_device_selection_version | integer<br>global | `0` | ≥0 | Always |

## Mixing and spatial follow

| Key / label | Type / scope | Default | Range / options | Display condition |
|---|---|---|---|---|
| `separation_mix_mode`<br>Mix mode | string<br>project | `bilingual` | bilingual, replace | Always |
| `separation_keep_original`<br>Original-vocal retention in replacement mode | string<br>project | `none` | none, unvoiced, manual | separation_mix_mode in ["replace"] |
| `separation_keep_ids`<br>Sentence IDs for original-vocal retention | string<br>project | Empty | — | separation_mix_mode in ["replace"] |
| `separation_keep_padding_ms`<br>Retention edge padding (ms) | number<br>project | `40.0` | ≥0; ≤500 | separation_mix_mode in ["replace"] |
| `spatial_rtf_enabled`<br>Spatial following | boolean<br>project | `false` | — | Always |
| `spatial_rtf_strength`<br>Strength | number<br>project | `1.0` | ≥0; ≤1 | spatial_rtf_enabled in [true] |
| `spatial_rtf_level_strength`<br>Distance strength | number<br>project | `0.7` | ≥0; ≤1 | spatial_rtf_enabled in [true] |
| `spatial_rtf_color_strength`<br>Tone coloration strength | number<br>project | `0.35` | ≥0; ≤1 | spatial_rtf_enabled in [true] |
| `spatial_rtf_fft_size`<br>FFT window | integer<br>project | `0` | 0, 1024, 2048, 4096, 8192 | spatial_rtf_enabled in [true] |
| `spatial_rtf_hop_divisor`<br>FFT hop divisor | integer<br>project | `8` | 4, 8, 16 | spatial_rtf_enabled in [true] |
| `spatial_rtf_block_seconds`<br>Processing block (s) | number<br>project | `10.0` | ≥1; ≤30 | spatial_rtf_enabled in [true] |
| `chinese_dubbing_offset_ms`<br>Dub offset (ms) | integer<br>project | `500` | ≥-30000; ≤30000 | Always |
| `chinese_max_auto_speed`<br>Maximum speed-up on overlap | number<br>project | `1.8` | ≥1.0; ≤4.0 | Always |
| `chinese_dubbing_timing_mode`<br>When the dub is longer than the line | string<br>project | `fit_window` | fit_window, sequential | Always |
| `chinese_gain_db`<br>Final gain adjustment | number<br>project | `0.0` | ≥-40.0; ≤20.0 | Always |
| `normalize_chinese_loudness`<br>Volume processing | boolean<br>project | `true` | — | Always |
| `match_source_loudness`<br>Match corresponding original | boolean<br>project | `true` | — | Always |
| `chinese_relative_loudness_db`<br>Relative to original | number<br>project | `-8.0` | ≥-24.0; ≤24.0 | Always |
| `chinese_min_active_rms_dbfs`<br>Minimum loudness | number<br>project | `-42.0` | ≥-60.0; ≤-20.0 | Always |
| `chinese_target_active_rms_dbfs`<br>Maximum loudness / uniform target | number<br>project | `-30.0` | ≥-50.0; ≤-16.0 | Always |
| `chinese_max_loudness_boost_db`<br>Maximum boost per sentence | number<br>project | `12.0` | ≥0.0; ≤30.0 | Always |
| `chinese_line_peak_dbfs`<br>Sentence peak ceiling | number<br>project | `-9.0` | ≥-20.0; ≤-1.0 | Always |
| `chinese_stem_peak_dbfs`<br>Dubbing track peak ceiling | number<br>project | `-3.0` | ≥-12.0; ≤-0.1 | Always |
| `chinese_fade_ms`<br>Sentence fade in and out | number<br>project | `8.0` | ≥0.0; ≤100.0 | Always |
| `replacement_chinese_gain_db`<br>Final gain adjustment | number<br>project | `0.0` | ≥-40.0; ≤20.0 | separation_mix_mode in ["replace"] |
| `replacement_normalize_chinese_loudness`<br>Volume processing | boolean<br>project | `true` | — | separation_mix_mode in ["replace"] |
| `replacement_match_source_loudness`<br>Match corresponding original | boolean<br>project | `true` | — | separation_mix_mode in ["replace"] |
| `replacement_chinese_relative_loudness_db`<br>Relative to original | number<br>project | `0.0` | ≥-24.0; ≤24.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_min_active_rms_dbfs`<br>Minimum loudness | number<br>project | `-42.0` | ≥-60.0; ≤-20.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_target_active_rms_dbfs`<br>Maximum loudness / uniform target | number<br>project | `-20.0` | ≥-50.0; ≤-16.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_max_loudness_boost_db`<br>Maximum boost per sentence | number<br>project | `12.0` | ≥0.0; ≤30.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_line_peak_dbfs`<br>Sentence peak ceiling | number<br>project | `-6.0` | ≥-20.0; ≤-1.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_stem_peak_dbfs`<br>Dubbing track peak ceiling | number<br>project | `-3.0` | ≥-12.0; ≤-0.1 | separation_mix_mode in ["replace"] |
| `replacement_chinese_fade_ms`<br>Sentence fade in and out | number<br>project | `8.0` | ≥0.0; ≤100.0 | separation_mix_mode in ["replace"] |
| `chinese_channel_routing`<br>Multichannel routing | string<br>project | `auto` | auto, all | Always |
| `mix_peak_protection`<br>Final mix peak protection | boolean<br>project | `true` | — | Always |
| `mix_peak_limit_dbfs`<br>Final peak ceiling (dBFS) | number<br>project | `-1.0` | ≥-6.0; ≤-0.1 | Always |
| `mix_output_mode`<br>Audio output layout | string<br>project | `both` | both, mixed, stem | Always |

## Subtitles

| Key / label | Type / scope | Default | Range / options | Display condition |
|---|---|---|---|---|
| `subtitle_timeline`<br>Time | string<br>project | `source` | source, dubbing | Always |
| `subtitle_max_chars_per_line`<br>Maximum characters per line (8–500) | integer<br>project | `22` | ≥8; ≤500 | Always |
| `subtitle_min_duration_seconds`<br>Minimum display duration (s) | number<br>project | `1.0` | ≥0.2; ≤10.0 | Always |
| `subtitle_max_cps`<br>Maximum characters per second | number<br>project | `18.0` | ≥5.0; ≤40.0 | Always |

## General

| Key / label | Type / scope | Default | Range / options | Display condition |
|---|---|---|---|---|
| `projects_root`<br>Project directory | string<br>global | Empty | — | Always |
| `huggingface_endpoint`<br>Hugging Face endpoint | string<br>global | Empty | — | Always |
| `pypi_index_url`<br>Python package index | string<br>global | Empty | — | Always |
| `ui_language`<br>Interface language | string<br>global | `zh` | ^(zh&#124;en)$; zh, en | Always |
| `open_browser`<br>Open browser on startup | boolean<br>global | `true` | — | Always |
| `download_source`<br>Download source | string<br>global | `modelscope` | ^(modelscope&#124;original)$; modelscope, original | Always |

## Batch

| Key / label | Type / scope | Default | Range / options | Display condition |
|---|---|---|---|---|
| `autoflow_output_folder_name`<br>Output folder name | string<br>global | `AutoFlow输出` | — | Always |
| `autoflow_default_mode`<br>Default output type | string<br>global | `video_normal` | audio, video_normal, video_harmonized | Always |
| `autoflow_default_layout`<br>Default output layout | string<br>global | `merged` | merged, separate, both | Always |
| `autoflow_include_bonus`<br>Include extra tracks by default | boolean<br>global | `false` | — | Always |
| `autoflow_background_policy`<br>Default video background | string<br>global | `auto` | auto, black | Always |
| `autoflow_embed_subtitles`<br>Embed video subtitles by default | boolean<br>global | `true` | — | Always |
| `autoflow_subtitle_language`<br>Default subtitle-only content | string<br>global | `bilingual` | bilingual, source, zh | Always |
| `autoflow_subtitle_naming`<br>Subtitle naming (subtitle-only mode) | string<br>global | `original` | original, standard, custom | Always |
| `autoflow_subtitle_custom_name`<br>Custom subtitle name (without extension) | string<br>global | `字幕` | — | Always |
| `autoflow_original_hard_subtitles`<br>Burn subtitles into the original-audio video too | boolean<br>global | `false` | — | Always |
| `autoflow_timestamp_footer_position`<br>Extra text position in timestamp document | string<br>global | `after` | before, after | Always |
| `autoflow_translate_work_title`<br>Translate work folder names | boolean<br>global | `true` | — | Always |
| `autoflow_translate_track_titles`<br>Translate track titles | boolean<br>global | `true` | — | Always |
| `autoflow_reference_wait_enabled`<br>Wait for manual reference selection | boolean<br>global | `true` | — | Always |
| `autoflow_reference_wait_seconds`<br>Maximum wait per work (s) | integer<br>global | `60` | ≥1; ≤3600 | Always |
| `autoflow_preferred_audio_formats`<br>Audio format preference order | string<br>global | `wav,flac,ape,m4a,mp3` | — | Always |
| `autoflow_harmonized_volume_reduction_db`<br>Original attenuation (dB) | number<br>global | `10.0` | ≥0.0; ≤60.0 | Always |
| `autoflow_harmonized_delay_minutes`<br>Delay original, dubbing & subtitles (min) | number<br>global | `20.0` | ≥0.0; ≤1440.0 | Always |
| `autoflow_timestamp_footer`<br>Timestamp document footer | string<br>global | `双语音声制作器：BV1f43G6YEov<br>内嵌字幕和配音为本地 AI 生成，内容仅供参考。<br>仅供日语学习，有能力请购买正版支持。` | — | Always |

