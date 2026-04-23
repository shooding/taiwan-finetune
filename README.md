# Fine-tune Whisper / Breeze-ASR-26 for Taiwanese ASR

Two Unsloth LoRA notebooks for Taiwanese speech recognition on Google Colab.

| Notebook | Base Model | Target Language | Output |
|---|---|---|---|
| `whisper_taiwan_finetune.ipynb` | [`unsloth/whisper-large-v3`](https://huggingface.co/unsloth/whisper-large-v3) | Taiwanese Mandarin (國語) | Traditional Chinese |
| `breezeasr26_taiwan_finetune.ipynb` | [`MediaTek-Research/Breeze-ASR-26`](https://huggingface.co/MediaTek-Research/Breeze-ASR-26) | Taiwanese Hokkien (台語) | Traditional Chinese (漢字) |

> Breeze-ASR-26 is MediaTek's fine-tune of **Whisper Large V2** (not V3) for Taigi. It outputs Mandarin Chinese characters, not Tâi-lô romanization.

## Key Design Decisions

- **`FastModel`** instead of raw `from_pretrained` — automatically patches conv1d mixed-precision, eliminating the `Input type (float) and bias type (c10::Half)` RuntimeError
- **LoRA** on `q_proj` + `v_proj` only — ~2% of parameters trained, 50%+ VRAM savings over full fine-tuning
- **`streaming=True`** — dataset streams from HuggingFace, never downloaded to Colab disk
- **`generation_config.language/task`** instead of deprecated `forced_decoder_ids`
- **Eval via logits argmax** (not `predict_with_generate`) — CER may read artificially high (e.g. 246%); this is expected. Watch validation loss instead.

## Datasets & Published Models

### Whisper Large V3 — Taiwanese Mandarin

| | |
|---|---|
| Dataset | [`adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw`](https://huggingface.co/datasets/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw) |
| Published model | [`shooding/faster-whisper-large-v3-zh-TW`](https://huggingface.co/shooding/faster-whisper-large-v3-zh-TW) |

### Breeze-ASR-26 — Taiwanese Hokkien

| | |
|---|---|
| Dataset | [`adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw`](https://huggingface.co/datasets/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw) (Mandarin regularizer) + custom Taigi recordings |
| Published model | [`shooding/taiwan-breeze-asr-26`](https://huggingface.co/shooding/taiwan-breeze-asr-26) |

## Custom Recordings (Optional)

Place WAV files and a `metadata.csv` in `custom_data/`:

```
custom_data/
├── metadata.csv
├── rec_001.wav
└── rec_002.wav
```

`metadata.csv` schema:
```csv
file_name,transcription
rec_001.wav,your transcription here
rec_002.wav,your transcription here
```

For Whisper V3, custom samples are interleaved at `prob=0.03` (~5 repeats for <200 samples).
For Breeze-ASR-26, set `USE_MAIN_DATASET=True` to keep a Mandarin regularizer; `CUSTOM_PROB=0.0625` targets ~10 repeats over 2000 steps.

## Google Drive Layout

```
MyDrive/taiwan-whisper/          ← Whisper Large V3
MyDrive/taiwan-breeze-asr-26/    ← Breeze-ASR-26
├── hf_cache/             ← HuggingFace model & dataset cache
├── checkpoints/          ← Training checkpoints (auto-resume on reconnect)
├── final_model/          ← LoRA adapters
├── merged_model/         ← Merged fp16 model (for ct2 conversion)
├── faster_whisper_ct2/   ← CTranslate2 export (faster-whisper compatible)
└── custom_data/          ← Your recordings
```

## Recording Tool

`recording_tool.html` — a browser-based audio recorder for collecting custom training samples. Open directly in Chrome/Safari; no server required.
