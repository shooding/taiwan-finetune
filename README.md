# Fine-tune Whisper Large V3 for Taiwanese Mandarin ASR

Fine-tunes **Whisper Large V3** for Taiwanese Mandarin ASR using [Unsloth](https://github.com/unslothai/unsloth) LoRA on Google Colab.

## Key Design Decisions

- **`FastModel`** instead of raw `from_pretrained` — automatically patches conv1d mixed-precision, eliminating the `Input type (float) and bias type (c10::Half)` RuntimeError
- **LoRA** on `q_proj` + `v_proj` only — ~2% of parameters trained, 50%+ VRAM savings over full fine-tuning
- **`streaming=True`** — dataset streams from HuggingFace, never downloaded to Colab disk
- **`generation_config.language/task`** instead of deprecated `forced_decoder_ids`
- **Eval via logits argmax** (not `predict_with_generate`) — CER may read artificially high (e.g. 246%); this is expected. Watch validation loss instead.

## Dataset & Model

| | |
|---|---|
| Base model | [`unsloth/whisper-large-v3`](https://huggingface.co/unsloth/whisper-large-v3) |
| Dataset | [`adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw`](https://huggingface.co/datasets/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw) |
| Hardware | Colab T4 (15 GB) / A100 |

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

Custom samples are interleaved into the main dataset at `prob=0.03` — targeting ~5 repeats, which avoids overfitting for small sets (<200 samples).

## Google Drive Layout

All artifacts are persisted to Drive automatically:

```
MyDrive/taiwan-whisper/
├── hf_cache/             ← HuggingFace model & dataset cache
├── checkpoints/          ← Training checkpoints (auto-resume on reconnect)
├── final_model/          ← LoRA adapters
├── merged_model/         ← Merged fp16 model (for ct2 conversion)
├── faster_whisper_ct2/   ← CTranslate2 export (faster-whisper compatible)
└── custom_data/          ← Your recordings
```

## Recording Tool

`recording_tool.html` — a browser-based audio recorder for collecting custom training samples. Open directly in Chrome/Safari; no server required.
