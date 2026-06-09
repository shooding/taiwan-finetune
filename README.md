# Whisper Large V3 — Taiwanese ASR fine-tune (Unsloth LoRA)

Fine-tunes `openai/whisper-large-v3` for Taiwanese / zh-TW ASR using Unsloth
LoRA, on a **DGX Spark GB10** host. Mixes the open-source
`adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw` (streamed) with a local
custom recording set (餐點語料) for domain adaptation.

The training script is `whisper_taiwan_finetune.py` (ported from the original
Colab notebook — no Google Drive / Colab dependencies).

## Hardware / platform assumed

- **DGX Spark GB10** — aarch64 (ARM), GPU compute capability **sm_121**
  (Grace-Blackwell)
- Host **CUDA 13.0** / driver 580
- Python **3.12**

## Required system packages (apt — needs sudo)

These are **not** pip-installable; install them on the host first:

```bash
sudo apt-get update
sudo apt-get install -y \
    python3.12-dev \   # Python.h — triton compiles a CUDA shim at first run
    ffmpeg             # libav* shared libs (audio decode; see note below)
```

- `python3.12-dev` — without `Python.h`, triton fails to build its CUDA driver
  module on the first training step (`fatal error: Python.h: No such file`).
- `ffmpeg` — provides FFmpeg shared libraries. **Note:** this project decodes
  audio via `soundfile` (libsndfile), not torchcodec, so ffmpeg is *not*
  strictly required for the current pipeline — but it's harmless and useful to
  have. See the torch/datasets note below.

## Python environment

Create a venv and install pinned deps:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the **cu130 aarch64** torch build and includes the
PyTorch extra index, so a plain `pip install -r requirements.txt` resolves the
GPU wheels.

### Why these specific pins (the non-obvious parts)

- **torch 2.10.0+cu130** — Unsloth caps `torch<2.11`, and PyPI's 2.10 is
  CPU-only. The cu130 aarch64 wheel ships `sm_120`/`compute_120` PTX which JITs
  forward to GB10's sm_121 (a cosmetic "capability 12.1 > max 12.0" warning is
  expected; GPU ops still run).
- **triton 3.6.0** — the version torch 2.10 expects; Unsloth compiles triton
  kernels, so a mismatch (e.g. 3.7) can crash mid-training.
- **datasets 3.6.0** — 4.x decodes audio via `torchcodec`, which needs FFmpeg
  *and* a torch ≥2.11 ABI (incompatible with our pinned 2.10). 3.6.0 decodes via
  `soundfile`/`librosa`; the bundled libsndfile 1.2.2 handles mp3 + wav, so no
  torchcodec/ffmpeg dependency.

If installing from scratch instead of `requirements.txt`, see the recipe in the
header comment of `whisper_taiwan_finetune.py` (install order matters — Unsloth
will pull a CPU torch, then force-reinstall the cu130 build).

## Data layout

Working dir defaults to `~/taiwan_finetune/work` (override with
`TAIWAN_WHISPER_DIR`). Custom recordings live in `custom_data/`:

```
~/taiwan_finetune/work/
├── hf_cache/          # HuggingFace model/dataset cache
├── checkpoints/       # training checkpoints (auto-resume)
└── custom_data/       # your recordings (override with CUSTOM_DATA_DIR)
    ├── metadata.csv   # columns: file_name,transcription
    ├── rec_001.wav
    └── ...
```

To unpack a custom-data zip into place:

```bash
mkdir -p ~/taiwan_finetune/work
unzip custom_data_2026-05-29.zip -d ~/taiwan_finetune/work/
```

## Run

```bash
./venv/bin/python whisper_taiwan_finetune.py
```

Training auto-resumes from the latest checkpoint in `work/checkpoints/` if one
exists. Outputs: LoRA adapters → `work/final_model/`, merged 16-bit model →
`work/merged_model/`, CTranslate2 (faster-whisper) → `work/faster_whisper_ct2/`.

## Autoresearch loop (recommended for tuning CER)

`autoresearch.py` is a budget-bounded controller (cf.
[karpathy/autoresearch](https://github.com/karpathy/autoresearch)) for this
project's actual goal: **drive down CER on the custom restaurant-order recordings
(餐點語料), and keep doing so as you add more recordings over time** — without
hand-watching the run.

### Why it exists

The plain `whisper_taiwan_finetune.py` had two problems that made "final CER not
good enough" hard to act on:

1. Its eval CER was computed from **teacher-forced argmax on decoder logits**, not
   real generation — a meaningless number (often >200), yet `metric_for_best_model`
   used it, so checkpoint selection was effectively random w.r.t. the CER you ship.
2. The open-source anchor data is **Taiwan-Tongues** (Mandarin + Taiwanese/Hakka/
   indigenous + code-switch). Training on the non-Mandarin parts spends capacity
   away from your Mandarin target audience.

`autoresearch.py` fixes both and wraps the training in a decision loop.

### How the loop works on this project

1. **Data.** Your `custom_data/` (~11k recordings, ~880 unique order phrases) is
   the domain target; the open-source stream is only an anti-forgetting anchor,
   **filtered to predominantly-CJK (Mandarin)** transcripts (`MANDARIN_THR`,
   default 0.6 → drops indigenous/romanized/heavy-English). The two are
   interleaved (`CUSTOM_PROB`, default 0.30 — custom up-sampled well above its
   natural share).
2. **Held-out-by-sentence eval.** `N_HELDOUT_TEXTS` (25) whole order phrases are
   held out — *all* their recordings go to eval, none to train — so domain CER
   measures generalization to **unseen menu phrases**, not memorized audio.
3. **Real CER.** Eval uses `predict_with_generate=True`, so the selection metric
   `eval_cer` comes from `model.generate()` — the number you actually deploy. A
   separate, smaller open-source eval is logged as an anti-forgetting reference
   (it does **not** drive selection).
4. **Decision loop (within `TIME_BUDGET_SEC`).** It trains a *trial* and
   early-stops when real domain CER stops improving (`EARLY_PATIENCE`). If CER is
   still above `TARGET_CER` and budget remains, it starts another trial at the
   next learning rate in `LR_LADDER`, continuing from the best weights so far.
   It keeps the **global-best adapter by real CER** and stops as soon as
   `TARGET_CER` is reached. A `TimeBudgetCallback` guarantees the wall-clock cap
   and reserves time to save. Every decision is appended to
   `work/autoresearch/journal.jsonl`.
5. **Export.** On finish it writes the best LoRA adapter to `work/final_model/`,
   merges it into the base (PEFT `merge_and_unload`) → `work/merged_model/`, and
   converts to CTranslate2 → `work/faster_whisper_ct2/`.

```bash
export TIME_BUDGET_SEC=7200      # wall-clock budget (default 2h)
./venv/bin/python autoresearch.py
tail -f /tmp/…  # or watch work/autoresearch/journal.jsonl
```

### Iterating as you collect more recordings

This is the intended workflow: drop new `.wav` + rows in `custom_data/metadata.csv`,
then re-run `autoresearch.py`. More domain coverage → lower held-out CER. Tips:
- The first real CER lands at step `EVAL_STEPS` (~25–30 min at ~7 s/step on GB10),
  so a 2h budget fits roughly one learning-rate trial; raise `TIME_BUDGET_SEC`
  (e.g. `21600`) to let the full `LR_LADDER` run.
- Lower `TARGET_CER` to push harder; raise `EVAL_CAP`/`N_HELDOUT_TEXTS` for a more
  stable CER estimate as the dataset grows.
- Compare runs via the per-trial `best_cer` (and `general_cer`) in the journal.

Reference result: first 2h run reached **6.07% real domain CER** (trial 0,
lr=1e-4); the deployed CT2 model verified at **0.00% CER** on held-out clips.

Tunable via env: `TIME_BUDGET_SEC`, `TARGET_CER`, `LR_LADDER` (comma-sep),
`EVAL_STEPS`, `EVAL_CAP`, `GENERAL_N` (0 disables the anti-forgetting eval),
`N_HELDOUT_TEXTS`, `CUSTOM_PROB`, `MANDARIN_THR`, `EARLY_PATIENCE`, `MAX_STEPS_TRIAL`,
`DO_CT2`, `BASE_MODEL_ID`. The plain `whisper_taiwan_finetune.py` remains the simple
single-run path.

> **Merge note.** Both scripts merge LoRA via PEFT `merge_and_unload()`, **not**
> unsloth's `save_pretrained_merged` — the latter corrupts the Whisper merge
> (adapter is fine, but the merged/CT2 model emits garbage). The merge reloads a
> clean base (`BASE_MODEL_ID`, default `openai/whisper-large-v3`) + the saved
> adapter.

### Sanity-check the CT2 model before deploying

Always verify the converted model actually transcribes (catches merge/convert
regressions):

```python
from faster_whisper import WhisperModel
m = WhisperModel('work/faster_whisper_ct2', device='cpu', compute_type='int8')
segs, _ = m.transcribe('work/custom_data/rec_001.wav', language='zh', beam_size=5)
print(''.join(s.text for s in segs))
```

Note: the installed `ctranslate2` wheel is CPU-only on this host; GPU serving
needs a CUDA-enabled ctranslate2 build.

## Publish to Hugging Face Hub

Uploading the CT2 model is a separate step in `upload_to_hf.py` (so training
doesn't require any HF credentials). After training finishes:

```bash
export HF_TOKEN=hf_...          # write-scope token (prompts if unset)
./venv/bin/python upload_to_hf.py
```

Repo defaults to `shooding/faster-whisper-large-v3-zh-TW`; override with
`HF_REPO_ID`. The model and the open-source dataset are public, so **no token is
needed for training** — only for this upload step.
