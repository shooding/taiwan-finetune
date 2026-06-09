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
[karpathy/autoresearch](https://github.com/karpathy/autoresearch)) for iterating
on CER as you grow the custom dataset. Key differences from the plain script:

- **Real CER.** Uses `predict_with_generate=True`, so eval CER comes from
  `model.generate()` — the number you actually deploy. The plain script's
  argmax-on-logits CER is teacher-forced and meaningless, yet it drove model
  selection; switching to real CER is the single biggest fix for "CER not good
  enough."
- **Mandarin-only anchor.** Filters the open-source Taiwan-Tongues stream to
  predominantly-CJK transcripts (drops indigenous / romanized / heavy-English),
  so the anti-forgetting anchor doesn't pull the model off Mandarin.
- **Controller.** Trains a trial with early-stopping on real domain CER; if it
  plateaus and budget remains and CER is above target, it retries at the next
  learning rate in a small ladder, keeping the global-best adapter. Every trial
  is logged to `work/autoresearch/journal.jsonl`.

```bash
export TIME_BUDGET_SEC=7200      # wall-clock budget (default 2h)
./venv/bin/python autoresearch.py
```

Tunable via env: `TIME_BUDGET_SEC`, `TARGET_CER` (stop once real CER ≤ this),
`LR_LADDER` (comma-sep), `EVAL_STEPS`, `EVAL_CAP`, `GENERAL_N` (0 disables the
anti-forgetting eval), `MANDARIN_THR` (0–1, default 0.6), `DO_CT2`. On finish it
writes the best LoRA adapter to `work/final_model/` and, when the best model is
live, the merged + CT2 artifacts too. The plain `whisper_taiwan_finetune.py`
remains the simple single-run path.

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
