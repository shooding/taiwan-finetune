# -*- coding: utf-8 -*-
"""autoresearch.py — budget-bounded LoRA fine-tune controller for Whisper Taiwan ASR.

Concept (cf. github.com/karpathy/autoresearch): optimize the metric you actually
deploy, within a wall-clock budget, journaling every trial and deciding
continue-vs-adjust automatically.

What's different from a plain single-run fine-tune
--------------------------------------------------
1. REAL CER. Uses predict_with_generate=True, so eval CER comes from
   model.generate() — the number you actually ship. (A teacher-forced
   argmax-on-logits CER is meaningless for selection, yet a plain run drove
   model selection; that alone can explain "CER not good enough".)
2. Mandarin-only anchor. Filters the open-source Taiwan-Tongues stream to
   predominantly-CJK transcripts (drops indigenous / romanized / heavy-English),
   so the anti-forgetting anchor doesn't pull the model away from Mandarin.
3. Controller. Trains a trial with EarlyStopping on real domain CER. If the
   trial plateaus and budget remains and CER is still above target, it starts a
   new trial at the next learning rate in a small ladder, continuing from the
   best weights so far. Keeps the global-best adapter by real CER and writes a
   journal under work/autoresearch/.

Run:
    export TIME_BUDGET_SEC=7200          # 2h (default)
    ./venv/bin/python autoresearch.py

Tunable via env: TIME_BUDGET_SEC, TARGET_CER, LR_LADDER (comma-sep),
EVAL_STEPS, EVAL_CAP, GENERAL_N, MANDARIN_THR, DO_CT2.
"""

import os
import sys
import re
import json
import time
import glob
import random
import shutil
import subprocess

START = time.time()

# ── Config (env-overridable) ─────────────────────────────────────────────────
BASE_DIR        = os.environ.get('TAIWAN_WHISPER_DIR',
                                 os.path.expanduser('~/taiwan_finetune/work'))
CACHE_DIR       = f'{BASE_DIR}/hf_cache'
CHECKPOINT_DIR  = f'{BASE_DIR}/checkpoints'
CUSTOM_DATA_DIR = os.environ.get('CUSTOM_DATA_DIR', f'{BASE_DIR}/custom_data')
AR_DIR          = f'{BASE_DIR}/autoresearch'          # journal + best adapter
JOURNAL         = f'{AR_DIR}/journal.jsonl'
BEST_ADAPTER    = f'{AR_DIR}/best_adapter'
FINAL_MODEL_DIR = f'{BASE_DIR}/final_model'
MERGED_DIR      = f'{BASE_DIR}/merged_model'
CT2_OUTPUT      = f'{BASE_DIR}/faster_whisper_ct2'

TIME_BUDGET_SEC = int(float(os.environ.get('TIME_BUDGET_SEC', 2 * 3600)))
SAVE_MARGIN_SEC = int(float(os.environ.get('SAVE_MARGIN_SEC', 240)))   # reserve for final save/CT2
MIN_TRIAL_SEC   = int(float(os.environ.get('MIN_TRIAL_SEC', 600)))     # don't start a trial with less left
TARGET_CER      = float(os.environ.get('TARGET_CER', 8.0))             # stop early once real domain CER ≤ this
LR_LADDER       = [float(x) for x in os.environ.get('LR_LADDER', '1e-4,5e-5,2e-4').split(',')]
EVAL_STEPS      = int(os.environ.get('EVAL_STEPS', 250))
MAX_STEPS_TRIAL = int(os.environ.get('MAX_STEPS_TRIAL', 4000))
EVAL_CAP        = int(os.environ.get('EVAL_CAP', 150))                 # domain eval size (generation eval is slow)
GENERAL_N       = int(os.environ.get('GENERAL_N', 80))                 # 0 disables general/anti-forgetting eval
GENERAL_SKIP    = int(os.environ.get('GENERAL_SKIP', 20000))
N_HELDOUT_TEXTS = int(os.environ.get('N_HELDOUT_TEXTS', 25))
CUSTOM_PROB     = float(os.environ.get('CUSTOM_PROB', 0.30))
MANDARIN_THR    = float(os.environ.get('MANDARIN_THR', 0.60))
# Canonical base for the final PEFT merge (verified to reproduce the adapter's CER).
BASE_MODEL_ID   = os.environ.get('BASE_MODEL_ID', 'openai/whisper-large-v3')
EARLY_PATIENCE  = int(os.environ.get('EARLY_PATIENCE', 2))
GEN_MAX_LEN     = int(os.environ.get('GEN_MAX_LEN', 225))
DO_CT2          = os.environ.get('DO_CT2', '1') == '1'

for d in [BASE_DIR, CACHE_DIR, CHECKPOINT_DIR, AR_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ['HF_HOME']            = CACHE_DIR
os.environ['HF_DATASETS_CACHE']  = f'{CACHE_DIR}/datasets'
os.environ['TRANSFORMERS_CACHE'] = f'{CACHE_DIR}/transformers'
os.environ['HF_HUB_CACHE']       = f'{CACHE_DIR}/hub'


def time_left():
    return TIME_BUDGET_SEC - (time.time() - START)


def log(msg):
    print(f'[autoresearch +{int(time.time() - START):5d}s | left {int(time_left()):5d}s] {msg}',
          flush=True)


def journal(record):
    record = {'t': round(time.time() - START, 1), **record}
    with open(JOURNAL, 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ── Mandarin filter ──────────────────────────────────────────────────────────
_CJK = re.compile(r'[㐀-䶿一-鿿豈-﫿]')


def mandarin_ratio(text):
    if not text:
        return 0.0
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return 0.0
    return len(_CJK.findall(text)) / len(nonspace)


def is_mandarin(text):
    # "Keep predominantly-CJK": transcript is mostly Chinese characters.
    return mandarin_ratio(text) >= MANDARIN_THR


# ── Model ────────────────────────────────────────────────────────────────────
log('Loading model (Unsloth FastModel + LoRA)...')
from unsloth import FastModel, is_bf16_supported
from transformers import WhisperForConditionalGeneration

model, tokenizer = FastModel.from_pretrained(
    model_name='unsloth/whisper-large-v3',
    dtype=None,
    load_in_4bit=False,
    auto_model=WhisperForConditionalGeneration,
    whisper_language='chinese',
    whisper_task='transcribe',
)
model = FastModel.get_peft_model(
    model,
    r=64,
    target_modules=['q_proj', 'v_proj'],
    lora_alpha=64,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    task_type=None,
)
model.generation_config.language = '<|zh|>'
model.generation_config.task = 'transcribe'
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None
log('Model + LoRA ready.')

# ── Data ─────────────────────────────────────────────────────────────────────
from datasets import load_dataset, Audio, Dataset, interleave_datasets

AUDIO_COL = 'mp3'
TEXT_COL  = 'txt'
ALL_COLS  = ['mp3', 'txt', 'json', '__key__', '__url__']


import io
import numpy as np
import soundfile as sf
import librosa


def _load_audio_16k(audio):
    """Return (float32 array, 16000). Handles BOTH a decoded
    {'array','sampling_rate'} dict (custom data) AND a raw {'bytes','path'} dict.
    The raw case happens because datasets' streaming .filter() yields undecoded
    audio even though the feature still reports decode=True — so we decode the
    mp3 bytes ourselves (libsndfile 1.2.2 handles mp3) and resample to 16 kHz."""
    if isinstance(audio, dict) and audio.get('array') is not None:
        arr = np.asarray(audio['array'], dtype='float32')
        sr = audio['sampling_rate']
    else:
        if isinstance(audio, dict):
            if audio.get('bytes') is not None:
                src = io.BytesIO(audio['bytes'])       # raw bytes (filtered oss stream)
            elif audio.get('path'):
                src = audio['path']                     # local file (custom data)
            else:
                raise ValueError(f'Undecodable audio entry: keys={list(audio)}')
        else:
            src = io.BytesIO(audio)
        arr, sr = sf.read(src, dtype='float32', always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        sr = 16000
    return arr, sr


def prepare_dataset(batch):
    arr, sr = _load_audio_16k(batch[AUDIO_COL])
    features = tokenizer.feature_extractor(arr, sampling_rate=sr)
    tokenized = tokenizer.tokenizer(batch[TEXT_COL])
    return {'input_features': features.input_features[0], 'labels': tokenized.input_ids}


log('Loading open-source Taiwan-Tongues dataset (streaming, Mandarin-filtered)...')
raw_ds = load_dataset('adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw', 'default', streaming=True)
# Mandarin-only anchor. input_columns=[TEXT_COL] → filter reads only the text
# (no audio decode). NOTE: streaming .filter() yields raw (undecoded) mp3 bytes
# regardless of cast, so prepare_dataset decodes audio itself (see _load_audio_16k).
oss_train = raw_ds['train'].filter(is_mandarin, input_columns=[TEXT_COL])
train_ds = oss_train

domain_eval_ds = None
metadata_path = f'{CUSTOM_DATA_DIR}/metadata.csv'
USE_CUSTOM_DATA = os.path.exists(metadata_path)

if USE_CUSTOM_DATA:
    import pandas as pd
    meta = pd.read_csv(metadata_path)
    log(f'Custom data: {len(meta)} samples')
    audio_paths = [os.path.join(CUSTOM_DATA_DIR, str(fn)) for fn in meta['file_name']]
    # Audio() (sampling_rate=None) to match the open-source stream's native-SR
    # feature so interleave_datasets can align. prepare_dataset resamples to 16k.
    custom_ds = Dataset.from_dict({
        AUDIO_COL: audio_paths,
        TEXT_COL: meta['transcription'].astype(str).tolist(),
    }).cast_column(AUDIO_COL, Audio())

    # Held-out BY SENTENCE so domain CER reflects unseen menu phrases, not memorized audio.
    all_texts = sorted(set(custom_ds[TEXT_COL]))
    rng = random.Random(42)
    rng.shuffle(all_texts)
    heldout_texts = set(all_texts[:N_HELDOUT_TEXTS])
    log(f'Custom texts: {len(all_texts)} unique → hold out {len(heldout_texts)} for domain eval.')

    custom_train_raw = custom_ds.filter(lambda b: b[TEXT_COL] not in heldout_texts)
    custom_eval_raw  = custom_ds.filter(lambda b: b[TEXT_COL] in heldout_texts).shuffle(seed=42)
    if len(custom_eval_raw) > EVAL_CAP:
        custom_eval_raw = custom_eval_raw.select(range(EVAL_CAP))
    log(f'Custom split → train {len(custom_train_raw)} / domain-eval {len(custom_eval_raw)}.')

    custom_iter = custom_train_raw.to_iterable_dataset()   # already Audio(); decode in prepare_dataset
    train_ds = interleave_datasets(
        [train_ds, custom_iter],
        probabilities=[1 - CUSTOM_PROB, CUSTOM_PROB],
        stopping_strategy='all_exhausted',
        seed=42,
    )
    log(f'Interleaved custom data (prob={CUSTOM_PROB}).')
    domain_eval_ds = custom_eval_raw.map(prepare_dataset, remove_columns=[AUDIO_COL, TEXT_COL])

train_ds = train_ds.shuffle(buffer_size=1000, seed=42)
train_ds = train_ds.map(prepare_dataset, remove_columns=ALL_COLS)

# General eval = open-source held-out (anti-forgetting reference, not the selection metric).
general_eval_ds = None
if GENERAL_N > 0:
    log(f'Building general eval (Mandarin, skip {GENERAL_SKIP}, take {GENERAL_N})...')
    general_stream = (
        oss_train.skip(GENERAL_SKIP).take(GENERAL_N)
        .map(prepare_dataset, remove_columns=ALL_COLS)
    )
    general_eval_ds = Dataset.from_list(list(general_stream))

# The Trainer evaluates a SINGLE dataset (domain if we have custom data) so that
# metric_for_best_model + EarlyStopping work cleanly — with multiple eval sets the
# callback fires once per set and misses the metric on alternate calls. The
# general/anti-forgetting set is evaluated separately once per trial (reference only).
trainer_eval_ds = domain_eval_ds if domain_eval_ds is not None else general_eval_ds
reference_eval_ds = general_eval_ds if domain_eval_ds is not None else None
SELECT_METRIC = 'eval_cer'
log(f'Trainer eval = {"domain" if domain_eval_ds is not None else "general"}; '
    f'reference eval = {"general" if reference_eval_ds is not None else "none"}; '
    f'selection metric: {SELECT_METRIC}')

# ── Collator + REAL (generation) CER ─────────────────────────────────────────
import torch
import numpy as np
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': f['input_features']} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')
        label_features = [{'input_ids': f['labels']} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch['labels'] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer)
cer_metric = evaluate.load('cer')


def compute_metrics(pred):
    # predict_with_generate=True → pred.predictions are GENERATED token ids.
    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    label_ids = pred.label_ids
    label_ids = np.where(label_ids == -100, tokenizer.tokenizer.pad_token_id, label_ids)
    pred_str  = tokenizer.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {'cer': cer}


# ── Trainer plumbing ─────────────────────────────────────────────────────────
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback,
    TrainerCallback,
)


class TimeBudgetCallback(TrainerCallback):
    """Stops the current trial when the wall-clock budget (minus save margin) is spent."""

    def _check(self, control):
        if time_left() <= SAVE_MARGIN_SEC:
            control.should_training_stop = True
        return control

    def on_step_end(self, args, state, control, **kw):
        return self._check(control)

    def on_evaluate(self, args, state, control, **kw):
        return self._check(control)


def run_trial(trial_idx, lr):
    """Train `model` (carries weights across trials) at `lr`; early-stop on real domain CER.
    Returns the trial's best real CER (lower is better)."""
    out_dir = f'{CHECKPOINT_DIR}/trial{trial_idx}'
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS_TRIAL,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type='cosine',
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        eval_strategy='steps',
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=8,
        predict_with_generate=True,            # ← real CER from model.generate()
        generation_max_length=GEN_MAX_LEN,
        save_strategy='steps',
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,           # leaves `model` at this trial's best weights
        metric_for_best_model=SELECT_METRIC,
        greater_is_better=False,
        logging_steps=25,
        report_to='none',
        dataloader_num_workers=0,
        optim='adamw_8bit',
        weight_decay=0.001,
        remove_unused_columns=False,
        label_names=['labels'],
        seed=3407,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=trainer_eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer.feature_extractor,
        callbacks=[TimeBudgetCallback(), EarlyStoppingCallback(early_stopping_patience=EARLY_PATIENCE)],
    )
    log(f'── Trial {trial_idx}: lr={lr:g}, max_steps={MAX_STEPS_TRIAL}, early_patience={EARLY_PATIENCE}')
    trainer.train()
    best = trainer.state.best_metric          # best real domain CER seen this trial
    steps = trainer.state.global_step
    # Anti-forgetting reference (separate eval; does NOT affect selection).
    general_cer = None
    if reference_eval_ds is not None and time_left() > SAVE_MARGIN_SEC:
        try:
            g = trainer.evaluate(reference_eval_ds, metric_key_prefix='general')
            general_cer = g.get('general_cer')
        except Exception as e:  # noqa: BLE001
            log(f'WARN: general eval failed ({e})')
    log(f'── Trial {trial_idx} done: best domain CER={best:.3f} @ {steps} steps'
        + (f' | general CER={general_cer:.3f}' if general_cer is not None else ''))
    journal({'event': 'trial_done', 'trial': trial_idx, 'lr': lr,
             'best_cer': best, 'general_cer': general_cer, 'global_step': steps})
    return best


# ── Controller ───────────────────────────────────────────────────────────────
journal({'event': 'run_start', 'budget_sec': TIME_BUDGET_SEC, 'target_cer': TARGET_CER,
         'lr_ladder': LR_LADDER, 'mandarin_thr': MANDARIN_THR, 'select_metric': SELECT_METRIC})

global_best = float('inf')
best_trial = None
live_is_best = False                 # does `model` currently hold the global-best weights?
for trial_idx, lr in enumerate(LR_LADDER):
    if time_left() < MIN_TRIAL_SEC:
        log(f'Stopping: {int(time_left())}s left < MIN_TRIAL_SEC={MIN_TRIAL_SEC}.')
        break
    trial_best = run_trial(trial_idx, lr)
    # After load_best_model_at_end, `model` holds THIS trial's best weights.
    if trial_best is not None and trial_best < global_best:
        global_best = trial_best
        best_trial = trial_idx
        model.save_pretrained(BEST_ADAPTER)
        tokenizer.save_pretrained(BEST_ADAPTER)
        live_is_best = True
        log(f'New global-best real CER={global_best:.3f} (trial {trial_idx}) → saved {BEST_ADAPTER}')
        journal({'event': 'new_best', 'trial': trial_idx, 'cer': global_best})
    else:
        live_is_best = False         # this (worse) trial's weights are now live
    if global_best <= TARGET_CER:
        log(f'Target reached: real CER {global_best:.3f} ≤ {TARGET_CER}. Stopping search.')
        break
    log(f'Continue decision: best so far {global_best:.3f}, {int(time_left())}s left.')

log(f'Search complete. Global-best real domain CER = {global_best:.3f} (trial {best_trial}).')
journal({'event': 'run_done', 'global_best_cer': global_best, 'best_trial': best_trial})

# ── Finalize ─────────────────────────────────────────────────────────────────
# The deployable LoRA adapter is always BEST_ADAPTER (the global best). Copy it
# to FINAL_MODEL_DIR unconditionally so the artifact is correct regardless of
# which trial the live model ended on.
if best_trial is not None and os.path.isdir(BEST_ADAPTER):
    if os.path.exists(FINAL_MODEL_DIR):
        shutil.rmtree(FINAL_MODEL_DIR)
    shutil.copytree(BEST_ADAPTER, FINAL_MODEL_DIR)
    log(f'Best LoRA adapter → {FINAL_MODEL_DIR}')

    # Merge via PEFT merge_and_unload — NOT unsloth save_pretrained_merged, which
    # produced a corrupted Whisper merge (garbage transcripts despite a good
    # adapter). Reload base + the best adapter (FINAL_MODEL_DIR always holds the
    # global best) and merge cleanly.
    log('Merging best adapter into base (PEFT merge_and_unload)...')
    merged_ok = False
    try:
        from transformers import WhisperForConditionalGeneration as _WFC, WhisperProcessor as _WP
        from peft import PeftModel as _PM
        _base = _WFC.from_pretrained(BASE_MODEL_ID, dtype=torch.float16)
        _merged = _PM.from_pretrained(_base, FINAL_MODEL_DIR).merge_and_unload()
        _merged.generation_config.language = '<|zh|>'
        _merged.generation_config.task = 'transcribe'
        _merged.generation_config.forced_decoder_ids = None
        _merged.save_pretrained(MERGED_DIR, safe_serialization=True)
        _WP.from_pretrained(FINAL_MODEL_DIR).save_pretrained(MERGED_DIR)
        del _base, _merged
        log(f'Merged model → {MERGED_DIR}')
        merged_ok = True
    except Exception as e:  # noqa: BLE001
        log(f'WARN: PEFT merge failed ({e}); LoRA adapter still at {FINAL_MODEL_DIR}.')

    if DO_CT2 and merged_ok and os.path.exists(f'{MERGED_DIR}/config.json'):
        log('Converting to CTranslate2 (faster-whisper)...')
        try:
            # Use the converter from THIS venv (not on PATH when run via
            # ./venv/bin/python without activating the venv).
            ct2_bin = os.path.join(os.path.dirname(sys.executable), 'ct2-transformers-converter')
            subprocess.run([
                ct2_bin, '--model', MERGED_DIR,
                '--output_dir', CT2_OUTPUT,
                '--copy_files', 'tokenizer.json', 'preprocessor_config.json',
                '--quantization', 'float16', '--force',
            ], check=True)
            log(f'CT2 model → {CT2_OUTPUT}. Publish: ./venv/bin/python upload_to_hf.py')
        except Exception as e:  # noqa: BLE001
            log(f'WARN: CT2 conversion failed ({e}).')
else:
    log('No completed trial produced a real-CER score (budget too small?). Nothing exported.')

log(f'Journal: {JOURNAL}')
print(f'\nDONE. Best real domain CER = {global_best:.3f}  |  artifacts under {BASE_DIR}')
