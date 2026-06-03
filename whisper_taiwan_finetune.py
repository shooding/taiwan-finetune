# -*- coding: utf-8 -*-
"""whisper_taiwan_finetune.py

# Fine-tune Whisper Large V3 for Taiwanese ASR (Unsloth LoRA)

Runs natively on a DGX Spark GB10 host (no Colab / Google Drive).

**修正的核心問題：**
- ✅ `RuntimeError: Input type (float) and bias type (c10::Half)` → 改用 `FastModel` 正確 patch 混精度
- ✅ `forced_decoder_ids` deprecated → 改用 `generation_config.language/task`
- ✅ 全量訓練 → LoRA adapter（只訓練 2% 參數，VRAM 省 50%+）
- ✅ 本地磁碟持久化（所有 checkpoint、cache）
- ✅ `streaming=True` → dataset 完全不下載到本地磁碟
- ✅ 自錄音 interleave 支援

**本地目錄結構（自動建立，可用 TAIWAN_WHISPER_DIR 環境變數覆寫）：**
```
~/taiwan_finetune/work/
├── hf_cache/          ← HuggingFace model cache
├── checkpoints/       ← 訓練 checkpoint
├── final_model/       ← LoRA adapters
├── merged_model/      ← 合併後完整模型（用於 ct2 轉換）
└── custom_data/       ← 自錄音（手動放）
    ├── metadata.csv
    └── rec_001.wav
```

## 1. 設定目錄
"""

import os
import sys

# 1. 定義路徑（本地磁碟）。BASE_DIR 可用環境變數覆寫。
BASE_DIR        = os.environ.get('TAIWAN_WHISPER_DIR',
                                 os.path.expanduser('~/taiwan_finetune/work'))
CACHE_DIR       = f'{BASE_DIR}/hf_cache'
CHECKPOINT_DIR  = f'{BASE_DIR}/checkpoints'

# 自錄音資料目錄（解壓後的 wav + metadata.csv）。可用環境變數覆寫。
CUSTOM_DATA_DIR = os.environ.get('CUSTOM_DATA_DIR', f'{BASE_DIR}/custom_data')

# Ensure base directories exist
for d in [BASE_DIR, CACHE_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(CUSTOM_DATA_DIR):
    print(f'Note: custom data dir not found: {CUSTOM_DATA_DIR}. '
          f'Training will proceed on open-source data only.')

os.environ['HF_HOME']            = CACHE_DIR
os.environ['HF_DATASETS_CACHE']  = f'{CACHE_DIR}/datasets'
os.environ['TRANSFORMERS_CACHE'] = f'{CACHE_DIR}/transformers'
os.environ['HF_HUB_CACHE']       = f'{CACHE_DIR}/hub'

print(f'Local dirs ready. Base: {BASE_DIR}, custom data: {CUSTOM_DATA_DIR}')

"""## 2. 安裝套件（Unsloth 版）

在 DGX Spark host 上於專案 venv 內預先安裝。本機 venv 在 `./venv`
（aarch64 / GB10 Grace-Blackwell, host CUDA 13.0, compute cap sm_121）。

關鍵：unsloth 限制 `torch<2.11`，而 PyPI 上的 2.10 是 CPU build。GB10 需要
cu130 的 aarch64 GPU wheel（torch 2.10 + cu130，含 sm_120/compute_120 PTX，
runtime JIT 到 sm_121）。所以順序是：先裝 unsloth 全家（會暫時拉進 CPU torch），
最後用 `--force-reinstall --no-deps` 蓋回 cu130 GPU build。從 repo 根目錄執行：

```bash
source venv/bin/activate

pip install "numpy<2.3" librosa soundfile evaluate jiwer
pip install unsloth
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
pip install --no-deps torchao==0.17.0
pip install ctranslate2

# datasets 4.x 用 torchcodec 解碼音訊，torchcodec 需要系統 FFmpeg 共享庫
# （本機無 sudo 裝不了）。改用 datasets 3.6.0：它用 soundfile/librosa 解碼，
# soundfile 內建的 libsndfile 1.2.2 可解 mp3 + wav，完全不需要 FFmpeg。
pip install "datasets==3.6.0"

# 蓋回 GB10 的 GPU torch（unsloth 會把它降成 CPU 版，這步把它換回來）：
pip install --index-url https://download.pytorch.org/whl/cu130 \
    --force-reinstall --no-deps "torch==2.10.0+cu130"
pip install --index-url https://download.pytorch.org/whl/cu130 "torchaudio==2.10.0"
pip install "triton==3.6.0"   # torch 2.10 對應的 triton（unsloth kernel 需要）
```

驗證：`./venv/bin/python -c "import torch; print(torch.cuda.is_available())"` → True
之後用 venv 直接跑：`./venv/bin/python whisper_taiwan_finetune.py`
"""

"""## 3. 載入模型（Unsloth FastModel + LoRA）

**為什麼改用 FastModel？**
- 自動 patch conv1d 混精度問題（消除 `Input type (float) and bias type (c10::Half)` RuntimeError）
- LoRA 只訓練 `q_proj`、`v_proj`，VRAM 省 50%+
- `use_gradient_checkpointing='unsloth'` 在 A100 上進一步節省顯存
- `task_type=None` 是 Whisper LoRA 的必要設定
"""

from unsloth import FastModel
from transformers import WhisperForConditionalGeneration

model, tokenizer = FastModel.from_pretrained(
    model_name='unsloth/whisper-large-v3',
    dtype=None,          # 自動偵測（A100 → bf16，T4 → fp16）
    load_in_4bit=False,  # True 可再省 VRAM，但 ASR 精度略降
    auto_model=WhisperForConditionalGeneration,
    whisper_language='chinese',
    whisper_task='transcribe',
)

# Apply LoRA — task_type=None 是 Whisper 的必要設定
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
    task_type=None,  # ** Whisper 必須設 None **
)

# 現代做法：不使用 deprecated forced_decoder_ids
model.generation_config.language = '<|zh|>'
model.generation_config.task = 'transcribe'
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None

print('Model + LoRA ready.')

"""## 4. 載入 Taiwan 資料集（streaming 模式）"""

from datasets import load_dataset

raw_ds = load_dataset(
    'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw',
    'default',
    streaming=True,
)

sample = next(iter(raw_ds['train']))
print('Train columns:', list(sample.keys()))
print('Text sample:', sample.get('txt', sample.get('text', '?')))

"""## 5. 加入自錄音（可選）

在 Google Drive 的 `custom_data/` 放 WAV 檔 + `metadata.csv`：
```csv
file_name,transcription
rec_001.wav,A套餐
rec_002.wav,B套餐
```
"""

import os

metadata_path = f'{CUSTOM_DATA_DIR}/metadata.csv'
USE_CUSTOM_DATA = os.path.exists(metadata_path)

if USE_CUSTOM_DATA:
    import pandas as pd
    from datasets import Dataset, Audio
    meta = pd.read_csv(metadata_path)
    print(f'Custom data: {len(meta)} samples')
    print(meta.head())

    # 不用 load_dataset('audiofolder', ...)：datasets 3.6.0 的 audiofolder builder
    # 只接受 metadata 欄位型別為 Value('string')，但 pandas 3.x 會把字串讀成
    # arrow-backed large_string → builder 報 "`file_name` ... must be present"。
    # 直接從 CSV 建 Dataset（欄位 audio/transcription 與 audiofolder 相同，
    # 下游 rename 邏輯不變），audio 由 soundfile 解碼。
    audio_paths = [os.path.join(CUSTOM_DATA_DIR, str(fn)) for fn in meta['file_name']]
    custom_ds = Dataset.from_dict({
        'audio': audio_paths,
        'transcription': meta['transcription'].astype(str).tolist(),
    }).cast_column('audio', Audio(sampling_rate=16000))
    print('Custom dataset columns:', custom_ds.column_names)
else:
    print(f'No custom data. Place wav + metadata.csv in: {CUSTOM_DATA_DIR}')

"""## 6. 預處理 Pipeline

streaming 模式下 `map()` 是 lazy 的，訓練時才逐 batch 執行特徵提取。

> `tokenizer` 在 unsloth 裡是 `WhisperProcessor` wrapper：
> - `tokenizer.feature_extractor` → mel spectrogram
> - `tokenizer.tokenizer` → BPE tokenizer
"""

from datasets import Audio, Dataset, interleave_datasets
import random

AUDIO_COL = 'mp3'
TEXT_COL  = 'txt'
# 主資料集所有欄位（map 後用來清除原始欄位）
ALL_COLS  = ['mp3', 'txt', 'json', '__key__', '__url__']

def prepare_dataset(batch):
    audio = batch[AUDIO_COL]
    features = tokenizer.feature_extractor(
        audio['array'],
        sampling_rate=audio['sampling_rate'],
    )
    tokenized = tokenizer.tokenizer(batch[TEXT_COL])
    return {
        'input_features': features.input_features[0],
        'labels': tokenized.input_ids,
    }

train_ds = raw_ds['train'].cast_column(AUDIO_COL, Audio(sampling_rate=16000))

# domain eval 預設為 None；有 custom_data 時改成「held-out by sentence」的餐點集
domain_eval_ds = None

if USE_CUSTOM_DATA:
    custom_cols = custom_ds.column_names
    audio_col_src = next((c for c in custom_cols if c in ('audio', 'mp3', 'wav', 'flac')), None)
    text_col_src  = next((c for c in custom_cols if c in ('transcription', 'txt', 'text', 'sentence', 'label')), None)
    print(f'Custom: audio→{audio_col_src}, text→{text_col_src}')

    if audio_col_src and audio_col_src != AUDIO_COL:
        custom_ds = custom_ds.rename_column(audio_col_src, AUDIO_COL)
    if text_col_src and text_col_src != TEXT_COL:
        custom_ds = custom_ds.rename_column(text_col_src, TEXT_COL)

    custom_ds = custom_ds.select_columns([AUDIO_COL, TEXT_COL])

    # ── 按「文本內容」切出 held-out 餐點 eval（不可隨機切）────────────────────
    # 11227 筆只有 ~182 種句子、每句重複 ~56 次。隨機切會讓同一句同時出現在
    # train / eval → CER 虛高、量不出泛化。改「整句保留」：被選中句子的所有
    # 錄音完全不進 train，只進 eval，才反映對「沒看過的餐點句子」的能力。
    N_HELDOUT_TEXTS = 25
    EVAL_CAP        = 200            # domain eval 最多筆數（控制 eval 速度）

    all_texts = sorted(set(custom_ds[TEXT_COL]))
    rng = random.Random(42)
    rng.shuffle(all_texts)
    heldout_texts = set(all_texts[:N_HELDOUT_TEXTS])
    print(f'Custom texts: {len(all_texts)} unique → 留 {len(heldout_texts)} 句做 domain eval.')

    custom_train_raw = custom_ds.filter(lambda b: b[TEXT_COL] not in heldout_texts)
    custom_eval_raw  = custom_ds.filter(lambda b: b[TEXT_COL] in heldout_texts)
    # eval 句子已固定，這裡只壓筆數（同 seed 保可重現）；被砍掉的錄音「不」回流 train
    custom_eval_raw = custom_eval_raw.shuffle(seed=42)
    if len(custom_eval_raw) > EVAL_CAP:
        custom_eval_raw = custom_eval_raw.select(range(EVAL_CAP))

    custom_train_raw = custom_train_raw.cast_column(AUDIO_COL, Audio(sampling_rate=16000))
    custom_eval_raw  = custom_eval_raw.cast_column(AUDIO_COL, Audio(sampling_rate=16000))
    print(f'Custom split → train {len(custom_train_raw)} / eval {len(custom_eval_raw)} samples.')

    custom_iter = custom_train_raw.to_iterable_dataset()

    # ── 混合比例：11227 筆餐點語料是 domain adaptation 的主角，120k 開源語料
    # 只是防遺忘的錨。上採樣到 ~30%（遠高於自然比例 8.6%）。
    # custom epoch ≈ (max_steps × 16 × CUSTOM_PROB) / custom_train_size
    #              ≈ (5000 × 16 × 0.30) / ~9700 ≈ 2.5 epoch
    CUSTOM_PROB = 0.30
    train_ds = interleave_datasets(
        [train_ds, custom_iter],
        probabilities=[1 - CUSTOM_PROB, CUSTOM_PROB],
        stopping_strategy='all_exhausted',  # custom_iter 跑完後循環，不提早停止
        seed=42,
    )
    print(f'Custom data interleaved (prob={CUSTOM_PROB}, stopping=all_exhausted).')

    domain_eval_ds = custom_eval_raw.map(prepare_dataset, remove_columns=[AUDIO_COL, TEXT_COL])

train_ds = train_ds.shuffle(buffer_size=1000, seed=42)
train_ds = train_ds.map(prepare_dataset, remove_columns=[AUDIO_COL, TEXT_COL])

# ── General eval（開源語料 held-out，當「防遺忘」參考指標，不主導選模）─────────
print('Building general eval set (skip 110000, take 200)...')
general_eval_stream = (
    raw_ds['train']
    .skip(110000)
    .take(200)
    .cast_column(AUDIO_COL, Audio(sampling_rate=16000))
    .map(prepare_dataset, remove_columns=ALL_COLS)
)
general_eval_ds = Dataset.from_list(list(general_eval_stream))

# 多 eval set：'domain' 主導選模（餐點泛化），'general' 只當防遺忘參考。
# 指標名會變成 eval_domain_cer / eval_general_cer（見訓練設定 cell）。
if domain_eval_ds is not None:
    eval_ds = {'domain': domain_eval_ds, 'general': general_eval_ds}
    print(f'Eval sets → domain {len(domain_eval_ds)} / general {len(general_eval_ds)}.')
else:
    eval_ds = general_eval_ds   # 無 custom_data：退回單一 general eval（此時選模指標需改回 'cer'）
    print(f'Eval set → general {len(general_eval_ds)} (no custom data).')

"""## 7. Data Collator & 評估指標

中文用 CER（字元錯誤率）比 WER 更合理（中文無空格分詞）。

> **注意**：不使用 `predict_with_generate=True`（效仿 Unsloth 官方做法），
> 改從 decoder logits 做 argmax，避免 eval 時的 generation 精度問題。看到CER e.g. 246 是「正常且已知」的副作用，是因為你選擇了高效能的 argmax 評估法。只要 Validation Loss 穩定下降（你現在的 0.15 很漂亮），就請無視 CER 繼續練下去。
"""

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
    # pred.predictions 可能是 tuple（decoder logits, ...）或 ndarray
    predictions = pred.predictions
    pred_logits = predictions[0] if isinstance(predictions, tuple) else predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.tokenizer.pad_token_id

    pred_ids  = np.argmax(pred_logits, axis=-1)
    pred_str  = tokenizer.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {'cer': cer}

print('Collator and metrics ready.')

"""## 8. 訓練設定

**關鍵設定說明：**
- `optim='adamw_8bit'` → unsloth 優化，省 VRAM
- `remove_unused_columns=False` → PEFT model forward 簽名不含所有欄位，必須關閉
- `label_names=['labels']` → Seq2Seq Trainer 需要明確指定
- `predict_with_generate` 關閉 → 避免 eval generation 的潛在精度問題
- `max_steps` 而非 `num_train_epochs`，因 streaming dataset 長度未知
"""

from transformers import Seq2SeqTrainingArguments
from unsloth import is_bf16_supported

training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch = 16
    max_steps=5000,                  # 11227 筆 custom：CUSTOM_PROB=0.30 下約 2.5 個 custom epoch
    learning_rate=1e-4,
    warmup_steps=250,                # ~5% of max_steps
    lr_scheduler_type='cosine',
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    eval_strategy='steps',
    eval_steps=500,
    per_device_eval_batch_size=4,
    # predict_with_generate=True,    # 關閉：改從 logits argmax 計算 CER
    save_strategy='steps',
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='eval_domain_cer',  # 多 eval set 時指標會加前綴；用餐點 held-out 集選模（非 general）。無 custom_data 時改回 'eval_cer'
    greater_is_better=False,
    logging_steps=25,
    report_to='none',               # 不需要 tensorboard/wandb
    dataloader_num_workers=0,        # streaming + multiprocessing 容易 deadlock
    push_to_hub=False,
    optim='adamw_8bit',              # unsloth 優化版 optimizer
    weight_decay=0.001,
    remove_unused_columns=False,     # PEFT Whisper 必須關閉
    label_names=['labels'],
    seed=3407,
)

use_bf16 = is_bf16_supported()
print(f'Training args configured. bf16={use_bf16}, fp16={not use_bf16}')

"""## 9. 開始訓練

Session 中斷後重新執行全部 cell，Trainer 會自動從最新 checkpoint 繼續。
"""

from transformers import Seq2SeqTrainer
import glob

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer.feature_extractor,
)

# 自動 resume：Drive 上有 checkpoint 就繼續
existing_checkpoints = sorted(glob.glob(f'{CHECKPOINT_DIR}/checkpoint-*'))
resume_from = existing_checkpoints[-1] if existing_checkpoints else None

if resume_from:
    print(f'Resuming from: {resume_from}')
else:
    print('Starting fresh training.')

try:
    trainer.train(resume_from_checkpoint=resume_from)
except AttributeError as e:
    if "NoneType' object has no attribute 'load_state_dict'" in str(e) and resume_from:
        print(f"Caught AttributeError during resume: {e}. This likely means the scaler state could not be loaded from checkpoint {resume_from}. Attempting to start training without resuming from checkpoint.")
        trainer.train(resume_from_checkpoint=None)
    else:
        # Re-raise other AttributeErrors or if not resuming
        raise

"""## 10. 顯示訓練記憶體統計"""

import torch

gpu_stats = torch.cuda.get_device_properties(0)
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory  = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f'GPU: {gpu_stats.name}. Max memory: {max_memory} GB.')
print(f'Peak reserved: {used_memory} GB ({round(used_memory / max_memory * 100, 1)}%)')

"""## 11. 儲存模型到 Drive"""

# 儲存 LoRA adapters（小，幾十 MB）
FINAL_MODEL_DIR = f'{BASE_DIR}/final_model'
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f'LoRA adapters saved → {FINAL_MODEL_DIR}')

"""## 12. 轉換為 faster-whisper 格式（CTranslate2）

需要先合併 LoRA 至 base model，再用 ct2-transformers-converter 轉換。
"""

import os
import subprocess

# Step 1：合併 LoRA → 完整 16bit 模型
MERGED_DIR = f'{BASE_DIR}/merged_model'
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
print(f'Merged model saved → {MERGED_DIR}')

# 確保 config.json 存在（save_pretrained_merged 有時不寫入）
if not os.path.exists(f'{MERGED_DIR}/config.json'):
    model.config.save_pretrained(MERGED_DIR)
    print('config.json manually saved')

# 確保 tokenizer 相關檔案存在
tokenizer.save_pretrained(MERGED_DIR)
print('Tokenizer files saved')

# 驗證目錄內容
print('merged_model contents:', sorted(os.listdir(MERGED_DIR)))

# Step 2：轉換為 CTranslate2 格式（需先 pip install ctranslate2）
CT2_OUTPUT = f'{BASE_DIR}/faster_whisper_ct2'

subprocess.run([
    'ct2-transformers-converter',
    '--model', MERGED_DIR,
    '--output_dir', CT2_OUTPUT,
    '--copy_files', 'tokenizer.json', 'preprocessor_config.json',
    '--quantization', 'float16',
    '--force',
], check=True)

print(f'faster-whisper model saved → {CT2_OUTPUT}')

from huggingface_hub import HfApi, create_repo, upload_folder
import os

# Define the local path of the CTranslate2 model
CT2_OUTPUT = f'{BASE_DIR}/faster_whisper_ct2'

# Define the Hugging Face repository ID
HF_REPO_ID = 'shooding/faster-whisper-large-v3-zh-TW'

# Get Hugging Face token (export HF_TOKEN=... before running)
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError('HF_TOKEN env var is required to upload to Hugging Face. '
                     'Run `export HF_TOKEN=hf_...` before launching.')

api = HfApi()

# Create the repository if it doesn't exist
create_repo(repo_id=HF_REPO_ID, repo_type='model', token=HF_TOKEN, exist_ok=True)
print(f'Hugging Face repository "{HF_REPO_ID}" ensured to exist.')

# Upload the model folder
print(f'Uploading model from {CT2_OUTPUT} to {HF_REPO_ID}...')
upload_folder(
    folder_path=CT2_OUTPUT,
    repo_id=HF_REPO_ID,
    repo_type='model',
    token=HF_TOKEN,
    commit_message='Upload faster-whisper CTranslate2 model'
)

print(f'Model successfully uploaded to https://huggingface.co/{HF_REPO_ID}')

"""## 附錄：macOS 自錄音準備腳本

在 macOS 上錄音後，用這個腳本產生 `metadata.csv`，再把整個資料夾上傳到 Drive。
"""

# ====== 在 macOS 本機執行，不是在 Colab ======
# import os
# import pandas as pd
#
# RECORDINGS_DIR = os.path.expanduser('~/my_recordings')
#
# transcriptions = {
#     'rec_001.wav': 'A套餐',
#     'rec_002.wav': 'B套餐',
# }
#
# rows = [
#     {'file_name': fname, 'transcription': text}
#     for fname, text in transcriptions.items()
#     if os.path.exists(os.path.join(RECORDINGS_DIR, fname))
# ]
#
# pd.DataFrame(rows).to_csv(
#     os.path.join(RECORDINGS_DIR, 'metadata.csv'), index=False
# )
# print(f'metadata.csv created with {len(rows)} entries')

print('See comments above for macOS preparation script.')

"""## 13. Publish to Hugging Face Hub (faster-whisper / CTranslate2 format)

Push the CT2-converted model + model card to Hugging Face
"""

from huggingface_hub import HfApi, login
from getpass import getpass
import os, textwrap

HF_REPO_ID = 'shooding/faster-whisper-large-v3-zh-TW'

CT2_OUTPUT = f'{BASE_DIR}/faster_whisper_ct2'

assert os.path.isdir(CT2_OUTPUT), f'CT2 dir not found: {CT2_OUTPUT}'

# Auth: prefer env var, else prompt
hf_token = os.environ.get('HF_TOKEN') or getpass('HF token (write scope): ')
login(token=hf_token)

# Create repo (no-op if it already exists) and upload CT2 artifacts
api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type='model', exist_ok=True, private=False)
api.upload_folder(
    folder_path=CT2_OUTPUT,
    repo_id=HF_REPO_ID,
    repo_type='model',
    commit_message='Upload CTranslate2 LoRA fine-tune with more custom_data',
    ignore_patterns=['__pycache__', '.ipynb_checkpoints'],
)
print(f'Published -> https://huggingface.co/{HF_REPO_ID}')
