# -*- coding: utf-8 -*-
"""upload_to_hf.py

Publish the CTranslate2 (faster-whisper) model produced by
`whisper_taiwan_finetune.py` to the Hugging Face Hub.

Prerequisite: training has finished and the CT2 artifact exists at
`{BASE_DIR}/faster_whisper_ct2` (the training script writes it in section 12).

Auth: export a write-scope token before running, or it will prompt:
    export HF_TOKEN=hf_...
    ./venv/bin/python upload_to_hf.py
"""

import os
from getpass import getpass
from huggingface_hub import HfApi

# Same path convention as whisper_taiwan_finetune.py (override via env var).
BASE_DIR   = os.environ.get('TAIWAN_WHISPER_DIR',
                            os.path.expanduser('~/taiwan_finetune/work'))
CT2_OUTPUT = f'{BASE_DIR}/faster_whisper_ct2'
HF_REPO_ID = os.environ.get('HF_REPO_ID', 'shooding/faster-whisper-large-v3-zh-TW')

assert os.path.isdir(CT2_OUTPUT), (
    f'CT2 dir not found: {CT2_OUTPUT}. Run whisper_taiwan_finetune.py first '
    f'(or set TAIWAN_WHISPER_DIR).'
)

# Auth: prefer env var, else prompt for a write-scope token.
hf_token = os.environ.get('HF_TOKEN') or getpass('HF token (write scope): ')

api = HfApi(token=hf_token)
api.create_repo(repo_id=HF_REPO_ID, repo_type='model', exist_ok=True, private=False)
print(f'Hugging Face repository "{HF_REPO_ID}" ensured to exist.')

print(f'Uploading {CT2_OUTPUT} → {HF_REPO_ID} ...')
api.upload_folder(
    folder_path=CT2_OUTPUT,
    repo_id=HF_REPO_ID,
    repo_type='model',
    commit_message='Upload faster-whisper CTranslate2 LoRA fine-tune',
    ignore_patterns=['__pycache__', '.ipynb_checkpoints'],
)
print(f'Published → https://huggingface.co/{HF_REPO_ID}')
