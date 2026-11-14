#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Quranic Speech Recognition (`سورة محمد` - Surah 47) — Custom PyTorch & Torchaudio Architecture
#
# ### Architecture Pipeline
# ```
# Audio
#   ↓
# Log-Mel
#   ↓
# CNN
#   ↓
# Conformer Layer 1
#   ↓
# Conformer Layer 2
#   ├── Intermediate CTC
#   └── Self-Conditioning
#   ↓
# Conformer Layer 3
#   ↓
# Conformer Layer 4
#   ↓
# Transformer Decoder
#   ↓
# Arabic Text
# ```
#
# * **Acoustic Front-End (`torchaudio`)**: 16 kHz mono waveform $\rightarrow$ 10-step deterministic preprocessing (Part A) + 7-step stochastic 2x waveform augmentation (Part B) $\rightarrow$ 80-bin Log-Mel Spectrogram (`25ms` window, `10ms` hop) + Per-Mel-Bin CMVN + Tajweed-Safe `SpecAugment`.
# * **CNN Subsampling (2 Layers)**: Two 1D Convolutional layers (`Conv1d` + `LayerNorm` + `GELU` + `Dropout`) with stride 2 in the first layer ($2\times$ temporal reduction to `20ms` frames) and strict inter-layer padding mask enforcement.
# * **Conformer Layer 1 & Conformer Layer 2**: Macaron-style Conformer blocks with Masked GroupNorm (`masked_group_norm`) inside the convolution module so padded frames never leak into valid frames.
# * **Mid-Conformer Intermediate CTC & Self-Conditioning (at Layer 2)**: Projects Layer 2 features to vocabulary probabilities (`inter_ctc_head`) for auxiliary CTC alignment loss and feeds softmax predictions back into Layer 2 representations via a gated linear projection (`inter_ctc_cond` + `inter_ctc_gate`).
# * **Conformer Layer 3 & Conformer Layer 4**: Deep contextual refinement conditioned on Layer 2 phonetic predictions.
# * **Transformer Decoder $\rightarrow$ Arabic Text**: 4-Layer Autoregressive Transformer Decoder with local causal self-attention window (`decoder_context_window=14`) and weight-tied output projection generating fully diacritized Quranic Arabic text.
# * **`EPOCHS = 60`** with Branch-Separated Checkpoints (`best_decoder_cer.pt`, `best_decoder_wer.pt`, `best_ctc_cer.pt`, `best_acoustic_balanced.pt`) + Decoder-driven early stopping.

# %% [markdown]
# ## Step 1 — Install Required Libraries

# %%
get_ipython().system(
    'pip -q install '
    '"datasets==3.3.2" '
    '"jiwer==3.0.5" '
    '"librosa>=0.10" '
    '"scipy>=1.10" '
    '"matplotlib>=3.7" "plotly>=5.18" pandas "arabic-reshaper>=3.0.0" "python-bidi>=0.4.2" '
    'torchaudio soundfile huggingface_hub'
)

# %% [markdown]
# ## Step 2 — Import Libraries & Configure GPU Device

# %%
import os
import io
import re
import math
import time
import shutil
import copy
import json
import random
import unicodedata
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
import jiwer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF
from torchaudio.models import Conformer
from torch.utils.data import Dataset as TorchDataset, DataLoader

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Active Device:", device)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## Step 3 — Mount Google Drive and Configure Project Directories (سورة محمد - ID: 47)

# %%
import glob
from datetime import datetime
from google.colab import drive

drive.mount("/content/drive")

SURAH_IDS = [47]                       # multi-Surah ready, e.g. [47, 48, 49]
SURAH_ID = SURAH_IDS[0]
SPLIT_MODE = "unseen_reciters"         # or "unseen_reciters_and_unseen_ayahs" (also holds out complete verses from training)
HELDOUT_AYAH_FRACTION = 0.15
SURAH_NAME_AR = "سورة محمد"
SURAH_SLUG = f"surah_{SURAH_ID}"
CONFORMER_NORM_MODE = "masked_group_norm"   # "masked_group_norm" | "unmasked_group_norm" | "frame_layer_norm"

PROJECT_DIR = f"/content/drive/MyDrive/quran-hybrid-asr-{SURAH_SLUG}"
RAW_DIR = os.path.join(PROJECT_DIR, "raw_dataset")
EXPERIMENT_TAG = f"conformer_interctc_selfcond_{CONFORMER_NORM_MODE}"
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, f"best_conformer_asr_{EXPERIMENT_TAG}.pt")
HISTORY_PATH = os.path.join(PROJECT_DIR, f"training_history_{EXPERIMENT_TAG}.json")
PERIODIC_CSV_PATH = os.path.join(PROJECT_DIR, f"periodic_5epoch_metrics_{EXPERIMENT_TAG}.csv")
LOCAL_RAW_DIR = f"/content/{SURAH_SLUG}_raw_dataset"
PLOTS_DIR = os.path.join(PROJECT_DIR, f"visualizations_{EXPERIMENT_TAG}")
BRANCH_CKPT_DIR = os.path.join(PROJECT_DIR, f"branch_checkpoints_{EXPERIMENT_TAG}")
BRANCH_CKPT_PATHS = {
    "decoder_cer": os.path.join(BRANCH_CKPT_DIR, "best_decoder_cer.pt"),
    "decoder_wer": os.path.join(BRANCH_CKPT_DIR, "best_decoder_wer.pt"),
    "ctc_cer": os.path.join(BRANCH_CKPT_DIR, "best_ctc_cer.pt"),
    "acoustic_balanced": os.path.join(BRANCH_CKPT_DIR, "best_acoustic_balanced.pt"),
}
CHECKPOINT_PATH = BRANCH_CKPT_PATHS["decoder_cer"]   # primary model = lowest Validation Decoder CER
EXTRA_CHECKPOINTS = {}

os.makedirs(PROJECT_DIR, exist_ok=True)

print("Project directory:", PROJECT_DIR)
print("Target Surah:", SURAH_NAME_AR, f"(ID: {SURAH_ID})")

# %% [markdown]
# ## Step 3.2 — Clean Up Old Training Artifacts in Google Drive to Start Fresh
# This cell scans your project directory in Google Drive, finds older training outputs (`checkpoints*`, `final_model*`, `visualizations*`, logs, the saved `best_hybrid_conformer_model.pt`, training history and old PNG plots), sorts folders from oldest to newest, and deletes them so you can start a fresh run.
# Note: `raw_dataset` is preserved so you do not need to re-download the dataset from Hugging Face.

# %%
# Safety Protection: Default to False to prevent accidental deletion of good checkpoints on rerun.
# Set CLEANUP_OLD_TRAINING_RUNS = True only when you explicitly desire a completely fresh run.
CLEANUP_OLD_TRAINING_RUNS = False
DELETE_ALL_TRAINING_ARTIFACTS = False  # Set to True only when intentionally wiping all previous artifacts
KEEP_NEWEST_N_FOLDERS = 3              # Keeps recent checkpoint folders intact if cleanup is enabled

if CLEANUP_OLD_TRAINING_RUNS and os.path.isdir(PROJECT_DIR):
    candidate_prefixes = ("checkpoints", "final_model", "visualizations", "logs", "runs")
    training_folders = []

    for entry in os.scandir(PROJECT_DIR):
        if entry.is_dir() and entry.name != "raw_dataset":
            if entry.name.startswith(candidate_prefixes):
                training_folders.append((entry.stat().st_mtime, entry.path, entry.name))

    training_folders.sort(key=lambda x: x[0])  # oldest -> newest

    if not training_folders:
        print("No old training folders found in Google Drive. Ready for a clean start!")
    else:
        print(f"Found {len(training_folders)} existing training folder(s) in Google Drive:")
        for mtime, path, name in training_folders:
            print(f"  - {name} (Last modified: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')})")

        if DELETE_ALL_TRAINING_ARTIFACTS:
            folders_to_delete = training_folders
        else:
            cutoff = max(0, len(training_folders) - KEEP_NEWEST_N_FOLDERS)
            folders_to_delete = training_folders[:cutoff]

        for mtime, path, name in folders_to_delete:
            print(f"Deleting old folder: {path} ...")
            shutil.rmtree(path, ignore_errors=True)

    if DELETE_ALL_TRAINING_ARTIFACTS:
        stale_files = (
            glob.glob(os.path.join(PROJECT_DIR, "*.png"))
            + glob.glob(os.path.join(PROJECT_DIR, "*.pt"))
            + glob.glob(os.path.join(PROJECT_DIR, "training_history.json"))
            + glob.glob(os.path.join(PROJECT_DIR, "evaluation_predictions.csv"))
        )
        for fpath in stale_files:
            try:
                os.remove(fpath)
                print(f"Removed old file: {os.path.basename(fpath)}")
            except OSError:
                pass

    print("Google Drive cleanup complete! Starting fresh.")

os.makedirs(PLOTS_DIR, exist_ok=True)
print("Active Checkpoint file:", CHECKPOINT_PATH)
print("Active Visualizations directory:", PLOTS_DIR)

# %% [markdown]
# ## Step 3.5 — Configure Hugging Face API Token (`HF_TOKEN`)

# %%
from getpass import getpass
from google.colab import userdata
from huggingface_hub import login

hf_token = None

try:
    hf_token = userdata.get("HF_TOKEN")
    print("Successfully retrieved HF_TOKEN from Colab Secrets.")
except Exception:
    print("HF_TOKEN not found in Colab Secrets.")

if not hf_token:
    hf_token = getpass("Enter your Hugging Face API token (or press Enter to skip): ").strip()

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token, add_to_git_credential=False)
    print("Logged in to Hugging Face Hub successfully!")
else:
    print("Proceeding without an API token (public access mode).")

# %% [markdown]
# ## Step 3.8 — Filter and Download Records for سورة محمد (Surah 47) Only

# %%
from huggingface_hub import HfApi
from datasets import load_dataset, load_from_disk, Audio

DATASET_ID = "Buraaq/quran-md-ayahs"

# Dictionary mapping standard reciter IDs to Arabic Reciter Names
RECITER_ARABIC_MAP = {
    "abdul_basit_murattal": "عبد الباسط عبد الصمد - مرتل",
    "abdul_basit_mujawwad": "عبد الباسط عبد الصمد - مجود",
    "abdullah_basfar": "عبد الله بصفر",
    "abdurrahmaan_as_sudais": "عبد الرحمن السديس",
    "abu_bakr_ash_shaatree": "أبو بكر الشاطري",
    "ahmed_ibn_ali_al_ajamy": "أحمد بن علي العجمي",
    "alafasy": "مشاري راشد العفاسي",
    "ali_jaber": "علي جابر",
    "ayman_suowaid": "أيمن سويد",
    "banna": "محمود علي البنا",
    "fares_abbad": "فارس عباد",
    "ghamadi": "سعد الغامدي",
    "hani_ar_rifai": "هاني الرفاعي",
    "hudhaify": "علي بن عبد الرحمن الحذيفي",
    "husary": "محمود خليل الحصري",
    "husary_muallim": "محمود خليل الحصري - معلم",
    "husary_mujawwad": "محمود خليل الحصري - مجود",
    "ibrahim_akhdar": "إبراهيم الأخضر",
    "khalefa_al_tunaiji": "خليفة الطنيجي",
    "maher_al_muaiqly": "ماهر المعيقلي",
    "minshawy_mujawwad": "محمد صديق المنشاوي - مجود",
    "minshawy_murattal": "محمد صديق المنشاوي - مرتل",
    "mohammad_al_tablaway": "محمد محمود الطبلاوي",
    "\x6d\x75\x68\x61\x6d\x6d\x61\x64_ayyoub": "محمد أيوب",
    "\x6d\x75\x68\x61\x6d\x6d\x61\x64_jibreel": "محمد جبريل",
    "muhsin_al_qasim": "عبد المحسن القاسم",
    "nasser_al_qatami": "ناصر القطامي",
    "salah_al_budair": "صلاح البدير",
    "salah_bukhatir": "صلاح بو خاطر",
    "saood_ash_shuraym": "سعود الشريم",
    "yasser_ad_dussary": "ياسر الدوسري",
}

def get_arabic_reciter_name(example_or_id):
    """Returns the Arabic name of the reciter from dataset metadata or mapping dictionary."""
    if isinstance(example_or_id, dict):
        if "reciter_ar" in example_or_id and example_or_id["reciter_ar"]:
            return str(example_or_id["reciter_ar"])
        rec_id = str(example_or_id.get("reciter_id", example_or_id.get("reciter_name", "")))
    else:
        rec_id = str(example_or_id)
    key = rec_id.strip().lower()
    if key in RECITER_ARABIC_MAP:
        return RECITER_ARABIC_MAP[key]
    for map_key, ar_name in RECITER_ARABIC_MAP.items():
        if map_key in key or key in map_key:
            return ar_name
    return rec_id

metadata_file = os.path.join(RAW_DIR, "dataset_info.json")

if os.path.isfile(metadata_file):
    print(f"Found cached dataset for {SURAH_NAME_AR} in Google Drive (skipping download).")
else:
    print(f"Searching Parquet files and filtering {SURAH_NAME_AR} (Surah ID: {SURAH_ID})...")
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=DATASET_ID, repo_type="dataset")
    parquet_files = [
        f"hf://datasets/{DATASET_ID}/{path}"
        for path in repo_files
        if path.startswith("data/") and path.endswith(".parquet")
    ]
    print("Parquet files found:", len(parquet_files))

    filtered_ds = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        filters=[("surah_id", "in", SURAH_IDS)],
    )

    print("Total records downloaded:", len(filtered_ds))
    print("Unique verses:", len(set(filtered_ds["ayah_id"])))
    if "reciter_id" in filtered_ds.column_names:
        print("Unique reciters:", len(set(filtered_ds["reciter_id"])))

    os.makedirs(RAW_DIR, exist_ok=True)
    filtered_ds.save_to_disk(RAW_DIR)
    print("Saved filtered dataset to Google Drive:", RAW_DIR)

# %% [markdown]
# ## Step 3.9 — Copy Dataset to Local Colab Disk for Fast I/O

# %%
if not os.path.isfile(os.path.join(LOCAL_RAW_DIR, "dataset_info.json")):
    if os.path.exists(LOCAL_RAW_DIR):
        shutil.rmtree(LOCAL_RAW_DIR)
    shutil.copytree(RAW_DIR, LOCAL_RAW_DIR)

raw_dataset = load_from_disk(LOCAL_RAW_DIR)
TOTAL_SURAH_VERSES = len(set(raw_dataset["ayah_id"]))
print("Local dataset ready:", len(raw_dataset), "records")
print("Total unique verses in Surah:", TOTAL_SURAH_VERSES)
print("Dataset columns:", raw_dataset.column_names)

# Build a lookup table from reciter_id to Arabic reciter name using dataset columns if available
reciter_col = "reciter_id" if "reciter_id" in raw_dataset.column_names else "reciter_name"
if "reciter_ar" in raw_dataset.column_names:
    for r_id, r_ar in zip(raw_dataset[reciter_col], raw_dataset["reciter_ar"]):
        if r_id and r_ar:
            RECITER_ARABIC_MAP[str(r_id).strip().lower()] = str(r_ar)

# %% [markdown]
# ## Step 4 — Split Dataset by Reciters (80% Train / 10% Val / 10% Test) & Validate Audio
# Whole reciters are held out, so validation and test measure generalization to **unseen voices**:
# - **80%** of reciters -> Training
# - **10%** of reciters -> Evaluation / Validation during training
# - **10%** of reciters -> Final Testing

# %%
all_reciters = sorted(set(raw_dataset[reciter_col]))

rng = random.Random(SEED)
rng.shuffle(all_reciters)

n_eval = max(1, round(len(all_reciters) * 0.10))
n_test = max(1, round(len(all_reciters) * 0.10))

eval_reciters = set(all_reciters[:n_eval])
test_reciters = set(all_reciters[n_eval : n_eval + n_test])
train_reciters = set(all_reciters[n_eval + n_test :])

reciter_column = raw_dataset[reciter_col]
n_train_rec = sum(r in train_reciters for r in reciter_column)
n_eval_rec = sum(r in eval_reciters for r in reciter_column)
n_test_rec = sum(r in test_reciters for r in reciter_column)

print(f"Surah: {SURAH_NAME_AR}")
print(f"Total Reciters: {len(all_reciters)}")
print(f"Training Reciters (80%):   {len(train_reciters)} reciters -> {n_train_rec} audio records")
print(f"Evaluation Reciters (10%): {len(eval_reciters)} reciters -> {n_eval_rec} audio records")
print(f"Testing Reciters (10%):    {len(test_reciters)} reciters -> {n_test_rec} audio records")

print("\nHeld-out Evaluation Reciters (10% - Arabic):")
for r_id in sorted(eval_reciters):
    print(f"  - {get_arabic_reciter_name(r_id)} ({r_id})")

print("\nHeld-out Testing Reciters (10% - Arabic):")
for r_id in sorted(test_reciters):
    print(f"  - {get_arabic_reciter_name(r_id)} ({r_id})")

print("\nTraining Reciters (80% - Arabic):")
for r_id in sorted(train_reciters):
    print(f"  - {get_arabic_reciter_name(r_id)}")

raw_dataset_no_decode = raw_dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

def filter_valid_split(ds, target_reciter_set, exclude_ayahs=None, only_ayahs=None):
    ayah_vals = ds["ayah_id"]
    indices = [
        i for i, r in enumerate(ds[reciter_col])
        if r in target_reciter_set
        and (not exclude_ayahs or ayah_vals[i] not in exclude_ayahs)
        and (not only_ayahs or ayah_vals[i] in only_ayahs)
    ]
    sub_ds = ds.select(indices)

    def is_valid(example):
        try:
            audio_dict = example["audio"]
            if not audio_dict or "bytes" not in audio_dict or audio_dict["bytes"] is None:
                return False
            with io.BytesIO(audio_dict["bytes"]) as f:
                arr, sr = sf.read(f)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            dur = len(arr) / sr
            return 0.4 <= dur <= 30.0 and np.isfinite(arr).all() and np.max(np.abs(arr)) > 1e-4
        except Exception:
            return False

    sub_ds = sub_ds.filter(is_valid, desc="Validating audio")
    return sub_ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))

all_ayah_ids = sorted(set(raw_dataset["ayah_id"]))
heldout_ayahs = set()
if SPLIT_MODE == "unseen_reciters_and_unseen_ayahs":
    _rng_a = random.Random(SEED + 1)
    _shuf = all_ayah_ids[:]
    _rng_a.shuffle(_shuf)
    heldout_ayahs = set(_shuf[: max(2, round(len(_shuf) * HELDOUT_AYAH_FRACTION))])
    print(f"SPLIT_MODE={SPLIT_MODE}: holding out {len(heldout_ayahs)} complete ayahs from TRAINING: {sorted(heldout_ayahs)}")
train_audio = filter_valid_split(raw_dataset_no_decode, train_reciters, exclude_ayahs=heldout_ayahs)
eval_audio = filter_valid_split(raw_dataset_no_decode, eval_reciters)
test_audio = filter_valid_split(raw_dataset_no_decode, test_reciters)
test_unseen_ayah_audio = (
    filter_valid_split(raw_dataset_no_decode, test_reciters, only_ayahs=heldout_ayahs) if heldout_ayahs else None
)
# Research hygiene: keep a fresh lock-box test for the final claim; Test results already used for exploration are exploratory only.

print(f"Train clips (80%): {len(train_audio)} | Val clips (10%): {len(eval_audio)} | Test clips (10%): {len(test_audio)}")

# %% [markdown]
# ## Step 5 — Comprehensive Audio/Text Preprocessing (Part A), Multi-Stage 2x Waveform Augmentation (Part B) & Tajweed-Safe SpecAugment
#
# ### Part A: Deterministic Preprocessing Pipeline (Applied to Training, Evaluation & Testing):
# 1. **Mono Conversion & NaN/Inf Sanitization.**
# 2. **DC Offset Removal & Transient Click De-Spiking (`despike_transients`):** Suppresses sudden non-speech microphone pops.
# 3. **6th-Order Butterworth Band-Pass Filter ($65\text{ Hz} - 7600\text{ Hz}$):** Removes low-frequency hum ($<65\text{ Hz}$) and high-frequency aliasing noise ($>7600\text{ Hz}$).
# 4. **Stationary Spectral Noise Gate (`spectral_noise_gate`):** Estimates stationary background floor and gently attenuates low-energy noise bins in the STFT domain.
# 5. **Subtle Pre-Emphasis / High-Frequency Clarity Boost:** Enhances Arabic consonant articulation (pharyngeal, emphatic, and sibilant letters).
# 6. **Boundary Silence Trimming (`top_db=35`) with $120\text{ ms}$ Breath Margin:** Preserves final *Qalqalah*, *Hams*, and *Madd* releases.
# 7. **Internal Long-Pause Compression (`compress_internal_pauses`):** Compresses long internal pauses ($>0.32\text{ s}$) so cross-attention and CTC alignment stay tight.
# 8. **Cosine Fade-In / Fade-Out Window ($15\text{ ms}$):** Eliminates boundary clicks.
# 9. **Target RMS Energy Normalization ($-20\text{ dBFS}$) + Soft-Knee Tanh Limiter (`0.95` ceiling).**
# 10. **Quranic Text Canonicalization (`preprocess_quran_text`):** Normalizes Unicode (`NFC`), strips zero-width characters, canonicalizes consecutive *Shaddah* + vowel ordering, and keeps all Quranic diacritics intact.
#
# ### Part B: Multi-Stage Waveform Augmentation (Applied to a 2nd Copy of Training Data for 2x Dataset Size):
# 1. **Time Stretching (`librosa.effects.time_stretch`):** $0.90\times - 1.12\times$ rate perturbation (with CTC $T \ge L+R$ & 38s tail-safety guards).
# 2. **Pitch Shifting (`librosa.effects.pitch_shift`):** $\pm 1.2$ semitones.
# 3. **Vocal Tract Length / Formant Perturbation (`perturb_formants`):** Simulates different vocal tract resonances across unseen reciters.
# 4. **Simulated Mosque/Room Reverb:** Impulse response convolution with exponential decay.
# 5. **Random Parametric Band-Pass / Microphone EQ Filter.**
# 6. **Random Temporal Shift & Silence Padding.**
# 7. **Combined White & Pink ($1/f$) Ambient Noise Injection ($20\text{ dB} - 35\text{ dB}$ SNR).**
#
# ### Tajweed-Safe `SpecAugment` (Spectrogram Augmentation):
# 1. **Time Masking:** Gentle random temporal masking (`max_time_ratio=0.045`).
# 2. **Frequency Masking:** Gentle random frequency-band masking (`num_freq_masks=2`, `freq_mask_param=8`).
# 3. **Optional Local Time Warping (`enable_time_warp=False` by default):** Local piecewise time-axis warp, off by default so delicate Tajweed durations are preserved.

# %%
SR = 16000
N_MELS = 80
MEL_FMIN = 60.0
MEL_FMAX = 7600.0
MIN_AUDIO_SECONDS = 0.4
MAX_AUDIO_SECONDS = 30.0
MAX_AUG_AUDIO_SECONDS = 38.0
BANDPASS_SOS = butter(6, [65.0, 7600.0], btype="bandpass", fs=SR, output="sos")
aug_rng = np.random.default_rng(SEED)


def preprocess_quran_text(text):
    """
    Part A (Step 10) — Quranic Text Canonicalization:
    Normalizes Unicode (NFC), strips invisible zero-width formatting and non-phonetic pause/ornament
    signs (Waqf marks ۖ ۗ ۘ ۙ ۚ ۛ ۜ, ۝, ۞, ۩), canonicalizes consecutive Shaddah (\\u0651) + vowel
    ordering so Shaddah always precedes the vowel, and keeps 100% of Quranic phonetic diacritics intact.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    # Remove invisible zero-width and directional formatting characters
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\uFEFF]", "", text)
    # Remove silent editorial pause/ornament signs (Waqf marks, Rub El Hizb ۞, Sajdah ۩, verse digits)
    text = re.sub(r"[\u06D6-\u06DE\u06E9\u0660-\u0669\u06F0-\u06F9]", "", text)
    # Canonicalize Shaddah (\u0651) + Harakah ordering so Shaddah always precedes the vowel
    text = re.sub(r"([\u064B-\u0650\u0652\u0670])\u0651", "\u0651\\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_arabic(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u064B-\u065F\u0670\u0640\u06D6-\u06ED]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class QuranCharTokenizer:
    """Character-level Arabic + Diacritics tokenizer for custom Seq2Seq & CTC decoding."""
    def __init__(self, texts):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        specials = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        unique_chars = sorted(set("".join(preprocess_quran_text(t) for t in texts)))
        self.vocab = specials + unique_chars

        self.char2id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id2char = {idx: ch for idx, ch in enumerate(self.vocab)}

        self.pad_id = self.char2id[self.pad_token]
        self.sos_id = self.char2id[self.sos_token]
        self.eos_id = self.char2id[self.eos_token]
        self.unk_id = self.char2id[self.unk_token]

    def __len__(self):
        return len(self.vocab)

    def encode(self, text):
        clean = preprocess_quran_text(text)
        ids = [self.sos_id] + [self.char2id.get(ch, self.unk_id) for ch in clean] + [self.eos_id]
        return ids

    def decode(self, token_ids):
        chars = []
        for idx in token_ids:
            idx = int(idx)
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.sos_id, self.unk_id):
                continue
            chars.append(self.id2char.get(idx, ""))
        return "".join(chars).strip()


tokenizer = QuranCharTokenizer(raw_dataset["ayah_ar"])
print("Vocabulary Size (Arabic letters + Tashkeel + Specials):", len(tokenizer))

# ---------------------------------------------------------------------------
# Native torchaudio Slaney Log-Mel transforms (focused on 60-7600 Hz vocal/Tajweed band)
# ---------------------------------------------------------------------------
mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=400,
    win_length=400,
    hop_length=160,
    f_min=MEL_FMIN,
    f_max=MEL_FMAX,
    n_mels=N_MELS,
    power=2.0,
    norm="slaney",
    mel_scale="slaney",
)
amplitude_to_db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

# Legacy single-mask instances kept for backward compatibility; training uses Tajweed-Safe SpecAugment below
time_mask_transform = T.TimeMasking(time_mask_param=20)
freq_mask_transform = T.FrequencyMasking(freq_mask_param=8)

MAX_SAMPLES = int(MAX_AUDIO_SECONDS * SR)
MAX_AUG_SAMPLES = int(MAX_AUG_AUDIO_SECONDS * SR)  # Prevents slow time-stretch (0.90x / 0.825x) from chopping verse tails

# Pre-cached polyphase resamplers (GCD-aligned frequencies -> ~8x faster CPU speed perturbation)
FAST_SPEED_FREQS = [13200, 13600, 14000, 14400, 14800, 15200, 15600, 16400, 16800, 17200, 17600, 18400, 19600]
FAST_SPEED_RESAMPLERS = [T.Resample(orig_freq=f, new_freq=SR) for f in FAST_SPEED_FREQS]


def _build_mosque_rir_bank(sr=SR, num_rirs=20, seed=42):
    """
    Pre-synthesizes a bank of realistic multi-reflection room & mosque impulse responses
    (RT60 0.10s studio booth up to 0.55s large reverberant Taraweeh hall).
    """
    gen = torch.Generator().manual_seed(seed)
    bank = []
    rt60_values = torch.linspace(0.10, 0.55, steps=num_rirs)
    for idx in range(num_rirs):
        rt60 = float(rt60_values[idx].item())
        ir_len = int(min(0.58, rt60 * 1.15) * sr)
        t = torch.linspace(0.0, ir_len / sr, ir_len)
        decay_rate = 6.9078 / max(0.08, rt60)
        diffuse = torch.randn(ir_len, generator=gen) * torch.exp(-decay_rate * t)
        if ir_len > 64:
            diffuse = AF.highpass_biquad(diffuse, sr, cutoff_freq=100.0)
            diffuse = AF.lowpass_biquad(diffuse, sr, cutoff_freq=5200.0)
        ir = 0.34 * diffuse
        ir[0] = 1.0  # Direct acoustic path
        early_delays = torch.randint(int(0.012 * sr), max(int(0.015 * sr), min(ir_len - 1, int(0.065 * sr))), (5,), generator=gen)
        early_gains = torch.linspace(0.44, 0.16, steps=5) * (1.0 - 2.0 * torch.randint(0, 2, (5,), generator=gen).float())
        for d_idx, g_val in zip(early_delays.tolist(), early_gains.tolist()):
            ir[d_idx] += g_val
        ir = ir / (ir.norm() + 1e-8)
        bank.append(ir)
    return bank


MOSQUE_RIR_BANK = _build_mosque_rir_bank(SR, num_rirs=24, seed=SEED)


# =============================================================================
# Part A: Deterministic Audio Preprocessing Pipeline (Train, Eval & Test)
# =============================================================================

def despike_transients(y: np.ndarray, sigma_thresh: float = 6.5) -> np.ndarray:
    """Part A (Step 2): Suppresses isolated single-sample digital clicks/pops without altering speech peaks."""
    diff = np.diff(y, prepend=y[0])
    std_diff = np.std(diff) + 1e-8
    spikes = np.abs(diff) > (sigma_thresh * std_diff)
    if np.any(spikes):
        y = y.copy()
        spike_indices = np.where(spikes)[0]
        for idx in spike_indices:
            if 0 < idx < len(y) - 1:
                y[idx] = 0.5 * (y[idx - 1] + y[idx + 1])
    return y


def spectral_noise_gate(y: np.ndarray, sample_rate: int = SR, prop_decrease: float = 0.60) -> np.ndarray:
    """
    Part A (Step 4): Gentle stationary spectral noise gating in the STFT domain.
    Estimates the noise floor from the quietest 10% frames and attenuates stationary background hiss.
    """
    if len(y) < 1024:
        return y
    stft = librosa.stft(y, n_fft=512, hop_length=160, win_length=400)
    mag = np.abs(stft)
    frame_energy = np.mean(mag, axis=0)
    quiet_thresh = np.percentile(frame_energy, 10)
    quiet_frames = mag[:, frame_energy <= quiet_thresh]
    if quiet_frames.shape[1] == 0:
        return y
    noise_profile = np.mean(quiet_frames, axis=1, keepdims=True)
    mask = np.clip((mag - noise_profile) / (mag + 1e-8), 1.0 - prop_decrease, 1.0)
    stft_denoised = stft * mask
    y_denoised = librosa.istft(stft_denoised, hop_length=160, win_length=400, length=len(y))
    return y_denoised.astype(np.float32)


def compress_internal_pauses(
    y: np.ndarray,
    sample_rate: int = SR,
    top_db: float = 34.0,
    max_pause_sec: float = 0.32,
) -> np.ndarray:
    """
    Part A (Step 7): Detects non-silent speech intervals and compresses overly long internal
    breathing pauses (>0.32s, common in slow Mujawwad recitations) down to `max_pause_sec`.
    """
    if len(y) < int(0.8 * sample_rate):
        return y

    intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    if len(intervals) <= 1:
        return y

    max_pause_samples = int(max_pause_sec * sample_rate)
    margin = int(0.06 * sample_rate)
    segments = []

    for idx, (start, end) in enumerate(intervals):
        s = max(0, start - margin) if idx == 0 else start
        e = min(len(y), end + margin) if idx == len(intervals) - 1 else end
        segments.append(y[s:e])

        if idx < len(intervals) - 1:
            next_start = intervals[idx + 1][0]
            gap_len = next_start - e
            if gap_len > 0:
                kept_gap = min(gap_len, max_pause_samples)
                segments.append(y[e : e + kept_gap])

    return np.concatenate(segments, axis=0).astype(np.float32)


def apply_fade_window(y: np.ndarray, sample_rate: int = SR, fade_ms: float = 15.0) -> np.ndarray:
    """Part A (Step 8): Applies a smooth 15ms cosine fade-in and fade-out window to eliminate boundary clicks."""
    fade_len = int((fade_ms / 1000.0) * sample_rate)
    if len(y) < 2 * fade_len or fade_len <= 1:
        return y
    y = y.copy()
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
    fade_out = fade_in[::-1]
    y[:fade_len] *= fade_in
    y[-fade_len:] *= fade_out
    return y


def rms_normalize_with_soft_limiter(
    y: np.ndarray,
    target_dbfs: float = -20.0,
    peak_ceiling: float = 0.95,
) -> np.ndarray:
    """Part A (Step 9): Normalizes waveform to -20 dBFS target RMS energy with a 0.95 soft-knee tanh limiter."""
    rms = np.sqrt(np.mean(y**2) + 1e-10)
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = min(target_rms / rms, 15.0)
    y = y * gain

    peak = np.max(np.abs(y))
    if peak > peak_ceiling:
        y = peak_ceiling * np.tanh(y / peak_ceiling)
    return y.astype(np.float32)


def trim_silence_torch(
    wav,
    sr=SR,
    top_db=35.0,
    lead_margin_sec=0.12,
    trail_margin_sec=0.12,
    margin_sec=None,
    frame_len=400,
    hop=160,
):
    """Torch helper for energy-based silence trimming with 120ms natural breath margin."""
    if wav.numel() < frame_len * 2:
        return wav
    if margin_sec is not None:
        lead_margin_sec = margin_sec
        trail_margin_sec = margin_sec
    frames = wav.unfold(0, frame_len, hop)
    rms = frames.pow(2).mean(dim=1).sqrt() + 1e-9
    db = 20.0 * torch.log10(rms / rms.max())
    voiced = torch.nonzero(db > -top_db).flatten()
    if voiced.numel() == 0:
        return wav
    lead_margin = int(lead_margin_sec * sr)
    trail_margin = int(trail_margin_sec * sr)
    start = max(0, int(voiced[0]) * hop - lead_margin)
    end = min(wav.numel(), int(voiced[-1]) * hop + frame_len + trail_margin)
    return wav[start:end]


def preprocess_clean_waveform(
    audio_array,
    sample_rate: int = SR,
    min_ctc_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Full 10-Step Deterministic Audio Preprocessing Pipeline (Part A — applied to Train, Eval & Test):
      1. Mono Conversion & NaN/Inf Sanitization.
      2. DC Offset Removal & Transient Click De-Spiking (`despike_transients`).
      3. 6th-Order Butterworth Band-Pass Filter (65 Hz - 7600 Hz).
      4. Stationary Spectral Noise Gate (`spectral_noise_gate`).
      5. Subtle Pre-Emphasis / High-Frequency Clarity Boost (15% blend, coef=0.95).
      6. Boundary Silence Trimming (`top_db=35`) with 120 ms Breath Margin.
      7. Internal Long-Pause Compression (>0.32 s) with CTC minimum-length safety check.
      8. Cosine Fade-In / Fade-Out Window (15 ms).
      9. Target RMS Energy Normalization (-20 dBFS) + Soft-Knee Tanh Limiter (0.95 ceiling).
     10. 30-second maximum window enforcement.
    """
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.detach().cpu().numpy()
    audio_array = np.asarray(audio_array, dtype=np.float32)
    audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Convert multi-channel audio to Mono
    if audio_array.ndim == 2:
        if audio_array.shape[0] <= 2:
            audio_array = audio_array.mean(axis=0)
        else:
            audio_array = audio_array.mean(axis=1)
    elif audio_array.ndim > 2:
        audio_array = audio_array.reshape(-1)

    if len(audio_array) == 0:
        return np.zeros(int(0.5 * SR), dtype=np.float32)

    if int(sample_rate) != SR:
        audio_array = librosa.resample(audio_array, orig_sr=int(sample_rate), target_sr=SR).astype(np.float32)
        sample_rate = SR

    # 2. Remove DC Offset & suppress isolated transient clicks
    audio_array = audio_array - np.mean(audio_array)
    audio_array = despike_transients(audio_array)

    # 3. 6th-Order Butterworth Band-Pass Filter (65 Hz - 7600 Hz)
    if len(audio_array) > 128:
        audio_array = sosfilt(BANDPASS_SOS, audio_array).astype(np.float32)

    # 4. Gentle Stationary Spectral Noise Gate
    audio_array = spectral_noise_gate(audio_array, sample_rate=sample_rate, prop_decrease=0.60)

    # 5. Subtle Pre-Emphasis blend (boosts high-frequency consonant clarity by 15%)
    pre_emph = librosa.effects.preemphasis(audio_array, coef=0.95)
    audio_array = (0.85 * audio_array + 0.15 * pre_emph).astype(np.float32)

    # Minimum waveform samples needed so 20ms Conformer frames (320 samples/frame) satisfy CTC T >= L + R
    min_required_samples = int((min_ctc_frames + 6) * 320) if min_ctc_frames is not None else 0

    # 6. Trim leading and trailing silence (`top_db=35`) while keeping a 120ms natural breath margin
    pre_trim = audio_array
    _, trim_idx = librosa.effects.trim(audio_array, top_db=35)
    pad_samples = int(0.12 * sample_rate)
    start_idx = max(0, int(trim_idx[0]) - pad_samples)
    end_idx = min(len(audio_array), int(trim_idx[1]) + pad_samples)
    if end_idx > start_idx and (end_idx - start_idx) >= min_required_samples:
        audio_array = audio_array[start_idx:end_idx]
    else:
        audio_array = pre_trim

    # 7. Compress overly long internal pauses (>0.32s) in slow recitations (only if CTC margin remains valid)
    compressed = compress_internal_pauses(audio_array, sample_rate=sample_rate, top_db=34, max_pause_sec=0.32)
    if len(compressed) >= min_required_samples:
        audio_array = compressed

    # 8. Apply 15ms cosine fade-in and fade-out
    audio_array = apply_fade_window(audio_array, sample_rate=sample_rate, fade_ms=15.0)

    # 9. Target RMS Normalization (-20 dBFS) with soft-knee limiter (0.95 ceiling)
    audio_array = rms_normalize_with_soft_limiter(audio_array, target_dbfs=-20.0, peak_ceiling=0.95)

    # 10. Enforce 30-second maximum clean window
    if len(audio_array) > MAX_SAMPLES:
        audio_array = audio_array[:MAX_SAMPLES]

    return np.clip(audio_array, -1.0, 1.0).astype(np.float32)


def preprocess_waveform(audio_array, sr=SR, min_ctc_frames: Optional[int] = None) -> torch.Tensor:
    """Torch wrapper around `preprocess_clean_waveform` (Part A) returning a 1D float32 Tensor."""
    clean_np = preprocess_clean_waveform(audio_array, sample_rate=sr, min_ctc_frames=min_ctc_frames)
    return torch.from_numpy(clean_np).float()


# =============================================================================
# Part B: Multi-Stage Waveform Augmentation (Applied to 2nd Copy of Training Data for 2x Dataset Size)
# =============================================================================

def generate_pink_noise(length: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generates realistic 1/f pink ambient noise in the frequency domain."""
    rng = rng or aug_rng
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.normal(0.0, 1.0, size=length).astype(np.float32)
    if length < 32:
        return white / (np.std(white) + 1e-8)
    fft_vals = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    scale = 1.0 / np.where(freqs == 0, 1.0, np.sqrt(freqs))
    scale[0] = 0.0
    pink = np.fft.irfft(fft_vals * scale, n=length).astype(np.float32)
    return pink / (np.std(pink) + 1e-8)


def perturb_formants(
    y: np.ndarray,
    sample_rate: int = SR,
    warp_range: Tuple[float, float] = (0.96, 1.04),
    max_samples: int = MAX_AUG_SAMPLES,
    min_samples: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Part B (Step 3) — Vocal Tract Length / Formant Perturbation (`perturb_formants`):
    Simulates different vocal tract resonances across unseen reciters via coupled resampling
    and inverse time-stretching so formant frequencies shift while duration is preserved.
    """
    rng = rng or aug_rng
    if len(y) < 256:
        return y.astype(np.float32)
    warp = float(rng.uniform(warp_range[0], warp_range[1]))
    if abs(warp - 1.0) < 1e-3:
        return y.astype(np.float32)
    warped_sr = max(8000, int(round(sample_rate * warp)))
    y_warped = librosa.resample(y, orig_sr=sample_rate, target_sr=warped_sr)
    if len(y_warped) == 0:
        return y.astype(np.float32)
    y_restored = librosa.effects.time_stretch(y_warped, rate=1.0 / warp)
    if min_samples <= len(y_restored) <= max_samples:
        return y_restored.astype(np.float32)
    return y.astype(np.float32)


def augment_waveform(
    audio_array: Union[np.ndarray, torch.Tensor],
    sample_rate: int = SR,
    min_ctc_frames: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Part B — Multi-Stage Stochastic Waveform Augmentation (Applied to a 2nd Copy of Training Data for 2x Dataset Size):
      1. Time Stretching (`librosa.effects.time_stretch`): 0.90x - 1.12x rate perturbation.
      2. Pitch Shifting (`librosa.effects.pitch_shift`): +/- 1.2 semitones.
      3. Vocal Tract Length / Formant Perturbation (`perturb_formants`): simulates unseen reciter resonances.
      4. Simulated Mosque/Room Reverb: impulse response convolution with exponential decay.
      5. Random Parametric Band-Pass / Microphone EQ Filter.
      6. Random Temporal Shift & Silence Padding (low-level natural room tone).
      7. Combined White & Pink (1/f) Ambient Noise Injection (20 dB - 35 dB SNR) + Soft-Knee Gain Scaling.
    """
    rng = rng or aug_rng
    if isinstance(audio_array, torch.Tensor):
        y = audio_array.detach().cpu().numpy().astype(np.float32).copy()
    else:
        y = np.asarray(audio_array, dtype=np.float32).copy()

    max_samples = MAX_AUG_SAMPLES  # 38s cap prevents slow 0.90x time-stretch from truncating verse tails
    min_samples = int((min_ctc_frames + 6) * 320) if min_ctc_frames is not None else int(0.4 * sample_rate)

    # 1. Time Stretching (0.90x to 1.12x) without altering pitch
    if rng.random() < 0.55 and len(y) > 512:
        rate = float(rng.uniform(0.90, 1.12))
        estimated_len = int(len(y) / rate)
        if min_samples <= estimated_len <= max_samples:
            y = librosa.effects.time_stretch(y, rate=rate).astype(np.float32)

    # 2. Pitch Shifting (-1.2 to +1.2 semitones)
    if rng.random() < 0.45 and len(y) > 512:
        n_steps = float(rng.uniform(-1.2, 1.2))
        y = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=n_steps).astype(np.float32)

    # 3. Vocal Tract Length / Formant Perturbation (`perturb_formants`)
    if rng.random() < 0.30 and len(y) > 512:
        y = perturb_formants(
            y,
            sample_rate=sample_rate,
            warp_range=(0.96, 1.04),
            max_samples=max_samples,
            min_samples=min_samples,
            rng=rng,
        )

    # 4. Simulated Mosque / Room Reverb via Impulse Response Convolution
    if rng.random() < 0.35 and len(y) > 256:
        reverb_duration = float(rng.uniform(0.06, 0.16))
        ir_len = int(reverb_duration * sample_rate)
        if ir_len > 10:
            t = np.linspace(0.0, 1.0, ir_len, dtype=np.float32)
            decay = np.exp(-6.5 * t)
            ir = rng.normal(0.0, 1.0, size=ir_len).astype(np.float32) * decay
            ir[0] = 1.0
            ir = ir / (np.linalg.norm(ir) + 1e-8)
            wet_mix = float(rng.uniform(0.08, 0.20))
            y_reverb = np.convolve(y, ir, mode="full")[: len(y)]
            y = ((1.0 - wet_mix) * y + wet_mix * y_reverb).astype(np.float32)

    # 5. Random Parametric Band-Pass / Microphone EQ Filter
    if rng.random() < 0.30 and len(y) > 128:
        low_cut = float(rng.uniform(70.0, 140.0))
        high_cut = float(rng.uniform(6200.0, 7500.0))
        eq_sos = butter(2, [low_cut, high_cut], btype="bandpass", fs=sample_rate, output="sos")
        y = sosfilt(eq_sos, y).astype(np.float32)

    # 6. Random Temporal Shift & Low-Level Breath/Silence Padding
    if rng.random() < 0.40:
        pad_front = int(rng.uniform(0.0, 0.20) * sample_rate)
        pad_back = int(rng.uniform(0.0, 0.20) * sample_rate)
        if len(y) + pad_front + pad_back <= max_samples:
            front_pad = generate_pink_noise(pad_front, rng=rng) * (10.0 ** (-52.0 / 20.0))
            back_pad = generate_pink_noise(pad_back, rng=rng) * (10.0 ** (-52.0 / 20.0))
            y = np.concatenate([front_pad, y, back_pad], axis=0).astype(np.float32)

    # 7. Combined White & Pink (1/f) Ambient Noise Injection (20 dB to 35 dB SNR)
    if rng.random() < 0.45 and len(y) > 32:
        sig_power = float(np.mean(y**2)) + 1e-9
        snr_db = float(rng.uniform(20.0, 35.0))
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        white_n = rng.normal(0.0, 1.0, size=len(y)).astype(np.float32)
        pink_n = generate_pink_noise(len(y), rng=rng)
        mixed_noise = 0.5 * white_n + 0.5 * pink_n
        mixed_noise = mixed_noise / (np.std(mixed_noise) + 1e-8)
        y = (y + mixed_noise * np.sqrt(noise_power)).astype(np.float32)

    # Dynamic Gain Scaling + Soft-Knee Limiter
    gain = float(rng.uniform(0.78, 1.03))
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = (y / peak) * gain
    if np.max(np.abs(y)) > 0.95:
        y = (0.95 * np.tanh(y / 0.95)).astype(np.float32)

    if len(y) > max_samples:
        y = y[:max_samples]

    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _colored_noise(num_samples: int, kind: str = "pink") -> torch.Tensor:
    """White / pink (1/f) / brownian (1/f^2) noise normalized to unit std."""
    white = torch.randn(num_samples)
    if kind == "white" or num_samples < 32:
        return white / (white.std() + 1e-8)
    spec = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(num_samples)
    freqs[0] = 1.0
    power = 0.5 if kind == "pink" else 1.0  # amplitude ~ f^-0.5 (pink) or f^-1 (brown)
    noise = torch.fft.irfft(spec / freqs.pow(power), n=num_samples)
    return noise / (noise.std() + 1e-8)


def make_room_tone(num_samples: int, level_dbfs: float = -52.0, jitter_db: float = 4.0, fade_sec: float = 0.010) -> torch.Tensor:
    """Low-level pinkish room tone (about -52 dBFS) with cosine fades, used instead of digital zeros."""
    if num_samples <= 0:
        return torch.zeros(0)
    tone = _colored_noise(num_samples, "pink") * (10.0 ** ((level_dbfs + random.uniform(-jitter_db, jitter_db)) / 20.0))
    fade = min(int(fade_sec * SR), num_samples // 2)
    if fade > 1:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, math.pi, fade))
        tone[:fade] = tone[:fade] * ramp
        tone[-fade:] = tone[-fade:] * ramp.flip(0)
    return tone


def stitch_with_room_tone(wav_a: torch.Tensor, wav_b: torch.Tensor, pause_sec: float) -> torch.Tensor:
    """Concatenates two clips with a natural room-tone pause and short cosine crossfades (no digital silence)."""
    pause = make_room_tone(int(pause_sec * SR))
    return torch.cat([wav_a, pause, wav_b], dim=0)


def augment_waveform_torch(wav, sr=SR, min_ctc_frames: Optional[int] = None) -> torch.Tensor:
    """Torch wrapper around Part B `augment_waveform` returning a 1D float32 Tensor."""
    aug_np = augment_waveform(wav, sample_rate=sr, min_ctc_frames=min_ctc_frames)
    return torch.from_numpy(aug_np).float()


def augment_waveform_fast(wav, sr=SR):
    """
    Fast on-the-fly CPU waveform perturbation for DataLoader workers (train only):
      1. 13 GCD-aligned speed factors (0.825x - 1.225x)
      2. Intra-utterance dynamic volume modulation
      3. Mosque / studio RIR reverb (24-RIR bank)
      4. White & pink (1/f) ambient noise at SNR 20-35 dB
      5. Consumer-microphone high-frequency roll-off (biquad low-pass)
      6. Boundary breath padding filled with room tone
      7. Random gain + soft limiter
    """
    if random.random() < 0.55:
        wav = random.choice(FAST_SPEED_RESAMPLERS)(wav)

    if random.random() < 0.40 and wav.numel() > sr // 2:
        t_env = torch.linspace(0.0, 1.0, wav.numel(), dtype=wav.dtype)
        cycles = random.uniform(1.0, 3.5)
        phase = random.uniform(0.0, 2.0 * math.pi)
        depth = random.uniform(0.15, 0.35)
        wav = wav * (1.0 + depth * torch.sin(2.0 * math.pi * cycles * t_env + phase))

    if random.random() < 0.35:
        ir = random.choice(MOSQUE_RIR_BANK)
        wet = random.uniform(0.08, 0.24)
        reverb = AF.fftconvolve(wav, ir)[: wav.numel()]
        wav = (1.0 - wet) * wav + wet * reverb

    if random.random() < 0.45 and wav.numel() > 32:
        noise = 0.5 * _colored_noise(wav.numel(), "white") + 0.5 * _colored_noise(wav.numel(), "pink")
        noise = noise / (noise.std() + 1e-8)
        snr = torch.tensor([random.uniform(20.0, 35.0)])
        wav = AF.add_noise(wav.unsqueeze(0), noise.unsqueeze(0), snr).squeeze(0)

    if random.random() < 0.20 and wav.numel() > 64:
        cutoff = random.uniform(4200.0, 7200.0)
        mix = random.uniform(0.5, 1.0)
        wav = (1.0 - mix) * wav + mix * AF.lowpass_biquad(wav, sr, cutoff_freq=cutoff)

    if random.random() < 0.35:
        front = int(random.uniform(0.0, 0.15) * sr)
        back = int(random.uniform(0.0, 0.15) * sr)
        wav = torch.cat([make_room_tone(front), wav, make_room_tone(back)], dim=0)

    if random.random() < 0.45:
        wav = T.Vol(gain=random.uniform(-4.0, 2.5), gain_type="db")(wav)
    peak = wav.abs().max()
    if peak > 0.98:
        wav = torch.tanh(wav / 0.95) * 0.95
    return wav[:MAX_AUG_SAMPLES]


# =============================================================================
# 5. Physics-Ordered Spectral Augmentation & Tajweed-Safe SpecAugment
# =============================================================================

def normalize_log_mel_cmvn(mel_bins_x_time: torch.Tensor) -> torch.Tensor:
    """Per-Mel-Bin Cepstral Mean Normalization + Global Utterance Std Scaling."""
    if mel_bins_x_time.size(-1) > 1:
        centered = mel_bins_x_time - mel_bins_x_time.mean(dim=-1, keepdim=True)
    else:
        centered = mel_bins_x_time - mel_bins_x_time.mean()
    return centered / (centered.std() + 1e-6)


def apply_vtlp_mel(
    db_mel_bins_x_time: torch.Tensor,
    alpha_min: float = 0.92,
    alpha_max: float = 1.08,
) -> torch.Tensor:
    """
    Piecewise-Linear Vocal Tract Length Perturbation (VTLP) in RAW dB Log-Mel space
    (MUST be called BEFORE Per-Mel-Bin CMVN so cross-bin interpolation operates on true dB energies).
    """
    squeeze_out = False
    if db_mel_bins_x_time.ndim == 2:
        db_mel_bins_x_time = db_mel_bins_x_time.unsqueeze(0)
        squeeze_out = True

    num_bins = db_mel_bins_x_time.size(1)
    alpha = random.uniform(alpha_min, alpha_max)
    if abs(alpha - 1.0) < 1e-3 or num_bins < 8:
        return db_mel_bins_x_time.squeeze(0) if squeeze_out else db_mel_bins_x_time

    device_m = db_mel_bins_x_time.device
    dtype_m = db_mel_bins_x_time.dtype
    bins = torch.arange(num_bins, device=device_m, dtype=torch.float32)
    top_bin = float(num_bins - 1)
    boundary = 0.85 * top_bin
    knot = boundary / max(1.0, alpha)

    low_warp = alpha * bins
    high_slope = (top_bin - alpha * knot) / max(1e-5, top_bin - knot)
    high_warp = top_bin - high_slope * (top_bin - bins)
    src_bins = torch.where(bins <= knot, low_warp, high_warp).clamp(0.0, top_bin)

    idx_floor = src_bins.floor().long().clamp(0, num_bins - 1)
    idx_ceil = (idx_floor + 1).clamp(0, num_bins - 1)
    weight = (src_bins - idx_floor.float()).to(dtype_m).view(1, num_bins, 1)

    warped = (1.0 - weight) * db_mel_bins_x_time[:, idx_floor, :] + weight * db_mel_bins_x_time[:, idx_ceil, :]
    return warped.squeeze(0) if squeeze_out else warped


def apply_mic_eq_tilt_db(
    db_mel_bins_x_time: torch.Tensor,
    max_tilt_db: float = 4.0,
    max_band_ripple_db: float = 2.0,
) -> torch.Tensor:
    """
    Physically accurate additive dB microphone frequency-response tilt + multi-band ripple + gentle
    vocal-proximity envelope drift in RAW dB space (applied BEFORE Per-Mel-Bin CMVN).
    """
    squeeze_out = False
    if db_mel_bins_x_time.ndim == 2:
        db_mel_bins_x_time = db_mel_bins_x_time.unsqueeze(0)
        squeeze_out = True

    _, num_bins, num_frames = db_mel_bins_x_time.shape
    dev = db_mel_bins_x_time.device
    dt = db_mel_bins_x_time.dtype

    b_axis = torch.linspace(-1.0, 1.0, num_bins, device=dev, dtype=dt)
    tilt_db = random.uniform(-max_tilt_db, max_tilt_db) * b_axis
    phase = random.uniform(0.0, 2.0 * math.pi)
    cycles = random.uniform(0.5, 1.75)
    ripple_db = random.uniform(-max_band_ripple_db, max_band_ripple_db) * torch.cos(
        math.pi * cycles * (b_axis + 1.0) + phase
    )
    out = db_mel_bins_x_time + (tilt_db + ripple_db).view(1, num_bins, 1)

    if num_frames > 25 and random.random() < 0.55:
        t_phase = random.uniform(0.0, 2.0 * math.pi)
        t_axis = torch.linspace(0.0, 1.0, num_frames, device=dev, dtype=dt)
        prox_db = random.uniform(-2.0, 2.0) * torch.sin(math.pi * t_axis + t_phase)
        out = out + prox_db.view(1, 1, num_frames)

    return out.squeeze(0) if squeeze_out else out


def apply_local_time_warp_mel(
    mel_bins_x_time: torch.Tensor,
    warp_param: int = 5,
) -> torch.Tensor:
    """
    Optional SpecAugment Component 3 — Gentle Local Time Warping:
    Piecewise-linearly warps the temporal axis around a random interior anchor frame `w0` by `+/- dw` frames
    (`dw <= warp_param`) while keeping total frame count `T` fixed. Disabled by default so delicate Tajweed
    elongation timings (Madd 2/4/6 harakat) remain undisturbed unless explicitly enabled.
    """
    squeeze_out = False
    if mel_bins_x_time.ndim == 2:
        mel_bins_x_time = mel_bins_x_time.unsqueeze(0)
        squeeze_out = True

    _, _, num_frames = mel_bins_x_time.shape
    if warp_param <= 0 or num_frames < max(24, 4 * warp_param):
        return mel_bins_x_time.squeeze(0) if squeeze_out else mel_bins_x_time

    w0 = random.randint(2 * warp_param, num_frames - 2 * warp_param - 1)
    dw = random.randint(-warp_param, warp_param)
    if dw == 0:
        return mel_bins_x_time.squeeze(0) if squeeze_out else mel_bins_x_time

    dev = mel_bins_x_time.device
    dt = mel_bins_x_time.dtype
    t = torch.arange(num_frames, device=dev, dtype=torch.float32)
    w1 = float(w0 + dw)
    left_src = t * (float(w0) / max(1.0, w1))
    right_src = float(w0) + (t - w1) * (float(num_frames - 1 - w0) / max(1.0, float(num_frames - 1) - w1))
    src_t = torch.where(t <= w1, left_src, right_src).clamp(0.0, float(num_frames - 1))

    idx_floor = src_t.floor().long().clamp(0, num_frames - 1)
    idx_ceil = (idx_floor + 1).clamp(0, num_frames - 1)
    weight = (src_t - idx_floor.float()).to(dt).view(1, 1, num_frames)
    warped = (1.0 - weight) * mel_bins_x_time[:, :, idx_floor] + weight * mel_bins_x_time[:, :, idx_ceil]
    return warped.squeeze(0) if squeeze_out else warped


def apply_adaptive_specaugment(
    cmvn_mel_bins_x_time: torch.Tensor,
    max_time_ratio: float = 0.045,
    num_freq_masks: int = 2,
    freq_mask_param: int = 8,
    max_time_masks: int = 4,
    micro_frame_drop_prob: float = 0.01,
    enable_time_warp: bool = False,
    time_warp_param: int = 5,
) -> torch.Tensor:
    """
    Tajweed-Safe SpecAugment applied AFTER Per-Mel-Bin CMVN (`0.0` mask fill = exact per-bin mean):
      1. Frequency Masking: masks `num_freq_masks` small random frequency bands (`<= freq_mask_param` bins).
      2. Time Masking: masks small random temporal segments (`<= max_time_ratio` of clip, capped at 40 frames
         so short Arabic diacritics, Qalqalah, and Hams are never erased).
      3. Optional Local Time Warping (`enable_time_warp=False` by default, gentle `time_warp_param=5`).
    """
    squeeze_out = False
    if cmvn_mel_bins_x_time.ndim == 2:
        cmvn_mel_bins_x_time = cmvn_mel_bins_x_time.unsqueeze(0)
        squeeze_out = True

    out = cmvn_mel_bins_x_time.clone()
    if enable_time_warp and random.random() < 0.35:
        out = apply_local_time_warp_mel(out, warp_param=time_warp_param)

    _, num_bins, num_frames = out.shape

    # 1. Frequency Masking (gentle: 1..freq_mask_param Mel bins)
    for _ in range(num_freq_masks):
        f_w = random.randint(1, min(max(1, freq_mask_param), max(1, num_bins - 1)))
        f_0 = random.randint(0, max(0, num_bins - f_w))
        out[:, f_0 : f_0 + f_w, :] = 0.0

    # 2. Time Masking (gentle: capped so delicate Tajweed durations are preserved)
    if num_frames > 30 and max_time_ratio > 0.0:
        n_t_masks = min(max_time_masks, max(2, num_frames // 260))
        max_w = max(4, min(40, int(num_frames * max_time_ratio)))
        for _ in range(n_t_masks):
            t_w = random.randint(1, min(max_w, num_frames - 1))
            t_0 = random.randint(0, num_frames - t_w)
            out[:, :, t_0 : t_0 + t_w] = 0.0

    if num_frames > 40 and micro_frame_drop_prob > 0.0 and random.random() < 0.4:
        drop_mask = torch.rand((1, 1, num_frames), device=out.device) < micro_frame_drop_prob
        out = out.masked_fill(drop_mask, 0.0)

    return out.squeeze(0) if squeeze_out else out


def augment_db_mel_correctly(
    raw_db_mel_bins_x_time: torch.Tensor,
    vtlp_prob: float = 0.65,
    vtlp_range: Tuple[float, float] = (0.92, 1.08),
    eq_prob: float = 0.50,
    specaug: bool = True,
    max_time_ratio: float = 0.045,
    num_freq_masks: int = 2,
    freq_mask_param: int = 8,
    enable_time_warp: bool = False,
) -> torch.Tensor:
    """
    Executes the physics-consistent spectral augmentation order:
      1. VTLP formant warp in raw dB space
      2. Additive microphone EQ tilt / proximity drift in raw dB space
      3. Per-Mel-Bin CMVN (producing exact 0.0 mean per bin)
      4. 0.0-filled Tajweed-Safe SpecAugment (Time Masking + Frequency Masking + optional Time Warping)
    Returns `(Time, N_MELS)` contiguous float32 tensor ready for `QuranConformerASR`.
    """
    mel_db = raw_db_mel_bins_x_time.float().clone()
    if vtlp_prob > 0.0 and random.random() < vtlp_prob:
        mel_db = apply_vtlp_mel(mel_db, alpha_min=vtlp_range[0], alpha_max=vtlp_range[1])
    if eq_prob > 0.0 and random.random() < eq_prob:
        mel_db = apply_mic_eq_tilt_db(mel_db)
    mel_cmvn = normalize_log_mel_cmvn(mel_db)
    if specaug:
        mel_cmvn = apply_adaptive_specaugment(
            mel_cmvn,
            max_time_ratio=max_time_ratio,
            num_freq_masks=num_freq_masks,
            freq_mask_param=freq_mask_param,
            enable_time_warp=enable_time_warp,
        )
    return mel_cmvn.transpose(0, 1).contiguous()


def dataloader_worker_init_fn(worker_id: int) -> None:
    """
    Ensures every PyTorch DataLoader worker process receives an independent deterministic seed
    for Python's `random` and `numpy.random` (preventing correlated/duplicate augmentations across workers).
    """
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed((worker_seed + int(worker_id) * 104729) % (2**32))


def extract_raw_db_mel_from_waveform(waveform) -> torch.Tensor:
    """Raw (pre-CMVN) Slaney Log-Mel in dB, shape (N_MELS, T)."""
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(np.asarray(waveform)).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return amplitude_to_db_transform(mel_spectrogram_transform(waveform)).squeeze(0)


def extract_log_mel_from_waveform(
    waveform,
    augment: bool = False,
    vtlp: bool = False,
    vtlp_range: tuple = (0.92, 1.08),
    band_contrast: bool = False,
    max_time_ratio: float = 0.045,
    num_freq_masks: int = 2,
    freq_mask_param: int = 8,
    enable_time_warp: bool = False,
):
    """
    Correct order: raw dB Mel -> (VTLP) -> (additive Mic-EQ tilt) -> per-bin CMVN -> (Tajweed-Safe SpecAugment).
    Returns (Time, N_MELS).
    """
    db = extract_raw_db_mel_from_waveform(waveform)
    if vtlp or augment:
        db = apply_vtlp_mel(db, alpha_min=vtlp_range[0], alpha_max=vtlp_range[1])
    if band_contrast or augment:
        db = apply_mic_eq_tilt_db(db)
    cmvn = normalize_log_mel_cmvn(db)
    if augment and cmvn.size(-1) > 30:
        cmvn = apply_adaptive_specaugment(
            cmvn,
            max_time_ratio=max_time_ratio,
            num_freq_masks=num_freq_masks,
            freq_mask_param=freq_mask_param,
            enable_time_warp=enable_time_warp,
        )
    return cmvn.transpose(0, 1).contiguous()


def extract_log_mel(audio_array, sr=SR, n_mels=N_MELS, augment=False):
    """Full pipeline: Part A preprocess -> (optional Part B waveform aug) -> dB Mel -> CMVN -> (SpecAugment). Returns (Time, N_MELS)."""
    waveform = preprocess_waveform(audio_array, sr)
    if augment:
        waveform = augment_waveform_torch(waveform, SR)
    return extract_log_mel_from_waveform(waveform, augment=augment)

# %% [markdown]
# ## Step 6 — Model Architecture (`Audio -> Log-Mel -> CNN -> Conformer L1-L2 [InterCTC + Self-Conditioning] -> Conformer L3-L4 -> Transformer Decoder -> Arabic Text`)
# ```
# Audio
#   ↓
# Log-Mel
#   ↓
# CNN (2 Layers, 20ms/frame)
#   ↓
# Conformer Layer 1
#   ↓
# Conformer Layer 2
#   ├── Intermediate CTC
#   └── Self-Conditioning
#   ↓
# Conformer Layer 3
#   ↓
# Conformer Layer 4
#   ↓
# Transformer Decoder (4 Layers)
#   ↓
# Arabic Text
# ```

# %%
try:
    from torchaudio.models import Conformer as TorchAudioConformer
except ImportError:
    TorchAudioConformer = None

# =============================================================================
# 1. Architecture Configuration
# =============================================================================

VALID_CONFORMER_NORM_MODES = (
    "masked_group_norm",
    "unmasked_group_norm",
    "frame_layer_norm",
)


@dataclass
class ConformerASRConfig:
    """
    Configuration for `QuranConformerASR`:
      Audio -> Log-Mel -> CNN (2 Layers) -> Conformer L1-L2 -> [Intermediate CTC + Self-Conditioning]
            -> Conformer L3-L4 -> Transformer Decoder (4 Layers) -> Arabic Text.
    """
    conformer_norm_mode: str = "masked_group_norm"  # "masked_group_norm" (default), "unmasked_group_norm", "frame_layer_norm"
    conformer_output_mode: str = "gated"            # "gated" = conformer_norm(x + conformer_gate * conf_out) | "direct"
    decoder_context_window: Optional[int] = 14      # 14 (local causal), 32, or 0 (full causal)
    use_specaugment: bool = True                    # Tajweed-safe Time Masking + Frequency Masking
    specaug_enable_time_warp: bool = False          # Optional local Time Warping (off by default to protect Tajweed)
    inter_ctc_layer_idx: int = 2                    # 1-based Conformer layer index for Intermediate CTC + Self-Conditioning (Layer 2 of 4)
    cnn_channels: Optional[int] = None              # Optional first-layer CNN channel width in [64, 256] (None = d_model)

    def validate(self) -> "ConformerASRConfig":
        if self.conformer_norm_mode not in VALID_CONFORMER_NORM_MODES:
            raise ValueError(
                f"conformer_norm_mode must be one of {VALID_CONFORMER_NORM_MODES}, got {self.conformer_norm_mode!r}"
            )
        if self.conformer_output_mode not in ("gated", "direct"):
            raise ValueError(
                f"conformer_output_mode must be 'gated' or 'direct', got {self.conformer_output_mode!r}"
            )
        if self.decoder_context_window is not None and self.decoder_context_window < 0:
            raise ValueError(
                f"decoder_context_window must be >= 0 (0 = full causal), got {self.decoder_context_window}"
            )
        if self.inter_ctc_layer_idx < 1:
            raise ValueError(f"inter_ctc_layer_idx must be >= 1, got {self.inter_ctc_layer_idx}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# 2. Pure-PyTorch `torchaudio.models.Conformer`-Compatible Fallback & Masked Norm
# =============================================================================

class _FallbackFeedForwardModule(nn.Module):
    """Matches `torchaudio.models.conformer._FeedForwardModule` exact `state_dict` layout."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class _FallbackConvolutionModule(nn.Module):
    """Matches `torchaudio.models.conformer._ConvolutionModule` exact `state_dict` layout."""
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_group_norm: bool = True,
    ):
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = nn.LayerNorm(input_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(input_dim, 2 * num_channels, 1, stride=1, padding=0, bias=bias),
            nn.GLU(dim=1),
            nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else nn.BatchNorm1d(num_channels),
            nn.SiLU(),
            nn.Conv1d(num_channels, input_dim, 1, stride=1, padding=0, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D) — intentionally unmasked here to match legacy torchaudio behavior
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(0, 1).transpose(1, 2)  # (B, D, T)
        x = self.sequential(x)
        x = x.transpose(1, 2).transpose(0, 1)  # (T, B, D)
        return x + residual


class _FallbackConformerLayer(nn.Module):
    """Matches `torchaudio.models.conformer.ConformerLayer` exact `state_dict` layout."""
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = True,
        convolution_first: bool = False,
    ):
        super().__init__()
        self.ffn1 = _FallbackFeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.self_attn = nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_module = _FallbackConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )
        self.ffn2 = _FallbackFeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_conv(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_module(x)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_conv(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_conv(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        x = self.final_layer_norm(x)
        return x


class _FallbackConformer(nn.Module):
    """1-to-1 drop-in fallback when `torchaudio` is unavailable locally."""
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = True,
        convolution_first: bool = False,
    ):
        super().__init__()
        self.conformer_layers = nn.ModuleList(
            [
                _FallbackConformerLayer(
                    input_dim=input_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_padding_mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = x.transpose(0, 1)  # (T, B, D)
        for layer in self.conformer_layers:
            h = layer(h, encoder_padding_mask)
        return h.transpose(0, 1), lengths


ConformerBackend = TorchAudioConformer if TorchAudioConformer is not None else _FallbackConformer


class FrameLayerNorm1d(nn.Module):
    """
    Per-frame LayerNorm over channel dimension `C` for `(B, C, T)` tensors (separate architectural option
    `conformer_norm_mode="frame_layer_norm"`). Keeps `.weight` and `.bias` of shape `(C,)`.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x_bct: torch.Tensor, valid_mask_bct: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_btc = x_bct.transpose(1, 2)
        out_btc = F.layer_norm(
            x_btc.float(),
            (self.num_channels,),
            weight=self.weight.float(),
            bias=self.bias.float(),
            eps=self.eps,
        ).to(x_bct.dtype)
        out_bct = out_btc.transpose(1, 2)
        if valid_mask_bct is not None:
            out_bct = out_bct.masked_fill(~valid_mask_bct, 0.0)
        return out_bct


def masked_group_norm_1d(
    x_bct: torch.Tensor,
    valid_lengths: torch.Tensor,
    valid_mask_bct: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float = 1e-5,
    num_groups: int = 1,
) -> torch.Tensor:
    """
    Computes exact `GroupNorm(num_groups, C)` statistics strictly over valid frames `t < valid_lengths[b]`
    in FP32 precision, eliminating Conformer batch/padding sensitivity while keeping 100% identical
    `weight` and `bias` parameter tensors to `nn.GroupNorm(1, C)`.
    """
    B, C, T = x_bct.shape
    G = max(1, int(num_groups))
    channels_per_group = C // G

    x_clean = x_bct.masked_fill(~valid_mask_bct, 0.0)
    x_fp32 = x_clean.float().view(B, G, channels_per_group, T)
    valid_g = valid_mask_bct.unsqueeze(1)  # (B, 1, 1, T)

    counts = (valid_lengths.clamp(min=1).to(torch.float32) * float(channels_per_group)).view(B, 1, 1, 1)
    mean = x_fp32.sum(dim=(2, 3), keepdim=True) / counts
    centered = (x_fp32 - mean).masked_fill(~valid_g, 0.0)
    var = centered.pow(2).sum(dim=(2, 3), keepdim=True) / counts

    normed = (centered / torch.sqrt(var + float(eps))).view(B, C, T)
    if weight is not None:
        normed = normed * weight.float().view(1, C, 1)
    if bias is not None:
        normed = normed + bias.float().view(1, C, 1)
    return normed.to(x_bct.dtype).masked_fill(~valid_mask_bct, 0.0)


def forward_conformer_with_masking(
    conformer_mod: nn.Module,
    x_btd: torch.Tensor,
    lengths: torch.Tensor,
    norm_mode: str = "masked_group_norm",
    inter_ctc_layer_idx: Optional[int] = None,
    inter_ctc_norm: Optional[nn.Module] = None,
    inter_ctc_head: Optional[nn.Module] = None,
    inter_ctc_cond: Optional[nn.Module] = None,
    inter_ctc_gate: Optional[torch.Tensor] = None,
    return_inter_ctc: bool = False,
):
    """
    Executes Conformer layers with strict valid-frame masking and Mid-Conformer Intermediate CTC +
    Self-Conditioning (after Layer 2 by default):
      - `"unmasked_group_norm"`: unmasked GroupNorm behavior.
      - `"masked_group_norm"`: masks padded frames before/after every sub-block, right after `GLU`
        (before `Depthwise Conv1d(kernel=15, padding=7)`), and computes `GroupNorm(1, D)` strictly over `t < lengths[b]`.
      - `"frame_layer_norm"`: strict conv masking + per-frame `LayerNorm(D)` across channels.
      - After Layer `inter_ctc_layer_idx` (Layer 2 of 4), computes `inter_ctc_logits` via `inter_ctc_head`
        and applies gated Self-Conditioning `inter_ctc_cond(softmax(inter_ctc_logits))` before Layers 3 and 4.
    """
    B, T, D = x_btd.shape
    clamped_lens = torch.clamp(lengths.long(), min=1, max=max(1, T))
    key_padding_mask = torch.arange(T, device=x_btd.device).unsqueeze(0) >= clamped_lens.unsqueeze(1)
    inter_ctc_logits: Optional[torch.Tensor] = None

    valid_mask_bct = (~key_padding_mask).unsqueeze(1)             # (B, 1, T)
    pad_mask_tbd = key_padding_mask.transpose(0, 1).unsqueeze(-1)  # (T, B, 1)

    x = x_btd.transpose(0, 1).masked_fill(pad_mask_tbd, 0.0)      # (T, B, D)

    def _run_masked_conv_module(conv_mod: nn.Module, h_tbd: torch.Tensor) -> torch.Tensor:
        if norm_mode == "unmasked_group_norm":
            return conv_mod(h_tbd)
        res_c = h_tbd
        c_in = conv_mod.layer_norm(h_tbd).masked_fill(pad_mask_tbd, 0.0)
        h = c_in.transpose(0, 1).transpose(1, 2)  # (B, D, T)
        seq = conv_mod.sequential
        h = seq[0](h)  # Pointwise Conv1d (D -> 2D)
        h = seq[1](h)  # GLU (2D -> D)
        # Zero padded frames BEFORE Depthwise Conv1d(kernel=15, padding=7)
        h = h.masked_fill(~valid_mask_bct, 0.0)
        h = seq[2](h)  # Depthwise Conv1d
        h = h.masked_fill(~valid_mask_bct, 0.0)

        norm_layer = seq[3]
        if norm_mode == "masked_group_norm" and isinstance(norm_layer, nn.GroupNorm):
            h = masked_group_norm_1d(
                h,
                clamped_lens,
                valid_mask_bct,
                weight=norm_layer.weight,
                bias=norm_layer.bias,
                eps=norm_layer.eps,
                num_groups=norm_layer.num_groups,
            )
        elif isinstance(norm_layer, FrameLayerNorm1d):
            h = norm_layer(h, valid_mask_bct)
        else:
            w_ln = norm_layer.weight.float() if getattr(norm_layer, "weight", None) is not None else None
            b_ln = norm_layer.bias.float() if getattr(norm_layer, "bias", None) is not None else None
            eps_ln = float(getattr(norm_layer, "eps", 1e-5))
            h_btc = F.layer_norm(h.transpose(1, 2).float(), (D,), weight=w_ln, bias=b_ln, eps=eps_ln).to(h.dtype)
            h = h_btc.transpose(1, 2).masked_fill(~valid_mask_bct, 0.0)

        h = seq[4](h)  # SiLU
        h = h.masked_fill(~valid_mask_bct, 0.0)
        h = seq[5](h)  # Pointwise Conv1d (D -> D)
        h = seq[6](h)  # Dropout
        h = h.masked_fill(~valid_mask_bct, 0.0)

        c_out = h.transpose(1, 2).transpose(0, 1)  # (T, B, D)
        return (res_c + c_out).masked_fill(pad_mask_tbd, 0.0)

    for layer_idx, layer in enumerate(conformer_mod.conformer_layers):
        # 1. Macaron FFN 1
        residual = x
        f1 = layer.ffn1(x).masked_fill(pad_mask_tbd, 0.0)
        x = (residual + 0.5 * f1).masked_fill(pad_mask_tbd, 0.0)

        conv_first = bool(getattr(layer, "convolution_first", False))
        if conv_first:
            x = _run_masked_conv_module(layer.conv_module, x)

        # 2. Multi-Head Self-Attention
        residual = x
        x_ln = layer.self_attn_layer_norm(x).masked_fill(pad_mask_tbd, 0.0)
        if getattr(layer.self_attn, "batch_first", False):
            x_b = x_ln.transpose(0, 1)
            attn_out, _ = layer.self_attn(
                query=x_b, key=x_b, value=x_b, key_padding_mask=key_padding_mask, need_weights=False
            )
            attn_out = attn_out.transpose(0, 1)
        else:
            attn_out, _ = layer.self_attn(
                query=x_ln, key=x_ln, value=x_ln, key_padding_mask=key_padding_mask, need_weights=False
            )
        attn_out = layer.self_attn_dropout(attn_out).masked_fill(pad_mask_tbd, 0.0)
        x = (residual + attn_out).masked_fill(pad_mask_tbd, 0.0)

        # 3. Masked Convolution Module
        if not conv_first:
            x = _run_masked_conv_module(layer.conv_module, x)

        # 4. Macaron FFN 2 + Final LayerNorm
        residual = x
        f2 = layer.ffn2(x).masked_fill(pad_mask_tbd, 0.0)
        x = (residual + 0.5 * f2).masked_fill(pad_mask_tbd, 0.0)
        x = layer.final_layer_norm(x).masked_fill(pad_mask_tbd, 0.0)

        # Mid-Conformer Intermediate CTC + Self-Conditioning after Conformer Layer 2
        if (
            inter_ctc_head is not None
            and inter_ctc_layer_idx is not None
            and (layer_idx + 1) == int(inter_ctc_layer_idx)
        ):
            h_mid_btd = x.transpose(0, 1)  # (B, T, D)
            if inter_ctc_norm is not None:
                h_mid_btd = inter_ctc_norm(h_mid_btd)
            h_mid_btd = h_mid_btd.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            inter_ctc_logits = inter_ctc_head(h_mid_btd)
            if inter_ctc_cond is not None:
                inter_probs = F.softmax(inter_ctc_logits.float(), dim=-1).to(x.dtype)
                cond_btd = inter_ctc_cond(inter_probs).masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
                gate_val = torch.tanh(inter_ctc_gate) if isinstance(inter_ctc_gate, torch.Tensor) else 0.10
                x = (x + gate_val * cond_btd.transpose(0, 1)).masked_fill(pad_mask_tbd, 0.0)

    out_btd = x.transpose(0, 1)
    if return_inter_ctc:
        return out_btd, clamped_lens, inter_ctc_logits
    return out_btd, clamped_lens


# =============================================================================
# 3. `QuranConformerASR`: CNN -> Conformer L1-L2 [InterCTC + Self-Conditioning] -> L3-L4 -> Transformer Decoder
# =============================================================================

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Conformer Encoder & Transformer Decoder."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class QuranConformerASR(nn.Module):
    """
    Quranic Speech Recognition Architecture:
      Audio -> Log-Mel -> 2-Layer Masked CNN (20ms/frame) -> Conformer Layer 1 -> Conformer Layer 2
            -> [Intermediate CTC + Self-Conditioning] -> Conformer Layer 3 -> Conformer Layer 4
            -> 4-Layer Transformer Decoder -> Arabic Text.
    """
    def __init__(
        self,
        n_mels: int = 80,
        vocab_size: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        conformer_layers: int = 4,
        conformer_ffn_dim: int = 512,
        conformer_kernel_size: int = 15,
        dropout: float = 0.15,
        decoder_dropout: float = 0.25,
        decoder_context_window: int = 14,
        pad_id: int = 0,
        sos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        cnn_channels: Optional[int] = None,
        conformer_norm_mode: Optional[str] = None,
        asr_config: Optional[ConformerASRConfig] = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pad_id = int(pad_id)
        self.conformer_layers_count = max(2, int(conformer_layers))
        raw_nhead = max(1, int(round(nhead)))
        if self.d_model % raw_nhead != 0:
            valid_heads = [h for h in (4, 8, 2, 16, 1) if self.d_model % h == 0]
            raw_nhead = min(valid_heads, key=lambda h: abs(h - raw_nhead))
        self.nhead = int(raw_nhead)

        cfg = copy.deepcopy(asr_config) if asr_config is not None else ConformerASRConfig()
        if conformer_norm_mode is not None:
            cfg.conformer_norm_mode = conformer_norm_mode
        if cnn_channels is not None:
            cfg.cnn_channels = int(cnn_channels)
        self.asr_config = cfg.validate()

        c1_channels = int(self.asr_config.cnn_channels) if self.asr_config.cnn_channels is not None else self.d_model
        c1_channels = max(32, c1_channels)
        self.cnn_channels = c1_channels

        effective_window = (
            self.asr_config.decoder_context_window
            if self.asr_config.decoder_context_window is not None
            else decoder_context_window
        )
        self.decoder_context_window = int(effective_window)
        self.conformer_norm_mode = self.asr_config.conformer_norm_mode
        self.inter_ctc_layer_idx = min(
            max(1, int(self.asr_config.inter_ctc_layer_idx)),
            max(1, self.conformer_layers_count - 1),
        )
        self._last_inter_ctc_logits: Optional[torch.Tensor] = None

        # 1. 2-Layer Masked CNN Frontend (2x temporal downsampling = 20ms/frame)
        self.conv1 = nn.Conv1d(n_mels, self.cnn_channels, kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.LayerNorm(self.cnn_channels)
        self.conv2 = nn.Conv1d(self.cnn_channels, d_model, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(d_model)
        self.cnn_dropout = nn.Dropout(dropout)

        # 2. 4-Layer Conformer Encoder
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)
        self.conformer = ConformerBackend(
            input_dim=d_model,
            num_heads=self.nhead,
            ffn_dim=conformer_ffn_dim,
            num_layers=self.conformer_layers_count,
            depthwise_conv_kernel_size=conformer_kernel_size,
            dropout=dropout,
            use_group_norm=True,
            convolution_first=False,
        )
        if self.conformer_norm_mode == "frame_layer_norm":
            for layer in self.conformer.conformer_layers:
                layer.conv_module.sequential[3] = FrameLayerNorm1d(d_model, eps=1e-5)

        if self.asr_config.conformer_output_mode == "gated":
            self.conformer_gate: Optional[nn.Parameter] = nn.Parameter(torch.tensor(0.35))
            self.conformer_norm: Optional[nn.LayerNorm] = nn.LayerNorm(d_model)
        else:
            self.conformer_gate = None
            self.conformer_norm = None

        # 3. Mid-Conformer Intermediate CTC + Self-Conditioning (after Conformer Layer 2)
        self.inter_ctc_norm = nn.LayerNorm(d_model)
        self.inter_ctc_head = nn.Linear(d_model, vocab_size)
        self.inter_ctc_cond = nn.Linear(vocab_size, d_model, bias=False)
        self.inter_ctc_gate = nn.Parameter(torch.tensor(0.1))
        with torch.no_grad():
            self.inter_ctc_head.bias.zero_()
            self.inter_ctc_head.bias[self.pad_id] = 2.0
            if sos_id is not None:
                self.inter_ctc_head.bias[int(sos_id)] = -0.5
            if eos_id is not None:
                self.inter_ctc_head.bias[int(eos_id)] = -0.5

        # 4. 4-Layer Transformer Decoder (Primary Recognition Head -> Arabic Text)
        self.decoder_layers_count = 4
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.token_embedding.weight[pad_id].zero_()

        self.pos_decoder = PositionalEncoding(d_model, max_len=1024, dropout=decoder_dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=self.nhead,
            dim_feedforward=dim_feedforward,
            dropout=decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.decoder_layers_count,
            norm=nn.LayerNorm(d_model),
        )

        # 5. Weight-Tied Transformer Decoder Output Head (Arabic Text Tokens)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.output_head.weight = self.token_embedding.weight
        self.last_diagnostics: Dict[str, Any] = {}

    def set_conformer_norm_mode(self, mode: str) -> None:
        """Switches Conformer normalization mode (`masked_group_norm` vs `unmasked_group_norm` vs `frame_layer_norm`) in-place."""
        if mode not in VALID_CONFORMER_NORM_MODES:
            raise ValueError(f"Invalid mode {mode!r}; choose from {VALID_CONFORMER_NORM_MODES}")
        self.conformer_norm_mode = mode
        self.asr_config.conformer_norm_mode = mode

    def describe_active_pipeline(self) -> str:
        return (
            f"Audio -> Log-Mel -> CNN (2L, C={self.cnn_channels}->{self.d_model}, 20ms) -> "
            f"Conformer L1 -> Conformer L2 [Intermediate CTC + Self-Conditioning] -> "
            f"Conformer L3 -> Conformer L4 ({self.conformer_norm_mode}) -> "
            f"Transformer Decoder ({self.decoder_layers_count}L, win={self.decoder_context_window}) -> Arabic Text"
        )

    def _compute_cnn_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(input_lengths + 2 * 1 - 3, 2, rounding_mode="floor") + 1
        lengths = torch.div(lengths + 2 * 1 - 3, 1, rounding_mode="floor") + 1
        return torch.clamp(lengths, min=1)

    def _build_local_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        row = idx.unsqueeze(1)
        col = idx.unsqueeze(0)
        if self.decoder_context_window and self.decoder_context_window > 0:
            return (col > row) | ((row - col) >= self.decoder_context_window)
        return col > row

    def encode_audio(
        self,
        mel_features: torch.Tensor,
        mel_lengths: torch.Tensor,
        return_inter_ctc: bool = False,
    ):
        # Mask padded Mel frames before Conv1
        input_mask = (
            torch.arange(mel_features.size(1), device=mel_features.device)[None, :]
            >= mel_lengths[:, None]
        )
        x = mel_features.masked_fill(input_mask.unsqueeze(-1), 0.0).transpose(1, 2)

        # CNN Layer 1 (stride=2) + LayerNorm + GELU + strict post-Conv1 padding mask
        x = self.conv1(x).transpose(1, 2)
        x = F.gelu(self.ln1(x))
        len1 = torch.clamp((mel_lengths.long() + 1) // 2, min=1, max=x.size(1))
        mask1 = torch.arange(x.size(1), device=x.device)[None, :] >= len1[:, None]
        x = x.masked_fill(mask1.unsqueeze(-1), 0.0)

        # CNN Layer 2 (stride=1) + LayerNorm + GELU + Dropout + strict post-Conv2 padding mask
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(self.ln2(x))
        x = self.cnn_dropout(x)

        max_enc_len = x.size(1)
        enc_lengths = torch.clamp(self._compute_cnn_lengths(mel_lengths), min=1, max=max_enc_len)
        src_key_padding_mask = (
            torch.arange(max_enc_len, device=x.device)[None, :] >= enc_lengths[:, None]
        )
        x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        # Sinusoidal Positional Encoding + 4-Layer Masked Conformer (with Layer-2 InterCTC + Self-Conditioning)
        x = self.pos_encoder(x).masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        max_valid_len = int(enc_lengths.max().item())
        conf_out, _, inter_ctc_logits = forward_conformer_with_masking(
            self.conformer,
            x[:, :max_valid_len, :],
            enc_lengths,
            norm_mode=self.conformer_norm_mode,
            inter_ctc_layer_idx=self.inter_ctc_layer_idx,
            inter_ctc_norm=self.inter_ctc_norm,
            inter_ctc_head=self.inter_ctc_head,
            inter_ctc_cond=self.inter_ctc_cond,
            inter_ctc_gate=self.inter_ctc_gate,
            return_inter_ctc=True,
        )
        if max_valid_len < max_enc_len:
            conf_out = F.pad(conf_out, (0, 0, 0, max_enc_len - max_valid_len))
            if inter_ctc_logits is not None:
                inter_ctc_logits = F.pad(inter_ctc_logits, (0, 0, 0, max_enc_len - max_valid_len))
        conf_out = conf_out.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        if (
            self.asr_config.conformer_output_mode == "direct"
            or self.conformer_gate is None
            or self.conformer_norm is None
        ):
            memory = conf_out
        else:
            memory = self.conformer_norm(x + self.conformer_gate * conf_out)
        memory = memory.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        self._last_inter_ctc_logits = inter_ctc_logits
        self.last_diagnostics = {
            "conformer_norm_mode": self.conformer_norm_mode,
            "inter_ctc_layer_idx": self.inter_ctc_layer_idx,
            "peak_gpu_memory_mb": (
                round(torch.cuda.max_memory_allocated(mel_features.device) / (1024.0 ** 2), 2)
                if (torch.cuda.is_available() and mel_features.is_cuda)
                else 0.0
            ),
        }
        if return_inter_ctc:
            return memory, src_key_padding_mask, inter_ctc_logits
        return memory, src_key_padding_mask

    def forward(
        self,
        mel_features: torch.Tensor,
        mel_lengths: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        return_inter_ctc: bool = False,
    ):
        memory, memory_key_padding_mask, inter_ctc_logits = self.encode_audio(
            mel_features, mel_lengths, return_inter_ctc=True
        )

        tgt_seq_len = tgt_input_ids.size(1)
        tgt_causal_mask = self._build_local_causal_mask(tgt_seq_len, tgt_input_ids.device)

        tgt_emb = self.token_embedding(tgt_input_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)

        # NOTE: no tgt_key_padding_mask on purpose: the causal mask already hides trailing pads from real
        # tokens, and masking pad keys inside a local window would create fully-masked rows (NaN risk).
        decoder_out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.output_head(decoder_out)
        if return_inter_ctc:
            enc_lengths = torch.clamp(self._compute_cnn_lengths(mel_lengths), min=1, max=memory.size(1))
            return logits, inter_ctc_logits, enc_lengths
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        mel_features: torch.Tensor,
        mel_lengths: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 350,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Pure Autoregressive Transformer Decoder Greedy Decoding."""
        self.eval()
        with torch.amp.autocast("cuda", enabled=(torch.cuda.is_available() and mel_features.is_cuda)):
            memory, memory_key_padding_mask = self.encode_audio(mel_features, mel_lengths)
            batch_size = mel_features.size(0)
            enc_lens = torch.clamp(self._compute_cnn_lengths(mel_lengths), min=1, max=memory.size(1))

            max_dynamic = min(max_len, int(enc_lens.max().item() * 0.45) + 20)
            generated = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=mel_features.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=mel_features.device)

            for step in range(max_dynamic):
                tgt_seq_len = generated.size(1)
                tgt_causal_mask = self._build_local_causal_mask(tgt_seq_len, generated.device)
                tgt_emb = self.pos_decoder(self.token_embedding(generated) * math.sqrt(self.d_model))

                dec_out = self.transformer_decoder(
                    tgt=tgt_emb,
                    memory=memory,
                    tgt_mask=tgt_causal_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                logits = self.output_head(dec_out[:, -1, :]).clone()
                logits[:, self.pad_id] = -float("inf")
                logits[:, sos_id] = -float("inf")
                if step == 0:
                    logits[:, eos_id] = -float("inf")

                if repetition_penalty != 1.0 and repetition_penalty > 0.0 and generated.size(1) > 1:
                    for b in range(batch_size):
                        if finished[b]:
                            continue
                        seen_tokens = {
                            t for t in generated[b, 1:].tolist() if t not in (self.pad_id, sos_id, eos_id)
                        }
                        for tok_id in seen_tokens:
                            val = logits[b, tok_id]
                            logits[b, tok_id] = val * repetition_penalty if val < 0 else val / repetition_penalty

                if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:
                    for b in range(batch_size):
                        if finished[b]:
                            continue
                        gen_list = generated[b].tolist()
                        cur_prefix = tuple(gen_list[-(no_repeat_ngram_size - 1) :])
                        search_start = max(1, len(gen_list) - 40)
                        for i in range(search_start, len(gen_list) - no_repeat_ngram_size + 1):
                            if tuple(gen_list[i : i + no_repeat_ngram_size - 1]) == cur_prefix:
                                banned_tok = gen_list[i + no_repeat_ngram_size - 1]
                                if banned_tok not in (sos_id, eos_id, self.pad_id):
                                    logits[b, banned_tok] = -float("inf")

                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                for b in range(batch_size):
                    if finished[b]:
                        next_tokens[b, 0] = self.pad_id
                    else:
                        ind_limit = int(enc_lens[b].item() * 0.45) + 20
                        if step >= ind_limit:
                            next_tokens[b, 0] = eos_id
                            finished[b] = True

                generated = torch.cat([generated, next_tokens], dim=1)
                finished = finished | (next_tokens.squeeze(1) == eos_id)
                if finished.all():
                    break

            final_sequences = []
            for b in range(batch_size):
                raw_seq = generated[b].tolist()
                if eos_id in raw_seq[1:]:
                    eos_pos = raw_seq.index(eos_id, 1)
                    seq = raw_seq[: eos_pos + 1]
                else:
                    seq = [t for t in raw_seq if t != self.pad_id] + [eos_id]
                final_sequences.append(torch.tensor(seq, device=mel_features.device, dtype=torch.long))

            return nn.utils.rnn.pad_sequence(final_sequences, batch_first=True, padding_value=self.pad_id)

    def _attention_beam_single_memory(
        self,
        clip_memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_steps: int,
        beam_size: int = 5,
        length_norm_alpha: float = 0.6,
        eos_bonus: float = 0.0,
        no_repeat_ngram_size: int = 0,
    ) -> List[Tuple[List[int], float, float]]:
        """
        Runs batched autoregressive beam search on one clip's valid encoder memory `(1, valid_T, D)`.
        Returns a sorted list of `(content_tokens_without_sos_eos, normalized_score, raw_log_prob)`.
        """
        dev = clip_memory.device
        B_beam = max(1, int(beam_size))

        def _gnmt_denom(seq_len_with_eos: int) -> float:
            if length_norm_alpha <= 0.0:
                return 1.0
            return ((5.0 + float(seq_len_with_eos)) / 6.0) ** float(length_norm_alpha)

        active_seqs: List[List[int]] = [[sos_id]]
        active_scores: List[float] = [0.0]
        finished: List[Tuple[float, float, List[int]]] = []  # (norm_score, raw_lp, content_tokens)

        for step in range(max_steps):
            num_active = len(active_seqs)
            tgt_tensor = torch.tensor(active_seqs, dtype=torch.long, device=dev)  # (num_active, step+1)
            mem_exp = clip_memory.expand(num_active, -1, -1)
            causal_mask = self._build_local_causal_mask(tgt_tensor.size(1), dev)

            tgt_emb = self.pos_decoder(self.token_embedding(tgt_tensor) * math.sqrt(self.d_model))
            dec_out = self.transformer_decoder(tgt=tgt_emb, memory=mem_exp, tgt_mask=causal_mask)
            step_logits = self.output_head(dec_out[:, -1, :]).float()
            log_probs = F.log_softmax(step_logits, dim=-1)  # (num_active, V)

            log_probs[:, self.pad_id] = -float("inf")
            log_probs[:, sos_id] = -float("inf")

            if no_repeat_ngram_size > 0 and tgt_tensor.size(1) >= no_repeat_ngram_size:
                for i, seq in enumerate(active_seqs):
                    cur_prefix = tuple(seq[-(no_repeat_ngram_size - 1) :])
                    search_start = max(1, len(seq) - 40)
                    for pos in range(search_start, len(seq) - no_repeat_ngram_size + 1):
                        if tuple(seq[pos : pos + no_repeat_ngram_size - 1]) == cur_prefix:
                            banned = seq[pos + no_repeat_ngram_size - 1]
                            if banned not in (self.pad_id, sos_id, eos_id):
                                log_probs[i, banned] = -float("inf")

            # 1. Evaluate completing each active hypothesis with <eos> (when non-empty)
            if step >= 1:
                for i in range(num_active):
                    eos_lp = float(log_probs[i, eos_id].item())
                    if math.isfinite(eos_lp):
                        raw_total = active_scores[i] + eos_lp + float(eos_bonus)
                        norm_sc = raw_total / _gnmt_denom(step + 1)
                        finished.append((norm_sc, active_scores[i] + eos_lp, active_seqs[i][1:]))

            # If at final step, force complete any remaining active beams
            if step == max_steps - 1:
                for i in range(num_active):
                    eos_lp = float(log_probs[i, eos_id].item())
                    add_lp = eos_lp if math.isfinite(eos_lp) else 0.0
                    raw_total = active_scores[i] + add_lp + float(eos_bonus)
                    norm_sc = raw_total / _gnmt_denom(step + 1)
                    finished.append((norm_sc, active_scores[i] + add_lp, active_seqs[i][1:]))
                break

            # 2. Expand non-<eos> continuations across all active beams simultaneously
            cont_lp = log_probs.clone()
            cont_lp[:, eos_id] = -float("inf")
            base_scores = torch.tensor(active_scores, dtype=torch.float32, device=dev).unsqueeze(1)
            cand_scores = (base_scores + cont_lp).view(-1)
            V = cont_lp.size(1)
            k_pick = min(B_beam, int((torch.isfinite(cand_scores)).sum().item()))
            if k_pick <= 0:
                break

            top_vals, top_idx = torch.topk(cand_scores, k=k_pick)
            next_seqs: List[List[int]] = []
            next_scores: List[float] = []
            for sc_val, flat_i in zip(top_vals.tolist(), top_idx.tolist()):
                beam_i = flat_i // V
                tok_i = flat_i % V
                next_seqs.append(active_seqs[beam_i] + [tok_i])
                next_scores.append(float(sc_val))

            active_seqs = next_seqs
            active_scores = next_scores

            # Safe early-stopping check once enough complete hypotheses beat best possible continuation
            if len(finished) >= B_beam:
                finished.sort(key=lambda item: item[0], reverse=True)
                finished = finished[: max(B_beam * 2, 8)]
                kth_best_done = finished[B_beam - 1][0]
                best_optimistic = (active_scores[0] + max(0.0, float(eos_bonus))) / _gnmt_denom(step + 2)
                if best_optimistic < kth_best_done:
                    break

        # Deduplicate content sequences keeping highest normalized score
        dedup: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        for norm_sc, raw_lp, content in finished:
            key = tuple(content)
            if key not in dedup or norm_sc > dedup[key][0]:
                dedup[key] = (norm_sc, raw_lp)

        ranked = [(list(k), v[0], v[1]) for k, v in dedup.items()]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    @torch.no_grad()
    def attention_beam_decode(
        self,
        mel_features: torch.Tensor,
        mel_lengths: torch.Tensor,
        sos_id: int,
        eos_id: int,
        beam_size: int = 5,
        length_norm_alpha: float = 0.6,
        eos_bonus: float = 0.0,
        max_len: int = 350,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """
        Standalone Attention Beam Search (completely independent of CTC):
          - `beam_size == 1` with `length_norm_alpha == 0.0` and `eos_bonus == 0.0` matches pure Greedy.
          - Supports `beam_size in {1, 3, 5, 8}`, GNMT length normalization `(5 + L)^alpha / 6^alpha`,
            and explicit `<eos>` transition scoring + `eos_bonus`.
        """
        if beam_size <= 1 and length_norm_alpha == 0.0 and eos_bonus == 0.0:
            return self.greedy_decode(
                mel_features,
                mel_lengths,
                sos_id=sos_id,
                eos_id=eos_id,
                max_len=max_len,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        self.eval()
        with torch.amp.autocast("cuda", enabled=(torch.cuda.is_available() and mel_features.is_cuda)):
            memory, _ = self.encode_audio(mel_features, mel_lengths)
            enc_lens = torch.clamp(self._compute_cnn_lengths(mel_lengths), min=1, max=memory.size(1))
            batch_size = mel_features.size(0)

            best_sequences = []
            for b in range(batch_size):
                valid_L = int(enc_lens[b].item())
                clip_mem = memory[b : b + 1, :valid_L, :]
                ind_limit = min(max_len, int(valid_L * 0.45) + 20)
                hyps = self._attention_beam_single_memory(
                    clip_mem,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    max_steps=max(2, ind_limit),
                    beam_size=beam_size,
                    length_norm_alpha=length_norm_alpha,
                    eos_bonus=eos_bonus,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
                best_content = hyps[0][0] if hyps else []
                best_sequences.append(
                    torch.tensor([sos_id] + best_content + [eos_id], dtype=torch.long, device=mel_features.device)
                )

            return nn.utils.rnn.pad_sequence(best_sequences, batch_first=True, padding_value=self.pad_id)


# =============================================================================
# 4. CTC Target Validation, Branch-Separated Checkpointing & Early Stopping
# =============================================================================

def prepare_1d_ctc_targets(
    padded_ctc_targets: torch.Tensor,
    ctc_lengths: torch.Tensor,
    blank_id: int = 0,
    sos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    slices = [
        padded_ctc_targets[b, : int(ctc_lengths[b].item())]
        for b in range(padded_ctc_targets.size(0))
    ]
    targets_1d = (
        torch.cat(slices, dim=0)
        if slices
        else torch.empty(0, dtype=torch.long, device=padded_ctc_targets.device)
    )
    if targets_1d.numel() > 0:
        if (targets_1d == blank_id).any():
            raise ValueError(f"Invalid CTC target: found blank_id={blank_id} inside valid target sequence!")
        if sos_id is not None and (targets_1d == sos_id).any():
            raise ValueError(f"Invalid CTC target: found sos_id={sos_id} inside valid target sequence!")
        if eos_id is not None and (targets_1d == eos_id).any():
            raise ValueError(f"Invalid CTC target: found eos_id={eos_id} inside valid target sequence!")
    return targets_1d


def get_ctc_minimum_lengths(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    required_lengths = []
    for b in range(targets.size(0)):
        length = int(target_lengths[b].item())
        sequence = targets[b, :length]
        repeated = int((sequence[1:] == sequence[:-1]).sum().item()) if length > 1 else 0
        required_lengths.append(length + repeated)
    return torch.tensor(required_lengths, dtype=torch.long, device=target_lengths.device)


def compute_acoustic_balanced_score(
    dec_wer: float,
    dec_cer: float,
    ctc_wer: float,
    ctc_cer: float,
) -> float:
    return 0.40 * float(ctc_wer) + 0.25 * float(ctc_cer) + 0.20 * float(dec_wer) + 0.15 * float(dec_cer)


class BranchCheckpointTracker:
    """
    Tracks and saves 4 independent branch-specific checkpoints:
      - `best_decoder_cer.pt` (Primary target: lowest Validation Decoder CER, tie-break Decoder WER)
      - `best_decoder_wer.pt` (Lowest Validation Decoder WER, tie-break Decoder CER)
      - `best_ctc_cer.pt`     (Lowest Validation Intermediate CTC CER at Conformer Layer 2, tie-break InterCTC WER)
      - `best_acoustic_balanced.pt` (Composite balanced score across Decoder & Intermediate CTC)
    Also manages Decoder-driven Early Stopping when Validation Decoder CER/WER stops improving.
    """
    def __init__(
        self,
        save_dir: str,
        min_early_stop_epoch: int = 65,
        patience_evals: int = 5,
    ):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.paths = {
            "decoder_cer": os.path.join(save_dir, "best_decoder_cer.pt"),
            "decoder_wer": os.path.join(save_dir, "best_decoder_wer.pt"),
            "ctc_cer": os.path.join(save_dir, "best_ctc_cer.pt"),
            "acoustic_balanced": os.path.join(save_dir, "best_acoustic_balanced.pt"),
        }
        self.best: Dict[str, Tuple[float, float]] = {
            k: (float("inf"), float("inf")) for k in self.paths
        }
        self.best_epochs: Dict[str, int] = {k: 0 for k in self.paths}
        self.min_early_stop_epoch = int(min_early_stop_epoch)
        self.patience_evals = int(patience_evals)
        self.evals_without_decoder_improvement = 0

    def update(
        self,
        epoch: int,
        raw_metrics: Dict[str, float],
        ema_metrics: Dict[str, float],
        raw_state: Dict[str, torch.Tensor],
        ema_state: Dict[str, torch.Tensor],
        extra_payload: Optional[Dict[str, Any]] = None,
        save_to_disk: bool = True,
    ) -> Dict[str, Any]:
        """
        Independently picks the better of (`Model` vs `EMA`) for EACH of the 4 criteria,
        updates best records, saves improved `.pt` files, and evaluates Decoder Early Stopping.
        """
        criteria = {
            "decoder_cer": lambda m: (float(m["dec_cer"]), float(m["dec_wer"])),
            "decoder_wer": lambda m: (float(m["dec_wer"]), float(m["dec_cer"])),
            "ctc_cer": lambda m: (float(m["ctc_cer"]), float(m["ctc_wer"])),
            "acoustic_balanced": lambda m: (
                compute_acoustic_balanced_score(m["dec_wer"], m["dec_cer"], m["ctc_wer"], m["ctc_cer"]),
                float(m["dec_cer"]),
            ),
        }

        saved_branches: Dict[str, Dict[str, Any]] = {}
        decoder_improved = False

        for branch_key, key_fn in criteria.items():
            raw_tuple = key_fn(raw_metrics)
            ema_tuple = key_fn(ema_metrics)
            if ema_tuple <= raw_tuple:
                cand_tuple = ema_tuple
                cand_src = "EMA"
                cand_metrics = ema_metrics
                cand_state = ema_state
            else:
                cand_tuple = raw_tuple
                cand_src = "Model"
                cand_metrics = raw_metrics
                cand_state = raw_state

            if cand_tuple < self.best[branch_key]:
                self.best[branch_key] = cand_tuple
                self.best_epochs[branch_key] = int(epoch)
                if branch_key in ("decoder_cer", "decoder_wer"):
                    decoder_improved = True

                payload = {
                    "epoch": int(epoch),
                    "branch_criterion": branch_key,
                    "best_weight_source": cand_src,
                    "best_eval_state_dict": copy.deepcopy(cand_state),
                    "best_val_cer": float(cand_metrics["dec_cer"]),
                    "best_val_wer": float(cand_metrics["dec_wer"]),
                    "best_ctc_cer": float(cand_metrics["ctc_cer"]),
                    "best_ctc_wer": float(cand_metrics["ctc_wer"]),
                    "best_val_score": compute_acoustic_balanced_score(
                        cand_metrics["dec_wer"],
                        cand_metrics["dec_cer"],
                        cand_metrics["ctc_wer"],
                        cand_metrics["ctc_cer"],
                    ),
                    **(extra_payload or {}),
                }
                if save_to_disk:
                    torch.save(payload, self.paths[branch_key])
                saved_branches[branch_key] = {
                    "source": cand_src,
                    "epoch": int(epoch),
                    "primary_metric": cand_tuple[0],
                    "path": self.paths[branch_key],
                }

        if decoder_improved:
            self.evals_without_decoder_improvement = 0
        elif epoch >= self.min_early_stop_epoch:
            self.evals_without_decoder_improvement += 1

        should_stop = (
            self.patience_evals > 0
            and epoch >= self.min_early_stop_epoch
            and self.evals_without_decoder_improvement >= self.patience_evals
        )
        return {
            "saved_branches": saved_branches,
            "decoder_improved": decoder_improved,
            "evals_without_decoder_improvement": self.evals_without_decoder_improvement,
            "should_early_stop": should_stop,
        }


def load_compatible_checkpoint(
    model: nn.Module,
    ckpt_or_state_dict: Union[str, Dict[str, Any]],
    device: Union[str, torch.device] = "cpu",
    map_device: Optional[Union[str, torch.device]] = None,
    prefer_best_eval: bool = True,
) -> Dict[str, Any]:
    """
    Loads branch-separated checkpoints cleanly into `QuranConformerASR`, safely ignoring any obsolete legacy keys if present.
    """
    target_dev = map_device if map_device is not None else device
    raw_ckpt = (
        torch.load(ckpt_or_state_dict, map_location=target_dev)
        if isinstance(ckpt_or_state_dict, str)
        else ckpt_or_state_dict
    )
    if isinstance(raw_ckpt, dict) and ("best_eval_state_dict" in raw_ckpt or "model_state_dict" in raw_ckpt):
        src_state = (
            raw_ckpt.get("best_eval_state_dict", raw_ckpt.get("model_state_dict"))
            if prefer_best_eval
            else raw_ckpt.get("model_state_dict", raw_ckpt.get("best_eval_state_dict"))
        )
    else:
        src_state = raw_ckpt

    target_state = model.state_dict()
    adapted_state: Dict[str, torch.Tensor] = {}
    skipped_shape_keys: List[str] = []
    ignored_legacy_keys: List[str] = []

    for key, tensor in src_state.items():
        if key in target_state:
            if target_state[key].shape == tensor.shape:
                adapted_state[key] = tensor
            else:
                skipped_shape_keys.append(key)
        else:
            ignored_legacy_keys.append(key)

    missing, unexpected = model.load_state_dict(adapted_state, strict=False)
    return {
        "migrated_count": len(adapted_state),
        "total_target_keys": len(target_state),
        "exact_match": len(missing) == 0 and len(skipped_shape_keys) == 0,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected) + ignored_legacy_keys,
        "skipped_shape_mismatch_keys": skipped_shape_keys,
        "raw_checkpoint": raw_ckpt if isinstance(raw_ckpt, dict) else {},
    }

# ---------------------------------------------------------------------------
# Instantiate Single Clean Model Architecture
# ---------------------------------------------------------------------------
ACTIVE_ASR_CONFIG = ConformerASRConfig(
    conformer_norm_mode=CONFORMER_NORM_MODE,
    inter_ctc_layer_idx=2,
).validate()

model = QuranConformerASR(
    n_mels=N_MELS,
    vocab_size=len(tokenizer),
    d_model=256,
    nhead=8,
    dim_feedforward=1024,
    conformer_layers=4,
    conformer_ffn_dim=512,
    conformer_kernel_size=15,
    dropout=0.15,
    decoder_dropout=0.25,
    decoder_context_window=14,
    pad_id=tokenizer.pad_id,
    sos_id=tokenizer.sos_id,
    eos_id=tokenizer.eos_id,
    asr_config=ACTIVE_ASR_CONFIG,
).to(device)

for _n, _m in model.named_modules():
    assert "gat" not in _n.lower(), f"Unexpected GAT module: {_n}"
    assert not isinstance(_m, nn.LSTM) and "bilstm" not in _n.lower(), f"Unexpected BiLSTM module: {_n}"
    assert "ctc_head" not in _n or "inter_ctc_head" in _n, f"Unexpected Final CTC head: {_n}"

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(
    f"\nPipeline        : {model.describe_active_pipeline()}"
    f"\nFlags           : SpecAugment={model.asr_config.use_specaugment} | InterCTC@Layer{model.inter_ctc_layer_idx}=True"
    f"\nTotal Trainable Parameters: {total_params:,}"
)

# Forward-pass sanity check on dummy tensors
with torch.no_grad():
    _dummy_mel = torch.randn(2, 60, N_MELS, device=device)
    _dummy_mel_lens = torch.tensor([60, 44], dtype=torch.long, device=device)
    _dummy_tgt_in = torch.full((2, 12), tokenizer.sos_id, dtype=torch.long, device=device)
    _test_dec_logits, _test_inter_ctc_logits, _test_enc_lens = model(
        _dummy_mel, _dummy_mel_lens, _dummy_tgt_in, return_inter_ctc=True
    )
    assert _test_dec_logits.shape == (2, 12, len(tokenizer)), _test_dec_logits.shape
    assert _test_inter_ctc_logits.shape[:2] == (2, int(_test_enc_lens.max().item())), _test_inter_ctc_logits.shape
    print(
        f"Sanity check OK : decoder_logits={tuple(_test_dec_logits.shape)} | "
        f"inter_ctc_logits={tuple(_test_inter_ctc_logits.shape)}"
    )

# %% [markdown]
# ## Step 7 — PyTorch Dataset & DataLoader Collation (with Part A 10-Step Preprocessing, Part B 2x Augmented Copy & Tajweed-Safe SpecAugment)

# %%
class CachedQuranASRDataset(TorchDataset):
    """
    In-memory cached dataset for Quranic ASR combining:
      - Part A: 10-Step Deterministic Audio/Text Preprocessing (`preprocess_clean_waveform` + `preprocess_quran_text`)
      - Part B: 7-Step Multi-Stage Waveform Augmentation (`augment_waveform`) creating a 2nd copy of training data
        (2x dataset size when `augment=True` and `build_2x_train_copy=True`)
      - Tajweed-Safe SpecAugment: Time Masking + Frequency Masking (+ optional Local Time Warping) after per-bin CMVN
      - Same-Reciter short-ayah stitching with natural room-tone pauses and exact `T_min = L + R` CTC guard
    """
    def __init__(self, hf_split, tokenizer, augment=False, desc="Dataset", build_2x_train_copy: bool = True):
        self.tokenizer = tokenizer
        self.augment = augment
        self.build_2x_train_copy = bool(augment and build_2x_train_copy)
        self.space_id = self.tokenizer.char2id.get(" ", self.tokenizer.unk_id)
        self._epoch_tensor = torch.tensor([1], dtype=torch.long).share_memory_()
        self.entries = []
        self.lengths = []
        self.reciter_short_indices = {}
        self.cache_schema_version = f"clean_ram_v4_step10_partAB_2x_{int(self.build_2x_train_copy)}_max_aug_{MAX_AUG_SAMPLES}"
        max_clean_samples = 0
        clean_at_cap_count = 0
        n_clean_cached = 0
        n_aug_copy_cached = 0

        print(f"Pre-caching {len(hf_split)} samples into RAM ({desc} | schema={self.cache_schema_version})...")
        for idx in range(len(hf_split)):
            row = hf_split[idx]
            raw_arr = np.asarray(row["audio"]["array"], dtype=np.float32)
            raw_text = preprocess_quran_text(row["ayah_ar"])
            token_ids = torch.tensor(self.tokenizer.encode(raw_text), dtype=torch.long)

            # Exact minimum encoder frames required by CTC (T_min = L + R) to guard Part A & Part B
            ctc_seq = token_ids[1:-1]
            repeats = int((ctc_seq[1:] == ctc_seq[:-1]).sum().item()) if ctc_seq.numel() > 1 else 0
            t_min = int(ctc_seq.numel()) + repeats

            # Part A: 10-Step Deterministic Audio Preprocessing
            clean_wav = preprocess_waveform(raw_arr, SR, min_ctc_frames=t_min)
            max_clean_samples = max(max_clean_samples, int(clean_wav.numel()))
            if int(clean_wav.numel()) >= MAX_SAMPLES:
                clean_at_cap_count += 1
            clean_db = extract_raw_db_mel_from_waveform(clean_wav)            # (N_MELS, T) pre-CMVN dB
            clean_mel = normalize_log_mel_cmvn(clean_db).transpose(0, 1).contiguous()  # (T, N_MELS)

            rec_id = row.get(reciter_col, row.get("reciter_id", row.get("reciter_name", "")))
            ayah_id = row.get("ayah_id", -1)

            entry_idx = len(self.entries)
            self.entries.append({
                "clean_wav": clean_wav,
                "clean_mel": clean_mel,
                "clean_db_mel": clean_db.to(torch.float16) if augment else None,
                "token_ids": token_ids,
                "raw_text": raw_text,
                "ayah_id": ayah_id,
                "reciter": rec_id,
                "is_augmented_copy": False,
            })
            self.lengths.append(clean_mel.size(0))
            n_clean_cached += 1

            if augment and clean_wav.numel() <= int(10.5 * SR) and token_ids.numel() <= 210:
                self.reciter_short_indices.setdefault(rec_id, []).append(entry_idx)

        # Part B: Build 2nd Copy of Training Data with 7-Step Multi-Stage Waveform Augmentation (2x Dataset Size)
        if self.build_2x_train_copy:
            print(f"Building Part B 7-step multi-stage waveform-augmented 2nd copy ({n_clean_cached} -> {2 * n_clean_cached} samples)...")
            rng_part_b = np.random.default_rng(SEED + 101)
            for base_idx in range(n_clean_cached):
                base_entry = self.entries[base_idx]
                ctc_seq = base_entry["token_ids"][1:-1]
                repeats = int((ctc_seq[1:] == ctc_seq[:-1]).sum().item()) if ctc_seq.numel() > 1 else 0
                t_min = int(ctc_seq.numel()) + repeats

                aug_np = augment_waveform(
                    base_entry["clean_wav"].numpy(),
                    SR,
                    rng=rng_part_b,
                    min_ctc_frames=t_min,
                )
                aug_wav = torch.from_numpy(aug_np).float()
                aug_db = extract_raw_db_mel_from_waveform(aug_wav)
                aug_mel = normalize_log_mel_cmvn(aug_db).transpose(0, 1).contiguous()

                self.entries.append({
                    "clean_wav": aug_wav,
                    "clean_mel": aug_mel,
                    "clean_db_mel": aug_db.to(torch.float16),
                    "token_ids": base_entry["token_ids"],
                    "raw_text": base_entry["raw_text"],
                    "ayah_id": base_entry["ayah_id"],
                    "reciter": base_entry["reciter"],
                    "is_augmented_copy": True,
                })
                self.lengths.append(aug_mel.size(0))
                n_aug_copy_cached += 1

        print(
            f"  -> cached {len(self.entries)} total samples ({n_clean_cached} Part-A clean + {n_aug_copy_cached} Part-B 2x aug copy | "
            f"max clean dur: {max_clean_samples / SR:.2f}s / {MAX_SAMPLES / SR:.1f}s cap, "
            f"clean-cap hits: {clean_at_cap_count}, aug ceiling: {MAX_AUG_SAMPLES / SR:.1f}s)."
        )

    def set_epoch(self, epoch: int):
        """Updates the shared curriculum epoch counter across DataLoader worker processes."""
        self._epoch_tensor[0] = int(epoch)

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def curriculum_for_epoch(cur_epoch: int) -> Dict[str, Any]:
        """
        60-Epoch Curriculum with Tajweed-Safe SpecAugment:
          - Epochs 1-10  : 25% clean | 45% spectral-only (SpecAugment) | 30% waveform+SpecAug, no stitching
          - Epochs 11-35 : 12% clean | 43% spectral-only (SpecAugment) | 45% waveform+SpecAug, 18% stitching
          - Epochs 36-60 :  8% clean | 40% spectral-only (SpecAugment) | 52% waveform+SpecAug, 25% stitching
        """
        if cur_epoch <= 10:
            return dict(stitch=0.0, p_clean=0.25, p_spec=0.70, vtlp=(0.94, 1.06), vtlp_p=0.40, eq_p=0.0,
                        t_ratio=0.025, n_f=1, f_param=6, wave_spec=True, wave_vtlp_p=0.0, wave_eq_p=0.0)
        if cur_epoch <= 35:
            return dict(stitch=0.18, p_clean=0.12, p_spec=0.55, vtlp=(0.91, 1.09), vtlp_p=0.75, eq_p=0.55,
                        t_ratio=0.042, n_f=2, f_param=8, wave_spec=True, wave_vtlp_p=0.45, wave_eq_p=0.40)
        return dict(stitch=0.25, p_clean=0.08, p_spec=0.48, vtlp=(0.90, 1.10), vtlp_p=0.80, eq_p=0.60,
                    t_ratio=0.048, n_f=2, f_param=9, wave_spec=True, wave_vtlp_p=0.50, wave_eq_p=0.45)

    def __getitem__(self, idx):
        item = self.entries[idx]
        if not self.augment:
            return item["clean_mel"], item["token_ids"], item["raw_text"], item["ayah_id"], item["reciter"]

        cur_epoch = int(self._epoch_tensor[0].item())
        cfg = self.curriculum_for_epoch(cur_epoch)

        base_wav = item["clean_wav"]
        tokens = item["token_ids"]
        raw_text = item["raw_text"]
        stitched = False

        # Same-Reciter short-ayah stitching with natural room-tone pause + exact CTC T_min guard
        if (
            cfg["stitch"] > 0.0
            and base_wav.numel() <= int(11.5 * SR)
            and tokens.numel() <= 230
            and random.random() < cfg["stitch"]
        ):
            candidates = self.reciter_short_indices.get(item["reciter"], ())
            if len(candidates) > 1:
                partner_idx = random.choice(candidates)
                if partner_idx == idx:
                    partner_idx = random.choice(candidates)
                if partner_idx != idx:
                    partner = self.entries[partner_idx]
                    cand_wav = stitch_with_room_tone(
                        base_wav, partner["clean_wav"], random.uniform(0.15, 0.34)
                    )[:MAX_AUG_SAMPLES]
                    space_tok = torch.tensor([self.space_id], dtype=torch.long)
                    cand_tokens = torch.cat([tokens[:-1], space_tok, partner["token_ids"][1:]], dim=0)
                    cand_ctc = cand_tokens[1:-1]
                    cand_repeats = int((cand_ctc[1:] == cand_ctc[:-1]).sum().item()) if cand_ctc.numel() > 1 else 0
                    cand_t_min = int(cand_ctc.numel()) + cand_repeats
                    # Guarantee T >= L + R even at the fastest speed perturbation (1.225x)
                    if int((cand_wav.numel() // 320) / 1.26) >= (cand_t_min + 4):
                        base_wav = cand_wav
                        tokens = cand_tokens
                        raw_text = f"{preprocess_quran_text(raw_text)} {preprocess_quran_text(partner['raw_text'])}"
                        stitched = True

        p = random.random()
        if p < cfg["p_clean"]:
            mel = extract_log_mel_from_waveform(base_wav, augment=False) if stitched else item["clean_mel"]
            return mel, tokens, raw_text, item["ayah_id"], item["reciter"]

        if p < cfg["p_spec"]:
            db = (
                extract_raw_db_mel_from_waveform(base_wav)
                if stitched
                else item["clean_db_mel"].float()
            )
            mel = augment_db_mel_correctly(
                db,
                vtlp_prob=cfg["vtlp_p"],
                vtlp_range=cfg["vtlp"],
                eq_prob=cfg["eq_p"],
                specaug=True,
                max_time_ratio=cfg["t_ratio"],
                num_freq_masks=cfg["n_f"],
                freq_mask_param=cfg["f_param"],
            )
            return mel, tokens, raw_text, item["ayah_id"], item["reciter"]

        aug_wav = augment_waveform_fast(base_wav, SR)
        db = extract_raw_db_mel_from_waveform(aug_wav)
        mel = augment_db_mel_correctly(
            db,
            vtlp_prob=cfg["wave_vtlp_p"],
            vtlp_range=cfg["vtlp"],
            eq_prob=cfg["wave_eq_p"],
            specaug=cfg["wave_spec"],
            max_time_ratio=cfg["t_ratio"] * 0.85,
            num_freq_masks=max(1, cfg["n_f"] - 1),
            freq_mask_param=cfg["f_param"],
        )
        return mel, tokens, raw_text, item["ayah_id"], item["reciter"]


class LengthBucketBatchSampler:
    """
    Groups audio clips of similar duration into the same batch to minimize padding zeros.
    Reduces self-attention wasted computation by ~60%.
    """
    def __init__(self, lengths, batch_size=16, shuffle=True, drop_last=False):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Sort indices by sequence length
        indices = np.argsort(self.lengths)
        batches = []

        if self.shuffle:
            # Group into mega-buckets (e.g. 8 batches each), create batches within bucket, then shuffle batches
            bucket_size = self.batch_size * 6
            for i in range(0, len(indices), bucket_size):
                megabucket = indices[i : i + bucket_size].copy()
                np.random.shuffle(megabucket)
                for j in range(0, len(megabucket), self.batch_size):
                    batch = megabucket[j : j + self.batch_size].tolist()
                    if not self.drop_last or len(batch) == self.batch_size:
                        batches.append(batch)
            random.shuffle(batches)
        else:
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size].tolist()
                if not self.drop_last or len(batch) == self.batch_size:
                    batches.append(batch)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def collate_asr_batch(batch):
    mels, tokens, raw_texts, ayah_ids, reciters = zip(*batch)

    mel_lengths = torch.tensor([m.size(0) for m in mels], dtype=torch.long)
    padded_mels = nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=0.0)

    padded_tokens = nn.utils.rnn.pad_sequence(
        tokens, batch_first=True, padding_value=tokenizer.pad_id
    )
    # Teacher forcing inputs: <sos>, c_1, c_2, ...
    tgt_input = padded_tokens[:, :-1]
    # Target labels: c_1, c_2, ..., <eos>
    tgt_labels = padded_tokens[:, 1:]

    # Dedicated clean CTC targets: c_1, c_2, ... (excluding both <sos> and <eos>)
    ctc_targets_list = [t[1:-1] for t in tokens]
    ctc_lengths = torch.tensor([len(c) for c in ctc_targets_list], dtype=torch.long)
    padded_ctc_targets = nn.utils.rnn.pad_sequence(
        ctc_targets_list, batch_first=True, padding_value=tokenizer.pad_id
    )

    return (
        padded_mels,
        mel_lengths,
        tgt_input,
        tgt_labels,
        padded_ctc_targets,
        ctc_lengths,
        raw_texts,
        ayah_ids,
        reciters,
    )


class CleanDatasetEvalView(TorchDataset):
    """
    Zero-copy read-only deterministic clean view over `CachedQuranASRDataset.entries`.
    Filters strictly to Part-A clean entries (`is_augmented_copy == False`) so periodic
    `Train CER` & `Train WER` evaluation runs on the exact 1x clean training set.
    """
    def __init__(self, cached_dataset: CachedQuranASRDataset):
        self.entries = [e for e in cached_dataset.entries if not e.get("is_augmented_copy", False)]
        self.lengths = [int(e["clean_mel"].size(0)) for e in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        return item["clean_mel"], item["token_ids"], item["raw_text"], item["ayah_id"], item["reciter"]


def audit_dataset_ctc_alignments(cached_dataset: CachedQuranASRDataset, split_label: str, check_fast_aug: bool = False):
    """
    Pre-flight verification BEFORE relying on `zero_infinity=True`:
    Checks every cached sample against exact CTC minimum required frames `T_min = L + R`
    both on clean audio and (for Train) under worst-case `1.225x` fast speed perturbation.
    """
    clean_impossible = 0
    fast_impossible = 0
    max_repeats = 0
    min_clean_margin = 10**9
    min_fast_margin = 10**9

    for entry in cached_dataset.entries:
        ctc_seq = entry["token_ids"][1:-1].unsqueeze(0)
        ctc_len = torch.tensor([ctc_seq.size(1)], dtype=torch.long)
        t_min = int(get_ctc_minimum_lengths(ctc_seq, ctc_len)[0].item())
        repeats = t_min - int(ctc_seq.size(1))
        max_repeats = max(max_repeats, repeats)

        clean_mel_len = int(entry["clean_mel"].size(0))
        clean_enc_len = (clean_mel_len + 1) // 2
        min_clean_margin = min(min_clean_margin, clean_enc_len - t_min)
        if clean_enc_len < t_min:
            clean_impossible += 1

        if check_fast_aug:
            # Worst-case shortest augmented clip: fastest speed resampler (orig_freq=19600 -> 16000 = 1.225x speed)
            fast_wav_samples = max(1, int(entry["clean_wav"].numel() * (float(SR) / 19600.0)))
            fast_mel_len = (fast_wav_samples // 160) + 1
            fast_enc_len = (fast_mel_len + 1) // 2
            min_fast_margin = min(min_fast_margin, fast_enc_len - t_min)
            if fast_enc_len < t_min:
                fast_impossible += 1

    n_total = len(cached_dataset.entries)
    msg = (
        f"  [Pre-Flight CTC Alignment Audit | {split_label}] "
        f"Clean Impossible (T < L+R): {clean_impossible} / {n_total} "
        f"(min frame margin={min_clean_margin:+d}, max adjacent repeats R={max_repeats})"
    )
    if check_fast_aug:
        msg += f" | Worst-Case 1.225x Speed Impossible: {fast_impossible} / {n_total} (min margin={min_fast_margin:+d})"
    print(msg)
    return {"clean_impossible": clean_impossible, "fast_impossible": fast_impossible, "total": n_total}


BATCH_SIZE = 16

train_dataset = CachedQuranASRDataset(train_audio, tokenizer, augment=True, desc="Train 80%")
train_clean_eval_dataset = CleanDatasetEvalView(train_dataset)
val_dataset = CachedQuranASRDataset(eval_audio, tokenizer, augment=False, desc="Validation 10%")
test_dataset = CachedQuranASRDataset(test_audio, tokenizer, augment=False, desc="Test 10%")

train_sampler = LengthBucketBatchSampler(train_dataset.lengths, batch_size=BATCH_SIZE, shuffle=True)
train_eval_sampler = LengthBucketBatchSampler(train_clean_eval_dataset.lengths, batch_size=BATCH_SIZE, shuffle=False)
val_sampler = LengthBucketBatchSampler(val_dataset.lengths, batch_size=BATCH_SIZE, shuffle=False)
test_sampler = LengthBucketBatchSampler(test_dataset.lengths, batch_size=BATCH_SIZE, shuffle=False)

train_generator = torch.Generator()
train_generator.manual_seed(SEED)
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=collate_asr_batch,
    num_workers=2,
    pin_memory=True,
    worker_init_fn=dataloader_worker_init_fn,  # independent Python/NumPy RNG stream per worker
    generator=train_generator,
)
train_eval_loader = DataLoader(
    train_clean_eval_dataset,
    batch_sampler=train_eval_sampler,
    collate_fn=collate_asr_batch,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    collate_fn=collate_asr_batch,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    collate_fn=collate_asr_batch,
    num_workers=2,
    pin_memory=True,
)
print(
    f"DataLoaders ready: Train batches/epoch: {len(train_loader)} | "
    f"Train-Clean-Eval batches: {len(train_eval_loader)} | "
    f"Val batches: {len(val_loader)} | Test batches: {len(test_loader)}"
)
audit_dataset_ctc_alignments(train_dataset, "Train 80%", check_fast_aug=True)
audit_dataset_ctc_alignments(val_dataset, "Validation 10%", check_fast_aug=False)
audit_dataset_ctc_alignments(test_dataset, "Test 10%", check_fast_aug=False)


class _RNGProbeDataset(TorchDataset):
    """Returns per-item Python-`random` and NumPy random numbers to verify worker RNG independence."""
    def __len__(self):
        return 16

    def __getitem__(self, idx):
        return torch.tensor([random.random(), float(np.random.rand())], dtype=torch.float64)


def verify_training_augmentation_pipeline(dataset, loader, probe_epochs=(1, 20, 45), n_probe=12):
    """
    Pre-flight verification that augmented data really reaches training correctly:
      1. stochasticity: the same item yields different augmented features across calls,
      2. DataLoader workers draw independent random streams (worker_init_fn),
      3. real train_loader batch: finite values, padded frames == 0.0, T_enc >= L + R, clean-vs-aug ratio.
    """
    print("=" * 96)
    print("PRE-FLIGHT AUGMENTATION PIPELINE VERIFICATION")
    saved_epoch = int(dataset._epoch_tensor[0].item())

    # 1. Stochasticity + augmented ratio per curriculum stage
    for ep in probe_epochs:
        dataset.set_epoch(ep)
        n_diff, n_changed_len = 0, 0
        for i in range(min(n_probe, len(dataset))):
            a = dataset[i][0]
            b = dataset[i][0]
            same_shape = tuple(a.shape) == tuple(b.shape)
            if (not same_shape) or float((a - b).abs().mean()) > 1e-6:
                n_diff += 1
            if tuple(a.shape) != tuple(dataset.entries[i]["clean_mel"].shape):
                n_changed_len += 1
        cfg = dataset.curriculum_for_epoch(ep)
        print(
            f"  Epoch {ep:>3}: {n_diff}/{min(n_probe, len(dataset))} items differ between two draws | "
            f"{n_changed_len} changed duration | expected clean ratio {cfg['p_clean']:.2f}, stitch {cfg['stitch']:.2f}"
        )

    # 2. Worker RNG independence
    probe_loader = DataLoader(
        _RNGProbeDataset(), batch_size=2, shuffle=False, num_workers=2, worker_init_fn=dataloader_worker_init_fn
    )
    vals = torch.cat([b for b in probe_loader], dim=0)
    n_unique_py = len(set(vals[:, 0].tolist()))
    n_unique_np = len(set(vals[:, 1].tolist()))
    print(f"  Worker RNG probe: {n_unique_py}/{vals.size(0)} unique Python-random values, {n_unique_np}/{vals.size(0)} unique NumPy values")
    assert n_unique_py == vals.size(0) and n_unique_np == vals.size(0), "DataLoader workers share RNG streams!"

    # 3. Real training batch checks at a late-curriculum epoch
    dataset.set_epoch(45)
    padded_mels, mel_lengths, _, _, padded_ctc, ctc_lens, _, _, _ = next(iter(loader))
    assert torch.isfinite(padded_mels).all(), "NaN/Inf found in augmented training batch"
    for b in range(padded_mels.size(0)):
        L = int(mel_lengths[b])
        if L < padded_mels.size(1):
            assert float(padded_mels[b, L:].abs().max()) == 0.0, "padded frames are not exactly zero"
    enc_lens = (mel_lengths + 1) // 2
    t_min = get_ctc_minimum_lengths(padded_ctc, ctc_lens)
    n_bad = int((enc_lens < t_min).sum())
    print(
        f"  Train batch: mel {tuple(padded_mels.shape)} | mean|x|={float(padded_mels.abs().mean()):.3f} | "
        f"CTC-impossible (T < L+R): {n_bad}/{padded_mels.size(0)}"
    )
    assert n_bad == 0, "Augmented batch contains CTC-impossible samples"
    dataset.set_epoch(saved_epoch)
    print("  -> OK: augmentation is stochastic, worker-independent, padded & CTC-feasible.")
    print("=" * 96)


verify_training_augmentation_pipeline(train_dataset, train_loader)


# %% [markdown]
# ## Step 7.5 — Visualize Raw vs. Torchaudio Features (Waveform, Log-Mel, SpecAugment) & Dataset Statistics

# %%
import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display
from IPython.display import display, Audio as IPythonAudio, HTML

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Tahoma"]
matplotlib.rcParams["axes.unicode_minus"] = False

def shape_ar(text):
    """Reshapes and applies bidirectional layout for Matplotlib rendering."""
    if not text:
        return ""
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)

def raw_ar(text):
    """Returns raw Arabic text for HTML/native browser rendering."""
    return str(text) if text else ""

# Select a representative training example with sufficient length for visualization and listening
sample_vis_idx = min(5, len(train_audio) - 1)
for candidate_idx in range(min(30, len(train_audio))):
    if train_audio[candidate_idx].get("ayah_id", 1) >= 2:
        sample_vis_idx = candidate_idx
        break

sample_row = train_audio[sample_vis_idx]
raw_wav = np.asarray(sample_row["audio"]["array"], dtype=np.float32)
reciter_ar_label = get_arabic_reciter_name(sample_row)
ayah_num_vis = sample_row.get("ayah_id", "-")
ayah_text_vis = preprocess_quran_text(sample_row["ayah_ar"])

# 1. Deterministic Clean Preprocessing (torchaudio biquads, preemphasis, silence trim, fade, peak norm)
clean_tensor = preprocess_waveform(raw_wav, SR)
clean_wav = clean_tensor.cpu().numpy()

# 2. Individual torchaudio augmentations for listening & inspection:
# A. Speed Perturbation (AF.resample: 1.10x faster)
aug_speed_tensor = AF.resample(clean_tensor, orig_freq=int(SR * 1.10), new_freq=SR)
aug_speed = aug_speed_tensor.cpu().numpy()

# B. Pitch Shifting (AF.pitch_shift: +1.5 semitones)
aug_pitch_tensor = AF.pitch_shift(clean_tensor, SR, n_steps=1.5)
aug_pitch = aug_pitch_tensor.cpu().numpy()

# C. Simulated Mosque / Room Reverb (AF.fftconvolve with exponential decay impulse response)
ir_len = int(0.22 * SR)
t_ir = torch.linspace(0.0, 1.0, ir_len)
ir = torch.randn(ir_len) * torch.exp(-6.5 * t_ir)
ir[0] = 1.0
ir = ir / (ir.norm() + 1e-8)
wet_mix = 0.25
reverb_tensor = AF.fftconvolve(clean_tensor, ir)[: clean_tensor.numel()]
aug_reverb_tensor = (1.0 - wet_mix) * clean_tensor + wet_mix * reverb_tensor
aug_reverb = aug_reverb_tensor.cpu().numpy()

# D. Ambient Pink Noise Injection (AF.add_noise: 1/f noise at 22 dB SNR)
white_n = torch.randn(clean_tensor.numel())
spec_n = torch.fft.rfft(white_n)
freqs_n = torch.fft.rfftfreq(clean_tensor.numel())
freqs_n[0] = 1.0
k_pink_n = torch.fft.irfft(spec_n / freqs_n.sqrt(), n=clean_tensor.numel())
pink_n = k_pink_n / (k_pink_n.std() + 1e-8)
snr_tensor = torch.tensor([22.0])
aug_noise_tensor = AF.add_noise(clean_tensor.unsqueeze(0), pink_n.unsqueeze(0), snr_tensor).squeeze(0)
aug_noise = aug_noise_tensor.cpu().numpy()

# E. Combined Multi-Stage Stochastic Augmentation (Full pipeline)
torch.manual_seed(42)
aug_combined_tensor = augment_waveform_torch(clean_tensor.clone(), SR)
aug_combined = aug_combined_tensor.cpu().numpy()

# Spectrograms
wf = torch.from_numpy(raw_wav).float()
mel_raw = amplitude_to_db_transform(mel_spectrogram_transform(wf.unsqueeze(0))).squeeze(0)
mel_clean = extract_log_mel(raw_wav, augment=False).T          # (n_mels, T)
torch.manual_seed(0)
mel_aug = extract_log_mel(raw_wav, augment=True).T             # SpecAugment (time+freq masks)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
t_axis = np.linspace(0, len(raw_wav) / SR, len(raw_wav))
axes[0, 0].plot(t_axis, raw_wav, color="#64748b", linewidth=0.7)
axes[0, 0].set_title(f"Raw Waveform ({len(raw_wav)/SR:.2f}s)", fontweight="bold")
axes[0, 0].set_xlabel("Time (s)"); axes[0, 0].set_ylabel("Amplitude")

# Torchaudio spectrogram / linear-power STFT view
spec = T.Spectrogram(n_fft=400, hop_length=160, power=2.0)(wf.unsqueeze(0))
spec_db = amplitude_to_db_transform(spec).squeeze(0)
im = axes[0, 1].imshow(spec_db.numpy(), origin="lower", aspect="auto", cmap="magma")
axes[0, 1].set_title("Linear STFT Spectrogram (torchaudio, dB)", fontweight="bold")
axes[0, 1].set_xlabel("Frames"); axes[0, 1].set_ylabel("Freq bin")
fig.colorbar(im, ax=axes[0, 1])

# Mel filterbank (Slaney area-normalized 60-7600 Hz vocal/Tajweed band)
fb = AF.melscale_fbanks(
    n_freqs=201,
    f_min=MEL_FMIN,
    f_max=MEL_FMAX,
    n_mels=N_MELS,
    sample_rate=SR,
    norm="slaney",
    mel_scale="slaney",
)
axes[0, 2].plot(fb[:, ::10].numpy(), linewidth=1)
axes[0, 2].set_title("Slaney Mel Filterbank (60-7600 Hz, every 10th)", fontweight="bold")
axes[0, 2].set_xlabel("FFT bin"); axes[0, 2].set_ylabel("Weight")

for ax, mel, ttl in [
    (axes[1, 0], mel_raw, "Raw Slaney Log-Mel (dB)"),
    (axes[1, 1], mel_clean, "Clean Per-Mel-Bin CMVN Log-Mel (model input)"),
    (axes[1, 2], mel_aug, "With Fast VTLP + Band Contrast + Adaptive SpecAugment"),
]:
    im = ax.imshow(mel.numpy(), origin="lower", aspect="auto", cmap="magma")
    ax.set_title(ttl, fontweight="bold")
    ax.set_xlabel("Frames"); ax.set_ylabel("Mel bin")
    fig.colorbar(im, ax=ax)

# Build cleanly directed suptitle blocks to prevent RTL boundary breaking
title_str = f"Torchaudio Feature Pipeline: {shape_ar(SURAH_NAME_AR)} | Ayah {ayah_num_vis} | {shape_ar(reciter_ar_label)}"
fig.suptitle(title_str, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "torchaudio_features.png"), dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Interactive Audio Players: Listen to Raw vs. Preprocessed vs. Augmentations
# ---------------------------------------------------------------------------
display(HTML(f"""
<div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 18px; margin: 12px 0 16px 0;">
  <h3 style="color: #0f172a; margin: 0 0 8px 0; font-size: 18px;">
    🎧 Interactive Audio Player Comparison — {raw_ar(SURAH_NAME_AR)} (Ayah {ayah_num_vis})
  </h3>
  <p style="margin: 4px 0; font-size: 14px; color: #334155;">
    <b>Reciter:</b> {raw_ar(reciter_ar_label)} &nbsp;|&nbsp; <b>ID:</b> <code>{sample_row.get(reciter_col, '-')}</code>
  </p>
  <p style="margin: 6px 0 0 0; font-size: 16px; direction: rtl; text-align: right; color: #0284c7; font-weight: bold; font-family: 'Amiri', 'Traditional Arabic', serif;">
    {raw_ar(ayah_text_vis)}
  </p>
</div>
"""))

display(HTML(f"<b>1. Raw Original Recording</b> <span style='color:#64748b;'>({len(raw_wav)/SR:.2f}s)</span>:"))
display(IPythonAudio(raw_wav, rate=SR))

display(HTML(f"<b>2. Clean Preprocessed Audio</b> <span style='color:#2563eb;'>(torchaudio bandpass 65-7600Hz + pre-emphasis + Tajweed-safe silence trim + 15ms fade + RMS/soft-clip norm - {len(clean_wav)/SR:.2f}s)</span>:"))
display(IPythonAudio(clean_wav, rate=SR))

display(HTML(f"<b>3. Augmentation: Speed Perturbation</b> <span style='color:#d97706;'>(torchaudio AF.resample 1.10x faster - {len(aug_speed)/SR:.2f}s)</span>:"))
display(IPythonAudio(aug_speed, rate=SR))

display(HTML(f"<b>4. Augmentation: Pitch Shifted</b> <span style='color:#9333ea;'>(torchaudio AF.pitch_shift +1.5 semitones - {len(aug_pitch)/SR:.2f}s)</span>:"))
display(IPythonAudio(aug_pitch, rate=SR))

display(HTML(f"<b>5. Augmentation: Simulated Mosque / Room Reverb</b> <span style='color:#059669;'>(torchaudio AF.fftconvolve with exponential decay IR - {len(aug_reverb)/SR:.2f}s)</span>:"))
display(IPythonAudio(aug_reverb, rate=SR))

display(HTML(f"<b>6. Augmentation: Ambient Pink Noise Injected</b> <span style='color:#ea580c;'>(torchaudio AF.add_noise with 1/f pink noise at SNR 22 dB - {len(aug_noise)/SR:.2f}s)</span>:"))
display(IPythonAudio(aug_noise, rate=SR))

display(HTML(f"<b>7. Augmentation: Multi-Stage Combined Augmentation</b> <span style='color:#16a34a;'>(Speed + Pitch + Reverb + Noise + Temporal Shift + SpecAugment input - {len(aug_combined)/SR:.2f}s)</span>:"))
display(IPythonAudio(aug_combined, rate=SR))

# Dataset statistics: split sizes, durations, transcript lengths, vocab frequencies
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
split_sizes = [len(train_audio), len(eval_audio), len(test_audio)]
bars = axes[0].bar(["Train (80%)", "Val (10%)", "Test (10%)"], split_sizes,
                   color=["#3b82f6", "#f59e0b", "#10b981"], edgecolor="black")
for b, c in zip(bars, split_sizes):
    axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(), str(c), ha="center", va="bottom", fontweight="bold")
axes[0].set_title("Split Sizes (by Reciter)", fontweight="bold"); axes[0].set_ylabel("Clips")

durs = {}
for name, ds in [("Train", train_audio), ("Val", eval_audio), ("Test", test_audio)]:
    idxs = np.linspace(0, len(ds) - 1, min(100, len(ds)), dtype=int)
    durs[name] = [len(ds[int(i)]["audio"]["array"]) / SR for i in idxs]
    axes[1].hist(durs[name], bins=20, alpha=0.55, label=name, edgecolor="black")
axes[1].set_title("Audio Duration Distribution", fontweight="bold")
axes[1].set_xlabel("Seconds"); axes[1].legend()

verse_texts = {a: preprocess_quran_text(t) for a, t in zip(raw_dataset["ayah_id"], raw_dataset["ayah_ar"])}
axes[2].hist([len(t) for t in verse_texts.values()], bins=15, color="#8b5cf6", edgecolor="black")
axes[2].set_title("Verse Length (characters = decoder steps)", fontweight="bold")
axes[2].set_xlabel("Characters"); axes[2].set_ylabel("Verses")

from collections import Counter
char_freq = Counter("".join(verse_texts.values())).most_common(15)
axes[3].bar([shape_ar(c) if c.strip() else "space" for c, _ in char_freq], [n for _, n in char_freq],
            color="#0ea5e9", edgecolor="black")
axes[3].set_title("Top-15 Character Frequencies", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "dataset_statistics.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 8 — Training & Evaluation Loop for سورة محمد (60 Epochs)

# %%
EPOCHS = 60
LEARNING_RATE = 3.5e-4
WEIGHT_DECAY = 1.5e-2
GRAD_ACCUM_STEPS = 1
TOKEN_CORRUPTION_PROB = 0.20  # Corrupts 20% of Teacher-Forcing tgt_input tokens to prevent verse memorization

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_id,
    label_smoothing=0.10,
)
ctc_criterion = nn.CTCLoss(
    blank=tokenizer.pad_id,
    zero_infinity=True,
)
INTER_CTC_WEIGHT = 0.30  # 30% Mid-Conformer Layer-2 Intermediate CTC + 70% Transformer Decoder Cross-Entropy
CTC_WEIGHT = INTER_CTC_WEIGHT

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=WEIGHT_DECAY,
    foreach=True,
)

# One scheduler step per optimizer update
updates_per_epoch = (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    epochs=EPOCHS,
    steps_per_epoch=updates_per_epoch,
    pct_start=0.10,
    anneal_strategy="cos",
    cycle_momentum=False,
    div_factor=10.0,
    final_div_factor=100.0,
)
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available(),
)

import copy

class ModelEMA:
    """Exponential Moving Average (EMA) of model weights with step-warmup decay for superior validation stability."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.num_updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        cur_decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(cur_decay).add_(p.data, alpha=1.0 - cur_decay)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)

ema = ModelEMA(model, decay=0.997)

best_val_score = float("inf")
best_val_wer = float("inf")
best_val_cer = float("inf")

# Decoder-driven early stopping (Attention Decoder is the primary success criterion)
EARLY_STOPPING_PATIENCE_EVALS = 5   # evaluations (every 5 epochs) without Decoder CER/WER improvement
EARLY_STOP_MIN_EPOCH = 35
tracker = BranchCheckpointTracker(
    BRANCH_CKPT_DIR,
    min_early_stop_epoch=EARLY_STOP_MIN_EPOCH,
    patience_evals=EARLY_STOPPING_PATIENCE_EVALS,
)

# Training history used by the visualization & diagnostics suite
history = {
    "epoch": [], "train_loss": [], "train_ce_loss": [], "train_ctc_loss": [],
    "train_ctc_impossible": [], "train_total_samples": [],
    "lr": [], "grad_norm": [], "peak_gpu_mem_mb": [],
    "val_epoch": [], "val_loss": [], "val_wer": [], "val_cer": [], "val_cer_exact": [],
    "val_ctc_wer": [], "val_ctc_cer": [], "val_ctc_impossible": [],
    "train_eval_loss": [], "train_dec_wer": [], "train_dec_cer": [], "train_dec_cer_exact": [],
    "train_ctc_wer": [], "train_ctc_cer": [], "train_clean_ctc_impossible": [],
    "ctc_cer_gap": [], "dec_cer_gap": [],
    "val_ctc_char_ins": [], "val_ctc_char_del": [], "val_ctc_char_sub": [],
    "val_dec_char_ins": [], "val_dec_char_del": [], "val_dec_char_sub": [],
    "val_ctc_word_ins": [], "val_ctc_word_del": [],
    "val_dec_word_ins": [], "val_dec_word_del": [],
}
periodic_eval_rows = []

START_EPOCH = 1
RESUME_TRAINING = False          # True: resume optimizer/scheduler/EMA from CHECKPOINT_PATH
INIT_FROM_CHECKPOINT = None      # e.g. path of the Exp-A Epoch-105 checkpoint: loads weights only (exact_match expected)

if INIT_FROM_CHECKPOINT and os.path.isfile(INIT_FROM_CHECKPOINT):
    init_report = load_compatible_checkpoint(model, INIT_FROM_CHECKPOINT, map_device=device, prefer_best_eval=True)
    print(f"Initialized weights from {INIT_FROM_CHECKPOINT}: exact_match={init_report['exact_match']} "
          f"({init_report['migrated_count']}/{init_report['total_target_keys']} keys)")
    ema = ModelEMA(model, decay=0.997)

if RESUME_TRAINING and os.path.isfile(CHECKPOINT_PATH):
    mig_report = load_compatible_checkpoint(model, CHECKPOINT_PATH, map_device=device, prefer_best_eval=False)
    ckpt = mig_report["raw_checkpoint"]
    print(f"Resume report: {mig_report['migrated_count']}/{mig_report['total_target_keys']} keys (exact_match={mig_report['exact_match']})")
    if mig_report["exact_match"] and ckpt:
        if "ema_model_state_dict" in ckpt:
            ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
            ema.num_updates = ckpt.get("ema_updates", 0)
        for key, obj in (("optimizer_state_dict", optimizer), ("scheduler_state_dict", scheduler), ("scaler_state_dict", scaler)):
            if key in ckpt:
                obj.load_state_dict(ckpt[key])
        for hk, hv in ckpt.get("history", {}).items():
            if hk in history:
                history[hk] = hv
        START_EPOCH = ckpt.get("epoch", 0) + 1
        print(f"Resuming from epoch {START_EPOCH}")


def print_validation_diagnostics(diag: Dict[str, Any], prefix: str = "Val Diagnostics") -> None:
    """Prints Section-9 CTC & Decoder validation diagnostics + exact CTC alignment & Insertion/Deletion counts."""
    if not diag:
        return
    print(f"  [{prefix}]")
    print(
        f"    CTC Impossible Alignments (enc_len < L + R): "
        f"{diag.get('ctc_impossible_alignments', 0)} / {diag.get('total_eval_samples', 0)}"
    )
    print("    CTC Mean Blank Probability:", round(float(diag["batch1_ctc_mean_blank_prob"]), 6))
    print("    CTC Valid-Frame Blank Ratio:", round(float(diag["split_valid_frame_blank_ratio"]), 6))
    print(
        f"    CTC Unique Predicted Tokens: {diag['batch1_ctc_unique_tokens']} "
        f"(Split Valid Frames: {diag['split_ctc_unique_valid_tokens']})"
    )
    print(
        f"    Decoder Unique Predicted Tokens: {diag['batch1_decoder_unique_tokens']} "
        f"(Split Valid Positions: {diag['split_decoder_unique_tokens']})"
    )
    print("    CTC Logits Shape:", diag["batch1_ctc_logits_shape"])
    print("    Decoder Logits Shape:", diag["batch1_decoder_logits_shape"])
    if diag.get("ctc_pred_ref_char_ratio") is not None:
        print(
            f"    Pred/Ref Char Length Ratio -> CTC: {diag['ctc_pred_ref_char_ratio']:.3f} | "
            f"Decoder: {diag['decoder_pred_ref_char_ratio']:.3f}"
        )
    if diag.get("ctc_char_insertions") is not None:
        print(
            f"    CTC Errors     -> Char Ins/Del/Sub: {diag['ctc_char_insertions']} / "
            f"{diag['ctc_char_deletions']} / {diag['ctc_char_substitutions']} | "
            f"Word Ins/Del/Sub: {diag['ctc_word_insertions']} / "
            f"{diag['ctc_word_deletions']} / {diag['ctc_word_substitutions']}"
        )
        print(
            f"    Decoder Errors -> Char Ins/Del/Sub: {diag['dec_char_insertions']} / "
            f"{diag['dec_char_deletions']} / {diag['dec_char_substitutions']} | "
            f"Word Ins/Del/Sub: {diag['dec_word_insertions']} / "
            f"{diag['dec_word_deletions']} / {diag['dec_word_substitutions']}"
        )


def evaluate_split(
    eval_model,
    dataloader,
    compute_metrics=True,
    max_len=300,
    no_repeat_ngram_size: int = 0,
    log_diagnostics: bool = False,
):
    """
    Evaluates loss (Transformer Decoder CE + Mid-Conformer Layer-2 Intermediate CTC) and optionally WER/CER.
    - Uses unbiased defaults (`no_repeat_ngram_size=0`) so evaluation measures true model quality.
    - Checks exact CTC alignability `enc_lengths < get_ctc_minimum_lengths(...)` (`T_min = L + R`) and
      records separate Intermediate-CTC vs Transformer Decoder Insertion/Deletion/Substitution counts.
    """
    eval_model.eval()
    total_loss = 0.0
    valid_eval_batches = 0
    all_refs, all_preds = [], []
    all_ctc_preds = []

    batch1_diag: Dict[str, Any] = {}
    total_valid_frames = 0
    total_valid_blank_frames = 0
    ctc_valid_token_set = set()
    dec_valid_token_set = set()
    ctc_impossible_alignments = 0
    total_eval_samples = 0

    with torch.no_grad():
        for batch_idx, (
            padded_mels,
            mel_lengths,
            tgt_input,
            tgt_labels,
            padded_ctc_targets,
            ctc_lengths,
            raw_texts,
            _,
            _,
        ) in enumerate(dataloader):
            padded_mels = padded_mels.to(device)
            mel_lengths = mel_lengths.to(device)
            tgt_input = tgt_input.to(device)
            tgt_labels = tgt_labels.to(device)
            padded_ctc_targets = padded_ctc_targets.to(device)
            ctc_lengths = ctc_lengths.to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits, inter_ctc_logits, enc_lengths = eval_model(
                    padded_mels, mel_lengths, tgt_input, return_inter_ctc=True
                )
                ce_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))

            # Prepare 1D non-blank CTC targets, check exact T_min = L + R, & compute InterCTC loss in float32:
            ctc_targets_1d = prepare_1d_ctc_targets(
                padded_ctc_targets,
                ctc_lengths,
                blank_id=tokenizer.pad_id,
                sos_id=tokenizer.sos_id,
                eos_id=tokenizer.eos_id,
            )
            required_lengths = get_ctc_minimum_lengths(padded_ctc_targets, ctc_lengths)
            invalid_alignment = enc_lengths < required_lengths
            ctc_impossible_alignments += int(invalid_alignment.sum().item())
            total_eval_samples += int(len(invalid_alignment))

            inter_ctc_log_probs = F.log_softmax(inter_ctc_logits.float(), dim=-1).transpose(0, 1)  # (T, B, V)
            inter_ctc_loss = F.ctc_loss(
                inter_ctc_log_probs,
                ctc_targets_1d,
                enc_lengths,
                ctc_lengths,
                blank=tokenizer.pad_id,
                reduction="mean",
                zero_infinity=True,
            )
            loss = (1.0 - INTER_CTC_WEIGHT) * ce_loss + INTER_CTC_WEIGHT * inter_ctc_loss
            if torch.isfinite(loss):
                total_loss += float(loss.item())
                valid_eval_batches += 1

            # --- InterCTC & Decoder Diagnostics ---
            ctc_probs = torch.softmax(inter_ctc_logits.float(), dim=-1)
            top_tokens = inter_ctc_logits.argmax(dim=-1)
            decoder_pred = logits.argmax(dim=-1)

            if batch_idx == 0:
                batch1_diag = {
                    "batch1_ctc_mean_blank_prob": float(ctc_probs[..., tokenizer.pad_id].mean().item()),
                    "batch1_ctc_unique_tokens": int(torch.unique(top_tokens).numel()),
                    "batch1_decoder_unique_tokens": int(torch.unique(decoder_pred).numel()),
                    "batch1_ctc_logits_shape": tuple(inter_ctc_logits.shape),
                    "batch1_decoder_logits_shape": tuple(logits.shape),
                }

            valid_frame_mask = (
                torch.arange(inter_ctc_logits.size(1), device=inter_ctc_logits.device)[None, :]
                < enc_lengths[:, None]
            )
            total_valid_frames += int(valid_frame_mask.sum().item())
            total_valid_blank_frames += float(
                ctc_probs[..., tokenizer.pad_id][valid_frame_mask].sum().item()
            )
            ctc_valid_token_set.update(top_tokens[valid_frame_mask].detach().cpu().tolist())
            dec_valid_mask = tgt_labels != tokenizer.pad_id
            dec_valid_token_set.update(decoder_pred[dec_valid_mask].detach().cpu().tolist())

            if compute_metrics:
                gen_ids = eval_model.greedy_decode(
                    padded_mels,
                    mel_lengths,
                    sos_id=tokenizer.sos_id,
                    eos_id=tokenizer.eos_id,
                    max_len=max_len,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=1.0,
                )
                for ref_str, pred_seq in zip(raw_texts, gen_ids):
                    all_refs.append(preprocess_quran_text(ref_str))
                    all_preds.append(tokenizer.decode(pred_seq))

                # Mid-Conformer Layer-2 Intermediate CTC greedy collapse (diagnostic)
                ctc_toks = torch.argmax(inter_ctc_logits, dim=-1)
                for b in range(padded_mels.size(0)):
                    c_seq, prev = [], -1
                    cur_len = enc_lengths[b].item()
                    for tok in ctc_toks[b, :cur_len].tolist():
                        if tok != prev:
                            if tok not in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id):
                                c_seq.append(tok)
                            prev = tok
                    all_ctc_preds.append(tokenizer.decode(c_seq))

    avg_loss = total_loss / max(1, valid_eval_batches)
    ref_char_total = sum(len(r) for r in all_refs) if all_refs else 0
    dec_char_total = sum(len(p) for p in all_preds) if all_preds else 0
    ctc_char_total = sum(len(p) for p in all_ctc_preds) if all_ctc_preds else 0

    err_counts: Dict[str, Any] = {}
    if compute_metrics and all_refs:
        norm_refs = [normalize_arabic(r) for r in all_refs]
        norm_preds = [normalize_arabic(p) for p in all_preds]
        norm_ctc = [normalize_arabic(p) for p in all_ctc_preds]

        # Corpus-Level metrics (Official JiWER standard)
        wer_norm = 100.0 * jiwer.wer(norm_refs, norm_preds)
        cer_norm = 100.0 * jiwer.cer(norm_refs, norm_preds)
        cer_exact = 100.0 * jiwer.cer(all_refs, all_preds)

        ctc_wer = 100.0 * jiwer.wer(norm_refs, norm_ctc)
        ctc_cer = 100.0 * jiwer.cer(norm_refs, norm_ctc)

        # Character-level & Word-level Insertions / Deletions / Substitutions for CTC and Attention Decoder
        ctc_char_out = jiwer.process_characters(norm_refs, norm_ctc)
        dec_char_out = jiwer.process_characters(norm_refs, norm_preds)
        ctc_word_out = jiwer.process_words(norm_refs, norm_ctc)
        dec_word_out = jiwer.process_words(norm_refs, norm_preds)
        err_counts = {
            "ctc_char_insertions": int(ctc_char_out.insertions),
            "ctc_char_deletions": int(ctc_char_out.deletions),
            "ctc_char_substitutions": int(ctc_char_out.substitutions),
            "dec_char_insertions": int(dec_char_out.insertions),
            "dec_char_deletions": int(dec_char_out.deletions),
            "dec_char_substitutions": int(dec_char_out.substitutions),
            "ctc_word_insertions": int(ctc_word_out.insertions),
            "ctc_word_deletions": int(ctc_word_out.deletions),
            "ctc_word_substitutions": int(ctc_word_out.substitutions),
            "dec_word_insertions": int(dec_word_out.insertions),
            "dec_word_deletions": int(dec_word_out.deletions),
            "dec_word_substitutions": int(dec_word_out.substitutions),
        }
    else:
        wer_norm, cer_norm, cer_exact = None, None, None
        ctc_wer, ctc_cer = None, None

    val_diag = {
        **batch1_diag,
        "split_valid_frame_blank_ratio": total_valid_blank_frames / max(1, total_valid_frames),
        "split_ctc_unique_valid_tokens": len(ctc_valid_token_set),
        "split_decoder_unique_tokens": len(dec_valid_token_set),
        "ctc_impossible_alignments": ctc_impossible_alignments,
        "total_eval_samples": total_eval_samples,
        "ctc_length_violations": ctc_impossible_alignments,
        "ctc_pred_ref_char_ratio": (ctc_char_total / max(1, ref_char_total)) if compute_metrics and all_refs else None,
        "decoder_pred_ref_char_ratio": (dec_char_total / max(1, ref_char_total)) if compute_metrics and all_refs else None,
        **err_counts,
    }
    eval_model.last_val_diagnostics = val_diag
    if log_diagnostics:
        print_validation_diagnostics(val_diag)

    return avg_loss, wer_norm, cer_norm, cer_exact, ctc_wer, ctc_cer, all_refs, all_preds


def corrupt_decoder_inputs(tgt_in: torch.Tensor, prob: float = 0.20) -> torch.Tensor:
    """
    Scheduled Teacher-Forcing Token Corruption (Anti-Memorization Regularizer):
    Replaces `prob` (20%) of non-special history tokens in `tgt_input` (50% with <unk>, 50% with random
    Arabic chars) while preserving <sos> and <pad>. Forces the Decoder to read acoustic Cross-Attention
    and teaches self-correction against exposure bias.
    """
    if prob <= 0.0:
        return tgt_in
    corrupted = tgt_in.clone()
    valid_mask = (
        (corrupted != tokenizer.pad_id)
        & (corrupted != tokenizer.sos_id)
        & (corrupted != tokenizer.eos_id)
    )
    valid_mask[:, 0] = False  # Never corrupt initial <sos>
    drop_mask = valid_mask & (torch.rand_like(corrupted, dtype=torch.float32) < prob)
    if drop_mask.any():
        rand_split = torch.rand_like(corrupted, dtype=torch.float32) < 0.5
        unk_mask = drop_mask & rand_split
        rnd_mask = drop_mask & (~rand_split)
        corrupted[unk_mask] = tokenizer.unk_id
        if rnd_mask.any():
            rand_tokens = torch.randint(
                4, len(tokenizer), size=(int(rnd_mask.sum().item()),), device=corrupted.device
            )
            corrupted[rnd_mask] = rand_tokens
    return corrupted


print(
    f"Starting {EXPERIMENT_TAG} on {SURAH_NAME_AR} for up to {EPOCHS} epochs ({len(train_loader)} batches/epoch) | "
    f"INTER_CTC_WEIGHT={INTER_CTC_WEIGHT} | Conformer norm: {model.conformer_norm_mode}"
)
print(f"  Pipeline: {model.describe_active_pipeline()}\n  Branch checkpoints -> {BRANCH_CKPT_DIR}\n")

stopped_early_epoch = None
for epoch in range(START_EPOCH, EPOCHS + 1):
    train_dataset.set_epoch(epoch)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    running_loss = running_ce = running_ctc = running_gn = 0.0
    valid_loss_batches = 0
    epoch_impossible_ctc = 0
    epoch_total_samples = 0
    cur_tok_drop = 0.10 if epoch <= 10 else TOKEN_CORRUPTION_PROB

    optimizer.zero_grad(set_to_none=True)
    for step_idx, (
        padded_mels, mel_lengths, tgt_input, tgt_labels, padded_ctc_targets, ctc_lengths, _, _, _,
    ) in enumerate(train_loader):
        padded_mels = padded_mels.to(device)
        mel_lengths = mel_lengths.to(device)
        tgt_input = tgt_input.to(device)
        tgt_labels = tgt_labels.to(device)
        padded_ctc_targets = padded_ctc_targets.to(device)
        ctc_lengths = ctc_lengths.to(device)

        noisy_tgt_input = corrupt_decoder_inputs(tgt_input, prob=cur_tok_drop)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits, inter_ctc_logits, enc_lengths = model(
                padded_mels, mel_lengths, noisy_tgt_input, return_inter_ctc=True
            )
            ce_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))

        required_lengths = get_ctc_minimum_lengths(padded_ctc_targets, ctc_lengths)
        invalid_alignment = enc_lengths < required_lengths
        batch_impossible = int(invalid_alignment.sum().item())
        epoch_impossible_ctc += batch_impossible
        epoch_total_samples += int(len(invalid_alignment))
        if (epoch == START_EPOCH and step_idx == 0) or batch_impossible > 0:
            print("CTC Impossible Alignments:", batch_impossible, "/", len(invalid_alignment))

        ctc_targets_1d = prepare_1d_ctc_targets(
            padded_ctc_targets, ctc_lengths,
            blank_id=tokenizer.pad_id, sos_id=tokenizer.sos_id, eos_id=tokenizer.eos_id,
        )
        inter_ctc_log_probs = F.log_softmax(inter_ctc_logits.float(), dim=-1).transpose(0, 1)  # (T, B, V)
        inter_ctc_loss = F.ctc_loss(
            inter_ctc_log_probs, ctc_targets_1d, enc_lengths, ctc_lengths,
            blank=tokenizer.pad_id, reduction="mean", zero_infinity=True,
        )
        raw_loss = (1.0 - INTER_CTC_WEIGHT) * ce_loss + INTER_CTC_WEIGHT * inter_ctc_loss
        loss = raw_loss / GRAD_ACCUM_STEPS if GRAD_ACCUM_STEPS > 1 else raw_loss

        scaler.scale(loss).backward()
        if torch.isfinite(raw_loss) and torch.isfinite(ce_loss) and torch.isfinite(inter_ctc_loss):
            running_loss += float(raw_loss.item())
            running_ce += float(ce_loss.item())
            running_ctc += float(inter_ctc_loss.item())
            valid_loss_batches += 1

        if (step_idx + 1) % GRAD_ACCUM_STEPS == 0 or (step_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            if math.isfinite(float(gn)):
                running_gn += float(gn)

    n_valid = max(1, valid_loss_batches)
    train_loss = running_loss / n_valid
    peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2) if torch.cuda.is_available() else 0.0

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_ce_loss"].append(running_ce / n_valid)
    history["train_ctc_loss"].append(running_ctc / n_valid)
    history["train_ctc_impossible"].append(epoch_impossible_ctc)
    history["train_total_samples"].append(epoch_total_samples)
    history["lr"].append(optimizer.param_groups[0]["lr"])
    history["grad_norm"].append(running_gn / max(1, updates_per_epoch))
    history["peak_gpu_mem_mb"].append(peak_mem_mb)

    should_eval_metrics = (epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS)
    raw_res = evaluate_split(
        model, val_loader, compute_metrics=should_eval_metrics, max_len=350, no_repeat_ngram_size=0
    )

    if not should_eval_metrics:
        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {raw_res[0]:.4f} | "
            f"CTC Impossible (Train): {epoch_impossible_ctc}/{epoch_total_samples} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        continue

    raw_diag = copy.deepcopy(getattr(model, "last_val_diagnostics", {}))
    ema_res = evaluate_split(
        ema.ema_model, val_loader, compute_metrics=True, max_len=350, no_repeat_ngram_size=0
    )
    ema_diag = copy.deepcopy(getattr(ema.ema_model, "last_val_diagnostics", {}))

    # res = (loss, dec_wer, dec_cer, dec_cer_exact, ctc_wer, ctc_cer, refs, preds)
    # Display/history source: the one with the lower Decoder CER (primary criterion)
    if ema_res[2] <= raw_res[2]:
        val_loss, val_wer, val_cer, val_cer_exact, ctc_wer, ctc_cer, _, _ = ema_res
        chosen_src, chosen_eval_model, chosen_diag = "EMA", ema.ema_model, ema_diag
    else:
        val_loss, val_wer, val_cer, val_cer_exact, ctc_wer, ctc_cer, _, _ = raw_res
        chosen_src, chosen_eval_model, chosen_diag = "Model", model, raw_diag

    tr_eval_loss, tr_dec_wer, tr_dec_cer, tr_dec_cer_exact, tr_ctc_wer, tr_ctc_cer, _, _ = evaluate_split(
        chosen_eval_model, train_eval_loader, compute_metrics=True, max_len=350, no_repeat_ngram_size=0
    )
    train_eval_diag = copy.deepcopy(getattr(chosen_eval_model, "last_val_diagnostics", {}))
    chosen_eval_model.last_val_diagnostics = chosen_diag

    ctc_cer_gap = float(ctc_cer - tr_ctc_cer)
    dec_cer_gap = float(val_cer - tr_dec_cer)

    history["val_epoch"].append(epoch)
    history["val_loss"].append(val_loss)
    history["val_wer"].append(val_wer)
    history["val_cer"].append(val_cer)
    history["val_cer_exact"].append(val_cer_exact)
    history["val_ctc_wer"].append(ctc_wer)
    history["val_ctc_cer"].append(ctc_cer)
    history["val_ctc_impossible"].append(int(chosen_diag.get("ctc_impossible_alignments", 0)))
    history["train_eval_loss"].append(tr_eval_loss)
    history["train_dec_wer"].append(tr_dec_wer)
    history["train_dec_cer"].append(tr_dec_cer)
    history["train_dec_cer_exact"].append(tr_dec_cer_exact)
    history["train_ctc_wer"].append(tr_ctc_wer)
    history["train_ctc_cer"].append(tr_ctc_cer)
    history["train_clean_ctc_impossible"].append(int(train_eval_diag.get("ctc_impossible_alignments", 0)))
    history["ctc_cer_gap"].append(ctc_cer_gap)
    history["dec_cer_gap"].append(dec_cer_gap)
    for h_key, d_key in (
        ("val_ctc_char_ins", "ctc_char_insertions"), ("val_ctc_char_del", "ctc_char_deletions"),
        ("val_ctc_char_sub", "ctc_char_substitutions"), ("val_dec_char_ins", "dec_char_insertions"),
        ("val_dec_char_del", "dec_char_deletions"), ("val_dec_char_sub", "dec_char_substitutions"),
        ("val_ctc_word_ins", "ctc_word_insertions"), ("val_ctc_word_del", "ctc_word_deletions"),
        ("val_dec_word_ins", "dec_word_insertions"), ("val_dec_word_del", "dec_word_deletions"),
    ):
        history[h_key].append(int(chosen_diag.get(d_key, 0)))

    periodic_eval_rows.append({
        "Epoch": epoch, "Source": chosen_src,
        "Train CTC CER (%)": round(tr_ctc_cer, 2), "Val CTC CER (%)": round(ctc_cer, 2), "CTC Gap (%)": round(ctc_cer_gap, 2),
        "Train Dec CER (%)": round(tr_dec_cer, 2), "Val Dec CER (%)": round(val_cer, 2), "Dec Gap (%)": round(dec_cer_gap, 2),
        "Val CTC WER (%)": round(ctc_wer, 2), "Val Dec WER (%)": round(val_wer, 2),
        "Val CTC Char Ins/Del": f"{chosen_diag.get('ctc_char_insertions', 0)}/{chosen_diag.get('ctc_char_deletions', 0)}",
        "Val Dec Char Ins/Del": f"{chosen_diag.get('dec_char_insertions', 0)}/{chosen_diag.get('dec_char_deletions', 0)}",
    })

    print(
        f"Epoch [{epoch:02d}/{EPOCHS:02d}] [{chosen_src}] | Loss: {train_loss:.4f} "
        f"(CE: {history['train_ce_loss'][-1]:.4f}, CTC: {history['train_ctc_loss'][-1]:.4f}) | Val Loss: {val_loss:.4f} | "
        f"Mem: {peak_mem_mb:.0f}MB | LR: {optimizer.param_groups[0]['lr']:.2e}"
    )
    print(
        f"  [Attention Decoder - PRIMARY] Train CER: {tr_dec_cer:6.2f}% | Val CER: {val_cer:6.2f}% (Gap {dec_cer_gap:+6.2f}%) | "
        f"Train WER: {tr_dec_wer:6.2f}% | Val WER: {val_wer:6.2f}% | "
        f"Val Char Ins/Del/Sub: {chosen_diag.get('dec_char_insertions', 0)}/{chosen_diag.get('dec_char_deletions', 0)}/{chosen_diag.get('dec_char_substitutions', 0)}"
    )
    print(
        f"  [CTC Aux Head]                Train CER: {tr_ctc_cer:6.2f}% | Val CER: {ctc_cer:6.2f}% (Gap {ctc_cer_gap:+6.2f}%) | "
        f"Train WER: {tr_ctc_wer:6.2f}% | Val WER: {ctc_wer:6.2f}%"
    )

    # Branch-separated checkpoints: each branch independently picks the better of Model vs EMA
    upd = tracker.update(
        epoch=epoch,
        raw_metrics={"dec_cer": raw_res[2], "dec_wer": raw_res[1], "ctc_cer": raw_res[5], "ctc_wer": raw_res[4]},
        ema_metrics={"dec_cer": ema_res[2], "dec_wer": ema_res[1], "ctc_cer": ema_res[5], "ctc_wer": ema_res[4]},
        raw_state=model.state_dict(),
        ema_state=ema.ema_model.state_dict(),
        extra_payload={
            "inter_ctc_layer_idx": model.inter_ctc_layer_idx,
            "experiment_tag": EXPERIMENT_TAG,
            "asr_config": model.asr_config.to_dict(),
            "conformer_norm_mode": model.conformer_norm_mode,
            "ctc_weight": CTC_WEIGHT,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema.ema_model.state_dict(),
            "ema_updates": ema.num_updates,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
        },
    )
    for br, info in upd["saved_branches"].items():
        print(f"  --> best_{br}.pt saved [{info['source']}] (epoch {info['epoch']}, metric {info['primary_metric']:.3f})")
    if upd["decoder_improved"]:
        best_val_cer, best_val_wer = tracker.best["decoder_cer"][0], tracker.best["decoder_wer"][0]
    elif epoch >= EARLY_STOP_MIN_EPOCH:
        print(f"  (Decoder has not improved for {upd['evals_without_decoder_improvement']}/{EARLY_STOPPING_PATIENCE_EVALS} evaluations)")
    if upd["should_early_stop"]:
        stopped_early_epoch = epoch
        print(
            f"\nEARLY STOPPING at epoch {epoch}: Validation Decoder CER/WER stopped improving "
            f"(best decoder CER epoch = {tracker.best_epochs['decoder_cer']}, best decoder WER epoch = {tracker.best_epochs['decoder_wer']})."
        )
        break
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f)
with open(os.path.join(PROJECT_DIR, "training_history.json"), "w") as f:
    json.dump(history, f)

if periodic_eval_rows:
    periodic_eval_df = pd.DataFrame(periodic_eval_rows)
    periodic_eval_df.to_csv(PERIODIC_CSV_PATH, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 116)
    print(f"5-EPOCH PROGRESSION SUMMARY — {EXPERIMENT_TAG} (Train vs Val CER | InterCTC@L2 vs Attention Decoder):")
    print("=" * 116)
    print(periodic_eval_df.to_string(index=False))
    print("=" * 116 + "\n")

# %% [markdown]
# ## Step 9 — Validation-First Model Selection, Attention-Beam Tuning on Validation ONLY, One Clean Test Evaluation
# 1. Compare all saved branch checkpoints on **Validation** (Attention Greedy) and pick the final model *before* touching Test.
# 2. Sweep Attention Beam Search (`B in {1,3,5,8}`, GNMT length-norm, EOS bonus) on **Validation only**.
# 3. Compare `unmasked_group_norm` vs `masked_group_norm` on the **same checkpoint** (drift measured on valid frames only).
# 4. Evaluate the single best Validation setting once on **Test**.

# %%
BEST_DECODE = {"mode": "greedy"}   # updated by the Validation sweep below; used by Step 11 / 12.5 / 14 / 15


@torch.no_grad()
def decode_with_best_setting(mels, lens, dec_model=None, setting=None):
    """Decodes a padded batch with the Validation-selected Attention Decoder setting."""
    m = dec_model if dec_model is not None else model
    cfg = setting if setting is not None else BEST_DECODE
    sos, eos = tokenizer.sos_id, tokenizer.eos_id
    if cfg["mode"] == "attention_beam":
        return m.attention_beam_decode(
            mels, lens, sos_id=sos, eos_id=eos, beam_size=cfg["beam_size"],
            length_norm_alpha=cfg["alpha"], eos_bonus=cfg["eos_bonus"], max_len=350,
        )
    return m.greedy_decode(mels, lens, sos_id=sos, eos_id=eos, max_len=350)


@torch.no_grad()
def decode_split(eval_model, dataloader, decode_fn):
    eval_model.eval()
    refs, preds = [], []
    for padded_mels, mel_lengths, _, _, _, _, raw_texts, _, _ in dataloader:
        gen = decode_fn(eval_model, padded_mels.to(device), mel_lengths.to(device))
        for r, g in zip(raw_texts, gen):
            refs.append(preprocess_quran_text(r))
            preds.append(tokenizer.decode(g))
    return refs, preds


def score_decode(refs, preds):
    nr = [normalize_arabic(r) for r in refs]
    npd = [normalize_arabic(p) for p in preds]
    return 100.0 * jiwer.wer(nr, npd), 100.0 * jiwer.cer(nr, npd)


def att_greedy_fn(m, mels, lens):
    return m.greedy_decode(mels, lens, sos_id=tokenizer.sos_id, eos_id=tokenizer.eos_id, max_len=350)


@torch.no_grad()
def ctc_greedy_fn(m, mels, lens):
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        mem, _, inter_ctc_logits = m.encode_audio(mels, lens, return_inter_ctc=True)
        toks = inter_ctc_logits.argmax(-1)
    enc = torch.clamp(m._compute_cnn_lengths(lens), min=1, max=mem.size(1))
    out = []
    for b in range(mels.size(0)):
        seq, prev = [], -1
        for t in toks[b, : int(enc[b])].tolist():
            if t != prev:
                if t not in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id):
                    seq.append(t)
                prev = t
        out.append(torch.tensor(seq, dtype=torch.long))
    return out


# ---- 1. Validation-first checkpoint selection --------------------------------
def select_best_checkpoint_on_validation(eval_model, candidates):
    rows, best = [], None
    for name, path in candidates.items():
        if not os.path.isfile(path):
            continue
        rep = load_compatible_checkpoint(eval_model, path, map_device=device, prefer_best_eval=True)
        refs, preds = decode_split(eval_model, val_loader, att_greedy_fn)
        wer, cer = score_decode(refs, preds)
        ck = rep["raw_checkpoint"]
        rows.append({
            "Checkpoint": name, "Epoch": ck.get("epoch", "?"), "Source": ck.get("best_weight_source", "?"),
            "Exact Match": rep["exact_match"], "Val Dec WER (%)": round(wer, 2), "Val Dec CER (%)": round(cer, 2),
        })
        if best is None or (cer, wer) < best[0]:
            best = ((cer, wer), name, path)
    df = pd.DataFrame(rows)
    if best is not None:
        load_compatible_checkpoint(eval_model, best[2], map_device=device, prefer_best_eval=True)
        print(f"Selected checkpoint on VALIDATION: '{best[1]}'  (Dec CER {best[0][0]:.2f}% | WER {best[0][1]:.2f}%)")
    return df, best


ckpt_candidates = {**BRANCH_CKPT_PATHS, **EXTRA_CHECKPOINTS}
if not any(os.path.isfile(p) for p in ckpt_candidates.values()) and os.path.isfile(CHECKPOINT_PATH):
    ckpt_candidates = {"legacy": CHECKPOINT_PATH}
ckpt_sel_df, ckpt_sel_best = select_best_checkpoint_on_validation(model, ckpt_candidates)
print("=" * 96)
print("VALIDATION-ONLY CHECKPOINT COMPARISON (Attention Greedy):")
print(ckpt_sel_df.to_string(index=False) if len(ckpt_sel_df) else "No checkpoints found - using current in-memory weights.")
print("=" * 96 + "\n")

# ---- 2. Attention beam search sweep on Validation ONLY ----------------------
ATT_BEAM_GRID = [(1, 0.0, 0.0)] + [(b, a, 0.0) for b in (3, 5, 8) for a in (0.0, 0.6, 1.0)] + [(5, 0.6, 1.0), (5, 0.6, -1.0)]
beam_rows = []
for b_size, alpha, eos_b in ATT_BEAM_GRID:
    fn = (lambda m, mels, lens, b_size=b_size, alpha=alpha, eos_b=eos_b: m.attention_beam_decode(
        mels, lens, sos_id=tokenizer.sos_id, eos_id=tokenizer.eos_id, beam_size=b_size,
        length_norm_alpha=alpha, eos_bonus=eos_b, max_len=350))
    refs_v, preds_v = decode_split(model, val_loader, fn)
    w_v, c_v = score_decode(refs_v, preds_v)
    beam_rows.append({"Beam": b_size, "Length-Norm alpha": alpha, "EOS bonus": eos_b,
                      "Val WER (%)": round(w_v, 2), "Val CER (%)": round(c_v, 2)})
beam_df = pd.DataFrame(beam_rows)
print("=" * 80)
print("ATTENTION BEAM SEARCH SWEEP (VALIDATION ONLY; Beam=1 is Greedy):")
print(beam_df.to_string(index=False))
print("=" * 80)
greedy_row = beam_df.iloc[0]
best_row = beam_df.sort_values(["Val CER (%)", "Val WER (%)"]).iloc[0]
if (best_row["Val CER (%)"], best_row["Val WER (%)"]) < (greedy_row["Val CER (%)"], greedy_row["Val WER (%)"]):
    BEST_DECODE = {"mode": "attention_beam", "beam_size": int(best_row["Beam"]),
                   "alpha": float(best_row["Length-Norm alpha"]), "eos_bonus": float(best_row["EOS bonus"])}
else:
    BEST_DECODE = {"mode": "greedy"}
best_att_val = (float(min(best_row["Val CER (%)"], greedy_row["Val CER (%)"])),
                float(best_row["Val WER (%)"] if best_row["Val CER (%)"] <= greedy_row["Val CER (%)"] else greedy_row["Val WER (%)"]))
print("Best Attention setting on Validation:", BEST_DECODE)

print("\nFINAL DEFAULT DECODING SETTING (selected on Validation only):", BEST_DECODE, "\n")

@torch.no_grad()
def evaluate_conformer_padding_sensitivity(eval_model, dataset, split_name="Validation", batch_size=BATCH_SIZE, seed=SEED):
    """
    Tests the impact of variable-length batch padding on `torchaudio.models.Conformer`
    by comparing on the exact same checkpoint:
      1. Length-Bucketed / Sorted Batching (similar clip lengths -> minimal padding)
      2. Random Batching (mixed short & long recordings -> high padding ratio)
      3. Single-Sample Zero-Padding (`batch_size=1`, 0% padding ground truth)
    """
    eval_model.eval()
    n_items = len(dataset)
    if n_items == 0:
        return pd.DataFrame()

    # 1. Exact zero-padding reference representations & predictions at batch_size = 1
    single_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_asr_batch)
    ref_memories: List[torch.Tensor] = []
    for padded_mels, mel_lengths, _, _, _, _, _, _, _ in single_loader:
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            mem_1, _ = eval_model.encode_audio(padded_mels.to(device), mel_lengths.to(device))
        ref_memories.append(mem_1[0].detach().float().cpu())

    # Deterministic random permutation of indices so every run is 100% reproducible
    rng = np.random.default_rng(seed)
    rand_indices = rng.permutation(n_items).tolist()
    rand_batches = [rand_indices[i : i + batch_size] for i in range(0, n_items, batch_size)]

    sorted_indices = np.argsort(np.asarray(dataset.lengths)).tolist()
    bucket_batches = [sorted_indices[i : i + batch_size] for i in range(0, n_items, batch_size)]
    single_batches = [[i] for i in range(n_items)]

    regimes = [
        ("1. Zero-Padding Reference (B=1)", single_batches),
        (f"2. Length-Bucketed / Sorted (B={batch_size})", bucket_batches),
        (f"3. Random Unsorted Batching (B={batch_size})", rand_batches),
    ]

    rows = []
    for regime_name, index_batches in regimes:
        total_slots = 0
        padded_slots = 0
        max_abs_drift = 0.0
        mean_abs_drifts = []
        refs_ord = [""] * n_items
        ctc_preds_ord = [""] * n_items
        dec_preds_ord = [""] * n_items

        for idx_batch in index_batches:
            raw_items = [dataset[idx] for idx in idx_batch]
            (
                padded_mels,
                mel_lengths,
                _,
                _,
                _,
                _,
                raw_texts,
                _,
                _,
            ) = collate_asr_batch(raw_items)
            padded_mels_d = padded_mels.to(device)
            mel_lengths_d = mel_lengths.to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                mem_b, _, ctc_logits = eval_model.encode_audio(padded_mels_d, mel_lengths_d, return_inter_ctc=True)
            enc_lens = torch.clamp(eval_model._compute_cnn_lengths(mel_lengths_d), min=1, max=mem_b.size(1))
            dec_gen = eval_model.greedy_decode(
                padded_mels_d,
                mel_lengths_d,
                sos_id=tokenizer.sos_id,
                eos_id=tokenizer.eos_id,
                max_len=350,
                no_repeat_ngram_size=0,
            )

            B_cur, T_cur = mem_b.size(0), mem_b.size(1)
            total_slots += B_cur * T_cur
            padded_slots += int((B_cur * T_cur) - enc_lens.sum().item())

            ctc_toks = ctc_logits.argmax(dim=-1)
            mem_cpu = mem_b.detach().float().cpu()
            for local_b, orig_idx in enumerate(idx_batch):
                valid_L = int(enc_lens[local_b].item())
                ref_m = ref_memories[orig_idx][:valid_L]
                cur_m = mem_cpu[local_b, :valid_L]
                diff = (cur_m - ref_m).abs()
                if diff.numel() > 0:
                    max_abs_drift = max(max_abs_drift, float(diff.max().item()))
                    mean_abs_drifts.append(float(diff.mean().item()))

                refs_ord[orig_idx] = preprocess_quran_text(raw_texts[local_b])
                dec_preds_ord[orig_idx] = tokenizer.decode(dec_gen[local_b])
                c_seq, prev = [], -1
                for tok in ctc_toks[local_b, :valid_L].tolist():
                    if tok != prev:
                        if tok not in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id):
                            c_seq.append(tok)
                        prev = tok
                ctc_preds_ord[orig_idx] = tokenizer.decode(c_seq)

        norm_r = [normalize_arabic(r) for r in refs_ord]
        norm_c = [normalize_arabic(p) for p in ctc_preds_ord]
        norm_d = [normalize_arabic(p) for p in dec_preds_ord]
        rows.append({
            "Split": split_name,
            "Batching Regime": regime_name,
            "Padding Ratio (%)": round(100.0 * padded_slots / max(1, total_slots), 2),
            "Mean Enc Drift vs B=1": round(float(np.mean(mean_abs_drifts)) if mean_abs_drifts else 0.0, 5),
            "Max Enc Drift vs B=1": round(max_abs_drift, 5),
            "InterCTC WER (%)": round(100.0 * jiwer.wer(norm_r, norm_c), 2),
            "InterCTC CER (%)": round(100.0 * jiwer.cer(norm_r, norm_c), 2),
            "Decoder WER (%)": round(100.0 * jiwer.wer(norm_r, norm_d), 2),
            "Decoder CER (%)": round(100.0 * jiwer.cer(norm_r, norm_d), 2),
        })
    return pd.DataFrame(rows)


def compare_conformer_norm_modes(eval_model, dataset, split_name="Validation"):
    """Same checkpoint, same recordings: legacy unmasked GroupNorm vs Masked GroupNorm (drift on valid frames only)."""
    original_mode = eval_model.conformer_norm_mode
    frames = []
    for mode in ("unmasked_group_norm", "masked_group_norm"):
        eval_model.set_conformer_norm_mode(mode)
        df_m = evaluate_conformer_padding_sensitivity(eval_model, dataset, split_name=split_name)
        df_m.insert(0, "Conformer Norm", mode)
        frames.append(df_m)
    eval_model.set_conformer_norm_mode(original_mode)
    return pd.concat(frames, ignore_index=True)


if model.conformer_norm_mode != "frame_layer_norm":
    pad_val_df = compare_conformer_norm_modes(model, val_dataset, split_name="Validation")
    print("=" * 120)
    print("CONFORMER PADDING SENSITIVITY ON THE SAME CHECKPOINT (Validation): unmasked vs masked GroupNorm")
    print("=" * 120)
    print(pad_val_df.to_string(index=False))
    print("=" * 120 + "\n")

# ---- 4. One clean evaluation on the held-out Test split ----------------------
print("RESEARCH-HYGIENE NOTE: every decoding choice below was made on Validation only; Test is evaluated once.\n")
test_loss, test_wer, test_cer, test_cer_exact, test_ctc_wer, test_ctc_cer, test_refs, test_preds = evaluate_split(
    model, test_loader, compute_metrics=True, no_repeat_ngram_size=0, log_diagnostics=True
)
refs_b, preds_best = decode_split(model, test_loader, lambda m, mels, lens: decode_with_best_setting(mels, lens, dec_model=m))
w_b, c_b = score_decode(refs_b, preds_best)
benchmark_results = [
    {"Strategy": "Attention Greedy (reference)", "WER (%)": round(test_wer, 2), "CER (%)": round(test_cer, 2)},
    {"Strategy": f"Attention Decoder - Validation-selected {BEST_DECODE}", "WER (%)": round(w_b, 2), "CER (%)": round(c_b, 2)},
]
bench_df = pd.DataFrame(benchmark_results)
print("\n" + "=" * 88)
print(f"FINAL TEST RESULT ({SURAH_NAME_AR} - Unseen Reciters):")
print("=" * 88)
print(bench_df.to_string(index=False))
print("=" * 88 + "\n")

if test_unseen_ayah_audio is not None and len(test_unseen_ayah_audio) > 0:
    ua_dataset = CachedQuranASRDataset(test_unseen_ayah_audio, tokenizer, augment=False, desc="Test (Unseen Ayahs x Unseen Reciters)")
    ua_loader = DataLoader(ua_dataset, batch_sampler=LengthBucketBatchSampler(ua_dataset.lengths, batch_size=BATCH_SIZE, shuffle=False),
                           collate_fn=collate_asr_batch, num_workers=0)
    ua_refs, ua_preds = decode_split(model, ua_loader, lambda m, mels, lens: decode_with_best_setting(mels, lens, dec_model=m))
    ua_w, ua_c = score_decode(ua_refs, ua_preds)
    print(f"UNSEEN-AYAH x UNSEEN-RECITER TEST: WER {ua_w:.2f}% | CER {ua_c:.2f}% ({len(ua_refs)} clips)\n")

for i in range(min(5, len(refs_b))):
    print(f"--- Sample {i+1} ---")
    print("Reference          :", refs_b[i])
    print("Attention Greedy   :", test_preds[i])
    print("Selected Decoding  :", preds_best[i])
    print()
# %% [markdown]
# ## Step 10 — Training Progression Dashboard (Train vs. Val CER for InterCTC & Attention, Insertions/Deletions, Alignment & LR)

# %%
fig, axes = plt.subplots(2, 3, figsize=(22, 11))

# Panel 1: Train vs Validation Loss (Total + InterCTC + CE)
axes[0, 0].plot(history["epoch"], history["train_loss"], color="#2563eb", linewidth=2, label="Train Total Loss")
axes[0, 0].plot(history["epoch"], history["train_ctc_loss"], color="#0ea5e9", linestyle="--", linewidth=1.5, label="Train InterCTC Loss")
axes[0, 0].plot(history["epoch"], history["train_ce_loss"], color="#6366f1", linestyle=":", linewidth=1.5, label="Train Decoder CE")
axes[0, 0].plot(history["val_epoch"], history["val_loss"], color="#dc2626", marker="o", linewidth=2, label="Validation Loss")
axes[0, 0].set_title("1. Train vs. Validation Loss", fontweight="bold")
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend(fontsize=8.5); axes[0, 0].grid(True, linestyle="--", alpha=0.5)

# Panel 2: Mid-Conformer InterCTC CER — Train vs Validation (Every 5 Epochs)
if history.get("train_ctc_cer"):
    axes[0, 1].plot(history["val_epoch"], history["train_ctc_cer"], color="#0284c7", marker="^", linestyle="--", linewidth=2, label="Train InterCTC CER")
axes[0, 1].plot(history["val_epoch"], history["val_ctc_cer"], color="#dc2626", marker="o", linewidth=2, label="Val InterCTC CER")
if history.get("val_ctc_cer"):
    bi_ctc = int(np.argmin(history["val_ctc_cer"]))
    axes[0, 1].scatter(
        [history["val_epoch"][bi_ctc]], [history["val_ctc_cer"][bi_ctc]],
        s=120, color="#991b1b", zorder=5,
        label=f"Best Val InterCTC CER: {history['val_ctc_cer'][bi_ctc]:.2f}% (Ep {history['val_epoch'][bi_ctc]})",
    )
axes[0, 1].set_title("2. Mid-Conformer InterCTC CER: Train vs. Validation", fontweight="bold")
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("CER (%)")
axes[0, 1].legend(fontsize=8.5); axes[0, 1].grid(True, linestyle="--", alpha=0.5)

# Panel 3: Attention Decoder CER — Train vs Validation (Generalization Gap Monitor)
if history.get("train_dec_cer"):
    axes[0, 2].plot(history["val_epoch"], history["train_dec_cer"], color="#16a34a", marker="^", linestyle="--", linewidth=2, label="Train Decoder CER")
axes[0, 2].plot(history["val_epoch"], history["val_cer"], color="#9333ea", marker="o", linewidth=2, label="Val Decoder CER (Norm)")
axes[0, 2].plot(history["val_epoch"], history["val_cer_exact"], color="#6b21a8", marker="d", linestyle=":", linewidth=1.6, label="Val Decoder Exact CER")
axes[0, 2].set_title("3. Attention Decoder CER: Train vs. Validation", fontweight="bold")
axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("CER (%)")
axes[0, 2].legend(fontsize=8.5); axes[0, 2].grid(True, linestyle="--", alpha=0.5)

# Panel 4: Validation WER — InterCTC vs Attention Decoder
axes[1, 0].plot(history["val_epoch"], history["val_ctc_wer"], color="#ea580c", marker="o", linewidth=2, label="Val InterCTC WER")
axes[1, 0].plot(history["val_epoch"], history["val_wer"], color="#16a34a", marker="s", linewidth=2, label="Val Decoder WER")
bi = int(np.argmin(history["val_wer"]))
axes[1, 0].scatter([history["val_epoch"][bi]], [history["val_wer"][bi]], s=130, color="#15803d", zorder=5,
                   label=f"Best Dec WER: {history['val_wer'][bi]:.2f}% (Ep {history['val_epoch'][bi]})")
axes[1, 0].set_title("4. Validation WER: InterCTC vs. Attention Decoder", fontweight="bold")
axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("WER (%)")
axes[1, 0].legend(fontsize=8.5); axes[1, 0].grid(True, linestyle="--", alpha=0.5)

# Panel 5: Validation Character Insertions & Deletions Over Epochs (InterCTC vs Decoder)
if history.get("val_ctc_char_ins"):
    axes[1, 1].plot(history["val_epoch"], history["val_ctc_char_ins"], color="#e11d48", marker="o", linewidth=1.9, label="InterCTC Char Insertions")
    axes[1, 1].plot(history["val_epoch"], history["val_ctc_char_del"], color="#f59e0b", marker="s", linestyle="--", linewidth=1.9, label="InterCTC Char Deletions")
    axes[1, 1].plot(history["val_epoch"], history["val_dec_char_ins"], color="#7c3aed", marker="^", linewidth=1.9, label="Decoder Char Insertions")
    axes[1, 1].plot(history["val_epoch"], history["val_dec_char_del"], color="#0d9488", marker="v", linestyle="--", linewidth=1.9, label="Decoder Char Deletions")
axes[1, 1].set_title("5. Validation Character Insertions & Deletions", fontweight="bold")
axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Error Count (Corpus)")
axes[1, 1].legend(fontsize=8.5); axes[1, 1].grid(True, linestyle="--", alpha=0.5)

# Panel 6: InterCTC Impossible Alignments (T < L+R) & LR / Grad Norm
axes[1, 2].plot(history["epoch"], history["lr"], color="#4f46e5", linewidth=2, label="OneCycleLR")
axes[1, 2].set_ylabel("Learning Rate", color="#4f46e5")
axes[1, 2].set_xlabel("Epoch"); axes[1, 2].grid(True, linestyle="--", alpha=0.5)
ax_gn = axes[1, 2].twinx()
ax_gn.plot(history["epoch"], history["grad_norm"], color="#e11d48", linestyle=":", linewidth=1.8, label="Grad Norm")
if history.get("train_ctc_impossible"):
    ax_gn.plot(history["epoch"], history["train_ctc_impossible"], color="#111827", linewidth=1.8, label="InterCTC Impossible (T<L+R)")
ax_gn.set_ylabel("Grad Norm / InterCTC Impossible Count", color="#e11d48")
axes[1, 2].set_title("6. LR, Grad Norm & InterCTC Impossible Alignments", fontweight="bold")

fig.suptitle(f"{EXPERIMENT_TAG} ({model.conformer_norm_mode}) — {shape_ar(SURAH_NAME_AR)}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "training_progression.png"), dpi=150, bbox_inches="tight")
plt.show()

# Parameter budget per architectural block in exact forward pipeline order
blocks = {
    "1. CNN (2x Conv1d+LN)": [model.conv1, model.ln1, model.conv2, model.ln2],
    "2. Conformer L1-L2": list(model.conformer.conformer_layers[: model.inter_ctc_layer_idx]),
    "3. InterCTC + Self-Cond": [
        m for m in (model.inter_ctc_norm, model.inter_ctc_head, model.inter_ctc_cond) if m is not None
    ],
    "4. Conformer L3-L4 + Norm": list(model.conformer.conformer_layers[model.inter_ctc_layer_idx :])
    + ([model.conformer_norm] if model.conformer_norm is not None else []),
    "5. Embedding (Weight-Tied)": [model.token_embedding],
    "6. Transformer Decoder x4": [model.transformer_decoder],
}
param_counts = {
    k: sum(p.numel() for m in v for p in m.parameters())
    + (model.inter_ctc_gate.numel() if (k.startswith("3.") and model.inter_ctc_gate is not None) else 0)
    + (model.conformer_gate.numel() if (k.startswith("4.") and model.conformer_gate is not None) else 0)
    for k, v in blocks.items()
}
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(list(param_counts.keys()), list(param_counts.values()), color="#6366f1", edgecolor="black")
for b, c in zip(bars, param_counts.values()):
    ax.text(c, b.get_y() + b.get_height() / 2, f" {c/1e6:.2f}M", va="center", fontweight="bold")
ax.set_title("Parameters per Architecture Block", fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "parameter_budget.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 11 — Detailed Evaluation (Validation 10%) & Testing (10%): Per-Sample, Per-Reciter, Alignment

# %%
@torch.no_grad()
def collect_diagnostics(dataloader, split_label):
    """Decodes a split with the Validation-selected setting (Attention Decoder by default) and returns a per-sample DataFrame, alignment counts,
    and corpus-level (global) WER/CER metrics to eliminate discrepancy between aggregate and per-verse means."""
    model.eval()
    refs, preds, ayahs, recs = [], [], [], []
    for (
        padded_mels,
        mel_lengths,
        _,
        _,
        _,
        _,
        raw_texts,
        ayah_ids,
        reciters,
    ) in dataloader:
        gen = decode_with_best_setting(padded_mels.to(device), mel_lengths.to(device))
        for r, p, a, rec in zip(raw_texts, gen, ayah_ids, reciters):
            refs.append(preprocess_quran_text(r))
            preds.append(tokenizer.decode(p))
            ayahs.append(a)
            recs.append(rec)

    hits = subs = dels = ins = 0
    rows = []
    norm_refs, norm_preds = [], []
    for ref, pred, a, rec in zip(refs, preds, ayahs, recs):
        nr, np_ = normalize_arabic(ref), normalize_arabic(pred)
        norm_refs.append(nr if nr else "")
        norm_preds.append(np_ if np_ else "")
        if nr:
            out = jiwer.process_words(nr, np_ if np_ else "")
            hits += out.hits; subs += out.substitutions; dels += out.deletions; ins += out.insertions
        rows.append({
            "Split": split_label, "Ayah ID": a, "Reciter": rec,
            "Words": len(nr.split()),
            "Norm WER (%)": round(100 * jiwer.wer(nr, np_), 2) if nr and np_ else (100.0 if nr else 0.0),
            "Exact WER (%)": round(100 * jiwer.wer(ref, pred), 2) if pred else 100.0,
            "Norm CER (%)": round(100 * jiwer.cer(nr, np_), 2) if nr and np_ else (100.0 if nr else 0.0),
            "Exact CER (%)": round(100 * jiwer.cer(ref, pred), 2) if pred else 100.0,
            "Reference": ref, "Predicted": pred,
        })
    df = pd.DataFrame(rows)
    # Corpus-level (global aggregate) metrics
    corpus_norm_wer = round(100 * jiwer.wer(norm_refs, norm_preds), 2) if norm_refs else 0.0
    corpus_exact_wer = round(100 * jiwer.wer(refs, preds), 2) if refs else 0.0
    corpus_norm_cer = round(100 * jiwer.cer(norm_refs, norm_preds), 2) if norm_refs else 0.0
    corpus_exact_cer = round(100 * jiwer.cer(refs, preds), 2) if refs else 0.0
    corpus_dict = {
        "Norm WER (%)": corpus_norm_wer,
        "Exact WER (%)": corpus_exact_wer,
        "Norm CER (%)": corpus_norm_cer,
        "Exact CER (%)": corpus_exact_cer,
    }
    return df, (hits, subs, dels, ins), corpus_dict

val_df, val_align, val_corpus = collect_diagnostics(val_loader, "Validation")
test_df, test_align, test_corpus = collect_diagnostics(test_loader, "Test")
all_df = pd.concat([val_df, test_df], ignore_index=True)
metric_cols = ["Norm WER (%)", "Exact WER (%)", "Norm CER (%)", "Exact CER (%)"]

print("=" * 80)
print(f"EVALUATION SUMMARY — {SURAH_NAME_AR} (ID: {SURAH_ID}): Corpus-Level vs Per-Verse Means")
print("=" * 80)
comp_table = pd.DataFrame({
    "Val (Corpus-Level)": pd.Series(val_corpus),
    "Val (Per-Verse Mean)": val_df[metric_cols].mean().round(2),
    "Test (Corpus-Level)": pd.Series(test_corpus),
    "Test (Per-Verse Mean)": test_df[metric_cols].mean().round(2),
})
print(comp_table.to_string())
print("=" * 80)
all_df.to_csv(os.path.join(PROJECT_DIR, "evaluation_predictions.csv"), index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# Panel 1: Validation vs Test
x = np.arange(len(metric_cols)); w = 0.35
b1 = axes[0].bar(x - w / 2, [val_df[c].mean() for c in metric_cols], w, label="Validation (10%)", color="#3b82f6", edgecolor="black")
b2 = axes[0].bar(x + w / 2, [test_df[c].mean() for c in metric_cols], w, label="Test (10%)", color="#10b981", edgecolor="black")
for grp in (b1, b2):
    for b in grp:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%",
                     ha="center", fontsize=8.5, fontweight="bold")
axes[0].set_xticks(x); axes[0].set_xticklabels(["Norm WER", "Exact WER", "Norm CER", "Exact CER"])
axes[0].set_title("1. Validation vs. Test Error Rates", fontweight="bold"); axes[0].legend()
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# Panel 2: per-reciter
rg = all_df.groupby(["Split", "Reciter"])[metric_cols].mean().reset_index()
xr = np.arange(len(rg)); bw = 0.2
for k, (c, col) in enumerate(zip(metric_cols, ["#10b981", "#f59e0b", "#0ea5e9", "#8b5cf6"])):
    axes[1].bar(xr + (k - 1.5) * bw, rg[c], bw, label=c, color=col)
axes[1].set_xticks(xr)
axes[1].set_xticklabels([f"[{s}]\n{r}" for s, r in zip(rg["Split"], rg["Reciter"])], fontsize=8, rotation=20)
axes[1].set_title("2. Per-Reciter Error (Unseen Voices)", fontweight="bold"); axes[1].legend(fontsize=8)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

# Panel 3: word alignment
tot = [sum(v[i] for v in (val_align, test_align)) for i in range(4)]
bars = axes[2].bar(["Correct", "Substitutions", "Deletions", "Insertions"], tot,
                   color=["#16a34a", "#f59e0b", "#dc2626", "#9333ea"], edgecolor="black", width=0.55)
for b, c in zip(bars, tot):
    axes[2].text(b.get_x() + b.get_width() / 2, b.get_height(), str(c), ha="center", va="bottom", fontweight="bold")
axes[2].set_title("3. Word Alignment (Val + Test)", fontweight="bold")
axes[2].grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "evaluation_dashboard.png"), dpi=150, bbox_inches="tight")
plt.show()

# Per-verse distribution, length vs error, and per-ayah heatmap (reciter x ayah, Norm CER)
fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))
axes[0].hist(val_df["Norm WER (%)"], bins=15, alpha=0.65, label="Validation", color="#3b82f6", edgecolor="black")
axes[0].hist(test_df["Norm WER (%)"], bins=15, alpha=0.65, label="Test", color="#10b981", edgecolor="black")
axes[0].set_title("Per-Verse Normalized WER", fontweight="bold"); axes[0].set_xlabel("WER (%)"); axes[0].legend()

axes[1].scatter(val_df["Words"], val_df["Exact CER (%)"], alpha=0.7, label="Validation", color="#2563eb")
axes[1].scatter(test_df["Words"], test_df["Exact CER (%)"], alpha=0.7, marker="^", label="Test", color="#16a34a")
axes[1].set_title("Verse Length vs. Exact CER", fontweight="bold")
axes[1].set_xlabel("Words in verse"); axes[1].set_ylabel("Exact CER (%)"); axes[1].legend()

pivot = all_df.pivot_table(index="Reciter", columns="Ayah ID", values="Norm CER (%)", aggfunc="mean")
im = axes[2].imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
axes[2].set_yticks(range(len(pivot.index))); axes[2].set_yticklabels(pivot.index, fontsize=8)
axes[2].set_xlabel("Ayah ID"); axes[2].set_title("Norm CER (%) Heatmap: Reciter x Ayah", fontweight="bold")
fig.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "verse_level_analysis.png"), dpi=150, bbox_inches="tight")
plt.show()

# Worst & best verses
display(HTML("<h3>Worst 8 verses (Norm CER)</h3>"))
display(all_df.sort_values("Norm CER (%)", ascending=False).head(8)
        .style.set_properties(subset=["Reference", "Predicted"], **{"direction": "rtl", "text-align": "right"}))
display(HTML("<h3>Best 8 verses (Norm CER)</h3>"))
display(all_df.sort_values("Norm CER (%)").head(8)
        .style.set_properties(subset=["Reference", "Predicted"], **{"direction": "rtl", "text-align": "right"}))

# %% [markdown]
# ## Step 12 — Single-Clip Inference with Decoder Confidence & Encoder Output (torchaudio pipeline)

# %%
@torch.no_grad()
def transcribe_with_confidence(
    audio_array,
    max_len=350,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
):
    """Autoregressive decode one clip under FP16 autocast, returning text + per-character softmax confidence.
    `no_repeat_ngram_size` defaults to 0 (disabled in reference Quranic evaluation); `repetition_penalty` scales
    previously emitted content token logits when != 1.0."""
    model.eval()
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        mel = extract_log_mel(audio_array, augment=False).unsqueeze(0).to(device)
        lens = torch.tensor([mel.size(1)], device=device)
        memory, mem_mask = model.encode_audio(mel, lens)
        enc_lens = torch.clamp(model._compute_cnn_lengths(lens), min=1, max=memory.size(1))
        # Realistic vocalized recitation ceiling (~22.5 chars+tashkeel/sec at 50 frames/sec)
        dyn_limit = min(max_len, int(enc_lens.item() * 0.45) + 20)

        gen = torch.full((1, 1), tokenizer.sos_id, dtype=torch.long, device=device)
        confs = []
        for _ in range(dyn_limit):
            mask = model._build_local_causal_mask(gen.size(1), device)
            emb = model.pos_decoder(model.token_embedding(gen) * math.sqrt(model.d_model))
            out = model.transformer_decoder(tgt=emb, memory=memory, tgt_mask=mask, memory_key_padding_mask=mem_mask)
            logits = model.output_head(out[:, -1, :])

            if repetition_penalty != 1.0 and repetition_penalty > 0.0 and gen.size(1) > 1:
                seen_tokens = {
                    t for t in gen[0, 1:].tolist()
                    if t not in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id)
                }
                for tok_id in seen_tokens:
                    val = logits[0, tok_id]
                    logits[0, tok_id] = val * repetition_penalty if val < 0 else val / repetition_penalty

            # Optional local tail long-n-gram loop guard (disabled by default: no_repeat_ngram_size=0)
            if no_repeat_ngram_size > 0 and gen.size(1) >= no_repeat_ngram_size:
                seq = gen[0].tolist()
                prefix = tuple(seq[-(no_repeat_ngram_size - 1) :])
                search_start = max(1, len(seq) - 40)
                for i in range(search_start, len(seq) - no_repeat_ngram_size + 1):
                    if tuple(seq[i : i + no_repeat_ngram_size - 1]) == prefix:
                        banned_tok = seq[i + no_repeat_ngram_size - 1]
                        if banned_tok not in (tokenizer.sos_id, tokenizer.eos_id, tokenizer.pad_id):
                            logits[0, banned_tok] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            conf, nxt = probs.max(dim=-1)
            gen = torch.cat([gen, nxt.unsqueeze(1)], dim=1)
            confs.append(conf.item())
            if nxt.item() == tokenizer.eos_id:
                break
    return tokenizer.decode(gen[0, 1:]), confs, memory.squeeze(0).cpu()

test_sample = test_audio[min(2, len(test_audio) - 1)]
pred_text, confs, enc_out = transcribe_with_confidence(np.asarray(test_sample["audio"]["array"], dtype=np.float32))
ref_text = preprocess_quran_text(test_sample["ayah_ar"])
print("Reference:", ref_text)
print("Predicted:", pred_text)
print(f"Mean char confidence: {np.mean(confs):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(18, 4.8))
axes[0].plot(confs, color="#2563eb", linewidth=1.6)
axes[0].fill_between(range(len(confs)), confs, alpha=0.2, color="#2563eb")
axes[0].axhline(0.5, color="#dc2626", linestyle="--", alpha=0.7)
axes[0].set_ylim(0, 1.02)
axes[0].set_title("Decoder Confidence per Generated Character", fontweight="bold")
axes[0].set_xlabel("Decoding step"); axes[0].set_ylabel("Max softmax probability")
axes[0].grid(True, linestyle="--", alpha=0.5)

im = axes[1].imshow(enc_out.T.numpy(), aspect="auto", origin="lower", cmap="viridis")
axes[1].set_title("Conformer Acoustic Memory (d_model x 20ms frames)", fontweight="bold")
axes[1].set_xlabel("Encoder frame (2x downsampled, 20ms)"); axes[1].set_ylabel("Feature dim")
fig.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "inference_diagnostics.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 12.5 — Audio-Grounding Probe (does the Decoder really listen?)
# Same decoder, five conditions per ayah: (1) original recording, (2) same ayah by a different unseen reciter, (3) a different ayah,
# (4) pure silence, (5) pink noise. If silence/noise/different-ayah still produce the *original* text, the Decoder relies on a text prior instead of audio.

# %%
def _decode_array(eval_model, arr):
    mel = extract_log_mel(np.asarray(arr, dtype=np.float32), augment=False).unsqueeze(0).to(device)
    lens = torch.tensor([mel.size(1)], dtype=torch.long, device=device)
    ids = decode_with_best_setting(mel, lens, dec_model=eval_model)
    return tokenizer.decode(ids[0].cpu().tolist())


def _cer_pct(ref_text, pred_text):
    r, p = normalize_arabic(preprocess_quran_text(ref_text)), normalize_arabic(pred_text)
    if not r:
        return 0.0
    return 100.0 * jiwer.cer(r, p) if p else 100.0


@torch.no_grad()
def evaluate_audio_grounding_probe(eval_model, primary_ds, other_ds_list, n_ayahs=6, seed=SEED):
    eval_model.eval()
    rng_p = random.Random(seed)
    ayahs = list(primary_ds["ayah_id"])
    recs = list(primary_ds[reciter_col])
    uniq = sorted(set(ayahs))
    picks = rng_p.sample(uniq, min(n_ayahs, len(uniq)))
    rows = []
    for a in picks:
        i0 = ayahs.index(a)
        row0 = primary_ds[i0]
        ref_text = row0["ayah_ar"]
        rec0 = recs[i0]
        arr0 = row0["audio"]["array"]

        # (2) same ayah, different reciter
        alt_arr, alt_rec = None, None
        for ds_alt in [primary_ds] + list(other_ds_list):
            a_ids, a_recs = list(ds_alt["ayah_id"]), list(ds_alt[reciter_col])
            for j, (aj, rj) in enumerate(zip(a_ids, a_recs)):
                if aj == a and rj != rec0:
                    alt_arr, alt_rec = ds_alt[j]["audio"]["array"], rj
                    break
            if alt_arr is not None:
                break

        # (3) different ayah
        other_a = rng_p.choice([x for x in uniq if x != a]) if len(uniq) > 1 else a
        j_other = ayahs.index(other_a)
        diff_arr, diff_text = primary_ds[j_other]["audio"]["array"], primary_ds[j_other]["ayah_ar"]

        n_s = int(3.0 * SR)
        silence = np.zeros(n_s, dtype=np.float32) + 1e-4 * np.random.randn(n_s).astype(np.float32)
        pink = (_colored_noise(n_s, "pink") * 0.05).numpy().astype(np.float32)

        conds = [("1. Original recording", arr0, rec0, ref_text),
                 ("3. Different ayah", diff_arr, recs[j_other], diff_text),
                 ("4. Pure silence", silence, "-", None),
                 ("5. Pink noise", pink, "-", None)]
        if alt_arr is not None:
            conds.insert(1, ("2. Same ayah, other reciter", alt_arr, alt_rec, ref_text))
        for name, arr, rec, played_text in conds:
            pred = _decode_array(eval_model, arr)
            rows.append({
                "Ayah": a, "Condition": name, "Source Reciter": rec,
                "CER vs ORIGINAL ayah text (%)": round(_cer_pct(ref_text, pred), 2),
                "CER vs PLAYED audio text (%)": round(_cer_pct(played_text, pred), 2) if played_text else None,
                "Pred chars": len(pred), "Prediction": pred[:70],
            })
    df = pd.DataFrame(rows)
    summary = df.groupby("Condition")[["CER vs ORIGINAL ayah text (%)", "CER vs PLAYED audio text (%)", "Pred chars"]].mean().round(2)
    return df, summary


grounding_df, grounding_summary = evaluate_audio_grounding_probe(model, test_audio, [eval_audio, train_audio])
print("=" * 100)
print("AUDIO-GROUNDING PROBE (mean over probed ayahs):")
print(grounding_summary.to_string())
print("-" * 100)
print("Interpretation: Silence / Noise / Different-ayah should have HIGH 'CER vs ORIGINAL text'. A LOW value there means a strong text prior.")
print("=" * 100)
grounding_df.to_csv(os.path.join(PROJECT_DIR, "audio_grounding_probe.csv"), index=False, encoding="utf-8-sig")

# %% [markdown]
# ## Step 13 — Full Surah Transcription for One Unseen Reciter (Verse-by-Verse)

# %%
target_reciter = sorted(test_reciters)[0]
subset = test_audio.filter(lambda x: x[reciter_col] == target_reciter).sort("ayah_id")
print(f"Reciter: {target_reciter} | {len(subset)} verses\n" + "=" * 60)
cers = []
for row in subset:
    p, _, _ = transcribe_with_confidence(np.asarray(row["audio"]["array"], dtype=np.float32), max_len=300)
    r = normalize_arabic(preprocess_quran_text(row["ayah_ar"]))
    cers.append(100 * jiwer.cer(r, normalize_arabic(p)) if r and p else 100.0)
    print(f"[{row['ayah_id']}] {p}")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(subset["ayah_id"], cers, color="#6366f1", edgecolor="black")
ax.set_title(f"Per-Verse Norm CER — {target_reciter} ({shape_ar(SURAH_NAME_AR)})", fontweight="bold")
ax.set_xlabel("Ayah ID"); ax.set_ylabel("CER (%)"); ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "full_surah_reciter_cer.png"), dpi=150, bbox_inches="tight")
plt.show()
print("All plots saved to:", PLOTS_DIR)

# %% [markdown]
# ## Step 14 — Transcribe Full Surah Audio Verse-by-Verse for محمود خليل الحصري

# %%
reciter1_id = next((k for k, v in RECITER_ARABIC_MAP.items() if v == "محمود خليل الحصري"), "husary")
reciter1_ds = raw_dataset.cast_column("audio", Audio(decode=False))
reciter1_ds = reciter1_ds.filter(lambda x: x[reciter_col] == reciter1_id).sort("ayah_id")

# Check reciter split to ensure transparent evaluation (seen vs unseen voice)
reciter1_split = "Train (Seen voice)" if reciter1_id in train_reciters else ("Validation" if reciter1_id in eval_reciters else "Test (Held-Out / Unseen voice)")

model.eval()
print(f"\nTranscribed Recitation ({SURAH_NAME_AR} - محمود خليل الحصري):")
print(f"Split Verification: '{reciter1_id}' belongs to [{reciter1_split}]")
if "Train" in reciter1_split:
    print("  -> Acoustic Fitting / Memorization Benchmark on Seen Reciter Voice.")
else:
    print("  -> Generalization Benchmark on Unseen Reciter Voice.")
print("=" * 60)

for row in reciter1_ds:
    audio_dict = row["audio"]
    if audio_dict and "bytes" in audio_dict:
        try:
            with io.BytesIO(audio_dict["bytes"]) as f:
                waveform, sample_rate = torchaudio.load(f)

            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != 16000:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            array = waveform.squeeze(0).numpy()
            mel_tensor = extract_log_mel(array, sr=16000, n_mels=80, augment=False).unsqueeze(0).to(device)
            mel_lens = torch.tensor([mel_tensor.size(1)], dtype=torch.long, device=device)

            with torch.no_grad():
                gen_ids = decode_with_best_setting(mel_tensor, mel_lens)

            prediction = tokenizer.decode(gen_ids[0].cpu().tolist())
            print(f"[{row['ayah_id']}] {prediction}")

        except Exception as e:
            print(f"[Error in Ayah {row['ayah_id']}] {e}")

print("=" * 60)
print("محمود خليل الحصري Transcription Complete.")

# %% [markdown]
# ## Step 15 — Transcribe Full Surah Audio Verse-by-Verse for محمد جبريل

# %%
reciter2_id = next((k for k, v in RECITER_ARABIC_MAP.items() if v == "محمد جبريل"), "\x6d\x75\x68\x61\x6d\x6d\x61\x64_jibreel")
reciter2_ds = raw_dataset.cast_column("audio", Audio(decode=False))
reciter2_ds = reciter2_ds.filter(lambda x: x[reciter_col] == reciter2_id).sort("ayah_id")

# Check reciter split to ensure transparent evaluation (seen vs unseen voice)
reciter2_split = "Train (Seen voice)" if reciter2_id in train_reciters else ("Validation" if reciter2_id in eval_reciters else "Test (Held-Out / Unseen voice)")

model.eval()
print(f"\nTranscribed Recitation ({SURAH_NAME_AR} - محمد جبريل):")
print(f"Split Verification: '{reciter2_id}' belongs to [{reciter2_split}]")
if "Train" in reciter2_split:
    print("  -> Acoustic Fitting / Memorization Benchmark on Seen Reciter Voice.")
else:
    print("  -> Generalization Benchmark on Unseen Reciter Voice.")
print("=" * 60)

for row in reciter2_ds:
    audio_dict = row["audio"]
    if audio_dict and "bytes" in audio_dict:
        try:
            with io.BytesIO(audio_dict["bytes"]) as f:
                waveform, sample_rate = torchaudio.load(f)

            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != 16000:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            array = waveform.squeeze(0).numpy()
            mel_tensor = extract_log_mel(array, sr=16000, n_mels=80, augment=False).unsqueeze(0).to(device)
            mel_lens = torch.tensor([mel_tensor.size(1)], dtype=torch.long, device=device)

            with torch.no_grad():
                gen_ids = decode_with_best_setting(mel_tensor, mel_lens)

            prediction = tokenizer.decode(gen_ids[0].cpu().tolist())
            print(f"[{row['ayah_id']}] {prediction}")

        except Exception as e:
            print(f"[Error in Ayah {row['ayah_id']}] {e}")

print("=" * 60)
print("محمد جبريل Transcription Complete.")
