#!/usr/bin/env python3
"""
Image-C3 9x evaluation script

Compares Image-C3 (9x) vs LowRes (9x) vs Teacher (uncompressed upper bound)

Supported methods:
- image_c3: Image-C3 9x compression (uses generate_with_rope_3d, 3D M-RoPE)
- lowres: reduced-resolution baseline (1/3 scale)
- teacher: original Qwen2.5-VL-7B-Instruct (performance upper bound)

Datasets:
- MMMU, MMBench, ChartQA, DocVQA, TextVQA
- RealWorldQA, MathVision, OCRBench

Usage:
    # Single-method evaluation (good for multi-GPU parallel runs)
    python scripts/eval_9x_quick.py \
        --methods image_c3 \
        --datasets mmmu mmbench chartqa docvqa textvqa realworldqa mathvision ocrbench \
        --n 0 \
        --output_dir eval_results/9x_3drope

    # Full comparison
    python scripts/eval_9x_quick.py \
        --methods image_c3 lowres teacher \
        --datasets mmmu mmbench chartqa docvqa textvqa \
        --n 0
"""

import os
import sys
import json
import argparse
import random
import re
import time
import gc
from pathlib import Path
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration"""

    # Model path
    decoder_path: str = "./Qwen/Qwen2.5-VL-7B-Instruct"

    # Image-C3 9x checkpoint (3D RoPE, Stage 2)
    image_c3_ckpt: str = "./outputs/image_c3_v7_9x_3drope_20260301_103916/stage2/checkpoint-best"

    # Fairness parameters
    max_new_tokens: int = 128
    caption_max_new_tokens: int = 64  # Caption tasks use shorter generation (lmms-eval standard)
    do_sample: bool = False
    seed: int = 42

    # Compression block size: 3 = 9x (3x3), 4 = 16x (4x4)
    block_size: int = 3

    # LowRes scale factor: 9x -> 1/3 ~= 0.333, 16x -> 1/4 = 0.25
    lowres_scale: float = 0.333

    # Max image pixel cap (avoid OOM on large images)
    # Qwen2.5-VL default max_pixels=16384*28*28~=12.8M; large document images can OOM
    # 1280*28*28=1,003,520 ~= 1M pixels, ~1280 visual tokens, safe on an 80GB GPU
    max_pixels: int = 1280 * 28 * 28
    min_pixels: int = 4 * 28 * 28

    # Data path
    data_root: str = "./data/eval"


# =============================================================================
# Prompt templates
# =============================================================================

# Multiple-choice prompt
MC_PROMPT_SUFFIX = "\nAnswer with the option letter only."

# ChartQA prompt
CHARTQA_PROMPT = "Look at the chart and answer: {question}\nAnswer with a short phrase or number."

# DocVQA prompt
DOCVQA_PROMPT = "Based on the document, answer: {question}\nAnswer concisely."

# TextVQA prompt
TEXTVQA_PROMPT = "Read the text in the image and answer: {question}\nAnswer with a word or phrase."

# VQAv2 prompt (standard VQA format)
VQAV2_PROMPT = "{question}\nAnswer the question using a single word or phrase."

# COCO Caption prompt (lmms-eval / InternVL standard)
CAPTION_PROMPT = "Provide a one-sentence caption for the provided image."

# TextCaps prompt (matches dataset original)
TEXTCAPS_PROMPT = "Please carefully observe the image and come up with a caption for the image."

# POPE prompt (yes/no, already in data)
POPE_PROMPT_SUFFIX = "\nAnswer with yes or no."


def format_mc_question(question: str, options: List[str]) -> str:
    """Format a multiple-choice question"""
    if options:
        opts_str = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        return f"{question}\n{opts_str}{MC_PROMPT_SUFFIX}"
    return question + MC_PROMPT_SUFFIX


# =============================================================================
# Data loading
# =============================================================================

def load_mmmu(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load MMMU data"""
    import pyarrow.parquet as pq

    mmmu_root = Path(data_root) / "mmmu"
    samples = []

    for subject_dir in mmmu_root.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith("."):
            continue

        for pq_file in subject_dir.glob("validation*.parquet"):
            try:
                table = pq.read_table(pq_file)
                df = table.to_pandas()

                for _, row in df.iterrows():
                    image = None
                    for img_key in ["image", "image_1", "image_2", "image_3"]:
                        img_data = row.get(img_key)
                        if img_data is not None and isinstance(img_data, dict) and "bytes" in img_data:
                            try:
                                image = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                                break
                            except:
                                continue

                    if image is None:
                        continue

                    options_raw = row.get("options", [])
                    if isinstance(options_raw, str):
                        try:
                            options_raw = eval(options_raw)
                        except:
                            options_raw = []

                    options = [str(o) for o in options_raw if o] if isinstance(options_raw, list) else []
                    question = str(row.get("question", ""))
                    gt = str(row.get("answer", "")).strip().upper()

                    samples.append({
                        "id": str(row.get("id", len(samples))),
                        "image": image,
                        "question": question,
                        "options": options,
                        "gt": gt,
                        "task_type": "mc",
                        "dataset": "mmmu",
                    })
            except Exception as e:
                continue

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)

    print(f"  [MMMU] Loaded {len(samples)} samples")
    return samples


def load_mmbench(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load MMBench data"""
    import pyarrow.parquet as pq
    import pandas as pd

    pq_path = Path(data_root) / "mmbench/en/dev-00000-of-00001.parquet"

    if not pq_path.exists():
        print(f"  [MMBench] File not found: {pq_path}")
        return []

    table = pq.read_table(pq_path)
    df = table.to_pandas()

    samples = []
    for idx, row in df.iterrows():
        img_data = row.get("image")
        if not isinstance(img_data, dict) or "bytes" not in img_data:
            continue
        try:
            image = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
        except:
            continue

        options = []
        for key in ["A", "B", "C", "D"]:
            val = row.get(key)
            if val is not None and pd.notna(val) and str(val).strip():
                options.append(str(val).strip())

        question = str(row.get("question", ""))
        gt = str(row.get("answer", "")).strip().upper()

        samples.append({
            "id": str(row.get("index", idx)),
            "image": image,
            "question": question,
            "options": options,
            "gt": gt,
            "task_type": "mc",
            "dataset": "mmbench",
        })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)

    print(f"  [MMBench] Loaded {len(samples)} samples")
    return samples


def load_chartqa(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load ChartQA data"""
    import pyarrow.parquet as pq

    pq_path = Path(data_root) / "chartqa/data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"
    if not pq_path.exists():
        pq_path = Path(data_root) / "chartqa/data/val-00000-of-00001-0f11003c77497969.parquet"

    if not pq_path.exists():
        print(f"  [ChartQA] File not found")
        return []

    table = pq.read_table(pq_path)
    df = table.to_pandas()

    samples = []
    for idx, row in df.iterrows():
        img_data = row.get("image")
        if not isinstance(img_data, dict) or "bytes" not in img_data:
            continue
        try:
            image = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
        except:
            continue

        question = str(row.get("query", row.get("question", "")))
        raw_label = row.get("label", row.get("answer", ""))
        if hasattr(raw_label, '__iter__') and not isinstance(raw_label, str):
            gt = str(raw_label[0]) if len(raw_label) > 0 else ""
        else:
            gt = str(raw_label)

        samples.append({
            "id": str(idx),
            "image": image,
            "question": question,
            "gt": gt,
            "task_type": "chartqa",
            "dataset": "chartqa",
        })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)

    print(f"  [ChartQA] Loaded {len(samples)} samples")
    return samples


def load_docvqa(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load DocVQA data"""
    import pyarrow.parquet as pq

    docvqa_root = Path(data_root) / "docvqa/DocVQA"
    samples = []

    # Prefer the validation split
    for pq_file in sorted(docvqa_root.glob("validation*.parquet")):
        try:
            table = pq.read_table(pq_file)
            df = table.to_pandas()

            for idx, row in df.iterrows():
                img_data = row.get("image")
                if img_data is None:
                    continue

                # Handle different image formats
                try:
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        image = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                    elif hasattr(img_data, 'tobytes'):
                        image = img_data.convert("RGB")
                    else:
                        continue
                except:
                    continue

                question = str(row.get("question", ""))

                # answers may be a list or a numpy ndarray
                answers = row.get("answers", row.get("answer", ""))
                if hasattr(answers, 'tolist'):
                    answers = answers.tolist()
                if isinstance(answers, list):
                    gt = answers[0] if answers else ""
                    all_answers = [str(a) for a in answers]
                else:
                    gt = str(answers)
                    all_answers = [gt]

                samples.append({
                    "id": str(row.get("questionId", f"docvqa_{len(samples)}")),
                    "image": image,
                    "question": question,
                    "gt": gt,
                    "all_answers": all_answers,
                    "task_type": "docvqa",
                    "dataset": "docvqa",
                })
        except Exception as e:
            print(f"  [DocVQA] Error loading {pq_file}: {e}")
            continue

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)

    print(f"  [DocVQA] Loaded {len(samples)} samples")
    return samples


def load_textvqa(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load TextVQA data"""
    import pyarrow.parquet as pq

    textvqa_root = Path(data_root) / "textvqa/data"
    samples = []

    # Prefer the validation split
    for pq_file in sorted(textvqa_root.glob("validation*.parquet")):
        try:
            table = pq.read_table(pq_file)
            df = table.to_pandas()

            for idx, row in df.iterrows():
                img_data = row.get("image")
                if img_data is None:
                    continue

                try:
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        image = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                    elif hasattr(img_data, 'tobytes'):
                        image = img_data.convert("RGB")
                    else:
                        continue
                except:
                    continue

                question = str(row.get("question", ""))

                answers = row.get("answers", row.get("answer", ""))
                # Convert numpy ndarray to list
                if hasattr(answers, 'tolist'):
                    answers = answers.tolist()
                if isinstance(answers, list):
                    gt = answers[0] if answers else ""
                    all_answers = [str(a) for a in answers]
                else:
                    gt = str(answers)
                    all_answers = [gt]

                samples.append({
                    "id": str(row.get("question_id", f"textvqa_{len(samples)}")),
                    "image": image,
                    "question": question,
                    "gt": gt,
                    "all_answers": all_answers,
                    "task_type": "textvqa",
                    "dataset": "textvqa",
                })
        except Exception as e:
            print(f"  [TextVQA] Error loading {pq_file}: {e}")
            continue

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)

    print(f"  [TextVQA] Loaded {len(samples)} samples")
    return samples


def load_realworldqa(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load RealWorldQA (TSV, base64 images, multiple-choice)"""
    import base64, csv
    csv.field_size_limit(100 * 1024 * 1024)
    tsv_path = Path(data_root) / "realworldqa" / "RealWorldQA.tsv"
    img_dir = Path(data_root) / "vlmeval_images" / "RealWorldQA"
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = row.get("index", str(len(samples)))
            img_path = img_dir / f"{idx}.jpg"
            try:
                if img_path.exists():
                    image = Image.open(img_path).convert("RGB")
                elif row.get("image"):
                    image = Image.open(BytesIO(base64.b64decode(row["image"]))).convert("RGB")
                else:
                    continue
            except:
                continue
            question = row.get("question", "")
            choices = {}
            for key in ["A", "B", "C", "D"]:
                if row.get(key):
                    choices[key] = row[key]
            if choices:
                choice_text = "\n".join([f"{k}. {v}" for k, v in choices.items()])
                question = f"{question}\n{choice_text}\nAnswer with the option letter."
            gt = row.get("answer", "")
            samples.append({
                "id": str(idx), "image": image, "question": question,
                "gt": str(gt), "all_answers": None,
                "task_type": "mc", "dataset": "realworldqa",
            })
    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    print(f"  [RealWorldQA] Loaded {len(samples)} samples")
    return samples


def load_mathvision(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load MathVision MINI (TSV, base64 images, mixed MC + QA)"""
    import base64, csv
    csv.field_size_limit(100 * 1024 * 1024)
    tsv_path = Path(data_root) / "mathvision" / "MathVision_MINI.tsv"
    img_dir = Path(data_root) / "vlmeval_images" / "MathVision_MINI"
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = row.get("index", str(len(samples)))
            img_path = img_dir / f"{idx}.jpg"
            try:
                if img_path.exists():
                    image = Image.open(img_path).convert("RGB")
                elif row.get("image"):
                    image = Image.open(BytesIO(base64.b64decode(row["image"]))).convert("RGB")
                else:
                    continue
            except:
                continue
            question = row.get("question", "")
            choices_str = row.get("choices", "[]")
            try:
                choices = eval(choices_str) if choices_str and choices_str != "[]" else []
            except:
                choices = []
            if choices:
                labels = "ABCDEFGH"
                choice_text = "\n".join([f"{labels[i]}. {c}" for i, c in enumerate(choices)])
                question = f"{question}\n{choice_text}\nAnswer with the option letter."
                task_type = "mc"
            else:
                task_type = "qa"
            gt = str(row.get("answer", ""))
            samples.append({
                "id": str(idx), "image": image, "question": question,
                "gt": gt, "all_answers": None,
                "task_type": task_type, "dataset": "mathvision",
            })
    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    print(f"  [MathVision] Loaded {len(samples)} samples")
    return samples


def load_ocrbench(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load OCRBench (parquet)"""
    import pyarrow.parquet as pq
    import numpy as np
    pq_path = Path(data_root) / "ocrbench" / "data" / "test-00000-of-00001.parquet"
    samples = []
    try:
        df = pq.read_table(pq_path).to_pandas()
        for _, row in df.iterrows():
            img_data = row.get("image")
            if img_data is None:
                continue
            try:
                if isinstance(img_data, dict) and "bytes" in img_data:
                    image = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                elif isinstance(img_data, bytes):
                    image = Image.open(BytesIO(img_data)).convert("RGB")
                else:
                    continue
            except:
                continue
            answers = row.get("answer", [])
            if isinstance(answers, np.ndarray):
                answers = answers.tolist()
            if isinstance(answers, str):
                answers = [answers]
            if not isinstance(answers, list):
                answers = [answers] if answers else []
            gt = answers[0] if len(answers) > 0 else ""
            samples.append({
                "id": str(len(samples)), "image": image,
                "question": str(row.get("question", "")),
                "gt": str(gt),
                "all_answers": [str(a) for a in answers] if len(answers) > 0 else None,
                "task_type": "qa", "dataset": "ocrbench",
            })
    except Exception as e:
        print(f"  [OCRBench] Error loading: {e}")
    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    print(f"  [OCRBench] Loaded {len(samples)} samples")
    return samples


def load_coco(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load COCO Caption (val2017, 5000 images)"""
    jsonl_path = Path(data_root) / "coco" / "extracted" / "coco_val2017.jsonl"
    img_base = Path(data_root) / "coco" / "extracted"
    samples = []

    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            # Try several candidate paths
            img_path = img_base / "val2017" / Path(d["image"]).name
            if not img_path.exists():
                img_path = Path(data_root).parent.parent / d["image"]
            if not img_path.exists():
                img_path = Path(d["image"])
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue
            samples.append({
                "id": d.get("image_id", len(samples)),
                "image": image,
                "question": CAPTION_PROMPT,
                "gt": str(d.get("caption", "")),
                "all_captions": d.get("all_captions", []),
                "task_type": "caption",
                "dataset": "coco",
            })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    print(f"  [COCO] Loaded {len(samples)} samples")
    return samples


def load_mathvista(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load MathVista testmini (1000 questions, mixed MC + free_form)

    Data source: AI4Math/MathVista, saved with datasets.save_to_disk
    question_type: multi_choice (540) / free_form (460)
    answer_type:   text (MC) / integer / float / list
    The `query` field already carries a formatted prompt, used directly as-is.
    """
    from datasets import load_from_disk

    ds_path = Path(data_root) / "mathvista" / "testmini"
    if not ds_path.exists():
        print(f"  [MathVista] Path not found: {ds_path}")
        return []

    ds = load_from_disk(str(ds_path))
    samples = []

    for row in ds:
        # Image
        image = row.get("decoded_image")
        if image is None:
            continue
        if not isinstance(image, Image.Image):
            continue
        image = image.convert("RGB")

        pid = str(row["pid"])
        q_type = row.get("question_type", "free_form")
        answer = str(row.get("answer", ""))

        if q_type == "multi_choice":
            choices = row.get("choices") or []
            # Convert gt to an option letter (A/B/C/D)
            gt_letter = ""
            for i, c in enumerate(choices):
                if str(c) == answer:
                    gt_letter = chr(65 + i)
                    break
            samples.append({
                "id": pid,
                "image": image,
                "question": row.get("query", row["question"]),
                "gt": gt_letter,
                "options": choices,
                "task_type": "mc",
                "dataset": "mathvista",
            })
        else:
            # free_form: integer / float / list
            samples.append({
                "id": pid,
                "image": image,
                "question": row.get("query", row["question"]),
                "gt": answer,
                "all_answers": None,
                "task_type": "qa",
                "dataset": "mathvista",
            })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    mc_count = sum(1 for s in samples if s["task_type"] == "mc")
    qa_count = len(samples) - mc_count
    print(f"  [MathVista] Loaded {len(samples)} samples (MC={mc_count}, QA={qa_count})")
    return samples


def load_textcaps(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load TextCaps val (3166 samples, captioning images with text)"""
    jsonl_path = Path(data_root) / "textcaps" / "textcaps_val.jsonl"
    if not jsonl_path.exists():
        print(f"  [TextCaps] File not found: {jsonl_path}")
        return []

    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            img_path = Path(d["image_path"])
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue
            samples.append({
                "id": d["id"],
                "image": image,
                "question": TEXTCAPS_PROMPT,
                "gt": d.get("caption", ""),
                "all_captions": d.get("all_captions", []),
                "task_type": "caption",
                "dataset": "textcaps",
            })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    print(f"  [TextCaps] Loaded {len(samples)} samples")
    return samples


def load_flickr30k(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load Flickr30k Karpathy test split (~1000 images, 5 captions each)"""
    jsonl_path = Path(data_root) / "flickr30k" / "flickr30k_test.jsonl"
    if not jsonl_path.exists():
        print(f"  [Flickr30k] File not found: {jsonl_path}")
        return []

    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            img_path = Path(d["image_path"])
            if not img_path.is_absolute():
                img_path = PROJECT_ROOT / img_path
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue
            samples.append({
                "id": d["id"],
                "image": image,
                "question": CAPTION_PROMPT,
                "gt": d.get("caption", ""),
                "all_captions": d.get("all_captions", []),
                "task_type": "caption",
                "dataset": "flickr30k",
            })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    print(f"  [Flickr30k] Loaded {len(samples)} samples")
    return samples


def load_nocaps(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load NoCaps validation split (~4500 images, ~11 captions each)"""
    jsonl_path = Path(data_root) / "nocaps" / "nocaps_val.jsonl"
    if not jsonl_path.exists():
        print(f"  [NoCaps] File not found: {jsonl_path}")
        return []

    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            img_path = Path(d["image_path"])
            if not img_path.is_absolute():
                img_path = PROJECT_ROOT / img_path
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue
            samples.append({
                "id": d["id"],
                "image": image,
                "question": CAPTION_PROMPT,
                "gt": d.get("caption", ""),
                "all_captions": d.get("all_captions", []),
                "task_type": "caption",
                "dataset": "nocaps",
            })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    print(f"  [NoCaps] Loaded {len(samples)} samples")
    return samples


def load_pope(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load POPE (3 splits x 3000 questions, yes/no hallucination eval)"""
    pope_dir = Path(data_root) / "pope"
    img_dir = pope_dir / "val2014"
    samples = []

    for split in ["random", "popular", "adversarial"]:
        json_path = pope_dir / f"coco_pope_{split}.json"
        if not json_path.exists():
            print(f"  [POPE] File not found: {json_path}")
            continue
        with open(json_path, "r") as f:
            for line in f:
                d = json.loads(line.strip())
                img_path = img_dir / d["image"]
                if not img_path.exists():
                    continue
                try:
                    image = Image.open(img_path).convert("RGB")
                except:
                    continue
                samples.append({
                    "id": f"{split}_{d['question_id']}",
                    "image": image,
                    "question": d["text"] + POPE_PROMPT_SUFFIX,
                    "gt": d["label"],  # "yes" or "no"
                    "task_type": "pope",
                    "dataset": "pope",
                    "pope_split": split,
                })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    split_counts = {}
    for s in samples:
        sp = s["pope_split"]
        split_counts[sp] = split_counts.get(sp, 0) + 1
    print(f"  [POPE] Loaded {len(samples)} samples ({split_counts})")
    return samples


def load_vqav2(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load VQAv2 validation (standard VQA accuracy: min(#match/3, 1))

    VQAv2 has 214K entries; we read the metadata first and pre-sample before
    loading images to avoid OOM / slow startup.
    """
    jsonl_path = Path(data_root) / "vqav2" / "vqav2_val.jsonl"
    if not jsonl_path.exists():
        print(f"  [VQAv2] File not found: {jsonl_path}")
        return []

    # Step 1: Read all metadata (fast, no image loading)
    all_entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            all_entries.append(json.loads(line.strip()))

    # Step 2: Pre-sample before loading images (VQAv2 has 214K entries)
    random.seed(seed)
    if n and len(all_entries) > n:
        all_entries = random.sample(all_entries, n)

    # Step 3: Load images only for sampled entries
    samples = []
    for d in all_entries:
        img_path = d["image"]
        if not Path(img_path).exists():
            fname = Path(d["image"]).name
            img_path = str(Path(data_root) / "vqav2" / "images" / fname)
        if not Path(img_path).exists():
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        all_answers = [a["answer"] for a in d.get("answers", [])]
        gt_entry = d.get("answer", {})
        gt = gt_entry.get("answer", all_answers[0] if all_answers else "")

        samples.append({
            "id": str(d["id"]),
            "image": image,
            "question": d["question"],
            "gt": gt,
            "all_answers": all_answers,
            "task_type": "vqav2",
            "dataset": "vqav2",
        })

    print(f"  [VQAv2] Loaded {len(samples)} samples")
    return samples


DATASET_LOADERS = {
    "vqav2": load_vqav2,
    "mmmu": load_mmmu,
    "mmbench": load_mmbench,
    "chartqa": load_chartqa,
    "docvqa": load_docvqa,
    "textvqa": load_textvqa,
    "realworldqa": load_realworldqa,
    "mathvision": load_mathvision,
    "ocrbench": load_ocrbench,
    "coco": load_coco,
    "mathvista": load_mathvista,
    "textcaps": load_textcaps,
    "pope": load_pope,
    "flickr30k": load_flickr30k,
    "nocaps": load_nocaps,
}


# =============================================================================
# Scoring functions (aligned with official evaluation protocols)
# =============================================================================

# --- EvalAIAnswerProcessor (official TextVQA/VQAv2 normalization) ---
# Source: https://github.com/facebookresearch/mmf/blob/main/pythia/tasks/processors.py
# Used by lmms-eval for TextVQA and VQAv2

class EvalAIAnswerProcessor:
    """Answer normalization following Eval AI / VQA official protocol."""

    CONTRACTIONS = {
        "aint": "ain't", "arent": "aren't", "cant": "can't",
        "couldve": "could've", "couldnt": "couldn't",
        "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
        "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't",
        "hed": "he'd", "hes": "he's", "howd": "how'd", "hows": "how's",
        "Id've": "I'd've", "Im": "I'm", "Ive": "I've",
        "isnt": "isn't", "itd": "it'd", "itll": "it'll",
        "let's": "let's", "maam": "ma'am",
        "mightnt": "mightn't", "mightve": "might've",
        "mustnt": "mustn't", "mustve": "must've",
        "neednt": "needn't", "oclock": "o'clock",
        "oughtnt": "oughtn't", "shant": "shan't",
        "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
        "thats": "that's", "thered": "there'd", "therere": "there're",
        "theres": "there's", "theyd": "they'd", "theyll": "they'll",
        "theyre": "they're", "theyve": "they've",
        "wasnt": "wasn't", "weve": "we've", "werent": "weren't",
        "whatll": "what'll", "whatre": "what're", "whats": "what's",
        "whatve": "what've", "whens": "when's", "whered": "where'd",
        "wheres": "where's", "whod": "who'd", "wholl": "who'll",
        "whos": "who's", "whove": "who've", "whyll": "why'll",
        "whyre": "why're", "whys": "why's", "wont": "won't",
        "wouldve": "would've", "wouldnt": "wouldn't",
        "yall": "y'all", "youd": "you'd", "youll": "you'll",
        "youre": "you're", "youve": "you've",
    }
    NUMBER_MAP = {
        "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+",
        "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!",
    ]

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        for word in in_text.lower().split():
            word = self.NUMBER_MAP.get(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
        for i, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[i] = self.CONTRACTIONS[word]
        return " ".join(out_text)

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


_eval_ai_processor = EvalAIAnswerProcessor()


# --- MC scoring (unchanged) ---

def extract_mc_answer(text: str, num_options: int = 4) -> Optional[str]:
    """Extract the answer letter from a multiple-choice response"""
    if not text:
        return None

    text = text.strip().upper()
    valid_letters = [chr(65 + i) for i in range(num_options)]

    if len(text) == 1 and text in valid_letters:
        return text

    patterns = [
        r"^([A-H])\s*[\.:\)]",
        r"^\(([A-H])\)",
        r"^([A-H])$",
        r"answer\s*[is:]*\s*\(?([A-H])\)?",
        r"^the\s+answer\s+is\s+([A-H])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).upper() in valid_letters:
            return match.group(1).upper()

    for char in text:
        if char in valid_letters:
            return char

    return None


def score_mc(pred: str, gt: str, num_options: int = 4) -> bool:
    """Scoring: multiple-choice"""
    pred_letter = extract_mc_answer(pred, num_options)
    gt_letter = gt.strip().upper()[:1] if gt else ""
    return pred_letter == gt_letter


# --- Simple text normalization (for MathVista QA fallback) ---

def _normalize_text(text: str) -> str:
    """Normalize text: strip punctuation, collapse whitespace, lowercase"""
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# --- MathVista answer extraction (rule-based, no GPT) ---

def _extract_short_answer(pred: str) -> str:
    """Extract a concise answer from a (possibly verbose) model response.

    Follows MathVista official-style extraction (rule-based, no GPT):
      1. Pattern matching: "Answer: X", "answer is X", "= X", "Final value: X"
      2. LaTeX \\boxed{X}
      3. Last number fallback
    Returns extracted answer, or original pred if nothing found.
    """
    if not pred or not pred.strip():
        return pred

    text = pred.strip()

    if len(text) < 20:
        return text

    patterns = [
        r'[Aa]nswer\s*(?:is|=|:)\s*["\']?([^"\'\n,;]+)',
        r'[Ff]inal\s+(?:value|answer|result)\s*(?:is|=|:)\s*["\']?([^"\'\n,;]+)',
        r'(?:^|\n)\s*=\s*(.+)',
        r'\\boxed\{([^}]+)\}',
        r'(?:Therefore|Thus|So|Hence),?\s+(?:the answer is\s+)?(.+?)\.?\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1).strip().rstrip('.')
            if candidate:
                return candidate

    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return text


# --- MathVista QA scoring (used for mathvista free_form and mathvision QA) ---

def score_qa(pred: str, gt: str, all_answers: List[str] = None) -> Tuple[bool, bool]:
    """Scoring: MathVista/MathVision QA (free_form)

    First extract a concise answer from the (possibly verbose) response,
    then match it.
    Returns: (exact_match, relaxed_match)
    """
    candidates = [pred]
    extracted = _extract_short_answer(pred)
    if extracted != pred:
        candidates.append(extracted)

    for candidate in candidates:
        exact, relaxed = _score_qa_inner(candidate, gt, all_answers)
        if exact:
            return True, True
        if relaxed:
            return False, True

    return False, False


def _score_qa_inner(pred: str, gt: str, all_answers: List[str] = None) -> Tuple[bool, bool]:
    """Core QA scoring: normalized exact match + 5% numeric tolerance."""
    pred_norm = _normalize_text(pred)
    gt_norm = _normalize_text(gt)

    if pred_norm == gt_norm:
        return True, True

    if all_answers:
        for ans in all_answers:
            if pred_norm == _normalize_text(str(ans)):
                return True, True

    try:
        pred_num = float(re.sub(r'[^\d.\-]', '', pred.strip()))
        gt_num = float(re.sub(r'[^\d.\-]', '', gt.strip()))
        if gt_num == 0:
            relaxed = abs(pred_num) < 0.05
        else:
            relaxed = abs(pred_num - gt_num) / abs(gt_num) <= 0.05
        if relaxed:
            return False, True
    except:
        pass

    return False, False


# --- DocVQA: ANLS (Average Normalized Levenshtein Similarity) ---
# Official metric: https://arxiv.org/abs/1907.00490
# Implementation aligned with lmms-eval (EffiVLM-Bench/lmms-eval/lmms_eval/api/metrics.py)

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def score_anls(pred: str, gt_answers: List[str], threshold: float = 0.5) -> float:
    """ANLS scoring for DocVQA (official protocol).

    For each GT answer, compute NLS = 1 - edit_distance/max_len.
    Return max NLS across all GT answers (0 if below threshold).
    """
    if not pred:
        return 0.0
    pred_clean = " ".join(pred.strip().lower().split())
    values = []
    for answer in gt_answers:
        gt_clean = " ".join(answer.strip().lower().split())
        dist = _levenshtein_distance(gt_clean, pred_clean)
        length = max(len(gt_clean), len(pred_clean))
        nls = 0.0 if length == 0 else 1.0 - float(dist) / float(length)
        values.append(nls)
    score = max(values) if values else 0.0
    return score if score >= threshold else 0.0


# --- TextVQA: EvalAI VQA Accuracy ---
# Official: EvalAIAnswerProcessor normalization + min(matches/3, 1) averaged across GT
# Source: lmms-eval/lmms_eval/tasks/textvqa/utils.py

def score_textvqa(pred: str, gt_answers: List[str]) -> float:
    """TextVQA official scoring: EvalAI normalization + VQA accuracy.

    Returns accuracy in [0, 1].
    """
    if not gt_answers:
        return 0.0
    pred_norm = _eval_ai_processor(pred)
    gt_norms = [_eval_ai_processor(str(a)) for a in gt_answers]

    accs = []
    for i in range(len(gt_norms)):
        others = [gt_norms[j] for j in range(len(gt_norms)) if j != i]
        match_count = sum(1 for a in others if a == pred_norm)
        accs.append(min(1.0, float(match_count) / 3.0))
    return sum(accs) / len(accs) if accs else 0.0


# --- VQAv2: EvalAI VQA Accuracy ---
# Same protocol as TextVQA (official VQA 2.0 evaluation)

def score_vqav2(pred: str, gt_answers: List[str]) -> float:
    """VQAv2 official scoring: EvalAI normalization + min(matches/3, 1).

    Returns accuracy in [0, 1].
    """
    if not gt_answers:
        return 0.0
    pred_norm = _eval_ai_processor(pred)
    gt_norms = [_eval_ai_processor(str(a)) for a in gt_answers]

    match_count = sum(1 for a in gt_norms if a == pred_norm)
    return min(1.0, float(match_count) / 3.0)


# --- ChartQA: Relaxed Correctness ---
# Official: https://arxiv.org/pdf/2203.10244.pdf (Section 5.1)
# Source: lmms-eval/lmms_eval/tasks/chartqa/utils.py

def _clean_short_answer(pred: str) -> str:
    """Strip common answer wrappers: 'Answer: X.', trailing period, etc.

    Applied uniformly to all methods before dataset-specific scoring.
    """
    text = pred.strip()
    # Strip "Answer: X" / "answer is X" prefix
    m = re.match(r'^[Aa]nswer\s*(?:is|=|:)\s*', text)
    if m:
        text = text[m.end():]
    # Strip trailing period (but not decimal like "3.14")
    if text.endswith('.') and not re.search(r'\d\.$', text):
        text = text[:-1]
    # Strip surrounding quotes
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1]
    return text.strip()


def score_chartqa(pred: str, gt: str, max_relative_change: float = 0.05) -> bool:
    """ChartQA official relaxed correctness.

    Numeric answers: within 5% tolerance.
    Non-numeric: case-insensitive exact match.
    Pred is cleaned (strip 'Answer:' prefix, trailing period) before matching.
    """
    pred_clean = _clean_short_answer(pred)

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # ChartQA ground truth is already in percentage form (e.g. "93.45" means 93.45%),
                # so predictions like "93.45%" should compare as 93.45, not 0.9345.
                return float(text.rstrip("%"))
            else:
                return float(text)
        except ValueError:
            return None

    pred_f = _to_float(pred_clean)
    gt_f = _to_float(gt.strip())
    if pred_f is not None and gt_f:
        relative_change = abs(pred_f - gt_f) / abs(gt_f)
        return relative_change <= max_relative_change
    else:
        return pred_clean.lower() == gt.strip().lower()


# --- OCRBench: Substring Matching ---
# Official: lmms-eval/lmms_eval/tasks/ocrbench/utils.py
# answer.lower() in prediction.lower() (case-insensitive substring)

def score_ocrbench(pred: str, gt_answers: List[str]) -> int:
    """OCRBench official scoring: substring matching.

    Returns 1 if any GT answer is a substring of prediction, else 0.
    """
    pred_clean = pred.lower().strip().replace("\n", " ")
    for answer in gt_answers:
        ans_clean = str(answer).lower().strip().replace("\n", " ")
        if ans_clean in pred_clean:
            return 1
    return 0


# =============================================================================
# Model classes
# =============================================================================

def setup_gpu():
    """Set up GPU"""
    cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = "cuda"
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}, {p.total_memory/1024**3:.1f}GB")
    else:
        device = "cpu"
    return device


class LowResModel:
    """LowRes baseline: reduced image resolution"""

    def __init__(self, config: EvalConfig, device: str):
        self.config = config
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        print(f"  [LowRes] Loading {self.config.decoder_path} (scale={self.config.lowres_scale})")
        self.processor = AutoProcessor.from_pretrained(
            self.config.decoder_path,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.decoder_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="sdpa",
            )
            print("  [LowRes] SDPA attention enabled")
        except Exception:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.decoder_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )

        self.model.eval()
        print("  [LowRes] Load complete")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Downscale image resolution to 1/3"""
        scale = self.config.lowres_scale
        new_w = max(28, int(image.width * scale))
        new_h = max(28, int(image.height * scale))
        return image.resize((new_w, new_h), Image.LANCZOS)

    def generate(self, image: Image.Image, prompt: str) -> str:
        image_lowres = self._resize_image(image)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_lowres},
            {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image_lowres], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                use_cache=True,
            )

        response = self.processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


class ImageC3Model:
    """Image-C3 V7 9x model (uses generate_with_rope_3d for 3D M-RoPE support)"""

    def __init__(self, config: EvalConfig, device: str):
        self.config = config
        self.device = device
        self.model = None

    def load(self):
        from crispr.model_v7 import ImageC3ConfigV7, ImageC3ModelV7
        from transformers import AutoProcessor

        bs = self.config.block_size
        compress_ratio = bs * bs
        print(f"  [Image-C3 {compress_ratio}x] Loading checkpoint: {self.config.image_c3_ckpt}")
        cfg = ImageC3ConfigV7(
            decoder_path=self.config.decoder_path,
            use_token_mixer=True,
            token_mixer_num_layers=2,
            use_local_c3=True,
            local_c3_block_size=bs,
            use_rope_alignment=True,  # Enable 3D M-RoPE, consistent with training
        )
        self.model = ImageC3ModelV7(cfg, device=self.device)
        self.model.setup()
        self.model.load_checkpoint(self.config.image_c3_ckpt)
        self.model.eval()
        # Override processor with max_pixels to prevent OOM on large images
        self.model.processor = AutoProcessor.from_pretrained(
            self.config.decoder_path,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )
        print(f"  [Image-C3 {compress_ratio}x] Load complete (rope_alignment=True, block_size={bs}, max_pixels={self.config.max_pixels})")

    def generate(self, image: Image.Image, prompt: str) -> str:
        processor = self.model.processor

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_only = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
        prompt_ids = processor.tokenizer(prompt_only, return_tensors="pt").input_ids.to(self.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate_with_rope_3d(
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                prompt_ids=prompt_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )

        response = processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return response

    def unload(self):
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


class TeacherModel:
    """Teacher model: original Qwen2.5-VL-7B-Instruct, uncompressed (performance upper bound)"""

    def __init__(self, config: EvalConfig, device: str):
        self.config = config
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        print(f"  [Teacher] Loading {self.config.decoder_path} (uncompressed upper bound)")
        self.processor = AutoProcessor.from_pretrained(
            self.config.decoder_path,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )
        print(f"  [Teacher] max_pixels={self.config.max_pixels} ({self.config.max_pixels//28//28}*28*28)")

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.decoder_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="sdpa",
            )
            print("  [Teacher] SDPA attention enabled")
        except Exception:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.decoder_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )

        self.model.eval()
        print("  [Teacher] Load complete")

    def generate(self, image: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                use_cache=True,
            )

        response = self.processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# Main evaluation logic
# =============================================================================

def build_prompt(sample: Dict[str, Any]) -> str:
    """Build the prompt for a sample"""
    task_type = sample["task_type"]

    if task_type == "mc":
        return format_mc_question(sample["question"], sample.get("options", []))
    elif task_type == "chartqa":
        return CHARTQA_PROMPT.format(question=sample["question"])
    elif task_type == "docvqa":
        return DOCVQA_PROMPT.format(question=sample["question"])
    elif task_type == "textvqa":
        return TEXTVQA_PROMPT.format(question=sample["question"])
    elif task_type == "caption":
        return sample.get("question", CAPTION_PROMPT)
    elif task_type == "vqav2":
        return VQAV2_PROMPT.format(question=sample["question"])
    elif task_type == "pope":
        return sample["question"]
    else:
        return sample["question"]


def evaluate_samples(
    model,
    samples: List[Dict[str, Any]],
    method_name: str,
    output_dir=None,
    checkpoint_interval: int = 500,
) -> Dict[str, Any]:
    """Evaluate samples (supports resuming from checkpoint)"""
    results = []
    metrics = {
        "mc_correct": 0,
        "mc_total": 0,
        "qa_exact": 0,
        "qa_relaxed": 0,
        "qa_total": 0,
    }

    # --- Checkpoint recovery ---
    ckpt_path = None
    done_ids = set()
    if output_dir and checkpoint_interval > 0:
        # method_name format: "teacher/docvqa"
        parts = method_name.replace("/", "_")
        ckpt_path = Path(output_dir) / f"ckpt_{parts}.json"
        if ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                results = ckpt["results"]
                metrics = ckpt["metrics"]
                done_ids = {r["id"] for r in results}
                print(f"  Resumed from checkpoint: {len(done_ids)}/{len(samples)} already done")
            except Exception as e:
                print(f"  Warning: failed to load checkpoint ({e}), starting from scratch")
                results = []
                done_ids = set()

    new_count = 0  # Count of newly added results in this run

    pbar = tqdm(samples, desc=f"  {method_name}")
    for sample in pbar:
        if sample["id"] in done_ids:
            continue
        try:
            prompt = build_prompt(sample)
            # Caption tasks use shorter max_new_tokens (64 vs 128)
            task_type = sample["task_type"]
            _orig_max_tokens = None
            if task_type == "caption" and hasattr(model, 'config'):
                _orig_max_tokens = model.config.max_new_tokens
                model.config.max_new_tokens = model.config.caption_max_new_tokens
            try:
                response = model.generate(sample["image"], prompt)
            finally:
                if _orig_max_tokens is not None:
                    model.config.max_new_tokens = _orig_max_tokens
            result = {
                "id": sample["id"],
                "pred": response,
                "gt": sample["gt"],
            }

            dataset_name = sample.get("dataset", "")

            if task_type == "mc":
                correct = score_mc(response, sample["gt"], len(sample.get("options", [])) or 4)
                result["correct"] = correct
                metrics["mc_correct"] += int(correct)
                metrics["mc_total"] += 1
            elif task_type == "caption":
                result["all_captions"] = sample.get("all_captions", [])
                metrics["caption_total"] = metrics.get("caption_total", 0) + 1
            elif task_type == "pope":
                pred_yn = "yes" if "yes" in response.lower() else "no"
                gt_yn = sample["gt"].lower().strip()
                correct = (pred_yn == gt_yn)
                result["pred_yn"] = pred_yn
                result["correct"] = correct
                result["pope_split"] = sample.get("pope_split", "")
                metrics["pope_correct"] = metrics.get("pope_correct", 0) + int(correct)
                metrics["pope_total"] = metrics.get("pope_total", 0) + 1
                sp = sample.get("pope_split", "all")
                metrics[f"pope_{sp}_correct"] = metrics.get(f"pope_{sp}_correct", 0) + int(correct)
                metrics[f"pope_{sp}_total"] = metrics.get(f"pope_{sp}_total", 0) + 1
                if pred_yn == "yes" and gt_yn == "yes":
                    metrics[f"pope_{sp}_tp"] = metrics.get(f"pope_{sp}_tp", 0) + 1
                elif pred_yn == "yes" and gt_yn == "no":
                    metrics[f"pope_{sp}_fp"] = metrics.get(f"pope_{sp}_fp", 0) + 1
                elif pred_yn == "no" and gt_yn == "no":
                    metrics[f"pope_{sp}_tn"] = metrics.get(f"pope_{sp}_tn", 0) + 1
                elif pred_yn == "no" and gt_yn == "yes":
                    metrics[f"pope_{sp}_fn"] = metrics.get(f"pope_{sp}_fn", 0) + 1

            # --- Dataset-specific official scoring ---
            elif task_type == "docvqa" or dataset_name == "docvqa":
                # DocVQA: ANLS (official)
                all_answers = sample.get("all_answers", [sample["gt"]])
                anls = score_anls(response, all_answers)
                result["anls"] = anls
                metrics["anls_sum"] = metrics.get("anls_sum", 0) + anls
                metrics["anls_total"] = metrics.get("anls_total", 0) + 1
            elif task_type == "textvqa" or dataset_name == "textvqa":
                # TextVQA: EvalAI VQA Accuracy (official)
                all_answers = sample.get("all_answers", [sample["gt"]])
                acc = score_textvqa(response, all_answers)
                result["textvqa_accuracy"] = acc
                metrics["textvqa_acc_sum"] = metrics.get("textvqa_acc_sum", 0) + acc
                metrics["textvqa_total"] = metrics.get("textvqa_total", 0) + 1
            elif task_type == "chartqa" or dataset_name == "chartqa":
                # ChartQA: Relaxed Correctness (official)
                correct = score_chartqa(response, sample["gt"])
                result["relaxed_correct"] = correct
                metrics["chartqa_correct"] = metrics.get("chartqa_correct", 0) + int(correct)
                metrics["chartqa_total"] = metrics.get("chartqa_total", 0) + 1
            elif dataset_name == "ocrbench":
                # OCRBench: Substring matching (official)
                all_answers = sample.get("all_answers", [sample["gt"]])
                score = score_ocrbench(response, all_answers)
                result["ocr_score"] = score
                metrics["ocr_score_sum"] = metrics.get("ocr_score_sum", 0) + score
                metrics["ocr_total"] = metrics.get("ocr_total", 0) + 1
            elif task_type == "vqav2" or dataset_name == "vqav2":
                # VQAv2: EvalAI VQA Accuracy (official)
                all_answers = sample.get("all_answers", [])
                vqa_acc = score_vqav2(response, all_answers)
                result["vqa_accuracy"] = vqa_acc
                metrics["vqav2_acc_sum"] = metrics.get("vqav2_acc_sum", 0) + vqa_acc
                metrics["vqav2_total"] = metrics.get("vqav2_total", 0) + 1
            else:
                # Generic QA (MathVista free_form, MathVision QA, etc.)
                exact, relaxed = score_qa(
                    response,
                    sample["gt"],
                    sample.get("all_answers")
                )
                result["exact"] = exact
                result["relaxed"] = relaxed
                metrics["qa_exact"] += int(exact)
                metrics["qa_relaxed"] += int(relaxed)
                metrics["qa_total"] += 1

            results.append(result)
            new_count += 1

            # Update progress bar
            if metrics["mc_total"] > 0:
                pbar.set_postfix(mc_acc=f"{metrics['mc_correct']/metrics['mc_total']:.1%}")
            if metrics["qa_total"] > 0:
                pbar.set_postfix(qa_acc=f"{metrics['qa_relaxed']/metrics['qa_total']:.1%}")
            if metrics.get("pope_total", 0) > 0:
                pbar.set_postfix(pope_acc=f"{metrics['pope_correct']/metrics['pope_total']:.1%}")
            if metrics.get("vqav2_total", 0) > 0:
                pbar.set_postfix(vqa_acc=f"{metrics['vqav2_acc_sum']/metrics['vqav2_total']:.1%}")
            if metrics.get("anls_total", 0) > 0:
                pbar.set_postfix(anls=f"{metrics['anls_sum']/metrics['anls_total']:.1%}")
            if metrics.get("textvqa_total", 0) > 0:
                pbar.set_postfix(tvqa=f"{metrics['textvqa_acc_sum']/metrics['textvqa_total']:.1%}")
            if metrics.get("chartqa_total", 0) > 0:
                pbar.set_postfix(cqa=f"{metrics['chartqa_correct']/metrics['chartqa_total']:.1%}")
            if metrics.get("ocr_total", 0) > 0:
                pbar.set_postfix(ocr=f"{metrics['ocr_score_sum']/metrics['ocr_total']:.1%}")

            # --- Periodically save checkpoint ---
            if ckpt_path and checkpoint_interval > 0 and new_count % checkpoint_interval == 0:
                _save_checkpoint(ckpt_path, results, metrics, len(samples))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n  Warning: sample {sample['id']} OOM, cleared GPU memory and skipped")
            metrics["oom_count"] = metrics.get("oom_count", 0) + 1
            continue
        except Exception as e:
            print(f"\n  Warning: sample {sample['id']} failed: {e}")
            continue

    # --- Delete checkpoint once done ---
    if ckpt_path and ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except OSError:
            pass

    return {"results": results, "metrics": metrics}


def _save_checkpoint(ckpt_path: Path, results, metrics, total: int):
    """Write checkpoint atomically (write to .tmp then rename)"""
    ckpt_data = {
        "results": results,
        "metrics": metrics,
        "processed": len(results),
        "total": total,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp_path = ckpt_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(ckpt_data, f, ensure_ascii=False)
    os.replace(tmp_path, ckpt_path)


def run_evaluation(args):
    """Run the evaluation"""
    config = EvalConfig(
        decoder_path=args.decoder_path,
        image_c3_ckpt=args.image_c3_ckpt,
        max_new_tokens=args.max_new_tokens,
        caption_max_new_tokens=args.caption_max_new_tokens,
        seed=args.seed,
        block_size=args.block_size,
        lowres_scale=args.lowres_scale,
        max_pixels=args.max_pixels,
        data_root=args.data_root,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    compress_ratio = config.block_size ** 2
    print(f"Image-C3 {compress_ratio}x quick evaluation")
    print("=" * 60)
    print(f"  Methods: {args.methods}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Samples per dataset: {args.n}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)

    # Set up GPU
    device = setup_gpu()

    # Load data
    print("\n[1] Loading data...")
    all_samples = {}
    for dataset in args.datasets:
        if dataset not in DATASET_LOADERS:
            print(f"  Warning: unknown dataset {dataset}")
            continue
        samples = DATASET_LOADERS[dataset](config.data_root, args.n, config.seed)
        if samples:
            all_samples[dataset] = samples

    if not all_samples:
        print("Error: no data loaded")
        return

    # Evaluation results
    all_results = {}

    # Evaluate each method
    for method in args.methods:
        print(f"\n[2] Evaluating method: {method}")

        # Load model
        if method == "image_c3":
            model = ImageC3Model(config, device)
        elif method == "lowres":
            model = LowResModel(config, device)
        elif method == "teacher":
            model = TeacherModel(config, device)
        else:
            print(f"  Warning: unknown method {method}")
            continue

        model.load()

        method_results = {}
        for dataset, samples in all_samples.items():
            print(f"\n  Evaluating {dataset}...")
            result = evaluate_samples(model, samples, f"{method}/{dataset}",
                                      output_dir=output_dir,
                                      checkpoint_interval=args.checkpoint_interval)
            method_results[dataset] = result

        model.unload()
        all_results[method] = method_results

    # Summarize results
    print("\n" + "=" * 60)
    print("Evaluation results summary")
    print("=" * 60)

    summary = {}
    for method, method_results in all_results.items():
        summary[method] = {}
        for dataset, result in method_results.items():
            metrics = result["metrics"]
            mc_t = metrics["mc_total"]
            qa_t = metrics["qa_total"]
            cap_t = metrics.get("caption_total", 0)
            vqav2_t = metrics.get("vqav2_total", 0)

            pope_t = metrics.get("pope_total", 0)

            if vqav2_t > 0:
                vqa_acc = metrics["vqav2_acc_sum"] / vqav2_t
                summary[method][dataset] = {
                    "vqa_accuracy": round(vqa_acc, 4),
                    "total": vqav2_t,
                }
                print(f"  {method:12s} | {dataset:10s} | VQA Acc: {vqa_acc:.2%} (n={vqav2_t})")
            elif pope_t > 0:
                # POPE: compute per-split accuracy, precision, recall, F1
                pope_summary = {}
                for sp in ["random", "popular", "adversarial"]:
                    sp_total = metrics.get(f"pope_{sp}_total", 0)
                    if sp_total == 0:
                        continue
                    sp_correct = metrics.get(f"pope_{sp}_correct", 0)
                    tp = metrics.get(f"pope_{sp}_tp", 0)
                    fp = metrics.get(f"pope_{sp}_fp", 0)
                    fn = metrics.get(f"pope_{sp}_fn", 0)
                    tn = metrics.get(f"pope_{sp}_tn", 0)
                    acc = sp_correct / sp_total
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                    yes_ratio = (tp + fp) / sp_total
                    pope_summary[sp] = {
                        "accuracy": round(acc, 4),
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1": round(f1, 4),
                        "yes_ratio": round(yes_ratio, 4),
                        "total": sp_total,
                    }
                    print(f"  {method:12s} | pope_{sp:12s} | Acc: {acc:.1%}  P: {precision:.1%}  R: {recall:.1%}  F1: {f1:.1%}  Yes%: {yes_ratio:.1%}")
                overall_acc = metrics["pope_correct"] / pope_t
                pope_summary["overall"] = {"accuracy": round(overall_acc, 4), "total": pope_t}
                summary[method][dataset] = pope_summary
            elif cap_t > 0:
                # Caption: PTBTokenizer + CIDEr/BLEU-4/METEOR/ROUGE-L (academic standard)
                try:
                    from pycocoevalcap.cider.cider import Cider
                    from pycocoevalcap.bleu.bleu import Bleu
                    from pycocoevalcap.meteor.meteor import Meteor
                    from pycocoevalcap.rouge.rouge import Rouge
                    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

                    gts = {}  # {id: [{caption: ref1}, {caption: ref2}, ...]}
                    res = {}  # {id: [{caption: pred}]}
                    for r in result["results"]:
                        img_id = int(r["id"]) if isinstance(r["id"], (int, float)) or str(r["id"]).isdigit() else r["id"]
                        refs = r.get("all_captions", [r["gt"]])
                        gts[img_id] = [{"caption": ref} for ref in refs]
                        res[img_id] = [{"caption": r["pred"]}]

                    # PTBTokenizer: lowercase + Stanford PTB tokenization (academic standard)
                    tokenizer = PTBTokenizer()
                    gts_tok = tokenizer.tokenize(gts)
                    res_tok = tokenizer.tokenize(res)

                    # Compute all 4 metrics
                    cider_score, _ = Cider().compute_score(gts_tok, res_tok)
                    bleu_scores, _ = Bleu(4).compute_score(gts_tok, res_tok)
                    bleu4_score = bleu_scores[3]  # Bleu_4
                    meteor_score, _ = Meteor().compute_score(gts_tok, res_tok)
                    rouge_score, _ = Rouge().compute_score(gts_tok, res_tok)

                    summary[method][dataset] = {
                        "CIDEr": round(cider_score, 4),
                        "BLEU-4": round(bleu4_score, 4),
                        "METEOR": round(meteor_score, 4),
                        "ROUGE-L": round(rouge_score, 4),
                        "total": cap_t,
                    }
                    print(f"  {method:12s} | {dataset:10s} | CIDEr: {cider_score:.3f}  B4: {bleu4_score:.3f}  M: {meteor_score:.3f}  R: {rouge_score:.3f} (n={cap_t})")
                except Exception as e:
                    import traceback
                    print(f"  {method:12s} | {dataset:10s} | Caption scoring failed: {e}")
                    traceback.print_exc()
            # --- Dataset-specific official metrics ---
            anls_t = metrics.get("anls_total", 0)
            textvqa_t = metrics.get("textvqa_total", 0)
            chartqa_t = metrics.get("chartqa_total", 0)
            ocr_t = metrics.get("ocr_total", 0)

            if anls_t > 0:
                # DocVQA: ANLS
                anls_acc = metrics["anls_sum"] / anls_t
                summary[method][dataset] = {"anls": round(anls_acc, 4), "total": anls_t}
                print(f"  {method:12s} | {dataset:10s} | ANLS: {anls_acc:.2%} (n={anls_t})")
            elif textvqa_t > 0:
                # TextVQA: VQA Accuracy
                tvqa_acc = metrics["textvqa_acc_sum"] / textvqa_t
                summary[method][dataset] = {"vqa_accuracy": round(tvqa_acc, 4), "total": textvqa_t}
                print(f"  {method:12s} | {dataset:10s} | VQA Acc: {tvqa_acc:.2%} (n={textvqa_t})")
            elif chartqa_t > 0:
                # ChartQA: Relaxed Correctness
                cqa_acc = metrics["chartqa_correct"] / chartqa_t
                summary[method][dataset] = {"relaxed_accuracy": round(cqa_acc, 4), "total": chartqa_t}
                print(f"  {method:12s} | {dataset:10s} | Relaxed: {cqa_acc:.2%} (n={chartqa_t})")
            elif ocr_t > 0:
                # OCRBench: /1000 score
                ocr_score = metrics["ocr_score_sum"]
                ocr_acc = ocr_score / ocr_t
                summary[method][dataset] = {"score": ocr_score, "accuracy": round(ocr_acc, 4), "total": ocr_t}
                print(f"  {method:12s} | {dataset:10s} | Score: {ocr_score}/{ocr_t} ({ocr_acc:.2%})")
            # Mixed datasets (MC + QA): combine
            elif mc_t > 0 and qa_t > 0:
                combined_correct = metrics["mc_correct"] + metrics["qa_relaxed"]
                combined_total = mc_t + qa_t
                acc = combined_correct / combined_total
                summary[method][dataset] = {
                    "accuracy": acc,
                    "correct": combined_correct,
                    "total": combined_total,
                }
                print(f"  {method:12s} | {dataset:10s} | Combined: {acc:.2%} ({combined_correct}/{combined_total})")
            elif mc_t > 0:
                acc = metrics["mc_correct"] / mc_t
                summary[method][dataset] = {
                    "accuracy": acc,
                    "correct": metrics["mc_correct"],
                    "total": mc_t,
                }
                print(f"  {method:12s} | {dataset:10s} | Accuracy: {acc:.2%} ({metrics['mc_correct']}/{mc_t})")
            elif qa_t > 0:
                relaxed_acc = metrics["qa_relaxed"] / qa_t
                oom_count = metrics.get("oom_count", 0)
                summary[method][dataset] = {
                    "relaxed_accuracy": relaxed_acc,
                    "total": qa_t,
                }
                if oom_count > 0:
                    summary[method][dataset]["oom_skipped"] = oom_count
                oom_str = f", OOM={oom_count}" if oom_count > 0 else ""
                print(f"  {method:12s} | {dataset:10s} | Relaxed: {relaxed_acc:.2%} (n={qa_t}{oom_str})")

    # Save results
    report = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "methods": args.methods,
            "datasets": args.datasets,
            "n": args.n,
            "max_new_tokens": args.max_new_tokens,
            "caption_max_new_tokens": config.caption_max_new_tokens,
            "seed": args.seed,
            "block_size": config.block_size,
            "lowres_scale": config.lowres_scale,
            "max_pixels": config.max_pixels,
            "image_c3_ckpt": config.image_c3_ckpt,
        },
        "summary": summary,
        "detailed_results": {
            method: {
                dataset: result["results"]  # Save all detailed results (useful for future re-scoring)
                for dataset, result in method_results.items()
            }
            for method, method_results in all_results.items()
        }
    }

    report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {report_path}")

    # Generate Markdown comparison table
    md_lines = [
        "# Image-C3 9x Evaluation Results",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Samples per dataset: {args.n}",
        "",
        "## Results comparison",
        "",
        "| Dataset | " + " | ".join(args.methods) + " |",
        "|" + "---|" * (len(args.methods) + 1),
    ]

    for dataset in args.datasets:
        row = [dataset]
        for method in args.methods:
            if method in summary and dataset in summary[method]:
                s = summary[method][dataset]
                if "accuracy" in s:
                    row.append(f"{s['accuracy']:.2%}")
                elif "relaxed_accuracy" in s:
                    row.append(f"{s['relaxed_accuracy']:.2%}")
                elif "CIDEr" in s:
                    row.append(f"C:{s['CIDEr']:.3f} B4:{s['BLEU-4']:.3f}")
                else:
                    row.append("-")
            else:
                row.append("-")
        md_lines.append("| " + " | ".join(row) + " |")

    md_path = output_dir / "COMPARISON.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Markdown report: {md_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Image-C3 9x quick evaluation")

    # Methods and datasets
    parser.add_argument("--methods", nargs="+", default=["image_c3", "lowres"],
                        help="Evaluation methods: image_c3, lowres, teacher")
    parser.add_argument("--datasets", nargs="+",
                        default=["mmmu", "mmbench", "realworldqa", "mathvista", "chartqa", "docvqa", "textvqa", "ocrbench", "coco"],
                        help="Evaluation datasets")

    # Sample count
    parser.add_argument("--n", type=int, default=200,
                        help="Number of samples per dataset (0=all)")

    # Model paths
    parser.add_argument("--decoder_path", type=str,
                        default="./Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_c3_ckpt", type=str,
                        default="./outputs/image_c3_v7_9x_3drope_20260301_103916/stage2/checkpoint-best")

    # Evaluation parameters
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--caption_max_new_tokens", type=int, default=64,
                        help="Caption tasks max_new_tokens (lmms-eval standard: 64)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=3,
                        help="Compression block size: 3=9x, 4=16x")
    parser.add_argument("--lowres_scale", type=float, default=0.333,
                        help="LowRes scale factor (0.333 ~= 9x, 0.25 ~= 16x)")
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28,
                        help="Max image pixel count (prevents OOM, default 1280*28*28~=1M)")

    # Data and output
    parser.add_argument("--data_root", type=str, default="./data/eval")
    parser.add_argument("--output_dir", type=str,
                        default="./eval_results/9x_comparison")
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="Save a checkpoint every N processed samples (0=disable)")

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
