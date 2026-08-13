#!/usr/bin/env python3
"""
CRISP Evaluation Script

General-purpose evaluation for CRISP visual token compression.
Supports any combination of model size (7B/3B) and compression ratio (9x/16x).

Methods:
- image_c3: CRISP compressed (9x or 16x via --block_size)
- lowres: Low-resolution baseline (matching compression ratio)
- teacher: Original Qwen2.5-VL (uncompressed upper bound)

Datasets:
- VQA: MMMU, MMBench, ChartQA, DocVQA, TextVQA, RealWorldQA, MathVista, OCRBench, VQAv2
- Perception: MME (14 categories, Perception/Cognition scoring), ScienceQA-IMG
- Caption: COCO, TextCaps, Flickr30k, NoCaps
- Hallucination: POPE

Usage:
    # 7B CRISP 9x - full eval
    CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_crisp.py \
        --methods image_c3 \
        --datasets mmmu mmbench chartqa docvqa textvqa realworldqa mathvista ocrbench mme scienceqa \
        --n 0 --output_dir eval_results/7b_9x

    # 3B CRISP 9x
    python scripts/eval_crisp.py \
        --methods image_c3 \
        --decoder_path ./Qwen/Qwen2.5-VL-3B-Instruct \
        --image_c3_ckpt ./outputs/3b_9x_checkpoint/checkpoint-best \
        --datasets mmmu mmbench chartqa docvqa textvqa mme scienceqa \
        --n 0 --output_dir eval_results/3b_9x

    # Quick comparison (200 samples per dataset)
    python scripts/eval_crisp.py \
        --methods image_c3 lowres teacher \
        --datasets mmmu mmbench chartqa docvqa textvqa \
        --n 200
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
from collections import defaultdict

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration"""
    decoder_path: str = "./Qwen/Qwen2.5-VL-7B-Instruct"
    image_c3_ckpt: str = "./outputs/image_c3_v7_9x_3drope_20260301_103916/stage2/checkpoint-best"
    max_new_tokens: int = 128
    caption_max_new_tokens: int = 64
    do_sample: bool = False
    seed: int = 42
    block_size: int = 3
    lowres_scale: float = 0.333
    max_pixels: int = 1280 * 28 * 28
    min_pixels: int = 4 * 28 * 28
    data_root: str = "./data/eval"

    # Ablation overrides (default = full CRISP config)
    no_token_mixer: bool = False
    no_rope_alignment: bool = False
    no_relative_pos_embed: bool = False
    compression_mode: str = "cross_attn"
    num_post_encoder_layers: int = 2


# =============================================================================
# Prompt templates
# =============================================================================

MC_PROMPT_SUFFIX = "\nAnswer with the option letter only."
CHARTQA_PROMPT = "Look at the chart and answer: {question}\nAnswer with a short phrase or number."
DOCVQA_PROMPT = "Based on the document, answer: {question}\nAnswer concisely."
TEXTVQA_PROMPT = "Read the text in the image and answer: {question}\nAnswer with a word or phrase."
CAPTION_PROMPT = "Provide a one-sentence caption for the provided image."
TEXTCAPS_PROMPT = "Please carefully observe the image and come up with a caption for the image."
POPE_PROMPT_SUFFIX = "\nAnswer with yes or no."
VQAV2_PROMPT = "{question}\nAnswer the question using a single word or phrase."
MME_PROMPT_SUFFIX = "\nPlease answer yes or no."


def format_mc_question(question: str, options: List[str]) -> str:
    """Format multiple choice question"""
    if options:
        opts_str = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        return f"{question}\n{opts_str}{MC_PROMPT_SUFFIX}"
    return question + MC_PROMPT_SUFFIX


# =============================================================================
# MME category classification
# =============================================================================

MME_PERCEPTION_CATS = {
    "existence", "count", "position", "color", "posters",
    "celebrity", "scene", "landmark", "artwork", "OCR"
}
MME_COGNITION_CATS = {
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning"
}


# =============================================================================
# Dataset loaders
# =============================================================================

def load_mmmu(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load MMMU validation data (parquet-based, multi-subject)"""
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
    """Load MMBench dev data (single parquet)"""
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
    """Load ChartQA test data (parquet)"""
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
    """Load DocVQA validation data (parquet)"""
    import pyarrow.parquet as pq

    docvqa_root = Path(data_root) / "docvqa/DocVQA"
    samples = []

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

                # answers can be a list
                answers = row.get("answers", row.get("answer", ""))
                if isinstance(answers, list):
                    gt = answers[0] if answers else ""
                else:
                    gt = str(answers)

                samples.append({
                    "id": str(row.get("questionId", f"docvqa_{len(samples)}")),
                    "image": image,
                    "question": question,
                    "gt": gt,
                    "all_answers": answers if isinstance(answers, list) else [str(answers)],
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
    """Load TextVQA validation data (parquet)"""
    import pyarrow.parquet as pq

    textvqa_root = Path(data_root) / "textvqa/data"
    samples = []

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
                if isinstance(answers, list):
                    gt = answers[0] if answers else ""
                else:
                    gt = str(answers)

                samples.append({
                    "id": str(row.get("question_id", f"textvqa_{len(samples)}")),
                    "image": image,
                    "question": question,
                    "gt": gt,
                    "all_answers": answers if isinstance(answers, list) else [str(answers)],
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
    """Load RealWorldQA (TSV + vlmeval_images, multiple choice)"""
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
    """Load MathVision MINI (TSV + vlmeval_images, MC + QA mixed)"""
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
    """Load OCRBench test data (parquet)"""
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
    """Load COCO Caption val2017 (5000 images, jsonl + images)"""
    jsonl_path = Path(data_root) / "coco" / "extracted" / "coco_val2017.jsonl"
    img_base = Path(data_root) / "coco" / "extracted"
    samples = []

    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            # Try multiple paths
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
    """Load MathVista testmini (1000 questions, MC + free_form mixed)

    Data source: AI4Math/MathVista, saved with datasets.save_to_disk
    question_type: multi_choice (540) / free_form (460)
    answer_type:   text (MC) / integer / float / list
    query field contains formatted prompt, use directly
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
            # Convert gt to option letter (A/B/C/D)
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


def load_mme(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load MME benchmark (yes/no questions across 14 categories)

    Standard MME scoring:
    - Each subcategory has paired yes/no questions
    - Acc+ = accuracy on yes-answer questions
    - Acc- = accuracy on no-answer questions
    - Category score = (Acc+ + Acc-) x 100 (max 200 per category)
    - Perception = 10 categories (max 2000)
    - Cognition = 4 categories (max 800)
    """
    jsonl_path = Path(data_root) / "mme" / "mme_test.jsonl"
    if not jsonl_path.exists():
        print(f"  [MME] File not found: {jsonl_path}")
        return []

    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            img_path = Path(d["image"])
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue

            question = d["question"]
            # Ensure question ends with yes/no prompt
            if "yes or no" not in question.lower():
                question = question.rstrip() + MME_PROMPT_SUFFIX

            samples.append({
                "id": str(d["id"]),
                "image": image,
                "question": question,
                "gt": d["answer"],
                "category": d.get("category", "unknown"),
                "task_type": "mme",
                "dataset": "mme",
            })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)

    # Print category distribution
    cats = defaultdict(int)
    for s in samples:
        cats[s["category"]] += 1
    print(f"  [MME] Loaded {len(samples)} samples ({len(cats)} categories)")
    return samples


def load_scienceqa(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load ScienceQA-IMG test split (multiple choice with optional hints)"""
    jsonl_path = Path(data_root) / "scienceqa" / "scienceqa_test.jsonl"
    if not jsonl_path.exists():
        print(f"  [ScienceQA] File not found: {jsonl_path}")
        return []

    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            img_path = Path(d["image"])
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue

            choices = d.get("choices", [])
            answer_idx = d.get("answer", 0)
            gt_letter = chr(65 + answer_idx) if isinstance(answer_idx, int) and answer_idx < len(choices) else str(answer_idx)

            # Build question with hint context if available
            question = d["question"]
            hint = d.get("hint", "")
            if hint:
                question = f"Context: {hint}\n\nQuestion: {question}"

            samples.append({
                "id": str(d["id"]),
                "image": image,
                "question": question,
                "options": choices,
                "gt": gt_letter,
                "task_type": "mc",
                "dataset": "scienceqa",
            })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    print(f"  [ScienceQA] Loaded {len(samples)} samples")
    return samples


def load_vqav2(data_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load VQAv2 validation set (214k samples)

    Standard VQA accuracy: min(#annotators_with_same_answer / 3, 1)
    Note: Full validation is 214k samples. Use --n to subsample for quick tests.
    """
    jsonl_path = Path(data_root) / "vqav2" / "vqav2_val.jsonl"
    if not jsonl_path.exists():
        print(f"  [VQAv2] File not found: {jsonl_path}")
        return []

    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            img_path = Path(d["image"])
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue

            answers = d.get("answers", [])
            # Get most common answer as gt
            gt = d.get("answer", "")
            if isinstance(gt, dict):
                gt = gt.get("answer", "")
            elif isinstance(gt, str):
                pass
            else:
                gt = str(gt) if gt else ""

            samples.append({
                "id": str(d["id"]),
                "image": image,
                "question": d["question"],
                "gt": str(gt),
                "all_answers": answers,
                "task_type": "vqav2",
                "dataset": "vqav2",
            })

    random.seed(seed)
    if n and len(samples) > n:
        samples = random.sample(samples, n)
    print(f"  [VQAv2] Loaded {len(samples)} samples")
    return samples


DATASET_LOADERS = {
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
    # New datasets
    "mme": load_mme,
    "scienceqa": load_scienceqa,
    "vqav2": load_vqav2,
}


# =============================================================================
# Scoring functions
# =============================================================================

def extract_mc_answer(text: str, num_options: int = 4) -> Optional[str]:
    """Extract multiple choice answer letter from response"""
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
    """Score multiple choice question"""
    pred_letter = extract_mc_answer(pred, num_options)
    gt_letter = gt.strip().upper()[:1] if gt else ""
    return pred_letter == gt_letter


def _normalize_text(text: str) -> str:
    """Normalize text: remove punctuation, unify whitespace, lowercase"""
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _clean_short_answer(pred: str) -> str:
    """Strip common answer wrappers: 'Answer: X.', trailing period, etc."""
    text = pred.strip()
    m = re.match(r'^[Aa]nswer\s*(?:is|=|:)\s*', text)
    if m:
        text = text[m.end():]
    if text.endswith('.') and not re.search(r'\d\.$', text):
        text = text[:-1]
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1]
    return text.strip()


def score_chartqa(pred: str, gt: str, max_relative_change: float = 0.05) -> bool:
    """ChartQA official relaxed correctness.

    Numeric answers: within 5% tolerance.
    Non-numeric: case-insensitive exact match.
    Pred is cleaned (strip 'Answer:' prefix, trailing period) before matching.
    Source: lmms-eval/lmms_eval/tasks/chartqa/utils.py
    """
    pred_clean = _clean_short_answer(pred)

    def _to_float(text: str):
        try:
            if text.endswith("%"):
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


def score_qa(pred: str, gt: str, all_answers: List[str] = None) -> Tuple[bool, bool]:
    """
    Score QA task (ChartQA/DocVQA/TextVQA)

    Relaxed matching strategy:
      1. Normalized exact match (remove punctuation/lowercase)
      2. Numeric 5% tolerance
      3. Normalized substring match (shorter string must be >= 3 chars to prevent "2" in "22")

    Returns: (exact_match, relaxed_match)
    """
    pred_norm = _normalize_text(pred)
    gt_norm = _normalize_text(gt)

    # 1. Exact match (normalized)
    if pred_norm == gt_norm:
        return True, True

    # Check all_answers (normalized exact match)
    if all_answers:
        for ans in all_answers:
            if pred_norm == _normalize_text(str(ans)):
                return True, True

    # 2. Numeric 5% tolerance
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

    # 3. Substring match (normalized, shorter string must be >= 3 chars)
    shorter_len = min(len(pred_norm), len(gt_norm))
    if shorter_len >= 3:
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return False, True
        if all_answers:
            for ans in all_answers:
                ans_norm = _normalize_text(str(ans))
                if len(ans_norm) >= 3 and (ans_norm in pred_norm or pred_norm in ans_norm):
                    return False, True

    return False, False


# =============================================================================
# Model classes
# =============================================================================

def setup_gpu():
    """Setup GPU"""
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
    """LowRes Baseline: reduce resolution"""

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
                attn_implementation="flash_attention_2",
            )
            print("  [LowRes] Flash Attention enabled")
        except:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.decoder_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )

        self.model.eval()
        print("  [LowRes] Loading complete")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Reduce resolution"""
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
    """Image-C3 V7 model (uses generate_with_rope_3d for 3D M-RoPE support)"""

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
            use_token_mixer=not self.config.no_token_mixer,
            token_mixer_num_layers=2,
            use_local_c3=True,
            local_c3_block_size=bs,
            use_rope_alignment=not self.config.no_rope_alignment,
            use_relative_pos_embed=not self.config.no_relative_pos_embed,
            compression_mode=self.config.compression_mode,
            num_post_encoder_layers=self.config.num_post_encoder_layers,
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
        print(f"  [Image-C3 {compress_ratio}x] Loading complete (rope_alignment=True, block_size={bs}, max_pixels={self.config.max_pixels})")

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
    """Teacher model: original Qwen2.5-VL, uncompressed (upper bound)"""

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
                attn_implementation="flash_attention_2",
            )
            print("  [Teacher] Flash Attention enabled")
        except:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.decoder_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )

        self.model.eval()
        print("  [Teacher] Loading complete")

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
# Evaluation logic
# =============================================================================

def build_prompt(sample: Dict[str, Any]) -> str:
    """Build prompt for each task type"""
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
    elif task_type == "pope":
        return sample["question"]
    elif task_type == "mme":
        return sample["question"]  # Already includes yes/no prompt
    elif task_type == "vqav2":
        return VQAV2_PROMPT.format(question=sample["question"])
    else:
        return sample["question"]


def evaluate_samples(
    model,
    samples: List[Dict[str, Any]],
    method_name: str,
    output_dir=None,
    checkpoint_interval: int = 500,
) -> Dict[str, Any]:
    """Evaluate samples (with checkpoint resume support)"""
    results = []
    metrics = {
        "mc_correct": 0,
        "mc_total": 0,
        "qa_exact": 0,
        "qa_relaxed": 0,
        "qa_total": 0,
    }

    # --- checkpoint resume ---
    ckpt_path = None
    done_ids = set()
    if output_dir and checkpoint_interval > 0:
        parts = method_name.replace("/", "_")
        ckpt_path = Path(output_dir) / f"ckpt_{parts}.json"
        if ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                results = ckpt["results"]
                metrics = ckpt["metrics"]
                done_ids = {r["id"] for r in results}
                print(f"  Resumed from checkpoint: {len(done_ids)}/{len(samples)} completed")
            except Exception as e:
                print(f"  Warning: failed to load checkpoint ({e}), starting from scratch")
                results = []
                done_ids = set()

    new_count = 0

    pbar = tqdm(samples, desc=f"  {method_name}")
    for sample in pbar:
        if sample["id"] in done_ids:
            continue
        try:
            prompt = build_prompt(sample)
            # Caption tasks use shorter max_new_tokens (64 vs 128)
            task_type = sample["task_type"]
            if task_type == "caption" and hasattr(model, 'config'):
                _orig_max_tokens = model.config.max_new_tokens
                model.config.max_new_tokens = model.config.caption_max_new_tokens
            response = model.generate(sample["image"], prompt)
            if task_type == "caption" and hasattr(model, 'config'):
                model.config.max_new_tokens = _orig_max_tokens
            result = {
                "id": sample["id"],
                "pred": response,
                "gt": sample["gt"],
            }

            if task_type == "mc":
                correct = score_mc(response, sample["gt"], len(sample.get("options", [])) or 4)
                result["correct"] = correct
                metrics["mc_correct"] += int(correct)
                metrics["mc_total"] += 1

            elif task_type == "mme":
                # MME: yes/no questions with category tracking
                pred_yn = "yes" if "yes" in response.lower() else "no"
                gt_yn = sample["gt"].lower().strip()
                correct = (pred_yn == gt_yn)
                result["pred_yn"] = pred_yn
                result["correct"] = correct
                result["category"] = sample.get("category", "")
                result["gt_yn"] = gt_yn
                metrics["mme_correct"] = metrics.get("mme_correct", 0) + int(correct)
                metrics["mme_total"] = metrics.get("mme_total", 0) + 1

            elif task_type == "vqav2":
                # VQA accuracy: min(count_matching / 3, 1)
                pred_norm = _normalize_text(response)
                answers = sample.get("all_answers", [])
                if isinstance(answers, list) and len(answers) > 0:
                    if isinstance(answers[0], dict):
                        count = sum(1 for a in answers if _normalize_text(str(a.get("answer", ""))) == pred_norm)
                    else:
                        count = sum(1 for a in answers if _normalize_text(str(a)) == pred_norm)
                    vqa_acc = min(count / 3.0, 1.0)
                else:
                    vqa_acc = 1.0 if pred_norm == _normalize_text(sample["gt"]) else 0.0
                result["vqa_accuracy"] = vqa_acc
                metrics["vqa_acc_sum"] = metrics.get("vqa_acc_sum", 0) + vqa_acc
                metrics["vqa_total"] = metrics.get("vqa_total", 0) + 1

            elif task_type == "caption":
                # Caption task: save all_captions for offline CIDEr scoring
                result["all_captions"] = sample.get("all_captions", [])
                metrics["caption_total"] = metrics.get("caption_total", 0) + 1

            elif task_type == "pope":
                # POPE: yes/no judgment
                pred_yn = "yes" if "yes" in response.lower() else "no"
                gt_yn = sample["gt"].lower().strip()
                correct = (pred_yn == gt_yn)
                result["pred_yn"] = pred_yn
                result["correct"] = correct
                result["pope_split"] = sample.get("pope_split", "")
                metrics["pope_correct"] = metrics.get("pope_correct", 0) + int(correct)
                metrics["pope_total"] = metrics.get("pope_total", 0) + 1
                # Track per-split and TP/FP/TN/FN
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

            elif task_type == "chartqa" or sample.get("dataset", "") == "chartqa":
                # ChartQA: official relaxed correctness (5% numeric tolerance, exact text match)
                correct = score_chartqa(response, sample["gt"])
                result["relaxed_correct"] = correct
                metrics["qa_exact"] += int(correct)
                metrics["qa_relaxed"] += int(correct)
                metrics["qa_total"] += 1

            else:
                # Default QA scoring
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
            if metrics.get("mme_total", 0) > 0:
                pbar.set_postfix(mme_acc=f"{metrics['mme_correct']/metrics['mme_total']:.1%}")
            if metrics.get("vqa_total", 0) > 0:
                pbar.set_postfix(vqa_acc=f"{metrics['vqa_acc_sum']/metrics['vqa_total']:.1%}")

            # --- periodic checkpoint save ---
            if ckpt_path and checkpoint_interval > 0 and new_count % checkpoint_interval == 0:
                _save_checkpoint(ckpt_path, results, metrics, len(samples))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n  Warning: sample {sample['id']} OOM, cleared GPU memory and skipped")
            metrics["oom_count"] = metrics.get("oom_count", 0) + 1
            continue
        except Exception as e:
            print(f"\n  Warning: error on sample {sample['id']}: {e}")
            continue

    # --- delete checkpoint after completion ---
    if ckpt_path and ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except OSError:
            pass

    return {"results": results, "metrics": metrics}


def _save_checkpoint(ckpt_path: Path, results, metrics, total: int):
    """Atomic checkpoint write (write .tmp then rename)"""
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
    """Run evaluation"""
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
        no_token_mixer=args.no_token_mixer,
        no_rope_alignment=args.no_rope_alignment,
        no_relative_pos_embed=args.no_relative_pos_embed,
        compression_mode=args.compression_mode,
        num_post_encoder_layers=args.num_post_encoder_layers,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    compress_ratio = config.block_size ** 2
    print(f"CRISP {compress_ratio}x Evaluation")
    print("=" * 60)
    print(f"  Methods: {args.methods}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Samples per dataset: {args.n}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)

    # Setup GPU
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
            pope_t = metrics.get("pope_total", 0)
            mme_total = metrics.get("mme_total", 0)
            vqa_total = metrics.get("vqa_total", 0)

            if mme_total > 0:
                # MME: compute perception/cognition scores
                by_cat = defaultdict(lambda: {"yes_correct": 0, "yes_total": 0, "no_correct": 0, "no_total": 0})
                for r in result["results"]:
                    cat = r.get("category", "unknown")
                    gt_yn = r.get("gt_yn", r["gt"].lower().strip())
                    pred_yn = r.get("pred_yn", "no")
                    if gt_yn == "yes":
                        by_cat[cat]["yes_total"] += 1
                        if pred_yn == "yes":
                            by_cat[cat]["yes_correct"] += 1
                    else:
                        by_cat[cat]["no_total"] += 1
                        if pred_yn == "no":
                            by_cat[cat]["no_correct"] += 1

                perception_score = 0.0
                cognition_score = 0.0
                cat_details = {}
                for cat, counts in sorted(by_cat.items()):
                    acc_plus = counts["yes_correct"] / max(counts["yes_total"], 1)
                    acc_minus = counts["no_correct"] / max(counts["no_total"], 1)
                    cat_score = (acc_plus + acc_minus) * 100
                    cat_details[cat] = {
                        "score": round(cat_score, 1),
                        "acc_plus": round(acc_plus, 4),
                        "acc_minus": round(acc_minus, 4),
                        "total": counts["yes_total"] + counts["no_total"],
                    }
                    if cat in MME_PERCEPTION_CATS:
                        perception_score += cat_score
                    elif cat in MME_COGNITION_CATS:
                        cognition_score += cat_score

                summary[method][dataset] = {
                    "perception": round(perception_score, 1),
                    "cognition": round(cognition_score, 1),
                    "total_score": round(perception_score + cognition_score, 1),
                    "overall_accuracy": round(metrics["mme_correct"] / mme_total, 4),
                    "num_samples": mme_total,
                    "categories": cat_details,
                }
                print(f"  {method:12s} | {dataset:10s} | P: {perception_score:.0f}  C: {cognition_score:.0f}  Total: {perception_score+cognition_score:.0f}  Acc: {metrics['mme_correct']/mme_total:.1%}")

            elif vqa_total > 0:
                # VQAv2: standard VQA accuracy
                vqa_acc = metrics["vqa_acc_sum"] / vqa_total
                summary[method][dataset] = {
                    "vqa_accuracy": round(vqa_acc, 4),
                    "total": vqa_total,
                }
                print(f"  {method:12s} | {dataset:10s} | VQA Acc: {vqa_acc:.2%} (n={vqa_total})")

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

            # Mixed dataset (MC + QA): combined scoring
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
                dataset: result["results"]
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
        "# CRISP Evaluation Results",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Samples per dataset: {args.n}",
        f"- Compression ratio: {compress_ratio}x (block_size={config.block_size})",
        "",
        "## Results Comparison",
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
                elif "perception" in s:
                    row.append(f"P:{s['perception']:.0f} C:{s['cognition']:.0f}")
                elif "vqa_accuracy" in s:
                    row.append(f"{s['vqa_accuracy']:.2%}")
                elif "overall" in s:  # POPE
                    row.append(f"{s['overall']['accuracy']:.2%}")
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
    parser = argparse.ArgumentParser(description="CRISP Evaluation Script")

    parser.add_argument("--methods", nargs="+", default=["image_c3"],
                        help="Methods: image_c3, lowres, teacher")
    parser.add_argument("--datasets", nargs="+",
                        default=["mmmu", "mmbench", "realworldqa", "mathvista",
                                 "chartqa", "docvqa", "textvqa", "ocrbench",
                                 "mme", "scienceqa"],
                        help="Datasets to evaluate")
    parser.add_argument("--n", type=int, default=200,
                        help="Samples per dataset (0=all)")
    parser.add_argument("--decoder_path", type=str,
                        default="./Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_c3_ckpt", type=str,
                        default="./outputs/image_c3_v7_9x_3drope_20260301_103916/stage2/checkpoint-best")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--caption_max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=3,
                        help="Compression block size: 3=9x, 4=16x")
    parser.add_argument("--lowres_scale", type=float, default=0.333,
                        help="LowRes scale (0.333 for 9x, 0.25 for 16x)")
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--data_root", type=str, default="./data/eval")
    parser.add_argument("--output_dir", type=str,
                        default="./eval_results/crisp_eval")
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="Save checkpoint every N samples (0=disable)")

    # Ablation-specific config overrides
    parser.add_argument("--no_token_mixer", action="store_true",
                        help="Ablation: disable TokenMixer (A-noTR)")
    parser.add_argument("--no_rope_alignment", action="store_true",
                        help="Ablation: disable 3D M-RoPE alignment (A-noRoPE)")
    parser.add_argument("--no_relative_pos_embed", action="store_true",
                        help="Ablation: disable block relative position embedding (A-noRelPos)")
    parser.add_argument("--compression_mode", type=str, default="cross_attn",
                        choices=["cross_attn", "avg_pool"],
                        help="Ablation: compression mode (A-avgpool)")
    parser.add_argument("--num_post_encoder_layers", type=int, default=2,
                        help="Ablation: number of GTF layers (A-noGTF uses 0)")

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
