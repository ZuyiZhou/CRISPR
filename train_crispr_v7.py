#!/usr/bin/env python3
"""
Image-C3 V7 training script

Features:
1. 7B-only architecture: shared Vision Encoder, avoids cross-space alignment
2. TokenMixer + LocalC3-2x2: learnable compression, preserves detail
3. RoPE position alignment: keeps text positions consistent with the Teacher
4. Hidden-State Distillation: stronger distillation, improves detail fidelity
5. Staged training: Stage-1a trains LocalC3 only, Stage-1b trains TokenMixer + LocalC3
6. Multi-task support: Caption + OCR (prompt selected based on the source field)

Usage:
    # Single GPU quick sanity check
    python train_image_c3_v7.py \
        --data_path ./data/train/finevisionmax_200k/caption_train.jsonl \
        --output_dir ./outputs/image_c3_v7_test \
        --epochs 2 \
        --max_samples 1000

    # Stage-1a: train LocalC3 only (warmup)
    python train_image_c3_v7.py \
        --data_path ./data/train/stage1_caption/combined.jsonl \
        --output_dir ./outputs/image_c3_v7_stage1a \
        --epochs 1 \
        --freeze_token_mixer \
        --use_hidden_distillation

    # Stage-1b: train TokenMixer + LocalC3
    python train_image_c3_v7.py \
        --data_path ./data/train/stage2_mixed/combined.jsonl \
        --output_dir ./outputs/image_c3_v7_stage1b \
        --epochs 3 \
        --resume_from ./outputs/image_c3_v7_stage1a/checkpoint-best \
        --use_hidden_distillation

    # Multi-GPU training
    torchrun --nproc_per_node=4 train_image_c3_v7.py \
        --data_path ./data/train/stage2_mixed/combined.jsonl \
        --output_dir ./outputs/image_c3_v7_full \
        --epochs 5 \
        --use_hidden_distillation

Author: Image-C3 Team
Date: 2026-02-01
"""

import os
import sys
import json
import signal
import random
import shutil
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast

from tqdm import tqdm
from PIL import Image

# Set HF mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoProcessor, get_linear_schedule_with_warmup
from crispr.model_v7 import ImageC3ConfigV7, ImageC3ModelV7


# =============================================================================
# Dataset - supports Caption and OCR multi-task training
# =============================================================================

class MultiTaskDataset(Dataset):
    """
    V7 multi-task training dataset

    Supported data formats:
    1. Caption format: {"id": "...", "image": "...", "caption": "...", "source": "..."}
    2. Conversation format: {"id": "...", "image": "...", "conversations": [...], "source": "..."}

    The prompt is chosen based on the source field:
    - source contains "ocr" -> OCR prompt
    - otherwise -> Caption prompt
    """

    # Task prompt mapping
    TASK_PROMPTS = {
        "caption": "Describe this image.",
        "ocr": "What text is shown in this image?",
    }

    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 768,
        max_image_size: int = 448,
        min_image_size: int = 0,
        max_samples: Optional[int] = None,
    ):
        self.processor = processor
        self.max_length = max_length
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size

        # Load data
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
                    if max_samples and len(self.samples) >= max_samples:
                        break

        # Get data directory
        self.data_dir = Path(data_path).parent

        # Count task distribution
        self._count_tasks()

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")

    def _count_tasks(self):
        """Count the task distribution"""
        task_counts = {}
        for sample in self.samples:
            source = sample.get('source', 'unknown')
            task = self._get_task_type(source)
            task_counts[task] = task_counts.get(task, 0) + 1

        print(f"[Dataset] Task distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"  - {task}: {count} ({count/len(self.samples)*100:.1f}%)")

    def _get_task_type(self, source: str) -> str:
        """Determine the task type based on source"""
        if source and 'ocr' in source.lower():
            return 'ocr'
        return 'caption'

    def _get_prompt(self, source: str) -> str:
        """Return the prompt corresponding to source"""
        task = self._get_task_type(source)
        return self.TASK_PROMPTS.get(task, self.TASK_PROMPTS['caption'])

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess an image"""
        # Try several path combinations
        full_path = Path(image_path)
        if not full_path.exists():
            full_path = self.data_dir / image_path
        if not full_path.exists():
            # Try relative to the project root
            full_path = self.data_dir.parent / image_path
        if not full_path.exists():
            # Try relative to the data directory
            full_path = self.data_dir.parent.parent / image_path

        image = Image.open(full_path).convert("RGB")

        # Cap image size
        if self.max_image_size:
            w, h = image.size
            if max(w, h) > self.max_image_size:
                scale = self.max_image_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                image = image.resize((new_w, new_h), Image.LANCZOS)

        return image

    def _extract_text(self, sample: dict) -> Tuple[str, str]:
        """
        Extract the prompt and answer from a sample

        Returns:
            (prompt, answer)
        """
        source = sample.get('source', '')

        # Format 1: Caption format
        if 'caption' in sample:
            prompt = self._get_prompt(source)
            answer = sample['caption']
            return prompt, answer

        # Format 2: Conversation format
        if 'conversations' in sample:
            convs = sample['conversations']
            # Take the first turn
            if len(convs) >= 2:
                prompt = convs[0].get('content', '')
                answer = convs[1].get('content', '')
                return prompt, answer

        # Default
        return self._get_prompt(source), ""

    def _tokenize_sample(self, image, prompt, answer):
        """Convert (image, prompt, answer) into model input tensors"""
        # ============================
        # Teacher input (with image tokens)
        # ============================
        teacher_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]

        teacher_text = self.processor.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=False
        )
        teacher_inputs = self.processor(
            text=[teacher_text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Teacher prompt (user turn only), used to build labels
        teacher_prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        teacher_prompt_text = self.processor.apply_chat_template(
            teacher_prompt_messages, tokenize=False, add_generation_prompt=True
        )
        teacher_prompt_inputs = self.processor(
            text=[teacher_prompt_text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        teacher_input_ids = teacher_inputs["input_ids"].squeeze(0)
        teacher_attention_mask = teacher_inputs["attention_mask"].squeeze(0)
        teacher_labels = teacher_input_ids.clone()
        teacher_prompt_len = int(teacher_prompt_inputs["attention_mask"].sum().item())
        teacher_labels[:teacher_prompt_len] = -100
        teacher_labels[teacher_attention_mask == 0] = -100

        # ============================
        # Student input (text only, no image tokens)
        # ============================
        student_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]
        student_text = self.processor.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=False
        )
        student_inputs = self.processor(
            text=[student_text],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Student prompt
        student_prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        student_prompt_text = self.processor.apply_chat_template(
            student_prompt_messages, tokenize=False, add_generation_prompt=True
        )
        student_prompt_inputs = self.processor(
            text=[student_prompt_text],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = student_inputs["input_ids"].squeeze(0)
        attention_mask = student_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = int(student_prompt_inputs["attention_mask"].sum().item())
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        # ============================
        # Return result
        # ============================
        return {
            "pixel_values": teacher_inputs["pixel_values"].squeeze(0),
            "image_grid_thw": teacher_inputs["image_grid_thw"].squeeze(0),

            # Student
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,

            # Teacher
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_labels": teacher_labels,
        }

    def _try_get_item(self, idx):
        """Try to fetch a single sample, return None on failure"""
        sample = self.samples[idx]

        # Get image path
        image_path = sample.get('image') or sample.get('image_paths')
        if isinstance(image_path, list):
            image_path = image_path[0]
        image = self._load_image(image_path)

        # Filter out small images
        if self.min_image_size > 0:
            w, h = image.size
            if min(w, h) < self.min_image_size:
                return None

        # Get prompt and answer
        prompt, answer = self._extract_text(sample)

        if not answer:
            return None

        return image, prompt, answer

    def __getitem__(self, idx):
        # DDP-safe: must always return a valid sample, otherwise other ranks will hang on AllReduce
        max_retries = 10
        for attempt in range(max_retries):
            try:
                current_idx = idx if attempt == 0 else random.randint(0, len(self.samples) - 1)
                result = self._try_get_item(current_idx)
                if result is not None:
                    image, prompt, answer = result
                    return self._tokenize_sample(image, prompt, answer)
            except Exception as e:
                if attempt == 0:
                    print(f"[Dataset] Error at index {idx}: {e}")

        # Last resort: use the first sample
        try:
            result = self._try_get_item(0)
            if result is not None:
                image, prompt, answer = result
                return self._tokenize_sample(image, prompt, answer)
        except Exception:
            pass
        raise RuntimeError(f"[Dataset] Cannot find any valid sample after {max_retries} retries")


def collate_fn(batch):
    """Collate function"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    return {
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch]),

        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),

        "teacher_input_ids": torch.stack([b["teacher_input_ids"] for b in batch]),
        "teacher_attention_mask": torch.stack([b["teacher_attention_mask"] for b in batch]),
        "teacher_labels": torch.stack([b["teacher_labels"] for b in batch]),
    }


# =============================================================================
# Trainer
# =============================================================================

class ImageC3V7Trainer:
    """Image-C3 V7 trainer"""

    def __init__(
        self,
        args,
        rank=0,
        world_size=1,
        local_rank=0,
        is_distributed=False,
    ):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = is_distributed
        self.is_main = (rank == 0)

        # Set device
        if is_distributed:
            self.device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
        else:
            self.device = f"cuda:{args.gpu}"

        self._print_rank0(f"\n{'='*60}")
        self._print_rank0("Image-C3 V7 Training")
        self._print_rank0(f"{'='*60}")

        # Create output directory
        self.output_dir = Path(args.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._save_config()

        if is_distributed:
            dist.barrier()

        # Create model
        self._create_model()

        # Create data loader
        self._create_dataloader()

        # Create optimizer
        self._create_optimizer()

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self._resume_start_epoch = None
        self._resume_batch_idx = -1
        self._current_batch_idx = 0

        # Async checkpoint support: write to local /tmp first, then copy to the output filesystem
        self._async_save_threads = []
        self._local_tmp_dir = Path(f"/tmp/crisp_ckpt_{os.getpid()}")

        # Graceful interruption support
        self._should_stop = False
        if self.is_main:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

        # Full training resume (model weights + optimizer + scheduler + epoch)
        if getattr(self.args, 'resume_training_from', None):
            self._print_rank0(f"[V7] Full training resume from {self.args.resume_training_from}")
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            model_ref.load_checkpoint(self.args.resume_training_from)
            self._resume_start_epoch = self._load_trainer_state(self.args.resume_training_from)

    def _print_rank0(self, msg):
        if self.is_main:
            print(msg)

    def _handle_signal(self, signum, frame):
        """On receiving SIGTERM/SIGINT, set the stop flag"""
        sig_name = signal.Signals(signum).name
        self._print_rank0(f"\n[Trainer] Received {sig_name}, will save checkpoint and exit after current step...")
        self._should_stop = True

    def _check_should_stop(self):
        """Check whether a stop is requested (DDP-synchronized)"""
        if self.is_distributed:
            stop_tensor = torch.tensor(
                [1.0 if self._should_stop else 0.0],
                device=self.device,
            )
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            return stop_tensor.item() > 0.5
        return self._should_stop

    def _save_config(self):
        """Save the config"""
        config = {
            "decoder_path": self.args.decoder_path,
            "use_token_mixer": self.args.use_token_mixer,
            "token_mixer_num_layers": self.args.token_mixer_num_layers,
            "token_mixer_num_heads": self.args.token_mixer_num_heads,
            "token_mixer_dropout": self.args.token_mixer_dropout,
            "freeze_token_mixer": self.args.freeze_token_mixer,
            "use_local_c3": self.args.use_local_c3,
            "local_c3_block_size": self.args.local_c3_block_size,
            "local_c3_num_heads": self.args.local_c3_num_heads,
            "num_post_encoder_layers": self.args.num_post_encoder_layers,
            "use_rope_alignment": self.args.use_rope_alignment,
            "use_kl_distillation": self.args.use_kl_distillation,
            "kl_loss_weight": self.args.kl_loss_weight,
            "teacher_temperature": self.args.teacher_temperature,
            "use_hidden_distillation": self.args.use_hidden_distillation,
            "hidden_loss_weight": self.args.hidden_loss_weight,
            "use_decoder_lora": self.args.use_decoder_lora,
            "data_path": self.args.data_path,
            "max_length": self.args.max_length,
            "max_samples": self.args.max_samples,
            "max_image_size": self.args.max_image_size,
            "output_dir": self.args.output_dir,
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "lr_local_c3": self.args.lr_local_c3,
            "lr_token_mixer": self.args.lr_token_mixer,
            "lr_decoder_lora": self.args.lr_decoder_lora,
            "weight_decay": self.args.weight_decay,
            "warmup_ratio": self.args.warmup_ratio,
            "max_grad_norm": self.args.max_grad_norm,
            "log_steps": self.args.log_steps,
            "save_steps": self.args.save_steps,
            "gpu": self.args.gpu,
            "num_workers": self.args.num_workers,
            "resume_from": self.args.resume_from,
            "timestamp": datetime.now().isoformat(),
            "world_size": self.world_size,
            # Ablation flags
            "no_token_mixer": getattr(self.args, 'no_token_mixer', False),
            "no_relative_pos_embed": getattr(self.args, 'no_relative_pos_embed', False),
            "no_rope_alignment": getattr(self.args, 'no_rope_alignment', False),
            "no_kl_distillation": getattr(self.args, 'no_kl_distillation', False),
            "no_hidden_distillation": getattr(self.args, 'no_hidden_distillation', False),
            "compression_mode": getattr(self.args, 'compression_mode', 'cross_attn'),
            "hidden_loss_weight_final": getattr(self.args, 'hidden_loss_weight_final', None),
        }

        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def _create_model(self):
        """Create the V7 model"""
        self._print_rank0("[V7] Loading model...")

        # Apply --no_* ablation overrides
        use_token_mixer = self.args.use_token_mixer and not getattr(self.args, 'no_token_mixer', False)
        use_rope_alignment = self.args.use_rope_alignment and not getattr(self.args, 'no_rope_alignment', False)
        use_kl_distillation = self.args.use_kl_distillation and not getattr(self.args, 'no_kl_distillation', False)
        use_hidden_distillation = self.args.use_hidden_distillation and not getattr(self.args, 'no_hidden_distillation', False)
        use_relative_pos_embed = not getattr(self.args, 'no_relative_pos_embed', False)
        compression_mode = getattr(self.args, 'compression_mode', 'cross_attn')

        # Log ablation overrides
        ablation_flags = []
        if not use_token_mixer: ablation_flags.append("no_token_mixer")
        if not use_relative_pos_embed: ablation_flags.append("no_relative_pos_embed")
        if not use_rope_alignment: ablation_flags.append("no_rope_alignment")
        if not use_kl_distillation: ablation_flags.append("no_kl_distillation")
        if not use_hidden_distillation: ablation_flags.append("no_hidden_distillation")
        if compression_mode != 'cross_attn': ablation_flags.append(f"compression_mode={compression_mode}")
        if ablation_flags:
            self._print_rank0(f"[V7] Ablation flags: {', '.join(ablation_flags)}")

        config = ImageC3ConfigV7(
            decoder_path=self.args.decoder_path,
            use_token_mixer=use_token_mixer,
            token_mixer_num_layers=self.args.token_mixer_num_layers,
            token_mixer_num_heads=self.args.token_mixer_num_heads,
            token_mixer_dropout=self.args.token_mixer_dropout,
            use_local_c3=self.args.use_local_c3,
            local_c3_block_size=self.args.local_c3_block_size,
            local_c3_num_heads=self.args.local_c3_num_heads,
            num_post_encoder_layers=self.args.num_post_encoder_layers,
            use_rope_alignment=use_rope_alignment,
            use_kl_distillation=use_kl_distillation,
            kl_loss_weight=self.args.kl_loss_weight,
            teacher_temperature=self.args.teacher_temperature,
            use_hidden_distillation=use_hidden_distillation,
            hidden_loss_weight=self.args.hidden_loss_weight,
            use_decoder_lora=self.args.use_decoder_lora,
            use_relative_pos_embed=use_relative_pos_embed,
            compression_mode=compression_mode,
        )

        self.model = ImageC3ModelV7(config, device=self.device)
        self.model.setup()

        # Load checkpoint
        if self.args.resume_from:
            self._print_rank0(f"[V7] Loading checkpoint from {self.args.resume_from}")
            self.model.load_checkpoint(self.args.resume_from)

        # Freeze TokenMixer (Stage-1a)
        if self.args.freeze_token_mixer:
            self.model.freeze_token_mixer()

        # Print parameter statistics
        self._print_trainable_params()

        # DDP
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )

    def _print_trainable_params(self):
        """Print trainable parameters"""
        self._print_rank0(f"\n{'='*60}")
        self._print_rank0("Image-C3 V7 - Trainable Parameters")
        self._print_rank0(f"{'='*60}")

        model = self.model.module if hasattr(self.model, 'module') else self.model

        # TokenMixer
        if model.token_mixer is not None:
            mixer_params = sum(p.numel() for p in model.token_mixer.parameters())
            mixer_trainable = sum(p.numel() for p in model.token_mixer.parameters() if p.requires_grad)
            self._print_rank0(f"  TokenMixer: {mixer_trainable:,} / {mixer_params:,} params")

        # LocalC3
        if model.local_c3 is not None:
            c3_params = sum(p.numel() for p in model.local_c3.parameters())
            c3_trainable = sum(p.numel() for p in model.local_c3.parameters() if p.requires_grad)
            block_size = model.config.local_c3_block_size
            self._print_rank0(f"  LocalC3-{block_size}×{block_size}: {c3_trainable:,} / {c3_params:,} params")

        # Total
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._print_rank0(f"\n  Total trainable: {total_trainable:,} / {total_params:,} ({total_trainable/total_params*100:.4f}%)")
        self._print_rank0(f"{'='*60}\n")

    def _create_dataloader(self):
        """Create the data loader"""
        model = self.model.module if hasattr(self.model, 'module') else self.model

        self.dataset = MultiTaskDataset(
            data_path=self.args.data_path,
            processor=model.processor,
            max_length=self.args.max_length,
            max_image_size=self.args.max_image_size,
            min_image_size=getattr(self.args, 'min_image_size', 0),
            max_samples=self.args.max_samples,
        )

        if self.is_distributed:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            shuffle = False
        else:
            self.sampler = None
            shuffle = True

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        self._print_rank0(f"[Trainer] DataLoader: {len(self.dataloader)} batches")

    def _create_optimizer(self):
        """Create the optimizer"""
        model = self.model.module if hasattr(self.model, 'module') else self.model

        param_groups = []

        # LocalC3
        if model.local_c3 is not None:
            c3_params = [p for p in model.local_c3.parameters() if p.requires_grad]
            if c3_params:
                param_groups.append({
                    "params": c3_params,
                    "lr": self.args.lr_local_c3,
                    "name": "local_c3",
                })

        # TokenMixer
        if model.token_mixer is not None:
            mixer_params = [p for p in model.token_mixer.parameters() if p.requires_grad]
            if mixer_params:
                param_groups.append({
                    "params": mixer_params,
                    "lr": self.args.lr_token_mixer,
                    "name": "token_mixer",
                })

        # Decoder LoRA
        if model.config.use_decoder_lora:
            lora_params = [p for p in model.decoder.parameters() if p.requires_grad]
            if lora_params:
                param_groups.append({
                    "params": lora_params,
                    "lr": self.args.lr_decoder_lora,
                    "name": "decoder_lora",
                })

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
        )

        # Scheduler
        total_steps = len(self.dataloader) * self.args.epochs // self.args.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        self.total_steps = total_steps

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Hidden loss weight scheduling
        self.hidden_loss_weight_init = self.args.hidden_loss_weight
        self.hidden_loss_weight_final = getattr(self.args, 'hidden_loss_weight_final', None)
        if self.hidden_loss_weight_final is None:
            self.hidden_loss_weight_final = self.hidden_loss_weight_init
        self.current_hidden_loss_weight = self.hidden_loss_weight_init

        self._print_rank0(f"[Trainer] Total steps: {total_steps}, Warmup: {warmup_steps}")
        if self.hidden_loss_weight_final != self.hidden_loss_weight_init:
            self._print_rank0(
                f"[Trainer] Hidden loss weight schedule: {self.hidden_loss_weight_init} -> {self.hidden_loss_weight_final}"
            )

    def train(self):
        """Training loop"""
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Log file
        log_file = None
        if self.is_main:
            log_file = open(self.output_dir / "train_log.jsonl", 'a')

        self.model.train()

        start_epoch = 0
        if self._resume_start_epoch is not None:
            start_epoch = self._resume_start_epoch
            self._print_rank0(f"[Trainer] Resuming from epoch {start_epoch + 1}/{self.args.epochs}")

        for epoch in range(start_epoch, self.args.epochs):
            if self.is_distributed:
                self.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_steps = 0

            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch+1}/{self.args.epochs}",
                disable=not self.is_main,
            )

            self.optimizer.zero_grad()

            # Resume: compute how many batches need to be skipped
            skip_batches = 0
            if epoch == start_epoch and self._resume_batch_idx > 0:
                skip_batches = self._resume_batch_idx + 1
                self._print_rank0(
                    f"[Resume] Skipping {skip_batches} batches in epoch {epoch+1}"
                )

            actual_step_in_epoch = 0  # count of batches actually processed (for gradient accumulation alignment)

            for batch_idx, batch in enumerate(pbar):
                # Resume: skip already-processed batches (read only, don't count)
                if batch_idx < skip_batches:
                    if batch_idx % 2000 == 0 and self.is_main:
                        pbar.set_description(
                            f"Epoch {epoch+1}/{self.args.epochs} "
                            f"[skipping {batch_idx}/{skip_batches}]"
                        )
                    continue

                self._current_batch_idx = batch_idx

                if batch is None:
                    continue

                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(
                        pixel_values=batch["pixel_values"],
                        image_grid_thw=batch["image_grid_thw"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        teacher_input_ids=batch["teacher_input_ids"],
                        teacher_attention_mask=batch["teacher_attention_mask"],
                        teacher_labels=batch["teacher_labels"],
                    )

                    loss = outputs["loss"]
                    loss = loss / self.args.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                epoch_loss += loss.item() * self.args.gradient_accumulation_steps
                epoch_steps += 1
                actual_step_in_epoch += 1

                # Gradient accumulation (based on the actual number of batches processed, not batch_idx)
                if actual_step_in_epoch % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.args.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Update the hidden loss weight (linear interpolation)
                    if self.hidden_loss_weight_final != self.hidden_loss_weight_init and self.total_steps > 0:
                        progress = self.global_step / self.total_steps
                        self.current_hidden_loss_weight = (
                            self.hidden_loss_weight_init
                            + (self.hidden_loss_weight_final - self.hidden_loss_weight_init) * progress
                        )
                        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                        model_ref.config.hidden_loss_weight = self.current_hidden_loss_weight

                    # Check whether a graceful stop is requested
                    if self._check_should_stop():
                        if self.is_main:
                            self._wait_async_saves()
                        self._save_checkpoint("checkpoint-interrupted", epoch=epoch)
                        self._print_rank0("[Trainer] Interrupted checkpoint saved. Exiting.")
                        if log_file:
                            log_file.close()
                        if self.is_main and self._local_tmp_dir.exists():
                            shutil.rmtree(self._local_tmp_dir, ignore_errors=True)
                        return

                    # Logging
                    if self.global_step % self.args.log_steps == 0 and self.is_main:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.scheduler.get_last_lr()[0]

                        losses = outputs.get("losses", {})
                        log_entry = {
                            "step": self.global_step,
                            "epoch": epoch + 1,
                            "loss": avg_loss,
                            "lr": lr,
                            "ce_loss": losses.get("ce_loss", torch.tensor(0)).item() if isinstance(losses.get("ce_loss"), torch.Tensor) else losses.get("ce_loss", 0),
                            "kl_loss": losses.get("kl_loss", torch.tensor(0)).item() if isinstance(losses.get("kl_loss"), torch.Tensor) else losses.get("kl_loss", 0),
                            "hidden_loss": losses.get("hidden_loss", torch.tensor(0)).item() if isinstance(losses.get("hidden_loss"), torch.Tensor) else losses.get("hidden_loss", 0),
                            "hidden_loss_weight": self.current_hidden_loss_weight,
                        }

                        if log_file:
                            log_file.write(json.dumps(log_entry) + "\n")
                            log_file.flush()

                        pbar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        })

                    # Save checkpoint (keep best + latest, auto-clean older ones)
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"checkpoint-step_{self.global_step}", epoch=epoch)
                        # Update the best checkpoint (based on running avg loss)
                        current_avg_loss = epoch_loss / max(epoch_steps, 1)
                        if current_avg_loss < self.best_loss:
                            self.best_loss = current_avg_loss
                            self._save_checkpoint("checkpoint-best", epoch=epoch)
                        # Clean up old step checkpoints (keep best + latest)
                        self._cleanup_old_checkpoints(keep_latest=f"checkpoint-step_{self.global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            self._print_rank0(f"\n[Epoch {epoch+1}] Average Loss: {avg_epoch_loss:.4f}")

            # Save epoch checkpoint
            self._save_checkpoint(f"checkpoint-epoch_{epoch+1}", epoch=epoch)

            # Save best (uses the same comparison logic as step checkpoints)
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self._save_checkpoint("checkpoint-best", epoch=epoch)

        if log_file:
            log_file.close()

        # Wait for all async checkpoint saves to finish
        if self.is_main:
            self._wait_async_saves()
            # Cleanup local tmp dir
            if self._local_tmp_dir.exists():
                shutil.rmtree(self._local_tmp_dir, ignore_errors=True)

        self._print_rank0("\n[Trainer] Training complete!")

    def _save_checkpoint(self, name: str, epoch: int = 0):
        """Save a checkpoint (model weights + training state)

        Uses local /tmp as a staging area to avoid blocking on network storage I/O.
        Step 1 (synchronous): save to /tmp (fast, local SSD)
        Step 2 (async thread): copy from /tmp to the output_dir on network storage

        For critical checkpoints (best, epoch, interrupted), waits for the copy to finish.
        For step checkpoints, the copy runs in the background.
        """
        if self.is_main:
            model = self.model.module if hasattr(self.model, 'module') else self.model

            # Wait for any previous async save to finish before overwriting same name
            self._wait_async_saves(name)

            # Step 1: Save to local /tmp (fast)
            local_path = self._local_tmp_dir / name
            local_path.mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(str(local_path))

            trainer_state = {
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "epoch": epoch,
                "batch_idx": getattr(self, '_current_batch_idx', 0),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "world_size": self.world_size,
                "effective_batch_size": self.args.batch_size * self.args.gradient_accumulation_steps * self.world_size,
                "total_steps": self.total_steps,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "hidden_loss_weight": self.current_hidden_loss_weight,
            }
            torch.save(trainer_state, local_path / "trainer_state.pt")

            # Step 2: Copy to the output filesystem
            final_path = self.output_dir / name
            is_critical = name.startswith("checkpoint-best") or name.startswith("checkpoint-epoch") or name == "checkpoint-interrupted"

            if is_critical:
                # Critical checkpoints: copy synchronously
                self._copy_checkpoint(local_path, final_path)
                self._print_rank0(f"[Trainer] Saved checkpoint: {final_path}")
            else:
                # Step checkpoints: copy in background thread
                t = threading.Thread(
                    target=self._copy_checkpoint,
                    args=(local_path, final_path),
                    name=f"ckpt-save-{name}",
                    daemon=True,
                )
                t.start()
                self._async_save_threads.append((name, t))
                self._print_rank0(f"[Trainer] Saved checkpoint (async copy): {final_path}")

        # DDP barrier: all ranks wait for rank-0's local save (fast)
        if self.is_distributed:
            dist.barrier()

    def _copy_checkpoint(self, src: Path, dst: Path):
        """Copy checkpoint from local /tmp to the output filesystem, with atomic rename."""
        try:
            tmp_dst = dst.parent / f".tmp_{dst.name}"
            if tmp_dst.exists():
                shutil.rmtree(tmp_dst)
            shutil.copytree(str(src), str(tmp_dst))
            # Atomic rename (same filesystem)
            if dst.exists():
                shutil.rmtree(dst)
            tmp_dst.rename(dst)
        except Exception as e:
            print(f"[Trainer] WARNING: Failed to copy checkpoint {src} -> {dst}: {e}")
            # Fallback: direct save
            try:
                if not dst.exists():
                    shutil.copytree(str(src), str(dst))
            except Exception as e2:
                print(f"[Trainer] ERROR: Fallback save also failed: {e2}")

    def _wait_async_saves(self, name: str = None):
        """Wait for async save threads to finish. If name given, only wait for that name."""
        remaining = []
        for save_name, t in self._async_save_threads:
            if name is None or save_name == name:
                t.join(timeout=300)
                if t.is_alive():
                    print(f"[Trainer] WARNING: Async save '{save_name}' still running after 300s")
                    remaining.append((save_name, t))
            else:
                if t.is_alive():
                    remaining.append((save_name, t))
        self._async_save_threads = remaining

    def _cleanup_old_checkpoints(self, keep_latest: str):
        """Clean up old step checkpoints, keeping only best + latest + epoch checkpoints"""
        if not self.is_main:
            return
        for d in sorted(self.output_dir.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            if name.startswith(".tmp_"):
                shutil.rmtree(d, ignore_errors=True)
                continue
            # Keep: best, latest, epoch checkpoints, interrupted, sanity_check
            if name == "checkpoint-best" or name == keep_latest:
                continue
            if name.startswith("checkpoint-epoch_") or name == "checkpoint-interrupted":
                continue
            if not name.startswith("checkpoint-step_"):
                continue
            # Remove the old step checkpoint (output filesystem + local tmp)
            shutil.rmtree(d, ignore_errors=True)
            local_copy = self._local_tmp_dir / name
            if local_copy.exists():
                shutil.rmtree(local_copy, ignore_errors=True)
            self._print_rank0(f"[Trainer] Removed old checkpoint: {name}")

    def _load_trainer_state(self, checkpoint_path: str):
        """Load the training state (optimizer, scheduler, counters)

        Supports safe resume when the GPU count changes:
        - Same GPU count: restore the scheduler normally
        - Different GPU count: rebuild the scheduler (remaining epochs), restore hidden_loss_weight

        Also sets self._resume_batch_idx for precise mid-epoch resume.

        Returns:
            The restored starting epoch (i.e. saved_epoch), or None if there is no trainer_state.pt
        """
        state_file = Path(checkpoint_path) / "trainer_state.pt"
        if not state_file.exists():
            self._print_rank0(
                f"[Trainer] Warning: trainer_state.pt not found in {checkpoint_path}, "
                "only model weights will be loaded (optimizer/scheduler reset)"
            )
            return None

        state = torch.load(str(state_file), map_location="cpu")

        self.global_step = state["global_step"]
        self.best_loss = state["best_loss"]
        self.optimizer.load_state_dict(state["optimizer"])

        saved_world_size = state.get("world_size", self.world_size)
        saved_grad_accum = state.get("gradient_accumulation_steps", self.args.gradient_accumulation_steps)
        # Continue from the saved epoch (not +1), skipping already-processed batches within the epoch
        start_epoch = state["epoch"]
        saved_batch_idx = state.get("batch_idx", -1)

        if saved_world_size != self.world_size and saved_batch_idx > 0:
            # GPU count changed: convert batch_idx based on processed samples
            # old: each batch = batch_size * saved_world_size samples globally
            # new: each batch = batch_size * self.world_size samples globally
            # saved_batch_idx is per-GPU, so global samples = (saved_batch_idx + 1) * batch_size * saved_world_size
            samples_processed = (saved_batch_idx + 1) * self.args.batch_size * saved_world_size
            new_batches_per_gpu = samples_processed // (self.args.batch_size * self.world_size)
            self._print_rank0(
                f"[Resume] GPU count {saved_world_size} -> {self.world_size}, "
                f"batch_idx {saved_batch_idx} -> {new_batches_per_gpu - 1} "
                f"(samples processed: {samples_processed})"
            )
            self._resume_batch_idx = new_batches_per_gpu - 1
        else:
            self._resume_batch_idx = saved_batch_idx

        if saved_world_size == self.world_size:
            # Same GPU count: restore the scheduler normally
            self.scheduler.load_state_dict(state["scheduler"])
            self._print_rank0(
                f"[Trainer] Restored training state: "
                f"global_step={self.global_step}, best_loss={self.best_loss:.4f}, "
                f"resume from epoch {start_epoch + 1}, batch_idx {self._resume_batch_idx}"
            )
        else:
            # GPU count changed: rebuild the scheduler with the correct remaining steps
            new_batches_per_epoch = len(self.dataloader)
            batches_done_in_epoch = self._resume_batch_idx + 1 if self._resume_batch_idx >= 0 else 0
            remaining_batches_this_epoch = new_batches_per_epoch - batches_done_in_epoch
            remaining_full_epochs = self.args.epochs - start_epoch - 1
            total_remaining_batches = remaining_batches_this_epoch + remaining_full_epochs * new_batches_per_epoch
            remaining_steps = total_remaining_batches // self.args.gradient_accumulation_steps

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,  # Already past warmup, no need to warm up again
                num_training_steps=remaining_steps,
            )

            self._print_rank0(
                f"[Resume] scheduler rebuilt for {remaining_steps} remaining steps (warmup=0)"
            )

        # Restore hidden_loss_weight (regardless of whether the GPU count changed)
        saved_hidden_weight = state.get("hidden_loss_weight")
        if saved_hidden_weight is not None:
            self.current_hidden_loss_weight = saved_hidden_weight
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            model_ref.config.hidden_loss_weight = saved_hidden_weight
            self._print_rank0(f"[Resume] hidden_loss_weight resumed at {saved_hidden_weight:.4f}")

        return start_epoch


# =============================================================================
# Distributed training
# =============================================================================

def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        nccl_timeout_sec = int(os.environ.get("NCCL_TIMEOUT_SEC", "3600"))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=nccl_timeout_sec),
        )
        dist.barrier()

        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Tear down the distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Image-C3 V7")

    # Model path
    parser.add_argument("--decoder_path", type=str, default="./Qwen/Qwen2.5-VL-7B-Instruct")

    # Data path
    parser.add_argument("--data_path", type=str, required=True)

    # TokenMixer configuration
    parser.add_argument("--use_token_mixer", action="store_true", default=True)
    parser.add_argument("--token_mixer_num_layers", type=int, default=2)
    parser.add_argument("--token_mixer_num_heads", type=int, default=16)
    parser.add_argument("--token_mixer_dropout", type=float, default=0.1)
    parser.add_argument("--freeze_token_mixer", action="store_true", default=False)

    # LocalC3 configuration
    parser.add_argument("--use_local_c3", action="store_true", default=True)
    parser.add_argument("--local_c3_block_size", type=int, default=3,
                        help="Block size for NxN compression (2=4x, 3=9x, 4=16x)")
    parser.add_argument("--local_c3_num_heads", type=int, default=8)
    parser.add_argument("--num_post_encoder_layers", type=int, default=2)

    # RoPE
    parser.add_argument("--use_rope_alignment", action="store_true", default=True)

    # Distillation configuration
    parser.add_argument("--use_kl_distillation", action="store_true", default=True)
    parser.add_argument("--kl_loss_weight", type=float, default=0.5)
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--use_hidden_distillation", action="store_true", default=True)
    parser.add_argument("--hidden_loss_weight", type=float, default=0.1)
    parser.add_argument("--hidden_loss_weight_final", type=float, default=None,
                        help="Final hidden_loss_weight for linear schedule (default: same as hidden_loss_weight, no schedule)")

    # Decoder LoRA
    parser.add_argument("--use_decoder_lora", action="store_true", default=False)

    # Ablation flags (override defaults — use --no_* to disable components)
    parser.add_argument("--no_token_mixer", action="store_true", default=False,
                        help="Ablation: disable TokenMixer (A-noTR)")
    parser.add_argument("--no_relative_pos_embed", action="store_true", default=False,
                        help="Ablation: disable block-internal relative position embedding (A-noRelPos)")
    parser.add_argument("--no_rope_alignment", action="store_true", default=False,
                        help="Ablation: disable 3D M-RoPE alignment (A-noRoPE)")
    parser.add_argument("--no_kl_distillation", action="store_true", default=False,
                        help="Ablation: disable KL distillation loss (C-CE+HD, C-CEonly)")
    parser.add_argument("--no_hidden_distillation", action="store_true", default=False,
                        help="Ablation: disable hidden-state distillation loss (C-CE+KL, C-CEonly)")
    parser.add_argument("--compression_mode", type=str, default="cross_attn",
                        choices=["cross_attn", "avg_pool"],
                        help="Ablation: compression mechanism (A-avgpool)")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_image_size", type=int, default=448)
    parser.add_argument("--min_image_size", type=int, default=0,
                        help="Skip images with min(w,h) < this value (0 = no filter)")
    parser.add_argument("--max_samples", type=int, default=None)

    # Continuous training configuration (Stage-1a -> Stage-1b auto-transition)
    parser.add_argument("--continuous_training", action="store_true", default=False,
                        help="Enable continuous training: Stage-1a (freeze TokenMixer) → Stage-1b (full training)")
    parser.add_argument("--stage1a_epochs", type=int, default=1,
                        help="Epochs for Stage-1a (LocalC3 only, TokenMixer frozen)")
    parser.add_argument("--stage1b_epochs", type=int, default=3,
                        help="Epochs for Stage-1b (TokenMixer + LocalC3)")

    # Learning rate
    parser.add_argument("--lr_local_c3", type=float, default=1e-4)
    parser.add_argument("--lr_token_mixer", type=float, default=5e-5)
    parser.add_argument("--lr_decoder_lora", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/image_c3_v7")
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)

    # Hardware
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Load model weights only (for cross-stage resume, e.g. stage1a→1b)")
    parser.add_argument("--resume_training_from", type=str, default=None,
                        help="Full training resume: model weights + optimizer + scheduler + epoch "
                             "(for same-stage resume after interruption, supports GPU count changes)")

    # DDP
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def main():
    args = parse_args()

    # Distributed setup
    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main = (rank == 0)

    try:
        if args.continuous_training:
            # =============================================
            # Continuous training mode: Stage-1a -> Stage-1b
            # =============================================
            if is_main:
                print("\n" + "=" * 60)
                print("CONTINUOUS TRAINING MODE")
                print(f"  Stage-1a: {args.stage1a_epochs} epochs (LocalC3 only, TokenMixer frozen)")
                print(f"  Stage-1b: {args.stage1b_epochs} epochs (TokenMixer + LocalC3)")
                print(f"  Compression: {args.local_c3_block_size}x{args.local_c3_block_size} = {args.local_c3_block_size**2}x")
                print("=" * 60 + "\n")

            # ----------------
            # Stage-1a: train LocalC3 only
            # ----------------
            if is_main:
                print("\n" + "=" * 60)
                print(">>> STAGE-1a: Training LocalC3 only (TokenMixer frozen)")
                print("=" * 60 + "\n")

            args_1a = argparse.Namespace(**vars(args))
            args_1a.epochs = args.stage1a_epochs
            args_1a.freeze_token_mixer = True
            args_1a.output_dir = str(Path(args.output_dir) / "stage1a")

            trainer_1a = ImageC3V7Trainer(
                args=args_1a,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                is_distributed=is_distributed,
            )
            trainer_1a.train()

            # Get the best checkpoint path from Stage-1a
            stage1a_checkpoint = str(Path(args_1a.output_dir) / "checkpoint-best")

            if is_main:
                print(f"\n[Stage-1a] Completed. Best checkpoint: {stage1a_checkpoint}\n")

            if is_distributed:
                dist.barrier()

            # ----------------
            # Stage-1b: train TokenMixer + LocalC3
            # ----------------
            if is_main:
                print("\n" + "=" * 60)
                print(">>> STAGE-1b: Training TokenMixer + LocalC3")
                print("=" * 60 + "\n")

            args_1b = argparse.Namespace(**vars(args))
            args_1b.epochs = args.stage1b_epochs
            args_1b.freeze_token_mixer = False
            args_1b.resume_from = stage1a_checkpoint
            args_1b.output_dir = str(Path(args.output_dir) / "stage1b")

            trainer_1b = ImageC3V7Trainer(
                args=args_1b,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                is_distributed=is_distributed,
            )
            trainer_1b.train()

            if is_main:
                print("\n" + "=" * 60)
                print("CONTINUOUS TRAINING COMPLETED!")
                print(f"  Stage-1a checkpoint: {stage1a_checkpoint}")
                print(f"  Stage-1b checkpoint: {Path(args_1b.output_dir) / 'checkpoint-best'}")
                print("=" * 60 + "\n")

        else:
            # =============================================
            # Normal training mode
            # =============================================
            trainer = ImageC3V7Trainer(
                args=args,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                is_distributed=is_distributed,
            )

            trainer.train()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
