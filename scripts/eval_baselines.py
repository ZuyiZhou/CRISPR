#!/usr/bin/env python3
"""
Comprehensive baseline evaluation script for CRISP paper.

Compares Teacher (uncompressed) / CRISP / VisionZip / PruMerge+ / FastV
across multiple datasets and compression ratios (9x, 16x).

Academic fairness guarantees:
  - Identical max_pixels / min_pixels for all methods
  - Same prompt templates and scoring functions
  - Sequential position_ids for baselines (simple, fast)
  - Standard VQA accuracy for VQAv2 (min(#match/3, 1))
  - CIDEr/BLEU-4/METEOR/ROUGE-L for caption tasks (PTBTokenizer)
  - Greedy decoding (do_sample=False) for all methods
  - Uses model.generate() with SDPA for speed

Usage:
    # Full evaluation (9x + 16x, all methods, all datasets)
    CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_baselines.py \
        --ratios 9 16 \
        --methods teacher visionzip prumerge fastv image_c3 \
        --datasets vqav2 mmbench chartqa textvqa mathvista textcaps flickr30k \
        --n 0 \
        --image_c3_ckpt_9x ./outputs/crisp_9x/checkpoint-best \
        --image_c3_ckpt_16x ./outputs/crisp_16x/checkpoint-best \
        --output_dir eval_results/paper_baselines

    # Quick sanity check
    python scripts/eval_baselines.py --ratios 9 --methods teacher visionzip --datasets mmbench --n 50
"""

import os
import sys
import json
import argparse
import math
import gc
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse shared infrastructure from eval_9x_quick.py
from scripts.eval_9x_quick import (
    EvalConfig,
    DATASET_LOADERS,
    build_prompt,
    evaluate_samples,
    score_mc,
    score_qa,
    setup_gpu,
    _normalize_text,
    _save_checkpoint,
    # Models
    TeacherModel,
    ImageC3Model,
    # Prompt constants
    MC_PROMPT_SUFFIX,
    CHARTQA_PROMPT,
    DOCVQA_PROMPT,
    TEXTVQA_PROMPT,
    VQAV2_PROMPT,
)

from crispr.baselines import (
    AttentionCaptureHook,
    VisionZipCompressor,
    PruMergePlusCompressor,
)


# =============================================================================
# 3D M-RoPE position_ids for baselines (matches CRISP's build_position_ids_3d)
# =============================================================================

def build_position_ids_for_baselines(
    image_grid_thw: torch.Tensor,   # [B, 3] (B=1 for inference)
    keep_indices: torch.Tensor,      # [M] indices into post-merger token grid
    text_length: int,
    device: torch.device,
    prefix_length: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build 3D M-RoPE position_ids for baseline compressed tokens.

    Handles the full sequence layout:
        [prefix_text (P), vis_start (1), image_tokens (M), vis_end (1), suffix_text (S)]

    - prefix_text + vis_start: sequential positions [0..P] on all 3 dims
    - image_tokens: spatial (t=0, row, col) in post-merger grid
    - vis_end + suffix_text: sequential continuing from max spatial coord

    Args:
        prefix_length: number of tokens BEFORE the image (prefix_text + vis_start)
        text_length: number of NON-image tokens (prefix + vis_start + vis_end + suffix)

    Returns:
        position_ids: [3, 1, prefix_length + M + (text_length - prefix_length)]
        rope_deltas:  [1, 1]
    """
    t, h, w = image_grid_thw[0].tolist()
    t, h, w = int(t), int(h), int(w)

    # Post-merger grid dimensions (Qwen2.5-VL 2x2 spatial merge)
    h_in = h // 2
    w_in = w // 2

    M = len(keep_indices)
    total_len = M + text_length

    position_ids = torch.zeros(3, 1, total_len, device=device, dtype=torch.long)

    # --- Prefix text tokens (before image): sequential [0, prefix_length) ---
    if prefix_length > 0:
        pre_pos = torch.arange(prefix_length, device=device, dtype=torch.long)
        position_ids[0, 0, :prefix_length] = pre_pos
        position_ids[1, 0, :prefix_length] = pre_pos
        position_ids[2, 0, :prefix_length] = pre_pos

    # --- Image tokens: spatial coordinates ---
    img_start = prefix_length
    img_end = prefix_length + M
    if M > 0 and w_in > 0:
        ki = keep_indices.to(device=device, dtype=torch.long)
        rows = ki // w_in
        cols = ki % w_in

        position_ids[0, 0, img_start:img_end] = prefix_length  # temporal = constant
        position_ids[1, 0, img_start:img_end] = rows
        position_ids[2, 0, img_start:img_end] = cols

    # --- Suffix text tokens (after image): sequential continuing ---
    suffix_count = text_length - prefix_length  # vis_end + suffix_text
    if suffix_count > 0:
        # Continue from max of (prefix_length, max_spatial_coord)
        text_start = max(prefix_length, max(h_in, w_in)) if (h_in > 0 and w_in > 0) else prefix_length
        suffix_pos = torch.arange(text_start, text_start + suffix_count,
                                  device=device, dtype=torch.long)
        position_ids[0, 0, img_end:] = suffix_pos
        position_ids[1, 0, img_end:] = suffix_pos
        position_ids[2, 0, img_end:] = suffix_pos

        max_pos = text_start + suffix_count - 1
    else:
        max_pos = prefix_length

    rope_deltas = torch.tensor(
        [[(max_pos + 1) - total_len]], device=device, dtype=torch.long
    )

    return position_ids, rope_deltas


# =============================================================================
# Manual autoregressive generation (bypasses model.generate() for M-RoPE compat)
# =============================================================================

@torch.no_grad()
def manual_generate(
    model,               # Qwen2_5_VLForConditionalGeneration
    inputs_embeds,       # [1, seq_len, dim]
    attention_mask,      # [1, seq_len]
    position_ids_3d,     # [3, 1, seq_len]
    max_new_tokens: int = 128,
    eos_token_id: int = 151645,
) -> List[int]:
    """
    Manual autoregressive generation with 3D M-RoPE position_ids.

    model.generate() is incompatible with custom 3D position_ids because it
    auto-generates attention masks of wrong size. This function manually does
    prefill + decode, matching CRISP's generate_with_rope_3d approach.
    """
    device = inputs_embeds.device
    # Access the language model backbone
    language_model = model.model.language_model

    # 1. Prefill
    outputs = language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids_3d,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state
    logits = model.lm_head(hidden[:, -1:, :])  # [1, 1, V]

    # 2. Decode loop
    generated_ids = []
    next_pos_val = position_ids_3d[:, :, -1:] + 1  # [3, 1, 1]

    for step in range(max_new_tokens):
        next_token = logits[:, -1:, :].argmax(dim=-1)  # [1, 1]
        generated_ids.append(next_token.item())

        if next_token.item() == eos_token_id:
            break

        next_embed = model.get_input_embeddings()(next_token)  # [1, 1, D]
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(1, 1, device=device, dtype=torch.long),
        ], dim=1)

        outputs = language_model(
            inputs_embeds=next_embed,
            attention_mask=attention_mask,
            position_ids=next_pos_val,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state
        logits = model.lm_head(hidden[:, -1:, :])

        next_pos_val = next_pos_val + 1

    return generated_ids


# =============================================================================
# VisionZip Model
# =============================================================================

class VisionZipModel:
    """
    VisionZip: attention-based dominant token selection + cosine merge.
    Operates at vision encoder level (block 30).
    """

    def __init__(self, config: EvalConfig, device: str, budgets: float):
        self.config = config
        self.device = device
        self.budgets = budgets
        self.model = None
        self.processor = None
        self.hook = None
        self.compressor = VisionZipCompressor()

    def load(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"  [VisionZip] Loading {self.config.decoder_path} (budgets={self.budgets:.4f})")
        self.processor = AutoProcessor.from_pretrained(
            self.config.decoder_path,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.decoder_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        self.model.eval()

        # Register attention capture hook on vision encoder block 30
        self.hook = AttentionCaptureHook(target_block_idx=30)
        self.hook.register(self.model)
        print("  [VisionZip] Ready")

    def generate(self, image: Image.Image, prompt: str) -> str:
        self.hook.clear()

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        # Run vision encoder → capture attention
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            image_tokens = self.model.visual(pixel_values, grid_thw=image_grid_thw)

        attn_weights = self.hook.attn_weights
        metric_k = self.hook.metric_k
        if attn_weights is None:
            raise RuntimeError("AttentionCaptureHook did not capture attention weights")

        # Compress
        with torch.no_grad():
            selected_tokens, keep_indices = self.compressor(
                image_tokens, attn_weights, metric_k, self.budgets
            )

        # Build inputs_embeds = [prefix, vision_start, compressed_image, vision_end, suffix]
        selected_tokens = selected_tokens.unsqueeze(0)  # [1, M, D]

        # Split chat template around vision placeholder
        vision_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        vis_idx = text.find(vision_placeholder)
        prefix_text = text[:vis_idx]
        suffix_text = text[vis_idx + len(vision_placeholder):]

        tokenizer = self.processor.tokenizer
        embed_fn = self.model.get_input_embeddings()

        # Encode prefix/suffix text
        prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(self.device)
        suffix_ids = tokenizer(suffix_text, return_tensors="pt").input_ids.to(self.device)
        prefix_embeds = embed_fn(prefix_ids)  # [1, P, D]
        suffix_embeds = embed_fn(suffix_ids)  # [1, S, D]

        # Vision start/end special token embeddings
        vis_start_id = torch.tensor([[151652]], device=self.device)  # <|vision_start|>
        vis_end_id = torch.tensor([[151653]], device=self.device)    # <|vision_end|>
        vis_start_embed = embed_fn(vis_start_id)  # [1, 1, D]
        vis_end_embed = embed_fn(vis_end_id)      # [1, 1, D]

        inputs_embeds = torch.cat([
            prefix_embeds, vis_start_embed, selected_tokens, vis_end_embed, suffix_embeds
        ], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        # 3D M-RoPE position_ids preserving spatial coordinates of kept tokens
        prefix_len = prefix_embeds.shape[1] + 1  # prefix_text + vis_start
        text_length = prefix_len + 1 + suffix_embeds.shape[1]  # + vis_end + suffix
        position_ids, _ = build_position_ids_for_baselines(
            image_grid_thw, keep_indices, text_length, self.device,
            prefix_length=prefix_len,
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            token_ids = manual_generate(
                self.model, inputs_embeds, attention_mask, position_ids,
                max_new_tokens=self.config.max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def unload(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        del self.model, self.processor
        self.model = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# PruMerge+ Model
# =============================================================================

class PruMergePlusModel:
    """
    PruMerge+: IQR outlier detection + attention-weighted merge.
    Operates at vision encoder level (block 30).
    """

    def __init__(self, config: EvalConfig, device: str, budgets: float):
        self.config = config
        self.device = device
        self.budgets = budgets
        self.model = None
        self.processor = None
        self.hook = None
        self.compressor = PruMergePlusCompressor()

    def load(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"  [PruMerge+] Loading {self.config.decoder_path} (budgets={self.budgets:.4f})")
        self.processor = AutoProcessor.from_pretrained(
            self.config.decoder_path,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.decoder_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        self.model.eval()
        self.hook = AttentionCaptureHook(target_block_idx=30)
        self.hook.register(self.model)
        print("  [PruMerge+] Ready")

    def generate(self, image: Image.Image, prompt: str) -> str:
        self.hook.clear()

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            image_tokens = self.model.visual(pixel_values, grid_thw=image_grid_thw)

        attn_weights = self.hook.attn_weights
        metric_k = self.hook.metric_k
        if attn_weights is None:
            raise RuntimeError("AttentionCaptureHook did not capture attention weights")

        with torch.no_grad():
            selected_tokens, keep_indices = self.compressor(
                image_tokens, attn_weights, metric_k, self.budgets
            )

        # Build inputs_embeds = [prefix, vision_start, compressed_image, vision_end, suffix]
        selected_tokens = selected_tokens.unsqueeze(0)  # [1, M, D]

        vision_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        vis_idx = text.find(vision_placeholder)
        prefix_text = text[:vis_idx]
        suffix_text = text[vis_idx + len(vision_placeholder):]

        tokenizer = self.processor.tokenizer
        embed_fn = self.model.get_input_embeddings()

        prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(self.device)
        suffix_ids = tokenizer(suffix_text, return_tensors="pt").input_ids.to(self.device)
        prefix_embeds = embed_fn(prefix_ids)
        suffix_embeds = embed_fn(suffix_ids)

        vis_start_embed = embed_fn(torch.tensor([[151652]], device=self.device))
        vis_end_embed = embed_fn(torch.tensor([[151653]], device=self.device))

        inputs_embeds = torch.cat([
            prefix_embeds, vis_start_embed, selected_tokens, vis_end_embed, suffix_embeds
        ], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        # 3D M-RoPE position_ids preserving spatial coordinates of kept tokens
        text_length = prefix_embeds.shape[1] + 1 + 1 + suffix_embeds.shape[1]
        position_ids, _ = build_position_ids_for_baselines(
            image_grid_thw, keep_indices, text_length, self.device,
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            token_ids = manual_generate(
                self.model, inputs_embeds, attention_mask, position_ids,
                max_new_tokens=self.config.max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def unload(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        del self.model, self.processor
        self.model = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# FastV Model
# =============================================================================

class FastVModel:
    """
    FastV: prune low-attention image tokens at LLM layer 2.

    Two-stage approach (compatible with transformers 4.56+):
    1. Prefill pass through first (target-1) layers to capture attention weights
    2. Use attention to select top-k image tokens, then generate with pruned tokens
    """

    def __init__(self, config: EvalConfig, device: str, budgets: float,
                 target_layer: int = 2, use_last_token_attn: bool = False):
        self.config = config
        self.device = device
        self.budgets = budgets
        self.target_layer = target_layer
        self.use_last_token_attn = use_last_token_attn
        self.model = None
        self.processor = None

    def load(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"  [FastV] Loading {self.config.decoder_path} (budgets={self.budgets:.4f}, layer={self.target_layer})")
        self.processor = AutoProcessor.from_pretrained(
            self.config.decoder_path,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.decoder_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()
        print("  [FastV] Ready")

    def _compute_attention_scores(self, inputs_embeds, attention_mask, position_ids):
        """Run partial forward to layer (target-1) and manually compute attention weights."""
        language_model = self.model.model.language_model

        hidden_states = inputs_embeds
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)

        # Run through layers up to target_layer - 1
        for i, layer in enumerate(language_model.layers[:self.target_layer - 1]):
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        # Manually compute attention at the target layer (works with any attn_implementation)
        attn_module = language_model.layers[self.target_layer - 1].self_attn
        normed = language_model.layers[self.target_layer - 1].input_layernorm(hidden_states)

        bsz, seq_len, _ = normed.size()
        query_states = attn_module.q_proj(normed)
        key_states = attn_module.k_proj(normed)

        query_states = query_states.view(bsz, seq_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, attn_module.rope_scaling["mrope_section"]
            )

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv
        key_states = repeat_kv(key_states, attn_module.num_key_value_groups)

        attn = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(attn_module.head_dim)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=normed.device), diagonal=1)
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        return attn

    def generate(self, image: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        image_grid_thw = inputs["image_grid_thw"]
        image_token_id = self.model.config.image_token_id
        text_image_mask = (input_ids != image_token_id)  # True = text, False = image

        # Step 1: Get full inputs_embeds (image + text)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pixel_values = inputs["pixel_values"]
            image_tokens = self.model.visual(pixel_values, grid_thw=image_grid_thw)

            text_embeds = self.model.get_input_embeddings()(input_ids)
            image_mask = ~text_image_mask
            image_indices = image_mask[0].nonzero(as_tuple=True)[0]

            if len(image_indices) > 0 and len(image_tokens) > 0:
                full_embeds = text_embeds.clone()
                n_img = min(len(image_indices), len(image_tokens))
                full_embeds[0, image_indices[:n_img]] = image_tokens[:n_img].to(full_embeds.dtype)
            else:
                full_embeds = text_embeds

            # Use simple sequential position_ids for attention scoring
            seq_len = full_embeds.shape[1]
            pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
            position_ids = pos.unsqueeze(0).expand(3, -1, -1)

            # Step 2: Compute attention at layer (target-1)
            attn_weights = self._compute_attention_scores(full_embeds, None, position_ids)

        # Step 3: Identify image tokens and select top-k by attention
        image_positions = image_mask[0].nonzero(as_tuple=True)[0]
        if len(image_positions) == 0:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False)
            return self.processor.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        image_start = int(image_positions[0])
        image_end = int(image_positions[-1])
        image_length = image_end - image_start + 1

        if self.use_last_token_attn:
            img_attn = attn_weights.mean(dim=1)[0][-1][image_start:image_end + 1]
        else:
            img_attn = attn_weights.mean(dim=1)[0][:, image_start:image_end + 1].mean(dim=0)

        n_keep = max(1, int(image_length * self.budgets))
        top_img_indices = img_attn.topk(n_keep).indices.sort().values + image_start

        # Step 4: Build pruned sequence
        keep_indices = torch.cat([
            torch.arange(image_start, device=self.device),
            top_img_indices,
            torch.arange(image_end + 1, full_embeds.shape[1], device=self.device),
        ]).long()

        pruned_embeds = full_embeds[:, keep_indices, :]
        pruned_attn_mask = torch.ones(1, len(keep_indices), dtype=torch.long, device=self.device)

        # Simple sequential position_ids for pruned sequence
        pruned_len = pruned_embeds.shape[1]
        pruned_pos = torch.arange(pruned_len, device=self.device).unsqueeze(0)
        pruned_position_ids = pruned_pos.unsqueeze(0).expand(3, -1, -1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            token_ids = manual_generate(
                self.model, pruned_embeds, pruned_attn_mask, pruned_position_ids,
                max_new_tokens=self.config.max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def unload(self):
        del self.model, self.processor
        self.model = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# Scoring helpers
# =============================================================================

def compute_summary(method: str, dataset: str, result: Dict) -> Dict:
    """Compute summary metrics from evaluate_samples result."""
    metrics = result["metrics"]
    mc_t = metrics["mc_total"]
    qa_t = metrics["qa_total"]
    cap_t = metrics.get("caption_total", 0)
    vqav2_t = metrics.get("vqav2_total", 0)
    # Dataset-specific official scoring metrics
    chartqa_t = metrics.get("chartqa_total", 0)
    textvqa_t = metrics.get("textvqa_total", 0)
    anls_t = metrics.get("anls_total", 0)
    ocr_t = metrics.get("ocr_total", 0)

    # VQAv2
    if vqav2_t > 0:
        vqa_acc = metrics["vqav2_acc_sum"] / vqav2_t
        return {"metric": "VQA Acc", "value": round(vqa_acc, 4), "total": vqav2_t}

    # DocVQA: ANLS (official)
    if anls_t > 0:
        anls = metrics["anls_sum"] / anls_t
        return {"metric": "ANLS", "value": round(anls, 4), "total": anls_t}

    # TextVQA: EvalAI VQA Accuracy (official)
    if textvqa_t > 0:
        acc = metrics["textvqa_acc_sum"] / textvqa_t
        return {"metric": "VQA Acc", "value": round(acc, 4), "total": textvqa_t}

    # ChartQA: Relaxed Correctness (official)
    if chartqa_t > 0:
        acc = metrics["chartqa_correct"] / chartqa_t
        return {"metric": "Relaxed Acc", "value": round(acc, 4), "total": chartqa_t}

    # OCRBench: Substring matching (official)
    if ocr_t > 0:
        score = metrics["ocr_score_sum"] / ocr_t
        return {"metric": "Substring", "value": round(score, 4), "total": ocr_t}

    # Caption (CIDEr, BLEU-4, etc.)
    if cap_t > 0:
        try:
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

            gts, res = {}, {}
            for r in result["results"]:
                img_id = int(r["id"]) if str(r["id"]).isdigit() else r["id"]
                refs = r.get("all_captions", [r["gt"]])
                gts[img_id] = [{"caption": ref} for ref in refs]
                res[img_id] = [{"caption": r["pred"]}]

            tokenizer = PTBTokenizer()
            gts_tok = tokenizer.tokenize(gts)
            res_tok = tokenizer.tokenize(res)

            cider_score, _ = Cider().compute_score(gts_tok, res_tok)
            bleu_scores, _ = Bleu(4).compute_score(gts_tok, res_tok)
            bleu4 = bleu_scores[3]
            meteor_score, _ = Meteor().compute_score(gts_tok, res_tok)
            rouge_score, _ = Rouge().compute_score(gts_tok, res_tok)

            return {
                "metric": "CIDEr",
                "value": round(cider_score, 4),
                "CIDEr": round(cider_score, 4),
                "BLEU-4": round(bleu4, 4),
                "METEOR": round(meteor_score, 4),
                "ROUGE-L": round(rouge_score, 4),
                "total": cap_t,
            }
        except Exception as e:
            print(f"  Warning: Caption scoring failed for {method}/{dataset}: {e}")
            return {"metric": "CIDEr", "value": 0, "total": cap_t, "error": str(e)}

    # Mixed MC + QA (e.g., MathVista)
    if mc_t > 0 and qa_t > 0:
        combined = (metrics["mc_correct"] + metrics["qa_relaxed"]) / (mc_t + qa_t)
        return {"metric": "Accuracy", "value": round(combined, 4), "total": mc_t + qa_t}

    # MC only
    if mc_t > 0:
        acc = metrics["mc_correct"] / mc_t
        return {"metric": "Accuracy", "value": round(acc, 4), "total": mc_t}

    # QA only
    if qa_t > 0:
        relaxed = metrics["qa_relaxed"] / qa_t
        return {"metric": "Relaxed Acc", "value": round(relaxed, 4), "total": qa_t}

    return {"metric": "N/A", "value": 0, "total": 0}


# =============================================================================
# Main evaluation logic
# =============================================================================

def run_evaluation(args):
    """Run comprehensive baseline evaluation."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CRISP Paper — Comprehensive Baseline Evaluation")
    print("=" * 70)
    print(f"  Methods:     {args.methods}")
    print(f"  Ratios:      {args.ratios}")
    print(f"  Datasets:    {args.datasets}")
    print(f"  Samples/set: {args.n} (0=all)")
    print(f"  Output:      {output_dir}")
    print(f"  Decoder:     {args.decoder_path}")
    print(f"  max_pixels:  {args.max_pixels}")
    print("=" * 70)

    device = setup_gpu()

    # --- Load datasets (once, shared across all methods/ratios) ---
    print("\n[1/3] Loading datasets...")
    all_samples = {}
    for dataset in args.datasets:
        if dataset not in DATASET_LOADERS:
            print(f"  Warning: unknown dataset '{dataset}', available: {list(DATASET_LOADERS.keys())}")
            continue
        samples = DATASET_LOADERS[dataset](args.data_root, args.n, args.seed)
        if samples:
            all_samples[dataset] = samples

    if not all_samples:
        print("Error: no data loaded")
        return

    # --- Evaluate ---
    # Structure: all_results[ratio_str][method][dataset] = evaluate_samples result
    all_results = {}
    # summary[ratio_str][method][dataset] = {metric, value, ...}
    summary = {}

    # Teacher only needs to run once (ratio-independent)
    teacher_results = None

    for ratio in args.ratios:
        ratio_str = f"{ratio}x"
        budgets = 1.0 / ratio
        block_size = int(math.sqrt(ratio))

        print(f"\n{'=' * 70}")
        print(f"[2/3] Compression ratio: {ratio_str} (budgets={budgets:.4f})")
        print(f"{'=' * 70}")

        all_results[ratio_str] = {}
        summary[ratio_str] = {}

        # Build EvalConfig for this ratio
        config = EvalConfig(
            decoder_path=args.decoder_path,
            max_new_tokens=args.max_new_tokens,
            caption_max_new_tokens=args.caption_max_new_tokens,
            seed=args.seed,
            block_size=block_size,
            max_pixels=args.max_pixels,
            data_root=args.data_root,
        )

        for method in args.methods:
            print(f"\n  --- Method: {method} ({ratio_str}) ---")

            # Teacher: run once, reuse across ratios
            if method == "teacher":
                if teacher_results is None:
                    model = TeacherModel(config, device)
                    model.load()
                    teacher_results = {}
                    for dataset, samples in all_samples.items():
                        print(f"    Evaluating {dataset}...")
                        teacher_results[dataset] = evaluate_samples(
                            model, samples, f"teacher/{dataset}",
                            output_dir=output_dir,
                            checkpoint_interval=args.checkpoint_interval,
                        )
                    model.unload()

                all_results[ratio_str]["teacher"] = teacher_results
                summary[ratio_str]["teacher"] = {}
                for ds, res in teacher_results.items():
                    summary[ratio_str]["teacher"][ds] = compute_summary("teacher", ds, res)
                continue

            # CRISP
            if method == "image_c3":
                ckpt = getattr(args, f"image_c3_ckpt_{ratio}x", None)
                if not ckpt:
                    print(f"    Skipping image_c3 ({ratio_str}): no checkpoint (use --image_c3_ckpt_{ratio}x)")
                    continue
                if not Path(ckpt).exists():
                    print(f"    Skipping image_c3 ({ratio_str}): checkpoint not found: {ckpt}")
                    continue
                config.image_c3_ckpt = ckpt
                config.block_size = block_size
                model = ImageC3Model(config, device)
            elif method == "visionzip":
                model = VisionZipModel(config, device, budgets)
            elif method == "prumerge":
                model = PruMergePlusModel(config, device, budgets)
            elif method == "fastv":
                model = FastVModel(config, device, budgets, target_layer=args.fastv_target_layer)
            else:
                print(f"    Unknown method: {method}")
                continue

            model.load()

            method_results = {}
            for dataset, samples in all_samples.items():
                print(f"    Evaluating {dataset}...")
                method_results[dataset] = evaluate_samples(
                    model, samples, f"{method}_{ratio_str}/{dataset}",
                    output_dir=output_dir,
                    checkpoint_interval=args.checkpoint_interval,
                )

            model.unload()

            all_results[ratio_str][method] = method_results
            summary[ratio_str][method] = {}
            for ds, res in method_results.items():
                summary[ratio_str][method][ds] = compute_summary(method, ds, res)

    # --- Report ---
    print(f"\n{'=' * 70}")
    print("[3/3] Results Summary")
    print(f"{'=' * 70}")

    # Console output
    for ratio_str in sorted(summary.keys()):
        print(f"\n  === {ratio_str} Compression ===")
        methods_with_data = [m for m in args.methods if m in summary[ratio_str]]
        if not methods_with_data:
            continue

        header = f"  {'Dataset':12s}"
        for m in methods_with_data:
            header += f" | {m:>12s}"
        print(header)
        print("  " + "-" * len(header))

        for dataset in args.datasets:
            row = f"  {dataset:12s}"
            for m in methods_with_data:
                s = summary[ratio_str].get(m, {}).get(dataset)
                if s and s.get("value"):
                    val = s["value"]
                    if val < 1:
                        row += f" | {val:>11.2%}"
                    else:
                        row += f" | {val:>12.3f}"
                else:
                    row += f" | {'—':>12s}"
            print(row)

    # JSON report
    report = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "methods": args.methods,
            "ratios": args.ratios,
            "datasets": args.datasets,
            "n": args.n,
            "max_new_tokens": args.max_new_tokens,
            "caption_max_new_tokens": args.caption_max_new_tokens,
            "max_pixels": args.max_pixels,
            "seed": args.seed,
            "decoder_path": args.decoder_path,
            "fastv_target_layer": args.fastv_target_layer,
        },
        "summary": summary,
        "detailed_results": all_results,
    }

    report_path = output_dir / f"baseline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report: {report_path}")

    # Markdown report
    md_lines = [
        "# CRISP Baseline Comparison",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Decoder: {args.decoder_path}",
        f"- max_pixels: {args.max_pixels} ({args.max_pixels // 28 // 28}*28*28)",
        f"- Samples per dataset: {args.n} (0=all)",
        f"- Decoding: greedy (do_sample=False), max_new_tokens={args.max_new_tokens}",
        "",
    ]

    for ratio_str in sorted(summary.keys()):
        methods_with_data = [m for m in args.methods if m in summary[ratio_str]]
        if not methods_with_data:
            continue

        md_lines.append(f"## {ratio_str} Compression")
        md_lines.append("")

        # Determine primary metric per dataset
        header = "| Dataset | Metric | " + " | ".join(methods_with_data) + " |"
        sep = "|" + "---|" * (len(methods_with_data) + 2)
        md_lines.append(header)
        md_lines.append(sep)

        for dataset in args.datasets:
            # Get metric name from first method that has data
            metric_name = "—"
            for m in methods_with_data:
                s = summary[ratio_str].get(m, {}).get(dataset)
                if s:
                    metric_name = s.get("metric", "—")
                    break

            row = f"| {dataset} | {metric_name} |"
            for m in methods_with_data:
                s = summary[ratio_str].get(m, {}).get(dataset)
                if s and s.get("value"):
                    val = s["value"]
                    if val < 1:
                        row += f" {val:.2%} |"
                    else:
                        row += f" {val:.3f} |"
                else:
                    row += " — |"
            md_lines.append(row)

        md_lines.append("")

        # Caption detail table (if any caption datasets)
        cap_datasets = [d for d in args.datasets
                        if any(summary[ratio_str].get(m, {}).get(d, {}).get("CIDEr") is not None
                               for m in methods_with_data)]
        if cap_datasets:
            md_lines.append(f"### {ratio_str} Caption Details")
            md_lines.append("")
            cap_header = "| Dataset | Method | CIDEr | BLEU-4 | METEOR | ROUGE-L |"
            cap_sep = "|---|---|---|---|---|---|"
            md_lines.append(cap_header)
            md_lines.append(cap_sep)
            for ds in cap_datasets:
                for m in methods_with_data:
                    s = summary[ratio_str].get(m, {}).get(ds, {})
                    if s.get("CIDEr") is not None:
                        md_lines.append(
                            f"| {ds} | {m} | {s['CIDEr']:.3f} | {s['BLEU-4']:.3f} "
                            f"| {s['METEOR']:.3f} | {s['ROUGE-L']:.3f} |"
                        )
            md_lines.append("")

    md_path = output_dir / "BASELINE_COMPARISON.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown report: {md_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="CRISP Paper — Comprehensive Baseline Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument("--methods", nargs="+",
                        default=["teacher", "visionzip", "prumerge", "fastv", "image_c3"],
                        help="Methods to evaluate")
    parser.add_argument("--ratios", nargs="+", type=int, default=[9],
                        help="Compression ratios (e.g., 9 16)")
    parser.add_argument("--datasets", nargs="+",
                        default=["vqav2", "mmbench", "chartqa", "textvqa", "mathvista", "textcaps", "flickr30k"],
                        help="Evaluation datasets")
    parser.add_argument("--n", type=int, default=0,
                        help="Samples per dataset (0=all)")

    # Model
    parser.add_argument("--decoder_path", type=str,
                        default="./Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_c3_ckpt_9x", type=str, default="",
                        help="CRISP 9x checkpoint path")
    parser.add_argument("--image_c3_ckpt_16x", type=str, default="",
                        help="CRISP 16x checkpoint path")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--caption_max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--data_root", type=str, default="./data/eval")
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--output_dir", type=str,
                        default="./eval_results/baseline_comparison")

    # Method-specific
    parser.add_argument("--fastv_target_layer", type=int, default=2,
                        help="FastV pruning layer (default: 2)")
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="Save checkpoint every N samples (0=disable)")

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
