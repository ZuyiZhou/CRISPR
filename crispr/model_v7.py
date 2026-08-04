#!/usr/bin/env python3
"""
Image-C3 V7 模型

架构：
    - 7B-only：共享 Vision Encoder，避免跨空间对齐问题
    - TokenMixer：轻量语义重编码（2-4 层 Transformer Encoder）
    - LocalC3-2×2：V5.3 风格的局部结构化压缩
    - RoPE 位置对齐：确保文本位置与 Teacher 一致

训练流程：
    Stage-1a: 只训练 LocalC3（TokenMixer 冻结）
    Stage-1b: 训练 TokenMixer + LocalC3
    Stage-2:  精调（可选 hidden-state distillation）
    Stage-3:  可选 7B decoder LoRA 微调

作者: Image-C3 Team
日期: 2026-01-24
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor, AutoModelForVision2Seq

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class ImageC3ConfigV7:
    """V7 配置"""
    
    # 模型路径 - 只需要 7B（共享 visual encoder）
    decoder_path: str = "./Qwen/Qwen2.5-VL-7B-Instruct"
    
    # 维度（Qwen2.5-VL-7B-Instruct: hidden_size=3584）
    hidden_size: int = 3584
    
    # TokenMixer 配置（必选）
    use_token_mixer: bool = True
    token_mixer_num_layers: int = 2       # 2-4 层
    token_mixer_num_heads: int = 16       # 8 或 16
    token_mixer_ffn_expansion: int = 4    # D → 4D → D
    token_mixer_dropout: float = 0.1
    
    # LocalC3 配置（必选，支持 NxN 压缩）
    use_local_c3: bool = True
    local_c3_block_size: int = 3          # 3x3=9x 压缩（可选 2=4x, 3=9x, 4=16x）
    local_c3_num_heads: int = 8
    local_c3_dropout: float = 0.1
    use_relative_pos_embed: bool = True   # 相对位置 embedding
    num_post_encoder_layers: int = 2      # 压缩后的全局 self-attn
    compression_mode: str = "cross_attn"  # "cross_attn" or "avg_pool" (ablation)
    
    # RoPE 位置对齐（必选）
    use_rope_alignment: bool = True
    
    # 蒸馏配置
    use_kl_distillation: bool = True
    kl_loss_weight: float = 0.5
    teacher_temperature: float = 1.0
    
    # Hidden-State 蒸馏（推荐）
    use_hidden_distillation: bool = True
    hidden_loss_weight: float = 0.1
    hidden_distill_layers: List[int] = field(default_factory=lambda: [-1, -2])  # 最后 1-2 层
    
    # 冻结策略
    freeze_visual: bool = True    # 冻结 7B.visual
    freeze_decoder: bool = True   # 冻结 7B.decoder
    
    # 可选：7B decoder 轻量 LoRA（Stage-3 上限增强）
    use_decoder_lora: bool = False
    decoder_lora_r: int = 8
    decoder_lora_alpha: int = 16
    decoder_lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    decoder_lora_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # 前 4 层


# =============================================================================
# TokenMixer（必选模块）
# =============================================================================

class TokenMixerLayer(nn.Module):
    """
    TokenMixer 单层：Pre-Norm Transformer Encoder Layer
    
    只处理图像 tokens，不读取 prompt（prompt-agnostic）
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Pre-norm
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ffn_expansion, hidden_size),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            mask: [B, N] 有效位置为 1，padding 为 0
        
        Returns:
            x: [B, N, D]
        """
        # mask → key_padding_mask: True = 忽略
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        
        # Self-attention (pre-norm)
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(x)
        
        # FFN (pre-norm)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class TokenMixer(nn.Module):
    """
    TokenMixer：轻量语义重编码模块
    
    目标：在 LocalC3 压缩之前，对 image_tokens 做可学习的全局信息传播，
    使 tokens 变得"更可压缩"、更适合后续的局部聚合。
    
    特点：
    - prompt-agnostic：只看图像 tokens，不读取 prompt
    - 轻量：2-4 层 pre-norm Transformer encoder
    - 输入输出维度不变：[B, N_img, D] → [B, N_img, D]
    """
    
    def __init__(self, config: ImageC3ConfigV7):
        super().__init__()
        self.config = config
        
        self.layers = nn.ModuleList([
            TokenMixerLayer(
                hidden_size=config.hidden_size,
                num_heads=config.token_mixer_num_heads,
                ffn_expansion=config.token_mixer_ffn_expansion,
                dropout=config.token_mixer_dropout,
            )
            for _ in range(config.token_mixer_num_layers)
        ])
        
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
    
    def forward(
        self,
        image_tokens: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image_tokens: [B, N_img, D]
            image_mask: [B, N_img] 有效位置为 1，padding 为 0
        
        Returns:
            mixed_tokens: [B, N_img, D]
        """
        x = image_tokens
        
        for layer in self.layers:
            x = layer(x, image_mask)
        
        x = self.final_norm(x)
        
        return x


# =============================================================================
# LocalC3-2×2（必选模块，V5.3 风格）
# =============================================================================

class PostEncoderLayer(nn.Module):
    """压缩后的全局 self-attention 层"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            mask: [B, L] 有效位置为 1，padding 为 0
        """
        # mask → key_padding_mask: True = 忽略
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        
        # Self-attention (pre-norm)
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(x)
        
        # FFN (pre-norm)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class LocalC3(nn.Module):
    """
    LocalC3：可配置 block_size 的局部结构化压缩器
    
    输入: N × D (来自 TokenMixer 或直接 7B.visual)
    输出: (N / block_size^2) × D (送入 7B.decoder 作为 prefix)
    
    支持压缩率:
    - block_size=2: 2×2=4x 压缩
    - block_size=3: 3×3=9x 压缩
    - block_size=4: 4×4=16x 压缩
    
    流程:
    1. Reshape to H×W grid
    2. 按 NxN 分组 → (H/N)×(W/N) 组，每组 N² tokens
    3. 组内 cross-attn: 共享 query, N² tokens 作为 K/V → 输出 1 token
    4. 相对位置 embedding：N² 个位置
    5. FFN（维度不变，D → D）
    6. Post-encoder: 1-2 层全局 self-attn
    
    关键优势：保留局部高频细节（OCR/图表/数字）
    """
    
    def __init__(self, config: ImageC3ConfigV7):
        super().__init__()
        self.config = config
        self.block_size = config.local_c3_block_size  # N for NxN compression
        self.compression_mode = getattr(config, 'compression_mode', 'cross_attn')
        hidden_size = config.hidden_size
        dropout = config.local_c3_dropout

        num_positions = self.block_size ** 2  # N² positions per group

        # 1. 相对位置 embedding (N² positions)
        if config.use_relative_pos_embed:
            self.rel_pos_embed = nn.Parameter(
                torch.randn(num_positions, hidden_size) * 0.02  # [N², D]
            )
        else:
            self.rel_pos_embed = None

        # 2-3. Compression mechanism: cross-attn (default) or avg_pool (ablation)
        if self.compression_mode == 'cross_attn':
            self.query = nn.Parameter(torch.randn(1, hidden_size) * 0.02)  # [1, D]
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=config.local_c3_num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attn_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        else:
            # avg_pool: no learnable query or cross-attention needed
            self.query = None
            self.cross_attn = None
            self.cross_attn_norm = None
        
        # 4. FFN（维度不变）
        hidden_dim = hidden_size * 4
        self.ffn = nn.Sequential(
            nn.RMSNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        
        # 5. Post-encoder: 全局 self-attn
        self.post_encoder_layers = nn.ModuleList([
            PostEncoderLayer(hidden_size, config.local_c3_num_heads, dropout)
            for _ in range(config.num_post_encoder_layers)
        ])
        
        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        
        mode_str = f", mode={self.compression_mode}" if self.compression_mode != "cross_attn" else ""
        print(f"[LocalC3] Initialized with block_size={self.block_size} ({self.block_size}x{self.block_size}={num_positions}x compression{mode_str})")
    
    def forward(
        self,
        image_tokens: torch.Tensor,       # [B, N, D]
        image_grid_thw: torch.Tensor,     # [B, 3] 每个样本的 (t, h, w)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_tokens: [B, N, D] 来自 TokenMixer 或直接 7B.visual
            image_grid_thw: [B, 3] 每个样本的 (t, h, w)
        
        Returns:
            latent_tokens: [B, max_M, D] 压缩后的 latent tokens
            latent_mask: [B, max_M] 有效位置 mask (1=有效, 0=padding)
        """
        B = image_tokens.shape[0]
        device = image_tokens.device
        dtype = image_tokens.dtype
        N = self.block_size  # block size for NxN grouping
        
        compressed_list = []
        lengths = []
        
        for i in range(B):
            # --- 每个样本单独处理（因为 grid 可能不同）---
            t, h, w = image_grid_thw[i].tolist()
            t, h, w = int(t), int(h), int(w)
            
            # Qwen-VL 2x2 spatial_merge 后的网格尺寸
            h_in = h // 2
            w_in = w // 2
            n_tokens = h_in * w_in
            
            # 取有效 tokens
            x = image_tokens[i, :n_tokens, :]  # [n_tokens, D]
            
            # 处理边界：pad 到能被 N 整除
            h_pad = ((h_in + N - 1) // N) * N  # 向上取整到 N 的倍数
            w_pad = ((w_in + N - 1) // N) * N
            n_padded = h_pad * w_pad
            
            if n_padded > n_tokens:
                pad = torch.zeros(n_padded - n_tokens, x.shape[-1], device=device, dtype=dtype)
                x = torch.cat([x, pad], dim=0)
            
            # Reshape to 2D grid: [H_pad, W_pad, D]
            x = x.view(h_pad, w_pad, -1)
            
            # 按 NxN 分组: [H/N, W/N, N, N, D] → [num_groups, N², D]
            h_out, w_out = h_pad // N, w_pad // N
            x = x.view(h_out, N, w_out, N, -1)
            x = x.permute(0, 2, 1, 3, 4)  # [H/N, W/N, N, N, D]
            x = x.reshape(h_out * w_out, N * N, -1)  # [num_groups, N², D]
            num_groups = x.shape[0]
            
            # 加相对位置 embedding (N² positions)
            if self.rel_pos_embed is not None:
                x = x + self.rel_pos_embed.unsqueeze(0).to(dtype)  # [num_groups, N², D]

            # Compression: cross-attn or avg_pool
            if self.compression_mode == 'avg_pool':
                pooled = x.mean(dim=1)  # [num_groups, D]
            else:
                q = self.query.expand(num_groups, 1, -1).to(dtype)  # [num_groups, 1, D]
                attn_out, _ = self.cross_attn(q, x, x)    # [num_groups, 1, D]
                pooled = self.cross_attn_norm(attn_out.squeeze(1))  # [num_groups, D]

            # FFN
            compressed_i = self.ffn_norm(self.ffn(pooled) + pooled)  # [num_groups, D]
            
            compressed_list.append(compressed_i)
            lengths.append(num_groups)
        
        # Pad to max length
        max_len = max(lengths)
        padded = []
        masks = []
        
        for comp, length in zip(compressed_list, lengths):
            if comp.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - comp.shape[0], comp.shape[-1],
                    device=device, dtype=dtype
                )
                comp = torch.cat([comp, pad], dim=0)
            padded.append(comp)
            
            mask = torch.zeros(max_len, device=device, dtype=torch.long)
            mask[:length] = 1
            masks.append(mask)
        
        latent_tokens = torch.stack(padded, dim=0)  # [B, max_len, D]
        latent_mask = torch.stack(masks, dim=0)     # [B, max_len]
        
        # Post-encoder (全局 self-attn)
        for layer in self.post_encoder_layers:
            latent_tokens = layer(latent_tokens, latent_mask)
        
        latent_tokens = self.final_norm(latent_tokens)
        
        return latent_tokens, latent_mask


# Alias for backward compatibility
LocalC3_2x2 = LocalC3


# =============================================================================
# RoPE 位置对齐工具
# =============================================================================

def build_position_ids(
    image_grid_thw: torch.Tensor,  # [B, 3]
    latent_lengths: torch.Tensor,  # [B] 每个样本的 latent token 数量
    text_length: int,              # 文本序列长度
    device: torch.device,
    block_size: int = 3,           # NxN block size for compression
) -> torch.Tensor:
    """
    构造 position_ids，实现 RoPE 对齐（block 中心坐标）

    策略：
    - Teacher 视觉 token: N_img = (t * h * w) // 4，对应 merge 后的 h_in x w_in 网格
    - Student latent token: 对应 h_in x w_in 网格上的 NxN block
    - 对齐方式: 取每个 NxN block 的中心点，映射为 1D index

    Args:
        image_grid_thw: [B, 3] 每个样本的 (t, h, w)
        latent_lengths: [B] 每个样本的 latent token 数量 M
        text_length: 文本序列长度 L_txt
        device: 目标设备
        block_size: NxN 压缩的 block 大小 (默认 3 for 9x compression)

    Returns:
        position_ids: [B, max_M + L_txt]
    """
    B = image_grid_thw.shape[0]
    max_M = latent_lengths.max().item()
    total_len = max_M + text_length
    N = block_size

    position_ids = torch.zeros(B, total_len, device=device, dtype=torch.long)

    for i in range(B):
        t, h, w = image_grid_thw[i].tolist()
        t, h, w = int(t), int(h), int(w)

        # Teacher 的 N_img
        N_img = (t * h * w) // 4

        # merge 后网格尺寸
        h_in = h // 2
        w_in = w // 2
        
        # pad 到能被 N 整除
        h_pad = ((h_in + N - 1) // N) * N
        w_pad = ((w_in + N - 1) // N) * N
        h_out = h_pad // N
        w_out = w_pad // N

        # 当前样本的 latent 长度 M
        M = int(latent_lengths[i].item())
        if M > 0 and h_in > 0 and w_in > 0 and w_out > 0:
            idx = torch.arange(M, device=device, dtype=torch.long)
            r = idx // w_out
            c = idx % w_out

            # block 中心坐标 (N*r + N//2, N*c + N//2)
            center_r = N * r + N // 2
            center_c = N * c + N // 2

            # 边界截断，避免 padding 区域
            center_r = torch.clamp(center_r, max=h_in - 1)
            center_c = torch.clamp(center_c, max=w_in - 1)

            latent_pos = center_r * w_in + center_c
            if N_img > 0:
                latent_pos = torch.clamp(latent_pos, max=N_img - 1)
        else:
            latent_pos = torch.zeros(M, device=device, dtype=torch.long)

        # 文本位置：从 N_img 开始
        text_pos = torch.arange(N_img, N_img + text_length, device=device, dtype=torch.long)

        # 填充（考虑 padding）
        position_ids[i, :M] = latent_pos
        position_ids[i, max_M:] = text_pos

    return position_ids


def build_position_ids_3d(
    image_grid_thw: torch.Tensor,  # [B, 3]
    latent_lengths: torch.Tensor,  # [B] 每个样本的 latent token 数量
    text_length: int,              # 文本序列长度
    device: torch.device,
    block_size: int = 3,           # NxN block size for compression
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造 3D M-RoPE position_ids [3, B, L]，匹配 Qwen2.5-VL 的多模态 RoPE 格式。

    Qwen2.5-VL 使用 3D 位置编码:
      dim 0: temporal (静态图为 0)
      dim 1: height (行坐标)
      dim 2: width (列坐标)
    文本 token 的三个维度值相同（退化为 1D）。

    策略:
    - Latent token 继承其对应 NxN block 中心点的 2D 空间坐标
    - 文本 token 从 max(h_in, w_in) 开始递增（不与视觉位置重叠）

    Args:
        image_grid_thw: [B, 3] 每个样本的 (t, h, w)
        latent_lengths: [B] 每个样本的 latent token 数量 M
        text_length: 文本序列长度 L_txt
        device: 目标设备
        block_size: NxN 压缩的 block 大小

    Returns:
        position_ids: [3, B, max_M + L_txt]
        mrope_position_deltas: [B, 1] 用于自回归生成时的位置递增
    """
    B = image_grid_thw.shape[0]
    max_M = int(latent_lengths.max().item())
    total_len = max_M + text_length
    N = block_size

    position_ids = torch.zeros(3, B, total_len, device=device, dtype=torch.long)
    mrope_position_deltas = torch.zeros(B, 1, device=device, dtype=torch.long)

    for i in range(B):
        t, h, w = image_grid_thw[i].tolist()
        t, h, w = int(t), int(h), int(w)

        # merge 后网格尺寸 (Qwen-VL 的 2×2 spatial merge)
        h_in = h // 2
        w_in = w // 2

        # pad 到能被 N 整除
        h_pad = ((h_in + N - 1) // N) * N
        w_pad = ((w_in + N - 1) // N) * N
        h_out = h_pad // N
        w_out = w_pad // N

        M = int(latent_lengths[i].item())

        # === Latent token 位置 (视觉部分) ===
        if M > 0 and h_in > 0 and w_in > 0 and w_out > 0:
            idx = torch.arange(M, device=device, dtype=torch.long)
            r = idx // w_out
            c = idx % w_out

            center_r = torch.clamp(N * r + N // 2, max=h_in - 1)
            center_c = torch.clamp(N * c + N // 2, max=w_in - 1)

            # dim 0: temporal = 0 (静态图)
            position_ids[0, i, :M] = 0
            # dim 1: height = block 中心行坐标
            position_ids[1, i, :M] = center_r
            # dim 2: width = block 中心列坐标
            position_ids[2, i, :M] = center_c

        # === 文本 token 位置 ===
        # 文本从 max(vision_pos) + 1 开始，三个维度相同
        text_start = max(h_in, w_in) if (h_in > 0 and w_in > 0) else 1
        text_pos = torch.arange(text_start, text_start + text_length, device=device, dtype=torch.long)

        position_ids[0, i, max_M:] = text_pos
        position_ids[1, i, max_M:] = text_pos
        position_ids[2, i, max_M:] = text_pos

        # === rope_deltas ===
        # delta = max_position + 1 - total_sequence_length
        # 用于自回归 decode 时：new_token_pos = cache_position + delta
        max_pos = text_start + text_length - 1
        mrope_position_deltas[i] = (max_pos + 1) - total_len

    return position_ids, mrope_position_deltas


# =============================================================================
# 主模型 V7
# =============================================================================

class ImageC3ModelV7(nn.Module):
    """
    Image-C3 V7 模型
    
    架构:
        pixel_values → 7B.visual (frozen) → image_tokens
                     → TokenMixer (trainable) → mixed_tokens
                     → LocalC3-2×2 (trainable) → latent_tokens
                     → concat(latent_tokens, text_embeds) + position_ids
                     → 7B.decoder (frozen) → logits
    
    Teacher:
        pixel_values + prompt → 7B (frozen, multimodal) → teacher_logits
    """
    
    def __init__(self, config: ImageC3ConfigV7, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        
        # 延迟加载
        self.decoder = None
        self.visual = None
        self.processor = None
        self.token_mixer = None
        self.local_c3 = None
        
        self._initialized = False
    
    def setup(self):
        """加载模型并应用冻结策略"""
        if self._initialized:
            return
        
        print(f"[V7] Loading 7B from {self.config.decoder_path}...")
        
        # 选择 dtype
        device_str = str(self.device)
        model_dtype = torch.bfloat16 if "cuda" in device_str else torch.float32
        
        # 1. 加载完整的 7B 模型
        self.decoder = AutoModelForVision2Seq.from_pretrained(
            self.config.decoder_path,
            torch_dtype=model_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        
        # 2. 共享 Vision Encoder
        self.visual = self.decoder.visual
        
        # 3. 获取 hidden_size
        if hasattr(self.decoder.config, 'hidden_size'):
            hidden_size = self.decoder.config.hidden_size
        elif hasattr(self.decoder.config, 'text_config'):
            hidden_size = self.decoder.config.text_config.hidden_size
        else:
            hidden_size = 3584  # Qwen2.5-VL-7B 默认值
        
        self.config.hidden_size = hidden_size
        print(f"[V7] Hidden size: {hidden_size}")
        
        # 4. 创建 TokenMixer（必选）
        if self.config.use_token_mixer:
            self.token_mixer = TokenMixer(self.config)
            self.token_mixer = self.token_mixer.to(self.device).to(model_dtype)
            print(f"[V7] ✓ TokenMixer created: {self.config.token_mixer_num_layers} layers")
        
        # 5. 创建 LocalC3（必选，支持 NxN 压缩）
        if self.config.use_local_c3:
            self.local_c3 = LocalC3(self.config)
            self.local_c3 = self.local_c3.to(self.device).to(model_dtype)
            block_size = self.config.local_c3_block_size
            compression_ratio = block_size ** 2
            print(f"[V7] ✓ LocalC3-{block_size}×{block_size} created: {compression_ratio}x compression, {self.config.num_post_encoder_layers} post-encoder layers")
        
        # 6. 冻结策略
        self._apply_freeze()
        
        # 7. 可选：对 decoder 应用 LoRA
        if self.config.use_decoder_lora:
            self._apply_decoder_lora()
        
        # 8. 加载 Processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.decoder_path,
            trust_remote_code=True,
        )
        
        self._initialized = True
        self._print_trainable_params()
    
    def _apply_freeze(self):
        """应用冻结策略"""
        # 冻结 Vision Encoder
        if self.config.freeze_visual:
            for param in self.visual.parameters():
                param.requires_grad = False
            print("[V7] ✓ 7B.visual frozen")
        
        # 冻结 Decoder
        if self.config.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            print("[V7] ✓ 7B.decoder frozen")
    
    def _apply_decoder_lora(self):
        """对 decoder 的指定层应用 LoRA（可选，Stage-3 上限增强）"""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("Please install peft: pip install peft")
        
        lora_config = LoraConfig(
            r=self.config.decoder_lora_r,
            lora_alpha=self.config.decoder_lora_alpha,
            lora_dropout=0.05,
            target_modules=self.config.decoder_lora_target_modules,
            layers_to_transform=self.config.decoder_lora_layers,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.decoder = get_peft_model(self.decoder, lora_config)
        print(f"[V7] ✓ LoRA applied to decoder layers {self.config.decoder_lora_layers}")
        self.decoder.print_trainable_parameters()
    
    def _print_trainable_params(self):
        """打印可训练参数统计"""
        print("\n" + "=" * 60)
        print("Image-C3 V7 - Trainable Parameters")
        print("=" * 60)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if self.token_mixer is not None:
            mixer_params = sum(p.numel() for p in self.token_mixer.parameters())
            print(f"  TokenMixer: {mixer_params:,} params")
        
        if self.local_c3 is not None:
            c3_params = sum(p.numel() for p in self.local_c3.parameters())
            print(f"  LocalC3-2×2: {c3_params:,} params")
        
        print(f"\n  Total trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
        print("=" * 60 + "\n")
    
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用共享的 7B Vision Encoder 提取图像特征
        
        Returns:
            image_tokens: [B, max_N, D]
            image_mask: [B, max_N] 有效位置为 1
        """
        with torch.no_grad():
            features = self.visual(pixel_values, grid_thw=image_grid_thw)
            if isinstance(features, tuple):
                features = features[0]
        
        B = image_grid_thw.shape[0]
        device = features.device
        dtype = features.dtype
        
        # Qwen-VL 的 visual() 在 batch>1 时会把 tokens 拼接返回 [sum_tokens, hidden]
        if features.dim() == 2:
            if B == 1:
                image_mask = torch.ones(1, features.shape[0], device=device, dtype=torch.long)
                return features.unsqueeze(0), image_mask
            
            # 按样本拆分
            token_counts = []
            for i in range(B):
                t, h, w = image_grid_thw[i].tolist()
                token_counts.append((int(t) * int(h) * int(w)) // 4)
            
            feat_list = []
            start = 0
            for cnt in token_counts:
                feat_list.append(features[start:start + cnt])
                start += cnt
            
            max_len = max(token_counts)
            padded = []
            masks = []
            for f, cnt in zip(feat_list, token_counts):
                if f.shape[0] < max_len:
                    pad = torch.zeros(
                        max_len - f.shape[0], f.shape[1],
                        device=device, dtype=dtype
                    )
                    f = torch.cat([f, pad], dim=0)
                padded.append(f)
                
                mask = torch.zeros(max_len, device=device, dtype=torch.long)
                mask[:cnt] = 1
                masks.append(mask)
            
            image_tokens = torch.stack(padded, dim=0)
            image_mask = torch.stack(masks, dim=0)
            return image_tokens, image_mask
        
        # 已经是 3D
        image_mask = torch.ones(features.shape[:2], device=device, dtype=torch.long)
        return features, image_mask
    
    def compress(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整压缩流程: visual → TokenMixer → LocalC3-2×2
        
        Returns:
            latent_tokens: [B, M, D]
            latent_mask: [B, M]
        """
        # 1. 获取 image tokens
        image_tokens, image_mask = self.get_image_features(pixel_values, image_grid_thw)
        
        # 2. TokenMixer（如果启用）
        if self.token_mixer is not None:
            mixed_tokens = self.token_mixer(image_tokens, image_mask)
        else:
            mixed_tokens = image_tokens
        
        # 3. LocalC3-2×2 压缩
        if self.local_c3 is not None:
            latent_tokens, latent_mask = self.local_c3(mixed_tokens, image_grid_thw)
        else:
            # 如果没有 LocalC3，直接返回（不压缩）
            latent_tokens, latent_mask = mixed_tokens, image_mask
        
        return latent_tokens, latent_mask
    
    @torch.no_grad()
    def get_teacher_outputs(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        获取 Teacher (7B multimodal) 的输出
        """
        outputs = self.decoder(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        result = {
            "logits": outputs.logits,
        }
        
        # Soft labels for KL distillation
        T = self.config.teacher_temperature
        if T > 0:
            result["soft_logits"] = outputs.logits / T
        else:
            result["soft_logits"] = outputs.logits
        
        # Hidden states（用于 hidden-state distillation）
        if output_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states
        
        return result
    
    def forward_with_distillation(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,           # student 文本 (无 image token)
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        teacher_input_ids: torch.Tensor,   # teacher 文本 (有 image token)
        teacher_attention_mask: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        蒸馏训练前向传播
        """
        B = image_grid_thw.shape[0]
        device = pixel_values.device
        
        # === 1. Student 分支 ===
        # 1.1 压缩: visual → TokenMixer → LocalC3
        latent_tokens, latent_mask = self.compress(pixel_values, image_grid_thw)
        latent_len = latent_tokens.shape[1]
        
        # 1.2 获取 7B 文本 embeddings
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        
        # 1.3 拼接: [latent, text]
        inputs_embeds = torch.cat([latent_tokens, text_embeds], dim=1)
        
        # 1.4 拼接 attention_mask
        combined_attention_mask = torch.cat([latent_mask, attention_mask], dim=1)
        
        # 1.5 构造 3D position_ids（M-RoPE 对齐）
        position_ids = None
        if self.config.use_rope_alignment:
            latent_lengths = latent_mask.sum(dim=1)  # [B]
            position_ids, _ = build_position_ids_3d(
                image_grid_thw=image_grid_thw,
                latent_lengths=latent_lengths,
                text_length=input_ids.shape[1],
                device=device,
                block_size=self.config.local_c3_block_size,
            )

        # 1.6 通过 7B decoder
        student_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            position_ids=position_ids,
            output_hidden_states=self.config.use_hidden_distillation,
        )
        student_logits = student_outputs.logits
        
        # === 2. Teacher 分支 (multimodal, no_grad) — skip if no distillation ===
        need_teacher = self.config.use_kl_distillation or self.config.use_hidden_distillation
        teacher_outputs = None
        if need_teacher:
            teacher_outputs = self.get_teacher_outputs(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                output_hidden_states=self.config.use_hidden_distillation,
            )

        # === 3. 计算损失 ===
        losses = {}

        # 3.1 CE Loss (student)
        label_padding = torch.full(
            (B, latent_len), -100,
            device=labels.device, dtype=labels.dtype
        )
        combined_labels = torch.cat([label_padding, labels], dim=1)

        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = combined_labels[..., 1:].contiguous()

        valid_mask = (shift_labels.view(-1) != -100)
        num_valid = valid_mask.sum().item()

        if num_valid > 0:
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            ce_loss = student_logits.sum() * 0.0

        losses["ce_loss"] = ce_loss

        # 3.2 KL Loss (labels-mask 对齐)
        if self.config.use_kl_distillation and teacher_outputs is not None:
            kl_loss = self._compute_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_outputs["soft_logits"],
                student_labels=combined_labels,
                teacher_labels=teacher_labels,
            )
            losses["kl_loss"] = kl_loss

        # 3.3 Hidden-State Distillation
        if self.config.use_hidden_distillation and teacher_outputs is not None:
            hidden_loss = self._compute_hidden_loss(
                student_hidden_states=student_outputs.hidden_states,
                teacher_hidden_states=teacher_outputs.get("hidden_states"),
                student_labels=combined_labels,
                teacher_labels=teacher_labels,
            )
            losses["hidden_loss"] = hidden_loss

        # 3.4 总损失
        total_loss = ce_loss
        if "kl_loss" in losses:
            total_loss = total_loss + self.config.kl_loss_weight * losses["kl_loss"]
        if "hidden_loss" in losses:
            total_loss = total_loss + self.config.hidden_loss_weight * losses["hidden_loss"]
        
        losses["total_loss"] = total_loss
        
        return {
            "loss": total_loss,
            "losses": losses,
            "student_logits": student_logits,
            "latent_tokens": latent_tokens,
            "latent_mask": latent_mask,
        }
    
    def _compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        用 labels-mask 对齐答案 token 计算 KL
        """
        B = student_logits.shape[0]
        T = self.config.teacher_temperature if self.config.teacher_temperature > 0 else 1.0
        device = student_logits.device
        
        sample_kl_losses = []
        
        for i in range(B):
            # Shift for next-token prediction
            s_logits = student_logits[i, :-1, :]
            t_logits = teacher_logits[i, :-1, :]
            s_labels = student_labels[i, 1:]
            t_labels = teacher_labels[i, 1:]
            
            # 用 labels mask 选取答案 token
            s_mask = (s_labels != -100)
            t_mask = (t_labels != -100)
            
            if s_mask.sum() == 0 or t_mask.sum() == 0:
                continue
            
            s_sel = s_logits[s_mask]  # [Ns, V]
            t_sel = t_logits[t_mask]  # [Nt, V]
            
            # 对齐长度
            m = min(s_sel.shape[0], t_sel.shape[0])
            if m == 0:
                continue
            
            s_sel = s_sel[:m]
            t_sel = t_sel[:m]
            
            # KL divergence
            student_log_probs = F.log_softmax(s_sel / T, dim=-1)
            teacher_probs = F.softmax(t_sel, dim=-1)  # teacher 已除过 T
            
            kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            sample_kl_losses.append(kl)
        
        if not sample_kl_losses:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(sample_kl_losses).mean() * (T ** 2)
    
    def _compute_hidden_loss(
        self,
        student_hidden_states: Tuple[torch.Tensor, ...],
        teacher_hidden_states: Optional[Tuple[torch.Tensor, ...]],
        student_labels: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Hidden-State Distillation Loss
        
        对齐最后 1-2 层的 hidden states（在答案 token 位置）
        """
        device = student_labels.device
        
        if teacher_hidden_states is None or student_hidden_states is None:
            return torch.tensor(0.0, device=device)
        
        B = student_labels.shape[0]
        losses = []
        
        # 对指定层计算 loss
        for layer_idx in self.config.hidden_distill_layers:
            try:
                s_hidden = student_hidden_states[layer_idx]  # [B, L_s, D]
                t_hidden = teacher_hidden_states[layer_idx]  # [B, L_t, D]
            except IndexError:
                continue
            
            for i in range(B):
                s_labels = student_labels[i]
                t_labels = teacher_labels[i]
                
                # 用 labels mask 选取答案位置
                s_mask = (s_labels != -100)
                t_mask = (t_labels != -100)
                
                if s_mask.sum() == 0 or t_mask.sum() == 0:
                    continue
                
                s_sel = s_hidden[i][s_mask]  # [Ns, D]
                t_sel = t_hidden[i][t_mask]  # [Nt, D]
                
                # 对齐长度
                m = min(s_sel.shape[0], t_sel.shape[0])
                if m == 0:
                    continue
                
                s_sel = s_sel[:m]
                t_sel = t_sel[:m]
                
                # L2 loss
                loss = F.mse_loss(s_sel, t_sel)
                losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(losses).mean()
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        teacher_input_ids: Optional[torch.Tensor] = None,
        teacher_attention_mask: Optional[torch.Tensor] = None,
        teacher_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """前向传播"""
        if self.training and labels is not None and teacher_input_ids is not None:
            return self.forward_with_distillation(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                teacher_input_ids=teacher_input_ids,
                teacher_attention_mask=teacher_attention_mask,
                teacher_labels=teacher_labels,
            )
        else:
            # 推理模式
            latent_tokens, latent_mask = self.compress(pixel_values, image_grid_thw)
            
            text_embeds = self.decoder.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([latent_tokens, text_embeds], dim=1)
            combined_attention_mask = torch.cat([latent_mask, attention_mask], dim=1)
            
            # 3D Position IDs (M-RoPE)
            position_ids = None
            if self.config.use_rope_alignment:
                latent_lengths = latent_mask.sum(dim=1)
                position_ids, _ = build_position_ids_3d(
                    image_grid_thw=image_grid_thw,
                    latent_lengths=latent_lengths,
                    text_length=input_ids.shape[1],
                    device=input_ids.device,
                    block_size=self.config.local_c3_block_size,
                )

            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention_mask,
                position_ids=position_ids,
                output_hidden_states=False,
            )
            
            return {
                "logits": outputs.logits,
                "latent_tokens": latent_tokens,
                "latent_mask": latent_mask,
            }
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> torch.Tensor:
        """生成文本"""
        # 1. 压缩
        latent_tokens, latent_mask = self.compress(pixel_values, image_grid_thw)
        
        # 2. 获取 prompt embeddings
        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids)
        
        # 3. 拼接
        inputs_embeds = torch.cat([latent_tokens, prompt_embeds], dim=1)
        attention_mask = torch.cat([
            latent_mask,
            torch.ones(prompt_ids.shape, device=prompt_ids.device, dtype=torch.long)
        ], dim=1)
        
        # 4. 3D Position IDs (M-RoPE) — 注意: generate() 受 KV cache 限制，
        #    推理推荐用 generate_with_rope_3d() 替代
        position_ids = None
        if self.config.use_rope_alignment:
            latent_lengths = latent_mask.sum(dim=1)
            position_ids, _ = build_position_ids_3d(
                image_grid_thw=image_grid_thw,
                latent_lengths=latent_lengths,
                text_length=prompt_ids.shape[1],
                device=prompt_ids.device,
                block_size=self.config.local_c3_block_size,
            )

        # 5. 生成
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generate_kwargs,
        )
        
        return outputs

    @torch.no_grad()
    def generate_with_rope_3d(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        手写自回归生成循环，支持 3D M-RoPE position_ids + KV cache。

        绕过 model.generate()，在每一步显式传入正确的 3D position_ids，
        解决 transformers KV cache 与自定义 position_ids 不兼容的问题。
        """
        device = pixel_values.device
        B = image_grid_thw.shape[0]
        eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id

        # 1. 压缩
        latent_tokens, latent_mask = self.compress(pixel_values, image_grid_thw)
        latent_len = latent_tokens.shape[1]

        # 2. Prompt embeddings
        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids)

        # 3. 拼接 inputs_embeds 和 attention_mask
        inputs_embeds = torch.cat([latent_tokens, prompt_embeds], dim=1)
        attention_mask = torch.cat([
            latent_mask,
            torch.ones(prompt_ids.shape, device=device, dtype=torch.long),
        ], dim=1)
        seq_len = inputs_embeds.shape[1]

        # 4. 构造 3D position_ids [3, B, L]
        latent_lengths = latent_mask.sum(dim=1)
        position_ids_3d, mrope_deltas = build_position_ids_3d(
            image_grid_thw=image_grid_thw,
            latent_lengths=latent_lengths,
            text_length=prompt_ids.shape[1],
            device=device,
            block_size=self.config.local_c3_block_size,
        )

        # 5. Prefill: 用完整序列做一次前向，启用 KV cache
        outputs = self.decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids_3d,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        # 取最后一个 hidden state 过 lm_head 得到 logits
        hidden = outputs.last_hidden_state
        logits = self.decoder.lm_head(hidden[:, -1:, :])  # [B, 1, V]

        # 6. 自回归 decode loop
        generated_ids = []
        # 下一个 token 的位置 = prefill 最后一个位置 + 1
        # position_ids_3d[:, :, -1] 是最后一个文本 token 的位置
        next_pos_val = position_ids_3d[:, :, -1:] + 1  # [3, B, 1]

        for step in range(max_new_tokens):
            # Sampling / greedy
            if do_sample and temperature > 0:
                scaled_logits = logits[:, -1, :] / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                next_token = logits[:, -1:, :].argmax(dim=-1)  # [B, 1]

            generated_ids.append(next_token)

            # EOS check
            if B == 1 and next_token.item() == eos_token_id:
                break

            # Embed next token
            next_embed = self.decoder.get_input_embeddings()(next_token)  # [B, 1, D]

            # Attention mask 扩展
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(B, 1, device=device, dtype=torch.long),
            ], dim=1)

            # 3D position for new token: 文本 token 三个维度相同
            step_position_ids = next_pos_val  # [3, B, 1]

            # Decode step with KV cache
            outputs = self.decoder.model(
                inputs_embeds=next_embed,
                attention_mask=attention_mask,
                position_ids=step_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            hidden = outputs.last_hidden_state
            logits = self.decoder.lm_head(hidden[:, -1:, :])

            # 递增位置
            next_pos_val = next_pos_val + 1

        if generated_ids:
            return torch.cat(generated_ids, dim=1)  # [B, num_generated]
        else:
            return torch.tensor([[eos_token_id]], device=device)

    def get_trainable_params(self) -> List[Dict[str, Any]]:
        """获取可训练参数组（用于分组学习率）"""
        param_groups = []
        
        # LocalC3
        if self.local_c3 is not None:
            c3_params = [p for p in self.local_c3.parameters() if p.requires_grad]
            if c3_params:
                param_groups.append({
                    "params": c3_params,
                    "lr": 1e-4,
                    "name": "local_c3",
                })
        
        # TokenMixer
        if self.token_mixer is not None:
            mixer_params = [p for p in self.token_mixer.parameters() if p.requires_grad]
            if mixer_params:
                param_groups.append({
                    "params": mixer_params,
                    "lr": 5e-5,
                    "name": "token_mixer",
                })
        
        # Decoder LoRA (如果启用)
        if self.config.use_decoder_lora:
            lora_params = [p for p in self.decoder.parameters() if p.requires_grad]
            if lora_params:
                param_groups.append({
                    "params": lora_params,
                    "lr": 1e-5,
                    "name": "decoder_lora",
                })
        
        return param_groups
    
    def save_checkpoint(self, path: str):
        """保存 checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            "config": {
                "decoder_path": self.config.decoder_path,
                "hidden_size": self.config.hidden_size,
                "use_token_mixer": self.config.use_token_mixer,
                "token_mixer_num_layers": self.config.token_mixer_num_layers,
                "use_local_c3": self.config.use_local_c3,
                "num_post_encoder_layers": self.config.num_post_encoder_layers,
                "use_rope_alignment": self.config.use_rope_alignment,
                "use_relative_pos_embed": self.config.use_relative_pos_embed,
                "compression_mode": getattr(self.config, 'compression_mode', 'cross_attn'),
            },
        }
        
        # TokenMixer
        if self.token_mixer is not None:
            checkpoint["token_mixer"] = self.token_mixer.state_dict()
        
        # LocalC3
        if self.local_c3 is not None:
            checkpoint["local_c3"] = self.local_c3.state_dict()
        
        # Decoder LoRA
        if self.config.use_decoder_lora:
            lora_state = {}
            for name, param in self.decoder.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    lora_state[name] = param.data.cpu()
            checkpoint["decoder_lora"] = lora_state
        
        torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))
        print(f"[V7] Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载 checkpoint (handles shape mismatches for cross-config resume, e.g. 9x→16x)"""
        checkpoint_file = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(checkpoint_file):
            print(f"[V7] Checkpoint not found: {checkpoint_file}")
            return

        checkpoint = torch.load(checkpoint_file, map_location="cpu")

        # TokenMixer
        if self.token_mixer is not None and "token_mixer" in checkpoint:
            self.token_mixer.load_state_dict(checkpoint["token_mixer"])
            print(f"[V7] TokenMixer loaded from checkpoint")

        # LocalC3 (handle shape mismatch when block_size differs)
        if self.local_c3 is not None and "local_c3" in checkpoint:
            try:
                self.local_c3.load_state_dict(checkpoint["local_c3"])
                print(f"[V7] LocalC3 loaded from checkpoint")
            except RuntimeError as e:
                # Shape mismatch (e.g. rel_pos_embed [9,D] vs [16,D]) — skip and keep random init
                print(f"[V7] LocalC3 shape mismatch, using fresh initialization: {e}")

        # Decoder LoRA
        if "decoder_lora" in checkpoint:
            for name, param in self.decoder.named_parameters():
                if name in checkpoint["decoder_lora"]:
                    param.data.copy_(checkpoint["decoder_lora"][name].to(param.device))

        print(f"[V7] Checkpoint loaded from {path}")
    
    def freeze_token_mixer(self):
        """冻结 TokenMixer（用于 Stage-1a: 只训练 LocalC3）"""
        if self.token_mixer is not None:
            for param in self.token_mixer.parameters():
                param.requires_grad = False
            print("[V7] TokenMixer frozen")
    
    def unfreeze_token_mixer(self):
        """解冻 TokenMixer（用于 Stage-1b: 训练 TokenMixer + LocalC3）"""
        if self.token_mixer is not None:
            for param in self.token_mixer.parameters():
                param.requires_grad = True
            print("[V7] TokenMixer unfrozen")


# =============================================================================
# 工厂函数
# =============================================================================

def create_model_v7(
    decoder_path: str = "./Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda",
    **kwargs,
) -> ImageC3ModelV7:
    """创建 V7 模型"""
    config = ImageC3ConfigV7(
        decoder_path=decoder_path,
        **kwargs,
    )
    
    model = ImageC3ModelV7(config, device)
    model.setup()
    
    return model


if __name__ == "__main__":
    # 测试
    print("Testing Image-C3 V7...")
    
    config = ImageC3ConfigV7(
        decoder_path="./Qwen/Qwen2.5-VL-7B-Instruct",
        use_token_mixer=True,
        token_mixer_num_layers=2,
        use_local_c3=True,
        use_rope_alignment=True,
    )
    
    model = ImageC3ModelV7(config, device="cuda")
    model.setup()
    
    print("\n✓ V7 model created successfully!")
