"""CRISPR inference modules, shared with the published Qwen2.5-VL architecture."""
from __future__ import annotations
from typing import Optional, Tuple
from types import SimpleNamespace
import torch
from torch import nn

class TokenMixerLayer(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int=16, ffn_expansion: int=4, dropout: float=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-06)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-06)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size * ffn_expansion), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size * ffn_expansion, hidden_size), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(x)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x

class TokenMixer(nn.Module):

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TokenMixerLayer(hidden_size=config.hidden_size, num_heads=config.token_mixer_num_heads, ffn_expansion=config.token_mixer_ffn_expansion, dropout=config.token_mixer_dropout) for _ in range(config.token_mixer_num_layers)])
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=1e-06)

    def forward(self, image_tokens: torch.Tensor, image_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = image_tokens
        for layer in self.layers:
            x = layer(x, image_mask)
        x = self.final_norm(x)
        return x

class PostEncoderLayer(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int=8, dropout: float=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout))
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-06)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(x)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x

class LocalC3(nn.Module):

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.block_size = config.local_c3_block_size
        self.compression_mode = getattr(config, 'compression_mode', 'cross_attn')
        hidden_size = config.hidden_size
        dropout = config.local_c3_dropout
        num_positions = self.block_size ** 2
        if config.use_relative_pos_embed:
            self.rel_pos_embed = nn.Parameter(torch.randn(num_positions, hidden_size) * 0.02)
        else:
            self.rel_pos_embed = None
        if self.compression_mode == 'cross_attn':
            self.query = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=config.local_c3_num_heads, dropout=dropout, batch_first=True)
            self.cross_attn_norm = nn.RMSNorm(hidden_size, eps=1e-06)
        else:
            self.query = None
            self.cross_attn = None
            self.cross_attn_norm = None
        hidden_dim = hidden_size * 4
        self.ffn = nn.Sequential(nn.RMSNorm(hidden_size, eps=1e-06), nn.Linear(hidden_size, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_size), nn.Dropout(dropout))
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=1e-06)
        self.post_encoder_layers = nn.ModuleList([PostEncoderLayer(hidden_size, config.local_c3_num_heads, dropout) for _ in range(config.num_post_encoder_layers)])
        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-06)

    def forward(self, image_tokens: torch.Tensor, image_grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = image_tokens.shape[0]
        device = image_tokens.device
        dtype = image_tokens.dtype
        N = self.block_size
        compressed_list = []
        lengths = []
        for i in range(B):
            t, h, w = image_grid_thw[i].tolist()
            t, h, w = (int(t), int(h), int(w))
            h_in = h // 2
            w_in = w // 2
            n_tokens = h_in * w_in
            x = image_tokens[i, :n_tokens, :]
            h_pad = (h_in + N - 1) // N * N
            w_pad = (w_in + N - 1) // N * N
            n_padded = h_pad * w_pad
            if n_padded > n_tokens:
                pad = torch.zeros(n_padded - n_tokens, x.shape[-1], device=device, dtype=dtype)
                x = torch.cat([x, pad], dim=0)
            x = x.view(h_pad, w_pad, -1)
            h_out, w_out = (h_pad // N, w_pad // N)
            x = x.view(h_out, N, w_out, N, -1)
            x = x.permute(0, 2, 1, 3, 4)
            x = x.reshape(h_out * w_out, N * N, -1)
            num_groups = x.shape[0]
            if self.rel_pos_embed is not None:
                x = x + self.rel_pos_embed.unsqueeze(0).to(dtype)
            if self.compression_mode == 'avg_pool':
                pooled = x.mean(dim=1)
            else:
                q = self.query.expand(num_groups, 1, -1).to(dtype)
                attn_out, _ = self.cross_attn(q, x, x)
                pooled = self.cross_attn_norm(attn_out.squeeze(1))
            compressed_i = self.ffn_norm(self.ffn(pooled) + pooled)
            compressed_list.append(compressed_i)
            lengths.append(num_groups)
        max_len = max(lengths)
        padded = []
        masks = []
        for comp, length in zip(compressed_list, lengths):
            if comp.shape[0] < max_len:
                pad = torch.zeros(max_len - comp.shape[0], comp.shape[-1], device=device, dtype=dtype)
                comp = torch.cat([comp, pad], dim=0)
            padded.append(comp)
            mask = torch.zeros(max_len, device=device, dtype=torch.long)
            mask[:length] = 1
            masks.append(mask)
        latent_tokens = torch.stack(padded, dim=0)
        latent_mask = torch.stack(masks, dim=0)
        for layer in self.post_encoder_layers:
            latent_tokens = layer(latent_tokens, latent_mask)
        latent_tokens = self.final_norm(latent_tokens)
        return (latent_tokens, latent_mask)
