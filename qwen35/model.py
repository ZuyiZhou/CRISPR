"""Fixed-budget CRISPR compressor for Qwen3.5 merged image features."""
from __future__ import annotations
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from ._modules import TokenMixer, LocalC3

def fixed_grid_axis(L: int, N: int, G: int) -> bool:
    return not N * (G - 1) < L <= N * G

def fixed_grid_resample(visual_tokens: torch.Tensor, h_in: int, w_in: int, N: int, G: int):
    D = visual_tokens.shape[-1]
    rs, cs = (fixed_grid_axis(h_in, N, G), fixed_grid_axis(w_in, N, G))
    if not (rs or cs):
        return (visual_tokens, h_in, w_in, None)
    g = visual_tokens.view(h_in, w_in, D)
    h2, w2 = (N * G if rs else h_in, N * G if cs else w_in)
    for axis, L, S, on in ((0, h_in, h2, rs), (1, w_in, w2, cs)):
        if not on:
            continue
        x = g.movedim(axis, -1)
        shp = x.shape
        y = F.interpolate(x.reshape(1, -1, 1, L).float(), size=(1, S), mode='bilinear', align_corners=False, antialias=S < L)
        g = y.reshape(*shp[:-1], S).movedim(-1, axis).to(visual_tokens.dtype)

    def ctr(L, on):
        return [min(int(math.floor((j + 0.5) * L / G)), L - 1) for j in range(G)] if on else [min(N * j + N // 2, L - 1) for j in range(G)]
    rows, cols = (ctr(h_in, rs), ctr(w_in, cs))
    return (g.reshape(h2 * w2, D), h2, w2, [(r, c) for r in rows for c in cols])


class CRISPRQwen35Compressor(nn.Module):
    """Map one image's [H*W, D] merged features to M features and grid centres."""

    def __init__(self, config):
        super().__init__()
        hidden = int(config["hidden_size"])
        block = int(config["block_size"])
        count = int(config["num_visual_tokens"])
        expected = {"Qwen/Qwen3.5-4B": 2560, "Qwen/Qwen3.5-9B": 4096}
        if config.get("format_version") != 1 or config.get("architecture") != "CRISPRQwen35Compressor":
            raise ValueError("Unsupported CRISPR checkpoint format")
        if expected.get(config["base_model"]) != hidden:
            raise ValueError("Backbone and feature width do not match")
        grid = math.isqrt(count)
        if count <= 0 or grid * grid != count or block <= 0:
            raise ValueError("The token count must be a positive square and block size positive")
        self.release_config = dict(config)
        self.hidden_size = hidden
        self.G = self.fixed_grid = grid
        self.block = self.block_size = block
        self.fixed_M = count
        self.config = SimpleNamespace(
            hidden_size=hidden,
            local_c3_block_size=block,
            token_mixer_num_layers=config["token_mixer_num_layers"],
            token_mixer_num_heads=config["token_mixer_num_heads"],
            token_mixer_ffn_expansion=config["token_mixer_ffn_expansion"],
            token_mixer_dropout=0.0,
            local_c3_num_heads=config["local_c3_num_heads"],
            local_c3_dropout=0.0,
            num_post_encoder_layers=config["num_post_encoder_layers"],
            use_relative_pos_embed=config["use_relative_pos_embed"],
            compression_mode=config["compression_mode"],
        )
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.token_mixer = TokenMixer(self.config)
        self.local_c3 = LocalC3(self.config)

    def forward(self, tokens, h_in, w_in):
        """Return (features [M, D], [(row, col), ...]) in the original merged grid.

        Use float32 features and eval mode. Coordinates are needed by the
        Qwen3.5 reader's multimodal rotary positions.
        """
        if self.training:
            raise RuntimeError("This release supports inference; call eval() first")
        if h_in <= 0 or w_in <= 0 or tokens.shape != (h_in * w_in, self.hidden_size):
            raise ValueError("Features must have shape [h_in * w_in, hidden_size]")
        if tokens.dtype != torch.float32 or self.out_scale.dtype != torch.float32:
            raise ValueError("Keep compressor weights and input features in float32")
        n, g = self.block, self.fixed_grid
        x = self.token_mixer(tokens[None])[0]
        x, h2, w2, centres = fixed_grid_resample(x, h_in, w_in, n, g)
        thw = torch.tensor([[1, 2 * h2, 2 * w2]], device=x.device)
        out, mask = self.local_c3(x[None], thw)
        if int(mask[0].sum()) != self.fixed_M:
            raise RuntimeError("Unexpected number of compressed visual tokens")
        if centres is None:
            centres = [(min(n * r + n // 2, h_in - 1), min(n * c + n // 2, w_in - 1))
                       for r in range(g) for c in range(g)]
        return out[0, :self.fixed_M] * self.out_scale, centres


def load_compressor(checkpoint_dir, *, device="cpu"):
    """Load a directory containing config.json and model.safetensors."""
    directory = Path(checkpoint_dir)
    config = json.loads((directory / "config.json").read_text())
    with torch.device("meta"):
        model = CRISPRQwen35Compressor(config)
    state = load_file(str(directory / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
