[Home](../README.md) / [Documentation](../docs/README.md) / Qwen3.5

# CRISPR for Qwen3.5

CRISPR compressor checkpoints for the frozen Qwen3.5-4B and Qwen3.5-9B
backbones. Each checkpoint produces a fixed number of visual tokens per image.

| Backbone | Visual tokens per image | Checkpoint |
|---|---:|---|
| Qwen3.5-4B | 16 | [`qwen3.5-4b-16tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-4b-16tokens) |
| Qwen3.5-4B | 36 | [`qwen3.5-4b-36tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-4b-36tokens) |
| Qwen3.5-4B | 64 | [`qwen3.5-4b-64tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-4b-64tokens) |
| Qwen3.5-9B | 36 | [`qwen3.5-9b-36tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-9b-36tokens) |
| Qwen3.5-9B | 64 | [`qwen3.5-9b-64tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-9b-64tokens) |

These are additional Qwen3.5 checkpoints, separate from the Qwen2.5-VL
checkpoints accompanying the ACM MM 2026 paper. The [Qwen2.5-VL reproduction guide](../docs/reproduction.md) applies to
the original release.

## Download and load

From the root of this GitHub repository:

```bash
pip install -r qwen35/requirements.txt
```

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from qwen35 import load_compressor

variant = "qwen3.5-4b-64tokens"
root = snapshot_download(
    repo_id="Rim2000/CRISPR",
    allow_patterns=[f"{variant}/*"],
)
compressor = load_compressor(Path(root) / variant, device="cpu")
```

Choose the directory matching your backbone and token budget. All weights are
float32; the loader checks tensor names and shapes strictly and sets eval mode.
The frozen vision encoder and language model are obtained separately from
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) or
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).

## Compressor interface

```python
import torch

# visual_features: the matching backbone's merged image features, [H * W, D].
# H and W describe the merged spatial grid, after the vision tower's 2x2 merge.
# D is 2560 for Qwen3.5-4B and 4096 for Qwen3.5-9B.
with torch.inference_mode():
    memory, centres = compressor(visual_features.float(), H, W)
# memory: [M, D]; centres: M (row, column) coordinates in the original merged grid.
```

Keep the compressor and its input features in float32 on the same device.
Use the backbone's image processor with `max_pixels=1003520` and its default
minimum image size. The compressor preserves the original merged-grid
coordinates for the reader's multimodal rotary positions. Keep these
coordinates when inserting the compressed features; a new dense M-token image
grid does not represent the same positions. Use vision start/end sentinels and
the empty thinking block produced by `enable_thinking=False`.

This package loads the compressor and processes merged visual features. It does
not wrap the full Qwen3.5 text-generation API. In particular, the legacy
`create_model_v7` loader is for Qwen2.5-VL and cannot load these checkpoints.

## File format

Each checkpoint directory contains:

- `model.safetensors`: 56 tensors for the CRISPR modules, including the output scale.
- `config.json`: backbone, architecture, token budget, and inference settings.
- `SHA256SUMS`: checksums for both files.

The safetensors weights preserve the source model tensors exactly. No backbone
weights, optimizer state, or training metadata are included. The 4B compressor
contains 393,408,001 parameters (about 1.57 GB); the 9B compressor contains
1,006,940,161 parameters (about 4.03 GB). These sizes refer to the compressor.

[checkpoint_manifest.json](checkpoint_manifest.json) records file sizes and
SHA-256 digests. Verify a downloaded directory with `sha256sum -c SHA256SUMS`.

The code and compressor weights use the repository's MIT license. The Qwen
backbones retain their own licenses. See the main README for the CRISPR citation.

---

[All checkpoints](../docs/checkpoints.md) · [Documentation](../docs/README.md) · [Back to home](../README.md)
