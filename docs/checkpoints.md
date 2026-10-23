[Home](../README.md) / [Documentation](README.md) / Checkpoints

# Checkpoint directory

Choose the checkpoint for your backbone, then follow its loading guide.

## Qwen2.5-VL checkpoints

CRISPR checkpoints are hosted on Hugging Face: **[Rim2000/CRISPR](https://huggingface.co/Rim2000/CRISPR)**.

| Backbone | Compression | File |
|---|:---:|---|
| Qwen2.5-VL-3B-Instruct | 9x  | [`3b_9x/checkpoint.pt`](https://huggingface.co/Rim2000/CRISPR/tree/main/3b_9x) |
| Qwen2.5-VL-3B-Instruct | 16x | [`3b_16x/checkpoint.pt`](https://huggingface.co/Rim2000/CRISPR/tree/main/3b_16x) |
| Qwen2.5-VL-7B-Instruct | 16x | [`7b_16x/checkpoint.pt`](https://huggingface.co/Rim2000/CRISPR/tree/main/7b_16x) |

## Qwen3.5 checkpoints

Additional CRISPR checkpoints are available for Qwen3.5-4B and Qwen3.5-9B,
with a fixed visual-token budget per image.

| Backbone | Visual tokens per image | Checkpoint |
|---|---:|---|
| Qwen3.5-4B | 16 | [`qwen3.5-4b-16tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-4b-16tokens) |
| Qwen3.5-4B | 36 | [`qwen3.5-4b-36tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-4b-36tokens) |
| Qwen3.5-4B | 64 | [`qwen3.5-4b-64tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-4b-64tokens) |
| Qwen3.5-9B | 36 | [`qwen3.5-9b-36tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-9b-36tokens) |
| Qwen3.5-9B | 64 | [`qwen3.5-9b-64tokens`](https://huggingface.co/Rim2000/CRISPR/tree/main/qwen3.5-9b-64tokens) |

Each directory contains float32 safetensors weights, an inference configuration,
and SHA-256 checksums. See [the Qwen3.5 guide](../qwen35/README.md) for downloading,
loading, and the compressor interface. These files contain the CRISPR modules;
the Qwen3.5 backbones are downloaded separately.

## Choose the matching loader

| Release | Entry point | Guide |
|---|---|---|
| Qwen2.5-VL | `crispr.create_model_v7` | [Reproduction](reproduction.md) |
| Qwen3.5 | `qwen35.load_compressor` | [Download and load](../qwen35/README.md#download-and-load) |

The Qwen2.5-VL loader takes a **directory** containing `checkpoint.pt`.
The Qwen3.5 loader takes a directory containing `config.json` and
`model.safetensors`. Match both the backbone and the compression setting.

The Qwen2.5-VL-7B 9× checkpoint is not available in this release.

---

[Documentation](README.md) · [Back to home](../README.md)
