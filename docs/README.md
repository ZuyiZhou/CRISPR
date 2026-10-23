[Home](../README.md) / Documentation

# Documentation

Find the right starting point for your model and task.

| I want to… | Start here |
|---|---|
| Understand the method | [Method overview](../README.md#method) |
| Download weights | [Checkpoint directory](checkpoints.md) |
| Use the Qwen3.5 compressor | [Qwen3.5 loading guide](../qwen35/README.md) |
| Reproduce the original Qwen2.5-VL release | [Reproduction guide](reproduction.md) |
| Find the model and evaluation code | [Repository map](../README.md#repository-map) |
| Report an issue or contribute | [Contributing](../CONTRIBUTING.md) |
| Cite CRISPR | [Citation](../README.md#citation) |

## Which guide should I use?

**Qwen2.5-VL** is the original ACM MM 2026 release. Its guide covers the
published model, training commands, and evaluation scripts already included
in this repository.

**Qwen3.5** provides compressor weights and an interface for merged visual
features. Follow its dedicated loading guide for the checkpoint format,
feature dimensions, token budgets, and position coordinates.

## Common questions

<details>
<summary><strong>Does a checkpoint include the Qwen backbone?</strong></summary>

No. The checkpoint contains CRISPR modules. Download the matching Qwen backbone
separately using the links in the appropriate guide.

</details>

<details>
<summary><strong>Can I use the same loader for both releases?</strong></summary>

Use `crispr.create_model_v7` for Qwen2.5-VL and `qwen35.load_compressor`
for Qwen3.5. Their checkpoint formats and integration interfaces differ.

</details>

<details>
<summary><strong>Where are the baseline comparison scripts?</strong></summary>

The included evaluation scripts cover CRISPR, the original-image model,
and low-resolution baselines. See [NOTICE.md](../NOTICE.md) for the separate
VisionZip / PruMerge+ / FastV comparison code and its redistribution status.

</details>

<details>
<summary><strong>Why do some files say CRISP or ImageC3?</strong></summary>

Those are historical code names for CRISPR. Existing names are retained for
compatibility with the published scripts and checkpoints.

</details>
