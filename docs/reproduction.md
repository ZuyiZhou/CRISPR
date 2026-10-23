[Home](../README.md) / [Documentation](README.md) / Qwen2.5-VL reproduction

# Reproduce the Qwen2.5-VL release

This section covers the original Qwen2.5-VL release. For the Qwen3.5 checkpoints, see [the Qwen3.5 guide](../qwen35/README.md).

Run the commands below from the repository root.

## Environment

- Python 3.12, PyTorch 2.10+ (cu128), `transformers>=4.56.1` (see [requirements.txt](../requirements.txt))
- Download the backbone weights and point `--decoder_path` at them:
  - `./Qwen/Qwen2.5-VL-7B-Instruct` or `./Qwen/Qwen2.5-VL-3B-Instruct`, from
    [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) (e.g. via
    `huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./Qwen/Qwen2.5-VL-7B-Instruct`)
  - If `huggingface.co` is unreachable, set `HF_ENDPOINT=https://hf-mirror.com`
    before downloading/running (both `train_crispr_v7.py` and the eval scripts
    already set this as a default if unset).
- `attn_implementation="eager"` is required — SDPA is not compatible with the
  custom `inputs_embeds`/3D position_ids used here.

## Data preparation

Training data is a JSONL file, one example per line:

```json
{"id": "unique_id", "image": "path/to/image.jpg", "conversations": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}], "source": "dataset_name"}
```

The `source` field is used to auto-select a task-specific prompt template
(e.g. OCR-style datasets get a specialized prompt) — see `MultiTaskDataset`
in `train_crispr_v7.py`. Evaluation datasets are expected under `--data_root`
(default `./data/eval`) in the layout each `load_*` function in
`scripts/eval_crisp.py` / `scripts/eval_9x_quick.py` expects (grep for
`def load_<dataset>` to see the exact file/column names for a given dataset).

## Training

`local_c3_block_size` sets the compression ratio: `3` → 9x (3×3 blocks), `4` → 16x (4×4 blocks).

```bash
# Stage-1a: warm up LocalC3 only, TokenMixer frozen
python3.12 train_crispr_v7.py \
    --data_path ./data/train/stage1_caption/combined.jsonl \
    --output_dir ./outputs/crispr_9x_stage1a \
    --decoder_path ./Qwen/Qwen2.5-VL-7B-Instruct \
    --local_c3_block_size 3 \
    --epochs 1 --freeze_token_mixer --use_hidden_distillation

# Stage-1b: train TokenMixer + LocalC3 jointly
python3.12 train_crispr_v7.py \
    --data_path ./data/train/stage2_mixed/combined_v4.jsonl \
    --output_dir ./outputs/crispr_9x_stage1b \
    --decoder_path ./Qwen/Qwen2.5-VL-7B-Instruct \
    --local_c3_block_size 3 \
    --epochs 3 \
    --resume_from ./outputs/crispr_9x_stage1a/checkpoint-best \
    --use_hidden_distillation

# Stage-2: multi-GPU, + hidden-state distillation schedule, + 3D RoPE alignment
torchrun --nproc_per_node=4 --master_port=29501 train_crispr_v7.py \
    --data_path ./data/train/stage2_mixed/combined_v4.jsonl \
    --output_dir ./outputs/crispr_9x_stage2 \
    --decoder_path ./Qwen/Qwen2.5-VL-7B-Instruct \
    --local_c3_block_size 3 \
    --epochs 3 --batch_size 1 --gradient_accumulation_steps 16 \
    --lr_local_c3 1e-5 --lr_token_mixer 1e-5 \
    --resume_from ./outputs/crispr_9x_stage1b/checkpoint-best \
    --use_hidden_distillation --hidden_loss_weight 0.1 --hidden_loss_weight_final 0.3

# Resume with full trainer state (weights + optimizer + epoch) after a crash
torchrun --nproc_per_node=4 train_crispr_v7.py \
    --resume_training_from ./outputs/crispr_9x_stage2/checkpoint-step_XXXX \
    ...  # same args as the run being resumed
```

For 16x, pass `--local_c3_block_size 4` and `--decoder_path` for the 3B
backbone if reproducing that configuration; per-module learning rates default
to `--lr_local_c3 1e-4`, `--lr_token_mixer 5e-5`, `--lr_decoder_lora 1e-5`
(AdamW, linear warmup + decay, grad clip 1.0). Loss is
`CE + 0.5·KL(T=1.0) + hidden_weight·Hidden(MSE on last decoder layers)`,
with `hidden_weight` scheduled linearly from `--hidden_loss_weight` to
`--hidden_loss_weight_final` across training.

## Evaluation

```bash
# CRISPR, 7B backbone, 9x — full benchmark suite
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.12 scripts/eval_crisp.py \
    --methods image_c3 \
    --decoder_path ./Qwen/Qwen2.5-VL-7B-Instruct \
    --image_c3_ckpt ./outputs/crispr_9x_stage2/checkpoint-best \
    --block_size 3 \
    --datasets mmmu mmbench chartqa docvqa textvqa realworldqa mathvista ocrbench mme scienceqa \
    --n 0 --output_dir eval_results/crisp/9x

# CRISPR, 3B backbone, 9x
python3.12 scripts/eval_crisp.py \
    --methods image_c3 \
    --decoder_path ./Qwen/Qwen2.5-VL-3B-Instruct \
    --image_c3_ckpt ./outputs/crispr_3b_9x/checkpoint-best \
    --block_size 3 \
    --datasets mmmu mmbench chartqa docvqa textvqa mme scienceqa \
    --n 0 --output_dir eval_results/crisp/3b_9x

# Teacher (uncompressed upper bound) and LowRes baseline use the same script
python3.12 scripts/eval_crisp.py --methods teacher lowres \
    --datasets mmmu mmbench chartqa --n 0 --output_dir eval_results/reference
```

`--n 0` evaluates the full dataset; use a smaller `--n` for a quick sanity
check. Results land at `eval_results/{method}/{ratio}/{dataset}/results.json`.
See each script's module docstring for the complete dataset list and more
usage examples (`scripts/eval_9x_quick.py` is an earlier, 9x/7B-only variant
of the same evaluation logic kept for reference).

Note: to reproduce the VisionZip/PruMerge+/FastV baseline numbers reported in
the paper, see [Third-party notices](../NOTICE.md) regarding EffiVLM-Bench.

---

[Checkpoint downloads](checkpoints.md) · [Qwen3.5 guide](../qwen35/README.md) · [Back to home](../README.md)
