# CRISPR: Context-Refined Information Spatial Pooling with Region-awareness for Efficient Visual Token Compression in VLMs

[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-blue)](https://doi.org/10.1145/3767308.3835007)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for our ACM MM 2026 paper **CRISPR**, accepted for publication.

> **The core model, training script, dataset loaders, and evaluation scripts
> (including baselines) are now available (see [Code](#code) below), along
> with checkpoints for both backbones (see [Checkpoints](#checkpoints)).
> Step-by-step reproduction documentation is being finalized for
> camera-ready and will follow. This page is the permanent, citable home
> for the project — watch/star the repo to be
> notified of updates.**

## Abstract

CRISPR is a visual token compression method for Vision-Language Models (VLMs).
It achieves 9x/16x token compression via 3x3 block cross-attention, distilled
from a frozen Qwen2.5-VL-7B/3B-Instruct teacher, with trainable parameters
amounting to roughly 0.1% of the 7B backbone.

<!-- TODO(camera-ready): paste final paper abstract here -->

<p align="center">
  <img src="assets/overview.png" width="55%" alt="Comparison of compression methods">
</p>

<p align="center"><sub>Comparison of compression methods. Token pruning and resolution downsampling lose fine-grained details, whereas CRISPR preserves them.</sub></p>

<p align="center">
  <img src="assets/architecture.png" width="85%" alt="CRISPR architecture">
</p>

<p align="center"><sub>CRISPR architecture: Token Refiner → Local Token Compressor (block cross-attention) → Global Token Fusion, trained via CE + KL + hidden-state distillation from a frozen Qwen2.5-VL teacher.</sub></p>

## Code

```bash
pip install -r requirements.txt
```

```python
from crispr import create_model_v7

model = create_model_v7(decoder_path="./Qwen/Qwen2.5-VL-7B-Instruct")
```

- `crispr/model_v7.py` — the CRISPR architecture: `TokenMixer` (Token Refiner),
  `LocalC3` (Local Token Compressor + Global Token Fusion), and the top-level
  `ImageC3ModelV7` model that wires them around a frozen Qwen2.5-VL decoder.
- `crispr/dataset.py`, `crispr/dataset_v3.py` — dataset loaders for training.
- `crispr/baselines.py` — training-free compression baselines (VisionZip,
  PruMerge+, FastV-style attention capture) used for comparison in the paper.
  **License note:** this file is ported/adapted from a third-party project
  (EffiVLM-Bench) whose license we have not been able to confirm — see
  [NOTICE.md](NOTICE.md) before reusing it outside of research comparison.
- `train_crispr_v7.py` — the training script (see docstring for stage-by-stage
  usage: Stage-1a/1b/2/3).
- `scripts/eval_crisp.py`, `scripts/eval_9x_quick.py` — evaluation scripts for
  CRISPR / teacher / low-resolution baselines.
- `scripts/eval_baselines.py` — evaluation script for the VisionZip/PruMerge+/FastV
  baselines.

Note: these scripts depend on a local copy of the Qwen2.5-VL backbone weights
(not included in this repository — see [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
for download instructions) and on training/eval data prepared in the JSONL
format described in the scripts' docstrings. Full step-by-step reproduction
documentation and launch scripts will be added in subsequent updates.

## Checkpoints

CRISPR checkpoints are hosted on Hugging Face: **[Rim2000/CRISPR](https://huggingface.co/Rim2000/CRISPR)**.

| Backbone | Compression | |
|---|---|---|
| Qwen2.5-VL-3B-Instruct | 9x | `3b_9x/checkpoint.pt` |
| Qwen2.5-VL-3B-Instruct | 16x | `3b_16x/checkpoint.pt` |
| Qwen2.5-VL-7B-Instruct | 16x | `7b_16x/checkpoint.pt` |

The Qwen2.5-VL-7B-Instruct / 9x checkpoint is not currently available (lost
prior to this release) and is not planned unless we retrain it.

## Citation

If you find this work useful, please cite:

**ACM Reference Format:**

Zuyi Zhou, Dizhan Xue, Shengsheng Qian, and Changsheng Xu. 2026. CRISPR:
Context-Refined Information Spatial Pooling with Region-awareness for
Efficient Visual Token Compression in VLMs. In Proceedings of the 34th ACM
International Conference on Multimedia (MM '26), November 10-14, 2026, Rio
de Janeiro, Brazil. ACM, New York, NY, USA. https://doi.org/10.1145/3767308.3835007

**BibTeX:**

```bibtex
@inproceedings{zhou2026crispr,
  author    = {Zhou, Zuyi and Xue, Dizhan and Qian, Shengsheng and Xu, Changsheng},
  title     = {CRISPR: Context-Refined Information Spatial Pooling with
               Region-awareness for Efficient Visual Token Compression in VLMs},
  booktitle = {Proceedings of the 34th ACM International Conference on
               Multimedia (MM '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3767308.3835007}
}
```

## Authors & Affiliations

- **Zuyi Zhou** — <zuyi.zhou@evermind.ai> — Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences; EverMind AI
- **Dizhan Xue** — <dizhan.xue@evermind.ai> — Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences; EverMind AI
- **Shengsheng Qian** (Corresponding author) — <shengsheng.qian@nlpr.ia.ac.cn> — Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences
- **Changsheng Xu** — <csxu@nlpr.ia.ac.cn> — Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences; Peng Cheng Laboratory

## Roadmap

- [x] Paper accepted, DOI reserved
- [x] Core model code (`crispr/model_v7.py`)
- [x] Training script (`train_crispr_v7.py`) and dataset loaders
- [x] Evaluation scripts, including baselines comparison
- [x] Model checkpoints, Qwen2.5-VL-3B backbone, 9x and 16x compression (see [Checkpoints](#checkpoints))
- [x] Model checkpoints, Qwen2.5-VL-7B backbone, 16x compression (7B/9x lost, not planned)
- [ ] Reproduction-friendly documentation (environment, data prep, exact launch commands)
- [ ] Resolve license status of the ported `crispr/baselines.py` (see [NOTICE.md](NOTICE.md))

## Naming Note

The paper uses **CRISPR** as the method name. Some historical code/comments may
still refer to the earlier internal codename **CRISP** / **ImageC3** — these
refer to the same method.

## License

Code in this repository is released under the [MIT License](LICENSE), unless
otherwise noted in individual files or subdirectories (e.g. third-party assets
with their own upstream licenses). See [NOTICE.md](NOTICE.md) for a
third-party file whose license status is currently unresolved.

## Acknowledgements

This work builds on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL). We
thank the authors of Qwen2.5-VL and the datasets used in this work.
