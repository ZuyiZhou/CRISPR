# CRISPR: Context-Refined Information Spatial Pooling with Region-awareness for Efficient Visual Token Compression in VLMs

[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-blue)](https://doi.org/10.1145/3767308.3835007)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for our ACM MM 2026 paper **CRISPR**, accepted for publication.

> **Code, checkpoints, and full reproduction instructions are being finalized for
> camera-ready and will be published here shortly. This page is the permanent,
> citable home for the project — check back soon, or watch/star the repo to be
> notified.**

## Abstract

CRISPR is a visual token compression method for Vision-Language Models (VLMs).
It achieves 9x/16x token compression via 3x3 block cross-attention, distilled
from a frozen Qwen2.5-VL-7B/3B-Instruct teacher, with trainable parameters
amounting to roughly 0.1% of the 7B backbone.

<!-- TODO(camera-ready): paste final paper abstract here -->

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
- [ ] De-anonymized code release
- [ ] Model checkpoints (Qwen2.5-VL-7B and 3B backbones, 9x and 16x compression)
- [ ] Reproduction-friendly documentation (environment, data prep, training, evaluation)

## Naming Note

The paper uses **CRISPR** as the method name. Some historical code/comments may
still refer to the earlier internal codename **CRISP** / **ImageC3** — these
refer to the same method.

## License

Code in this repository is released under the [MIT License](LICENSE), unless
otherwise noted in individual files or subdirectories (e.g. third-party assets
with their own upstream licenses).

## Acknowledgements

This work builds on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL). We
thank the authors of Qwen2.5-VL and the datasets used in this work.
