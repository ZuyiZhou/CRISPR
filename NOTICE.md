# Third-Party Notices

This repository's original code is released under the [MIT License](LICENSE).
The following file is a ported/adapted derivative of third-party code and is
called out separately:

## `crispr/baselines.py`

Ported and adapted from **EffiVLM-Bench** (`kv_cache_compression/qwen2vl_model.py`,
`siglip_model.py`), adapted here from Qwen2-VL to Qwen2.5-VL-7B-Instruct for use
as training-free comparison baselines (VisionZip, PruMerge+, FastV-style
attention capture) in our paper's experiments.

**License status: unresolved.** We have not been able to confirm the license
terms of EffiVLM-Bench at the time of this release. This file is included with
attribution for research transparency and reproducibility of our comparison
baselines, but its licensing may differ from the MIT license covering the rest
of this repository. If you are the EffiVLM-Bench maintainers, or can point us
to the applicable license, please open an issue — we will update this notice
(and the file's license header, or remove/replace the file if required)
accordingly.
