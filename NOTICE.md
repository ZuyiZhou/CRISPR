# Third-Party Notices

This repository's original code is released under the [MIT License](LICENSE).

## Removed: baseline comparison code (`crispr/baselines.py`, `scripts/eval_baselines.py`)

Our paper's comparisons against VisionZip, PruMerge+, and a FastV-style
attention-capture baseline were run using code ported and adapted from
[EffiVLM-Bench](https://github.com/EffiVLM-Bench/EffiVLM-Bench)
(`kv_cache_compression/qwen2vl_model.py`, `siglip_model.py`), adapted from
Qwen2-VL to Qwen2.5-VL-7B/3B-Instruct.

**These two files have been removed from this repository.** EffiVLM-Bench's
GitHub repository does not declare a license, which by default means all
rights are reserved by its authors — we are not able to confirm we have
permission to redistribute a derivative of their code, so we removed it
rather than keep it published without that permission.

**To reproduce the baseline numbers in our paper:** obtain the relevant code
directly from [EffiVLM-Bench](https://github.com/EffiVLM-Bench/EffiVLM-Bench)
under their own terms, and adapt it to Qwen2.5-VL following the same approach
we used (see the paper for method-level details of the VisionZip/PruMerge+/
FastV configurations we evaluated). If you are an EffiVLM-Bench maintainer and
can clarify the license, or grant permission to redistribute an adapted
version, please open an issue on this repository — we would be glad to restore
this code with proper licensing.
