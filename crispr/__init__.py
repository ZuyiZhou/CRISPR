"""
CRISPR: Context-Refined Information Spatial Pooling with Region-awareness
for Efficient Visual Token Compression in VLMs.

This is the core model implementation. Training scripts, evaluation
scripts, and checkpoints will follow in subsequent updates (see the
repository README roadmap).
"""

from .model_v7 import (
    ImageC3ConfigV7,
    TokenMixer,
    LocalC3,
    ImageC3ModelV7,
    create_model_v7,
)

__all__ = [
    "ImageC3ConfigV7",
    "TokenMixer",
    "LocalC3",
    "ImageC3ModelV7",
    "create_model_v7",
]
