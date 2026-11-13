"""Calpha: protein folding through a general-purpose chat model.

A chat model (Qwen3.5-9B, fine-tuned) reads a protein sequence together with frozen ESM2-3B
features; its hidden states at the residue positions condition a frozen SimpleFold-3B
flow-matching head that generates all-atom coordinates.
"""

__version__ = "0.1.0"

from .constants import CANONICAL_TEMPLATE, COND_LAYERS, STRUCT_TOKEN  # noqa: F401


def __getattr__(name):  # heavy imports only on demand
    if name == "Calpha":
        from .pipeline import Calpha
        return Calpha
    raise AttributeError(name)
