"""Frozen ESM2-3B encoder and the adapter that adds its features to the token embeddings.

The adapter is: softmax-weighted sum over the 37 ESM2 layer outputs -> LayerNorm -> Linear ->
GELU -> Linear -> per-dimension gate (initialised to zero before stage-1 training, so a fresh
adapter leaves the language model unchanged).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ESM_NAME = "esm2_t36_3B_UR50D"
ESM_LAYERS = 37   # embedding output + 36 transformer layers
ESM_DIM = 2560
ESM_MAX = 1022    # residues per window (1024 positions including BOS/EOS)


class EsmEncoder:
    """ESM2-3B in bf16. Returns, per sequence, a tensor (L, 37, 2560)."""

    def __init__(self, device: str = "cuda"):
        import esm  # fair-esm

        model, alphabet = getattr(esm.pretrained, ESM_NAME)()
        self.model = model.eval().to(device).to(torch.bfloat16).requires_grad_(False)
        self.conv = alphabet.get_batch_converter()
        self.device = device

    @torch.no_grad()
    def _run(self, seqs: list[str]) -> list[torch.Tensor]:
        _, _, toks = self.conv([(str(i), s) for i, s in enumerate(seqs)])
        out = self.model(toks.to(self.device), repr_layers=list(range(ESM_LAYERS)))
        rep = torch.stack([out["representations"][l] for l in range(ESM_LAYERS)], 2)
        return [rep[i, 1:1 + len(s)] for i, s in enumerate(seqs)]

    @torch.no_grad()
    def __call__(self, seq: str) -> torch.Tensor:
        """Sequences longer than 1022 residues are encoded as two windows (first and last 1022
        residues); each residue takes the window it is more central in."""
        if len(seq) <= ESM_MAX:
            return self._run([seq])[0]
        a = self._run([seq[:ESM_MAX]])[0]
        b = self._run([seq[-ESM_MAX:]])[0]
        cut = len(seq) // 2
        return torch.cat([a[:cut], b[cut - (len(seq) - ESM_MAX):]])


class Adapter(torch.nn.Module):
    """ESM2 per-residue representation -> language-model embedding space."""

    def __init__(self, hidden: int, esm_dim: int = ESM_DIM, n_layers: int = ESM_LAYERS):
        super().__init__()
        self.layer_logits = torch.nn.Parameter(torch.zeros(n_layers))
        self.norm = torch.nn.LayerNorm(esm_dim)
        self.fc1 = torch.nn.Linear(esm_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.gate = torch.nn.Parameter(torch.zeros(hidden))

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """rep: (N, 37, esm_dim) -> (N, hidden)."""
        w = self.layer_logits.softmax(0).to(rep.dtype)
        x = torch.einsum("nld,l->nd", rep, w)
        return self.fc2(F.gelu(self.fc1(self.norm(x)))) * self.gate
