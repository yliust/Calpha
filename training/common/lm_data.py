"""Language-side data stream and FSDP2 helpers shared by the stage-1 trainer.

This is the subset of the project's earlier LM-only trainer that stage 1 actually uses; the
LM-only training loop itself and its options (pair-attention bias, soft structure-token targets,
bit-level losses, prefix noise) are not part of Calpha and were left out.

* Data: pre-tokenized shards (data/pretokenize.py) streamed in file order (epochs are already
  shuffled). Windows of WINDOW samples are length-sorted and cut into micro-batches with at
  most `micro_tokens` padded tokens; micro-batches are shuffled and dealt round-robin to
  ranks, so every rank runs the same number of forwards (required by FSDP collectives).
  No cross-sample packing: right padding keeps the causal GatedDeltaNet / attention
  outputs of real tokens exact without needing varlen kernels.
* Loss: next-token CE on every non-pad token. Logits are computed in checkpointed chunks so the
  252k-vocab logits never fully materialise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from pretokenize import TASKS  # noqa: E402
from esm_adapter import Feats  # noqa: E402

WINDOW = 8192
CE_CHUNK = 2048
IGNORE = -100


# ----------------------------------------------------------------------------- data
class Shards:
    def __init__(self, files: list[Path]):
        self.files = files
        self.idx = [np.load(f.with_suffix(".idx.npy")) for f in files]
        self._tok: dict[int, np.ndarray] = {}
        self._ids: dict[int, list[str]] = {}

    def sample_id(self, f: int, off: int) -> str:
        """Protein / document id of the sample at token offset off of file f (from <file>.meta.parquet)."""
        if f not in self._ids:
            import pyarrow.parquet as pq
            self._ids[f] = pq.read_table(self.files[f].with_suffix(".meta.parquet"), columns=["id"]).column("id").to_pylist()
        k = int(np.searchsorted(self.idx[f][:, 0], off))
        assert self.idx[f][k, 0] == off
        return self._ids[f][k]

    def tokens(self, f: int) -> np.ndarray:
        if f not in self._tok:
            self._tok = {f: np.load(self.files[f].with_suffix(".tok.npy"), mmap_mode="r")}
        return self._tok[f]

    def samples(self):
        for f, idx in enumerate(self.idx):
            for off, n, task in idx:
                yield f, int(off), int(n), int(task)


def micro_batches(shards: Shards, micro_tokens: int, max_len: int, world: int, seed: int):
    """Yield (global_mb_index, [samples]) in a deterministic order; each sample = (file, off, len, task)."""
    rng = np.random.default_rng(seed)
    gidx = 0
    buf: list = []
    stream = shards.samples()
    while True:
        buf.clear()
        for s in stream:
            buf.append((s[0], s[1], min(s[2], max_len), s[3]))
            if len(buf) == WINDOW:
                break
        if not buf:
            return
        buf.sort(key=lambda s: s[2])
        mbs, cur = [], []
        for s in buf:
            if cur and (len(cur) + 1) * s[2] > micro_tokens:
                mbs.append(cur)
                cur = []
            cur.append(s)
        if cur:
            mbs.append(cur)
        rng.shuffle(mbs)
        mbs = mbs[: len(mbs) - len(mbs) % world]
        for mb in mbs:
            yield gidx, mb
            gidx += 1


def collate(shards: Shards, mb: list, pad_id: int):
    n = max(s[2] for s in mb)
    ids = torch.full((len(mb), n), pad_id, dtype=torch.long)
    labels = torch.full((len(mb), n), IGNORE, dtype=torch.long)
    tasks = torch.zeros((len(mb), n), dtype=torch.long)
    for i, (f, off, ln, task) in enumerate(mb):
        t = torch.from_numpy(shards.tokens(f)[off:off + ln].astype(np.int64))
        ids[i, :ln] = t
        labels[i, : ln - 1] = t[1:]
        tasks[i, :ln] = task
    return ids, labels, tasks


# ----------------------------------------------------------------------------- loss
def _ce_chunk(h, y, w):
    """Per-token cross-entropy and argmax prediction for one chunk of hidden states."""
    logits = F.linear(h, w).float()
    return F.cross_entropy(logits, y, reduction="none"), logits.argmax(-1)


def mask_leaky_targets(labels: torch.Tensor, feats: Feats | None) -> None:
    """No loss on amino-acid tokens of samples that get ESM2 features. ESM2 is bidirectional, so the
    feature added at residue i already encodes residue i+1: predicting the next <seq> amino acid or
    the next anchor becomes a shortcut that drew the gradient away from structure (an early run
    had to be stopped at step ~210 because of it). Those tokens are inputs, not targets. labels is
    modified in place."""
    if feats is None:
        return
    T = labels.shape[1]
    p = feats.tok_pos.cpu()
    p = p[p % T > 0]
    labels.view(-1)[p - 1] = IGNORE


def run_injector(injector, shards: Shards, mb: list, ids: torch.Tensor, train: bool = True):
    if injector is None:
        return None
    tasks = [TASKS[s[3]] for s in mb]
    return injector(ids, tasks)


# ----------------------------------------------------------------------------- FSDP
def reshard_all(root) -> None:
    """Return every FSDP module to its sharded state. Forward-only passes (eval) leave the root
    module (lm_head, final norm) unsharded because no backward follows; state-dict code expects
    sharded DTensors."""
    for m in root.modules():
        if isinstance(m, FSDPModule):
            m.reshard()
