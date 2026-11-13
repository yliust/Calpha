"""Frozen ESM2-3B encoder whose per-residue representations are projected into Qwen's embedding
space and added to the residue tokens.

Pieces
* EsmEncoder   frozen ESM2-3B (fair-esm, bf16), returns all 37 layer outputs per residue. Not an
               nn.Module attribute of the trained model, so FSDP never shards it and checkpoints
               never contain it (it is reloaded from the torch hub cache).
* Adapter      softmax-weighted sum over ESM2 layers (as in ESMFold) -> LayerNorm -> Linear ->
               GELU -> Linear -> per-dimension gate initialised to 0, so a fresh adapter leaves the
               model's outputs exactly unchanged.
* ResidueMapper parse one tokenized sample: which token positions carry which residue index,
               per protein segment (each <seq> starts one; a training sample has one, an
               in-context prompt has one per example plus the query).
               <seq> amino-acid tokens are residues 0..L-1; amino-acid anchors after </seq> are
               residues in order of appearance, except infill, whose <struct> holds the prefix,
               a <span> placeholder and the suffix, and whose answer <span> holds the cut residues.
               Tasks whose output *is* the sequence (inverse_fold) and general text get no features.
* Injector     batches the residue maps of a micro-batch through ESM2 and returns what the model's
               forward needs to add the projected features to the embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

ESM_NAME = "esm2_t36_3B_UR50D"
ESM_LAYERS = 37  # embedding output + 36 transformer layers
ESM_DIM = 2560
ESM_MAX = 1022  # residues per window (1024 positions incl. BOS/EOS)
AMINO = "ACDEFGHIKLMNPQRSTVWY"
INJECT_TASKS = {"fold", "secondary", "infill", "contacts", "fold_with_ss", "fold_c", "fold_cx"}


class EsmEncoder:
    def __init__(self, device: str = "cuda"):
        import esm
        model, alphabet = getattr(esm.pretrained, ESM_NAME)()
        self.model = model.eval().to(device).to(torch.bfloat16)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.alphabet = alphabet
        self.conv = alphabet.get_batch_converter()
        self.device = device

    @torch.no_grad()
    def _run(self, seqs: list[str]) -> list[torch.Tensor]:
        _, _, toks = self.conv([(str(i), s) for i, s in enumerate(seqs)])
        out = self.model(toks.to(self.device), repr_layers=list(range(ESM_LAYERS)))
        rep = torch.stack([out["representations"][l] for l in range(ESM_LAYERS)], 2)  # B, T, 37, D
        return [rep[i, 1:1 + len(s)] for i, s in enumerate(seqs)]

    @torch.no_grad()
    def __call__(self, seqs: list[str], max_residues: int = 12000) -> list[torch.Tensor]:
        """Per sequence: (L, 37, 2560) bf16. Sequences longer than ESM_MAX are encoded as two
        windows (first / last ESM_MAX residues); each residue takes the window it is more central in."""
        jobs = []  # (seq index, window start, window seq)
        for i, s in enumerate(seqs):
            if len(s) <= ESM_MAX:
                jobs.append((i, 0, s))
            else:
                jobs += [(i, 0, s[:ESM_MAX]), (i, len(s) - ESM_MAX, s[-ESM_MAX:])]
        jobs.sort(key=lambda j: len(j[2]))
        reps: dict[tuple[int, int], torch.Tensor] = {}
        k = 0
        while k < len(jobs):
            b = [jobs[k]]
            while k + len(b) < len(jobs) and (len(b) + 1) * len(jobs[k + len(b)][2]) <= max_residues:
                b.append(jobs[k + len(b)])
            for (i, start, _), r in zip(b, self._run([j[2] for j in b])):
                reps[(i, start)] = r
            k += len(b)
        out = []
        for i, s in enumerate(seqs):
            if len(s) <= ESM_MAX:
                out.append(reps[(i, 0)])
            else:
                a, b2 = reps[(i, 0)], reps[(i, len(s) - ESM_MAX)]
                cut = len(s) // 2
                out.append(torch.cat([a[:cut], b2[cut - (len(s) - ESM_MAX):]]))
        return out


class Adapter(torch.nn.Module):
    """Per-residue ESM2 representations -> Qwen embedding space.

    The output end has two forms, depending on whether the projection is **added to** the token
    embedding or **replaces** it:

    * `scale_init=None` (add; the form Calpha uses): a per-dimension gate, **initialised to 0**.
      A freshly built adapter therefore leaves the model's outputs bit-for-bit unchanged, which is
      what the pre-training equivalence test relies on. The scale is anchored for free: the output
      is only a perturbation on top of the real token embedding.

    * `scale_init=<median token-embedding norm>` (replace; not used by Calpha): **a zero gate is
      not an option here**. In replace mode gate=0 would zero the residue positions entirely --
      the model would see a block of zeros instead of a protein vector, which is not "no
      injection" but "injection of zero information". Nothing constrains the output magnitude
      either: the output **is** the token embedding. An earlier concatenation variant missed this;
      after training its projector output norm was 12.80 against 0.884 for real Qwen embeddings,
      a factor of 14.5. The fix used here is the same: RMS-normalise, then multiply by a learnable
      scalar, so training starts inside the distribution Qwen is used to.
    """

    def __init__(self, hidden: int, esm_dim: int = ESM_DIM, n_layers: int = ESM_LAYERS,
                 scale_init: float | None = None):
        super().__init__()
        self.layer_logits = torch.nn.Parameter(torch.zeros(n_layers))
        self.norm = torch.nn.LayerNorm(esm_dim)
        self.fc1 = torch.nn.Linear(esm_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        # The two are mutually exclusive: which one is registered depends on the form. Do not keep
        # both -- a parameter that takes no part in the forward raises "did not receive grad"
        # under DDP and silently gets a zero gradient under FSDP.
        if scale_init is None:
            self.gate = torch.nn.Parameter(torch.zeros(hidden))
            self.scale = None
        else:
            self.gate = None
            self.scale = torch.nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """rep: (N, 37, esm_dim) -> (N, hidden)."""
        w = self.layer_logits.softmax(0).to(rep.dtype)
        x = torch.einsum("nld,l->nd", rep, w)
        y = self.fc2(F.gelu(self.fc1(self.norm(x))))
        if self.scale is None:
            return y * self.gate
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-4) * self.scale


def adapter_from_state(hidden: int, state: dict) -> "Adapter":
    """Pick the form from **which parameter the checkpoint actually stores**, instead of making
    the caller guess.

    The add form stores `gate`, the replace form stores `scale` (see the Adapter docstring).
    If inference always built the add form, loading a replace-form checkpoint would fail in
    load_state_dict -- failing is good, but only after training has finished. This function gets
    it right from the start.
    """
    return Adapter(hidden, scale_init=0.0 if "scale" in state else None)


@dataclass
class Feats:
    """Everything the model's forward needs: rep (N_res, 37, D) and, for every injected token,
    its flat position in the (B, T) batch and the row of rep it takes."""
    rep: torch.Tensor
    tok_pos: torch.Tensor
    res_row: torch.Tensor


class ResidueMapper:
    def __init__(self, tok):
        ids = tok.convert_tokens_to_ids
        self.aa = {ids(f"<aa{a}>"): a for a in AMINO}
        self.aa[ids("<aaX>")] = "X"
        self.seq_open, self.seq_close = ids("<seq>"), ids("</seq>")
        self.struct_open, self.struct_close = ids("<struct>"), ids("</struct>")
        self.span = ids("<span>")

    def __call__(self, ids: list[int], task: str) -> list[tuple[str, list[int], list[int]]]:
        """-> one (sequence, token positions, residue indices) per protein segment; [] = no injection.

        Each <seq> starts a segment that runs to the next <seq> (or the end). Training samples hold
        one segment; in-context prompts hold k example folds followed by the query, each mapped to
        its own sequence. A segment truncated inside its <seq> is dropped without affecting the
        segments before it."""
        if task not in INJECT_TASKS:
            return []
        starts = [p for p, t in enumerate(ids) if t == self.seq_open]
        bounds = starts + [len(ids)]
        segs = (self._segment(ids, s, e, task) for s, e in zip(bounds, bounds[1:]))
        return [m for m in segs if m is not None]

    def _segment(self, ids: list[int], start: int, end: int, task: str) -> tuple[str, list[int], list[int]] | None:
        a = start + 1
        if self.seq_close not in ids[a:end]:
            return None  # truncated inside the prompt
        b = ids.index(self.seq_close, a, end)
        seq = "".join(self.aa[t] for t in ids[a:b] if t in self.aa)
        if len(seq) != b - a:
            return None
        pos, res = list(range(a, b)), list(range(len(seq)))
        rest = [p for p in range(b + 1, end) if ids[p] in self.aa]
        # infill numbers its suffix from len(seq) - len(prefix) - len(suffix), so it assumes the
        # sample is not truncated inside <struct>; true for the training data
        if task == "infill" and self.struct_open in ids[start:end]:
            so = ids.index(self.struct_open, start, end)
            ph = ids.index(self.span, so, end)  # placeholder inside <struct>
            sc = ids.index(self.struct_close, ph, end) if self.struct_close in ids[ph:end] else end
            pre = [p for p in rest if so < p < ph]
            suf = [p for p in rest if ph < p < sc]
            ans = [p for p in rest if p > sc]
            s = len(pre)
            span = len(seq) - s - len(suf)
            idx = list(range(s)) + list(range(s + span, s + span + len(suf))) + list(range(s, s + len(ans)))
            rest = pre + suf + ans
        else:
            idx = list(range(len(rest)))
        keep = [(p, r) for p, r in zip(rest, idx) if r < len(seq)]
        pos += [p for p, _ in keep]
        res += [r for _, r in keep]
        return seq, pos, res


class Injector:
    def __init__(self, tok, encoder: EsmEncoder):
        self.map = ResidueMapper(tok)
        self.enc = encoder

    def __call__(self, ids: torch.Tensor, tasks: list[str]) -> Feats | None:
        """ids: (B, T) token ids (CPU or GPU); tasks: task name per row."""
        B, T = ids.shape
        rows = ids.tolist()
        seqs, tok_pos, res_row = [], [], []
        base = 0
        for i, (row, task) in enumerate(zip(rows, tasks)):
            for seq, pos, res in self.map(row, task):
                seqs.append(seq)
                tok_pos += [i * T + p for p in pos]
                res_row += [base + r for r in res]
                base += len(seq)
        if not seqs:
            return None
        rep = torch.cat(self.enc(seqs))
        dev = rep.device
        return Feats(rep, torch.tensor(tok_pos, device=dev), torch.tensor(res_row, device=dev))


def inject(embeds: torch.Tensor, adapter: Adapter, feats: Feats | None,
           mode: str = "add") -> torch.Tensor:
    """embeds: (B, T, H) -> put the projected ESM features at the residue positions.

    mode="add" (what Calpha uses): `embeds[residue positions] += adapter(ESM2)`.
        At those positions the model sees "amino-acid token embedding + ESM feature".

    mode="replace" (implemented and tested, not used for the released model):
        `embeds[residue positions] = adapter(ESM2)`, **replacing the token embedding outright**.
        The model then sees only the protein-modality vector there and no amino-acid embedding at
        all -- the amino-acid tokens become pure placeholders, the same role an <image>
        placeholder plays in a vision-language model.
        The amino-acid tokens still stay in the stream because the sequence ESM needs can only be
        recovered from the token stream: the data pipeline (jsonl -> pretokenize -> Shards ->
        collate) passes token ids only and carries no sequence field.
    """
    if mode not in ("add", "replace"):
        raise ValueError(f"unknown inject mode {mode}")
    if feats is None:
        # keep every adapter parameter in the graph (zero contribution) so all ranks produce the
        # same set of gradients for FSDP's reduce-scatter
        dummy = adapter(torch.zeros(1, adapter.layer_logits.numel(), adapter.norm.normalized_shape[0],
                                    device=embeds.device, dtype=embeds.dtype))
        return embeds + 0.0 * dummy.sum()
    proj = adapter(feats.rep.to(embeds.dtype))
    flat = embeds.reshape(-1, embeds.shape[-1])
    src = proj[feats.res_row].to(flat.dtype)
    if mode == "add":
        add = torch.zeros_like(flat).index_add(0, feats.tok_pos, src)
        return (flat + add).reshape(embeds.shape)
    # replace: use index_add to subtract the original embedding at those positions and add the new
    # vector. Equivalent to assignment but stays differentiable and does not break autograd with
    # an in-place write (flat is the output of embed_tokens).
    delta = torch.zeros_like(flat).index_add(0, feats.tok_pos, src)
    keep = torch.ones_like(flat).index_fill(0, feats.tok_pos, 0.0)
    return (flat * keep + delta).reshape(embeds.shape)
