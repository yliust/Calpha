"""Stage 1: joint training of a dual-output model (LM head + flow structure head).

Unlike stage 2 (which trains only a conditioning front-end with everything else frozen), here the
flow gradient **passes through the condition projection back into the Qwen trunk**, so the trunk
adapts to "provide a good condition for the structure head". The only frozen component of the
whole system is ESM2-3B.

                            +--> LM head ----------------> text / discrete protein tokens
Qwen3.5-9B trunk -----------+
    ^                       +--> hidden states at <seq> residues --> condition projection
    |                                                                --> flow head --> all-atom coordinates
ESM2-3B (frozen) -> adapter +

Every micro-step takes two micro-batches:
  LM path         pre-tokenized shards (protein tasks in chat format + general text), chunked CE
  structure path  SimpleFold's boltz-style data, rendered on the fly into a chat-format prompt,
                  flow-matching MSE + smooth LDDT

    L = L_LM + L_trigger + lambda_f * (L_flow + alpha(t) * L_LDDT)

lambda_f is fixed for the whole run (--lambda-flow); it is not adjusted during training.

The FSDP2 plumbing (sharded data stream, chunked CE, reshard_all) lives in common/lm_data.py.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import lm_data as mt  # noqa: E402  data stream + FSDP2 helpers
from esm_adapter import AMINO, Adapter, EsmEncoder  # noqa: E402

# Upstream apple/ml-simplefold checkout, directory <ml-simplefold>/src/simplefold.
if not os.environ.get("SIMPLEFOLD_SRC"):
    raise SystemExit("SIMPLEFOLD_SRC is not set; point it at <ml-simplefold>/src/simplefold")
SF_SRC = Path(os.environ["SIMPLEFOLD_SRC"])
sys.path.insert(0, str(SF_SRC))

STRUCT_TOKEN = "<structure>"   # the model emits it by itself to trigger folding


# ---------------------------------------------------------------- chat-format rendering

# English only.
# The wording must be clearly different from build_mixture.CHAT_ASK: if the same question were
# answered with discrete structure tokens on the LM path and with <structure> here, the model
# could not learn a deterministic trigger. In an earlier round "Fold this sequence:" and
# "Fold this sequence for me." were nearly the same sentence, which was the direct cause of an
# English trigger rate of only 5% at 2B tokens.
# The data mixture used for Calpha removes the discrete-token folding tasks altogether; keeping
# the questions apart here is a second safeguard.
CHAT_TEMPLATES = [
    ("Fold this protein and give me the 3D structure.\nSequence: {seq}",
     "Sure, here is the predicted structure. {tag}"),
    ("What is the 3D structure of this protein?\n{seq}", "Here is the predicted structure. {tag}"),
    ("What does this protein look like in 3D? Please show the structure.\n{seq}", "{tag}"),
    ("Model the three-dimensional fold of:\n{seq}", "{tag}"),
    ("Given the sequence {seq}, output its folded structure.", "The structure is: {tag}"),
]


def render_chat_parts(seq: str, rng: np.random.Generator) -> tuple[str, str]:
    """-> (head, body). head = user turn + assistant role header, body = the assistant's answer.

    They are kept apart so that the "trigger loss" is computed on the body only: it teaches the
    model to answer <structure> when asked to fold. The boundary between the two is a special
    token (right after <|im_start|>assistant\\n), so tokenizing them separately and concatenating
    equals tokenizing the whole string.
    """
    user, assistant = CHAT_TEMPLATES[rng.integers(len(CHAT_TEMPLATES))]
    aa = "".join(f"<aa{a if a in AMINO else 'X'}>" for a in seq)
    head = (f"<|im_start|>user\n{user.format(seq=f'<seq>{aa}</seq>')}<|im_end|>\n"
            f"<|im_start|>assistant\n")
    body = f"{assistant.format(tag=STRUCT_TOKEN)}<|im_end|>"
    return head, body


def render_chat(seq: str, rng: np.random.Generator) -> str:
    """Wrap one folding request as a chat exchange; <structure> is emitted by the model and then
    handed to the flow head."""
    user, assistant = CHAT_TEMPLATES[rng.integers(len(CHAT_TEMPLATES))]
    aa = "".join(f"<aa{a if a in AMINO else 'X'}>" for a in seq)
    return (f"<|im_start|>user\n{user.format(seq=f'<seq>{aa}</seq>')}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant.format(tag=STRUCT_TOKEN)}<|im_end|>")


# ---------------------------------------------------------------- dual-head root module

class DualHead(torch.nn.Module):
    """LM head + flow structure head sharing the Qwen trunk.

    This is the FSDP root module, so lm_head / norm / the flow head are all covered by the root's
    forward and backward hooks -- lm_data.reshard_all() applies here as well (that is the fix for
    a root cause that crashed an early pilot three times in a row).
    """

    def __init__(self, model, adapter: Adapter, flow_head, cond_dim: int, layers: tuple[int, ...],
                 inject_mode: str = "add"):
        super().__init__()
        self.lm = model
        self.adapter = adapter
        # "add" (used for Calpha): ESM features are added to the amino-acid token embeddings.
        # "replace": they replace the token embeddings outright -- at residue positions the model
        #   sees only the protein modality and the amino-acid tokens become placeholders.
        self.inject_mode = inject_mode
        self.flow = flow_head
        self.cond_layers = layers
        hidden = model.get_input_embeddings().weight.shape[1]
        # Condition projection: a weighted sum of several hidden-state layers mapped to the flow
        # head's condition width. Same structure as SimpleFold's esm_s_combine (softmax layer
        # weights); the gate starts at 0, so at the start of training the flow branch contributes
        # no gradient to the trunk, which makes an equivalence test against the plain LM easy.
        self.cond_logits = torch.nn.Parameter(torch.zeros(len(layers)))
        self.cond_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, cond_dim))
        self.cond_gate = torch.nn.Parameter(torch.zeros(cond_dim))

    def forward(self, mode: str, *args):
        """Every computation must be dispatched through forward.

        FSDP2 all-gathers the sharded parameters into full tensors only at the entry of the root
        module's forward. Calling self._lm_loss(...) / self._condition(...) directly bypasses the
        forward hook: the parameters are still DTensors, and any operation with a plain tensor
        (ESM features, atom coordinates) fails with "got mixed torch.Tensor and DTensor".
        """
        if mode == "lm":
            return self._lm_loss(*args)
        if mode == "cond":
            return self._condition(*args)
        if mode == "cond_lm":
            return self._condition_and_lm(*args)
        raise ValueError(mode)

    def trunk(self, ids, feats):
        """Trunk forward; returns the last hidden state and all hidden_states layers."""
        emb = self.lm.model.embed_tokens(ids)
        if self.adapter is not None and feats is not None:
            from esm_adapter import inject
            emb = inject(emb, self.adapter, feats, self.inject_mode)
        out = self.lm.model(inputs_embeds=emb, use_cache=False, output_hidden_states=True)
        return out.last_hidden_state, out.hidden_states

    def _lm_loss(self, ids, labels, feats):
        """Chunked next-token CE over all non-ignored positions."""
        h, _ = self.trunk(ids, feats)
        h, y = h.reshape(-1, h.shape[-1]), labels.reshape(-1)
        keep = (y != mt.IGNORE).nonzero().squeeze(1)
        h, y = h[keep], y[keep]
        w = self.lm.lm_head.weight
        if h.shape[0] == 0:
            return h.new_zeros(0) + 0.0 * w.sum(), y, keep
        from torch.utils.checkpoint import checkpoint
        losses, preds = [], []
        for i in range(0, h.shape[0], mt.CE_CHUNK):
            lc, pc_ = checkpoint(mt._ce_chunk, h[i:i + mt.CE_CHUNK], y[i:i + mt.CE_CHUNK], w, use_reentrant=False)
            losses.append(lc)
            preds.append(pc_)
        return torch.cat(losses), torch.cat(preds), keep

    def _condition_and_lm(self, ids, feats, res_pos: list[list[int]], max_len: int, labels):
        """One trunk forward gives both the structure condition and the per-token CE of the
        assistant segment (the trigger loss)."""
        h_last, hs = self.trunk(ids, feats)
        cond = self._cond_from_hidden(hs, res_pos, max_len)
        h, y = h_last.reshape(-1, h_last.shape[-1]), labels.reshape(-1)
        keep = (y != mt.IGNORE).nonzero().squeeze(1)
        w = self.lm.lm_head.weight
        if keep.numel() == 0:
            return cond, h.new_zeros(0) + 0.0 * w.sum()
        from torch.utils.checkpoint import checkpoint
        ce, _ = checkpoint(mt._ce_chunk, h[keep], y[keep], w, use_reentrant=False)
        return cond, ce

    def _cond_from_hidden(self, hs, res_pos, max_len):
        sel = torch.stack([hs[l] for l in self.cond_layers], 2)        # (B, T, n_layers, H)
        w = self.cond_logits.softmax(0)
        mixed = torch.einsum("n,btnh->bth", w, sel.float())
        out = mixed.new_zeros(len(res_pos), max_len, mixed.shape[-1])
        for b, pos in enumerate(res_pos):
            n = min(len(pos), max_len)
            if n:
                out[b, :n] = mixed[b, torch.as_tensor(pos[:n], device=mixed.device)]
        return self.cond_proj(out) * self.cond_gate

    def _condition(self, ids, feats, res_pos: list[list[int]], max_len: int):
        """-> (B, L, cond_dim): hidden states at the residue positions of the <seq> segment,
        layer-weighted and projected."""
        _, hs = self.trunk(ids, feats)
        sel = torch.stack([hs[l] for l in self.cond_layers], 2)        # (B, T, n_layers, H)
        w = self.cond_logits.softmax(0)
        mixed = torch.einsum("n,btnh->bth", w, sel.float())            # (B, T, H)
        out = mixed.new_zeros(len(res_pos), max_len, mixed.shape[-1])
        for b, pos in enumerate(res_pos):
            n = min(len(pos), max_len)
            if n:
                out[b, :n] = mixed[b, torch.as_tensor(pos[:n], device=mixed.device)]
        return self.cond_proj(out) * self.cond_gate

    def flow_dummy(self):
        """Keep the flow parameters in the graph when a rank's micro-batch has no structure sample.

        Under FSDP, if one rank skips a module it does not take part in the reduce-scatter and the
        other ranks hang forever (this was hit before with another module; the 8-GPU hang was only
        found by the watchdog).
        """
        z = 0.0 * self.cond_logits.sum() + 0.0 * self.cond_gate.sum()
        for p in self.cond_proj.parameters():
            z = z + 0.0 * p.sum()
        for p in self.flow.parameters():
            z = z + 0.0 * p.sum()
        return z


# ---------------------------------------------------------------- structure data stream

class StructStream:
    """SimpleFold's boltz-style training set -> (chat prompt tokens, residue positions, structure
    features, coordinates)."""

    def __init__(self, token_dir: Path, target_dir: Path, tok, max_res: int, seed: int,
                 rank: int, world: int):
        from boltz_data_pipeline.crop.boltz import BoltzCropper
        from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
        from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
        from boltz_data_pipeline.types import Manifest
        from datasets.train_datamodule import SimpleFoldTrainingDataset
        from utils.datamodule_utils import Dataset, collate

        ds = Dataset(tokenized_dir=str(token_dir), target_dir=target_dir, esm_dir="",
                     connectivity_dir="", cheap_dir="", manifest=Manifest.load(token_dir / "manifest.json"),
                     cropper=BoltzCropper(), tokenizer=BoltzTokenizer(), featurizer=BoltzFeaturizer())
        self.ds = SimpleFoldTrainingDataset(datasets=[ds], symmetries={}, max_atoms=max_res * 9,
                                            max_tokens=max_res, pad_to_max_atoms=False,
                                            pad_to_max_tokens=False)
        _guard_dataset(self.ds)
        self.collate = collate
        self.tok = tok
        self.rng = np.random.default_rng(seed + rank)
        self.order = np.random.default_rng(seed).permutation(self.ds.num_samples)[rank::world]
        self.cursor = 0

    def batch(self, n: int):
        items, seqs = [], []
        while len(items) < n:
            idx = int(self.order[self.cursor % len(self.order)])
            self.cursor += 1
            it = self.ds[idx]
            items.append(it)
            seqs.append(it["aa_seq"])
        batch = self.collate(items)
        enc, heads = [], []
        for s in seqs:
            head, body = render_chat_parts(s, self.rng)
            h = self.tok(head, add_special_tokens=False).input_ids
            b = self.tok(body, add_special_tokens=False).input_ids
            enc.append(h + b)
            heads.append(len(h))
        self.last_heads = heads
        return batch, seqs, enc


def _guard_dataset(ds) -> None:
    """Resample when a record fails. Lesson from an earlier experiment: nucleic-acid residues
    crashed four training runs at once, and a 4-step smoke test had not drawn any of them."""
    raw = ds.__class__.__getitem__

    def safe(self, idx: int, _d: int = 0):
        try:
            return raw(self, idx)
        except Exception:  # noqa: BLE001
            if _d >= 8:
                raise
            import random as _r
            return safe(self, _r.randint(0, self.num_samples - 1), _d + 1)

    ds.__class__.__getitem__ = safe


def residue_positions(ids: list[int], seq_open: int, seq_close: int) -> list[int]:
    """Positions of the residue tokens between <seq> and </seq> in the prompt token stream."""
    if seq_open not in ids or seq_close not in ids:
        return []
    a = ids.index(seq_open) + 1
    b = ids.index(seq_close, a)
    return list(range(a, b))


def pad_ids(enc: list[list[int]], pad_id: int, device) -> torch.Tensor:
    m = max(len(e) for e in enc)
    out = torch.full((len(enc), m), pad_id, dtype=torch.long)
    for i, e in enumerate(enc):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out.to(device)


def smooth_lddt(pred, true, mask, alpha):
    """SimpleFold's smooth LDDT (same implementation as in train_stage2.py), with the alpha(t) weight."""
    dp, dt = torch.cdist(pred, pred), torch.cdist(true, true)
    err = (dp - dt).abs()
    eps = sum(torch.sigmoid(c - err) for c in (0.5, 1.0, 2.0, 4.0)) / 4.0
    m = mask[:, :, None] * mask[:, None, :] * (dt < 15.0).float()
    m = m * (1 - torch.eye(m.shape[1], device=m.device))[None]
    lddt = (eps * m).sum((1, 2)) / (m.sum((1, 2)) + 1e-6)
    return ((1.0 - lddt) * alpha).mean()


def logit_normal(n: int, m: float, s: float) -> torch.Tensor:
    return torch.sigmoid(torch.randn(n) * s + m)


# ---------------------------------------------------------------- model construction

def build_flow_head(ckpt: Path | None, model_name: str = "simplefold_100M"):
    """SimpleFold structure head. Initialised from the public pretrained weights (default) or
    randomly; **it is trained throughout and never frozen**."""
    import hydra
    import omegaconf
    from inference import get_config_path

    cfg = omegaconf.OmegaConf.load(get_config_path(f"configs/model/architecture/foldingdit_{model_name[11:]}.yaml"))
    head = hydra.utils.instantiate(cfg)
    if ckpt is not None and Path(ckpt).exists():
        head.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=True)
    return head


def build_dual(path: str, flow_ckpt: Path | None, cond_layers: tuple[int, ...], esm: bool = True,
               shard: bool | None = None, inject_mode: str = "add",
               flow_model: str = "simplefold_100M"):
    """With shard=None this is decided automatically: FSDP sharding only if the process group is
    initialised.

    Single-GPU checks and inference have no RANK etc. in the environment, and calling fully_shard
    there raises 'environment variable RANK expected'."""
    from transformers import AutoModelForCausalLM
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    import conditioning as sc

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    # In replace mode the adapter output **is** the embedding at the residue positions, so it has
    # to start at the same magnitude as real Qwen embeddings; in add mode it is only a
    # perturbation and keeps the zero gate (see the Adapter docstring).
    emb_w = model.get_input_embeddings().weight
    scale_init = float(emb_w.float().norm(dim=1).median()) if inject_mode == "replace" else None
    adapter = Adapter(emb_w.shape[1], scale_init=scale_init) if esm else None
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    head = build_flow_head(flow_ckpt, flow_model)
    cond_dim = head.esm_s_proj.proj[0].in_features
    # The condition now comes from the LLM: both the layer count and the width change, so these
    # two modules are rebuilt (the same function stage 2 uses)
    sc.retarget_condition(head, n_layers=1, dim=cond_dim, reinit=True)

    root = DualHead(model, adapter, head, cond_dim, cond_layers, inject_mode)
    if shard is None:
        shard = dist.is_available() and dist.is_initialized()
    if shard:
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model.model.embed_tokens, mp_policy=mp)
        # The flow head and the condition projection are not sharded: together they are about
        # 130 M parameters, so a full copy per GPU is cheap, whereas after sharding the bmm inside
        # them fails on "DTensor mixed with plain tensor" (the atom features in the batch are
        # plain tensors and cannot become DTensors).
        keep = {p for n, p in root.named_parameters() if n.startswith(("flow.", "cond_"))}
        fully_shard(root, mp_policy=mp, ignored_params=keep)
    return root


# ---------------------------------------------------------------- structure loss

def trigger_labels(enc: list[list[int]], heads: list[int], device) -> torch.Tensor:
    """Next-token labels for every token of the assistant segment (after head); IGNORE elsewhere."""
    T = max(len(e) for e in enc)
    lab = torch.full((len(enc), T), mt.IGNORE, dtype=torch.long)
    for b, (e, h) in enumerate(zip(enc, heads)):
        for t in range(h - 1, len(e) - 1):
            lab[b, t] = e[t + 1]
    return lab.to(device)


def flow_step(root, batch, enc, processor, path_, device, tok, copies: int, phase: str,
              injector=None, heads: list[int] | None = None):
    """One structure forward -> (flow_mse, lddt, trig). The condition is taken from the trunk's
    hidden states at the <seq> residue positions. When heads is given the trigger loss (teaching
    the model to answer <structure>) is computed in the same forward."""
    from utils.boltz_utils import center_random_augmentation, weighted_rigid_align

    seq_open = tok.convert_tokens_to_ids("<seq>")
    seq_close = tok.convert_tokens_to_ids("</seq>")
    res_pos = [residue_positions(e, seq_open, seq_close) for e in enc]
    ids = pad_ids(enc, tok.pad_token_id or 0, device)

    batch = processor.preprocess_training(batch)
    L = batch["res_type"].shape[1]
    # The structure path must inject ESM2 as well: the condition is taken from the hidden states
    # of the <seq> segment, and the embeddings at those positions need adapter(ESM2) added first.
    # The first version passed None here -- the structure head then saw a representation without
    # any evolutionary information and the main experiment would silently have become the no-ESM
    # ablation (measured elsewhere: that only reaches the floor, TM about 0.35).
    # In the --no-esm ablation the injector is None, so nothing is injected.
    feats = injector(ids, ["fold"] * len(enc)) if injector is not None else None
    if heads is not None:
        cond, trig_ce = root("cond_lm", ids, feats, res_pos, L, trigger_labels(enc, heads, device))
        trig = trig_ce.mean() if trig_ce.numel() else trig_ce.sum()
    else:
        cond = root("cond", ids, feats, res_pos, L)            # (B, L, cond_dim)
        trig = cond.new_zeros(())
    batch["esm_s"] = cond.unsqueeze(2).repeat_interleave(copies, dim=0)   # (B*c, L, 1, C)

    n = batch["coords"].shape[0]
    t = (0.98 * logit_normal(n, 0.8, 1.7) + 0.02 * torch.rand(n)).to(device) * (1 - 2e-4) + 1e-4
    noise = torch.randn_like(batch["coords"])
    _, y_t, v_t = path_.interpolant(t, noise, batch["coords"])
    out = root.flow(y_t, t, batch)

    resolved = batch["atom_resolved_mask"].float()
    w = y_t.new_ones(y_t.shape[:2])
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        vd = out["predict_velocity"].detach().float()
        denoised = y_t + vd * (1.0 - t[:, None, None])
        aligned = weighted_rigid_align(batch["coords"].detach().float(), denoised, w.float(), mask=resolved)
        _, _, target = path_.interpolant(t, noise, aligned)

    mse = F.mse_loss(out["predict_velocity"], target, reduction="none")
    mse = ((mse * (resolved * w)[..., None]).sum(1) / (resolved.sum(1, keepdim=True) * 3 + 1e-6)).mean()

    denoised = y_t + out["predict_velocity"] * (1.0 - t[:, None, None])
    dc = center_random_augmentation(denoised, batch["atom_pad_mask"], augmentation=False, centering=True) * processor.scale
    tc = center_random_augmentation(batch["coords"], batch["atom_pad_mask"], augmentation=False, centering=True) * processor.scale
    alpha = torch.ones_like(t) if phase == "pretrain" else 1.0 + 8.0 * F.relu(t - 0.5)
    return mse, smooth_lddt(dc, tc, resolved, alpha), trig


# ---------------------------------------------------------------- in-training generation probe

class StructProbe:
    """Periodically fold a small batch of proteins for real during training (an early signal must
    not consist of losses only).

    Takes the targets of the given evaluation parquet that were released after 2020-05-01 (clean
    with respect to SimpleFold's initial weights) and are at most max_len residues long.
    All ranks compute the condition for every target in the same order (the trunk forward has
    FSDP collectives and must stay in sync); each rank then runs the sampler only for the targets
    assigned to it (the flow head is not sharded, so that part is purely local).
    The metrics come in two halves:
      reference-based    TM, lDDT (CA)
      reference-free     fraction of consecutive CA-CA distances within 3.6-4.0 A, Rg / L^0.34
                         (about 2.8 for native structures)
    """

    def __init__(self, eval_parquet: Path, out_dir: Path, tok, rank: int, world: int, device,
                 ccd: Path, n: int = 16, max_len: int = 200, steps: int = 100):
        import argparse as _ap
        import pyarrow.parquet as pq
        from inference import initialize_others
        from utils.fasta_utils import process_fastas

        rows = pq.read_table(eval_parquet, columns=["id", "sequence", "coords", "length",
                                                     "release_date"]).to_pylist()
        rows = [r for r in rows if r["length"] <= max_len and str(r["release_date"]) >= "2020-05-01"]
        self.rows = rows[:n]
        self.rank, self.world, self.device, self.tok = rank, world, device, tok
        self.work = out_dir / "probe_cache" / f"rank{rank}"
        fdir = self.work / "fastas"
        fdir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(self.rows):
            (fdir / f"p{i:03d}.fasta").write_text(f">p{i:03d}|protein|\n{r['sequence']}\n")
        if not (self.work / "structures").exists():
            process_fastas(data=sorted(fdir.glob("*.fasta")), out_dir=self.work, ccd_path=ccd)
        sfa = _ap.Namespace(backend="torch", num_steps=steps, tau=0.01, nsample_per_protein=1)
        (self.tokenizer, self.featurizer, self.processor,
         self.path, self.sampler) = initialize_others(sfa, device)
        self.open = tok.convert_tokens_to_ids("<seq>")
        self.close = tok.convert_tokens_to_ids("</seq>")

    @torch.no_grad()
    def run(self, root, injector) -> dict:
        from copy import deepcopy

        from ca_metrics import all_metrics
        from utils.boltz_utils import process_structure, save_structure
        from utils.datamodule_utils import process_one_inference_structure

        acc = torch.zeros(5, device=self.device)          # tm, lddt, bond_ok, rg_norm, n
        user, assistant = CHAT_TEMPLATES[0]
        for i, r in enumerate(self.rows):
            batch, structure, record = process_one_inference_structure(
                self.work / "structures" / f"p{i:03d}.npz", self.work / "records" / f"p{i:03d}.json",
                self.tokenizer, self.featurizer, self.processor)
            L = batch["res_type"].shape[1]
            aa = "".join(f"<aa{a if a in AMINO else 'X'}>" for a in r["sequence"])
            prompt = (f"<|im_start|>user\n{user.format(seq=f'<seq>{aa}</seq>')}<|im_end|>\n"
                      f"<|im_start|>assistant\n{assistant.format(tag=STRUCT_TOKEN)}<|im_end|>")
            ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            pos = residue_positions(ids[0].tolist(), self.open, self.close)
            feats = injector(ids, ["fold"]) if injector is not None else None
            cond = root("cond", ids, feats, [pos], L)       # computed on every rank: the trunk forward has collectives
            if i % self.world != self.rank:
                continue
            batch["esm_s"] = cond.unsqueeze(2).float()
            noise = torch.randn_like(batch["coords"])
            out = self.processor.postprocess(self.sampler.sample(root.flow, self.path, noise, batch), batch)
            st = process_structure(deepcopy(structure), out["denoised_coords"][0],
                                   batch["atom_pad_mask"][0], record, backend="torch")
            cif_dir = self.work / "pred"
            cif_dir.mkdir(exist_ok=True)
            save_structure(st, cif_dir, f"p{i:03d}", "mmcif")
            ca = _ca_from_cif(cif_dir / f"p{i:03d}.cif", r["length"])
            ref = np.asarray(r["coords"], dtype=np.float32).reshape(-1, 5, 3)[:, 1]
            m = all_metrics(ca, ref)
            ok = np.isfinite(ca).all(1)
            d = np.linalg.norm(np.diff(ca[ok], axis=0), axis=1)
            c = ca[ok] - ca[ok].mean(0)
            rg = float(np.sqrt((c ** 2).sum(1).mean())) / max(ok.sum(), 1) ** 0.34
            acc += torch.tensor([m["tm"], m["lddt"], float(((d > 3.6) & (d < 4.0)).mean()), rg, 1.0],
                                device=self.device)
        dist.all_reduce(acc)
        n = max(float(acc[4]), 1.0)
        return {"probe_tm": round(float(acc[0]) / n, 4), "probe_lddt": round(float(acc[1]) / n, 4),
                "probe_bond_ok": round(float(acc[2]) / n, 4), "probe_rg_norm": round(float(acc[3]) / n, 3),
                "probe_n": int(n)}


def _ca_from_cif(path: Path, length: int) -> np.ndarray:
    import gemmi
    ca = np.full((length, 3), np.nan, dtype=np.float64)
    st = gemmi.read_structure(str(path))
    for i, res in enumerate(st[0][0]):
        if i >= length:
            break
        at = res.find_atom("CA", "*")
        if at is not None:
            ca[i] = (at.pos.x, at.pos.y, at.pos.z)
    return ca


# ---------------------------------------------------------------- checkpoints

def _local(t):
    """For a sharded parameter, this rank's slice; unsharded parameters such as the flow head
    (ignored_params) are plain tensors already."""
    from torch.distributed.tensor import DTensor
    return t.to_local() if isinstance(t, DTensor) else t


def save_ckpt(root, opt, out: Path, step: int, cursor: dict, keep: int = 2) -> None:
    """Every rank saves its own share of the parameters and optimizer state (this avoids
    torch.distributed.checkpoint, whose collective planning phase hung on earlier runs).

    The flow head and the condition projection are not FSDP-sharded: they are plain tensors and
    have no .to_local() -- the very first save of this trainer died on exactly that.
    """
    import shutil
    rank = dist.get_rank()
    mt.reshard_all(root)
    path = out / "ckpt" / f"step_{step:07d}"
    path.mkdir(parents=True, exist_ok=True)
    model = {n: _local(p.detach()).cpu() for n, p in root.named_parameters()}
    osd = opt.state_dict()
    state = {i: {k: _local(v).cpu() if torch.is_tensor(v) else v for k, v in st.items()}
             for i, st in osd["state"].items()}
    tmp = path / f"rank{rank:02d}.pt.tmp"
    # The random-number state is saved too: the structure loss draws timesteps and noise every
    # step, and without restoring it the loss after a resume cannot be compared against an
    # uninterrupted run, which would make the resume check meaningless
    import random as _pyrandom
    # The global numpy / python generators as well: SimpleFold's cropper draws crop windows with
    # random=np.random
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
           "struct": rng_state_of(root), "np": np.random.get_state(), "py": _pyrandom.getstate()}
    torch.save({"model": model, "optim_state": state, "param_groups": osd["param_groups"],
                "rng": rng}, tmp)
    tmp.rename(path / f"rank{rank:02d}.pt")
    dist.barrier()
    if rank == 0:
        (path / "cursor.json").write_text(json.dumps(cursor))
        (out / "ckpt" / "latest").write_text(path.name)
        olds = sorted(q for q in (out / "ckpt").glob("step_*") if q.is_dir())
        for q in olds[:-keep]:
            shutil.rmtree(q, ignore_errors=True)
    dist.barrier()


def load_ckpt(root, opt, out: Path):
    """-> cursor dict, or None (no checkpoint). Sharded and unsharded parameters are restored separately."""
    from torch.distributed.tensor import DTensor
    latest = out / "ckpt" / "latest"
    if not latest.exists():
        return None
    path = out / "ckpt" / latest.read_text().strip()
    ck = torch.load(path / f"rank{dist.get_rank():02d}.pt", map_location="cpu", weights_only=False)
    with torch.no_grad():
        for n, p in root.named_parameters():
            _local(p).copy_(ck["model"][n])
    params = [p for g in opt.param_groups for p in g["params"]]
    state = {}
    for i, st in ck["optim_state"].items():
        p = params[int(i)]
        if isinstance(p, DTensor):
            state[i] = {k: DTensor.from_local(v.to(p.to_local().device), p.device_mesh, p.placements,
                                              run_check=False, shape=p.shape, stride=p.stride())
                        if torch.is_tensor(v) and v.dim() > 0 else v for k, v in st.items()}
        else:
            state[i] = {k: v.to(p.device) if torch.is_tensor(v) and v.dim() > 0 else v
                        for k, v in st.items()}
    opt.load_state_dict({"state": state, "param_groups": ck["param_groups"]})
    if "rng" in ck:
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state(ck["rng"]["cuda"])
        _RNG_RESTORE["struct"] = ck["rng"]["struct"]
        import random as _pyrandom
        np.random.set_state(ck["rng"]["np"])
        _pyrandom.setstate(ck["rng"]["py"])
    return json.loads((path / "cursor.json").read_text())


_RNG_RESTORE: dict = {}
_STRUCT_REF: dict = {}


def rng_state_of(_root):
    st = _STRUCT_REF.get("stream")
    return st.rng.bit_generator.state if st is not None else None


# ---------------------------------------------------------------- optimizer and schedule

def param_groups(root, wd: float):
    """Embedding / decay / no-decay / adapter groups, plus a flow group: the structure head and
    the condition projection use a larger lr (1e-4, as in head-only training)."""
    g = {"embed": [], "decay": [], "no_decay": [], "adapter": [], "flow": []}
    for name, p in root.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("flow.", "cond_")):
            g["flow"].append(p)
        elif name.startswith("adapter."):
            g["adapter"].append(p)
        elif "embed_tokens" in name or "lm_head" in name:
            g["embed"].append(p)
        elif p.ndim < 2 or "norm" in name or name.endswith(("A_log", "dt_bias")):
            g["no_decay"].append(p)
        else:
            g["decay"].append(p)
    out = [{"name": "embed", "params": g["embed"], "weight_decay": 0.0},
           {"name": "decay", "params": g["decay"], "weight_decay": wd},
           {"name": "no_decay", "params": g["no_decay"], "weight_decay": 0.0}]
    out += [{"name": k, "params": g[k], "weight_decay": 0.0} for k in ("adapter", "flow") if g[k]]
    return out


def lr_at(a, step: int, tokens: float) -> float:
    """Trunk lr: after warmup, cosine annealing over token progress."""
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    # Ablation groups can be aligned to the main run by step: the number of counted tokens per
    # step differs between groups (without ESM2 injection the residue positions are not masked,
    # about 57% more per step), and a cosine over each group's own tokens would give different
    # learning rates at the same step
    progress = step * a.lr_tokens_per_step if a.lr_tokens_per_step else tokens
    frac = min(1.0, max(0.0, progress / max(a.total_tokens, 1)))
    import math
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * frac)))


# ---------------------------------------------------------------- training

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="starting LM: Qwen3.5-9B with the protein vocabulary and the <structure> "
                         "special token (output of data/add_structure_token.py)")
    ap.add_argument("--lm-data", type=Path, required=True,
                    help="pre-tokenized mixture (protein tasks + general text); reads <dir>/train/epoch*.jsonl shards")
    ap.add_argument("--struct-tokens", type=Path, required=True, help="SimpleFold tokenized dir")
    ap.add_argument("--struct-targets", type=Path, required=True)
    ap.add_argument("--flow-ckpt", required=True,
                    help="initial weights of the flow head (e.g. simplefold_100M.ckpt); pass 'none' for random init")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--total-tokens", type=float, default=2e9)
    ap.add_argument("--micro-tokens", type=int, default=16384)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--struct-per-step", type=int, default=2, help="proteins per micro-batch (B_p)")
    ap.add_argument("--struct-copies", type=int, default=8, help="timestep copies per protein (B_c)")
    ap.add_argument("--max-res", type=int, default=256)
    ap.add_argument("--lambda-flow", type=float, default=1.0,
                    help="lambda_f, fixed for the whole run (28.9 for the released model)")
    ap.add_argument("--phase", default="finetune", choices=["pretrain", "finetune"])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--flow-lr", type=float, default=1e-4)
    ap.add_argument("--embed-lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--no-esm", action="store_true", help="ablation: remove ESM2")
    ap.add_argument("--cond-layers", default="0,4,8,12,16,20,24,28,32")
    ap.add_argument("--inject-mode", default="add", choices=["add", "replace"],
                    help="add = ESM features added to the amino-acid token embeddings (used for Calpha); "
                         "replace = they replace the token embeddings (protein as a separate modality)")
    ap.add_argument("--flow-model", default="simplefold_100M",
                    help="size of the structure head")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr-tokens-per-step", type=float, default=0,
                    help=">0: learning-rate progress is step x this value instead of the actual token "
                         "count (aligns ablation groups with the main run step by step)")
    ap.add_argument("--stop-at-tokens", type=float, default=0,
                    help="save and stop after this many tokens, while the learning rate still follows "
                         "the full --total-tokens schedule. For an intermediate check: afterwards "
                         "resume from the checkpoint and the curve continues without a break")
    ap.add_argument("--probe-every", type=int, default=250,
                    help="run the generation probe every this many steps; 0 disables it")
    ap.add_argument("--probe-eval", type=Path, default=None,
                    help="evaluation parquet for the probe (columns id, sequence, coords, length, "
                         "release_date); required unless --probe-every 0")
    ap.add_argument("--ccd", type=Path, default=None,
                    help="SimpleFold's ccd.pkl; required unless --probe-every 0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.probe_every and (a.probe_eval is None or a.ccd is None):
        ap.error("--probe-eval and --ccd are required unless --probe-every 0")

    from model.flow import LinearPath
    from processor.protein_processor import ProteinDataProcessor
    from transformers import AutoTokenizer

    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda")
    torch.manual_seed(a.seed + rank)
    a.out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (a.out / "args.json").write_text(json.dumps(vars(a), indent=1, default=str))

    tok = AutoTokenizer.from_pretrained(a.model)
    layers = tuple(int(x) for x in a.cond_layers.split(","))
    flow_ckpt = None if str(a.flow_ckpt).lower() == "none" else Path(a.flow_ckpt)
    root = build_dual(a.model, flow_ckpt, layers, esm=not a.no_esm,
                      inject_mode=a.inject_mode, flow_model=a.flow_model)
    root = root.to(device)

    encoder = None if a.no_esm else EsmEncoder(str(device))
    from esm_adapter import Injector
    injector = None if a.no_esm else Injector(tok, encoder)

    processor = ProteinDataProcessor(device=device, scale=16.0, ref_scale=5.0,
                                     multiplicity=a.struct_copies, inference_multiplicity=1, backend="torch")
    path_ = LinearPath()
    struct = StructStream(a.struct_tokens, a.struct_targets, tok, a.max_res, a.seed, rank, world)
    _STRUCT_REF["stream"] = struct
    probe = StructProbe(a.probe_eval, a.out, tok, rank, world, device, a.ccd) if a.probe_every else None
    shards = mt.Shards(sorted(a.lm_data.glob("train/epoch*.jsonl")))
    stream = mt.micro_batches(shards, a.micro_tokens, a.max_len, world, a.seed)

    opt = torch.optim.AdamW(param_groups(root, a.weight_decay), lr=a.lr)
    metrics = (a.out / "metrics.jsonl").open("a") if rank == 0 else None

    step, tokens, t0 = 0, 0.0, time.time()
    cursor = load_ckpt(root, opt, a.out)
    if cursor is not None:
        step, tokens = cursor["step"], cursor["tokens"]
        struct.cursor = cursor["struct_cursor"]
        if _RNG_RESTORE.get("struct") is not None:
            struct.rng.bit_generator.state = _RNG_RESTORE["struct"]
        # Fast-forward the LM stream by the number of global micro-batches already consumed:
        # micro_batches only walks the index and reads no tokens, so this is quick
        for _ in range(cursor["lm_mbs"]):
            next(stream)
        if rank == 0:
            print(json.dumps({"resume": cursor}), flush=True)
    lm_mbs = cursor["lm_mbs"] if cursor else 0
    if probe is not None and cursor is None:
        # Probe once at step 0, before any training: "the model itself before training" is the
        # best floor and directly answers "how far did this training move it". Not repeated on
        # resume.
        root.eval()
        res = probe.run(root, injector)
        root.train()
        mt.reshard_all(root)
        if rank == 0:
            metrics.write(json.dumps({"step": 0, "tokens": 0, **res}) + "\n")
            metrics.flush()
            print(json.dumps({"step": 0, **res}), flush=True)
    acc = {"lm": [], "flow": [], "lddt": [], "trig": [], "gn": [], "gn_body": [], "gn_flow": []}
    opt.zero_grad(set_to_none=True)
    def keep_going() -> bool:
        return (tokens < a.total_tokens and (not a.max_steps or step < a.max_steps)
                and (not a.stop_at_tokens or tokens < a.stop_at_tokens))

    while keep_going():
        n_local = torch.zeros((), device=device)
        for micro in range(a.grad_accum):
            # --- LM path
            # micro_batches yields (global index, sample list); take `world` of them per round and
            # use this rank's own. Every rank must draw all `world`, otherwise the data cursors of
            # the ranks drift apart.
            mbs = [next(stream)[1] for _ in range(world)]
            lm_mbs += world
            ids, labels, _tasks = mt.collate(shards, mbs[rank], tok.pad_token_id or 0)
            ids, labels = ids.to(device), labels.to(device)
            feats = mt.run_injector(injector, shards, mbs[rank], ids) if injector is not None else None
            if feats is not None:
                mt.mask_leaky_targets(labels, feats)
            lm_losses, _, _ = root("lm", ids, labels, feats)
            lm = lm_losses.mean() if lm_losses.numel() else lm_losses.sum()
            n_local += labels.ne(mt.IGNORE).sum()

            # --- structure path
            sb, seqs, enc = struct.batch(a.struct_per_step)
            mse, lddt, trig = flow_step(root, sb, enc, processor, path_, device, tok,
                                        a.struct_copies, a.phase, injector, struct.last_heads)
            # The trigger loss is on the same scale as the language loss (both per-token CE),
            # weight 1; the structure loss is weighted by the fixed lambda_f
            loss = lm + trig + a.lambda_flow * (mse + lddt)
            (loss / a.grad_accum).backward()
            acc["lm"].append(float(lm)); acc["flow"].append(float(mse)); acc["lddt"].append(float(lddt))
            acc["trig"].append(float(trig))

        # The token count must agree across ranks: both the stop test (--stop-at-tokens) and the
        # learning rate depend on it. The old "this rank's count x world" was a per-rank estimate;
        # at one stop point some ranks decided they had reached it and others had not -- the ones
        # that stopped went into the probe's all_reduce while the rest kept training, and after
        # waiting on each other for 30 minutes NCCL timed out.
        dist.all_reduce(n_local)
        tokens += float(n_local)

        # Clip separately: the trunk is FSDP-sharded into DTensors, the flow head and condition
        # projection are plain tensors (ignored_params); computing one norm over both raises
        # "mixed Tensor/DTensor".
        sharded, plain = [], []
        for n_, p_ in root.named_parameters():
            if p_.grad is None:
                continue
            (plain if n_.startswith(("flow.", "cond_")) else sharded).append(p_)
        # The two gradient norms are **logged separately** (logging only the merged value leaves
        # the question "should lambda_f be adjusted" without data)
        gn_body = float(torch.nn.utils.clip_grad_norm_(sharded, a.max_grad_norm)) if sharded else 0.0
        gn_flow = float(torch.nn.utils.clip_grad_norm_(plain, a.max_grad_norm)) if plain else 0.0
        acc["gn"].append(max(gn_body, gn_flow))
        acc["gn_body"].append(gn_body)
        acc["gn_flow"].append(gn_flow)
        # The trunk follows the cosine schedule; embed and flow head each keep a constant lr and
        # only share the warmup
        lr_body = lr_at(a, step, tokens)
        warm = min(1.0, (step + 1) / max(a.warmup_steps, 1))
        for g in opt.param_groups:
            g["lr"] = {"embed": a.embed_lr * warm, "flow": a.flow_lr * warm}.get(g["name"], lr_body)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if rank == 0 and step % a.log_every == 0:
            rec = {"step": step, "tokens": round(tokens), "sec_per_step": round((time.time() - t0) / step, 2),
                   **{k: round(sum(v) / len(v), 5) for k, v in acc.items() if v}}
            metrics.write(json.dumps(rec) + "\n"); metrics.flush()
            print(json.dumps(rec), flush=True)
            acc = {k: [] for k in acc}
        if probe is not None and step % a.probe_every == 0:
            root.eval()
            res = probe.run(root, injector)
            root.train()
            mt.reshard_all(root)   # forward only, no backward: the root module's parameters are still unsharded
            if rank == 0:
                metrics.write(json.dumps({"step": step, "tokens": round(tokens), **res}) + "\n")
                metrics.flush()
                print(json.dumps({"step": step, **res}), flush=True)
        if step % a.save_every == 0:
            # save_ckpt calls reshard_all first
            save_ckpt(root, opt, a.out, step, {"step": step, "tokens": tokens,
                                               "struct_cursor": struct.cursor, "lm_mbs": lm_mbs})
    # Wrap-up: if the last step is not on a save boundary, save once more and run the probe --
    # when stopping at --stop-at-tokens / --max-steps the checkpoint has to correspond to the step
    # training really stopped at, otherwise a resume would start from an earlier point
    if step and step % a.save_every != 0:
        if probe is not None:
            root.eval()
            res = probe.run(root, injector)
            root.train()
            mt.reshard_all(root)
            if rank == 0:
                metrics.write(json.dumps({"step": step, "tokens": round(tokens), **res}) + "\n")
                metrics.flush()
        save_ckpt(root, opt, a.out, step, {"step": step, "tokens": tokens,
                                           "struct_cursor": struct.cursor, "lm_mbs": lm_mbs})
    if rank == 0:
        print(json.dumps({"done": True, "step": step, "tokens": round(tokens)}), flush=True)
        if metrics:
            metrics.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
