"""Condition providers: swap what feeds SimpleFold's residue trunk.

SimpleFold conditions its trunk on one tensor:

    esm_s = (esm_s_combine.softmax(0) @ feats['esm_s']).squeeze(2)   # (B, L, nLayers, C) -> (B, L, C)
    esm_emb = esm_s_proj(esm_s)                                      # C -> hidden_size
    latent = esm_cat_proj(cat([latent, esm_emb], -1))

so a condition is fully described by a `(B, L, nLayers, C)` tensor. Every provider here returns
that shape, and `retarget_condition` resizes the two modules that depend on `(nLayers, C)` while
the rest of the pretrained network is kept.

Providers:
  onehot       (B, L, 1, 21)        amino-acid identity only -- the floor
  esm2         (B, L, 37, 2560)     what SimpleFold was trained with (head-only control)
  routec       (B, L, 1, C)         the stage-1 model's own condition projection output
  routec_raw   (B, L, 9, 4096)      raw hidden states of the stage-1 language model at the <seq>
                                    residue tokens, layers 0/4/.../32 -- what Calpha stage 2 uses

("routec" is the internal name of the stage-1 dual-head model; the option values are kept so that
the archived run configurations still apply.)

The language-model providers only ever run the *prompt*, never ground-truth structure, so no
structure can leak into the condition. The frozen ESM2 features are injected through the
checkpoint's own adapter first, exactly as in stage-1 training.

Two further providers of the research code (hidden states of a plain or LM-only-trained Qwen, and
ESM features concatenated into Qwen's input through a trainable projector) are not used by Calpha
and were removed from this release.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1"))

AMINO = "ACDEFGHIKLMNPQRSTVWY"


def aa_tok(a: str) -> str:
    return f"<aa{a if a in AMINO else 'X'}>"


class OneHotProvider:
    """Amino-acid identity only: the floor that says how much any language model adds."""

    n_layers, dim = 1, 21

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.index = {a: i for i, a in enumerate(AMINO)}

    @torch.no_grad()
    def __call__(self, seqs: list[str], max_len: int) -> torch.Tensor:
        out = torch.zeros(len(seqs), max_len, 1, self.dim, device=self.device)
        for b, s in enumerate(seqs):
            for i, a in enumerate(s[:max_len]):
                out[b, i, 0, self.index.get(a, 20)] = 1.0
        return out


class Esm2Provider:
    """SimpleFold's own condition: all 37 ESM2-3B layers."""

    n_layers, dim = 37, 2560

    def __init__(self, device: str = "cuda") -> None:
        import esm as fair_esm

        self.device = device
        model, self.alphabet = fair_esm.pretrained.esm2_t36_3B_UR50D()
        self.model = model.to(device).eval().requires_grad_(False)
        self.batch_converter = self.alphabet.get_batch_converter()

    @torch.no_grad()
    def __call__(self, seqs: list[str], max_len: int) -> torch.Tensor:
        _, _, toks = self.batch_converter([(str(i), s) for i, s in enumerate(seqs)])
        out = self.model(toks.to(self.device), repr_layers=list(range(self.n_layers)))
        rep = torch.stack([out["representations"][l] for l in range(self.n_layers)], 2)
        rep = rep[:, 1:]  # drop BOS; EOS/padding is cut by the per-sequence copy below
        dst = torch.zeros(len(seqs), max_len, self.n_layers, self.dim, device=self.device)
        for b, s in enumerate(seqs):
            n = min(len(s), max_len)
            dst[b, :n] = rep[b, :n]
        return dst


class RouteCProvider:
    """The stage-1 dual-head model as the condition source: trunk (with ESM2 injection) ->
    hidden states at the <seq> residue positions -> stage-1 condition projection.

    ckpt is the output directory of stage1/export_stage1.py (merged.pt + meta.json).
    Exactly the training path: chat-format prompt, ESM2 injected by the Injector,
    DualHead._condition. Returns (B, L, 1, C), the shape of batch["esm_s"] during stage-1 training.
    """

    n_layers = 1

    def __init__(self, ckpt: str, device: str = "cuda") -> None:
        from transformers import AutoTokenizer

        import train_stage1 as rc
        from esm_adapter import EsmEncoder, Injector

        d = Path(ckpt)
        meta = json.loads((d / "meta.json").read_text())
        layers = tuple(int(x) for x in str(meta["cond_layers"]).split(","))
        self.root = rc.build_dual(meta["model"], None, layers, esm=not meta["no_esm"], shard=False)
        # The checkpoint stores parameters only (named_parameters), not buffers. Buffers
        # (positional-encoding frequencies, RoPE inv_freq, ...) are generated deterministically
        # from the config at construction, so the freshly built model already has them. Missing
        # keys are allowed here, but every missing key must be a buffer and none a parameter --
        # otherwise weights really are missing.
        res = self.root.load_state_dict(torch.load(d / "merged.pt", map_location="cpu",
                                                   weights_only=False), strict=False)
        params = {n for n, _ in self.root.named_parameters()}
        lost = [k for k in res.missing_keys if k in params]
        if lost or res.unexpected_keys:
            raise RuntimeError(f"checkpoint does not match the model: missing parameters {lost[:5]}, "
                               f"unexpected {res.unexpected_keys[:5]}")
        self.root = self.root.to(device).eval().requires_grad_(False)
        self.dim = self.root.cond_proj[1].out_features
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(meta["model"])
        self.inj = None if meta["no_esm"] else Injector(self.tok, EsmEncoder(device))
        self.open = self.tok.convert_tokens_to_ids("<seq>")
        self.close = self.tok.convert_tokens_to_ids("</seq>")
        self.rc = rc

    def __call__(self, seqs: list[str], max_len: int) -> torch.Tensor:
        # When the language model is unfrozen (--train-llm) the gradient **must be kept**. The
        # whole function used to run under no_grad, so unfreezing the trunk did nothing -- the
        # folding loss never reached the language model.
        ctx = torch.enable_grad() if getattr(self, "_train_lm", False) else torch.no_grad()
        with ctx:
            return self._call(seqs, max_len)

    def _call(self, seqs: list[str], max_len: int) -> torch.Tensor:
        user, assistant = self.rc.CHAT_TEMPLATES[0]       # fixed template for evaluation, not random
        out = torch.zeros(len(seqs), max_len, 1, self.dim, device=self.device)
        for b, s in enumerate(seqs):
            aa = "".join(aa_tok(a) for a in s)
            prompt = (f"<|im_start|>user\n{user.format(seq=f'<seq>{aa}</seq>')}<|im_end|>\n"
                      f"<|im_start|>assistant\n{assistant.format(tag=self.rc.STRUCT_TOKEN)}<|im_end|>")
            ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            pos = self.rc.residue_positions(ids[0].tolist(), self.open, self.close)
            feats = self.inj(ids, ["fold"]) if self.inj is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cond = self.root._condition(ids, feats, [pos], max_len)    # (1, L, C)
            out[b, :, 0] = cond[0].float()
        return out


    def unfreeze_lm(self, shard: bool = True):
        """Unfreeze the Qwen trunk (--train-llm; not used by Calpha, where the trunk stays frozen
        in stage 2).

        Memory: Qwen goes through FSDP2 sharding (9B x 16 B / 8 = about 18 GB per GPU); the
        structure head keeps a full copy on every GPU -- the bmm inside the head fails outright on
        "sharded tensor mixed with plain tensor" (the same pitfall as in train_stage1.build_dual).

        **Only the transformer layers are sharded, not the root module**: wrapping the root would
        also shard the final norm and lm_head outside the layers, which was observed to fail in
        `output * (1.0 + self.weight.float())` (the norm weight became a DTensor while the
        activation was a plain tensor). Parameters outside the layers keep full copies and their
        gradients are all-reduced by hand in train_stage2.

        ESM2 and the adapter always stay frozen: they are external encoders and not a variable of
        that experiment.
        """
        lm = self.root.lm
        lm.requires_grad_(True)
        lm.train()
        # the adapter / ESM2 are not variables; the injection path stays frozen
        if getattr(self.root, "adapter", None) is not None:
            self.root.adapter.requires_grad_(False)
        self._train_lm = True
        if shard:
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            for layer in lm.model.layers:
                fully_shard(layer, mp_policy=mp)
        return [q for q in lm.parameters() if q.requires_grad]

    def unsharded_lm_params(self):
        """Parameters outside the layers (embedding / final norm / lm_head): FSDP does not manage
        them, so they need a manual all_reduce."""
        sharded = {id(q) for layer in self.root.lm.model.layers for q in layer.parameters()}
        return [q for q in self.root.lm.parameters()
                if q.requires_grad and id(q) not in sharded]


class RouteCRawProvider(RouteCProvider):
    """The stage-1 trunk, but returning the raw hidden states **before** cond_proj.

    RouteCProvider returns the output of the stage-1 model's own condition projection
    (B, L, 1, C), and that projection was trained for the 100M head: its width is set by the 100M
    head and its weights are only optimal for it. Feeding it to the 3B head would make the new
    head look at the trunk through an old bottleneck.

    This class returns (B, L, 9, 4096) instead -- the same layout as Esm2Provider -- and the new
    head's own esm_s_combine / esm_s_proj learn the layer weighting and the projection. The layer
    indices come from cond_layers in the export's meta.json, exactly those used in stage-1
    training. Loading, chat template and ESM2 injection are all inherited.
    """

    def __init__(self, ckpt: str, device: str = "cuda") -> None:
        super().__init__(ckpt, device)
        self.layers = self.root.cond_layers
        self.n_layers = len(self.layers)
        self.dim = self.root.lm.get_input_embeddings().weight.shape[1]

    def __call__(self, seqs: list[str], max_len: int) -> torch.Tensor:
        # When Qwen is unfrozen the gradient **must be kept**, otherwise the trunk receives no
        # signal at all and unfreezing does nothing -- a "runs fine but the variable never took
        # effect" failure, which is the hardest kind to notice.
        ctx = torch.enable_grad() if getattr(self, "_train_lm", False) else torch.no_grad()
        with ctx:
            return self._forward(seqs, max_len)

    def _forward(self, seqs: list[str], max_len: int) -> torch.Tensor:
        user, assistant = self.rc.CHAT_TEMPLATES[0]       # same fixed template as RouteCProvider
        out = torch.zeros(len(seqs), max_len, self.n_layers, self.dim, device=self.device)
        for b, s in enumerate(seqs):
            aa = "".join(aa_tok(a) for a in s)
            prompt = (f"<|im_start|>user\n{user.format(seq=f'<seq>{aa}</seq>')}<|im_end|>\n"
                      f"<|im_start|>assistant\n{assistant.format(tag=self.rc.STRUCT_TOKEN)}<|im_end|>")
            ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            pos = self.rc.residue_positions(ids[0].tolist(), self.open, self.close)
            feats = self.inj(ids, ["fold"]) if self.inj is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, hs = self.root.trunk(ids, feats)
            idx = torch.as_tensor(pos, device=self.device)
            rep = torch.stack([hs[l][0, idx] for l in self.layers], 1)   # (L, n_layers, D)
            n = min(len(pos), max_len)
            out[b, :n] = rep[:n].float()
        return out


def make_provider(kind: str, ckpt: str | None = None, device: str = "cuda", bf16: bool = False):
    """With bf16=True the frozen condition models are cast to bf16 weights.

    They only run forward, and that forward is under bf16 autocast anyway, so the numerical
    difference is small (measured max|diff| 2e-4 against fp32 weights, itself bf16-level noise);
    the memory saved (Qwen 36 -> 18 GB, ESM2 11 -> 5.5 GB) decides whether training the 3B head
    fits at all. **Training and evaluation must use the same setting**, otherwise the condition
    distributions do not match.
    """
    if kind == "onehot":
        return OneHotProvider(device)
    if kind == "esm2":
        p = Esm2Provider(device)
    elif kind == "routec":
        if ckpt is None:
            raise ValueError("routec needs --cond-ckpt <export_stage1.py output directory>")
        p = RouteCProvider(ckpt, device)
    elif kind == "routec_raw":
        if ckpt is None:
            raise ValueError("routec_raw needs --cond-ckpt <export_stage1.py output directory>")
        p = RouteCRawProvider(ckpt, device)
    else:
        raise ValueError(f"unknown condition {kind}")

    if bf16:
        for attr in ("model", "root", "enc", "adapter", "inj"):
            m = getattr(p, attr, None)
            if m is None:
                continue
            inner = getattr(m, "model", m)          # Injector / EsmEncoder wrap the real model in .model
            if isinstance(inner, torch.nn.Module):
                inner.to(torch.bfloat16)
    return p


def retarget_condition(model, n_layers: int, dim: int, reinit: bool = False) -> None:
    """Resize the two condition-dependent modules in place, keeping the rest of the weights.

    `esm_s_combine` is a softmax over layers and `esm_s_proj` reads the combined vector, so both
    have to match the new (n_layers, dim). Everything downstream -- atom encoder, residue trunk,
    atom decoder -- is untouched, which is the point: only the condition changes between groups.
    """
    import torch.nn as nn

    from model.torch.layers import ConditionEmbedder

    dev = model.esm_s_combine.device
    if reinit or model.esm_s_combine.numel() != n_layers:
        model.esm_s_combine = nn.Parameter(torch.zeros(n_layers, device=dev))
    old = model.esm_s_proj
    if reinit or old.proj[0].in_features != dim:
        new = ConditionEmbedder(input_dim=dim, hidden_size=old.proj[0].out_features,
                                dropout_prob=old.dropout_prob).to(dev)
        model.esm_s_proj = new
