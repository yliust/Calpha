"""The language side of Calpha: Qwen3.5-9B (fine-tuned in stage 1) with ESM2 features added to
the embeddings of the amino-acid tokens.

Two things are read from it:
* `generate`  - its text answer; a request for a structure is answered with `<structure>`;
* `residue_states` - hidden states of nine layers at the residue positions, the only
  information the structure head receives.

The model is causal: the hidden states at the residue positions are a function of the tokens up
to and including the sequence. Numerically they are not always bit-identical when more tokens
follow (for some prompt lengths we measured a difference of about 2% in relative norm, for
others none), so the benchmark path always feeds the full turn, answer included, as in training.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .constants import AMINO, COND_LAYERS, SEQ_CLOSE, SEQ_OPEN
from .esm_adapter import Adapter, EsmEncoder


class CalphaLM:
    def __init__(self, llm_dir: str | Path, adapter_path: str | Path, device: str = "cuda",
                 layers: tuple[int, ...] = COND_LAYERS):
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device, self.layers = device, tuple(layers)
        self.tok = AutoTokenizer.from_pretrained(str(llm_dir))
        self.lm = AutoModelForCausalLM.from_pretrained(str(llm_dir), dtype=torch.bfloat16)
        # `.to(bfloat16)` also casts the rotary-embedding frequency buffers, which
        # `from_pretrained(dtype=...)` leaves in float32. The front-end was trained with the
        # bf16 buffers; with float32 ones the hidden states move by about 2% in relative norm.
        self.lm = self.lm.to(device).to(torch.bfloat16).eval().requires_grad_(False)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.hidden = self.lm.get_input_embeddings().weight.shape[1]

        self.adapter = Adapter(self.hidden)
        self.adapter.load_state_dict(load_file(str(adapter_path)), strict=True)
        self.adapter = self.adapter.to(device).to(torch.bfloat16).eval().requires_grad_(False)
        self.esm = EsmEncoder(device)

        ids = self.tok.convert_tokens_to_ids
        self.seq_open, self.seq_close = ids(SEQ_OPEN), ids(SEQ_CLOSE)
        self.aa = {ids(f"<aa{a}>"): a for a in AMINO}
        self.aa[ids("<aaX>")] = "X"

    # ---- tokens -> embeddings with ESM2 features

    def _segments(self, ids: list[int]) -> list[tuple[str, list[int]]]:
        """Every complete <seq>...</seq> span: (sequence, token positions of its residues)."""
        out, start = [], 0
        while self.seq_open in ids[start:]:
            a = ids.index(self.seq_open, start) + 1
            if self.seq_close not in ids[a:]:
                break
            b = ids.index(self.seq_close, a)
            if all(t in self.aa for t in ids[a:b]) and b > a:
                out.append(("".join(self.aa[t] for t in ids[a:b]), list(range(a, b))))
            start = b + 1
        return out

    def embed(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, list[int]]]]:
        """-> (token ids (1, T), input embeddings (1, T, H), protein segments)."""
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        emb = self.lm.get_input_embeddings()(ids)
        segs = self._segments(ids[0].tolist())
        if segs:
            flat = emb.reshape(-1, emb.shape[-1])
            add = torch.zeros_like(flat)
            for seq, pos in segs:
                feat = self.adapter(self.esm(seq).to(emb.dtype))
                add.index_add_(0, torch.as_tensor(pos, device=self.device), feat.to(flat.dtype))
            emb = (flat + add).reshape(emb.shape)
        return ids, emb, segs

    # ---- the two read-outs

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Greedy continuation of a rendered chat prompt, up to the end of the turn.
        Special tokens such as `<structure>` are kept in the returned text."""
        ids, emb, _ = self.embed(prompt)
        self.lm.config.use_cache = True
        out = self.lm.generate(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                               max_new_tokens=max_new_tokens, do_sample=False,
                               eos_token_id=self.im_end, pad_token_id=self.im_end)
        return self.tok.decode(out[0], skip_special_tokens=False).split("<|im_end|>")[0]

    @torch.no_grad()
    def residue_states(self, prompt: str) -> torch.Tensor:
        """Hidden states at the residue positions of the last protein in the prompt:
        (L, n_layers, H), float32."""
        # The embedding, the adapter and the trunk all run under bf16 autocast, as they did when
        # the front-end was trained (autocast changes how LayerNorm is evaluated).
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device != "cpu"):
            _, emb, segs = self.embed(prompt)
            if not segs:
                raise ValueError("the prompt contains no <seq>...</seq> protein")
            pos = torch.as_tensor(segs[-1][1], device=self.device)
            hs = self.lm.model(inputs_embeds=emb, use_cache=False, output_hidden_states=True).hidden_states
        return torch.stack([hs[l][0, pos] for l in self.layers], 1).float()
