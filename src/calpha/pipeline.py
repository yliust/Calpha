"""End-to-end Calpha: a chat request in, text or a structure out."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .constants import (CANONICAL_TEMPLATE, COND_LAYERS, HF_REPO, NUM_STEPS, SEED, STRUCT_TOKEN,
                        TAU, seq_tokens, user_turn)

WEIGHT_FILES = ["calpha_config.json", "esm_adapter.safetensors", "frontend.safetensors", "llm/*"]


def resolve_weights(weights: str | Path | None) -> Path:
    """A local directory, or a Hugging Face repo id that is downloaded to the local cache."""
    if weights is not None and Path(weights).is_dir():
        return Path(weights)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(weights or HF_REPO), allow_patterns=WEIGHT_FILES))


@dataclass
class Answer:
    text: str                                   # what the language model said
    triggered: bool                             # did it emit <structure>?
    structures: list[Path] = field(default_factory=list)   # mmCIF files, one per sample


class Calpha:
    """
    >>> model = Calpha.from_pretrained()
    >>> out = model.fold("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", out_dir="pred")
    >>> out.structures
    """

    def __init__(self, weights: Path, simplefold_dir: str | Path, device: str = "cuda",
                 num_steps: int = NUM_STEPS, tau: float = TAU, nsample: int = 1, seed: int = SEED):
        import lightning.pytorch as pl

        from .language_model import CalphaLM
        from .structure_head import StructureHead

        cfg = json.loads((weights / "calpha_config.json").read_text())
        layers = tuple(cfg.get("cond_layers", COND_LAYERS))
        pl.seed_everything(seed, workers=True)
        self.lm = CalphaLM(weights / "llm", weights / "esm_adapter.safetensors", device, layers)
        self.head = StructureHead(weights / "frontend.safetensors", simplefold_dir,
                                  n_layers=len(layers), dim=self.lm.hidden,
                                  num_steps=num_steps, tau=tau, nsample=nsample, seed=seed)
        # Seed again once everything is built, so that the noise drawn for the first target does
        # not depend on how many random numbers model construction happened to consume.
        pl.seed_everything(seed, workers=True)

    @classmethod
    def from_pretrained(cls, weights: str | Path | None = None,
                        simplefold_dir: str | Path = "~/.cache/calpha/simplefold", **kw) -> "Calpha":
        sf = Path(simplefold_dir).expanduser()
        sf.mkdir(parents=True, exist_ok=True)
        return cls(resolve_weights(weights), sf, **kw)

    # ---- prompts

    @staticmethod
    def canonical_prompt(seq: str, with_answer: bool = True) -> str:
        """The turn under which the front-end was trained and every reported number was
        produced: the canonical request followed by the assistant's answer ending in
        `<structure>`. With `with_answer=False` only the request is rendered, for generation."""
        prompt = user_turn(CANONICAL_TEMPLATE[0].format(seq=seq_tokens(seq)))
        if with_answer:
            prompt += CANONICAL_TEMPLATE[1].format(tag=STRUCT_TOKEN) + "<|im_end|>"
        return prompt

    # ---- the two entry points

    def fold(self, seq: str, out_dir: str | Path, name: str = "target", work: Path | None = None) -> Answer:
        """Benchmark path: fold `seq` with the canonical turn. The answer `<structure>` is
        supplied, not generated, so the head always runs. Use `chat` to let the model decide
        whether to fold."""
        return self._fold({name: seq}, lambda s: self.lm.residue_states(self.canonical_prompt(s)),
                          Path(out_dir), work)

    def chat(self, request: str, seq: str | None = None, out_dir: str | Path = "pred",
             name: str = "target", max_new_tokens: int = 64) -> Answer:
        """Full path: the model reads a request in plain language (write `{seq}` where the
        protein goes) and answers. If the answer contains `<structure>`, the hidden states of
        this very prompt condition the head and a structure is generated.

        Accuracy has only been measured for the canonical request; see the README."""
        text = request.format(seq=seq_tokens(seq)) if seq is not None else request
        prompt = user_turn(text)
        reply = self.lm.generate(prompt, max_new_tokens=max_new_tokens)
        if STRUCT_TOKEN not in reply or seq is None:
            return Answer(reply, STRUCT_TOKEN in reply)
        turn = prompt + reply + "<|im_end|>"     # the request and the model's own answer
        ans = self._fold({name: seq}, lambda s: self.lm.residue_states(turn), Path(out_dir), None)
        return Answer(reply, True, ans.structures)

    def _fold(self, seqs: dict[str, str], condition, out_dir: Path, work: Path | None) -> Answer:
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(work) if work is not None else Path(tmp)
            files = self.head.prepare(seqs, w)
            paths = [p for f in files for p in self.head.sample(f, w, condition, out_dir)]
        return Answer(STRUCT_TOKEN, True, paths)
