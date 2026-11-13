"""Create the base checkpoint: Qwen3.5-9B (text-only) with the protein vocabulary.

* Loads Qwen3_5ForCausalLM (vision tower dropped) in bf16.
* Resizes input embeddings and the (untied) lm_head to len(tokenizer) padded to 128.
* New rows: mean of existing rows + small Gaussian noise (std = 0.1 x per-dim std), so the
  4096 structure tokens start distinguishable; <aaX> rows copy the single-letter token
  "X" and <aaA>.. copy their single-letter token (input and output side).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab import AA_TOKENS, ALL_TOKENS, AMINO  # noqa: E402

# Directory of the original Qwen3.5-9B weights and tokenizer (same variable as in build_mixture.py).
# Not needed with --from-extended.
QWEN = os.environ.get("QWEN_BASE")


def init_rows(weight: torch.Tensor, n_old: int, gen: torch.Generator) -> None:
    old = weight[:n_old].float()
    mean, std = old.mean(0), old.std(0)
    n_new = weight.shape[0] - n_old
    noise = torch.randn(n_new, weight.shape[1], generator=gen) * std * 0.1
    weight[n_old:] = (mean + noise).to(weight.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, required=True,
                    help="tokenizer directory = Qwen tokenizer + vocab.ALL_TOKENS "
                         "(build_mixture.py writes one to <out>/tokenizer)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from-extended", type=Path, default=None,
                    help="rebuild from a checkpoint whose vocabulary **has already been extended** instead of "
                         "from the original Qwen. The two are equivalent: extending only touches rows "
                         ">= base_vocab, and init_rows below re-initialises all of those rows; the first "
                         "base_vocab rows are untouched, and mean/std and the random seed are the same. "
                         "For machines where the original Qwen weights are not available. Needs --base-vocab")
    ap.add_argument("--base-vocab", type=int, default=None,
                    help="vocabulary size of the original Qwen (base_vocab in the extend_vocab.json of the "
                         "extended checkpoint)")
    args = ap.parse_args()
    if not args.from_extended and not QWEN:
        raise SystemExit("set QWEN_BASE to the original Qwen3.5-9B directory (or use --from-extended)")

    src_model = str(args.from_extended) if args.from_extended else QWEN
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.from_extended:
        if args.base_vocab is None:
            raise SystemExit("--from-extended needs --base-vocab")
        base_tok, n_base_vocab = AutoTokenizer.from_pretrained(src_model), args.base_vocab
    else:
        base_tok = AutoTokenizer.from_pretrained(QWEN)
        n_base_vocab = len(base_tok)
    assert tok.convert_tokens_to_ids(ALL_TOKENS[0]) == n_base_vocab, "protein tokens must follow the base vocab"
    model = AutoModelForCausalLM.from_pretrained(src_model, dtype=torch.bfloat16)
    n_rows_old = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok), pad_to_multiple_of=128, mean_resizing=False)
    emb, head = model.get_input_embeddings().weight.data, model.get_output_embeddings().weight.data
    gen = torch.Generator().manual_seed(0)
    n_base = n_base_vocab
    for w in (emb, head):
        init_rows(w, n_base, gen)  # rows >= n_base: protein tokens + padding rows
    copied = {}
    for tok_str, letter in zip(AA_TOKENS, AMINO + "X"):
        src = base_tok.convert_tokens_to_ids(letter)
        dst = tok.convert_tokens_to_ids(tok_str)
        emb[dst], head[dst] = emb[src].clone(), head[src].clone()
        copied[tok_str] = letter
    model.config.get_text_config().vocab_size = emb.shape[0]
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(args.out)
    meta = {"base": src_model, "rebuilt_from_extended": bool(args.from_extended), "base_vocab": n_base, "base_embedding_rows": n_rows_old, "tokenizer_len": len(tok),
            "embedding_rows": emb.shape[0], "added_tokens": len(ALL_TOKENS), "aa_copied_from_letters": copied,
            "init": "mean + N(0, (0.1*std)^2) per dim; aa tokens copy single-letter rows"}
    (args.out / "extend_vocab.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "aa_copied_from_letters"}))


if __name__ == "__main__":
    main()
