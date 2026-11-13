"""Stage-1 starting model: the vocabulary-extended Qwen plus one <structure> special token.

Why a dedicated token: the stage-1 model emits <structure> by itself in a conversation to trigger
the flow structure head. The extended vocabulary does not contain it, and the tokenizer would
split it into '<' 'structure' '>' -- "the model decides to fold" would then not be one clean
single-step decision, and inference could not stop on a single token.

The input and output vectors of the new token are both initialised with the mean of the existing
token vectors, so that at the start of training it is not biased towards any existing token.

  python data/add_structure_token.py --src <extend_vocab.py output> --out <stage-1 base model>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STRUCT_TOKEN = "<structure>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="output directory of extend_vocab.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.src)
    n_before = len(tok)
    # add_tokens(special_tokens=True): the transformers development build used here does not
    # support the replace_additional_special_tokens argument of add_special_tokens
    added = tok.add_tokens([STRUCT_TOKEN], special_tokens=True)
    if added != 1:
        raise SystemExit(f"expected 1 new token, got {added} (does {STRUCT_TOKEN} already exist?)")
    new_id = tok.convert_tokens_to_ids(STRUCT_TOKEN)

    model = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.bfloat16)
    rows_before = model.get_input_embeddings().weight.shape[0]
    # The embedding matrix of the extended model is already padded (252288 rows > 252209 tokens).
    # If the new id falls into the padding rows the matrix shape is **left unchanged** -- a resize
    # would trim the padding rows and break the alignment; only this one row is reset.
    if new_id >= rows_before:
        model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)
    with torch.no_grad():
        for emb in {id(e): e for e in (model.get_input_embeddings().weight,
                                       model.get_output_embeddings().weight)}.values():
            # mean over the rows of real tokens only (the first n_before rows), excluding the
            # padding rows and the new row itself
            emb[new_id] = emb[:n_before].float().mean(0).to(emb.dtype)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    # Self-check: after reloading, <structure> is a single token whose id is the one assigned above
    t2 = AutoTokenizer.from_pretrained(out)
    ids = t2(STRUCT_TOKEN, add_special_tokens=False).input_ids
    info = {"src": a.src, "vocab_before": n_before, "vocab_after": len(t2),
            "struct_token_id": new_id, "tokenized_as": ids,
            "embedding_rows_before": rows_before,
            "embedding_rows_after": model.get_input_embeddings().weight.shape[0]}
    (out / "base_rc.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    if ids != [new_id]:
        raise SystemExit(f"{STRUCT_TOKEN} is still split: {ids}")


if __name__ == "__main__":
    main()
