"""Pre-tokenize mixture jsonl shards into flat token arrays for the trainer.

For every <split>/<name>.jsonl (+ .meta.parquet) writes <name>.tok.npy (uint32, all samples
concatenated) and <name>.idx.npy (int64 [n, 3]: offset, length, task id). Task ids follow
TASKS. No BOS is added; each sample is terminated with the tokenizer's eos token so the
model learns where a sample ends.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# Append only: the indices are written into the pre-tokenized idx.npy files. The data mixture used
# for Calpha contains inverse_fold, secondary, infill, contacts, fineweb_edu and mot_reasoning; the
# other names belong to earlier experiments and stay in the list so that task ids do not shift.
TASKS = ["fold", "inverse_fold", "secondary", "infill", "contacts", "fold_with_ss", "fineweb_edu", "mot_reasoning",
         "fold_c", "fold_cx"]
_tok = None


def _init(tok_path: str) -> None:
    global _tok
    _tok = AutoTokenizer.from_pretrained(tok_path)


def process(path: str) -> str:
    jsonl = Path(path)
    out_tok, out_idx = jsonl.with_suffix(".tok.npy"), jsonl.with_suffix(".idx.npy")
    if out_tok.exists() and out_idx.exists():
        return f"skip {jsonl.name}"
    meta = pq.read_table(jsonl.with_suffix(".meta.parquet")).to_pandas()
    texts = [json.loads(line)["messages"][0]["content"] for line in jsonl.open()]
    assert len(texts) == len(meta)
    eos = _tok.eos_token_id
    ids_all, idx = [], []
    off = 0
    for i in range(0, len(texts), 1024):
        for ids in _tok(texts[i:i + 1024], add_special_tokens=False)["input_ids"]:
            ids.append(eos)
            ids_all.append(np.asarray(ids, dtype=np.uint32))
    for ids, task in zip(ids_all, meta["task"]):
        idx.append((off, len(ids), TASKS.index(task)))
        off += len(ids)
    np.save(out_tok, np.concatenate(ids_all))
    np.save(out_idx, np.asarray(idx, dtype=np.int64))
    return f"{jsonl.name}: {len(idx)} samples, {off} tokens"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", type=Path, required=True,
                    help="output directory of build_mixture.py (holds train/, val/, tokenizer/)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    files = sorted(str(p) for p in args.mixture.glob("*/*.jsonl"))
    with mp.Pool(args.workers, initializer=_init, initargs=(str(args.mixture / "tokenizer"),)) as pool:
        for msg in pool.imap_unordered(process, files):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
