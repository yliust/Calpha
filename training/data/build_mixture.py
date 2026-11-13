"""Build the language-side training mixture (protein multi-task + general text) as jsonl shards.

Protein samples (one chain may yield several tasks per epoch). `--task-mix default`:
  fold            seq -> struct                          p=1.00
  inverse_fold    struct -> seq                          p=0.50
  secondary       seq -> 3-state DSSP                    p=0.25
  infill          seq + struct with one hidden span -> span tokens   p=0.25
  contacts        seq -> long-range contacts (|i-j|>=12, CB<8A)       p=0.15
  fold_with_ss    seq -> DSSP -> struct                  p=0.10
`--task-mix vlm` (what Calpha stage 1 was trained on, with `--format chat --general-frac 0.25`)
drops the two "fold into discrete structure tokens" tasks and doubles the rest:
  inverse_fold 0.5, secondary 0.5, infill 0.5, contacts 0.3
Redundancy control: each epoch draws ceil(n ** CLUSTER_EXP) members from every 30%-identity
cluster (0.67: ~0.38B protein tokens/epoch, top-1% largest clusters ~17% of samples). Chains longer than MAX_RES residues get a random contiguous crop per sample.
General data (FineWeb-Edu text, Mixture-of-Thoughts science/math chats) is added to
make up GENERAL_FRAC of all tokens.

Inputs (not built by this script, see training/README.md): under --root a split index
(`split/index.parquet`: id, split, rep = 30%-identity cluster representative), per-chain feature
parquets (`features/parsed/<source>/*.parquet`: id, sequence, struct_tokens, dssp, contacts,
masked_frac) and `features/eval/foldbench334.parquet`; the FineWeb-Edu and Mixture-of-Thoughts
parquet files located by FINEWEB_GLOB / MOT_GLOB; the Qwen3.5-9B tokenizer at QWEN_BASE.

Output: <out>/train/epoch{e}-{k:04d}.jsonl ({"messages": [{"role": "assistant", "content": ...}]}),
<out>/train/epoch{e}-{k:04d}.meta.parquet (id, task, n_tokens), <out>/val/*.jsonl, <out>/tokenizer/,
<out>/summary.json.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from vocab import (ALL_TOKENS, N_STRUCT, interleave_ss, interleave_ss_struct, interleave_struct, interleave_struct_then_aa,
                   seq_str, ss_str, struct_str)

# QWEN_BASE: directory of the base tokenizer -- the original Qwen3.5-9B, or a checkpoint whose
# vocabulary has already been extended (see main()).
QWEN = os.environ.get("QWEN_BASE")
# MOT_GLOB: glob for the open-r1/Mixture-of-Thoughts parquet files with a `{}` placeholder for the
# subset name (science / math). Defaults to the Hugging Face hub cache in the home directory.
MOT = os.environ.get(
    "MOT_GLOB",
    str(Path.home() / ".cache/huggingface/hub/datasets--open-r1--Mixture-of-Thoughts/snapshots/*/{}/*.parquet"))
# FINEWEB_GLOB: glob for the FineWeb-Edu parquet shards (the sample/10BT subset was used). There is
# no default: with a wrong path nothing reports a missing file -- drawing general text just raises
# StopIteration, which does not look like a path problem. main() therefore checks it up front.
FINEWEB = os.environ.get("FINEWEB_GLOB", "")

MAX_RES = 1024
CLUSTER_EXP = 0.67
MAX_MASKED_FRAC = 0.5
TASK_P = {"fold": 1.0, "inverse_fold": 0.5, "secondary": 0.25, "infill": 0.25, "contacts": 0.15, "fold_with_ss": 0.10}
# --task-mix vlm (used for Calpha):
# drops the two "fold -> discrete structure tokens" tasks. They compete with the structure path's
# <structure> answer for the same question, which was the direct cause of an English trigger rate
# of only 5% at 2B tokens in an earlier round; together they made up 49.6% of the protein tokens.
# The remaining weights are raised to bring the amount of protein-task data from 51% back to
# about 80% of the original -- an ablation showed that the gradient flowing back from the
# structure head alone cannot drive the 9B trunk; the trunk learns protein representations from
# these dense per-token tasks.
TASK_P_VLM = {"inverse_fold": 0.5, "secondary": 0.5, "infill": 0.5, "contacts": 0.3}
CONTACT_MIN_SEP = 12
GENERAL_FRAC = 0.15   # override with --general-frac (0.25 was used for Calpha)
GENERAL_SPLIT = {"fineweb_edu": 2 / 3, "mot_reasoning": 1 / 3}
MAX_GENERAL_CHARS = 24000
SHARD = 50000


def render(task: str, f: dict, rng: np.random.Generator, fmt: str = "v1") -> str:
    if fmt == "interleave":
        return render_interleave(task, f, rng)
    if fmt == "chat":
        return render_chat(task, f, rng)
    seq, tok, ss = f["sequence"], f["struct_tokens"], f["dssp"]
    if task == "fold":
        return f"Task: fold\n<seq>{seq_str(seq)}</seq>\n<struct>{struct_str(tok)}</struct>"
    if task == "fold_with_ss":
        return f"Task: fold_with_ss\n<seq>{seq_str(seq)}</seq>\n<ss>{ss_str(ss)}</ss>\n<struct>{struct_str(tok)}</struct>"
    if task == "inverse_fold":
        return f"Task: inverse_fold\n<struct>{struct_str(tok)}</struct>\n<seq>{seq_str(seq)}</seq>"
    if task == "secondary":
        return f"Task: secondary_structure\n<seq>{seq_str(seq)}</seq>\n<ss>{ss_str(ss)}</ss>"
    if task == "infill":
        n = len(seq)
        span = int(rng.integers(max(5, n // 10), max(6, int(n * 0.3)) + 1))
        s = int(rng.integers(0, n - span + 1))
        return (f"Task: infill_structure\n<seq>{seq_str(seq)}</seq>\n"
                f"<struct>{struct_str(tok[:s])}<span>{struct_str(tok[s + span:])}</struct>\n"
                f"<span>{struct_str(tok[s:s + span])}</span>")
    if task == "contacts":
        pairs = f["contacts"].reshape(-1, 2)
        pairs = pairs[pairs[:, 1] - pairs[:, 0] >= CONTACT_MIN_SEP]
        cap = max(1, len(seq) // 4)
        if len(pairs) > cap:
            pairs = pairs[np.sort(rng.choice(len(pairs), cap, replace=False))]
        body = "; ".join(f"{i + 1} {j + 1}" for i, j in pairs)
        return f"Task: contacts\n<seq>{seq_str(seq)}</seq>\n<contacts>{body}</contacts>"
    raise ValueError(task)


def render_interleave(task: str, f: dict, rng: np.random.Generator) -> str:
    """v2 format: the full sequence stays in the prompt (global context), and every output
    residue symbol is preceded by its amino-acid token, e.g. <struct><aaM><s0812><aaK><s0033>..."""
    seq, tok, ss = f["sequence"], f["struct_tokens"], f["dssp"]
    if task == "fold":
        return f"Task: fold\n<seq>{seq_str(seq)}</seq>\n<struct>{interleave_struct(seq, tok)}</struct>"
    if task == "fold_with_ss":
        return f"Task: fold_with_ss\n<seq>{seq_str(seq)}</seq>\n<struct>{interleave_ss_struct(seq, ss, tok)}</struct>"
    if task == "inverse_fold":
        return f"Task: inverse_fold\n<struct>{struct_str(tok)}</struct>\n<seq>{interleave_struct_then_aa(seq, tok)}</seq>"
    if task == "secondary":
        return f"Task: secondary_structure\n<seq>{seq_str(seq)}</seq>\n<ss>{interleave_ss(seq, ss)}</ss>"
    if task == "infill":
        n = len(seq)
        span = int(rng.integers(max(5, n // 10), max(6, int(n * 0.3)) + 1))
        s = int(rng.integers(0, n - span + 1))
        return (f"Task: infill_structure\n<seq>{seq_str(seq)}</seq>\n"
                f"<struct>{interleave_struct(seq[:s], tok[:s])}<span>{interleave_struct(seq[s + span:], tok[s + span:])}</struct>\n"
                f"<span>{interleave_struct(seq[s:s + span], tok[s:s + span])}</span>")
    return render(task, f, rng, "v1")  # contacts: residue numbers are already explicit


# --format chat: wrap the tasks in the chat template. The bare format `Task: fold\n...` never
# occurs in a conversation, so "the model can still chat" does not imply "it can still do the
# task when asked in a conversation".
# English only. The wording deliberately **avoids** the structure-path questions in
# train_stage1.CHAT_TEMPLATES -- in an earlier round "Fold this sequence for me." and
# "Fold this sequence:" were nearly the same sentence; if the same question is answered with
# discrete tokens here and with <structure> on the structure path, the model cannot learn a
# deterministic trigger.
CHAT_ASK = {
    "fold": ["Predict the 3D structure of this protein.", "Write out the structure tokens.",
             "Emit the structure token sequence.", "Produce the discrete structure encoding."],
    "inverse_fold": ["Which sequence folds into this structure?",
                     "Recover the amino-acid sequence from this structure."],
    "secondary": ["What is the secondary structure here?",
                  "Label each residue with its secondary structure."],
    "infill": ["Fill in the missing span of this structure.",
               "Complete the gap marked in this structure."],
    "contacts": ["Which residues are in contact?", "List the long-range residue contacts."],
    "fold_with_ss": ["Predict the structure using the secondary structure.",
                     "Use the given secondary structure to write the structure tokens."],
}


def render_chat(task: str, f: dict, rng: np.random.Generator) -> str:
    """Split the output of render_interleave into a user question and an assistant answer.

    In the bare format prompt and answer are written back to back; here the text is cut at the
    task's first result tag: everything before it (including the sequence) goes to the user turn,
    the rest to the assistant turn."""
    body = render_interleave(task, f, rng)
    head, sep, tail = body.partition("\n")
    ask = CHAT_ASK.get(task, CHAT_ASK["fold"])[int(rng.integers(len(CHAT_ASK.get(task, CHAT_ASK["fold"]))))]
    # the first line of tail is the input (sequence or structure), the rest is the answer
    lines = tail.split("\n", 1)
    given, answer = (lines[0], lines[1] if len(lines) > 1 else "")
    return (f"<|im_start|>user\n{ask}\n{given}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>")


def crop(f: dict, rng: np.random.Generator) -> dict:
    n = len(f["sequence"])
    if n <= MAX_RES:
        return f
    s = int(rng.integers(0, n - MAX_RES + 1))
    e = s + MAX_RES
    pairs = f["contacts"].reshape(-1, 2)
    keep = (pairs[:, 0] >= s) & (pairs[:, 1] < e)
    return {"id": f["id"], "sequence": f["sequence"][s:e], "struct_tokens": f["struct_tokens"][s:e],
            "dssp": f["dssp"][s:e], "contacts": (pairs[keep] - s).reshape(-1)}


def feature_dirs(root: Path, features_root: Path | None) -> tuple[Path, Path]:
    """Return (root directory of the per-chain features, feature directory of the evaluation sets).

    Defaults to `<root>/features/parsed` and `<root>/features/eval` (ESM3 structure tokens).
    For another structure tokenizer, point --features-root at a parallel copy that holds
    `<source>/*.parquet` plus an `eval/` directory, row-aligned with the ESM3 version and
    differing only in struct_tokens.
    """
    if features_root is None:
        return root / "features/parsed", root / "features/eval"
    return features_root, features_root / "eval"


def load_features(parsed: Path, index: pd.DataFrame) -> dict[str, dict]:
    keep = dict(zip(index["id"], index["split"]))
    feats = {}
    # `eval/` and `logs/` are sibling directories, not the features of a source: skip them
    for p in sorted(glob.glob(str(parsed / "*/*.parquet"))):
        if Path(p).parent.name in ("eval", "logs"):
            continue
        for r in pq.read_table(p).to_pylist():
            if r["id"] in keep and r["masked_frac"] <= MAX_MASKED_FRAC:
                r["struct_tokens"] = np.asarray(r["struct_tokens"], dtype=np.int16)
                r["contacts"] = np.asarray(r["contacts"], dtype=np.int32)
                feats[r["id"]] = r
    return feats


def iter_general(kind: str, rng: np.random.Generator, tokenizer):
    if kind == "fineweb_edu":
        files = sorted(glob.glob(FINEWEB))
        for p in files:
            for batch in pq.ParquetFile(p).iter_batches(columns=["text", "id"], batch_size=4096):
                for text, id_ in zip(batch.column("text").to_pylist(), batch.column("id").to_pylist()):
                    yield f"fineweb:{id_}", text[:MAX_GENERAL_CHARS]
    else:
        # Safety net: if science + math (<= MAX_GENERAL_CHARS) runs out, reshuffle and repeat it. The
        # first pass is unchanged, so earlier mixtures are reproduced exactly; v3's 5 epochs did not
        # exhaust it. (fineweb-edu is not cycled: add shards under FINEWEB instead -- v3 needs two.)
        files = sorted(glob.glob(MOT.format("science"))) + sorted(glob.glob(MOT.format("math")))
        rng.shuffle(files)
        for rep in itertools.count():
            n = 0
            for p in files:
                t = pq.read_table(p, columns=["messages"])
                for k, msgs in enumerate(t.column("messages").to_pylist()):
                    text = tokenizer.apply_chat_template(msgs, tokenize=False)
                    if len(text) <= MAX_GENERAL_CHARS:
                        n += 1
                        yield (f"mot:{Path(p).stem}:{k}" + (f":r{rep}" if rep else "")), text
            if not n:
                raise RuntimeError("no Mixture-of-Thoughts samples found")
            print(json.dumps({"mot_pass_done": rep, "samples": n}), flush=True)
            rng.shuffle(files)


def count_tokens(tokenizer, texts: list[str]) -> list[int]:
    out = []
    for i in range(0, len(texts), 2048):
        enc = tokenizer(texts[i:i + 2048], add_special_tokens=False)["input_ids"]
        out.extend(len(x) for x in enc)
    return out


def write_shards(samples: list[tuple[str, str, str]], n_tokens: list[int], out: Path, prefix: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(samples), SHARD):
        part = samples[k:k + SHARD]
        name = f"{prefix}-{k // SHARD:04d}"
        with (out / f"{name}.jsonl").open("w") as fh:
            for _, _, text in part:
                fh.write(json.dumps({"messages": [{"role": "assistant", "content": text}]}, ensure_ascii=False) + "\n")
        pq.write_table(pa.table({"id": [s[0] for s in part], "task": [s[1] for s in part],
                                 "n_tokens": n_tokens[k:k + SHARD]}), out / f"{name}.meta.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="data root holding split/index.parquet and features/ (see the module docstring)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["v1", "interleave", "chat"], default="v1")
    ap.add_argument("--task-mix", choices=["default", "vlm"], default="default",
                    help="default = TASK_P; vlm = TASK_P_VLM (used for Calpha: folding-to-token tasks "
                         "dropped, the rest re-weighted)")
    ap.add_argument("--general-frac", type=float, default=None,
                    help="fraction of general-text tokens, overrides GENERAL_FRAC (0.25 for Calpha)")
    ap.add_argument("--general-samples-from", type=Path, default=None,
                    help="copy the **number** of general-text samples per epoch and kind from another "
                         "build's summary.json instead of stopping at the token target. "
                         "Why: the last step of every epoch is rng.permutation(len(samples)), and len "
                         "includes the general samples; after a tokenizer change the token counts of a "
                         "few general documents differ slightly, stopping at the token target then "
                         "yields a few samples more or fewer, and once len changes **the whole order "
                         "changes** -- same protein samples in a different order, i.e. one more "
                         "confound. With the counts locked the order matches the reference sample by sample")
    ap.add_argument("--features-root", type=Path, default=None,
                    help="root of the per-chain features (holding <source>/*.parquet and eval/). "
                         "Defaults to features/parsed under --root")
    ap.add_argument("--index", default="split/index.parquet",
                    help="which split index to draw from, relative to --root")
    ap.add_argument("--start-epoch", type=int, default=0,
                    help="build only epochs >= this, into a fresh --out; the general-text streams are "
                         "first advanced past what epochs < start consumed (--prior-stats), so the "
                         "result is identical to the same epochs of a full build")
    ap.add_argument("--prior-stats", type=Path,
                    help="JSON {epoch: stats} of the epochs already built (the per-epoch lines build_mixture prints)")
    args = ap.parse_args()

    if not QWEN:
        raise SystemExit("set QWEN_BASE to the directory of the base tokenizer (Qwen3.5-9B, or a "
                         "checkpoint with the extended vocabulary)")
    needs_general = (args.general_frac if args.general_frac is not None else GENERAL_FRAC) > 0 \
        or args.general_samples_from is not None
    if needs_general and not glob.glob(FINEWEB):
        raise SystemExit(f"FINEWEB_GLOB matches no files: {FINEWEB!r}")

    tokenizer = AutoTokenizer.from_pretrained(QWEN)
    # When QWEN_BASE points at an incomplete directory that only has config.json / merges.txt,
    # AutoTokenizer **does not raise**; it silently degrades to a nearly empty vocabulary (observed:
    # len 8229 after adding the protein tokens, with <aaA> at id 1). Data built with it has wrong
    # token ids throughout, and that only shows when training starts, so it is caught here.
    if len(tokenizer) < 100_000:
        raise SystemExit(f"the base tokenizer has only {len(tokenizer)} tokens; {QWEN} is not a "
                         "complete tokenizer directory")
    n_added = tokenizer.add_tokens(ALL_TOKENS)
    # QWEN_BASE may point at a checkpoint whose vocabulary **has already been extended**. Then
    # add_tokens returns 0, which is not an error -- what matters is that all tokens are present
    # and that their ids agree with previously built data, otherwise the .tok.npy files do not
    # line up.
    if n_added not in (0, len(ALL_TOKENS)):
        raise SystemExit(f"unexpected vocabulary state: {n_added} / {len(ALL_TOKENS)} tokens were new -- "
                         "the base vocabulary contains only part of the protein tokens, so id "
                         "alignment cannot be guaranteed")
    missing = [t for t in (ALL_TOKENS[0], ALL_TOKENS[-1], "<seq>", "</seq>")
               if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if missing:
        raise SystemExit(f"vocabulary lacks {missing}")
    tokenizer.save_pretrained(args.out / "tokenizer")

    # The global default TASK_P is left unchanged so that older mixtures can be rebuilt exactly.
    task_p = TASK_P_VLM if args.task_mix == "vlm" else TASK_P
    print(f"task_mix={args.task_mix} -> {task_p}", flush=True)

    ref_general = None
    if args.general_samples_from is not None:
        ref_general = {k: v["general"] for k, v in
                       json.loads(args.general_samples_from.read_text())["epochs"].items()}
        print(json.dumps({"general_samples_locked_to": str(args.general_samples_from)}), flush=True)

    index = pd.read_parquet(args.root / args.index)
    parsed_dir, eval_dir = feature_dirs(args.root, args.features_root)
    feats = load_features(parsed_dir, index)
    index = index[index["id"].isin(feats)]
    print(f"features: {len(feats)} chains usable", flush=True)

    summary: dict = {"tokenizer_base": QWEN, "added_tokens": n_added, "vocab_after": len(tokenizer),
                     "format": args.format, "task_p": task_p,
                     "n_struct": N_STRUCT, "features_root": str(parsed_dir),
                     "general_samples_from": (str(args.general_samples_from) if args.general_samples_from else None), "task_mix": args.task_mix, "max_res": MAX_RES, "cluster_exp": CLUSTER_EXP, "general_frac": (args.general_frac if args.general_frac is not None else GENERAL_FRAC), "index": args.index, "epochs": {}}

    # validation sets (fold task, no crop needed: val chains are 50..600 aa)
    val_rng = np.random.default_rng(args.seed + 999)
    val_ids = index[index.split == "val"]["id"].tolist()
    val = [(i, "fold", render("fold", feats[i], val_rng, args.format)) for i in val_ids]
    write_shards(val, count_tokens(tokenizer, [v[2] for v in val]), args.out / "val", "val_id")
    fb = [r for r in pq.read_table(eval_dir / "foldbench334.parquet").to_pylist()]
    for r in fb:
        r["struct_tokens"] = np.asarray(r["struct_tokens"], dtype=np.int16)
    fb_samples = [(r["id"], "fold", render("fold", r, val_rng, args.format)) for r in fb]
    write_shards(fb_samples, count_tokens(tokenizer, [s[2] for s in fb_samples]), args.out / "val", "foldbench334")
    summary["val"] = {"val_id": len(val), "foldbench334": len(fb_samples)}

    train = index[index.split == "train"]
    clusters = train.groupby("rep")["id"].apply(list)
    general_iters = {k: iter_general(k, np.random.default_rng(args.seed), tokenizer) for k in GENERAL_SPLIT}
    if args.start_epoch:
        prior = json.loads(args.prior_stats.read_text())
        for epoch in range(args.start_epoch):
            st = prior[str(epoch)]
            for kind in GENERAL_SPLIT:  # each epoch drew whole 2048-chunks and dropped the tail of the last
                for _ in range(math.ceil(st["general"][kind]["samples"] / 2048) * 2048):
                    next(general_iters[kind])
            summary["epochs"][epoch] = st
        print(json.dumps({"skipped_epochs": args.start_epoch}), flush=True)
    for epoch in range(args.start_epoch, args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        samples: list[tuple[str, str, str]] = []
        for members in clusters:
            k = math.ceil(len(members) ** CLUSTER_EXP)
            for cid in rng.choice(members, size=k, replace=False):
                f = feats[cid]
                for task, p in task_p.items():
                    if rng.random() < p:
                        samples.append((cid, task, render(task, crop(f, rng), rng, args.format)))
        n_tok = count_tokens(tokenizer, [s[2] for s in samples])
        protein_tokens = int(sum(n_tok))
        gfrac = args.general_frac if args.general_frac is not None else GENERAL_FRAC
        general_target = protein_tokens * gfrac / (1 - gfrac)
        stats = {"protein_samples": len(samples), "protein_tokens": protein_tokens, "general": {}}
        for kind, frac in GENERAL_SPLIT.items():
            target, got, buf = general_target * frac, 0, []
            want_n = None if ref_general is None else ref_general[str(epoch)][kind]["samples"]
            while (len(buf) < want_n) if want_n is not None else (got < target):
                chunk = [next(general_iters[kind]) for _ in range(2048)]
                counts = count_tokens(tokenizer, [c[1] for c in chunk])
                for (gid, text), n in zip(chunk, counts):
                    buf.append(((gid, kind, text), n))
                    got += n
                    if (len(buf) >= want_n) if want_n is not None else (got >= target):
                        break
            samples.extend(b[0] for b in buf)
            n_tok.extend(b[1] for b in buf)
            stats["general"][kind] = {"samples": len(buf), "tokens": int(got)}
        order = rng.permutation(len(samples))
        samples = [samples[i] for i in order]
        n_tok = [n_tok[i] for i in order]
        write_shards(samples, n_tok, args.out / "train", f"epoch{epoch}")
        by_task = pd.DataFrame({"task": [s[1] for s in samples], "n": n_tok}).groupby("task")["n"].agg(["count", "sum", "mean"])
        stats["by_task"] = {t: {"samples": int(r["count"]), "tokens": int(r["sum"]), "mean": round(float(r["mean"]), 1)}
                            for t, r in by_task.iterrows()}
        stats["total_tokens"] = int(sum(n_tok))
        summary["epochs"][epoch] = stats
        print(json.dumps({"epoch": epoch, **stats}), flush=True)
    summary["total_tokens"] = sum(e["total_tokens"] for e in summary["epochs"].values())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
