"""Command line: calpha fold | chat | benchmark | score | compare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import HF_REPO, NUM_STEPS, SEED, TAU

DEFAULT_SF = "~/.cache/calpha/simplefold"


def _read_fasta(path: Path) -> dict[str, str]:
    seqs, name = {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0].replace("|", "_").replace(":", "__") or f"seq{len(seqs)}"
            seqs[name] = ""
        elif line and name is not None:
            seqs[name] += line
    return seqs


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--weights", default=None, help=f"local directory or Hugging Face repo (default {HF_REPO})")
    p.add_argument("--simplefold-dir", default=DEFAULT_SF,
                   help="where the public SimpleFold-3B checkpoint is stored (downloaded on first use)")
    p.add_argument("--num-steps", type=int, default=NUM_STEPS)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--nsample", type=int, default=1)
    p.add_argument("--seed", type=int, default=SEED)


def main() -> None:
    ap = argparse.ArgumentParser(prog="calpha", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fold", help="fold sequences with the canonical request")
    p.add_argument("input", help="a FASTA file or a raw amino-acid sequence")
    p.add_argument("--out", type=Path, default=Path("predictions"))
    _common(p)

    p = sub.add_parser("chat", help="ask in plain language; the model decides whether to fold")
    p.add_argument("request", help="the request; write {seq} where the protein goes")
    p.add_argument("--seq", default=None)
    p.add_argument("--out", type=Path, default=Path("predictions"))
    _common(p)

    p = sub.add_parser("benchmark", help="generate predictions for a benchmark set")
    p.add_argument("--set", type=Path, required=True, help="benchmarks/<name>.parquet")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    _common(p)

    p = sub.add_parser("score", help="score predictions against a benchmark set")
    p.add_argument("--set", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="directory with predictions/ or pred_ca.npz")
    p.add_argument("--hits", type=Path, default=Path("benchmarks/homology_hits.tsv.gz"))

    p = sub.add_parser("compare", help="paired bootstrap comparison of two scores.csv files")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--metric", default="tm")

    a = ap.parse_args()
    if a.cmd in ("fold", "chat"):
        from .pipeline import Calpha

        model = Calpha.from_pretrained(a.weights, a.simplefold_dir, num_steps=a.num_steps, tau=a.tau,
                                       nsample=a.nsample, seed=a.seed)
        if a.cmd == "fold":
            seqs = _read_fasta(Path(a.input)) if Path(a.input).is_file() else {"target": a.input.strip().upper()}
            for name, seq in seqs.items():
                for path in model.fold(seq, a.out, name=name).structures:
                    print(path)
        else:
            ans = model.chat(a.request, a.seq, a.out)
            print(ans.text)
            for path in ans.structures:
                print(path)
    elif a.cmd == "benchmark":
        from .eval.benchmark import generate

        generate(a.set, a.out, a.weights, a.simplefold_dir, a.rank, a.world, a.nsample,
                 a.num_steps, a.tau, a.seed)
    elif a.cmd == "score":
        from .eval.score import run

        s = run(a.set, a.out, a.hits)
        o = s["overall"]
        print(f"TM {o['tm']}  lDDT {o['lddt']}  GDT-TS {o['gdt_ts']}  ({o['n_ok']}/{o['n']} targets)  "
              f"no-homolog TM {s['no_family']['tm']}")
    elif a.cmd == "compare":
        from .eval.compare import compare

        print(json.dumps(compare(a.a, a.b, a.metric), indent=1))


if __name__ == "__main__":
    main()
