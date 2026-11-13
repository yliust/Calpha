"""Run Calpha over a benchmark set, the way the reported numbers were produced.

A set is a parquet file with columns id, sequence, length, coords (see benchmarks/README.md).
Targets are sharded over GPUs with --rank/--world; each shard seeds its random generator with
`seed + rank` and folds its targets in sorted order, one sample each. The reported numbers used
--world 8. With a different --world the noise drawn for a given target differs, so individual
scores move slightly while the mean stays within the sample-to-sample spread reported in the
paper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def stem(target_id: str) -> str:
    """Target ids such as 'foldbench:5sbj_A' have to become file names."""
    return target_id.replace(":", "__")


def generate(set_file: Path, out: Path, weights, simplefold_dir, rank: int = 0, world: int = 1,
             nsample: int = 1, num_steps: int = 200, tau: float = 0.01, seed: int = 42) -> None:
    import pyarrow.parquet as pq
    import torch

    from ..pipeline import Calpha, resolve_weights

    work, pred = out / "work", out / "predictions"
    pred.mkdir(parents=True, exist_ok=True)
    model = Calpha(resolve_weights(weights), Path(simplefold_dir).expanduser(), num_steps=num_steps,
                   tau=tau, nsample=nsample, seed=seed + rank)
    if rank == 0:
        rows = pq.read_table(set_file, columns=["id", "sequence"]).to_pylist()
        (work).mkdir(parents=True, exist_ok=True)
        (work / "stems.json").write_text(json.dumps({stem(r["id"]): r["id"] for r in rows}, indent=1))
        model.head.prepare({stem(r["id"]): r["sequence"] for r in rows}, work)
        (work / "READY").write_text("ok")
    else:
        while not (work / "READY").exists():
            time.sleep(5)

    files = sorted((work / "structures").glob("*.npz"))[rank::world]
    print(f"[rank {rank}] {len(files)} targets", flush=True)
    for n, f in enumerate(files, 1):
        if (pred / f"{f.stem}_sampled_{nsample - 1}.cif").exists():
            continue
        model.head.sample(f, work, lambda s: model.lm.residue_states(model.canonical_prompt(s)), pred)
        print(f"[rank {rank}] {n}/{len(files)} {f.stem}", flush=True)
        torch.cuda.empty_cache()
