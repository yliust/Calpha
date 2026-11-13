"""The scorer behind every number in the report.

Every system's predictions are first reduced to one alpha-carbon coordinate per residue
(`pred_ca.npz`: target id -> (L, 3), NaN where a residue is missing) and then scored by this
one program, so that numbers of different systems are comparable.

Rules
* a target with no usable prediction scores 0 (TM / lDDT / GDT) and stays in the mean;
* a prediction that covers only part of the chain is scaled by its coverage;
* three reference-free checks are reported per target: fraction of consecutive CA-CA distances
  in the physical range (3.6-4.0 A), fraction of CA pairs closer than 3 A (|i-j| >= 3), and the
  radius of gyration divided by L^0.34;
* buckets: four length ranges, and homolog / no-homolog from an MMseqs2 search of each target
  against the training pool (identity >= 0.30 and coverage >= 0.5 on either side).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import stem
from .metrics import all_metrics

HITS_COLUMNS = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]
BUCKETS = ["<100", "100-300", "300-600", ">600"]


def length_bucket(n: int) -> str:
    return "<100" if n < 100 else "100-300" if n < 300 else "300-600" if n <= 600 else ">600"


def validity(ca: np.ndarray) -> dict:
    ok = np.isfinite(ca).all(-1)
    pair = ok[1:] & ok[:-1]
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)[pair]
    c = ca[ok].astype(np.float64)
    if len(c) < 4 or len(d) == 0:
        return {"ca_ca_ok": np.nan, "clash": np.nan, "rg_norm": np.nan}
    dm = np.linalg.norm(c[:, None] - c[None], axis=-1)
    iu = np.triu_indices(len(c), k=3)
    rg = float(np.sqrt(((c - c.mean(0)) ** 2).sum(-1).mean()))
    return {"ca_ca_ok": float(((d > 3.6) & (d < 4.0)).mean()),
            "clash": float((dm[iu] < 3.0).mean()) if len(iu[0]) else 0.0,
            "rg_norm": rg / len(c) ** 0.34}


def ca_from_cif(path: Path, length: int) -> np.ndarray:
    """(L, 3) alpha-carbon coordinates of the first chain; residues indexed by their order."""
    import gemmi

    ca = np.full((length, 3), np.nan, dtype=np.float32)
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    for i, res in enumerate(st[0][0]):
        if i >= length:
            break
        at = res.find_atom("CA", "*")
        if at is not None:
            ca[i] = (at.pos.x, at.pos.y, at.pos.z)
    return ca


def collect_cif(pred_dir: Path, targets: list[dict], sample: int = 0) -> dict[str, np.ndarray]:
    d = {}
    for r in targets:
        cif = pred_dir / f"{stem(r['id'])}_sampled_{sample}.cif"
        if cif.exists():
            d[r["id"]] = ca_from_cif(cif, r["length"])
    return d


def load_targets(set_file: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(set_file, columns=["id", "sequence", "coords", "length"]).to_pylist()


def reference_ca(row: dict) -> np.ndarray:
    # coords is stored flat as (L, 5 atoms [N, CA, C, O, CB], 3)
    return np.asarray(row["coords"], dtype=np.float32).reshape(-1, 5, 3)[:, 1]


def homolog_ids(hits_file: Path) -> set[str]:
    hits = pd.read_csv(hits_file, sep="\t", names=HITS_COLUMNS)
    return set(hits[(hits.fident >= 0.30) & ((hits.qcov >= 0.5) | (hits.tcov >= 0.5))]["query"])


def score(targets: list[dict], pred: dict[str, np.ndarray], family: set[str]) -> pd.DataFrame:
    rows = []
    for r in targets:
        ref = reference_ca(r)
        row = {"id": r["id"], "length": r["length"], "family": r["id"] in family,
               "len_bucket": length_bucket(r["length"])}
        p = pred.get(r["id"])
        if p is not None and p.shape != ref.shape:
            q = np.full(ref.shape, np.nan, dtype=np.float32)
            n = min(len(p), len(ref))
            q[:n] = p[:n]
            p = q
        both = None if p is None else int((np.isfinite(p).all(-1) & np.isfinite(ref).all(-1)).sum())
        if p is None or both < 4:
            row.update({"ok": False, "tm": 0.0, "lddt": 0.0, "gdt_ts": 0.0, "rmsd": np.nan,
                        "coverage": 0.0, "ca_ca_ok": np.nan, "clash": np.nan, "rg_norm": np.nan})
        else:
            m = all_metrics(p, ref)
            cov = both / max(int(np.isfinite(ref).all(-1).sum()), 1)
            row.update({"ok": True, "tm": m["tm"] * cov, "lddt": m["lddt"] * cov,
                        "gdt_ts": m["gdt_ts"] * cov, "rmsd": m["rmsd"], "coverage": cov, **validity(p)})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    def summ(d: pd.DataFrame) -> dict:
        ok = d[d.ok]
        return {"n": int(len(d)), "n_ok": int(len(ok)),
                "completion": round(len(ok) / max(len(d), 1), 4),
                "tm": round(float(d.tm.mean()), 4), "lddt": round(float(d.lddt.mean()), 4),
                "gdt_ts": round(float(d.gdt_ts.mean()), 4),
                "rmsd_median_ok": round(float(ok.rmsd.median()), 3) if len(ok) else None,
                "tm>0.5": round(float((d.tm > 0.5).mean()), 4),
                "ca_ca_ok": round(float(ok.ca_ca_ok.mean()), 4) if len(ok) else None,
                "clash": round(float(ok.clash.mean()), 5) if len(ok) else None,
                "rg_norm": round(float(ok.rg_norm.mean()), 3) if len(ok) else None}

    s = {"overall": summ(df), "family": summ(df[df.family]), "no_family": summ(df[~df.family])}
    for b in BUCKETS:
        d = df[df.len_bucket == b]
        if len(d):
            s[f"len{b}"] = summ(d)
    return s


def run(set_file: Path, out: Path, hits_file: Path, sample: int = 0) -> dict:
    """Score <out>/predictions/*.cif (or an existing <out>/pred_ca.npz) against a benchmark set."""
    targets = load_targets(set_file)
    if (out / "predictions").is_dir():
        pred = collect_cif(out / "predictions", targets, sample)
        np.savez_compressed(out / "pred_ca.npz", **pred)
    else:
        pred = dict(np.load(out / "pred_ca.npz"))
    df = score(targets, pred, homolog_ids(hits_file))
    df.to_csv(out / "scores.csv", index=False)
    s = summarize(df)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    return s
