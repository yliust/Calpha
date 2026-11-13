"""Structure metrics on CA coordinates with known residue correspondence (numpy only).

All functions take pred / ref arrays of shape (L, 3) and score only residues where
both are finite. TM-score follows the TM-score program's fixed-correspondence search
(seeded fragment superpositions refined iteratively); lDDT is the CA-only variant
(15 A inclusion radius, thresholds 0.5/1/2/4 A).
"""

from __future__ import annotations

import numpy as np


def _valid(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.isfinite(pred).all(-1) & np.isfinite(ref).all(-1)


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |(p @ R.T + t) - q|."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, qc - pc @ r.T


def rmsd(pred: np.ndarray, ref: np.ndarray) -> float:
    m = _valid(pred, ref)
    p, q = pred[m].astype(np.float64), ref[m].astype(np.float64)
    r, t = kabsch(p, q)
    return float(np.sqrt(((p @ r.T + t - q) ** 2).sum(-1).mean()))


def tm_d0(n: int) -> float:
    return max(1.24 * np.cbrt(max(n, 19) - 15) - 1.8, 0.5)


def tm_score(pred: np.ndarray, ref: np.ndarray, norm_len: int | None = None) -> float:
    """TM-score of pred vs ref, normalised by norm_len (default: number of ref residues with coords)."""
    m = _valid(pred, ref)
    p, q = pred[m].astype(np.float64), ref[m].astype(np.float64)
    n = len(p)
    lnorm = norm_len or int(np.isfinite(ref).all(-1).sum())
    d0 = tm_d0(lnorm)
    d0_search = min(max(d0, 4.5), 8.0)
    best = 0.0
    frag = n
    while True:
        step = max(1, frag // 2) if frag < n else 1
        for start in range(0, n - frag + 1, step):
            idx = np.arange(start, start + frag)
            for _ in range(20):
                r, t = kabsch(p[idx], q[idx])
                d = np.linalg.norm(p @ r.T + t - q, axis=-1)
                best = max(best, float((1.0 / (1.0 + (d / d0) ** 2)).sum() / lnorm))
                cut = d0_search
                new = np.where(d < cut)[0]
                while len(new) < 3 and cut < 50:
                    cut += 0.5
                    new = np.where(d < cut)[0]
                if len(new) < 3 or np.array_equal(new, idx):
                    break
                idx = new
        if frag <= 4:
            break
        frag = max(4, frag // 2)
    return best


def lddt_ca(pred: np.ndarray, ref: np.ndarray, radius: float = 15.0) -> float:
    m = _valid(pred, ref)
    p, q = pred[m].astype(np.float64), ref[m].astype(np.float64)
    dq = np.linalg.norm(q[:, None] - q[None], axis=-1)
    dp = np.linalg.norm(p[:, None] - p[None], axis=-1)
    pair = (dq < radius) & ~np.eye(len(q), dtype=bool)
    if not pair.any():
        return float("nan")
    err = np.abs(dp - dq)[pair]
    return float(np.mean([(err < t).mean() for t in (0.5, 1.0, 2.0, 4.0)]))


def gdt_ts(pred: np.ndarray, ref: np.ndarray) -> float:
    """Approximate GDT-TS: best fraction under each cutoff over TM-style seed superpositions."""
    m = _valid(pred, ref)
    p, q = pred[m].astype(np.float64), ref[m].astype(np.float64)
    n = int(np.isfinite(ref).all(-1).sum())
    out = []
    for cut in (1.0, 2.0, 4.0, 8.0):
        best, frag = 0, len(p)
        while frag >= 4:
            for start in range(0, len(p) - frag + 1, max(1, frag // 2)):
                idx = np.arange(start, start + frag)
                for _ in range(10):
                    r, t = kabsch(p[idx], q[idx])
                    d = np.linalg.norm(p @ r.T + t - q, axis=-1)
                    new = np.where(d < cut)[0]
                    best = max(best, len(new))
                    if len(new) < 3 or np.array_equal(new, idx):
                        break
                    idx = new
            frag //= 2
        out.append(best / n)
    return float(np.mean(out))


def all_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    return {"rmsd": rmsd(pred, ref), "tm": tm_score(pred, ref), "lddt": lddt_ca(pred, ref), "gdt_ts": gdt_ts(pred, ref)}
