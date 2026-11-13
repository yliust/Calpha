"""Build SimpleFold training targets from raw mmCIF files.

SimpleFold's own `process_mmcif.py` reads the CCD component dictionary out of a Redis server
(`redis-server --dbfilename ccd.rdb`). The same data is in the `ccd.pkl` that SimpleFold downloads
for inference, so this wrapper swaps the Redis-backed `Resource` for a plain dict and otherwise
calls SimpleFold's pipeline unchanged. That removes a server dependency and makes the step
reproducible from files we archive.

Three stages:
  targets    mmCIF -> <out>/structures/*.npz + <out>/records/*.json + manifest
  tokenize   targets -> <token_out>/tokens/*.pkl + records + manifest.json
  filter     drop records that are not protein-only, rewriting manifest.json

  export SIMPLEFOLD_SRC=<ml-simplefold>/src/simplefold SIMPLEFOLD_CCD=<path to ccd.pkl>
  python data/simplefold_targets.py targets \
      --data-dir <mmcif_dir> --out-dir <targets_dir> --num-processes 32
  python data/simplefold_targets.py tokenize \
      --target-dir <targets_dir> --token-dir <tokens_dir>
  python data/simplefold_targets.py filter --token-dir <tokens_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# SIMPLEFOLD_SRC: <ml-simplefold>/src/simplefold of the upstream checkout (stages targets, tokenize).
# SIMPLEFOLD_CCD: SimpleFold's ccd.pkl component dictionary (stage targets).
SF_SRC = Path(os.environ.get("SIMPLEFOLD_SRC", ""))
CCD = Path(os.environ.get("SIMPLEFOLD_CCD", ""))


def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"environment variable(s) not set: {', '.join(missing)}")


_CCD_CACHE: dict | None = None


def _ccd() -> dict:
    global _CCD_CACHE
    if _CCD_CACHE is None:
        with CCD.open("rb") as fh:
            _CCD_CACHE = pickle.load(fh)
    return _CCD_CACHE


class DictResource:
    """Drop-in for SimpleFold's Redis-backed Resource, served from ccd.pkl.

    Holds no state: the 345 MB dict of rdkit Mols is loaded lazily into a module global,
    once per worker process. Keeping it off the instance matters because p_umap pickles the
    resource to every worker -- carrying the dict there made processing 8 s/entry, which is
    ~200 h for the 99k PDB entries.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def get(self, key: str):
        return _ccd().get(key)

    def __getitem__(self, key: str):
        out = _ccd().get(key)
        if out is None:
            raise KeyError(key)
        return out


def _stub_redis() -> None:
    """process_mmcif does `from redis import Redis` at module level, but Redis is never used
    here -- Resource is replaced by a dict below. If the package is not installed (e.g. on a
    machine without internet access) a stub module is put in its place; with the real redis
    package installed this does nothing."""
    try:
        import redis  # noqa: F401
    except ModuleNotFoundError:
        import types

        m = types.ModuleType("redis")
        m.Redis = object
        sys.modules["redis"] = m


def stage_targets(a: argparse.Namespace) -> None:
    _require_env("SIMPLEFOLD_SRC", "SIMPLEFOLD_CCD")
    sys.path.insert(0, str(SF_SRC))
    import rdkit
    _stub_redis()
    import process_mmcif as pm

    # p_umap pickles `resource` to each worker; a dict of 43k rdkit Mols is ~350 MB per
    # process, so keep the worker count modest on top of --num-processes.
    rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)
    pm.Resource = DictResource

    if not a.use_assembly:
        # `parse_mmcif` defaults to use_assembly=True and process_mmcif.parse never overrides it,
        # so every entry goes through gemmi's assembly expansion -- which raises "no subchain X"
        # on a large share of older PDB entries (all 40 of our first sample failed, gemmi 0.7.3,
        # the version SimpleFold pins). We fold single chains cropped to 256 residues, so the
        # asymmetric unit is also the right unit for us.
        import utils.mmcif_utils as mu

        _orig = mu.parse_mmcif
        pm.parse_mmcif = lambda path, components, **_: _orig(path, components, use_assembly=False)

    pm.process(argparse.Namespace(
        data_dir=a.data_dir, out_dir=a.out_dir, num_processes=a.num_processes,
        redis_host="", redis_port=0, use_assembly=a.use_assembly, max_file_size=None,
    ))


def stage_tokenize(a: argparse.Namespace) -> None:
    """SimpleFold's process_structure.py keeps its driver inline under __main__; repeat it here."""
    _require_env("SIMPLEFOLD_SRC")
    sys.path.insert(0, str(SF_SRC))
    from tqdm import tqdm

    import process_structure as ps
    from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
    from boltz_data_pipeline.types import Manifest

    manifest = Manifest.load(a.target_dir / "manifest.json")
    print(f"Number of records after filtering: {len(manifest.records)}")
    tok_dir = a.token_dir / "tokens"
    rec_dir = a.token_dir / "records"
    tok_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(manifest.records):
        ps.tokenize_structure(record, BoltzTokenizer(), a.target_dir, str(tok_dir), rec_dir)
    ps.finalize(a.token_dir)


def stage_filter(a: argparse.Namespace) -> None:
    """Keep only protein-only entries (every chain mol_type == 0).

    SimpleFold's `extract_sequence_from_tokens` maps token residue names through a 20-amino-acid
    table, so a DNA/RNA residue raises `KeyError: 'DG'` in a dataloader worker and kills the run.
    The cropper picks a random neighbourhood, so a protein-DNA complex can fail on some epochs
    and not others -- which is why a short smoke run over the same data passed. Dropping mixed
    entries is also closer to our own pool, which is protein chains only.
    """
    man = a.token_dir / "manifest.json"
    orig = a.token_dir / "manifest.all.json"
    if not orig.exists():
        orig.write_bytes(man.read_bytes())
    data = json.loads(orig.read_text())
    records = data["records"] if isinstance(data, dict) else data
    kept = [r for r in records
            if r.get("chains") and all(c.get("mol_type") == 0 for c in r["chains"])]
    out = {**data, "records": kept} if isinstance(data, dict) else kept
    man.write_text(json.dumps(out))
    print(f"protein-only: {len(kept)} / {len(records)} records "
          f"({len(records) - len(kept)} dropped)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("targets")
    t.add_argument("--data-dir", type=Path, required=True)
    t.add_argument("--out-dir", type=Path, required=True)
    t.add_argument("--num-processes", type=int, default=16)
    t.add_argument("--use-assembly", action="store_true")
    t.set_defaults(fn=stage_targets)
    k = sub.add_parser("tokenize")
    k.add_argument("--target-dir", type=Path, required=True)
    k.add_argument("--token-dir", type=Path, required=True)
    k.set_defaults(fn=stage_tokenize)
    f = sub.add_parser("filter")
    f.add_argument("--token-dir", type=Path, required=True)
    f.set_defaults(fn=stage_filter)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
