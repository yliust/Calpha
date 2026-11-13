"""Merge the per-rank checkpoints of a stage-1 run back into one full model.

During training every rank saves its own file (train_stage1.save_ckpt):
  - trunk, LM head, ESM adapter: sharded by FSDP along dimension 0 into one piece per rank
    (8 pieces for an 8-GPU run), each piece living on a single rank;
  - flow head and condition projection (names starting with flow. / cond_): not sharded, every
    rank holds a full copy.
Merging follows from that: the former are concatenated along dimension 0 in rank order, the latter
are taken from rank 0.

Output (--out directory):
  merged.pt         full state_dict (CPU, fp32), loadable with DualHead.load_state_dict
  flow_head.ckpt    the flow head only ("flow." prefix removed), self-contained
  meta.json         base model path, condition layers, whether ESM2 is used, source checkpoint

  python stage1/export_stage1.py --run <stage-1 run dir> [--step N] --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

UNSHARDED = ("flow.", "cond_")


def fill_flow_buffers(flow_sd: dict) -> dict:
    """Add the flow head's buffers so that flow_head.ckpt is self-contained and loads with strict=True.

    The training checkpoint stores parameters only, not buffers (positional-encoding frequencies
    etc. are generated deterministically from the config at construction). A fresh head of the
    same architecture is built here and its buffers are copied in; if any key that has to be
    filled in turns out to be a parameter, the checkpoint really lacks weights and this raises.
    """
    import train_stage1 as rc
    import conditioning as sc

    head = rc.build_flow_head(None)
    sc.retarget_condition(head, n_layers=1, dim=flow_sd["esm_s_proj.proj.0.weight"].shape[1], reinit=True)
    params = {n for n, _ in head.named_parameters()}
    for k, v in head.state_dict().items():
        if k not in flow_sd:
            if k in params:
                raise RuntimeError(f"checkpoint lacks flow-head parameter {k}")
            flow_sd[k] = v.clone()
    return flow_sd


def merge(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    files = sorted(ckpt_dir.glob("rank*.pt"))
    if not files:
        raise SystemExit(f"no rank*.pt under {ckpt_dir}")
    shards = [torch.load(f, map_location="cpu", weights_only=False)["model"] for f in files]
    out = {}
    for name in shards[0]:
        if name.startswith(UNSHARDED):
            out[name] = shards[0][name].float().clone()
        else:
            parts = [s[name] for s in shards]
            out[name] = torch.cat(parts, 0).float() if parts[0].dim() > 0 else parts[0].float()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="training output directory (holds ckpt/ and args.json)")
    ap.add_argument("--step", type=int, default=0, help="0 = ckpt/latest")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ck = a.run / "ckpt"
    name = f"step_{a.step:07d}" if a.step else (ck / "latest").read_text().strip()
    src = ck / name
    if not src.is_dir() and (a.run / "keep" / name).is_dir():
        src = a.run / "keep" / name     # milestone checkpoints copied aside before rotation deleted them
    merged = merge(src)
    args = json.loads((a.run / "args.json").read_text())

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(merged, a.out / "merged.pt")
    flow = fill_flow_buffers({k[len("flow."):]: v for k, v in merged.items() if k.startswith("flow.")})
    torch.save(flow, a.out / "flow_head.ckpt")
    meta = {"source": str(src), "model": args["model"], "cond_layers": args["cond_layers"],
            "no_esm": bool(args.get("no_esm", False)), "lambda_flow": args.get("lambda_flow"),
            "n_params": sum(v.numel() for v in merged.values())}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
