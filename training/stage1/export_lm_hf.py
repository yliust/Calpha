"""Extract the plain language model from a merged stage-1 checkpoint as a standard HF directory.

Purpose: general-ability benchmarks (e.g. MMLU-Pro) need a model that from_pretrained can load
directly. In the dual-head model the LM parameters are named "lm.<original HF name>"; dropping the
prefix puts them back into the base architecture unchanged. The ESM adapter, the flow head and the
condition projection take no part in pure text generation and are discarded.

  python stage1/export_lm_hf.py --export <export_stage1.py output directory> [--out <dir>]
  (--out defaults to <export>/hf)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True, help="output directory of export_stage1.py")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or a.export / "hf"

    meta = json.loads((a.export / "meta.json").read_text())
    sd = torch.load(a.export / "merged.pt", map_location="cpu", weights_only=False)
    lm = {k[len("lm."):]: v.to(torch.bfloat16) for k, v in sd.items() if k.startswith("lm.")}

    model = AutoModelForCausalLM.from_pretrained(meta["model"], dtype=torch.bfloat16)
    res = model.load_state_dict(lm, strict=False)
    # The checkpoint stores parameters only: missing buffers (RoPE frequencies etc.) are allowed,
    # a missing parameter or an unexpected key is not
    params = {n for n, _ in model.named_parameters()}
    lost = [k for k in res.missing_keys if k in params]
    if lost or res.unexpected_keys:
        raise SystemExit(f"LM parameters do not match: missing {lost[:5]}, unexpected {res.unexpected_keys[:5]}")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(meta["model"]).save_pretrained(out)
    info = {"source": meta["source"], "base": meta["model"], "n_lm_params": sum(v.numel() for v in lm.values())}
    (out / "route_c_hf.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
