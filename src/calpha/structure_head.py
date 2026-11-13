"""The structure side of Calpha: the public SimpleFold-3B flow-matching head, with its
conditioning entry rebuilt for language-model hidden states.

SimpleFold conditions its residue trunk on one tensor `esm_s` of shape (B, L, n_layers, C):

    mixed = softmax(esm_s_combine) @ esm_s      # weighted sum over layers
    cond  = esm_s_proj(mixed)                   # C -> head width

In SimpleFold `esm_s` holds the 37 layers of ESM2-3B (C = 2560). In Calpha it holds nine layers
of the language model (C = 4096). `esm_s_combine` and `esm_s_proj` are therefore rebuilt for
(9, 4096); these two modules are the 8.4M-parameter conditioning front-end, the only weights
trained in stage 2. Every other weight of the head is the public SimpleFold-3B checkpoint,
which this code downloads from its original location and does not redistribute.
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch


def _add_simplefold_to_path() -> Path:
    """SimpleFold's modules import each other as top-level packages (`from utils... import`),
    so its source directory has to be on sys.path."""
    src = os.environ.get("SIMPLEFOLD_SRC")
    if src is None:
        try:
            import simplefold
        except ImportError as e:
            raise ImportError(
                "SimpleFold is not installed. Install it with\n"
                "  pip install git+https://github.com/apple/ml-simplefold.git\n"
                "or set SIMPLEFOLD_SRC to <ml-simplefold>/src/simplefold.") from e
        src = str(Path(simplefold.__file__).resolve().parent)
    if src not in sys.path:
        sys.path.insert(0, src)
    return Path(src)


def _use_fair_esm_registry() -> None:
    """SimpleFold loads ESM2 through torch.hub from GitHub. fair-esm ships the same loaders and
    fetches the same weights without the GitHub round trip."""
    import esm as fair_esm
    import inference
    from utils import esm_utils

    esm_utils.esm_registry = {
        "esm2_3B": fair_esm.pretrained.esm2_t36_3B_UR50D,
        "esm2_650M": fair_esm.pretrained.esm2_t33_650M_UR50D,
    }
    inference.esm_registry = esm_utils.esm_registry


def retarget_condition(model, n_layers: int, dim: int) -> None:
    """Rebuild the two modules that depend on the shape of the condition; keep everything else."""
    from model.torch.layers import ConditionEmbedder

    dev = model.esm_s_combine.device
    model.esm_s_combine = torch.nn.Parameter(torch.zeros(n_layers, device=dev))
    old = model.esm_s_proj
    model.esm_s_proj = ConditionEmbedder(input_dim=dim, hidden_size=old.proj[0].out_features,
                                         dropout_prob=old.dropout_prob).to(dev)


class StructureHead:
    def __init__(self, frontend_path: str | Path, simplefold_dir: str | Path,
                 n_layers: int = 9, dim: int = 4096, model: str = "simplefold_3B",
                 num_steps: int = 200, tau: float = 0.01, nsample: int = 1, seed: int = 42):
        from safetensors.torch import load_file

        _add_simplefold_to_path()
        _use_fair_esm_registry()
        from inference import initialize_folding_model, initialize_others

        self.dir = Path(simplefold_dir)
        self.args = argparse.Namespace(
            simplefold_model=model, ckpt_dir=str(self.dir), backend="torch", num_steps=num_steps,
            tau=tau, nsample_per_protein=nsample, plddt=False, output_format="mmcif", seed=seed)
        # downloads <simplefold_dir>/simplefold_3B.ckpt from Apple's CDN on first use
        self.model, self.device = initialize_folding_model(self.args)
        retarget_condition(self.model, n_layers, dim)
        state = load_file(str(frontend_path))
        res = self.model.load_state_dict(state, strict=False)
        if res.unexpected_keys or any(k.startswith("esm_s_") for k in res.missing_keys):
            raise RuntimeError(f"front-end weights do not match the head: unexpected "
                               f"{res.unexpected_keys[:3]}, missing "
                               f"{[k for k in res.missing_keys if k.startswith('esm_s_')][:3]}")
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        self.tokenizer, self.featurizer, self.processor, self.flow, self.sampler = \
            initialize_others(self.args, self.device)

    # ---- inputs

    def prepare(self, sequences: dict[str, str], work: Path) -> list[Path]:
        """Turn {name: sequence} into SimpleFold's input files. One FASTA per target: SimpleFold
        makes one record per file, so a combined FASTA would be folded as one complex.
        Returns the structure files, sorted by name."""
        from utils import fasta_utils
        from utils.fasta_utils import process_fastas

        fdir = work / "fastas"
        fdir.mkdir(parents=True, exist_ok=True)
        for name, seq in sequences.items():
            (fdir / f"{name}.fasta").write_text(f">{name}|protein|\n{seq}\n")
        ccd = self.dir / "cache" / "ccd.pkl"     # chemical component dictionary, fetched once
        if not ccd.exists():
            import urllib.request

            ccd.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading the CCD dictionary to {ccd}", flush=True)
            urllib.request.urlretrieve(fasta_utils.CCD_URL, str(ccd))
        process_fastas(data=sorted(fdir.glob("*.fasta")), out_dir=work, ccd_path=ccd)
        return sorted((work / "structures").glob("*.npz"))

    # ---- sampling

    @torch.no_grad()
    def sample(self, struct_file: Path, work: Path, condition, out_dir: Path) -> list[Path]:
        """Fold one prepared target. `condition(sequence) -> (L, n_layers, dim)` supplies the
        language-model hidden states. Writes <name>_sampled_<i>.cif and returns the paths."""
        from inference import generate_structure
        from utils.boltz_utils import process_structure, save_structure
        from utils.datamodule_utils import process_one_inference_structure

        n = self.args.nsample_per_protein
        batch, structure, record = process_one_inference_structure(
            struct_file, work / "records" / f"{struct_file.stem}.json",
            self.tokenizer, self.featurizer, self.processor, None, None, None)
        L = batch["res_type"].shape[1]
        seq = batch["aa_seq"][0]
        cond = condition(seq)
        esm_s = torch.zeros(1, L, cond.shape[1], cond.shape[2], device=self.device)
        m = min(L, cond.shape[0])
        esm_s[0, :m] = cond[:m].to(self.device)
        batch["esm_s"] = esm_s.repeat_interleave(n, dim=0)
        coord, pad_mask, _ = generate_structure(self.args, batch, self.sampler, self.flow,
                                                self.processor, self.model, None, None, self.device)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            s = process_structure(deepcopy(structure), coord[i], pad_mask[i], record, backend="torch")
            save_structure(s, out_dir, f"{record.id}_sampled_{i}", "mmcif", plddts=None)
            paths.append(out_dir / f"{record.id}_sampled_{i}.cif")
        return paths
