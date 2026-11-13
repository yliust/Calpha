"""Stage 2 (and the head-only control): train SimpleFold's structure head under a swapped condition.

Every group runs the same data, steps and hyper-parameters; only `--cond` (and what is frozen)
differs, so the groups are directly comparable:

  --cond onehot                                        floor (amino-acid identity only)
  --cond esm2                                          SimpleFold's own condition (head-only control)
  --cond routec_raw --cond-ckpt <stage-1 export>       hidden states of the stage-1 language model
       --freeze-flow                                   (Calpha stage 2: only the 8.4M front-end trains)
  --cond routec     --cond-ckpt <stage-1 export>       the stage-1 model's own condition projection

Loss, timestep resampling and rigid alignment are SimpleFold's (model/simplefold.py
`training_step`), reused rather than reimplemented so the groups stay comparable to the
pretrained checkpoint they start from.

  SIMPLEFOLD_SRC=<ml-simplefold>/src/simplefold TORCH_HOME=... \
  python stage2/train_stage2.py --cond esm2 --ckpt-dir <weights_dir> \
      --data <tokens_dir> --targets <targets_dir> --out <run_dir> --steps 20000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Sequence lengths differ from step to step (crop <= max_tokens, atoms <= max_atoms), so the caching
# allocator accumulates more and more fragments; expandable segments let it merge and reuse free
# segments. Must be set before `import torch`. PYTORCH_ALLOC_CONF is the name used by torch >= 2.9
# (formerly PYTORCH_CUDA_ALLOC_CONF); it is only filled in here when neither is set, so that
# launching this script directly gets it too.
if not (os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F

# Upstream apple/ml-simplefold checkout, directory <ml-simplefold>/src/simplefold.
if not os.environ.get("SIMPLEFOLD_SRC"):
    raise SystemExit("SIMPLEFOLD_SRC is not set; point it at <ml-simplefold>/src/simplefold")
SF_SRC = Path(os.environ["SIMPLEFOLD_SRC"])
sys.path.insert(0, str(SF_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    return rank() == 0


def sync_grads(params) -> None:
    """Average gradients by hand.

    DDP only synchronises the parameters that "the module it wraps uses in its forward"; anything
    trained inside a condition provider is invisible to DDP, and without synchronisation every GPU
    would train on its own -- equivalent to the batch not having grown at all.
    """
    import torch.distributed as dist

    ws = world_size()
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad /= ws


def setup_distributed():
    """Initialise only for processes started by torchrun; when run directly on one GPU the
    environment variables are absent and the single-GPU path is taken.

    With the 3B head a single GPU measured 7.06 s/step, about 20 h for 10k steps. Multi-GPU is
    used to **split the effective batch** (1 protein x 8 copies per GPU x 8 GPUs = 64), not to
    grow it eightfold -- the effective batch has to stay at 64, otherwise differences between
    groups cannot be compared with those of the earlier single-GPU experiments.
    """
    if world_size() <= 1:
        return torch.device("cuda")
    import datetime

    import torch.distributed as dist

    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    # explicit timeout: a hang should raise an error instead of blocking forever
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    return torch.device(f"cuda:{local}")


def build_loader(a: argparse.Namespace):
    from boltz_data_pipeline.crop.boltz import BoltzCropper
    from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
    from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
    from boltz_data_pipeline.types import Manifest
    from datasets.train_datamodule import SimpleFoldTrainingDataset
    from utils.datamodule_utils import Dataset, collate

    ds = Dataset(
        tokenized_dir=str(a.data), target_dir=a.targets, esm_dir="", connectivity_dir="",
        cheap_dir="", manifest=Manifest.load(a.data / "manifest.json"),
        cropper=BoltzCropper(), tokenizer=BoltzTokenizer(), featurizer=BoltzFeaturizer(),
    )
    train = SimpleFoldTrainingDataset(
        datasets=[ds], symmetries={}, max_atoms=a.max_atoms, max_tokens=a.max_tokens,
        pad_to_max_atoms=False, pad_to_max_tokens=False,  # collate pads to the batch max (as in SimpleFold's own PDB data config)
    )
    _guard_dataset(train)
    if is_main():
        print(f"dataset: {train.num_samples} records", flush=True)
    # With several GPUs each rank reads only its own share; otherwise every GPU would get the same
    # samples and the effective batch would not actually grow
    sampler = None
    if world_size() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train, num_replicas=world_size(), rank=rank(), shuffle=True, drop_last=True)
    return torch.utils.data.DataLoader(
        train, batch_size=a.proteins, shuffle=(sampler is None), sampler=sampler,
        num_workers=a.workers, collate_fn=collate, drop_last=True,
        persistent_workers=a.workers > 0), train


def _guard_dataset(ds) -> None:
    """Resample instead of dying when one record cannot be featurised.

    SimpleFold's __getitem__ guards cropping and featurising but not
    `extract_sequence_from_tokens`, so a single odd residue raises inside a dataloader worker and
    takes the whole run with it -- hours in, because the cropper only reaches that record on some
    epochs. A run of this length has to survive one bad record.
    """
    raw = ds.__class__.__getitem__

    def safe(self, idx: int, _depth: int = 0):
        try:
            return raw(self, idx)
        except Exception as e:  # noqa: BLE001 - any record-level failure should not kill the run
            if _depth >= 8:
                raise
            if _depth == 0:
                print(f"[dataset] skipping idx={idx}: {type(e).__name__}: {e}", flush=True)
            import random as _r

            return safe(self, _r.randint(0, self.num_samples - 1), _depth + 1)

    ds.__class__.__getitem__ = safe


def load_head(a: argparse.Namespace, device):
    import hydra
    import omegaconf
    from inference import get_config_path

    cfg = omegaconf.OmegaConf.load(get_config_path(f"configs/model/architecture/foldingdit_{a.model[11:]}.yaml"))
    model = hydra.utils.instantiate(cfg)
    model.load_state_dict(torch.load(Path(a.ckpt_dir) / f"{a.model}.ckpt", map_location="cpu",
                                     weights_only=False), strict=True)
    return model.to(device)


class Ema:
    """SimpleFold keeps an EMA of all weights (decay 0.999) in both training phases.

    An fp32 shadow is kept only for tensors that change: trainable parameters + floating-point
    buffers. The EMA of a frozen parameter is always the parameter itself, so on saving its value
    is taken straight from the model; keys and shapes of the exported ema_*.ckpt are unchanged.
    With --freeze-flow this saves 2.86 billion parameters x fp32 = about 11.5 GB per GPU.
    The only difference from keeping a shadow of everything: that version also applied
    0.999*v + 0.001*v to frozen parameters, and fp32 rounding moved them about 6e-8 away from the
    frozen value; here they equal the frozen value exactly. Trainable parameters are bit-identical;
    without freezing every parameter is trainable and the behaviour is bit-identical to before.
    Must be constructed after requires_grad has been set.
    """

    def __init__(self, model, decay: float) -> None:
        self.decay = decay
        self.model = model
        names = dict(model.named_parameters(remove_duplicate=False))
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point and (k not in names or names[k].requires_grad)}

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def state_dict(self) -> dict:
        full = {k: v.detach().float() for k, v in self.model.state_dict().items() if v.dtype.is_floating_point}
        full.update(self.shadow)
        return full


def log_mem(tag: str) -> None:
    """Memory breakdown during start-up (printed by the main rank): current usage after each load."""
    if is_main() and torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[mem] {tag}: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)


def logit_normal_sample(n: int, m: float, s: float) -> torch.Tensor:
    """Copied from model/simplefold.py; importing that module would pull in fairscale/lightning."""
    return torch.sigmoid(torch.randn(n) * s + m)


def lddt_weight(t: torch.Tensor, phase: str) -> torch.Tensor:
    """alpha(t): 1 in pretraining, 1 + 8*ReLU(t - 0.5) in finetuning (SimpleFold paper, sec. 4.1)."""
    return torch.ones_like(t) if phase == "pretrain" else 1.0 + 8.0 * F.relu(t - 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True,
                    choices=["onehot", "esm2", "routec", "routec_raw"])
    ap.add_argument("--cond-ckpt", default=None,
                    help="stage-1 export directory (stage1/export_stage1.py) for --cond routec / routec_raw")
    ap.add_argument("--data", type=Path, required=True, help="tokenized dir (tokens/, records/, manifest.json)")
    ap.add_argument("--targets", type=Path, required=True, help="targets dir (structures/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="simplefold_100M")
    ap.add_argument("--ckpt-dir", required=True,
                    help="directory holding the public SimpleFold weights (<model>.ckpt)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--proteins", type=int, default=32, help="distinct proteins per step (B_p)")
    ap.add_argument("--copies", type=int, default=8, help="timestep copies per protein (B_c)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimiser step; effective batch = proteins*copies*grad_accum. "
                         "SimpleFold finetunes at 32*8=256 across GPUs; one GPU holds far less because the "
                         "smooth-LDDT loss materialises a (B, N_atom, N_atom) distance matrix.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-atoms", type=int, default=2304)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--llm-lr", type=float, default=2e-5, help="learning rate of the trunk with --train-llm")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--phase", default="finetune", choices=["pretrain", "finetune"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from a last.pt. The learning rate is a constant 1e-4 after warmup "
                         "(no decay), so extending the step count does not conflict with the schedule.")
    ap.add_argument("--train-llm", action="store_true",
                    help="also unfreeze the LLM inside the condition (not used by Calpha). Qwen is "
                         "FSDP2-sharded; the structure head keeps a full copy per GPU and its gradients "
                         "are synchronised by hand in sync_grads.")
    ap.add_argument("--cond-bf16", action="store_true",
                    help="bf16 weights for the condition model (Qwen / ESM2). They are frozen and only run "
                         "forward, and that forward is under bf16 autocast anyway; once the 3B head fills "
                         "the GPU this is the cheapest memory saving.")
    ap.add_argument("--init-head", type=Path, default=None,
                    help="initialise from an existing structure head (e.g. the flow_head.ckpt of a stage-1 "
                         "export). Loaded after retarget_condition, so the condition front-end shapes must "
                         "match (cannot be combined with --reinit-cond).")
    ap.add_argument("--reinit-cond", action="store_true",
                    help="rebuild esm_s_combine/esm_s_proj even if the shape fits (fair control for --cond esm2)")
    ap.add_argument("--freeze-flow", action="store_true",
                    help="freeze the body of the structure head and train only the condition front-end "
                         "(esm_s_proj / esm_s_combine)")
    ap.add_argument("--cond-depth", type=int, default=1,
                    help="number of layers of the condition projection (default 1 = the official "
                         "structure). >1 deepens it, replacing only .proj")
    ap.add_argument("--cond-width", type=int, default=0,
                    help="width of the intermediate layers when deepening; 0 = same as the output width")
    a = ap.parse_args()

    import conditioning as sc
    from model.flow import LinearPath
    from processor.protein_processor import ProteinDataProcessor
    from utils.boltz_utils import center_random_augmentation, weighted_rigid_align

    torch.manual_seed(a.seed)
    device = setup_distributed()
    a.out.mkdir(parents=True, exist_ok=True)
    if is_main():
        (a.out / "args.json").write_text(json.dumps(vars(a), indent=1, default=str))

    loader, _ = build_loader(a)
    model = load_head(a, device)
    log_mem("after loading the structure head")
    provider = sc.make_provider(a.cond, a.cond_ckpt, str(device), bf16=a.cond_bf16)
    log_mem("after loading the condition model")
    sc.retarget_condition(model, provider.n_layers, provider.dim, reinit=a.reinit_cond)
    if a.init_head is not None:
        assert not a.reinit_cond, "--init-head cannot be combined with --reinit-cond"
        sd = torch.load(a.init_head, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        res = model.load_state_dict(sd, strict=False)
        own = dict(model.named_parameters())
        bad_shape = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
        lost = [k for k in res.missing_keys if k in own]
        print(f"[init-head] {a.init_head}: loaded {len(sd) - len(res.unexpected_keys)} tensors; "
              f"missing parameters {len(lost)}, unexpected {len(res.unexpected_keys)}, "
              f"shape mismatches {len(bad_shape)}", flush=True)
        assert not lost and not res.unexpected_keys and not bad_shape, \
            (f"initial structure head does not match the model: missing {lost[:3]} "
             f"unexpected {res.unexpected_keys[:3]} shape {bad_shape[:3]}")
    model = model.to(device).train()
    core = model          # EMA / saving / optimizer always use the unwrapped module, avoiding DDP's "module." prefix

    # --freeze-flow: freeze the **body** of the structure head (atom encoder / residue trunk /
    # atom decoder) and train only the condition front-end. An earlier round retrained the whole
    # head in every group, and retraining has a cost of its own: with the same direct ESM2
    # condition the official weights scored TM 0.842 and our retrained head 0.822 (-0.020), and
    # the >600-residue bucket dropped from 0.821 to 0.598. Freezing the body avoids both.
    #
    # The condition front-end (esm_s_combine / esm_s_proj) **must stay trainable**:
    # retarget_condition has just rebuilt it for the new (n_layers, dim), so it is randomly
    # initialised, and freezing it would feed the head a random projection.
    # --cond-depth: deepen the condition projection. The official esm_s_proj is a single
    # Linear(4096 -> 2048) + LayerNorm + SiLU (8.4M parameters).
    # The 10k -> 20k step comparison showed that **the amount of training is not the
    # bottleneck** (doubling it changed the overall score by +0.0001, all seven length buckets
    # crossing zero), so if more is to be gained, capacity is one candidate.
    #
    # Only .proj is replaced; null_token and token_drop are kept -- they are the classifier-free
    # guidance interface, and replacing them would break condition dropping at sampling time.
    if a.cond_depth > 1:
        import torch.nn as nn
        old_proj = core.esm_s_proj.proj
        d_in = old_proj[0].in_features
        d_out = old_proj[0].out_features
        d_mid = a.cond_width or d_out
        layers: list = []
        prev = d_in
        for i in range(a.cond_depth - 1):
            layers += [nn.Linear(prev, d_mid), nn.LayerNorm(d_mid), nn.SiLU()]
            prev = d_mid
        layers += [nn.Linear(prev, d_out), nn.LayerNorm(d_out), nn.SiLU()]
        core.esm_s_proj.proj = nn.Sequential(*layers).to(device)
        if is_main():
            n_old = sum(q.numel() for q in old_proj.parameters())
            n_new = sum(q.numel() for q in core.esm_s_proj.proj.parameters())
            print(f"condition projection deepened: {a.cond_depth} layers, intermediate width {d_mid}, "
                  f"parameters {n_old/1e6:.1f}M -> {n_new/1e6:.1f}M", flush=True)

    if a.freeze_flow:
        keep = ("esm_s_proj", "esm_s_combine")
        froze = kept = 0
        for n, q in core.named_parameters():
            if n.startswith(keep):
                kept += q.numel()
            else:
                q.requires_grad_(False)
                froze += q.numel()
        if is_main():
            print(f"structure head frozen: {froze/1e6:.0f}M frozen, {kept/1e6:.1f}M left trainable "
                  f"(condition front-end {keep})", flush=True)
    # A provider may own trainable parameters of its own. They live in the provider, not in the
    # head's forward, so DDP cannot track them; in that case gradients are synchronised with a
    # manual all_reduce instead (see sync_grads below). None of the providers in this release
    # has any, so `extra` is always empty here.
    extra = list(provider.trainable_parameters()) if hasattr(provider, "trainable_parameters") else []
    if extra:
        provider.train()
    llm_params: list = []
    sync_params: list = []     # parameters that need a manual all_reduce (those FSDP does not manage)
    if a.train_llm:
        if not hasattr(provider, "unfreeze_lm"):
            raise SystemExit("--train-llm needs a provider with unfreeze_lm (routec / routec_raw)")
        llm_params = provider.unfreeze_lm(shard=world_size() > 1)
        # The embedding is excluded from FSDP (see unfreeze_lm): it still belongs to llm_params
        # (trunk learning rate), but FSDP does not handle its gradient, so it joins the manually
        # synchronised group. It must not be put into `extra` -- it would then sit in two
        # optimizer parameter groups at once and AdamW raises an error.
        if world_size() > 1 and hasattr(provider, "unsharded_lm_params"):
            sync_params += provider.unsharded_lm_params()
        if is_main():
            print(f"LLM unfrozen and sharded: {sum(q.numel() for q in llm_params)/1e9:.2f}B parameters", flush=True)
    use_ddp = world_size() > 1 and not extra
    if use_ddp:
        # find_unused_parameters=True is required: part of the 3B head's parameters receive no
        # gradient in the forward of some batches (an 8-GPU smoke test raised "Expected to have
        # finished reduction..."). The cost is one extra autograd graph traversal per step,
        # negligible against a 7 s step.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=True)
    if is_main():
        print(f"condition={a.cond} reinit={a.reinit_cond} n_layers={provider.n_layers} dim={provider.dim}  "
              f"head params={sum(p.numel() for p in core.parameters())/1e6:.0f}M  world={world_size()}  "
              f"effective batch={a.proteins * a.copies * a.grad_accum * world_size()}", flush=True)

    # `multiplicity` makes the processor repeat each protein `copies` times with independent
    # augmentation; the condition is computed once per protein and repeated the same way.
    processor = ProteinDataProcessor(device=device, scale=16.0, ref_scale=5.0,
                                     multiplicity=a.copies, inference_multiplicity=1, backend="torch")
    path = LinearPath()
    # The LLM gets a smaller learning rate: the structure head is fine-tuned from pretrained
    # weights and the front-end is learned from scratch, whereas the 9B trunk only needs
    # fine-tuning; the same 1e-4 would wreck it (stage 1 uses 2e-5).
    head_params = [q for q in core.parameters() if q.requires_grad]
    groups = [{"params": head_params + extra, "lr": a.lr, "base_lr": a.lr}]
    if llm_params:
        # **Sharded and plain parameters must go into different parameter groups.** AdamW's
        # batched update operators (_foreach_lerp_ etc.) do not accept DTensors and plain tensors
        # in one group and raise "got mixed torch.Tensor and DTensor" (seen in a smoke test).
        # Only the transformer layers are sharded; embedding / final norm / lm_head stay plain.
        sh = [q for q in llm_params if hasattr(q, "device_mesh")]
        pl = [q for q in llm_params if not hasattr(q, "device_mesh")]
        for part in (sh, pl):
            if part:
                groups.append({"params": part, "lr": a.llm_lr, "base_lr": a.llm_lr})
        if is_main():
            print(f"LLM parameter groups: {len(sh)} sharded / {len(pl)} unsharded", flush=True)
    # Self-check: one parameter must not fall into two groups (AdamW would raise, but it is
    # better to say which one up front than to wait for that).
    seen: dict[int, str] = {}
    for gi, g in enumerate(groups):
        for q in g["params"]:
            if id(q) in seen:
                raise SystemExit(f"parameter appears in both group {seen[id(q)]} and group {gi}, "
                                 f"shape {tuple(q.shape)}")
            seen[id(q)] = str(gi)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.0)
    ema = Ema(core, a.ema)
    log_mem(f"after building optimizer and EMA (EMA shadow: {sum(v.numel() for v in ema.shadow.values()) / 1e6:.1f}M values)")
    metrics = (a.out / "metrics.jsonl").open("a") if is_main() else None

    step, t0, running, micro = 0, time.time(), [], 0

    def save_all(step_):
        if is_main():
            blob = {"step": step_, "model": core.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict()}
            if extra:
                blob["projector"] = {k: v.detach().cpu() for k, v in provider.state_dict().items()
                                     if k.startswith(("proj.", "layer_logits"))}
            torch.save(blob, a.out / "last.pt")
            torch.save(ema.state_dict(), a.out / f"ema_{step_}.ckpt")
            if extra:
                torch.save(blob["projector"], a.out / f"projector_{step_}.ckpt")
        # The other ranks have to wait while rank 0 saves: otherwise they move on and their next
        # collective no longer lines up with the rank that is still writing.
        if world_size() > 1:
            torch.distributed.barrier()
    if a.resume is not None and a.resume.exists():
        blob = torch.load(a.resume, map_location="cpu", weights_only=False)
        core.load_state_dict(blob["model"])
        ema.shadow = {k: blob["ema"][k].to(device) for k in ema.shadow}  # older checkpoints store the full set; take only what needs a shadow
        opt.load_state_dict(blob["opt"])
        if extra and "projector" in blob:
            provider.load_state_dict(blob["projector"], strict=False)
        step = int(blob["step"])
        if is_main():
            print(f"resumed from {a.resume} at step {step}"
                  f"{' (with projector)' if extra and 'projector' in blob else ''}", flush=True)
        # Note: the data stream does not continue from where the checkpoint left off; it is
        # reshuffled and starts over. These runs go over the same pool several times and the
        # sample order is reshuffled every epoch anyway, so this does not change what is being
        # measured -- but it is **not a bit-reproducible resume**.
        del blob
    opt.zero_grad(set_to_none=True)
    while step < a.steps:
        for batch in loader:
            if step >= a.steps:
                break
            seqs = batch["aa_seq"]
            batch = processor.preprocess_training(batch)  # no esm_model -> leaves esm_s unset
            L = batch["res_type"].shape[1]
            cond = provider(seqs, L).repeat_interleave(a.copies, dim=0)
            batch["esm_s"] = cond

            n = batch["coords"].shape[0]
            t = 0.98 * logit_normal_sample(n=n, m=0.8, s=1.7) + 0.02 * torch.rand(n)
            t = (t.to(device) * (1 - 2e-4) + 1e-4)
            noise = torch.randn_like(batch["coords"])
            _, y_t, v_t = path.interpolant(t, noise, batch["coords"])

            out = model(y_t, t, batch)
            resolved = batch["atom_resolved_mask"].float()
            w = y_t.new_ones(y_t.shape[:2])
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                vd = out["predict_velocity"].detach().float()
                denoised = y_t + vd * (1.0 - t[:, None, None])
                aligned = weighted_rigid_align(batch["coords"].detach().float(), denoised,
                                               w.float(), mask=resolved)
                _, _, target = path.interpolant(t, noise, aligned)

            mse = F.mse_loss(out["predict_velocity"], target, reduction="none")
            mse = (mse * (resolved * w)[..., None]).sum(1) / (resolved.sum(1, keepdim=True) * 3 + 1e-6)
            mse = mse.mean()

            denoised = y_t + out["predict_velocity"] * (1.0 - t[:, None, None])
            dc = center_random_augmentation(denoised, batch["atom_pad_mask"], augmentation=False,
                                            centering=True) * processor.scale
            tc = center_random_augmentation(batch["coords"], batch["atom_pad_mask"], augmentation=False,
                                            centering=True) * processor.scale
            lddt = _smooth_lddt(dc, tc, resolved, lddt_weight(t, a.phase))
            loss = mse + lddt

            (loss / a.grad_accum).backward()
            micro += 1
            if micro < a.grad_accum:
                continue
            micro = 0
            if world_size() > 1 and (extra or sync_params):
                # Synchronise only what FSDP does not manage: the structure head, provider
                # parameters, and the unsharded embedding. The Qwen layers under fully_shard are
                # reduced by FSDP itself; a second all_reduce would give wrong values.
                sync_grads(list(core.parameters()) + extra + sync_params)
            # Clipping is done in two passes: FSDP-sharded parameters are DTensors, and computing
            # one norm together with plain tensors raises
            # "aten._foreach_norm.Scalar: got mixed torch.Tensor and DTensor" (seen in a smoke test).
            # The price of clipping separately is that each side is cut at 10.0 on its own rather
            # than on the global norm; the two sides have different learning rates anyway, so
            # per-group clipping is closer to the intent.
            plain = list(core.parameters()) + extra + sync_params
            gnorm = torch.nn.utils.clip_grad_norm_(plain, 10.0)
            if llm_params:
                sharded = [q for q in llm_params if hasattr(q, "device_mesh")]
                rest = [q for q in llm_params if not hasattr(q, "device_mesh")
                        and all(q is not r for r in plain)]
                if sharded:
                    torch.nn.utils.clip_grad_norm_(sharded, 10.0)
                if rest:
                    torch.nn.utils.clip_grad_norm_(rest, 10.0)
            warm = min(1.0, (step + 1) / max(a.warmup, 1))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * warm
            opt.step()
            opt.zero_grad(set_to_none=True)
            ema.update(core)
            step += 1
            running.append((loss.item(), mse.item(), lddt.item(), float(gnorm)))

            if step % a.log_every == 0:
                # Memory: peak allocated within this logging interval, memory reserved by the
                # caching allocator, and device memory in use (about the nvidia-smi reading);
                # maximum over all GPUs
                free, total = torch.cuda.mem_get_info()
                mem = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved(), total - free],
                                   device=device, dtype=torch.float64) / 2**30
                if world_size() > 1:
                    torch.distributed.all_reduce(mem, op=torch.distributed.ReduceOp.MAX)
                torch.cuda.reset_peak_memory_stats()
            if is_main() and step % a.log_every == 0:
                m = [sum(c) / len(running) for c in zip(*running)]
                rec = {"step": step, "loss": round(m[0], 5), "mse": round(m[1], 5),
                       "lddt": round(m[2], 5), "gnorm": round(m[3], 3),
                       "lr": opt.param_groups[0]["lr"], "sec_per_step": round((time.time() - t0) / step, 3),
                       "mem_peak_gb": round(mem[0].item(), 1), "mem_reserved_gb": round(mem[1].item(), 1),
                       "mem_device_used_gb": round(mem[2].item(), 1)}
                metrics.write(json.dumps(rec) + "\n")
                metrics.flush()
                print(json.dumps(rec), flush=True)
                running = []
            if step % a.save_every == 0 or step == a.steps:
                save_all(step)
    if metrics is not None:
        metrics.close()
    if world_size() > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if is_main():
        print("done", flush=True)


def _smooth_lddt(pred, true, mask, alpha):
    """SimpleFold's smooth LDDT loss (model/simplefold.py), with the alpha(t) weighting."""
    dp = torch.cdist(pred, pred)
    dt = torch.cdist(true, true)
    err = (dp - dt).abs()
    eps = sum(torch.sigmoid(c - err) for c in (0.5, 1.0, 2.0, 4.0)) / 4.0
    near = (dt < 15.0).float()
    m = mask[:, :, None] * mask[:, None, :] * near
    m = m * (1 - torch.eye(m.shape[1], device=m.device))[None]
    lddt = (eps * m).sum((1, 2)) / (m.sum((1, 2)) + 1e-6)
    return ((1.0 - lddt) * alpha).mean()


if __name__ == "__main__":
    main()
