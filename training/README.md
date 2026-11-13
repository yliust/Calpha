# Calpha training code

This directory holds the code that trained Calpha, in two stages, plus the head-only control,
the vocabulary extension, the data builders and the export step between the stages.

> **Status of this code.** It is the code as used for the paper, with hard-coded paths removed,
> comments and messages translated to English, files renamed, and code paths that Calpha does not
> use taken out (listed under [What was removed](#what-was-removed)). **It has not been re-run end
> to end after this cleanup.** What was checked: every file compiles, every module imports on CPU
> in the training environment, and an AST-level comparison against the original files shows no
> change outside paths, messages, argument defaults that were personal paths, and the removals
> listed below. Expect to fix small things when you run it. The training data is not included.

## What each stage trains

```
Qwen3.5-9B  +  frozen ESM2-3B features (added to the amino-acid token embeddings via an adapter)
    -> hidden states of 9 layers at the residue positions
    -> conditioning front-end (8.4M parameters)
    -> frozen SimpleFold-3B flow-matching head -> all-atom coordinates
```

### Stage 1: joint training (`stage1/train_stage1.py`)

| Component | Initialised from | Trained |
|---|---|---|
| Qwen3.5-9B trunk, embeddings, LM head | public Qwen3.5-9B weights with the vocabulary extended (see below) | yes, all parameters |
| ESM2-3B (`esm2_t36_3B_UR50D`, fair-esm) | public weights, bf16 | no (the only frozen part) |
| ESM adapter (layer mix, LayerNorm, 2-layer MLP, per-dimension gate) | new; gate initialised to 0 | yes |
| Condition projection (softmax mix of hidden states 0,4,...,32, LayerNorm, Linear, gate) | new; gate initialised to 0 | yes |
| SimpleFold-100M flow head | public `simplefold_100M.ckpt`; its condition input modules are rebuilt for the new condition | yes |

Each optimizer step accumulates 8 micro-steps on each of 8 GPUs. A micro-step takes one
language micro-batch (at most 8192 padded tokens) and one structure micro-batch (2 proteins x 8
noise/timestep copies, cropped to 256 residues), and back-propagates

```
L = L_LM + L_trigger + 28.9 * (L_flow_MSE + alpha(t) * L_smooth_lDDT),   alpha(t) = 1 + 8 * relu(t - 0.5)
```

`L_LM` is next-token cross-entropy on the language mixture; `L_trigger` is the cross-entropy of
the assistant turn of the folding prompt, which teaches the model to answer `<structure>`. The
flow gradient reaches the trunk through the condition projection.

Run length is counted in tokens: `--total-tokens 2e9` counts the language-path target tokens
(labels that are not masked) summed over GPUs. The run behind the released model ended at step
6198.

Optimizer as run: AdamW (PyTorch default betas); trunk and adapter lr 2e-5 with 100 warmup
steps and cosine decay over counted tokens to 0.1x; token embeddings and LM head 1e-4, flow head
and condition projection 1e-4 (both constant after the same warmup); weight decay 0.1 on trunk
weight matrices only; gradient clipping at 1.0, applied separately to the FSDP-sharded parameters
and to the flow head + condition projection; seed 0. The trunk is trained with FSDP2 (bf16
parameters in the forward, fp32 master weights and gradient reduction).

Every 250 steps (and at step 0) the trainer folds 16 held-out proteins with the current weights
and logs TM-score, CA-lDDT, the fraction of physically plausible CA-CA distances and a normalised
radius of gyration (`probe_*` in `metrics.jsonl`).

### Export between the stages (`stage1/export_stage1.py`)

Stage 1 saves one checkpoint file per GPU. `export_stage1.py` merges them into `merged.pt`
(full state dict), `flow_head.ckpt` and `meta.json`. Stage 2 reads this directory.
`meta.json` records the path of the stage-1 base model, which must still exist when stage 2
runs (tokenizer and model config are loaded from it). `stage1/export_lm_hf.py` optionally writes
the language model alone as a Hugging Face directory (for text benchmarks).

### Stage 2: conditioning front-end (`stage2/train_stage2.py`)

| Component | Initialised from | Trained |
|---|---|---|
| Stage-1 language model + adapter, ESM2-3B | stage-1 export, cast to bf16 | no |
| SimpleFold-3B atom encoder, residue trunk, atom decoder | public `simplefold_3B.ckpt` | no (`--freeze-flow`) |
| `esm_s_combine` (9 layer weights) + `esm_s_proj` (Linear 4096->2048, LayerNorm, SiLU, plus SimpleFold's null token): 8.4M parameters | re-initialised (`--reinit-cond`) | yes |

The condition is the raw hidden states of the stage-1 trunk at layers 0,4,...,32 (0 = embedding
output) at the residue positions of a fixed folding prompt (`--cond routec_raw`). The stage-1
condition projection and the 100M head are not used. Loss is SimpleFold's fine-tuning loss (flow
MSE + smooth lDDT with the same `alpha(t)`).

As run: effective batch 64 (8 GPUs x 1 protein x 8 copies), AdamW lr 1e-4 without weight decay,
500 warmup steps then constant, gradient clipping at 10, EMA 0.999, seed 0. Schedule:

| Phase | Steps | Crop | Started from | Wall-clock (8x H200) |
|---|---|---|---|---|
| a | 0 -> 10,000 | 256 residues / 2304 atoms | public SimpleFold-3B | about 2.5 h |
| b | 10,000 -> 20,000 | 256 residues / 2304 atoms | `last.pt` of phase a | about 2.3 h |
| c | 20,000 -> 30,000 | 512 residues / 4608 atoms | `last.pt` of phase b | about 6.3 h |

The released front-end is the EMA at step 30,000 (`ema_30000.ckpt` of phase c; the file holds
the whole head, of which only `esm_s_combine` and `esm_s_proj.*` differ from the public weights).

### Head-only control (`scripts/run_head_only_control.sh`)

Same trainer, data, batch and optimizer settings as stage 2 phase a, but with SimpleFold's native
condition (`--cond esm2`, all 37 ESM2-3B layers, no language model) and the **whole 3B head
trained** (no `--freeze-flow`), 10,000 steps at crop 256. The condition front-end is
re-initialised here too.

## Layout

| Path | Purpose |
|---|---|
| `stage1/train_stage1.py` | stage-1 trainer (dual-head model, structure data stream, probe, checkpoints) |
| `stage1/export_stage1.py` | merge per-GPU checkpoints into the export directory used by stage 2 |
| `stage1/export_lm_hf.py` | extract the language model as a Hugging Face directory (optional) |
| `stage2/train_stage2.py` | stage-2 trainer; also runs the head-only control |
| `common/conditioning.py` | condition providers for stage 2 and `retarget_condition` |
| `common/esm_adapter.py` | frozen ESM2 encoder, adapter, mapping of residues to token positions |
| `common/lm_data.py` | language data stream, chunked cross-entropy, FSDP helpers |
| `common/ca_metrics.py` | TM-score / lDDT / RMSD / GDT on CA atoms (numpy), used by the probe |
| `data/vocab.py` | protein token lists |
| `data/extend_vocab.py` | add the protein vocabulary to Qwen3.5-9B |
| `data/add_structure_token.py` | add the `<structure>` special token (stage-1 base model) |
| `data/build_mixture.py` | build the language-side mixture as jsonl shards |
| `data/pretokenize.py` | tokenize the shards into flat arrays |
| `data/simplefold_targets.py` | mmCIF files -> SimpleFold training targets and tokens |
| `scripts/run_stage1.sh`, `run_stage2.sh`, `run_head_only_control.sh` | example launchers carrying the hyperparameters as run |
| `requirements.txt` | export of the training environment |

Scripts are run as files (`python stage1/train_stage1.py ...`); each adds the sibling
directories it needs to `sys.path`. Some option values and output file names keep internal names
so that archived run configurations still apply: `--cond routec` / `routec_raw` ("route C" was the
internal name of the stage-1 dual-head model), `base_rc.json`, `route_c_hf.json`.

## Data and weights you need

**Nothing below is shipped in this repository**: not the training data, not the general-text
corpus, not PDB downloads, not any third-party weights.

Third-party weights and code:

- Qwen3.5-9B (weights and tokenizer).
- ESM2-3B, downloaded by fair-esm into the torch hub cache (`TORCH_HOME`).
- SimpleFold source, unmodified upstream `apple/ml-simplefold` at commit
  `94e2cd704cf7b86ef8e7cb27a1e77154c4e35b34`, and its public `simplefold_100M.ckpt` (stage 1),
  `simplefold_3B.ckpt` (stage 2, control) and `ccd.pkl` component dictionary (data building,
  probe).

### Structure data (stage 1 structure path, stage 2, control)

Built from PDB mmCIF files with `data/simplefold_targets.py`, which calls SimpleFold's own
processing pipeline with the Redis-backed component dictionary replaced by `ccd.pkl`:

```bash
export SIMPLEFOLD_SRC=<ml-simplefold>/src/simplefold SIMPLEFOLD_CCD=<ml-simplefold cache>/ccd.pkl
python data/simplefold_targets.py targets  --data-dir <mmcif_dir> --out-dir <targets_dir> --num-processes 32
python data/simplefold_targets.py tokenize --target-dir <targets_dir> --token-dir <tokens_dir>
python data/simplefold_targets.py filter   --token-dir <tokens_dir>
```

`targets` parses the asymmetric unit (not the biological assembly). `filter` keeps the records
that have at least one chain and in which every chain is a protein, and rewrites `manifest.json`
(the unfiltered manifest is kept as `manifest.all.json`). In our build the manifest went from
98,254 records (one per PDB entry) to 84,658. Training crops each sample to at most 256 (or 512)
residues. Not included here: the mmCIF files themselves and the list of PDB entries that went in
(see the paper for how the pool was selected).

### Language-side mixture (stage 1 language path)

`data/build_mixture.py` renders protein tasks and general text into chat-format jsonl shards;
`data/pretokenize.py` turns them into token arrays. As built for the released model
(`--task-mix vlm --format chat --general-frac 0.25`, 5 epochs, seed 0):

- protein tasks per chain and epoch: inverse folding (p=0.5), secondary structure (0.5),
  structure infilling (0.5), long-range contacts (0.3); chains are sampled per 30%-identity
  cluster (`ceil(n^0.67)` members per epoch) and cropped to 1024 residues;
- general text: 25% of the tokens, two thirds FineWeb-Edu (`sample/10BT`) and one third
  Mixture-of-Thoughts (`science` and `math` subsets, rendered with the Qwen chat template);
- size: about 415k protein samples / 410M protein tokens plus 137M general-text tokens per
  epoch, 2.74B tokens in total. Stage 1 stops at 2e9 *counted* tokens, which excludes
  amino-acid tokens that receive ESM2 features (they are inputs, not targets).

The script expects inputs that are **not built by anything in this directory**:

- `<root>/split/index.parquet`: columns `id`, `split` (`train` / `val`), `rep` (cluster
  representative);
- `<root>/features/parsed/<source>/*.parquet`: per chain `id`, `sequence`, `struct_tokens`
  (ESM3 structure-tokenizer codes, 4096-entry codebook, negative = unresolved), `dssp`
  (3-state string), `contacts` (flattened residue index pairs), `masked_frac`;
- `<root>/features/eval/foldbench334.parquet`: the same features for an evaluation set (used
  only to write validation shards);
- the FineWeb-Edu and Mixture-of-Thoughts parquet files (`FINEWEB_GLOB`, `MOT_GLOB`).

Parsing the chains, running the ESM3 structure tokenizer and DSSP, clustering and splitting were
done by other parts of our pipeline and are not part of this release. Note that the discrete
structure tokens are still part of the vocabulary and of the inverse-folding and infilling
tasks, although Calpha does not generate structures through them.

### Probe set (stage 1)

`--probe-eval` is a parquet with columns `id`, `sequence`, `coords` (flattened L x 5 x 3 backbone
array, CA at index 1), `length`, `release_date`. The trainer takes the first 16 rows with
`length <= 200` and `release_date >= 2020-05-01`. Not included. `--probe-every 0` turns the probe
off; because the probe draws from the same torch random stream as training, turning it off or
changing its interval changes the noise drawn afterwards.

## Commands, as run

All commands are run from this directory with `SIMPLEFOLD_SRC` set.

```bash
# 0. vocabulary: Qwen3.5-9B + 4,132 protein tokens (248,077 -> 252,209), then <structure> (252,210)
export QWEN_BASE=<Qwen3.5-9B directory>
(cd data && python - <<'EOF'
import os
from transformers import AutoTokenizer
from vocab import ALL_TOKENS
tok = AutoTokenizer.from_pretrained(os.environ["QWEN_BASE"])
tok.add_tokens(ALL_TOKENS)
tok.save_pretrained("../work/tokenizer_ext")
EOF
)
python data/extend_vocab.py --tokenizer work/tokenizer_ext --out work/base_ext
python data/add_structure_token.py --src work/base_ext --out work/base_model

# 1. language mixture (inputs described above)
QWEN_BASE=work/base_model FINEWEB_GLOB='<fineweb-edu>/sample/10BT/*.parquet' \
MOT_GLOB='<Mixture-of-Thoughts>/{}/*.parquet' \
python data/build_mixture.py --root <root> --out work/mixture \
    --task-mix vlm --format chat --general-frac 0.25 --epochs 5 --seed 0
python data/pretokenize.py --mixture work/mixture --workers 32

# 2. stage 1 (8 GPUs)
torchrun --nproc_per_node=8 stage1/train_stage1.py \
    --model work/base_model --lm-data work/mixture \
    --struct-tokens <tokens_dir> --struct-targets <targets_dir> \
    --flow-ckpt <simplefold weights>/simplefold_100M.ckpt --flow-model simplefold_100M \
    --probe-eval <probe.parquet> --ccd <ccd.pkl> --out runs/stage1 \
    --total-tokens 2e9 --inject-mode add --cond-layers 0,4,8,12,16,20,24,28,32 \
    --lambda-flow 28.9 --phase finetune \
    --lr 2e-5 --min-lr-ratio 0.1 --flow-lr 1e-4 --embed-lr 1e-4 \
    --weight-decay 0.1 --warmup-steps 100 --max-grad-norm 1.0 \
    --max-res 256 --max-len 4096 \
    --struct-per-step 2 --struct-copies 8 --micro-tokens 8192 --grad-accum 8 \
    --log-every 10 --save-every 250 --probe-every 250 --seed 0

# 3. export
python stage1/export_stage1.py --run runs/stage1 --out runs/stage1/export

# 4. stage 2 (8 GPUs); phase a shown, b and c add --resume and change --steps / the crop
torchrun --nproc_per_node=8 stage2/train_stage2.py \
    --cond routec_raw --cond-ckpt runs/stage1/export --cond-bf16 --reinit-cond --freeze-flow \
    --model simplefold_3B --ckpt-dir <simplefold weights> \
    --data <tokens_dir> --targets <targets_dir> --out runs/stage2/a_crop256_10k --steps 10000 \
    --proteins 1 --copies 8 --grad-accum 1 --max-tokens 256 --max-atoms 2304 \
    --lr 1e-4 --warmup 500 --ema 0.999 --phase finetune \
    --workers 6 --save-every 2500 --log-every 20 --seed 0
#    phase b: --out runs/stage2/b_crop256_20k --steps 20000 --resume runs/stage2/a_crop256_10k/last.pt
#    phase c: --out runs/stage2/c_crop512_30k --steps 30000 --max-tokens 512 --max-atoms 4608 \
#             --resume runs/stage2/b_crop256_20k/last.pt

# 5. head-only control: as phase a, with
#    --cond esm2 --cond-bf16 --reinit-cond     (no --cond-ckpt, no --freeze-flow)
```

The launchers in `scripts/` wrap steps 2, 4 and 5 with the same arguments and take their paths
from variables at the top (`SIMPLEFOLD_ROOT`, `CALPHA_DATA`, `STAGE1_EXPORT`, ...).
`scripts/run_stage1.sh` adds a restart loop and a stale-log watchdog; `SMOKE=1` runs a 9-step
test that exercises probing, saving and resuming.

In the original runs the tokenizer directory given to `extend_vocab.py` was the one
`build_mixture.py` writes (`<out>/tokenizer`) from an earlier mixture build; the snippet in step 0
performs the same three calls.

## Hardware and wall-clock

All runs used one node with 8x NVIDIA H200 (141 GB each).

- Stage 1: about 31 hours from first launch to the last step, including three interruptions for
  intermediate evaluation; steady-state about 17 s per optimizer step. Full-parameter training of
  the 9B trunk used about 127 GB per GPU.
- Stage 2: about 2.5 h, 2.3 h and 6.3 h for the three phases (table above).
- The 3B head leaves little memory headroom at 8 copies per GPU; one GPU cannot hold 8 proteins
  x 8 copies.

## Environment

`requirements.txt` is the frozen package list of the environment both stages ran in (Python
3.12, PyTorch 2.10.0 + CUDA 12.8). Three things are not plain PyPI installs; the header of
`requirements.txt` gives the commands:

- **transformers is a development build** (5.15.0.dev0, commit `dff4572d`), needed for
  Qwen3.5 support at the time of training. A later release may work but was not tested.
- `causal-conv1d` 1.6.2.post1 from the prebuilt wheel on its GitHub releases page.
- SimpleFold from source at the commit given above, imported through `SIMPLEFOLD_SRC`.

Qwen3.5's linear-attention layers use the `fla` (flash-linear-attention 0.5.2) kernels. The
environment pins `triton==3.7.1`: with triton 3.4-3.7.0 the gated chunk backward of `fla`
computes wrong gradients on Hopper GPUs (H100/H200), and `fla` refuses to run. This environment
did not contain the TileLang backend; if your `fla`/triton combination is rejected on Hopper
hardware, that backend is the alternative.

## Known caveats

- **Not re-run after cleanup** (see the note at the top).
- **One training run per configuration.** No configuration was repeated with another seed, so
  run-to-run variance is not measured.
- **Stage 2 resumes are not bit-reproducible.** `--resume` restores weights, EMA, optimizer and
  step count, but the data loader starts again with a fresh shuffle. The released front-end went
  through two such resumes.
- **Stage 1 was interrupted and resumed** from its checkpoints at steps 1500, 3000 and 4750
  (it was stopped for evaluation shortly after each of them; after step 1500 between 10 and 19
  steps had already been taken and were repeated). The trainer restores data cursors and
  random-number state on resume; we have not verified that a resumed run matches an
  uninterrupted one bit for bit.
- `sec_per_step` in the stage-1 `metrics.jsonl` divides the time since the *current* process
  started by the global step, so it is wrong after a resume.
- Stage-1 checkpoints only fit the number of GPUs they were written with (one file per rank).
- `ca_metrics.py` is a numpy re-implementation used for the in-training probe only; it is not
  the scorer used for the reported benchmark numbers.

## What was removed

Relative to the research code, without changing the training paths Calpha used:

- from the stage-2 trainer: a joint LoRA / text-understanding training branch and its options,
  and an unreachable block of old checkpoint code;
- from the condition providers: two providers Calpha does not use (hidden states of a Qwen that
  was not trained in stage 1; ESM features concatenated into Qwen's input through a trainable
  projector);
- from the language-data module: the earlier LM-only training loop and its options (pair
  attention bias, soft structure-token targets, bit-level losses, prefix noise, in-loop
  teacher-forced evaluation). Only the data stream, the plain chunked cross-entropy and the FSDP
  helper that stage 1 imports were kept;
- from the launchers: code snapshotting, GPU-availability waiting, evaluation and scoring steps,
  and the other experiment groups.

Kept although Calpha does not use them, because removing them was not a purely mechanical edit:
`--inject-mode replace` and `--no-esm` in stage 1; `--train-llm`, `--init-head`,
`--cond-depth/--cond-width`, `--cond onehot` and `--cond routec` in stage 2; the non-chat
formats and `--task-mix default` in `build_mixture.py`.
