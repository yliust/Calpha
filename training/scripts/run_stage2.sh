#!/usr/bin/env bash
# Stage 2: train only the conditioning front-end (esm_s_combine + esm_s_proj, 8.4M parameters) of
# SimpleFold-3B on hidden states of the frozen stage-1 language model. Language model, ESM2 and
# the rest of the 3B head are frozen.
#
# Schedule as run for the released front-end (EMA at step 30000):
#   phase a   steps     0 -> 10000   crop 256 residues / 2304 atoms
#   phase b   steps 10000 -> 20000   crop 256 residues / 2304 atoms   (resumed from a)
#   phase c   steps 20000 -> 30000   crop 512 residues / 4608 atoms   (resumed from b)
# Each phase wrote to its own output directory and the next one resumed from its last.pt.
#
#   scripts/run_stage2.sh a|b|c|all
#
# Note: a resume restores model, EMA, optimizer and step, but the data loader starts over with a
# fresh shuffle, so the three phases are not bit-identical to one uninterrupted 30000-step run.
set -euo pipefail

# ---- paths: edit these or export them before calling ---------------------------------------
SIMPLEFOLD_ROOT=${SIMPLEFOLD_ROOT:?set SIMPLEFOLD_ROOT to the apple/ml-simplefold checkout (commit 94e2cd7)}
CALPHA_DATA=${CALPHA_DATA:?set CALPHA_DATA to the directory holding the training data}
STAGE1_EXPORT=${STAGE1_EXPORT:?set STAGE1_EXPORT to the output directory of stage1/export_stage1.py}
RUNS=${RUNS:-runs/stage2}

STRUCT_TOKENS=${STRUCT_TOKENS:-$CALPHA_DATA/sf/tokens_pdb}
STRUCT_TARGETS=${STRUCT_TARGETS:-$CALPHA_DATA/sf/targets_pdb}
SF_WEIGHTS=${SF_WEIGHTS:-$SIMPLEFOLD_ROOT/weights}             # holds simplefold_3B.ckpt (public weights)

GPUS=${GPUS:-0,1,2,3,4,5,6,7}     # the runs used 8 x H200 (141 GB)
TORCHRUN=${TORCHRUN:-torchrun}
# ---------------------------------------------------------------------------------------------

PHASE=${1:?usage: run_stage2.sh a|b|c|all}
HERE=$(cd "$(dirname "$0")/.." && pwd)
NG=$(awk -F, '{print NF}' <<<"$GPUS")
# Effective batch = GPUs x proteins x copies x grad-accum = 8 x 1 x 8 x 1 = 64.
PROTEINS=1; COPIES=8; ACCUM=1
EFF=$((NG * PROTEINS * COPIES * ACCUM))
[ "$EFF" -ne 64 ] && echo "warning: effective batch is $EFF, the released model was trained with 64" >&2

export SIMPLEFOLD_SRC=$SIMPLEFOLD_ROOT/src/simplefold
export TORCH_HOME=${TORCH_HOME:-$CALPHA_DATA/torch_home}    # torch hub cache holding ESM2-3B
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPUS
# the 3B head leaves little headroom on a 141 GB GPU; fragmentation turns directly into OOM
export PYTORCH_ALLOC_CONF=expandable_segments:True

train() {   # train <out_dir> <total_steps> <max_tokens> <max_atoms> [resume_ckpt]
  local out=$1 steps=$2 max_tokens=$3 max_atoms=$4 resume=${5:-}
  mkdir -p "$out"
  local args=(
    --cond routec_raw --cond-ckpt "$STAGE1_EXPORT" --cond-bf16
    --reinit-cond --freeze-flow
    --model simplefold_3B --ckpt-dir "$SF_WEIGHTS"
    --data "$STRUCT_TOKENS" --targets "$STRUCT_TARGETS"
    --out "$out" --steps "$steps"
    --proteins "$PROTEINS" --copies "$COPIES" --grad-accum "$ACCUM"
    --max-tokens "$max_tokens" --max-atoms "$max_atoms"
    --lr 1e-4 --warmup 500 --ema 0.999 --phase finetune
    --workers 6 --save-every 2500 --log-every 20 --seed 0
  )
  [ -n "$resume" ] && args+=(--resume "$resume")
  echo "[$(date '+%F %T')] stage 2 -> $out (steps $steps, crop $max_tokens/$max_atoms, resume: ${resume:-none})"
  "$TORCHRUN" --nproc_per_node="$NG" --master_port="${PORT:-29537}" \
    "$HERE/stage2/train_stage2.py" "${args[@]}" >> "$out/train.log" 2>&1
}

phase_a() { train "$RUNS/a_crop256_10k" 10000 256 2304; }
phase_b() { train "$RUNS/b_crop256_20k" 20000 256 2304 "$RUNS/a_crop256_10k/last.pt"; }
phase_c() { train "$RUNS/c_crop512_30k" 30000 512 4608 "$RUNS/b_crop256_20k/last.pt"; }

case "$PHASE" in
  a) phase_a ;;
  b) phase_b ;;
  c) phase_c ;;
  all) phase_a; phase_b; phase_c ;;
  *) echo "unknown phase $PHASE (a|b|c|all)" >&2; exit 1 ;;
esac
echo "done. Released front-end = esm_s_combine / esm_s_proj.* of $RUNS/c_crop512_30k/ema_30000.ckpt"
