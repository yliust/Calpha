#!/usr/bin/env bash
# Head-only control ("C0"): fine-tune the whole SimpleFold-3B head on its native condition
# (all 37 ESM2-3B layers) with the same data, steps and optimizer settings as stage 2 phase a.
# No language model is involved. The condition front-end is re-initialised (--reinit-cond), as in
# stage 2, and the head is NOT frozen.
#
#   scripts/run_head_only_control.sh
set -euo pipefail

# ---- paths: edit these or export them before calling ---------------------------------------
SIMPLEFOLD_ROOT=${SIMPLEFOLD_ROOT:?set SIMPLEFOLD_ROOT to the apple/ml-simplefold checkout (commit 94e2cd7)}
CALPHA_DATA=${CALPHA_DATA:?set CALPHA_DATA to the directory holding the training data}
OUT=${OUT:-runs/head_only_control}

STRUCT_TOKENS=${STRUCT_TOKENS:-$CALPHA_DATA/sf/tokens_pdb}
STRUCT_TARGETS=${STRUCT_TARGETS:-$CALPHA_DATA/sf/targets_pdb}
SF_WEIGHTS=${SF_WEIGHTS:-$SIMPLEFOLD_ROOT/weights}             # holds simplefold_3B.ckpt (public weights)

GPUS=${GPUS:-0,1,2,3,4,5,6,7}     # the run used 8 x H200 (141 GB)
STEPS=${STEPS:-10000}
TORCHRUN=${TORCHRUN:-torchrun}
# ---------------------------------------------------------------------------------------------

HERE=$(cd "$(dirname "$0")/.." && pwd)
NG=$(awk -F, '{print NF}' <<<"$GPUS")
# Effective batch = GPUs x proteins x copies x grad-accum = 8 x 1 x 8 x 1 = 64.
PROTEINS=1; COPIES=8; ACCUM=1
EFF=$((NG * PROTEINS * COPIES * ACCUM))
[ "$EFF" -ne 64 ] && echo "warning: effective batch is $EFF, the control was trained with 64" >&2

export SIMPLEFOLD_SRC=$SIMPLEFOLD_ROOT/src/simplefold
export TORCH_HOME=${TORCH_HOME:-$CALPHA_DATA/torch_home}    # torch hub cache holding ESM2-3B
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"
"$TORCHRUN" --nproc_per_node="$NG" --master_port="${PORT:-29537}" \
  "$HERE/stage2/train_stage2.py" \
  --cond esm2 --cond-bf16 --reinit-cond \
  --model simplefold_3B --ckpt-dir "$SF_WEIGHTS" \
  --data "$STRUCT_TOKENS" --targets "$STRUCT_TARGETS" \
  --out "$OUT" --steps "$STEPS" \
  --proteins "$PROTEINS" --copies "$COPIES" --grad-accum "$ACCUM" \
  --max-tokens 256 --max-atoms 2304 \
  --lr 1e-4 --warmup 500 --ema 0.999 --phase finetune \
  --workers 6 --save-every 2500 --log-every 20 --seed 0 \
  >> "$OUT/train.log" 2>&1
echo "done: $OUT/ema_${STEPS}.ckpt"
