#!/usr/bin/env bash
# Stage 1: joint training of Qwen3.5-9B + ESM2 adapter + SimpleFold-100M head.
# Hyperparameters are those of the run behind the released model (2e9 counted tokens, 6198 steps).
#
#   scripts/run_stage1.sh              # full run
#   SMOKE=1 scripts/run_stage1.sh      # 9-step end-to-end test (save / probe / resume paths)
#
# The trainer resumes from <OUT>/ckpt/latest by itself, so re-running this script (or the restart
# loop below) continues an interrupted run.
set -uo pipefail

# ---- paths: edit these or export them before calling ---------------------------------------
SIMPLEFOLD_ROOT=${SIMPLEFOLD_ROOT:?set SIMPLEFOLD_ROOT to the apple/ml-simplefold checkout (commit 94e2cd7)}
CALPHA_DATA=${CALPHA_DATA:?set CALPHA_DATA to the directory holding the training data}
OUT=${OUT:-runs/stage1}

BASE_MODEL=${BASE_MODEL:-$CALPHA_DATA/base_model}                # data/add_structure_token.py output
LM_DATA=${LM_DATA:-$CALPHA_DATA/mixture}                        # build_mixture.py + pretokenize.py output
STRUCT_TOKENS=${STRUCT_TOKENS:-$CALPHA_DATA/sf/tokens_pdb}      # simplefold_targets.py tokenize + filter
STRUCT_TARGETS=${STRUCT_TARGETS:-$CALPHA_DATA/sf/targets_pdb}   # simplefold_targets.py targets
FLOW_CKPT=${FLOW_CKPT:-$SIMPLEFOLD_ROOT/weights/simplefold_100M.ckpt}   # public SimpleFold-100M weights
CCD=${CCD:-$SIMPLEFOLD_ROOT/cache/ccd.pkl}                      # SimpleFold component dictionary
PROBE_EVAL=${PROBE_EVAL:-$CALPHA_DATA/eval/dev.parquet}         # small eval set for the in-training probe

NPROC=${NPROC:-8}                 # GPUs; the run used 8 x H200 (141 GB)
TOTAL_TOKENS=${TOTAL_TOKENS:-2e9}
TORCHRUN=${TORCHRUN:-torchrun}
MAX_RESTARTS=${MAX_RESTARTS:-10}  # automatic restarts after a crash
STALE_SECS=${STALE_SECS:-2700}    # restart if train.log has not changed for this long
# ---------------------------------------------------------------------------------------------

HERE=$(cd "$(dirname "$0")/.." && pwd)
export SIMPLEFOLD_SRC=$SIMPLEFOLD_ROOT/src/simplefold
export TORCH_HOME=${TORCH_HOME:-$CALPHA_DATA/torch_home}    # torch hub cache holding ESM2-3B
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

EXTRA=""
if [ "${SMOKE:-0}" = 1 ]; then
  EXTRA="--max-steps 9 --save-every 3 --probe-every 3 --log-every 1"
  TOTAL_TOKENS=1e7
fi

mkdir -p "$OUT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT/train.log"; }

attempt=0
while :; do
  attempt=$((attempt + 1))
  log "===== launch $attempt"
  "$TORCHRUN" --nproc_per_node="$NPROC" --master_port="${PORT:-29533}" \
    "$HERE/stage1/train_stage1.py" \
    --model          "$BASE_MODEL" \
    --lm-data        "$LM_DATA" \
    --struct-tokens  "$STRUCT_TOKENS" \
    --struct-targets "$STRUCT_TARGETS" \
    --flow-ckpt      "$FLOW_CKPT" \
    --flow-model     simplefold_100M \
    --probe-eval     "$PROBE_EVAL" \
    --ccd            "$CCD" \
    --out            "$OUT" \
    --total-tokens   "$TOTAL_TOKENS" \
    --inject-mode add \
    --cond-layers 0,4,8,12,16,20,24,28,32 \
    --lambda-flow 28.9 --phase finetune \
    --lr 2e-5 --min-lr-ratio 0.1 --flow-lr 1e-4 --embed-lr 1e-4 \
    --weight-decay 0.1 --warmup-steps 100 --max-grad-norm 1.0 \
    --max-res 256 --max-len 4096 \
    --struct-per-step 2 --struct-copies 8 --micro-tokens 8192 --grad-accum 8 \
    --log-every 10 --save-every 250 --probe-every 250 \
    --seed 0 \
    $EXTRA ${EXTRA_ARGS:-} >> "$OUT/train.log" 2>&1 &
  pid=$!
  # watchdog: a silent hang would otherwise cost a whole night
  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
    age=$(( $(date +%s) - $(stat -c %Y "$OUT/train.log") ))
    if [ "$age" -gt "$STALE_SECS" ]; then
      log "train.log unchanged for $age s: assuming a hang, restarting"
      kill -9 "$pid" 2>/dev/null
      # also kill the per-GPU workers of this run, or the restart would compete with them for memory
      pkill -9 -f "[t]rain_stage1.py .*--out $OUT " 2>/dev/null
      break
    fi
  done
  wait "$pid"; rc=$?
  [ $rc = 0 ] && { log "finished"; break; }
  [ "$attempt" -ge "$MAX_RESTARTS" ] && { log "failed $attempt times, giving up"; exit 1; }
  log "exit code $rc, resuming from the last checkpoint"
done
