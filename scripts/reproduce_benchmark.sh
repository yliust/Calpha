#!/usr/bin/env bash
# Generate and score Calpha on one benchmark set, sharded over GPUs.
#   bash scripts/reproduce_benchmark.sh <foldbench334|casp16|cameo2025> [n_gpus=8] [out_dir]
# The reported numbers used 8 shards (seed 42 + shard index).
set -euo pipefail
SET=${1:?set name}; NG=${2:-8}; OUT=${3:-runs/$SET}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
pids=()
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i calpha benchmark --set "benchmarks/$SET.parquet" --out "$OUT" \
      --rank "$i" --world "$NG" ${CALPHA_WEIGHTS:+--weights "$CALPHA_WEIGHTS"} \
      ${SIMPLEFOLD_DIR:+--simplefold-dir "$SIMPLEFOLD_DIR"} > "$OUT/gen.rank$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
calpha score --set "benchmarks/$SET.parquet" --out "$OUT"
for ref in head_only_control esmfold; do
  echo "== Calpha (this run) minus $ref"
  calpha compare "$OUT/scores.csv" "results/scores/$ref/$SET/scores.csv" | head -12
done
