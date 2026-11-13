# What "reproduced" means here

## What was checked before release

1. **Weights.** The released language model is bit-identical (427 tensors) to the stage-1
   checkpoint cast to bf16. Every weight of the structure head other than the front-end equals
   the public SimpleFold-3B checkpoint, except the buffer `pos_embedder.freq_bands`, which is
   shipped inside `frontend.safetensors` with the values used in our runs.
2. **Hidden states.** On benchmark proteins of 28 to 1,414 residues, the conditioning tensor
   computed by this package from the released files is bit-identical to the one computed by the
   code and checkpoint that produced the reported numbers.
3. **End to end.** The benchmark was re-run with this package and the released files; see the
   table below.

## Two numerical details that matter

- The rotary-embedding frequency buffers of the language model must be in bf16, as they were
  when the front-end was trained. `from_pretrained(dtype=bfloat16)` leaves them in float32,
  which moves the hidden states by about 2% in relative norm. The package casts them.
- The hidden states are read from the full chat turn, including the assistant answer ending in
  `<structure>`. The model is causal, but for some prompt lengths the states at the residue
  positions differ by about 2% in relative norm depending on whether the answer tokens follow.
  `Calpha.fold` always feeds the full turn.

## Sampling noise

The head starts from Gaussian noise drawn from a generator seeded with `seed + shard index`, and
each shard folds its targets in sorted order. Use the same number of shards as the reported run
(8) to draw the same noise. With a different number of shards individual scores move; the mean
is stable (see `results/multisample/`).

## Rerun with the released files

FoldBench-334, 8 shards, seed 42, run on 2026-11-13 with this package and the released files
(`llm/`, `esm_adapter.safetensors`, `frontend.safetensors`) plus the public SimpleFold-3B
checkpoint:

| | Reported | Rerun |
|---|---|---|
| TM-score, all 334 targets | 0.8075 | 0.8075 |
| TM-score, no-homolog targets (128) | 0.7785 | 0.7785 |
| lDDT / GDT-TS | 0.7841 / 0.7408 | 0.7841 / 0.7408 |
| per-target TM difference, 90th percentile | | 0.0000 |

On this hardware and software stack (H200, the versions in `requirements-frozen.txt`) the rerun
matched the archived run target by target. On other GPUs or library versions expect small
per-target differences from floating-point and sampling noise. CASP16 and CAMEO-2025 were not
re-run for this check.

A smoke test of `calpha chat`: an unseen phrasing ("Can you show me what this protein looks like
in 3D?") was answered with `<structure>` and a 129-residue structure was written; "How many
residues does this protein have?" was answered in text and no structure was produced. The
text answers of the model to protein questions follow the formats of its training tasks and
are often not useful prose.

