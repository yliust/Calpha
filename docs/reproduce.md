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

The head starts from Gaussian noise. The archived run and a rerun draw different noise for the
same target, because the random generator is consumed in a different order during model
construction. Scores of individual targets therefore differ; the mean is stable (see
`results/multisample/`).

## Rerun with the released files

<!-- RERUN_TABLE -->
