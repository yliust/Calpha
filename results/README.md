# Results

Everything here is produced by the scorer in `src/calpha/eval/score.py`.

```
scores/<system>/<set>/scores.csv           one row per target
scores/<system>/<set>/summary.json         means overall, by homology bucket, by length bucket
scores/<system>/<set>/*.subset180.*        the same on the 180-target subset used for chat models
scores/<system>/<set>/meta.json            how the system was run (third-party baselines)
comparisons/calpha__vs__<system>.json      paired difference in TM-score, bootstrap 95% interval
multisample/                               16 samples per target from Calpha
trigger/                                   how often the model answers with <structure>, by phrasing
mmlu_pro/                                  general-ability checks of third-party language models
```

Columns of `scores.csv`: `id`, `length`, `family` (has a homolog in the training pool),
`len_bucket`, `ok` (a usable prediction exists), `tm`, `lddt`, `gdt_ts`, `rmsd`, `coverage`,
`ca_ca_ok` (fraction of consecutive CA–CA distances in 3.6–4.0 Å), `clash`, `rg_norm`.
A target without a usable prediction scores 0 and stays in the mean.

Systems: `calpha` (the released model), `calpha_joint` (stage-1 model with its co-trained 100M
head), `calpha_token` (our earlier model that writes structure tokens; scores imported from its
own evaluation run), `head_only_control` (SimpleFold-3B fine-tuned on ESM2 features with our
data, no language model), `simplefold_3b` / `simplefold_100m` (public weights), `esmfold`,
`esm3_open`, `dplm2_650m`, `dplm2_bit_650m`, `multiflow`, `biomatrix_1p7b` / `biomatrix_4b`,
`opendde_single` / `opendde_msa`, `chat_gpt` / `chat_claude` / `chat_gemini` / `chat_deepseek`,
`nearest_training_structure`, `random_walk`, `ideal_helix`, `ground_truth`.

Raw predictions (`pred_ca.npz`, one alpha-carbon array per target) and the raw text returned by
the chat models are in `benchmark_outputs/` of the Hugging Face repository.

Rows for third-party models describe their public weights under our protocol. Several could not
be matched to a number reported by their authors; the report lists which.
