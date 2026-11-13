# Benchmark sets

| File | Targets | What it is |
|---|---|---|
| `foldbench334.parquet` | 334 | Single chains from FoldBench, released between January 2023 and October 2024. Homologs of these targets were removed from the data used to train the language model. This was our development set. |
| `casp16.parquet` | 155 | Evaluation domains of CASP16 (2024), each predicted as a separate chain. |
| `cameo2025.parquet` | 200 | Single-chain X-ray structures released by the PDB between 2025-01-01 and 2025-02-19, selected with CAMEO-style criteria (the CAMEO server was unavailable when the set was built, so targets were taken from RCSB directly). |

Columns: `id`, `source`, `sequence`, `length`, `coords` (flat list, reshape to `(L, 5, 3)`:
N, CA, C, O, CB; NaN where unobserved), `observed_frac`, `resolution`, `release_date`.
`<set>.summary.json` records how each set was filtered.

- `homology_hits.tsv.gz`: MMseqs2 hits of every target against the training pool (PDB chains
  plus AFDB-SwissProt, 667,331 chains). Columns: query, target, identity, alignment length,
  query coverage, target coverage, e-value, bit score. A target is a *homolog* target if it has
  a hit with identity ≥ 0.30 and coverage ≥ 0.5 on either side.
- `subset180.json`: the fixed 180 targets (60 per set, at most 300 residues) on which the
  frontier chat models were evaluated.
