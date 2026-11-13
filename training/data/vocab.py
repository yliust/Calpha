"""Protein vocabulary added to the Qwen3.5 tokenizer.

Every residue-level symbol is one atomic token so sequence, structure and secondary
structure stay residue-aligned (residue i of <seq> <-> token i of <struct> / <ss>).
"""

from __future__ import annotations

import os

AMINO = "ACDEFGHIKLMNPQRSTVWY"
# Codebook size of the discrete structure tokens. The default 4096 is the ESM3 VQ-VAE codebook;
# all data used for Calpha was built with this value, so do not change the default.
# It can be overridden through the environment for another structure tokenizer.
# Names are always <sNNNN> (4 decimal digits), which covers every size up to 10000, so the format
# does not depend on the codebook size.
N_STRUCT = int(os.environ.get("FOLD_N_STRUCT", "4096"))
if not 1 <= N_STRUCT <= 10000:
    raise ValueError(f"FOLD_N_STRUCT={N_STRUCT} is outside the range <sNNNN> can represent")

AA_TOKENS = [f"<aa{a}>" for a in AMINO] + ["<aaX>"]
STRUCT_TOKENS = [f"<s{i:04d}>" for i in range(N_STRUCT)]
STRUCT_MASK = "<smask>"  # unresolved (PDB) or low-confidence (AFDB pLDDT < 50) residue
SS_TOKENS = {"H": "<ssH>", "E": "<ssE>", "-": "<ssC>", "X": "<ssX>"}
TAGS = ["<seq>", "</seq>", "<struct>", "</struct>", "<ss>", "</ss>", "<contacts>", "</contacts>",
        "<span>", "</span>"]

ALL_TOKENS = AA_TOKENS + STRUCT_TOKENS + [STRUCT_MASK] + list(SS_TOKENS.values()) + TAGS


def seq_str(seq: str) -> str:
    return "".join(f"<aa{a}>" if a in AMINO else "<aaX>" for a in seq)


def struct_str(tokens) -> str:
    return "".join(STRUCT_MASK if t < 0 else f"<s{t:04d}>" for t in tokens)


def ss_str(ss: str) -> str:
    return "".join(SS_TOKENS[c] for c in ss)


# ---- v2 "interleave" format: every output residue symbol is anchored by its amino acid token,
# so position i of the output never has to be aligned to position i of the prompt by offset.
def aa_tok(a: str) -> str:
    return f"<aa{a}>" if a in AMINO else "<aaX>"


def struct_tok(t: int) -> str:
    return STRUCT_MASK if t < 0 else f"<s{t:04d}>"


def interleave_struct(seq: str, tokens) -> str:
    return "".join(aa_tok(a) + struct_tok(int(t)) for a, t in zip(seq, tokens))


def interleave_ss(seq: str, ss: str) -> str:
    return "".join(aa_tok(a) + SS_TOKENS[c] for a, c in zip(seq, ss))


def interleave_ss_struct(seq: str, ss: str, tokens) -> str:
    return "".join(aa_tok(a) + SS_TOKENS[c] + struct_tok(int(t)) for a, c, t in zip(seq, ss, tokens))


def interleave_struct_then_aa(seq: str, tokens) -> str:
    """Inverse folding output: structure anchor first, then the residue to predict."""
    return "".join(struct_tok(int(t)) + aa_tok(a) for a, t in zip(seq, tokens))
