"""Constants shared by training-time and inference-time code. Changing any of these changes
what the released weights were trained on."""

AMINO = "ACDEFGHIKLMNPQRSTVWY"

# The token the language model emits to hand over to the structure head.
STRUCT_TOKEN = "<structure>"
SEQ_OPEN, SEQ_CLOSE = "<seq>", "</seq>"

# Hidden states read from the 32-layer language model: index 0 is the embedding output,
# index k the output of layer k. Nine of the 33 entries.
COND_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32)

# Folding requests used in stage-1 training: (user turn, assistant turn).
CHAT_TEMPLATES = [
    ("Fold this protein and give me the 3D structure.\nSequence: {seq}",
     "Sure, here is the predicted structure. {tag}"),
    ("What is the 3D structure of this protein?\n{seq}", "Here is the predicted structure. {tag}"),
    ("What does this protein look like in 3D? Please show the structure.\n{seq}", "{tag}"),
    ("Model the three-dimensional fold of:\n{seq}", "{tag}"),
    ("Given the sequence {seq}, output its folded structure.", "The structure is: {tag}"),
]

# The front-end (stage 2) was trained, and every reported benchmark number was produced, with
# the hidden states of this one request.
CANONICAL_TEMPLATE = CHAT_TEMPLATES[0]

# Sampling settings of the reported numbers.
NUM_STEPS = 200
TAU = 0.01
SEED = 42

HF_REPO = "liuyong9975/Calpha"
SIMPLEFOLD_COMMIT = "94e2cd704cf7b86ef8e7cb27a1e77154c4e35b34"


def aa_token(a: str) -> str:
    return f"<aa{a if a in AMINO else 'X'}>"


def seq_tokens(seq: str) -> str:
    """A protein as it appears inside a chat turn: <seq><aaM><aaK>...</seq>."""
    return SEQ_OPEN + "".join(aa_token(a) for a in seq) + SEQ_CLOSE


def user_turn(text: str) -> str:
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
