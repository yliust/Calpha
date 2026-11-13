"""Tests that need no GPU and no weights."""
import numpy as np

from calpha.constants import CANONICAL_TEMPLATE, seq_tokens, user_turn
from calpha.eval.metrics import all_metrics, tm_score
from calpha.eval.score import length_bucket, validity


def test_seq_tokens():
    assert seq_tokens("ACB") == "<seq><aaA><aaC><aaX></seq>"
    assert user_turn("hi").endswith("<|im_start|>assistant\n")
    assert "{seq}" in CANONICAL_TEMPLATE[0]


def test_metrics_identity_and_rigid_motion():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(80, 3)) * 2.2, axis=0)
    m = all_metrics(x, x)
    assert abs(m["tm"] - 1) < 1e-6 and m["rmsd"] < 1e-6 and abs(m["lddt"] - 1) < 1e-6
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    assert tm_score(x @ q.T + 5.0, x) > 0.999


def test_buckets_and_validity():
    assert [length_bucket(n) for n in (99, 100, 299, 300, 600, 601)] == \
        ["<100", "100-300", "100-300", "300-600", "300-600", ">600"]
    line = np.stack([np.arange(30) * 3.8, np.zeros(30), np.zeros(30)], 1)
    assert validity(line)["ca_ca_ok"] == 1.0
