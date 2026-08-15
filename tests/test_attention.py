"""L4 stages 2 and 3 - the tensor attention head.

The last test is the point of the milestone: given identical weights, the tensor
version must produce the same numbers as the plain-Python loops. Everything else
here is structure.
"""

import math

import pytest
import torch

import attention as tensor
import attention_plain as plain
from embeddings import D_MODEL

X_ROWS, WORDS = plain.example_input()
X = torch.tensor(X_ROWS)

WQ = plain.build_matrix(D_MODEL, plain.D_HEAD, plain.SEED + 1)
WK = plain.build_matrix(D_MODEL, plain.D_HEAD, plain.SEED + 2)
WV = plain.build_matrix(D_MODEL, plain.D_HEAD, plain.SEED + 3)


@pytest.fixture
def head():
    return tensor.load_plain_matrices(tensor.CausalSelfAttention(), WQ, WK, WV)


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_shapes(head):
    out, parts = head(X)
    assert tuple(out.shape) == (plain.T, plain.D_HEAD)
    assert tuple(parts["scores"].shape) == (plain.T, plain.T)
    assert tuple(parts["weights"].shape) == (plain.T, plain.T)


def test_no_bias():
    """The plain version has no bias, so these must not either - otherwise the
    cross-check fails for a reason that is not a logic error."""
    head = tensor.CausalSelfAttention()
    assert head.to_query.bias is None
    assert head.to_key.bias is None
    assert head.to_value.bias is None


def test_mask_is_a_buffer_not_a_parameter():
    """It is a fixed fact about position, not something to learn."""
    head = tensor.CausalSelfAttention()
    assert "future" in dict(head.named_buffers())
    assert "future" not in dict(head.named_parameters())


def test_weight_rows_sum_to_one(head):
    _, parts = head(X)
    assert torch.allclose(parts["weights"].sum(dim=-1), torch.ones(plain.T), atol=1e-6)


def test_future_gets_exactly_zero_weight(head):
    _, parts = head(X)
    upper = torch.triu(torch.ones(plain.T, plain.T, dtype=torch.bool), diagonal=1)
    assert (parts["weights"][upper] == 0.0).all()


def test_changing_a_later_token_cannot_move_an_earlier_output(head):
    """The leak test, on the tensor path this time."""
    before, _ = head(X)

    tampered = X.clone()
    tampered[-1] = 9.9

    after, _ = head(tampered)

    assert torch.allclose(after[:-1], before[:-1], atol=1e-6)
    assert not torch.allclose(after[-1], before[-1], atol=1e-6)


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------

def test_batch_shape(head):
    out, _ = head(torch.stack([X, X, X]))
    assert tuple(out.shape) == (3, plain.T, plain.D_HEAD)


def test_batch_matches_single(head):
    """Sequences in a batch never interact."""
    single, _ = head(X)
    batched, _ = head(torch.stack([X, X, X]))
    for i in range(3):
        assert torch.allclose(batched[i], single, atol=1e-6)


def test_shorter_sequence_works(head):
    """The mask is sliced to the sequence length, so shorter inputs are fine."""
    out, _ = head(X[:2])
    assert tuple(out.shape) == (2, plain.D_HEAD)


# --------------------------------------------------------------------------
# stage 3 - the one that matters
# --------------------------------------------------------------------------

def test_tensor_matches_plain_at_every_step(head):
    """Given identical weights, every intermediate must match - not just the
    final answer. A difference in the scores could cancel out by the output and
    hide a real bug."""
    out, parts = head(X)
    plain_out, plain_parts = plain.attention_plain(X_ROWS, WQ, WK, WV)

    for name in ("queries", "keys", "values", "weights"):
        assert torch.allclose(
            parts[name], torch.tensor(plain_parts[name]), atol=1e-6
        ), f"{name} differs"

    assert torch.allclose(out, torch.tensor(plain_out), atol=1e-6)


def test_a_missing_transpose_would_be_caught():
    """nn.Linear stores (out, in) while build_matrix gives (in, out). With a
    square matrix a missing .T does not crash - it silently computes something
    else, and only the cross-check notices."""
    wrong = tensor.CausalSelfAttention()
    with torch.no_grad():
        wrong.to_query.weight.copy_(torch.tensor(WQ))      # no .T
        wrong.to_key.weight.copy_(torch.tensor(WK))
        wrong.to_value.weight.copy_(torch.tensor(WV))

    wrong_out, _ = wrong(X)
    plain_out, _ = plain.attention_plain(X_ROWS, WQ, WK, WV)

    assert not torch.allclose(wrong_out, torch.tensor(plain_out), atol=1e-6)


def test_scaling_is_present():
    """scores must be divided by sqrt(d_head)."""
    head = tensor.load_plain_matrices(tensor.CausalSelfAttention(), WQ, WK, WV)
    _, parts = head(X)

    q, k = parts["queries"], parts["keys"]
    unscaled = q @ k.transpose(-2, -1)
    scaled = parts["scores"]

    # compare where nothing was masked: the first column
    assert torch.allclose(
        scaled[:, 0], unscaled[:, 0] / math.sqrt(plain.D_HEAD), atol=1e-5
    )
