"""L5 - multi-head attention.

The test that matters is the last one: the fused version, which reshapes rather
than looping, must produce exactly what the explicit list of heads produces. A
reshape that splits the wrong dimension does not crash - it quietly mixes the
wrong numbers together, and only a comparison catches it.
"""

import pytest
import torch

import multihead as mh
from attention import CausalSelfAttention
from attention_plain import example_input
from embeddings import D_MODEL

X = torch.tensor(example_input()[0])


@pytest.fixture
def pair():
    """An explicit head-list and a fused version holding identical weights."""
    explicit = mh.MultiHeadExplicit()
    fused = mh.copy_explicit_into_fused(explicit, mh.MultiHeadFused())
    return explicit, fused


# --------------------------------------------------------------------------
# the divisibility constraint
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_head_counts_that_divide_d_model_work(n_heads):
    out, _ = mh.MultiHeadExplicit(n_heads=n_heads)(X)
    assert tuple(out.shape) == (len(X), D_MODEL)


@pytest.mark.parametrize("n_heads", [3, 5, 7])
def test_head_counts_that_do_not_divide_are_rejected(n_heads):
    """The heads are concatenated, so the total must come back to d_model.
    Better a clear error than a confusing shape failure later."""
    with pytest.raises(ValueError, match="divisible"):
        mh.MultiHeadExplicit(n_heads=n_heads)
    with pytest.raises(ValueError, match="divisible"):
        mh.MultiHeadFused(n_heads=n_heads)


def test_shape_is_unchanged_whatever_the_head_count():
    """d_model in, d_model out - which is what lets L6 add the answer back on."""
    for n_heads in (1, 2, 4, 8):
        for model in (mh.MultiHeadExplicit(n_heads=n_heads),
                      mh.MultiHeadFused(n_heads=n_heads)):
            out, _ = model(X)
            assert out.shape == X.shape


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_parameter_count_matches_between_versions(pair):
    explicit, fused = pair
    assert (sum(p.numel() for p in explicit.parameters())
            == sum(p.numel() for p in fused.parameters()))


def test_four_matrices_not_three(pair):
    """Q, K, V per head plus one shared output projection."""
    explicit, _ = pair
    assert explicit.to_output.weight.shape == (D_MODEL, D_MODEL)
    assert explicit.to_output.bias is None


def test_heads_have_their_own_weights(pair):
    explicit, _ = pair
    first, second = explicit.heads[0], explicit.heads[1]
    assert not torch.allclose(first.to_query.weight, second.to_query.weight)


def test_heads_actually_disagree(pair):
    """Different weights should give different opinions - otherwise the extra
    heads are costing parameters and buying nothing."""
    explicit, _ = pair
    _, parts = explicit(X)
    first = parts["heads"][0]["weights"]
    second = parts["heads"][1]["weights"]
    assert not torch.allclose(first, second, atol=1e-3)


# --------------------------------------------------------------------------
# causality survives the extra machinery
# --------------------------------------------------------------------------

def test_every_head_still_blocks_the_future(pair):
    explicit, _ = pair
    _, parts = explicit(X)
    upper = torch.triu(torch.ones(len(X), len(X), dtype=torch.bool), diagonal=1)
    for head_parts in parts["heads"]:
        assert (head_parts["weights"][upper] == 0.0).all()


def test_changing_a_later_token_cannot_move_an_earlier_output(pair):
    """The leak test, now with concatenation and an output projection in the
    way. W_O mixes the heads, so a leak in any one of them would show here."""
    _, fused = pair
    before, _ = fused(X)

    tampered = X.clone()
    tampered[-1] = 9.9
    after, _ = fused(tampered)

    assert torch.allclose(after[:-1], before[:-1], atol=1e-6)
    assert not torch.allclose(after[-1], before[-1], atol=1e-6)


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------

def test_batch_matches_single(pair):
    _, fused = pair
    single, _ = fused(X)
    batched, _ = fused(torch.stack([X, X, X]))
    assert tuple(batched.shape) == (3, len(X), D_MODEL)
    for i in range(3):
        assert torch.allclose(batched[i], single, atol=1e-6)


# --------------------------------------------------------------------------
# the reshape - the one that matters
# --------------------------------------------------------------------------

def test_explicit_and_fused_agree(pair):
    explicit, fused = pair
    explicit_out, explicit_parts = explicit(X)
    fused_out, fused_parts = fused(X)

    assert torch.allclose(explicit_parts["joined"], fused_parts["joined"], atol=1e-6)
    assert torch.allclose(explicit_out, fused_out, atol=1e-6)


def test_split_heads_gives_each_head_its_own_slice():
    """Head h must own numbers h*d_head onward - the same slice the explicit
    version hands its separate heads. Anything else mixes the wrong numbers."""
    fused = mh.MultiHeadFused(n_heads=2)
    rows = torch.arange(float(len(X) * D_MODEL)).reshape(len(X), D_MODEL)
    split = fused.split_heads(rows)

    assert tuple(split.shape) == (2, len(X), fused.d_head)
    assert torch.equal(split[0], rows[:, :fused.d_head])
    assert torch.equal(split[1], rows[:, fused.d_head:])


def test_merge_undoes_split():
    fused = mh.MultiHeadFused(n_heads=2)
    rows = torch.randn(len(X), D_MODEL)
    assert torch.allclose(fused.merge_heads(fused.split_heads(rows)), rows)


def test_one_head_reduces_to_the_l4_head():
    """With n_heads=1 and W_O set to the identity, multi-head must reproduce the
    single head from L4 exactly. If this fails the plumbing is wrong before any
    of the interesting part."""
    single = CausalSelfAttention(d_model=D_MODEL, d_head=D_MODEL)
    multi = mh.MultiHeadExplicit(n_heads=1)

    with torch.no_grad():
        multi.heads[0].to_query.weight.copy_(single.to_query.weight)
        multi.heads[0].to_key.weight.copy_(single.to_key.weight)
        multi.heads[0].to_value.weight.copy_(single.to_value.weight)
        multi.to_output.weight.copy_(torch.eye(D_MODEL))

    single_out, _ = single(X)
    multi_out, _ = multi(X)
    assert torch.allclose(single_out, multi_out, atol=1e-6)
