"""L4 stage 1 - one causal attention head in plain Python.

The structural tests hold whatever the random weights are. The leak test is the
important one: it checks the thing the mask exists to prevent.
"""

import math

import pytest

import attention_plain as att

X, WORDS = att.example_input()          # the real L3 embeddings
WQ = att.build_matrix(att.D_MODEL, att.D_HEAD, att.SEED + 1)
WK = att.build_matrix(att.D_MODEL, att.D_HEAD, att.SEED + 2)
WV = att.build_matrix(att.D_MODEL, att.D_HEAD, att.SEED + 3)


@pytest.fixture
def run():
    return att.attention_plain(X, WQ, WK, WV)


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------

def test_project_changes_width():
    """d_model numbers in, d_head numbers out."""
    assert len(att.project(X[0], WQ)) == att.D_HEAD


def test_the_matrix_is_shared_but_the_rows_are_not():
    """Same matrix, different inputs, different outputs."""
    a, b = att.project(X[0], WQ), att.project(X[1], WQ)
    assert a != b


def test_softmax_sums_to_one_and_is_positive():
    out = att.softmax([1.0, 2.0, 3.0])
    assert sum(out) == pytest.approx(1.0)
    assert all(v > 0 for v in out)


def test_softmax_handles_large_scores():
    """Subtracting the max first stops exp() overflowing."""
    assert sum(att.softmax([1000.0, 1001.0])) == pytest.approx(1.0)


def test_softmax_turns_minus_infinity_into_exactly_zero():
    out = att.softmax([1.0, float("-inf")])
    assert out[1] == 0.0


def test_scores_grid_is_square():
    q = att.project_all(X, WQ)
    k = att.project_all(X, WK)
    s = att.scores_for(q, k)
    assert len(s) == att.T and all(len(row) == att.T for row in s)


def test_mask_blanks_only_the_future():
    masked = att.apply_causal_mask([[1.0] * att.T for _ in range(att.T)])
    for i in range(att.T):
        for j in range(att.T):
            if j <= i:
                assert masked[i][j] == 1.0
            else:
                assert masked[i][j] == float("-inf")


# --------------------------------------------------------------------------
# the whole head
# --------------------------------------------------------------------------

def test_output_shape(run):
    output, _ = run
    assert len(output) == att.T
    assert all(len(row) == att.D_HEAD for row in output)


def test_position_count_is_unchanged(run):
    """Attention changes what each position holds, not how many there are."""
    output, _ = run
    assert len(output) == len(X)


def test_every_weight_row_sums_to_one(run):
    _, parts = run
    for row in parts["weights"]:
        assert sum(row) == pytest.approx(1.0)


def test_no_attention_paid_to_the_future(run):
    """The second L4 task: masked positions receive no attention probability.
    Exactly zero, not merely small."""
    _, parts = run
    for i, row in enumerate(parts["weights"]):
        for j in range(i + 1, att.T):
            assert row[j] == 0.0


def test_first_position_gets_its_own_value_back(run):
    """Position 0 has nothing before it, so its weighted average has one term."""
    output, parts = run
    assert output[0] == pytest.approx(parts["values"][0])


def test_changing_a_later_token_cannot_move_an_earlier_output():
    """The leak test.

    Replace the LAST position's input entirely. Every earlier output must be
    byte-identical. If any moves, information flowed backwards and the mask is
    not doing its job - which is exactly the bug that would let the model copy
    the answer sitting one slot to its right.
    """
    before, _ = att.attention_plain(X, WQ, WK, WV)

    tampered = [row[:] for row in X]
    tampered[-1] = [9.9] * att.D_MODEL

    after, _ = att.attention_plain(tampered, WQ, WK, WV)

    for i in range(att.T - 1):
        assert after[i] == pytest.approx(before[i])

    assert after[-1] != pytest.approx(before[-1])   # the last one SHOULD move


def test_weights_are_a_blend_of_the_values(run):
    """Each output is inside the range of the values it averaged - an average
    cannot land outside the things being averaged."""
    output, parts = run
    values = parts["values"]
    for i, row in enumerate(output):
        for d in range(att.D_HEAD):
            visible = [values[j][d] for j in range(i + 1)]
            assert min(visible) - 1e-9 <= row[d] <= max(visible) + 1e-9


# --------------------------------------------------------------------------
# scaling and initialisation - the reason attention did not saturate
# --------------------------------------------------------------------------

def test_input_is_the_real_l3_output():
    """Not a stand-in. An invented input hid the saturation problem."""
    import torch

    import embeddings as emb
    from dataset import EXAMPLES

    torch.manual_seed(att.SEED)
    expected = emb.Embeddings()(torch.tensor(EXAMPLES[0][0])).tolist()
    for actual_row, expected_row in zip(X, expected):
        assert actual_row == pytest.approx(expected_row)
    assert len(X) == att.T
    assert len(X[0]) == att.D_MODEL


def test_scaling_shrinks_the_score_gap(run):
    """Dividing by sqrt(d_head) halves the spread at d_head=4."""
    _, parts = run
    unscaled = att.scores_for(parts["queries"], parts["keys"], scale=False)
    gap_unscaled = max(max(r) - min(r) for r in unscaled)
    gap_scaled = max(max(r) - min(r) for r in parts["raw"])
    assert gap_scaled == pytest.approx(gap_unscaled / math.sqrt(att.D_HEAD))


def test_attention_is_usually_not_saturated_at_initialisation():
    """The check that would have caught the uniform(-1, 1) initialisation.

    Before training, a row with several visible positions should usually SPREAD
    its attention. A weight near 1.0 means softmax is pinned, and a pinned
    softmax has almost no gradient - the query and key matrices would barely
    learn.

    Measured across many seeds, not one. A single seed saturates about 15% of
    the time by chance, so a one-seed version of this test passes or fails on
    luck rather than on anything real - which is exactly what happened when
    D_HEAD changed and this test flipped for no meaningful reason.
    """
    biggest = []
    for seed in range(100):
        wq = att.build_matrix(att.D_MODEL, att.D_HEAD, seed * 3 + 1)
        wk = att.build_matrix(att.D_MODEL, att.D_HEAD, seed * 3 + 2)
        wv = att.build_matrix(att.D_MODEL, att.D_HEAD, seed * 3 + 3)
        _, parts = att.attention_plain(X, wq, wk, wv)
        biggest.append(max(parts["weights"][-1]))

    average = sum(biggest) / len(biggest)
    saturated = sum(1 for m in biggest if m > 0.95)

    assert average < 0.85, f"attention is pinned on average ({average:.3f})"
    assert saturated < 35, f"{saturated}/100 seeds saturated"


def test_score_size_does_not_grow_with_head_width():
    """What sqrt(d_head) is FOR: scores must not get bigger just because the
    head got wider. Averaged over seeds, since any single seed is noisy."""
    def average_score_spread(d_head):
        spreads = []
        for seed in range(100):
            wq = att.build_matrix(att.D_MODEL, d_head, seed * 3 + 1)
            wk = att.build_matrix(att.D_MODEL, d_head, seed * 3 + 2)
            scores = att.scores_for(att.project_all(X, wq), att.project_all(X, wk))
            flat = [v for row in scores for v in row]
            spreads.append(max(flat) - min(flat))
        return sum(spreads) / len(spreads)

    narrow, wide = average_score_spread(4), average_score_spread(8)
    assert wide == pytest.approx(narrow, rel=0.25)


def test_initialisation_scales_with_fan_in():
    """std = 1/sqrt(fan_in), so wider matrices hold smaller numbers and the
    outputs stay roughly the same size however wide the layer is."""
    narrow = att.build_matrix(4, 4, 0)
    wide = att.build_matrix(64, 4, 0)
    spread = lambda m: (sum(v ** 2 for row in m for v in row) / (len(m) * len(m[0]))) ** 0.5
    assert spread(wide) < spread(narrow)
