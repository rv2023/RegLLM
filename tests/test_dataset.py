"""L2 - encode/decode and the shifted training pairs."""

import pytest

import dataset as ds
import tokenizer as tk


# --------------------------------------------------------------------------
# encode / decode
# --------------------------------------------------------------------------

def test_encode_known_sentence():
    assert ds.encode("the king ruled the kingdom") == [12, 5, 11, 12, 6]


def test_round_trip():
    for sentence in [
        "the king ruled the kingdom",
        "the queen entered the castle",
        "the kingdom had a castle",
    ]:
        assert ds.decode(ds.encode(sentence)) == sentence


def test_encode_uses_the_same_tokenizer():
    """encode must not tokenize differently from tokenizer.py, or the two drift."""
    assert ds.encode(tk.TEXT) == [tk.STOI[t] for t in tk.TOKENS]


def test_encode_strips_periods_like_the_tokenizer():
    assert ds.encode("the king.") == ds.encode("the king")


def test_out_of_vocabulary_word_raises():
    """The closed vocabulary from L1, made concrete. No <UNK> exists here."""
    with pytest.raises(KeyError):
        ds.encode("the dragon ruled the kingdom")


def test_corpus_encodes_to_the_expected_length():
    assert len(ds.IDS) == 42
    assert all(0 <= i < tk.VOCAB_SIZE for i in ds.IDS)


# --------------------------------------------------------------------------
# the sliding window
# --------------------------------------------------------------------------

@pytest.mark.parametrize("block_size,expected", [(3, 39), (4, 38), (5, 37)])
def test_example_count(block_size, expected):
    """len(ids) - block_size, NOT + 1: the last window needs one token beyond
    its own end to have a target."""
    assert len(ds.build_examples(ds.IDS, block_size)) == expected


def test_every_pair_has_matching_lengths():
    for block_size in (3, 4, 5):
        for x, y in ds.build_examples(ds.IDS, block_size):
            assert len(x) == len(y) == block_size


def test_targets_are_inputs_shifted_by_one():
    """The defining property: y[i] is the token that follows x[i]."""
    for x, y in ds.EXAMPLES:
        assert y[:-1] == x[1:]


def test_first_example_is_exact():
    x, y = ds.EXAMPLES[0]
    assert x == [12, 5, 11, 12]      # the king ruled the
    assert y == [5, 11, 12, 6]       # king ruled the kingdom


def test_last_example_ends_at_the_final_token():
    """Nothing is dropped off the end and nothing runs past it."""
    x, y = ds.EXAMPLES[-1]
    assert y[-1] == ds.IDS[-1]
    assert x == ds.IDS[-ds.BLOCK_SIZE - 1:-1]


def test_every_token_transition_is_covered():
    """Each adjacent pair in the corpus appears as some (input, target)
    position - no transition is skipped by the windowing."""
    transitions = {(ds.IDS[i], ds.IDS[i + 1]) for i in range(len(ds.IDS) - 1)}
    covered = {
        (x[position], y[position])
        for x, y in ds.EXAMPLES
        for position in range(len(x))
    }
    assert transitions == covered


def test_windows_are_continuous_not_per_line():
    """Deliberate choice: the corpus is one stream, so windows span line breaks.
    Per-line windowing would give 10 examples at block_size=4, not 38."""
    assert len(ds.EXAMPLES) == 38
