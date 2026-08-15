"""L1 - the tokenizer.

This module is imported by every milestone from L2 to L11, so a change here can
break something several files away. These tests pin the numbers those milestones
depend on.
"""

from collections import Counter

import pytest

import tokenizer as tk

EXPECTED_VOCAB = [
    "a", "castle", "entered", "had", "in", "king", "kingdom",
    "lived", "prince", "protected", "queen", "ruled", "the", "visited",
]


def test_corpus_shape():
    assert len(tk.TEXT.strip().splitlines()) == 8
    assert len(tk.TOKENS) == 42


def test_vocabulary_is_exact():
    assert tk.VOCABULARY == EXPECTED_VOCAB
    assert tk.VOCAB_SIZE == 14


def test_vocabulary_is_sorted():
    """Sorting makes the IDs a deterministic function of the text. A set has no
    defined order, and at L3 the embedding table is indexed by these IDs."""
    assert tk.VOCABULARY == sorted(tk.VOCABULARY)


def test_round_trip():
    for word in tk.VOCABULARY:
        assert tk.ITOS[tk.STOI[word]] == word


def test_ids_are_contiguous_from_zero():
    """IDs are row indices into the embedding table (L3). A gap would be an
    unused row; an ID >= vocab_size would be an out-of-bounds lookup."""
    assert sorted(tk.STOI.values()) == list(range(tk.VOCAB_SIZE))
    assert sorted(tk.ITOS) == list(range(tk.VOCAB_SIZE))


@pytest.mark.parametrize("word,expected_id", [("a", 0), ("castle", 1), ("the", 12), ("visited", 13)])
def test_known_ids(word, expected_id):
    assert tk.STOI[word] == expected_id
    assert tk.ITOS[expected_id] == word


def test_every_token_is_in_the_vocabulary():
    """True by construction for a closed dataset - which is why no <UNK> token
    is needed here, and why one would be needed on unseen text."""
    assert all(token in tk.STOI for token in tk.TOKENS)


def test_rebuilding_is_deterministic():
    for _ in range(5):
        assert tk.build_vocab(tk.TOKENS) == EXPECTED_VOCAB


def test_tokenize_strips_periods():
    assert tk.tokenize("the king.") == ["the", "king"]


def test_data_path_is_module_relative():
    """Built from __file__, so importing from any working directory works."""
    assert tk.DATA_PATH.is_absolute()
    assert tk.DATA_PATH.exists()


def test_majority_class_baseline():
    """34.1% on the next-token task. Anything at or below this at L11 has learned
    word frequency, not context."""
    counts = Counter(tk.TOKENS)
    most_common, n = counts.most_common(1)[0]
    assert most_common == "the"
    assert n == 15

    targets = tk.TOKENS[1:]                       # next-token targets
    baseline = targets.count("the") / len(targets)
    assert baseline == pytest.approx(14 / 41)
    assert baseline == pytest.approx(0.341, abs=0.001)
