"""L3 - embeddings.

Two lookup tables and an addition, built twice: plain Python lists, then
PyTorch. The last test is the one that matters — it proves nn.Embedding does
exactly what the list version does, the same cross-check that caught the missing
reduction when NumPy replaced lists at R6.
"""

import pytest
import torch

import embeddings as emb
import tokenizer as tk


@pytest.fixture
def tables():
    word = emb.build_table(tk.VOCAB_SIZE, emb.D_MODEL, emb.SEED)
    pos = emb.build_table(emb.BLOCK_SIZE, emb.D_MODEL, emb.SEED + 1)
    return word, pos


X = [12, 5, 11, 12]      # the king ruled the - "the" at positions 0 AND 3


# --------------------------------------------------------------------------
# Stage 1: plain Python
# --------------------------------------------------------------------------

def test_table_shapes(tables):
    word, pos = tables
    assert len(word) == tk.VOCAB_SIZE          # one row per vocabulary word
    assert len(pos) == emb.BLOCK_SIZE          # one row per position
    assert all(len(row) == emb.D_MODEL for row in word + pos)


def test_tables_are_seeded(tables):
    word, _ = tables
    assert emb.build_table(tk.VOCAB_SIZE, emb.D_MODEL, emb.SEED) == word


def test_lookup_is_just_indexing(tables):
    word, _ = tables
    assert emb.lookup(word, [12, 5]) == [word[12], word[5]]


def test_plain_output_shape(tables):
    word, pos = tables
    out = emb.embed_plain(X, word, pos)
    assert len(out) == len(X)
    assert all(len(row) == emb.D_MODEL for row in out)


def test_same_word_same_row(tables):
    """Positions 0 and 3 are both token 12, so their WORD rows are identical."""
    word, _ = tables
    assert word[X[0]] == word[X[3]]


def test_position_breaks_the_tie(tables):
    """...but their totals are not, which is the entire point of L3."""
    word, pos = tables
    out = emb.embed_plain(X, word, pos)
    assert out[0] != out[3]


def test_position_rows_do_not_depend_on_the_word(tables):
    """The position table is indexed by slot, not by token. Different sentences
    reuse the same four position rows."""
    word, pos = tables
    other = [5, 5, 5, 5]
    a = emb.embed_plain(X, word, pos)
    b = emb.embed_plain(other, word, pos)
    for position in range(len(X)):
        assert [round(v - w, 9) for v, w in zip(a[position], word[X[position]])] == \
               [round(v - w, 9) for v, w in zip(b[position], word[other[position]])]


# --------------------------------------------------------------------------
# Stage 2: PyTorch
# --------------------------------------------------------------------------

def test_module_shapes():
    model = emb.Embeddings()
    ids = torch.tensor(X)
    assert tuple(model.token.weight.shape) == (tk.VOCAB_SIZE, emb.D_MODEL)
    assert tuple(model.position.weight.shape) == (emb.BLOCK_SIZE, emb.D_MODEL)
    assert tuple(model(ids).shape) == (len(X), emb.D_MODEL)


def test_parameter_count():
    """14*8 + 4*8 = 144. Phase 1 had two parameters."""
    model = emb.Embeddings()
    assert sum(p.numel() for p in model.parameters()) == 144


def test_batch_shape():
    """(batch, T) -> (batch, T, d_model). Position rows broadcast across batch."""
    model = emb.Embeddings()
    batch = torch.tensor([X, X, X])
    assert tuple(model(batch).shape) == (3, len(X), emb.D_MODEL)


def test_sequence_length_is_preserved():
    """Embeddings change WHAT each position holds, never HOW MANY there are."""
    model = emb.Embeddings()
    for length in (1, 2, 3, 4):
        ids = torch.tensor(X[:length])
        assert model(ids).shape[0] == length


def test_ids_must_be_integers():
    model = emb.Embeddings()
    with pytest.raises((RuntimeError, IndexError)):
        model(torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_out_of_vocabulary_id_raises():
    """The table has exactly VOCAB_SIZE rows - L1's closed vocabulary, enforced."""
    model = emb.Embeddings()
    with pytest.raises(IndexError):
        model(torch.tensor([tk.VOCAB_SIZE, 0, 0, 0]))


# --------------------------------------------------------------------------
# Stage 3: the two implementations must agree
# --------------------------------------------------------------------------

def test_plain_and_torch_agree(tables):
    """Given identical tables, nn.Embedding must produce identical numbers.
    This is what proves nn.Embedding is a table plus a lookup, and nothing else."""
    word, pos = tables
    model = emb.load_plain_tables(emb.Embeddings(), word, pos)

    torch_out = model(torch.tensor(X))
    plain_out = torch.tensor(emb.embed_plain(X, word, pos))

    assert torch.allclose(torch_out, plain_out, atol=1e-6)
