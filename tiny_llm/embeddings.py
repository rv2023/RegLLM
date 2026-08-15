"""L3 - embeddings: two lookup tables and an addition.

Token IDs cannot be fed to a model directly. "the" = 12 and "king" = 5 came from
sorting the vocabulary alphabetically - they are labels, not values, and a model
doing arithmetic would treat 12 as more than twice 5.

So each word gets a ROW OF NUMBERS instead, stored in a table with one row per
word. Looking a word up is list indexing; nothing more. The rows start random and
are learned, exactly as m and c were in Phase 1.

A second table holds one row per POSITION, because "the king ruled the" has "the"
at positions 0 and 3 and their word rows are bit-for-bit identical. Attention
(L4) is permutation-invariant, so without positional information
"the king ruled the" and "ruled the the king" are the same input.

The two are ADDED, which is why they must be the same width. Adding keeps the
width fixed at d_model, and it has to stay fixed because L6 does
`x = x + Attention(x)`.

Built twice, per the scalars-before-matrices rule: plain lists first, then
PyTorch, then checked against each other.
"""

import random

import torch
from torch import nn

from tokenizer import ITOS, VOCAB_SIZE

D_MODEL    = 8      # numbers per word. 768 in GPT-2; 8 is plenty for 14 words.
BLOCK_SIZE = 4      # positions in the window, matching dataset.BLOCK_SIZE
SEED       = 0


# --------------------------------------------------------------------------
# Stage 1: plain Python. No PyTorch, no matrices - just lists and indexing.
# --------------------------------------------------------------------------

def build_table(rows, cols, seed):
    """A table of random numbers: `rows` rows, `cols` numbers each."""
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]


def lookup(table, indices):
    """The entire embedding operation: fetch a row per index."""
    return [table[i] for i in indices]


def add_rows(left, right):
    """Element-by-element addition of two equally shaped tables of rows."""
    return [[a + b for a, b in zip(row_l, row_r)] for row_l, row_r in zip(left, right)]


def embed_plain(ids, word_table, pos_table):
    """ids -> one row per position, word row + position row."""
    token_rows    = lookup(word_table, ids)
    position_rows = lookup(pos_table, range(len(ids)))
    return add_rows(token_rows, position_rows)


# --------------------------------------------------------------------------
# Stage 2: the same thing in PyTorch.
# --------------------------------------------------------------------------

class Embeddings(nn.Module):
    """nn.Embedding is a table plus a lookup. Nothing else."""

    def __init__(self, vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, d_model=D_MODEL):
        super().__init__()
        self.token    = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(block_size, d_model)

    def forward(self, ids):
        """ids: (T,) or (batch, T) of int64. Returns (..., T, d_model).

        The position lookup does not depend on the tokens - arange(T) is the same
        every time - so it broadcasts across the batch dimension for free.
        """
        seq_len = ids.shape[-1]
        positions = torch.arange(seq_len, device=ids.device)
        return self.token(ids) + self.position(positions)


# --------------------------------------------------------------------------
# Stage 3: prove they agree.
# --------------------------------------------------------------------------

def load_plain_tables(module, word_table, pos_table):
    """Copy the stage-1 lists into the module's weights, so both stages hold
    identical numbers and any difference in output is a difference in logic."""
    with torch.no_grad():
        module.token.weight.copy_(torch.tensor(word_table))
        module.position.weight.copy_(torch.tensor(pos_table))
    return module


if __name__ == "__main__":
    from dataset import EXAMPLES

    x, _ = EXAMPLES[0]
    word_table = build_table(VOCAB_SIZE, D_MODEL, SEED)
    pos_table  = build_table(BLOCK_SIZE, D_MODEL, SEED + 1)

    print(f"x = {x}  {[ITOS[i] for i in x]}")
    print("positions 0 and 3 are both token 12 (\"the\")")
    print()

    # ---- stage 1 ----------------------------------------------------------
    print("stage 1: plain Python lists")
    plain = embed_plain(x, word_table, pos_table)
    for position, (token_id, row) in enumerate(zip(x, plain)):
        word_row = word_table[token_id]
        pos_row  = pos_table[position]
        print(f"  position {position}  word {ITOS[token_id]!r}")
        print(f"     word_table[{token_id:2}] = {[round(v, 3) for v in word_row]}")
        print(f"     pos_table[{position}]   = {[round(v, 3) for v in pos_row]}")
        print(f"     added         = {[round(v, 3) for v in row]}")
    print()

    print("  same word, different totals:")
    print(f"    word rows equal at positions 0 and 3? {word_table[x[0]] == word_table[x[3]]}")
    print(f"    sums equal?                           {plain[0] == plain[3]}")
    print()

    # ---- stage 2 ----------------------------------------------------------
    print("stage 2: PyTorch")
    torch.manual_seed(SEED)
    model = Embeddings()
    ids = torch.tensor(x)
    # print(f"  ids              {tuple(ids.shape)}  dtype {ids.dtype}")
    # print(f"  ids              {ids}")
    out = model(ids)
    # print(f"  out              {tuple(out.shape)}  dtype {out.dtype}")
    # print(f"  out              {out}")
    
    print(f"  word table       {tuple(model.token.weight.shape)}"
          f"  = {model.token.weight.numel():3} values")
    print(f"  position table   {tuple(model.position.weight.shape)}"
          f"  = {model.position.weight.numel():3} values")
    print(f"  total learnable  {sum(p.numel() for p in model.parameters())}")
    print()
    print(f"  ids              {tuple(ids.shape)}  dtype {ids.dtype}")
    print(f"  token emb        {tuple(model.token(ids).shape)}")
    print(f"  pos emb          {tuple(model.position(torch.arange(BLOCK_SIZE)).shape)}")
    print(f"  sum              {tuple(out.shape)}")
    print()

    # ---- a batch: several sequences at once -------------------------------
    def short(vector, keep=3):
        """First few numbers of a vector, so the output stays readable."""
        return "[" + ", ".join(f"{v:6.3f}" for v in vector[:keep]) + ", ...]"

    batch = torch.tensor([EXAMPLES[i][0] for i in range(3)])
    batch_out = model(batch)

    print("a batch: three training examples stacked")
    print("  INPUT")
    for i, row in enumerate(batch.tolist()):
        print(f"    sequence {i}:  {row}  {[ITOS[t] for t in row]}")
    print(f"    shape {tuple(batch.shape)}  =  3 sequences x 4 tokens")
    print()

    print("  OUTPUT")
    for i in range(len(batch)):
        print(f"    sequence {i}:")
        for position in range(BLOCK_SIZE):
            word = ITOS[batch[i][position].item()]
            print(f"       position {position} ({word:7}) -> {short(batch_out[i][position])}")
    print(f"    shape {tuple(batch_out.shape)}  =  3 sequences x 4 positions x {D_MODEL} numbers")
    print("      B = batch (how many sequences)")
    print("      T = time  (how many positions in each)")
    print("      C = channels (numbers per position, i.e. d_model)")
    print()

    print("  the batch dimension is just independent copies, stacked:")
    print(f"    m(batch[0])            -> {tuple(model(batch[0]).shape)}")
    print(f"    equals batch_out[0]?      {torch.allclose(model(batch[0]), batch_out[0])}")
    print()

    print("  position rows are SHARED across the batch.")
    print("  subtract each position's word row and the same position row is left:")
    position_row_0 = model.position(torch.tensor(0))
    for i in range(len(batch)):
        word_row  = model.token(batch[i][0])
        remainder = batch_out[i][0] - word_row
        word      = ITOS[batch[i][0].item()]
        print(f"    sequence {i}:  out[{i},0] - word({word:7}) = {short(remainder)}"
              f"   matches position 0? {torch.allclose(remainder, position_row_0)}")
    print(f"    the position-0 row itself   = {short(position_row_0)}")
    print()

    # ---- stage 3 ----------------------------------------------------------
    print("stage 3: do the two agree?")
    load_plain_tables(model, word_table, pos_table)
    torch_out = model(ids)
    plain_out = torch.tensor(plain)
    print(f"  max difference   {(torch_out - plain_out).abs().max().item():.2e}")
    print(f"  agree            {torch.allclose(torch_out, plain_out, atol=1e-6)}")
    print()
    print("  nn.Embedding is a table and a lookup. The check proves it.")
