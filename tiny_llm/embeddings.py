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

    def show(vector, keep=4):
        """First few numbers of a vector, so the screen stays readable."""
        numbers = ", ".join(f"{v:7.3f}" for v in list(vector)[:keep])
        return f"[{numbers}, ...]"

    x, _ = EXAMPLES[0]
    words = [ITOS[i] for i in x]

    print("=" * 70)
    print("THE PROBLEM")
    print("=" * 70)
    print()
    print(f"  Our first training example is:  {words}")
    print(f"  As token IDs:                   {x}")
    print()
    print("  A model does arithmetic. It cannot multiply the word 'the' by")
    print("  anything. And the IDs are no help either - 'the' is 12 only")
    print("  because the vocabulary was sorted alphabetically, so 12 does not")
    print("  mean 'more than' 5.")
    print()
    print("  So we give every word its own row of numbers instead.")
    print()

    word_table = build_table(VOCAB_SIZE, D_MODEL, SEED)
    pos_table  = build_table(BLOCK_SIZE, D_MODEL, SEED + 1)

    print("=" * 70)
    print("STEP 1 - LOOK UP EACH WORD  (plain Python, no PyTorch)")
    print("=" * 70)
    print()
    print(f"  The word table has one row per vocabulary word: {VOCAB_SIZE} rows,")
    print(f"  {D_MODEL} numbers each. Looking a word up just means taking its row.")
    print()
    for position, token_id in enumerate(x):
        print(f"    {words[position]:7} is word {token_id:2}  ->  {show(word_table[token_id])}")
    print()
    print("  Notice positions 0 and 3 are both 'the', so they got the SAME row.")
    print("  The model cannot tell them apart. That is a problem: the first")
    print("  'the' and the last 'the' should not mean the same thing.")
    print()

    print("=" * 70)
    print("STEP 2 - ADD SOMETHING THAT SAYS WHERE THE WORD IS")
    print("=" * 70)
    print()
    print(f"  A second table, one row per slot: {BLOCK_SIZE} rows, {D_MODEL} numbers each.")
    print("  It is looked up by POSITION, not by word.")
    print()
    for position in range(BLOCK_SIZE):
        print(f"    slot {position}  ->  {show(pos_table[position])}")
    print()
    print("  Now add the two rows together, number by number:")
    print()
    plain = embed_plain(x, word_table, pos_table)
    for position, token_id in enumerate(x):
        print(f"    position {position} ({words[position]}):")
        print(f"        word row  {show(word_table[token_id])}")
        print(f"      + slot row  {show(pos_table[position])}")
        print(f"      = result    {show(plain[position])}")
    print()
    print("  Positions 0 and 3 are still both 'the', but look at the results:")
    print(f"      position 0  {show(plain[0])}")
    print(f"      position 3  {show(plain[3])}")
    print()
    print("  Different. That is the whole point of the second table.")
    print()

    print("=" * 70)
    print("STEP 3 - THE SAME THING IN PYTORCH")
    print("=" * 70)
    print()
    torch.manual_seed(SEED)
    model = Embeddings()
    ids = torch.tensor(x)
    out = model(ids)

    print("  nn.Embedding is a table plus a lookup. Nothing more.")
    print()
    print(f"    word table      {VOCAB_SIZE} rows x {D_MODEL} numbers = "
          f"{model.token.weight.numel():3} numbers to learn")
    print(f"    slot table      {BLOCK_SIZE} rows x {D_MODEL} numbers = "
          f"{model.position.weight.numel():3} numbers to learn")
    print(f"    total                                  "
          f"{sum(p.numel() for p in model.parameters()):3}")
    print()
    print("  Phase 1 had two numbers to learn: m and c. This has 144, and")
    print("  none of them will be set by hand.")
    print()
    print("  What went in and what came out:")
    print(f"    in    {len(x)} token IDs                     shape {tuple(ids.shape)}")
    print(f"    out   {len(x)} rows of {D_MODEL} numbers          shape {tuple(out.shape)}")
    print()
    print("  The number of positions did not change. Only what each one holds.")
    print()

    print("=" * 70)
    print("STEP 4 - DO THE TWO VERSIONS AGREE?")
    print("=" * 70)
    print()
    print("  Copy the plain-Python tables into PyTorch, so both hold the exact")
    print("  same numbers. If the logic matches, the results must match.")
    print()
    load_plain_tables(model, word_table, pos_table)
    torch_out = model(ids)
    plain_out = torch.tensor(plain)
    difference = (torch_out - plain_out).abs().max().item()
    print(f"    plain Python  {show(plain_out[0])}")
    print(f"    PyTorch       {show(torch_out[0])}")
    print()
    print(f"    biggest difference anywhere: {difference:.2e}")
    print()
    print("  That is float32 rounding, not a mistake - Python keeps about 16")
    print("  digits, PyTorch keeps about 7. Anything below 1e-6 is the same")
    print("  number as far as PyTorch can tell.")
    print()
    print("  So nn.Embedding really is just a table and a lookup. Proved,")
    print("  not assumed.")
    print()

    print("=" * 70)
    print("STEP 5 - SEVERAL SEQUENCES AT ONCE (a batch)")
    print("=" * 70)
    print()
    batch = torch.tensor([EXAMPLES[i][0] for i in range(3)])
    batch_out = model(batch)

    print("  Three training examples, stacked:")
    for i, row in enumerate(batch.tolist()):
        print(f"    sequence {i}:  {[ITOS[t] for t in row]}")
    print()
    print(f"    in    shape {tuple(batch.shape)}      3 sequences, 4 tokens each")
    print(f"    out   shape {tuple(batch_out.shape)}   3 sequences, 4 positions, {D_MODEL} numbers")
    print()
    print("  Nothing new happens. The sequences never interact - stacking")
    print("  them is a convenience. Running one on its own gives the same:")
    print(f"    m(batch[0]) == batch_out[0]  ->  {torch.allclose(model(batch[0]), batch_out[0])}")
    print()
    print("  And every sequence uses the SAME slot rows. Sequence 0 starts")
    print("  with 'the', sequence 1 with 'king', sequence 2 with 'ruled' -")
    print("  three different words. Take each result and subtract its word")
    print("  row, and see what is left over:")
    print()
    slot_row_0 = model.position(torch.tensor(0))
    for i in range(len(batch)):
        token = batch[i][0]
        leftover = batch_out[i][0] - model.token(token)
        print(f"    sequence {i} ({ITOS[token.item()]:7}) leftover  {show(leftover)}")
    print(f"    the slot-0 row itself           {show(slot_row_0)}")
    print()
    print("  Same leftover every time. The slot rows do not depend on which")
    print("  word is there - slot 0 means 'first', whatever sits in it.")
    print()
