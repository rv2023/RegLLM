# Q&A Notes — Phase 2, the tiny language model

Questions asked while building the tiny GPT-style model, and the answers, kept so
they don't have to be re-derived. Organised by topic rather than by date.
Milestone references point at [MILESTONES.md](../MILESTONES.md).

Phase 1's notes live separately in
[qa-notes-regression.md](qa-notes-regression.md) — gradients, optimizers,
batching, regularization. Everything there still applies; this file covers only
what is new about language models.

Every number here was produced by running the project's own code.

Covers L1–L5.

---

## Part 1 — Tokenization

### What a tokenizer is, and why it is not learned

A neural network does arithmetic. It cannot multiply `"kingdom"` by a weight.
Everything from L3 onward operates on numbers, so something must convert text
into integers first.

**It is the one component of a language model that is never learned.** Every
weight is found by gradient descent; the vocabulary is *chosen*, before training
starts, and the model is permanently stuck with it. A word missing from the
vocabulary can never be produced, however long you train.

### Its size sets the model's shape

`vocab_size` is not just a statistic:

- at **L3** it is the height of the embedding table — one row per token
- at **L7** it is the width of the output layer — the model emits one logit per
  possible next token

Change the vocabulary and the model's architecture changes with it. This is why
tokenizer choices are made once, at the start, and are painful to revisit.

### The kingdom dataset

```
lines            8
total tokens     42
vocabulary size  14
vocabulary       ['a', 'castle', 'entered', 'had', 'in', 'king', 'kingdom',
                  'lived', 'prince', 'protected', 'queen', 'ruled', 'the', 'visited']

  the 15   kingdom 5   castle 4   king 3
  ruled 2  queen 2     prince 2   lived 2   in 2
  entered 1  visited 1  had 1     a 1       protected 1
```

### The dataset's repetitiveness is the design, not a limitation

A first instinct is to write "better" sentences with more variety. That breaks
the milestone. Compared against a set of eight generic English sentences of
similar length:

```
generic   tokens= 47  vocab= 43  ratio=1.09  words appearing once: 40/43
kingdom   tokens= 42  vocab= 14  ratio=3.00  words appearing once:  5/14
```

A language model learns "given this context, predict the next token" from
**repeated structure**. The kingdom text has it deliberately: `"the ___ ruled the
kingdom"` occurs with both `king` and `queen`; `"the king ___"` occurs with
`ruled`, `lived` and `protected`. That recurrence is what makes
`"the king ruled the"` -> `"kingdom"` learnable at all.

With near-unique words there is exactly one example of nearly every transition —
nothing to generalise from, only memorisation, and no way at L11 to tell the two
apart. The larger vocabulary also makes the output layer wider with less evidence
per class.

### Sorting the vocabulary is not cosmetic

```python
vocabulary = sorted(set(tokens))
```

A Python set has **no defined order**, so building IDs by iterating one directly
can produce different assignments on different runs. At L3 the embedding table is
indexed by those IDs — different IDs make a saved model meaningless.

Sorting makes the mapping a deterministic function of the text. Same discipline
as seeding the RNG in R6.

### The two mappings

```python
stoi = {token: index for index, token in enumerate(vocabulary)}
itos = {index: token for index, token in enumerate(vocabulary)}
```

`stoi` (string to integer) for encoding, `itos` for decoding back. Verified:

```
round trip: itos[stoi[w]] == w for every word   -> True
ids are exactly 0..13                            -> True
stoi['a'] = 0   stoi['castle'] = 1   stoi['the'] = 12   stoi['visited'] = 13
```

Contiguous IDs from `0` matter because they will be used as **row indices** into
the embedding table. A gap would mean an unused row; an ID at or above
`vocab_size` would be an out-of-bounds lookup.

### Unknown tokens, and why this dataset needs none

The vocabulary is built from the entire corpus being processed, so every token is
captured by `set()` and indexed. An out-of-vocabulary token is impossible **by
construction**.

That is a property of a *closed* dataset, not a general one. What changes
otherwise:

1. Reserve a special token (e.g. `<UNK>`) in the vocabulary during training.
2. On unseen data, any token missing from `stoi` falls back to that index via a
   safe lookup: `stoi.get(token, stoi['<UNK>'])`.
3. Deliberately replace a small percentage of low-frequency words with `<UNK>`
   during training, so the network learns how to handle it. Otherwise `<UNK>`
   exists in the vocabulary but the model has never seen one.

### The majority-class baseline: 34.1%

`the` is 15 of 42 tokens. A "model" with no parameters that reads nothing and
always answers `"the"` is right about a third of the time:

```
   after the       -> true next = king      | always-say-"the" is wrong
   after king      -> true next = ruled     | always-say-"the" is wrong
   after ruled     -> true next = the       | always-say-"the" is RIGHT
   after kingdom   -> true next = the       | always-say-"the" is RIGHT

always predicting "the": 14/41 = 34.1% correct
random guess (1 of 14 words): 7.1%
```

Two figures, and the second is the one that matters: `15/42 = 35.7%` of all
tokens are `the`, but on the **next-token task** it is `14/41 = 34.1%`, since the
first token has no predecessor. That is the task the model performs.

**Write it down before there is a model to be impressed by.** At L11 an accuracy
number means nothing without it:

| model accuracy | what it actually means |
| --- | --- |
| 7% | no better than random guessing |
| 25% | **worse than saying "the" every time** |
| 34% | learned that "the" is common. Nothing about context. |
| 60% | genuinely using context |

This is the **majority-class baseline**. The classic cautionary case: a screening
model for a disease affecting 1% of people is **99% accurate** by always
answering "healthy" — a perfect-looking metric with zero clinical value that
never detects a single case.

It is also why `"the king ruled the"` -> `"kingdom"` is the interesting test at
L10. `kingdom` is not the most common word, so producing it requires reading the
context. A model that only ever generates `"the the the the"` has found the
baseline and stopped — a real failure mode for undertrained language models, not
a hypothetical.

### Module structure

`tiny_llm/` holds **shared modules**, not per-milestone scripts. Unlike the
regression files, `tokenizer.py` is imported by every milestone from L2 to L11;
duplicating it would recreate exactly the drift that let the R3 sign error
survive three review rounds.

Two things that follow, both learned the hard way in Phase 1:

- **Build the data path from `__file__`**, not relative to the working
  directory. `open("data/kingdom.txt")` works from `tiny_llm/` and fails
  everywhere else — including from every later milestone that imports it.
  `Path(__file__).parent / "data" / "kingdom.txt"` is stable.
- **Guard the prints with `if __name__ == "__main__":`.** This module is imported
  by ten later milestones; unguarded output would fire on every one of them.

---

## Part 2 — Training examples: encode, decode, shifted targets

### Where the shift actually happens

Not in any display code — in the slice offsets inside `build_examples`:

```python
(ids[i : i + block_size],  ids[i + 1 : i + block_size + 1])
      ^                          ^^^^^
   starts at i                starts at i+1
```

That `+1` is the entire mechanism. Both slices are the same length; `y`'s simply
begins one position later.

```
the ID stream, with positions:
  index        0        1        2        3        4        5
  id          12        5       11       12        6       12
  word       the     king    ruled      the  kingdom      the

example i=0, block_size=4:
  x = ids[0:4] ->     the     king    ruled      the
  y = ids[1:5] ->             king    ruled      the  kingdom

example i=1:
  x = ids[1:5] ->             king    ruled      the  kingdom
  y = ids[2:6] ->                     ruled      the  kingdom      the
```

Three consequences:

- **`x` and `y` mostly overlap** — three of four tokens are shared. Only `y`'s
  last token is new. `y` is not a different sequence; it is the same stream,
  offset. Asserted in one line: `y[:-1] == x[1:]`.
- **`y[i]` is always the token following `x[i]`.** Every position carries its own
  answer.
- **Consecutive examples overlap too** — example `i=1`'s `x` *is* example `i=0`'s
  `y`. The window advances one token at a time.

### Every position is a training example

A language model does not produce one prediction per sequence. **It produces one
at every position** — at L7 the output shape is `[sequence_length, vocab_size]`,
one vocab-sized row per input position.

So a single forward pass over `x = [the, king, ruled, the]` gives:

```
  position  model sees                        must predict
         0  the                                       king
         1  the king                                 ruled
         2  the king ruled                             the
         3  the king ruled the                     kingdom
```

Four predictions, four targets, four errors, one pass. The loss is their average.

### Why not (context, single_target) pairs?

Storing it the obvious way:

```
(["the"],                        "king")
(["the","king"],                 "ruled")
(["the","king","ruled"],         "the")
(["the","king","ruled","the"],   "kingdom")
```

gives the same four lessons but needs **four forward passes**, each discarding
everything except its final position's output.

It is not more *data* — the corpus is identical either way. It is four times the
**compute** to extract the same lessons. At `block_size=4` that is a 4× saving; at
a real model's `block_size=1024` it is 1024×, and training the other way would be
flatly impossible. This is why every language model uses the shifted form.

### And this is where causal masking comes from

Look at position 1. Its target is `"ruled"`, and it must predict that from
`"the king"` — but the full input `[the, king, ruled, the]` sits in front of the
model, with `"ruled"` at position 2.

If position 1 can see position 2, the task is free: the answer is in the input.
The model would score perfectly during training and generate nonsense at
inference, where future tokens genuinely do not exist yet.

**The causal mask blocks each position from seeing anything after itself.** That
is what makes "predict every position at once" legitimate rather than cheating,
and it is the constraint implemented at L4.

### The off-by-one

```
len(build_examples(ids, block_size)) == len(ids) - block_size
```

Not `- block_size + 1`. The final window needs one token *beyond* its own end to
have a target, so it stops one position earlier. Measured: 42 tokens give 39
examples at `block_size=3`, 38 at 4, 37 at 5.

The test that actually catches an error here is not the count — it is
`test_every_token_transition_is_covered`, which builds the set of all 41 adjacent
pairs in the corpus and the set of all `(x[i], y[i])` pairs across every example
and asserts they are equal. An off-by-one at either end drops a transition; a
count test alone would not notice.

### Continuous stream or per-line?

The corpus is eight separate sentences, so whether windows may span line breaks is
a real choice:

| | examples at `block_size=4` |
| --- | --- |
| continuous (all 42 tokens as one stream) | **38** |
| per-line (windows stay inside a sentence) | 10 |

**Continuous was chosen.** Nearly 4× the training data from the same corpus, it is
what real language models do, and in this text every line begins with `"the"`, so
cross-boundary transitions are consistent rather than noise. If generation runs
sentences together at L10, this is why.

### Materialising the examples is a convenience, not the norm

38 examples of 4 tokens is 152 stored tokens from a 42-token corpus — the overlap
is real duplication. It is accepted here because it is simple and inspectable.

Real implementations do not materialise the list at all: they keep the token
stream once and slice a window on demand when a batch is needed, picking random
offsets into the stream. That is what L9 will do.

---

## Part 3 — Embeddings, tensors, and autograd

*L3.*

### Why token IDs cannot be fed in directly

`"the" = 12` and `"king" = 5`. Those numbers came from sorting the vocabulary
alphabetically. They are **labels, not values**.

A model does arithmetic. Give it `12` and it treats that as bigger than `5` —
more than twice as big. Meaningless: `"the"` is not twice `"king"`. And `"the"=12`
sits beside `"visited"=13` only because of the alphabet, not because the words are
related.

### The fix: one row of numbers per word

Instead of one number per word, give each word a **list of numbers**, stored in a
table with one row per word. With three numbers per word for legibility (the real
thing uses eight):

```
  id  word          the 3 numbers for that word
   0  a             [0.69, 0.52, -0.16]
   5  king          [-0.5, 0.82, 0.97]
  11  ruled         [0.61, 0.1, -0.97]
  12  the           [0.44, -0.2, 0.65]
  13  visited       [0.34, -1.0, -0.01]

looking up a word is just indexing the table:
  table[12] = [0.44, -0.2, 0.65]   (the)
  table[5]  = [-0.5, 0.82, 0.97]   (king)
```

**That table is the embedding. That is all it is.** The lookup is list indexing —
no clever operation. It is mathematically equivalent to multiplying a one-hot
vector by a weight matrix, which is how textbooks introduce it and how nobody
implements it: a matrix multiply that is thirteen multiplications by zero and one
by one is a waste of a lookup.

Two facts about those numbers:

- **They start random** — the values above are literally `random.uniform(-1, 1)`.
  Nothing in there knows what any word means yet.
- **They are learned.** Gradient descent adjusts them exactly as it adjusted `m`
  and `c`. `king` and `queen` drift toward similar rows because they occur in
  similar places and the loss falls when they do. Nobody programs that in.

### The position problem

The first training example is `x = [12, 5, 11, 12]` — `the king ruled the`.

`"the"` is at position 0 **and** position 3. Both look up `table[12]`. Both get
`[0.44, -0.2, 0.65]` — bit-for-bit identical. The model has no way to tell the
first word from the fourth.

This matters because attention (L4) is **permutation-invariant**: shuffle the
input and it computes the same thing. Without positional information,
`the king ruled the` and `ruled the the king` are the same input.

### The second table

One row per **position**, not per word:

```
position 0 -> [-0.19, -0.60, -0.64]
position 1 -> [-0.50,  0.52, -0.50]
position 2 -> [-0.23,  0.37,  0.08]
position 3 -> [ 0.88, -0.02, -0.16]
```

The two tables are indexed differently:

| | rows | looked up by |
| --- | --- | --- |
| **word table** | 14 (one per vocabulary word) | the token ID — `12` for `"the"` |
| **position table** | 4 (one per slot in the window) | where it sits — `0, 1, 2, 3` |

**The position table does not care which word is there.** Position 0's row is
always the same, whatever token occupies it. Every training example uses the same
four position rows and different word rows.

### Adding them

For each position: look up the word's row, look up the position's row, add them
number by number.

```
  position 0  word 'the'
     word_table[12] = [0.44, -0.2, 0.65]
     pos_table[0]   = [-0.19, -0.6, -0.64]
     added          = [0.25, -0.8, 0.01]

  position 3  word 'the'
     word_table[12] = [0.44, -0.2, 0.65]     <- same row
     pos_table[3]   = [0.88, -0.02, -0.16]
     added          = [1.32, -0.22, 0.49]    <- different total
```

Same word, different totals. The tie is broken.

The result — four rows of numbers — **is the model's actual input**. From L4
onward the transformer never sees `"the"` or `12` again, only these vectors.

### Why the position rows must be the same width

Because they are **added**, and addition needs matching sizes. A 2-number position
row and a 3-number word row leaves no third number to add to.

So the width is not chosen independently. Pick one width — call it `d_model`, 8 in
this project, 768 in GPT-2 — and every table uses it.

### Why add rather than concatenate?

Concatenation is the obvious alternative: stick them side by side, letting
position use fewer numbers. It is a real design and some early models used it.

Addition won because of what comes later. At **L6** the transformer block does:

```
x = x + Attention(x)
```

That `+` requires `Attention(x)` to be exactly as wide as `x`, so the width has to
stay constant through the whole model. Concatenating at each stage would grow it
without bound. Once the width is pinned, adding is free and keeps it pinned.

### The part that should feel odd

Two unrelated things — *which word* and *where it sits* — are added into the same
numbers. That looks like it should scramble them irrecoverably.

It does not, because there is room. With eight numbers there are many independent
directions in that space, and the model learns to use some for word identity and
others for position. They coexist rather than collide. With one or two numbers it
genuinely would break; with 768 there is ample space.

Worth being slightly suspicious of. It works better than it has any right to, and
"just add them" is one of those choices justified mainly by the fact that it
works.

### Sizes

```
word table       14 rows x 8 numbers = 112 values
position table    4 rows x 8 numbers =  32 values
                                       ---
                                       144 learnable numbers
```

Phase 1 had two parameters, both watched by hand. This has 144, and none of them
will be set by hand.

### Shapes

```
x tensor       (4,)      int64      <- IDs are indices, so integers
token emb      (4, 8)
pos emb        (4, 8)
sum            (4, 8)
```

Four tokens in, four vectors of eight numbers out. **The sequence length is
preserved through every step** — embeddings change *what* each position holds,
never *how many* positions there are. That stays true all the way to L7.

### Built twice, then checked

Per the scalars-before-matrices rule, `embeddings.py` has three stages:

1. **Plain Python** — `build_table`, `lookup`, `add_rows`, `embed_plain`. About
   twenty lines, no PyTorch. `lookup` is a single line,
   `[table[i] for i in indices]`, which is the entire embedding operation.
2. **PyTorch** — `Embeddings(nn.Module)` holding two `nn.Embedding` tables.
3. **The cross-check** — `load_plain_tables` copies the stage-1 lists into the
   module's weights, so both hold identical numbers and any output difference
   would be a logic difference.

```
stage 3: do the two agree?
  max difference   1.19e-07
  agree            True
```

That is the point of the milestone: `nn.Embedding` is a table and a lookup, and
this is proof rather than a claim.

### What `1.19e-07` means

It is **float32 machine epsilon** — the smallest relative step float32 can
represent, `2**-23 = 1.1920928955078125e-07`. The measured difference is exactly
that, which is the giveaway.

The two stages store numbers at different precision:

- **Python floats are float64** — about 16 significant digits
- **PyTorch defaults to float32** — about 7

```
a Python float (float64):  0.6888437030500962
stored as float32       :  0.6888437271118164
lost in the conversion  :  2.406e-08
```

Copying the list into `.weight.data` rounds every value to the nearest float32.
Add two of them and the total can be off by about one epsilon. **Not an error and
nothing to fix** — the same computation, held in containers of different
precision.

This is why the test uses a tolerance:

```python
assert torch.allclose(torch_out, plain_out, atol=1e-6)
```

`==` would fail. `1e-6` sits comfortably above `1.19e-07`, so real logic errors
still fail while float32 rounding does not. Same rule as the finite-difference
check in R3: **compare floats with a tolerance, chosen above the noise floor of
the representation.**

Where it stops being trivia:

```
  torch.float64     epsilon 2.220e-16    bytes 8
  torch.float32     epsilon 1.192e-07    bytes 4
  torch.float16     epsilon 9.766e-04    bytes 2
  torch.bfloat16    epsilon 7.812e-03    bytes 2
```

At L12, moving to 16-bit halves memory and speeds up arithmetic — bfloat16's
epsilon is about 65,000× coarser than float32's. Numbers agreeing to `1.19e-07`
today would agree to two or three decimal places there. That is why mixed
precision keeps gradient accumulation and optimizer state in float32 while
running the bulk in 16-bit.

bfloat16 and float16 are both 2 bytes but split the bits differently: bfloat16
trades precision for range, keeping float32's exponent so it does not overflow.
That is why it is preferred for training, where gradients span many orders of
magnitude.

### Why PyTorch at all, when stage 1 works?

For L3 alone it is not needed. The reason is **autograd**.

Phase 1 derived `dL/dm` and `dL/dc` by hand — two parameters. L3 has 144, and by
L7 it is thousands, through attention and softmax and several layers. Deriving
those by hand is not a matter of effort; it is not practical.

PyTorch records every operation performed on a tensor and computes all gradients
from one call:

```
loss = m(x).sum()
loss.backward()          # <- one line, 144 gradients

  token row   word        gradient sum
     5        king           8.0   <- appeared once
    11        ruled          8.0   <- appeared once
    12        the           16.0   <- appeared twice (positions 0 and 3)
   (all other rows)          0.0
```

**`the` got exactly twice the gradient**, because it appeared at two positions
and its contributions accumulated. That is the multi-path chain rule from Phase 1
happening automatically — the same rule worked out by hand for `dL/dm` summing
over five data points. Words absent from the input get `0.0`; they contributed
nothing, so they get no update.

Secondary reasons: speed (tensor ops run in optimised C/BLAS), and the building
blocks needed next — `nn.Linear` for Q/K/V at L4, `softmax` at L4 and L7,
`cross_entropy` at L8.

**Phase 1 was not wasted by this.** Autograd is exactly what was done by hand, so
when gradients misbehave at L8 there are three tools available that most people
lack: knowing gradients accumulate (hence `zero_grad`), knowing what a sign error
looks like in a loss curve, and being able to run a finite-difference check on any
single parameter to test whether autograd's answer is right.

### block_size is a hard ceiling

```
dataset.IDS has 42 ids;  position table has 4 rows
model(torch.tensor(IDS)) ->  IndexError: index out of range in self
model(torch.tensor(IDS[:4])).shape = (4, 8)
```

`forward` computes `torch.arange(T)`, so 42 tokens asks for position rows 0–41.
Rows 4–41 do not exist. The *word* table is fine — all 42 IDs are valid words. It
is the position table that runs out.

That number is the **context window**. Not a soft preference or a tuning knob:
there is no row for position 5, so position 5 cannot be represented. It is the
same figure quoted for real models — 1024 for GPT-2, 128k+ today. When a model
"cannot handle a longer prompt", this is usually why, and extending it means new
parameters that were never trained.

**This is also why L2 windowed the data.** The corpus is 42 tokens and the model
sees 4, so the stream is chopped into 38 overlapping windows of exactly
`BLOCK_SIZE`. The two constants must agree — `dataset.BLOCK_SIZE` (how wide each
example is) and `embeddings.BLOCK_SIZE` (how many position rows exist). Defined
separately they could silently drift; one should import from the other.

### Shape is part of a tensor's identity

```
a = [[1.0, 2.0]]      shape (1, 2)    one row,  two columns
b = [[1.0], [2.0]]    shape (2, 1)    two rows, one column
same numbers? yes.   same tensor? False

a @ b  ->  (1, 1)  =  [[5.0]]                    the dot product
b @ a  ->  (2, 2)  =  [[1.0, 2.0], [2.0, 4.0]]   the outer product
```

Same two vectors; swapping the order gives a single number or a 2x2 matrix.
**Matrix multiplication is not commutative** — order changes the result and the
shape.

The rule:

```
(n, k) @ (k, m)  ->  (n, m)
     ^     ^
     these must match, and they disappear
```

Both orders are legal, which is why getting it backwards does not crash. It
silently produces the wrong shape and surfaces several layers later.

Why it matters at L4:

```
Q     (4, 8)      one row per position
K.T   (8, 4)      transposed
      ------
      (4, 4)      one score for every PAIR of positions
```

That `(4, 4)` is the attention matrix — row `i`, column `j` is "how much should
position `i` attend to position `j`", and it is what the causal mask applies to.
Compute `K.T @ Q` instead and you get `(8, 8)`: a valid matrix of complete
nonsense relating embedding dimensions to each other rather than positions.

This is why Codex.md insists on predicting shapes by hand before writing
attention.

### The batch dimension: [B, T, C]

```
input   (3, 4)      3 sequences  x  4 tokens each
output  (3, 4, 8)   3 sequences  x  4 positions  x  8 numbers each
```

The embedding added a dimension — each ID became 8 numbers — and left the others
alone. Nothing new happens: `m(batch[0])` gives `(4, 8)`, exactly matching
`out[0]`. The batch dimension is several independent copies of the same
computation, stacked.

The standard naming, used everywhere from here to L11:

- **B** — batch: how many sequences
- **T** — time: how many positions in each
- **C** — channels: how many numbers per position, i.e. `d_model`

Position rows are **shared across the batch**. Subtracting the word row from
`out[i, 0]` gives the same position-0 row in every sequence, because position 0
means "first slot" regardless of the sentence. `torch.arange` does not look at
the tokens, so broadcasting copies the same rows across the batch for free.

Why batch at all: the R8 answer. One update from several examples instead of one
each, with the sequences processed in parallel. At L9 the same trade-off returns
— bigger batches give less noisy gradients but fewer updates per pass.

---

## Part 4 — Attention

*L4 — concepts worked through before implementation.*

### The problem it solves

After L3 each position's vector says two things: **which word** and **which
position**. Nothing else.

```
position 0   "the"     [ 1.4, -3.8]
position 1   "king"    [-0.7,  0.3]
position 2   "ruled"   [-1.1,  1.9]
```

Position 2 describes `"ruled"` and contains nothing about `"the"` or `"king"`
sitting before it. **Every position is an island.** To predict what comes next it
has to gather information from earlier positions. That is attention's job.

### Why not just average the earlier vectors?

That would mix in context, but it treats every earlier word as equally
important. Predicting after `"the king ruled the"`, the word `"king"` matters far
more than the first `"the"`. A flat average throws that away.

So: **a weighted average, where the weights depend on relevance.**

### Query, key, value

Every position produces three things:

| | plain meaning | library analogy |
| --- | --- | --- |
| **query** | what I am looking for | your search term |
| **key** | what I advertise | a book's index card |
| **value** | what I actually contribute | the book's contents |

Position `i` compares **its query** against **every position's key**. Good match
-> high weight. It then reads the **values** in proportion to those weights.

Key and value are separate because *being findable* and *being useful* are
different jobs. A book's title helps you find it; the contents are what you take
away.

### Shared matrices, per-position vectors

These are two different objects and running them together causes confusion:

- **3 matrices** — `W_Q`, `W_K`, `W_V`. One set for the whole layer. Shared.
- **3 vectors per position** — each position gets its own query, key and value by
  putting **its own** `x` through those shared matrices.

```
W_Q (shared by all positions) = [[2, 0], [0, 3]]

   the     x=[1, 0]  @ W_Q  ->  query=[2, 0]
   king    x=[0, 1]  @ W_Q  ->  query=[0, 3]
   ruled   x=[1, 1]  @ W_Q  ->  query=[2, 3]

one matrix, three different queries - because the inputs differ.
```

Same recipe, different ingredients, different results. Those three matrices are
**the only learned parameters in attention** — the embedding says what the word
is, and the matrices decide what it looks for, offers and contributes.

### The five steps

```
1.  query, key, value  =  x @ W_Q,  x @ W_K,  x @ W_V
2.  scores             =  every query DOT every key        -> (T, T)
3.  mask               =  set future scores to -inf
4.  weights            =  softmax each row                  -> rows sum to 1
5.  output             =  weights @ values                  -> (T, d_head)
```

Everything else in L4 is those five with shapes attached.

### Worked through, by hand

Three positions, two numbers each, hand-picked so every number is readable.

```
STEP 0 - what each position has
   the     query=[1, 0]  key=[1, 0]  value=[10, 0]
   king    query=[0, 1]  key=[0, 1]  value=[0, 20]
   ruled   query=[1, 1]  key=[1, 1]  value=[5, 5]

STEP 1 - score = my query DOT your key. Position 2 ("ruled"), query=[1,1]:
   vs key of the    [1, 0]:  1*1 + 1*0 = 1
   vs key of king   [0, 1]:  1*0 + 1*1 = 1
   vs key of ruled  [1, 1]:  1*1 + 1*1 = 2

STEP 2 - do that for every position. One score per PAIR:
              looking at:  the  king  ruled
   the    is asking        1     0     1
   king   is asking        0     1     1
   ruled  is asking        1     1     2

STEP 3 - MASK. Position i may not look at anything after i.
              looking at:   the   king  ruled
   the    is asking          1   -inf   -inf
   king   is asking          0      1   -inf
   ruled  is asking          1      1      2

STEP 4 - SOFTMAX each row. exp(-inf) = 0, so blocked positions get EXACTLY zero.
              looking at:   the   king  ruled     sum
   the    attention    1.00   0.00   0.00    1.00
   king   attention    0.27   0.73   0.00    1.00
   ruled  attention    0.21   0.21   0.58    1.00

STEP 5 - OUTPUT = weighted average of the VALUES (not the keys).
   position 0 (the):    1.0000*10 + 0.0000*0 + 0.0000*5   =  10.0000
                        1.0000*0 + 0.0000*20 + 0.0000*5   =   0.0000
   position 1 (king):   0.2689*10 + 0.7311*0 + 0.0000*5   =   2.6894
                        0.2689*0 + 0.7311*20 + 0.0000*5   =  14.6212
   position 2 (ruled):  0.2119*10 + 0.2119*0 + 0.5761*5   =   5.0000
                        0.2119*0 + 0.2119*20 + 0.5761*5   =   7.1194
```

Reading it:

- **Position 0 gets its own value back unchanged.** It is first; there is nothing
  else to look at, so the weighted average has one term.
- **Every row of weights sums to 1.00.** That is what makes it an *average*. A
  position can redistribute its attention but never manufacture more.
- **The weights form a triangle** — 1 non-zero, then 2, then 3. That shape *is*
  causality.

A rounding trap: the display shows `0.21` and `0.58`, but the true weights are
`0.211942` and `0.576117`. Hand-checking with the rounded values will not
reconcile.

### The count that grows is positions, not output width

The triangle (1, 2, 3 non-zero weights) is **how many positions each one may look
at**. It is not the size of the output.

Every output is 2 numbers, because every value is 2 numbers and an average of
2-number vectors is a 2-number vector:

```
3 weights  x  3 values (2 numbers each)  ->  1 output (2 numbers)
```

The 3 is **consumed by the summing** — the same collapse as `np.mean` turning 40
residuals into one gradient in Phase 1.

### Why the mask exists — the concrete version

This is the part that took several attempts. Forget the mechanics; look at what
`y` is.

```
   input  x = ['the', 'king', 'ruled', 'the']
   target y = ['king', 'ruled', 'the', 'kingdom']

   position 0 must predict 'king'   -> 'king'  is at x[1]
   position 1 must predict 'ruled'  -> 'ruled' is at x[2]
   position 2 must predict 'the'    -> 'the'   is at x[3]
```

`y` is `x` shifted one position left — that is how L2 built it. So for every
position but the last, **the correct answer is sitting one slot to the right in
the input.**

Without a mask, position 0 can look at position 1 and **just copy it**. It needs
to understand nothing. The model would learn one rule — *"output whatever is one
slot to my right"* — which scores 3 out of 4 in training, learns nothing about
language, and is useless at generation, because when generating **there is no
slot to the right.**

The mask removes the shortcut. Position 0 must actually predict `'king'` from
`'the'`.

### Why there is no slot to the right at generation

At training the whole sentence is already known — all four tokens go in at once.
At generation the sequence **grows**:

```
   start:  ['the', 'king']   <- this is all the text that exists
   step 0: model reads ['the', 'king']
           positions available: 0..1. There is no position 2 yet.
           model outputs -> 'ruled'
   step 1: model reads ['the', 'king', 'ruled']
           positions available: 0..2. There is no position 3 yet.
           model outputs -> 'the'
```

When the model is producing `'ruled'`, position 2 does not exist — inventing it is
the model's job.

It is like practising for an exam with the answer sheet in front of you: perfect
in practice, useless in the real thing, because what was learned was "read the
answer" rather than "work it out".

**The mask forces training to match generation.**

### Turning scores into shares: what softmax does

The scores are just numbers — some negative, no particular total. Shares are
needed instead: all positive, adding to 1, usable as proportions. Two steps get
there. Taking row 2 (`ruled`) of the real masked scores:

```
   scores   [-1.1599, 2.5561, -0.7664, -inf]

   STEP ONE - raise 2.718 to the power of each score (that is exp).
              always positive, and it grows fast:

        -1.1599   ->     0.3135    (the)
         2.5561   ->    12.8855    (king)
        -0.7664   ->     0.4647    (ruled)
           -inf   ->     0.0000    (the)
                     ---------
        total          13.6637

   STEP TWO - divide each by that total, so they add up to 1:

         0.3135 / 13.6637  =  0.0229    (the)
        12.8855 / 13.6637  =  0.9430    (king)
         0.4647 / 13.6637  =  0.0340    (ruled)
         0.0000 / 13.6637  =  0.0000    (the)
```

Those four numbers *are* row 2 of the weights grid. Note `exp(-inf) = 0.0000`
contributing nothing to the total — the mask's mechanism, in plain sight.

### Why the size of a score gap matters: exp turns a difference into a ratio

This is the link that makes the scaling argument readable. A gap between two
scores becomes a **ratio** after exp:

```
   gap between two scores    exp(gap)            the odds
         0.5                      1.6            1.6 to 1
           1                      2.7            2.7 to 1
           3                     20.1           20.1 to 1
          10                  22026.5        22026.5 to 1
          30         10686474581524.5           huge to 1
```

A gap of 3 is 20-to-1. A gap of 10 is 22,000-to-1. exp grows fast enough that a
modest gap becomes total dominance:

```
   two scores   0.5 apart  ->  shares 0.3775 / 0.6225
   two scores     3 apart  ->  shares 0.0474 / 0.9526
   two scores    10 apart  ->  shares 0.0000 / 1.0000
```

Which is why score size matters at all, and why `sqrt(d_head)` exists.

### This example does saturate, and the file says so

Honest state at `D_HEAD = D_MODEL = 8`:

```
unscaled           biggest gap  29.81   biggest weight on last row 1.0000
scaled by sqrt(8)  biggest gap  10.54   biggest weight on last row 0.9881
```

`10.5` is still past the point where one position takes nearly everything. The
scaling moved this run **from hopeless to merely unlucky** — it did not fix it.
Across 200 random starts the average biggest weight is `0.69` and about 30 in 200
saturate; this seed is one of the 30.

An earlier draft claimed "10.5, which is fine" while the grid directly above
showed `0.9991` and `0.9881`. The text contradicted its own output. Worth
remembering: **a claim next to a table has to survive reading the table.**

### What saturation actually costs

Not just "no gradient". Look at the outputs:

```
       position 0 after  [  1.185,  -0.698,   0.030,   0.989, ...]
       position 1 after  [  1.184,  -0.698,   0.029,   0.987, ...]
```

Nearly identical. Row 1's shares were `0.9991` on position 0 and `0.0009` on
itself, so `king` handed back almost a straight copy of `the` and kept nearly
none of its own value.

**A pinned softmax does not just stop learning — it throws information away.**

### What changed, and what deliberately did not

```
   in    4 positions x 8 numbers   (straight from L3)
   out   4 positions x 8 numbers
```

The **shape** is untouched, on purpose: the next stage expects the same shape
back, which is what makes `x = x + Attention(x)` possible at L6.

The **contents** are completely different:

```
    position 3 (the)
       before  [  1.569,  -1.244,  -0.249,  -0.724, ...]
       after   [ -0.288,   0.795,   2.393,  -1.723, ...]
```

Before, each row said only "I am this word, in this slot". After, each row is a
blend of the values of every position up to and including itself, in the
proportions from step 4.

A shape line alone reads as "nothing happened" when `D_HEAD = D_MODEL`. The
before/after rows are what show the work.

### Mechanically, the mask is nothing

```
scores for position 1:   [0, 1, -inf, -inf]
                               |
                          softmax
                               |
weights for position 1:  [0.27, 0.73, 0.00, 0.00]
```

A weight of `0.00` times that position's value contributes **exactly zero**. The
information is not reduced or discounted — it is absent.

`-inf` is used because `exp(-inf) = 0`, giving a true zero rather than something
merely small.

### Shapes

```
x         (3, 2)     3 positions, 2 numbers each
q, k, v   (3, 2)     one query/key/value per position
scores    (3, 3)     one per PAIR of positions
weights   (3, 3)     same, after masking and softmax
output    (3, 2)     back to one vector per position
```

**In `(3,2)`, out `(3,2)`.** Attention changes *what each position holds* — it now
contains gathered context — without changing how many positions there are or how
wide they are. That is what makes `x = x + Attention(x)` possible at L6.

Deferred to the implementation: scores are divided by `sqrt(d_head)` before
softmax. Easier to justify after watching softmax misbehave without it.

### Stage 1, built and run

`attention.py` implements the five steps in plain Python — lists and loops, no
PyTorch and no matrix syntax, so every intermediate is printable. Run on three
positions with `d_model=4`, `d_head=2` and random untrained weights:

```
scores (nothing blocked yet)
                the     king    ruled
      the    0.027   -0.186   -0.142
     king   -0.028   -0.210   -0.122
    ruled   -0.069   -0.063    0.004

after masking
                the     king    ruled
      the    0.027      -inf     -inf
     king   -0.028   -0.210      -inf
    ruled   -0.069   -0.063    0.004

after softmax
                the     king    ruled
      the   1.0000   0.0000   0.0000
     king   0.5453   0.4547   0.0000
    ruled   0.3245   0.3265   0.3490
```

Two things to notice.

**Untrained attention is nearly uniform.** Row 2 is `0.32 / 0.33 / 0.35` — an
almost equal split. Random weights produce scores clustered near zero, and
softmax over near-equal scores returns near-equal shares. At this point the head
is doing little more than averaging its visible positions. The *structure* is
correct and the *behaviour* is uninformative, which is exactly what an untrained
model should look like. Training is what makes those shares uneven.

**Position 0's output is its own value, unchanged.** Its weighted average has a
single term with weight `1.0000`. That is not a special case in the code — it
falls out of having nothing before it.

### The three sizes

```
                <-------------- D_MODEL = 8 numbers -------------->
        the       1.427  -3.795   0.272   0.486  -1.651  -1.347  -1.150   0.224
       king      -0.725   0.329   1.632  -0.350   0.647   3.467  -0.546  -0.199   <-- T = 4 rows
      ruled      -1.129   1.884  -2.687   0.516  -0.435  -0.490   1.984  -0.353
        the       1.569  -1.244  -0.249  -0.724  -2.715  -2.730  -0.541   0.526

    T       = 4   how many words we look at at once   (the rows)
    D_MODEL = 8   numbers describing each word        (the columns)
    D_HEAD  = 8   numbers per word once inside the head
```

**T never changes.** Four words in, four out. Only the width changes,
`D_MODEL -> D_HEAD`, and the step-1 matrices are what do it.

### How one query number is built

Every input number gets a say, so one output number needs one weight per input
number — 8 of them:

```
        input number       weight        product
            1.427     x     0.455   =    0.6499
           -3.795     x    -0.386   =    1.4655
            0.272     x     0.070   =    0.0191
            ...
                                        --------
          query number 0 =              -0.8531
```

That is a **dot product**: 8 numbers against 8 numbers giving **one** number.

### Why the matrix has that shape

8 weights produced one number. Eight output numbers need **8 sets of 8** = 64
weights, stored as a table with 8 rows and 8 columns where each column is one
set. That is all "matrix" means here — the sets, side by side.

**The shape is always `(numbers coming in) x (numbers going out)`.** True for the
Q/K/V projections here, the output projection at L5, the MLP at L6, and the LM
head at L7.

### A set of weights belongs to a query NUMBER, not to a word

This misreading came up repeatedly, and counting settles it:

```
4 words x 8 query numbers = 32 numbers altogether
but only 8 sets of weights. They cannot be per word.
```

The set for query number 0 builds the **first** number of *every* word; the set
for query number 1 builds the second number of every word. A set fills a
**column** of the query table, not a row.

```
                 out 0    out 1    out 2   ...
                 set 0    set 1    set 2
      the       -0.853    0.910   -0.639
     king        2.138    1.271    0.928
    ruled       -0.610   -3.659   -0.178
      the       -1.579   -0.403   -0.839

    ACROSS a row  - one word, all 8 sets -> its 8 query numbers
    DOWN a column - 4 different words, all using the SAME set
```

The word "set" invited the confusion. Naming things after **what they build** —
"the 8 weights for query number 3" — rather than numbering them removes it.

### The same weights are used for every word

There is one query matrix, not one per position:

```
        set 0:   0.455,  0.070,  0.113,  0.077,  0.076,  0.700,  0.254,  0.257

            the row x set 0, summed  ->   -0.8531
           king row x set 0, summed  ->    2.1384
          ruled row x set 0, summed  ->   -0.6098
            the row x set 0, summed  ->   -1.5786
```

Same weights every time; different rows in, different numbers out. **64 weights
per matrix however many words go through them.**

Note the two `the` rows give different answers, because L3 already made their
rows differ by slot.

### Three matrices, not one

The sets above are the **query** matrix only. Keys and values have their own,
built the same way with their own numbers and no weights shared:

```
        W_Q  builds queries    set 0:   0.455,  0.070,  0.113, ...
        W_K  builds keys       set 0:   0.827, -0.380,  0.421, ...
        W_V  builds values     set 0:   0.033, -0.015, -0.471, ...

        each matrix   8 sets x 8 weights = 64
        three of them                    = 192 weights in this head
```

Which is why one input row comes out as three different rows.

### From one word to the matrix form

The whole of step 1, built up from a single word:

```
one word:
    (1 x 8) row  .  (8) W_Q set 0   =  query number 0
    (1 x 8) row  .  (8) W_Q set 1   =  query number 1
    ...
    (1 x 8) row  .  (8) W_Q set 7   =  query number 7

collect them:
    (1 x 8) word row   @   (8 x 8) W_Q   =   (1 x 8) query

and since the SAME W_Q serves every word, stack the rows:
    (4 x 8) all words  @   (8 x 8) W_Q   =   (4 x 8) all queries
```

Verified: the stacked multiply gives numbers identical to the four separate
loops.

**That is stage 2 in one line.** The tensor version is not a different algorithm
— it is the same dot products with the loops handed to the library:

```
    Q = X @ W_Q        (4 x 8) @ (8 x 8) = (4 x 8)
    K = X @ W_K
    V = X @ W_V
```

Counting: 8 dots for a query, 8 for a key, 8 for a value = 24 per word, x 4 words
= **96 dot products**, using the **192 weights**.

Step 2 then does something different in kind — it dots the *queries against the
keys* rather than against weights. `Q @ K.T` is `(4x8) @ (8x4) = (4x4)`, one score
per pair of words. Get the transpose wrong and you compute `(8x4) @ (4x8) =
(8x8)`, a valid matrix relating *numbers* to each other instead of *words*.

### Why 8 numbers per word, and not 1?

Because of what one number cannot express. Keep only `W_K`'s first column, so
each word advertises a single value. A score is query times key, so try every
query there is:

```
        the  advertises   5.08
       king  advertises  -0.34
      ruled  advertises  -3.48
        the  advertises   5.80

        query =    -5   ->  top choice: ruled
        query =  -0.1   ->  top choice: ruled
        query =   0.1   ->  top choice: the
        query =     5   ->  top choice: the
```

**Only two words can ever win** — the biggest key and the smallest. A positive
query picks one, a negative query the other, and that is the entire range of
opinions available. Here that strands `'the' (5.08)` and `'king' (-0.34)`: two of
the four words **can never be anyone's first choice, whatever query you write.**

With 8 numbers a query can ask for a *combination* — high on this, low on that —
so any word becomes reachable, and each position can prefer a different order.

### A score is a dot product, not every-number-against-every-number

A natural misreading: "every Q number multiplies every K number, so 8 x 8 x 4
scores". It does not. Query number 0 meets key number 0, number 1 meets number 1,
and so on — **8 products, position-matched, summed to ONE number**:

```
cell [row 0, col 0] = query of "the" dotted with key of "the", scaled

   position     query      key     product
       0        -0.853    5.081    -4.3350
       1         0.910    1.811     1.6485
       2        -0.639    2.342    -1.4968
       3        -0.153   -0.419     0.0642
       4        -1.416    1.324    -1.8742
       5         2.189   -0.604    -1.3225
       6         1.887    1.030     1.9443
       7        -0.968    5.433    -5.2579
                                 ---------
   raw dot product:             -10.6294

   divide by sqrt(D_HEAD) = sqrt(8) = 2.8284
      -10.6294 / 2.8284 = -3.7581

   and the grid shows: -3.7581
```

So the counting is:

```
   one pair    = 8 products summed  -> 1 number
   16 pairs                          -> 16 numbers
   arranged as                          4 x 4
```

**The 8 disappears** because it is the axis being summed over — the inner
dimension of `Q @ K.T`, `(4x8) @ (8x4) = (4x4)`. What survives is one score per
pair of *words*, not per pair of numbers.

If it really were every-number-against-every-number you would compute
`(8x4) @ (4x8) = (8x8)` — a valid matrix relating *query numbers* to each other
rather than words. That is the transpose mistake, and it does not crash.

Reading the grid:

```
                      the      king     ruled       the
          the   -3.7581   -0.0886    3.7947   -2.5871
         king    6.0381   -0.9794   -4.2429    5.3038
        ruled   -1.1599    2.5561   -0.7664   -1.5451
          the   -5.9131    0.1924    4.6261   -4.3019

   row i     word i's query, tested against all four keys
   column j  word j's key, tested against all four queries
   diagonal  each word against itself
```

Nothing is blocked yet — row 0 has values in columns 1–3, which are `the`'s
future. Masking replaces those with `-inf` next.

### Every number traces back

Any value in this file can be followed to its source. `-0.336`, for instance, is
`king`'s key number 0:

```
"king" the word
   -> token id 5                         (tokenizer.py, L1)
   -> 8-number row: word row + slot row  (embeddings.py, L3)
   -> dot with W_K set 0                 (attention.py, step 1)
   = -0.336
```

The same row dotted with **`W_Q` set 0** instead gives `king`'s *query* number 0.
Same input, different weights, different meaning — one is what `king` advertises,
the other what it is looking for.

Every word gets all three:

```
    position 1  'king'
       row in (L3)   -0.725   0.329   1.632  -0.350   0.647   3.467  -0.546  -0.199
       query          2.138   1.271   0.928   0.623   0.617  -1.933   1.981  -0.373
       key           -0.336  -2.555  -2.011   2.761   1.215   0.069  -0.188  -2.944
       value          0.412  -0.815  -1.783  -0.394  -1.405   0.404  -1.276   1.022
```

Two things visible there. **Positions 0 and 3 are both `the` and all six of their
rows differ**, because the rows going in already differed by the slot numbers L3
added. And **key and value look nothing alike for the same word** — independent
matrices, so *what a word advertises* and *what it hands over* are unrelated. A
word can be easy to find for one reason and useful for another.

### Why exactly 8?

Nothing derives it. It is `D_MODEL`, inherited from L3, because one head is as
wide as its input. Measuring what width actually buys — how many of the 4
positions can want a different order, over 60 random starts:

```
         1 numbers per word  ->  1.98 distinct orders out of 4
         2 numbers per word  ->  3.05
         4 numbers per word  ->  3.50
         8 numbers per word  ->  3.55   <- ours
        16 numbers per word  ->  3.72
```

`1.98` at one number is the "only two orders exist" result, measured. The jump to
2 buys most of the gain; past 4 it flattens, because there are only 4 words to
rank. **8 is generous here rather than necessary.** Real models use 64 or 128 per
head because they rank thousands of positions from a 50,000-word vocabulary.

**How that count is worked out:** ask each word to rank all four words by score,
then count how many of those rankings are actually different. With one number:

```
       the ranks them: ['king', 'ruled', 'the', 'the']
      king ranks them: ['the', 'the', 'ruled', 'king']
     ruled ranks them: ['king', 'ruled', 'the', 'the']   <- same as an earlier row
       the ranks them: ['the', 'the', 'ruled', 'king']   <- same as an earlier row

   4 rankings, but only 2 are DIFFERENT
```

**Why two numbers escapes the trap.** With one number a query was just a sign.
With two it has a *direction*, and there are many directions:

```
   queries:  the [-0.22, 1.94]   king [-0.01, -1.38]
             ruled [0.45, -1.35]  the [0.04, 2.37]

       the ranks them: ['ruled', 'the', 'the', 'king']
      king ranks them: ['king', 'the', 'the', 'ruled']
     ruled ranks them: ['the', 'king', 'the', 'ruled']
       the ranks them: ['ruled', 'the', 'the', 'king']   <- same as an earlier row

   4 rankings, 3 DIFFERENT
```

`king` and `ruled` both point roughly downward, but their first numbers differ
(`-0.01` vs `0.45`) — enough to swap their top two choices. The duplicate that
remains is `the` at positions 0 and 3, both pointing up (`1.94` and `2.37`):
similar directions, same ranking.

| numbers | what a query is | distinct rankings |
| --- | --- | --- |
| 1 | a sign | 2 |
| 2 | a direction in a plane | 3 |
| 4+ | a direction in more room | ~3.5 of a possible 4 |

### How the weights start

```
        spread = 1 / sqrt(fan_in)
        weight = rng.gauss(0, spread)      for every cell
```

Random, centred on zero. The model starts with no preferences whatsoever;
training is what makes the numbers mean anything. Drawing 2000 of them:

```
        -0.40 to -0.15  #################################       403
        -0.15 to  0.00  ##########################              322
         0.00 to  0.15  #########################               300
         0.15 to  0.40  ####################################    442
```

Two thirds land within `+/-0.354`, effectively all within three times that.

The one real decision is the spread, and it matters. Measured against an input of
typical size 1.53:

```
   spread 1.000 (too big)      ->  typical output size   4.79
   spread 0.354 (1/sqrt(8))    ->  typical output size   1.69
   spread 0.050 (too small)    ->  typical output size   0.24
```

Only `1/sqrt(fan_in)` keeps the output about the size of the input. Bigger and
every layer amplifies; smaller and the signal fades.

### Two mistakes worth keeping

**A stand-in input hid a real problem.** The first draft used small invented
numbers because they fitted the screen. On L3's actual embeddings the score gap
reached 32, and softmax saturates past a gap of about 10 — so attention came out
winner-take-all, `1.0000 / 0.0000`. The toy example made a broken configuration
look healthy. Measured fixes:

```
  as written then                        score gap 32.56   0.118 0.000 0.000 0.882
  + divide by sqrt(d_head)               score gap 16.28   0.267 0.000 0.001 0.732
  + smaller init (std=1/sqrt(d_model))   score gap  7.20   0.008 0.947 0.008 0.038
  both fixes together                    score gap  3.60   0.067 0.724 0.065 0.144
```

The `sqrt(d_head)` scaling everyone quotes was **not sufficient on its own** —
the initialisation mattered more here.

(Those figures were measured at `D_HEAD = 4`. At the current `D_HEAD = 8` the
same run gives 29.8 unscaled and 10.5 scaled — see "This example does saturate"
above.)

**An invented number crept back in, at a smaller scale.** The "why not one
number" demo built a throwaway `build_matrix(8, 1, seed=12)` to produce its
"advertises" values — correct arithmetic on a made-up matrix, in a file that uses
the real `W_K` everywhere else. It now takes `W_K`'s actual first column, so the
hypothetical is honest: *what if the key matrix had only the one column it
already has?*

The knock-on was the giveaway. The section ended with a hardcoded "there is no
way to say 'I want ruled'" — but with the real numbers `ruled` is the *minimum*,
so it is reachable, and the stranded words are `the` and `king` instead. A
sentence that contradicted its own printed numbers. It is now computed from the
data rather than written down.

**The rule that would have prevented both:** if a real value is available, use
it. Never fabricate one for readability. The first invented input hid a
saturation bug; the second produced a false sentence.

**A one-seed test measures the seed.** `test_attention_is_not_saturated_at_
initialisation` checked a single random start. It passed at `D_HEAD=4` and failed
at `D_HEAD=8`, and that looked like evidence that wider heads saturate. Over 200
seeds at each width:

```
D_HEAD=4:  score std 2.35   saturated in 34/200 runs
D_HEAD=8:  score std 2.20   saturated in 30/200 runs
```

Identical. The `sqrt(d_head)` division does exactly its job, and the flip was
luck. Both tests now average over 100 seeds and assert something about the
distribution.

### D_HEAD = D_MODEL, and a rule broken

`D_HEAD` was set to 4 with a comment justifying it by L5's multi-head split. That
violates the one-milestone-at-a-time rule — importing a later milestone's
reasoning to justify a number here.

The honest value for a single head is `D_HEAD = D_MODEL`: narrowing it throws
information away for no gain, because there is nothing else to use the space.
The comment now states the *condition* under which it changes ("this becomes a
real choice when there is more than one head") without describing the later
design.

### What "head" and "layer" actually mean

Both words get used constantly without being defined. Pointing at the things
themselves:

**A HEAD is one complete copy of the machinery** — its own `W_Q`, `W_K`, `W_V`
(192 weights here) plus the five steps. In this example the head is exactly three
things:

```
1. its weights - 192 numbers it owns and nothing else uses
     W_Q  8 x 8   first row [0.455, 0.512, 0.023, -0.27] ...
     W_K  8 x 8   first row [0.827, -0.234, 0.14, 0.052] ...
     W_V  8 x 8   first row [0.033, 0.442, -0.329, 0.351] ...

2. its opinion - one row of shares per position
                  the      king     ruled       the
      the    1.0000    0.0000    0.0000    0.0000
     king    0.9991    0.0009    0.0000    0.0000
    ruled    0.0229    0.9430    0.0340    0.0000
      the    0.0000    0.0117    0.9881    0.0001

3. its answer - 4 rows of 8 numbers, each a blend of the values above it
```

**What the head does NOT own: the input rows.** Those came from L3, and a second
head would read exactly the same ones. That grid of shares *is* the head — one
opinion per position, and only one. A second opinion needs a second head with its
own 192 weights.

**A LAYER is all the heads at one stage, run together and combined** — one round
of every position gathering from the positions before it.

With one head, the layer *is* that head and the combining step is nothing. With
two heads it would be: run both on the same input rows, get two opinions and two
answers, then join the answers back together. Same input, different weights, so
genuinely different opinions:

```
    head 1 thinks the last position should look at:
      the=0.000  king=0.012  ruled=0.988  the=0.000
    head 2, same input, its own weights:
      the=0.169  king=0.063  ruled=0.301  the=0.466
```

### Heads sit side by side; layers stack

That is the whole distinction, and the consequences differ:

```
after LAYER 1: each row is itself + a blend of earlier rows
after LAYER 2: the rows it blends ALREADY carry context, so
               position 3 now draws on second-hand information too
```

- **More heads** — more simultaneous views of the *same* input.
- **More layers** — indirect relationships, because layer 2 reads rows that layer
  1 already filled with context, so a position can reach information it never
  looked at directly.

Counting weights the two ways:

```
    one head              192 weights
    a layer of 2 heads    384 weights   (heads side by side)
    6 such layers        2304 weights   (layers stacked)
```

This project is currently **1 head, 1 layer** — the smallest thing that works.

One honesty note: a real transformer layer holds more than attention — a small
feed-forward network, normalisation, and a residual add. Until those are
assembled, "layer" here means the attention part only.

### Did one head achieve anything?

The test: run the same head on two sentences differing by **one earlier word**.

```
        A:  the king ruled the
        B:  the queen ruled the      <- word 1 changed

  Position 3 is 'the' in both. Its row going IN:
        A  [  1.569,  -1.244,  -0.249,  -0.724, ...]
        B  [  1.569,  -1.244,  -0.249,  -0.724, ...]
        the same? True

  Its row coming OUT of attention:
        A  [ -0.288,   0.795,   2.393,  -1.723, ...]
        B  [ -0.331,   0.648,   2.257,  -1.753, ...]
        the same? False
```

Going in, position 3's row is **identical** in both — it is `"the"` at slot 3
either way, and L3 gave it no way to know what preceded it. Coming out, the two
differ.

**That is what one attention head did: it turned isolated positions into
context-aware ones.** The file opens with "every position is an island"; this is
the bridge.

And it is exactly what the task requires. To guess `kingdom` after
`the king ruled the`, position 3 must know `king` and `ruled` came earlier.
Before attention it could not; now it can.

### Head width does not have to divide anything (yet)

`(8 x 7)` works. So does `(8 x 3)` and `(8 x 12)`. The only hard requirement is
that the matrix takes `D_MODEL` numbers in, because that is what a word row has.
What comes out is a free choice.

A constraint appears once heads are combined, because their outputs are stuck
together and the total has to come back to `D_MODEL`:

```
   n_heads   d_head   total
       1         8       1 x 8 = 8   ok
       2         4       2 x 4 = 8   ok
       4         2       4 x 2 = 8   ok
       8         1       8 x 1 = 8   ok

      ?          7       need 8/7 = 1.14 heads   not a whole number
```

`7` does not divide `8`, so no whole number of heads totals 8. The reason it has
to total `D_MODEL` is that attention's output is added back onto the original
word row a step later, and a 7-number row cannot be added to an 8-number row.

Not about oddness or powers of two — just arithmetic. It is also why `D_MODEL` is
usually chosen with divisors in mind: GPT-2's 768 gives `12 heads x 64`, and
divides many other ways besides.

### Shapes are architecture, not something fine-tuning changes

| | what changes |
| --- | --- |
| **architecture** | the *shapes*: `D_MODEL`, `D_HEAD`, number of heads, number of layers |
| **training / fine-tuning** | the *values* inside those shapes |

Four separate knobs: `D_MODEL` is the width of the pipe running through the
model, `D_HEAD` the width inside one head, heads sit beside each other, layers
stack on top.

Training and fine-tuning both adjust the 64 numbers in `W_Q`. Neither can make it
56. Change `D_HEAD` from 8 to 7 and you have a **different model** — the old
weights cannot be loaded, because there is nowhere for them to go. Same reason
`vocab_size` was called painful to revisit at L1: anything that sets a shape is a
commitment.

Shapes are chosen once; then a great deal of compute is spent finding good values
for them. Fine-tuning continues that search from someone else's values. It never
reshapes anything.

### The test that actually matters

Rows summing to 1 and zeros above the diagonal are worth asserting, but neither
would catch information leaking backwards through a subtler route. This one does:

```python
def test_changing_a_later_token_cannot_move_an_earlier_output():
    before, _ = attention_plain(X, WQ, WK, WV)

    tampered = [row[:] for row in X]
    tampered[-1] = [9.9] * D_MODEL          # replace the LAST position entirely

    after, _ = attention_plain(tampered, WQ, WK, WV)

    for i in range(T - 1):
        assert after[i] == pytest.approx(before[i])    # earlier outputs frozen
    assert after[-1] != pytest.approx(before[-1])      # the last one SHOULD move
```

Replace the final position's input with garbage. Every **earlier** output must be
identical, because none of them were allowed to look at it. The last output must
change, because it was.

This tests the property the mask exists for — *the future cannot influence the
past* — rather than the mechanism that currently implements it. It would survive
a rewrite from `-inf` to any other masking approach, and it fails loudly if a
future refactor lets information flow the wrong way.

Also worth asserting: **each output lies within the range of the values it
averaged.** An average cannot land outside the things being averaged, so a
violation means the weights are not really weights.

---

## Part 5 — The tensor version, and proving it matches

*L4 stages 2 and 3, in `attention.py` - the module later milestones import.
`attention_plain.py` keeps the loops as the reference.*

### Forty lines of loops become five

```
    Q = x @ W_Q                        (T x 8) @ (8 x 8) = (T x 8)
    scores = Q @ K.T / sqrt(8)         (T x 8) @ (8 x T) = (T x T)
    scores.masked_fill(future, -inf)
    weights = scores.softmax(dim=-1)
    out = weights @ V                  (T x T) @ (T x 8) = (T x 8)
```

Nothing new happens. Same arithmetic, same weights — the only difference is who
writes the loops.

`dim=-1` on the softmax means "along each row". Get it wrong and it normalises
down the columns instead, which produces plausible numbers that mean nothing.

The causal mask is a **buffer**, not a parameter: it is a fixed fact about
position, identical for every sequence, and nothing about it is learned.

### The check is the milestone

Given identical weights, the two versions must agree. Comparing **every
intermediate**, not just the output — a difference in the scores could cancel out
by step 5 and hide a real bug:

```
        queries   biggest difference 2.38e-07   agree: True
        keys      biggest difference 4.77e-07   agree: True
        values    biggest difference 2.38e-07   agree: True
        weights   biggest difference 5.96e-08   agree: True
        output    biggest difference 2.38e-07   agree: True
```

Float32 rounding, as at L3. The tensor version computes exactly what the loops
computed — **proved, not assumed**, which is the whole reason stage 1 exists.

A tensor implementation that looks right is not the same as one that is right.
A wrong one rarely crashes; it returns plausible numbers of the correct shape.

### Two traps that break the check for non-reasons

Both make stage 3 fail without any logic being wrong — which is exactly when
people conclude the check itself is broken.

**`nn.Linear` stores its weight transposed.**

```
   build_matrix(8, 8) gives     (8, 8)  as (in, out)
   nn.Linear(8, 8).weight is    (8, 8)  as (out, in)
```

Loading one into the other needs `.T`. And here `d_head = d_model`, so the
matrix is **square** — a missing `.T` does not crash, does not even produce a
shape error. It silently computes something else. There is a test asserting the
un-transposed version *disagrees*, so the check is verified to be capable of
failing.

**`nn.Linear` adds a bias unless told not to.** The plain version has none, so
all three projections are built with `bias=False`.

### Batching came free

```
        one sequence   (4, 8)      -> (4, 8)
        three stacked  (3, 4, 8)   -> (3, 4, 8)
        batch[0] matches the single run? True
```

Zero code changes. The plain version would need another loop; the tensor version
carries the extra dimension along, because every sequence gets the same weights
and the same mask.

**That is the real argument for tensors** — not elegance. Batching, and running
on a GPU, arrive without rewriting anything.

---

## Part 6 — Multi-head attention

*L5, in `multihead.py`.*

### What one head cannot do

On our own numbers. Take head 0 and look at how position 3's answer is built:

```
   weights:  the=0.369  king=0.045  ruled=0.057  the=0.529

   out[0] = 0.369*-0.045 + 0.045* 1.331 + 0.057*-1.054 + 0.529*-1.146 =  -0.622
   out[1] = 0.369*-0.467 + 0.045*-0.606 + 0.057*-0.008 + 0.529* 0.065 =  -0.166
   out[2] = 0.369*-1.759 + 0.045* 0.805 + 0.057* 0.002 + 0.529*-0.790 =  -1.031
   out[3] = 0.369* 0.866 + 0.045* 0.840 + 0.057*-1.592 + 0.529*-0.668 =  -0.085
```

**The same four weights appear in every line.** A head's weights do not depend on
which output number is being produced, so all of them attend to exactly the same
places. One head means one opinion, applied to everything it outputs.

Now the two heads together, same position:

```
   out[0..3] used  0.369  0.045  0.057  0.529
   out[4..7] used  0.088  0.405  0.262  0.245
```

The first half leans on `the`; the second half leans on `king`. **Different parts
of the same answer looked at different words** — and that is precisely what a
single head cannot do, however wide you make it.

`W_O` then mixes those halves so a later layer sees one answer carrying both.

**More heads is not more capacity — it is more than one place to look at once.**

### Extra heads are free

```
        1 head  of 8:  Q/K/V  192 + W_O 64 = 256 weights
        2 heads of 4:  Q/K/V  192 + W_O 64 = 256 weights
        4 heads of 2:  Q/K/V  192 + W_O 64 = 256 weights
        8 heads of 1:  Q/K/V  192 + W_O 64 = 256 weights
```

Identical, every time. One head of 8 uses three `8x8` matrices; two heads of 4
use six `8x4` matrices. Same total, same arithmetic. So the question is never
"are extra heads worth the cost" — it is what to do with a fixed budget.

The catch is that each head gets narrower, and narrow heads are cruder. Using the
width measurement from Part 4:

```
        1 head  of 8   1 view,  each head ~3.6 distinct orders of 4
        2 heads of 4   2 views, each head ~3.5 - nearly as good
        8 heads of 1   8 views, each head 2.0 - almost useless
```

Splitting 8 into 2 costs almost nothing per head. Splitting into 8 destroys it —
back to the "only two orderings" trap. The middle wins, which is why GPT-2 uses
`12 heads x 64` rather than `1 x 768` or `768 x 1`.

### Why the heads get narrower

A word row is `d_model` numbers, and the layer's answer has to be usable wherever
the input was — fed onward, or combined with the original row. So it must come
back out as `d_model`. Since the heads are concatenated, that pins their total.

What happens if each head keeps the full width:

```
        2 heads x 8 numbers each, stuck side by side
        -> 16 numbers per word
        the layer received 8 and handed back 16.

   feed that to another such layer:
        RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x16 and 8x8)
```

Narrow each to 4 and the concatenation is 8, which feeds onward fine.

### It is one rule, and L4 was the n_heads=1 case

`attention_plain.py` outputs `(4 x 8)` and each head in `multihead.py` outputs
`(4 x 4)`. Not a contradiction — the same class with a different setting. **A
head's output width simply IS its `d_head`:**

```
   d_head = d_model / n_heads

   n_heads=1  d_head=8  each head gives (4 x 8),  1 of them = (4 x 8)   <- attention_plain.py
   n_heads=2  d_head=4  each head gives (4 x 4),  2 of them = (4 x 8)   <- multihead.py
   n_heads=4  d_head=2  each head gives (4 x 2),  4 of them = (4 x 8)
   n_heads=8  d_head=1  each head gives (4 x 1),  8 of them = (4 x 8)
```

`multihead.py` imports `CausalSelfAttention` and constructs it with `d_head=4`.
Nothing about the head changed, only the number passed to it. And the right-hand
column never moves — every split lands back at `d_model`.

### The whole chain, no batch

```
        input                 (4 x 8)
        head 0 output         (4 x 4)     <- narrow, not 8
        head 1 output         (4 x 4)
        concatenated          (4 x 8)
        after W_O             (4 x 8)

   in short:  (4x8) -> 2 x (4x4) -> (4x8) -> (4x8)
```

### Why there is an output projection

Concatenating alone leaves the heads next to each other, never mixed:

```
        [ -0.622,  -0.166,  -1.031,  -0.085,   0.100,  -0.153,  -0.024,   0.301]
         <------- head 0 -------> <------- head 1 ------->
```

Numbers 0–3 came **only** from head 0, numbers 4–7 **only** from head 1. The
heads sat side by side and never spoke.

`W_O` is a `(d_model x d_model)` matrix, so every output number becomes a blend
of every head. Without it a later layer would see two separate halves rather than
one combined answer.

### Explicit and fused, and the reshape that hides bugs

Both stages are tensors this time, which is a deliberate reading of the
scalars-before-matrices rule rather than a lapse. Nothing at L5 is new
arithmetic: running a head several times is a loop, concatenation is sticking
lists together, and `W_O` is the same operation `attention_plain.py` already
shows number by number.

What *is* new is a **reshape**, and no amount of Python loops teaches that. So
"explicit" here means a list of heads:

```
   stage 1   a list of CausalSelfAttention modules, concatenated, then W_O
   stage 2   one fused Linear per role, reshaped into heads
   stage 3   check the two agree
```

```
        joined rows   biggest difference 1.19e-07
        final output  biggest difference 8.94e-08
        agree: True
```

The fused version splits the last dimension into `(n_heads, d_head)` and moves
the head axis in front of `T`. Getting either wrong **does not crash** — it
quietly mixes the wrong numbers together. `test_split_heads_gives_each_head_its_own_slice`
uses `arange` rather than random data so the assertion is exact equality: head 0
must get columns 0–3 and head 1 columns 4–7. With random data a subtly wrong
reshape can still pass a tolerance check.

`test_one_head_reduces_to_the_l4_head` is the other guard: with `n_heads=1` and
`W_O` set to the identity, multi-head must reproduce the L4 head exactly. If that
fails, the plumbing is wrong before any of the interesting part.

---

## Part 7 — Scalars before matrices

The working rule for this phase: **build every component explicitly first, then
convert it to tensors.** Plain Python numbers and loops, on a hand-sized example,
covering the full path — and only then `Q @ K.T`.

This is the sequence Phase 1 used. Lists and loops carried R1–R4; NumPy arrived
at R6, and only after the list version had been checked against known values
(`274.0`, `-108.0`, `-32.0`). The list version was not scaffolding to be thrown
away — it was the reference that proved the vectorised one correct, and it caught
a real bug: the first NumPy `gradients` returned arrays of 40 because the
reduction was missing, which would have been invisible without something to
compare against.

What this means concretely:

| component | explicit first | then |
| --- | --- | --- |
| embedding lookup (L3) | index into a list of lists | a row of a matrix |
| attention (L4) | nested loops over a 3-token sequence, printing every score | `Q @ K.T / sqrt(d_k)` |
| causal mask (L4) | check `j > i` inside the loop | an upper-triangular mask |
| forward pass (L7) | one sequence, token IDs to logits to loss, by hand | batched tensors |

Why it is worth the extra pass:

- **Shapes stop being guesswork.** After computing attention for three tokens by
  hand you know what `[sequence, d_head]` contains, rather than trusting that the
  dimensions line up because the code ran.
- **A silent bug has a reference.** A matrix version that produces plausible
  numbers is nearly impossible to check alone. The scalar version is the
  finite-difference check of Phase 2.
- **The mechanism is visible once.** `softmax(QK^T/sqrt(d))V` is four operations
  hiding a great deal; a loop that prints each score before and after masking
  shows what causality actually does.

The matrix form is the destination — it is what runs, what scales, and what every
implementation you read will use. It is just not the starting point.

---

## Part 8 — Why Phase 1 had no softmax or temperature

Neither is missing by oversight — there was nothing for them to act on.

Regression outputs a **single number**: `y_hat = mx + c`. There is no
distribution and no choice to make; the prediction *is* the answer. Softmax turns
a vector of scores into a probability distribution, and temperature reshapes that
distribution before sampling from it. Both need a model that outputs
probabilities over discrete options.

That arrives in Phase 2:

- **L4** — softmax over attention scores, turning them into weights that sum to 1
- **L7** — logits over the vocabulary, one score per possible next token
- **L10** — temperature, then top-k, then top-p, added **one at a time**

The plan is explicit about adding the three sampling controls separately rather
than together, because their effects overlap and a combined change is hard to
attribute.
