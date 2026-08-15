# Q&A Notes — Phase 2, the tiny language model

Questions asked while building the tiny GPT-style model, and the answers, kept so
they don't have to be re-derived. Organised by topic rather than by date.
Milestone references point at [MILESTONES.md](../MILESTONES.md).

Phase 1's notes live separately in
[qa-notes-regression.md](qa-notes-regression.md) — gradients, optimizers,
batching, regularization. Everything there still applies; this file covers only
what is new about language models.

Every number here was produced by running the project's own code.

Covers L1–L2.

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

## Part 3 — Scalars before matrices

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

## Part 4 — Why Phase 1 had no softmax or temperature

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
