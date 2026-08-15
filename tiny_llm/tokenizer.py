"""L1 - word-level tokenizer for the medieval-kingdom dataset.

The tokenizer is the one component of a language model that is never learned.
Every weight is found by gradient descent; the vocabulary is chosen before
training starts, and the model is permanently stuck with it. A word missing here
can never be produced, however long you train.

Its size also sets the model's shape: vocab_size becomes the width of the output
layer (L7) and the height of the embedding table (L3).
"""

from pathlib import Path

# Relative to THIS file, not to whatever directory Python was started from, so
# every later milestone can import this module from anywhere in the repo.
DATA_PATH = Path(__file__).parent / "data" / "kingdom.txt"


def read_text(path=DATA_PATH):
    with open(path, "r") as file:
        return file.read()


def tokenize(text):
    """Word-level split on whitespace. The kingdom text has no punctuation, but
    stripping periods documents the intent and costs nothing. Note it handles
    periods only - not commas, apostrophes or anything else."""
    return [token.replace(".", "") for token in text.split()]


def build_vocab(tokens):
    """Sorted, so the IDs are a deterministic function of the text.

    A set has no defined order, so building IDs by iterating one directly can
    hand you different assignments on different runs - and at L3 the embedding
    table is indexed by those IDs, which would make a saved model meaningless.
    Same discipline as seeding the RNG in R6.
    """
    return sorted(set(tokens))


def build_mappings(vocabulary):
    stoi = {token: index for index, token in enumerate(vocabulary)}
    itos = {index: token for index, token in enumerate(vocabulary)}
    return stoi, itos


# Built once at import so later milestones can just `from tokenizer import stoi`.
TEXT       = read_text()
TOKENS     = tokenize(TEXT)
VOCABULARY = build_vocab(TOKENS)
VOCAB_SIZE = len(VOCABULARY)
STOI, ITOS = build_mappings(VOCABULARY)


# Unknown tokens are impossible here because the vocabulary is built from the
# entire corpus being processed - every token is captured by set() and indexed
# into STOI. That is a property of a CLOSED dataset, not a general one.
#
# What would change if it were not closed:
#   1. Reserve a special token (e.g. '<UNK>') in the vocabulary during training.
#   2. On unseen data, any token missing from STOI falls back to that index via a
#      safe lookup: STOI.get(token, STOI['<UNK>']).
#   3. Deliberately replace a small percentage of low-frequency words with
#      '<UNK>' during training, so the network learns how to handle it.


if __name__ == "__main__":
    from collections import Counter

    print(f"lines            {len(TEXT.strip().splitlines())}")
    print(f"total tokens     {len(TOKENS)}")
    print(f"vocabulary size  {VOCAB_SIZE}")
    print(f"vocabulary       {VOCABULARY}")
    print()

    print("round trip: itos[stoi[w]] == w for every word")
    print(f"  all match: {all(ITOS[STOI[w]] == w for w in VOCABULARY)}")
    print(f"  ids are exactly 0..{VOCAB_SIZE - 1}: {sorted(ITOS) == list(range(VOCAB_SIZE))}")
    print()

    print("spot checks")
    for word in ("a", "castle", "the", "visited"):
        print(f"  stoi[{word!r}] = {STOI[word]:2}   itos[{STOI[word]}] = {ITOS[STOI[word]]!r}")
    print()

    counts = Counter(TOKENS)
    print("token frequencies")
    for word, n in counts.most_common():
        print(f"  {word:10} {n:2}   {n / len(TOKENS):5.1%}")
    print()

    # The majority-class baseline. A model that ignores its input entirely and
    # always predicts the most common token would be right this often. Anything
    # at or below this at L11 has learned word frequency, not context.
    most_common, n = counts.most_common(1)[0]
    print(f"baseline: always predict {most_common!r} -> {n}/{len(TOKENS)} = {n / len(TOKENS):.1%}")
