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
#print(VOCABULARY)
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

    print("=" * 70)
    print("THE PROBLEM")
    print("=" * 70)
    print()
    print("  A neural network multiplies and adds numbers. It cannot do")
    print("  anything with the word 'kingdom'.")
    print()
    print("  So before any model exists, every word has to be turned into a")
    print("  number. That is this file's whole job.")
    print()

    print("=" * 70)
    print("STEP 1 - SPLIT THE TEXT INTO WORDS")
    print("=" * 70)
    print()
    lines = TEXT.strip().splitlines()
    print(f"  The corpus is {len(lines)} sentences:")
    for line in lines:
        print(f"      {line}")
    print()
    print(f"  Split on spaces and we have {len(TOKENS)} words in total.")
    print(f"  The first ten:  {TOKENS[:10]}")
    print()

    print("=" * 70)
    print("STEP 2 - KEEP ONE COPY OF EACH WORD")
    print("=" * 70)
    print()
    print(f"  {len(TOKENS)} words, but many repeat. Keeping only the distinct ones")
    print(f"  leaves {VOCAB_SIZE}. That list is called the VOCABULARY:")
    print()
    for word in VOCABULARY:
        print(f"      {word}")
    print()
    print("  It is sorted on purpose. A Python set has no reliable order, so")
    print("  without sorting the numbering could come out different on another")
    print("  run - and a model trained with one numbering is meaningless under")
    print("  another.")
    print()

    print("=" * 70)
    print("STEP 3 - GIVE EACH WORD A NUMBER")
    print("=" * 70)
    print()
    print("  Number them by their place in the sorted list:")
    print()
    for word in VOCABULARY:
        print(f"      {word:11} -> {STOI[word]:2}")
    print()
    print("  And the same thing backwards, so we can turn numbers into words")
    print("  again later when the model produces some:")
    print()
    for token_id in (0, 5, 12, 13):
        print(f"      {token_id:2} -> {ITOS[token_id]!r}")
    print()

    print("=" * 70)
    print("STEP 4 - DOES IT SURVIVE A ROUND TRIP?")
    print("=" * 70)
    print()
    print("  Turn every word into its number and back again. If any word comes")
    print("  back different, the mapping is broken.")
    print()
    broken = [w for w in VOCABULARY if ITOS[STOI[w]] != w]
    if broken:
        print(f"      BROKEN for: {broken}")
    else:
        print(f"      All {VOCAB_SIZE} words survive the round trip.")
    print()
    print(f"  The numbers run 0 to {VOCAB_SIZE - 1} with no gaps, which matters because")
    print("  they will be used as row numbers into a table at L3. A gap would")
    print("  mean a wasted row; a number too big would be an error.")
    print()

    print("=" * 70)
    print("STEP 5 - THE SCORE TO BEAT, BEFORE ANY MODEL EXISTS")
    print("=" * 70)
    print()
    counts = Counter(TOKENS)
    print("  How often each word appears:")
    print()
    for word, n in counts.most_common():
        bar = "#" * n
        print(f"      {word:11} {n:2}  {bar}")
    print()
    common, n = counts.most_common(1)[0]
    targets = TOKENS[1:]
    baseline = targets.count(common) / len(targets)
    print(f"  {common!r} is {n} of the {len(TOKENS)} words - far more than any other.")
    print()
    print("  So imagine a 'model' that reads nothing, understands nothing, and")
    print(f"  answers {common!r} every single time. On the job we actually care")
    print("  about - given some words, guess the next one - it would be right:")
    print()
    print(f"      {targets.count(common)} times out of {len(targets)}  =  {baseline:.1%}")
    print()
    print("  Write that number down now. When the real model reports its score")
    print("  at L11, anything at or below 34% means it learned which word is")
    print("  common and nothing else.")
    print()
    print(f"  For comparison, guessing at random from {VOCAB_SIZE} words would be right")
    print(f"  about {1 / VOCAB_SIZE:.1%} of the time.")
    print()
