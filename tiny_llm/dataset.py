"""L2 - encoding, decoding, and shifted next-token training pairs.

The training signal for a language model is a sequence and the same sequence
shifted one position later:

    x = [the, king, ruled, the]
    y = [king, ruled, the, kingdom]

Every position in x is a training example, not just the last one. A model
processing this sequence emits logits at all four positions, and y supplies a
target for each - four lessons from one forward pass. Stored as
(context, single_target) pairs instead, the same four lessons would cost four
forward passes; at a real block_size of 1024 that factor makes training
impossible.

It is also where causal masking comes from (L4): position 1 must predict "ruled"
while "ruled" sits at position 2 of its own input. Without a mask the answer is
visible and the task is free.
"""

from tokenizer import ITOS, STOI, TEXT, tokenize

BLOCK_SIZE = 4      # context length: how many tokens the model sees at once


def encode(text):
    """Text -> list of token IDs.

    Tokenizes exactly as tokenizer.py does, so the two cannot drift. A word
    outside the vocabulary raises KeyError - the concrete face of the closed
    vocabulary from L1.
    """
    return [STOI[token] for token in tokenize(text)]


def decode(ids):
    """Token IDs -> text. Inverse of encode for any in-vocabulary input."""
    return " ".join(ITOS[i] for i in ids)


def build_examples(ids, block_size=BLOCK_SIZE):
    """Slide a window over the ID stream, pairing each window with itself
    shifted one position later.

    The corpus is treated as ONE CONTINUOUS STREAM, so windows may span line
    breaks. That gives 38 examples at block_size=4 rather than 10 for per-line
    windows, it is what real language models do, and in this corpus every line
    begins with "the", so cross-boundary transitions are consistent rather than
    noise. If generation ever runs sentences together (L10), this is why.

    The range stops at len(ids) - block_size, not one later: the final window
    needs a token beyond its own end to have a target.
    """
    return [
        (ids[i:i + block_size], ids[i + 1:i + block_size + 1])
        for i in range(len(ids) - block_size)
    ]


IDS      = encode(TEXT)
EXAMPLES = build_examples(IDS)


if __name__ == "__main__":

    print("=" * 70)
    print("THE PROBLEM")
    print("=" * 70)
    print()
    print("  L1 turned words into numbers. But a model cannot learn from a")
    print("  pile of numbers - it needs QUESTIONS and ANSWERS.")
    print()
    print("  The question we are training on is always the same one:")
    print("  given some words, what word comes next?")
    print()
    print("  This file builds those question-and-answer pairs.")
    print()

    print("=" * 70)
    print("STEP 1 - TEXT IN, NUMBERS OUT (AND BACK)")
    print("=" * 70)
    print()
    sentence = "the king ruled the kingdom"
    ids = encode(sentence)
    print(f"      encode({sentence!r})")
    print(f"          -> {ids}")
    print(f"      decode({ids})")
    print(f"          -> {decode(ids)!r}")
    print()
    if decode(encode(sentence)) == sentence:
        print("  The sentence comes back exactly as it went in, so nothing is")
        print("  lost in the conversion.")
    else:
        print("  BROKEN: the sentence did not come back unchanged.")
    print()
    print(f"  Doing that to the whole corpus gives {len(IDS)} numbers:")
    print(f"      {IDS[:12]} ...")
    print()

    print("=" * 70)
    print("STEP 2 - CUT IT INTO PIECES THE MODEL CAN HOLD")
    print("=" * 70)
    print()
    print(f"  The model can only look at {BLOCK_SIZE} words at a time. So slide a")
    print(f"  window of {BLOCK_SIZE} along the {len(IDS)} numbers, one step at a time.")
    print()
    print("  The first three windows:")
    print()
    for i in range(3):
        window = IDS[i:i + BLOCK_SIZE]
        print(f"      starting at {i}:  {[ITOS[t] for t in window]}")
    print()
    print("  Each window overlaps the one before it by all but one word.")
    print()

    print("=" * 70)
    print("STEP 3 - WHERE THE ANSWER COMES FROM")
    print("=" * 70)
    print()
    print("  For each window we need the answers. They are already in the text:")
    print("  the answer to 'what follows this word' is simply the NEXT word.")
    print()
    print("  So take the window, and take the same window shifted along by one:")
    print()
    x, y = EXAMPLES[0]
    print(f"      the question (x)  {[ITOS[t] for t in x]}")
    print(f"      the answers  (y)  {[ITOS[t] for t in y]}")
    print()
    print("  Line them up and you can see the shift:")
    print()
    print("      x:   the    king   ruled  the")
    print("      y:          king   ruled  the    kingdom")
    print()
    print("  y is x moved one place left. Only its last word is new.")
    print()

    print("=" * 70)
    print("STEP 4 - ONE PAIR TEACHES FOUR THINGS, NOT ONE")
    print("=" * 70)
    print()
    print("  It is tempting to read that pair as a single lesson. It is four.")
    print("  The model makes a guess at EVERY position, so every position has")
    print("  its own question and its own answer:")
    print()
    print(f"      {'the model can see':30} {'it must guess':>12}")
    for position in range(len(x)):
        seen = " ".join(ITOS[t] for t in x[:position + 1])
        print(f"      {seen:30} {ITOS[y[position]]:>12}")
    print()
    print("  Four lessons from one pass through the model. Storing them as four")
    print("  separate examples would need four passes for the same learning -")
    print("  and with a real model looking at 1024 words instead of 4, that")
    print("  difference is what makes training possible at all.")
    print()

    print("=" * 70)
    print("STEP 5 - HOW MANY PAIRS ALTOGETHER")
    print("=" * 70)
    print()
    print(f"      {len(IDS)} numbers in the corpus")
    print(f"    - {BLOCK_SIZE} for the window width")
    print(f"    = {len(EXAMPLES)} training pairs")
    print()
    print("  Not 39. The last window needs one extra word past its end to have")
    print("  an answer, so it has to stop one step early.")
    print()
    print("  The first three pairs:")
    print()
    for x_ids, y_ids in EXAMPLES[:3]:
        print(f"      x  {[ITOS[t] for t in x_ids]}")
        print(f"      y  {[ITOS[t] for t in y_ids]}")
        print()
