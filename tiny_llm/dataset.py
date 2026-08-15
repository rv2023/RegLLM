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
    print(f"tokens in corpus   {len(IDS)}")
    print(f"block size         {BLOCK_SIZE}")
    print(f"training examples  {len(EXAMPLES)}    (= {len(IDS)} - {BLOCK_SIZE})")
    print()

    print("encode / decode round trip")
    sentence = "the king ruled the kingdom"
    ids = encode(sentence)
    print(f"  encode({sentence!r})")
    print(f"    -> {ids}")
    print(f"  decode(...) -> {decode(ids)!r}")
    print(f"  round trip holds: {decode(encode(sentence)) == sentence}")
    print()
    #print(EXAMPLES)
    print("first three examples")
    for x, y in EXAMPLES[:3]:
        print(f"  x={x}  {[ITOS[i] for i in x]}")
        print(f"  y={y}  {[ITOS[i] for i in y]}")
        print()

    print("what one example actually teaches")
    x, y = EXAMPLES[0]
    print(f"  {'position':>8}  {'model sees':32}  {'must predict':>12}")
    for position in range(len(x)):
        context = " ".join(ITOS[i] for i in x[:position + 1])
        print(f"  {position:>8}  {context:32}  {ITOS[y[position]]:>12}")
    print()
    print(f"  {len(x)} lessons from one forward pass, not {len(x)} separate examples.")
