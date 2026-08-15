"""L4 stages 2 and 3 - the same attention head, written with tensors.

attention.py holds stage 1: the five steps as plain Python loops, every number
printable. This file is the same five steps as tensor operations, and then the
check that the two produce identical numbers.

That check is the point. A tensor version that looks right is not the same as one
that computes what the loops computed - and a wrong one rarely crashes, it just
returns plausible numbers of the correct shape. The loops are the reference; this
is the thing being verified against them.

Nothing new is happening here. Same arithmetic, same weights, loops handed to the
library:

    1.  Q = x @ W_Q                        one multiply instead of T x D_HEAD dots
    2.  scores = Q @ K.T / sqrt(D_HEAD)    one multiply instead of T x T dots
    3.  scores.masked_fill(future, -inf)
    4.  scores.softmax(dim=-1)
    5.  weights @ V

TWO TRAPS, both of which make the stage-3 check fail for reasons that are not
logic errors:

  - nn.Linear(8, 4) stores its weight as (4, 8), TRANSPOSED relative to
    build_matrix(8, 4). Loading one into the other needs .T
  - nn.Linear adds a bias by default. The plain version has none, so these are
    built with bias=False
"""

import math

import torch
from torch import nn

from attention import D_HEAD, SEED, T, attention_plain, build_matrix, example_input
from embeddings import D_MODEL


class CausalSelfAttention(nn.Module):
    """One causal self-attention head.

    Accepts (T, d_model) or (batch, T, d_model). The batch dimension is carried
    along untouched - every sequence gets the same weights and the same mask.
    """

    def __init__(self, d_model=D_MODEL, d_head=D_HEAD, block_size=T):
        super().__init__()
        # bias=False so this matches the plain version, which has no bias
        self.to_query = nn.Linear(d_model, d_head, bias=False)
        self.to_key   = nn.Linear(d_model, d_head, bias=False)
        self.to_value = nn.Linear(d_model, d_head, bias=False)
        self.d_head = d_head

        # True where a position must NOT look. Registered as a buffer so it
        # moves with the module but is not a learned parameter - it is a fixed
        # fact about position, the same for every sequence in every batch.
        future = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("future", future)

    def forward(self, x):
        seq_len = x.shape[-2]

        queries = self.to_query(x)                       # (..., T, d_head)
        keys    = self.to_key(x)
        values  = self.to_value(x)

        scores = queries @ keys.transpose(-2, -1)        # (..., T, T)
        scores = scores / math.sqrt(self.d_head)

        blocked = self.future[:seq_len, :seq_len]
        scores = scores.masked_fill(blocked, float("-inf"))

        # dim=-1 means "along each row". Get this wrong and it normalises down
        # the columns instead, which produces plausible numbers that mean
        # nothing.
        weights = scores.softmax(dim=-1)

        return weights @ values, {                       # (..., T, d_head)
            "queries": queries, "keys": keys, "values": values,
            "scores": scores, "weights": weights,
        }


def load_plain_matrices(module, w_query, w_key, w_value):
    """Copy stage 1's lists into the tensor module.

    .T because nn.Linear stores (out_features, in_features) while build_matrix
    produces (in_features, out_features).
    """
    with torch.no_grad():
        module.to_query.weight.copy_(torch.tensor(w_query).T)
        module.to_key.weight.copy_(torch.tensor(w_key).T)
        module.to_value.weight.copy_(torch.tensor(w_value).T)
    return module


if __name__ == "__main__":
    x_rows, words = example_input()
    w_query = build_matrix(D_MODEL, D_HEAD, SEED + 1)
    w_key   = build_matrix(D_MODEL, D_HEAD, SEED + 2)
    w_value = build_matrix(D_MODEL, D_HEAD, SEED + 3)

    x = torch.tensor(x_rows)
    model = load_plain_matrices(CausalSelfAttention(), w_query, w_key, w_value)
    out, parts = model(x)

    plain_out, plain_parts = attention_plain(x_rows, w_query, w_key, w_value)

    def show(row):
        return "[" + ", ".join(f"{v:7.3f}" for v in row) + "]"

    def grid(rows, labels, fmt="{:9.4f}"):
        print("               " + "".join(f"{w:>10}" for w in labels))
        for i, row in enumerate(rows):
            cells = "".join(
                "      -inf" if v == float("-inf") else fmt.format(v) + " "
                for v in row
            )
            print(f"    {labels[i]:>9} {cells}")

    print("=" * 74)
    print("THE SAME FIVE STEPS, IN FIVE LINES")
    print("=" * 74)
    print()
    print("  attention.py spends about forty lines of loops on this. Here it is")
    print("  with the loops handed to the library:")
    print()
    print("        Q = x @ W_Q                        (T x 8) @ (8 x 8) = (T x 8)")
    print("        scores = Q @ K.T / sqrt(8)         (T x 8) @ (8 x T) = (T x T)")
    print("        scores.masked_fill(future, -inf)")
    print("        weights = scores.softmax(dim=-1)")
    print("        out = weights @ V                  (T x T) @ (T x 8) = (T x 8)")
    print()
    print("  Nothing new is happening. Same arithmetic, same weights - the only")
    print("  difference is who writes the loops.")
    print()

    print("=" * 74)
    print("WHAT CAME OUT")
    print("=" * 74)
    print()
    print("  scores, after masking:")
    print()
    grid(parts["scores"].tolist(), words)
    print()
    print("  shares, after softmax:")
    print()
    grid(parts["weights"].tolist(), words)
    print()
    print("  the answer:")
    print()
    for i in range(T):
        print(f"        {words[i]:>7}  {show(out[i].tolist())}")
    print()

    print("=" * 74)
    print("STAGE 3 - DO THE TWO VERSIONS AGREE?")
    print("=" * 74)
    print()
    print("  Both were given the exact same weights, so if the logic matches")
    print("  the numbers must match. Comparing every intermediate, not just the")
    print("  final answer - a difference in step 2 could still cancel out by")
    print("  step 5 and hide a real bug:")
    print()
    checks = [
        ("queries", parts["queries"], plain_parts["queries"]),
        ("keys",    parts["keys"],    plain_parts["keys"]),
        ("values",  parts["values"],  plain_parts["values"]),
        ("weights", parts["weights"], plain_parts["weights"]),
        ("output",  out,              plain_out),
    ]
    for name, tensor_value, plain_value in checks:
        plain_tensor = torch.tensor(plain_value)
        difference = (tensor_value - plain_tensor).abs().max().item()
        agree = torch.allclose(tensor_value, plain_tensor, atol=1e-6)
        print(f"        {name:9} biggest difference {difference:.2e}   agree: {agree}")
    print()
    print("  Those differences are float32 rounding - Python keeps about 16")
    print("  digits, PyTorch about 7. Anything below 1e-6 is the same number as")
    print("  far as PyTorch can tell.")
    print()
    print("  So the tensor version computes exactly what the loops computed.")
    print("  Proved, not assumed - which is the whole reason stage 1 exists.")
    print()

    print("=" * 74)
    print("TWO TRAPS WORTH KNOWING")
    print("=" * 74)
    print()
    print("  Both make the check above fail for reasons that are not logic")
    print("  errors, which is when people wrongly conclude the check is broken.")
    print()
    print(f"  1. nn.Linear stores its weight TRANSPOSED.")
    print(f"        build_matrix(8, 8) gives          (8, 8)  as (in, out)")
    print(f"        nn.Linear(8, 8).weight is         {tuple(model.to_query.weight.shape)}  as (out, in)")
    print("     load_plain_matrices uses .T for exactly this reason. With a")
    print("     square matrix the shapes still line up, so a missing .T does")
    print("     not crash - it silently computes something else.")
    print()
    print("  2. nn.Linear adds a bias unless told not to. The plain version has")
    print("     none, so these are built with bias=False:")
    print(f"        our layer's bias:  {model.to_query.bias}")
    print()

    print("=" * 74)
    print("A BATCH, FOR FREE")
    print("=" * 74)
    print()
    batch = torch.stack([x, x, x])
    batch_out, _ = model(batch)
    print(f"        one sequence   {tuple(x.shape)}      -> {tuple(out.shape)}")
    print(f"        three stacked  {tuple(batch.shape)}   -> {tuple(batch_out.shape)}")
    print(f"        batch[0] matches the single run? {torch.allclose(batch_out[0], out, atol=1e-6)}")
    print()
    print("  The plain version would need another loop for this. The tensor")
    print("  version needed no change at all - every sequence gets the same")
    print("  weights and the same mask, so the extra dimension is just carried")
    print("  along.")
    print()
    print("  That is the real reason to move to tensors. Not elegance - it is")
    print("  that batching, and running on a GPU, come without rewriting")
    print("  anything.")
    print()
