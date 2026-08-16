"""L5 - several attention heads side by side, plus an output projection.

A NOTE ON THE SCALARS-BEFORE-MATRICES RULE. Earlier milestones built everything
twice: plain Python loops first, then tensors, then a check that the two agree.
This file keeps the two-stage discipline but BOTH stages are tensors, because
nothing here is new arithmetic:

    - running the same head several times is a loop
    - sticking the outputs side by side is concatenation
    - the output projection is the same operation as the Q/K/V projections,
      which attention_plain.py already shows number by number

What IS new and error-prone is a reshape, and no amount of Python loops teaches
that. So "explicit" here means a LIST of heads rather than a return to loops:

    stage 1   a list of CausalSelfAttention modules, concatenated, then W_O
              readable, and it reuses the head verified at L4
    stage 2   one fused Linear per role, reshaped into heads - what real
              implementations do
    stage 3   check the two agree

WHY MORE THAN ONE HEAD. A head's weights do not depend on which output number is
being produced, so every output number attends to exactly the same places. One
head has one opinion and applies it to everything it produces, and widening it
does not help - a wider head still has one row of weights per position. Several
heads let different parts of the same answer look at different words.

WHY d_head SHRINKS. A word row is d_model numbers, and the answer this layer
hands back has to be usable wherever the input was - fed to another layer, or
combined with the original row. So it has to come back out as d_model numbers.
Since the heads' answers are concatenated, that pins their total:

    n_heads x d_head = d_model          2 x 4 = 8

Keep each head at the full d_model and two heads give 16 numbers per word. The
layer received 8 and handed back 16, and feeding that to another such layer
fails: "mat1 and mat2 shapes cannot be multiplied (4x16 and 8x8)". With one head
there was nothing to share the width with, so it kept all of it.

WHY THERE IS AN OUTPUT PROJECTION. Concatenating alone leaves the heads sitting
next to each other, never mixed:

    [0.02, -0.152, -0.287, -0.444, | -0.315, 0.09, 0.057, -0.27]
     <-------- head 1 -----------> | <-------- head 2 ------->

Numbers 0-3 came only from head 1, numbers 4-7 only from head 2. W_O is an
(d_model x d_model) matrix that makes every output number a blend of every head.
Without it the heads never speak to each other.
"""

import math

import torch
from torch import nn

from attention import CausalSelfAttention
from attention_plain import SEED, T, example_input
from embeddings import D_MODEL

N_HEADS = 2                     # 2 x 4 = 8 = D_MODEL


class MultiHeadExplicit(nn.Module):
    """Stage 1: a plain list of heads, concatenated, then projected.

    Slower than the fused version - n_heads separate small matrix multiplies
    instead of one big one - but it is obvious what it does, which is what makes
    it useful as the reference.
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, block_size=T):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(
                f"d_model={d_model} is not divisible by n_heads={n_heads}. "
                f"The heads are concatenated, so n_heads * d_head must come "
                f"back to d_model."
            )
        d_head = d_model // n_heads

        self.heads = nn.ModuleList(
            CausalSelfAttention(d_model, d_head, block_size) for _ in range(n_heads)
        )
        self.to_output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        answers, per_head = [], []
        for head in self.heads:
            answer, parts = head(x)
            answers.append(answer)
            per_head.append(parts)

        joined = torch.cat(answers, dim=-1)          # (..., T, d_model)
        return self.to_output(joined), {"joined": joined, "heads": per_head}


class MultiHeadFused(nn.Module):
    """Stage 2: one Linear per role for ALL heads, then reshaped.

    Every head's query matrix is a slice of one (d_model x d_model) matrix, so
    all of them can be computed in a single multiply. The heads are then carved
    out of the last dimension.
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, block_size=T):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(
                f"d_model={d_model} is not divisible by n_heads={n_heads}."
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.to_query = nn.Linear(d_model, d_model, bias=False)
        self.to_key   = nn.Linear(d_model, d_model, bias=False)
        self.to_value = nn.Linear(d_model, d_model, bias=False)
        self.to_output = nn.Linear(d_model, d_model, bias=False)

        future = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("future", future)

    def split_heads(self, tensor):
        """(..., T, d_model) -> (..., n_heads, T, d_head).

        Two moves, and both matter:
          1. unflatten the LAST dimension into (n_heads, d_head), so head h owns
             numbers h*d_head .. (h+1)*d_head - the same slices the explicit
             version gives its separate heads
          2. move the head dimension in front of T, so the matrix multiplies
             below treat each head as an independent (T, d_head) problem
        """
        *leading, seq_len, _ = tensor.shape
        tensor = tensor.reshape(*leading, seq_len, self.n_heads, self.d_head)
        return tensor.transpose(-3, -2)

    def merge_heads(self, tensor):
        """(..., n_heads, T, d_head) -> (..., T, d_model). The inverse."""
        tensor = tensor.transpose(-3, -2).contiguous()
        *leading, seq_len, n_heads, d_head = tensor.shape
        return tensor.reshape(*leading, seq_len, n_heads * d_head)

    def forward(self, x):
        seq_len = x.shape[-2]

        queries = self.split_heads(self.to_query(x))     # (..., H, T, d_head)
        keys    = self.split_heads(self.to_key(x))
        values  = self.split_heads(self.to_value(x))

        scores = queries @ keys.transpose(-2, -1)        # (..., H, T, T)
        scores = scores / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.future[:seq_len, :seq_len], float("-inf"))

        weights = scores.softmax(dim=-1)
        joined = self.merge_heads(weights @ values)      # (..., T, d_model)

        return self.to_output(joined), {"joined": joined, "weights": weights}


def copy_explicit_into_fused(explicit, fused):
    """Give the fused version exactly the weights the explicit one has.

    Head h owns columns h*d_head .. (h+1)*d_head of the big matrix. nn.Linear
    stores (out, in), so those columns are ROWS of .weight - which is why the
    slicing below is on dimension 0.
    """
    d_head = fused.d_head
    with torch.no_grad():
        for h, head in enumerate(explicit.heads):
            lo, hi = h * d_head, (h + 1) * d_head
            fused.to_query.weight[lo:hi] = head.to_query.weight
            fused.to_key.weight[lo:hi]   = head.to_key.weight
            fused.to_value.weight[lo:hi] = head.to_value.weight
        fused.to_output.weight.copy_(explicit.to_output.weight)
    return fused


if __name__ == "__main__":
    x_rows, words = example_input()
    x = torch.tensor(x_rows)

    torch.manual_seed(SEED)
    explicit = MultiHeadExplicit()
    fused = copy_explicit_into_fused(explicit, MultiHeadFused())

    explicit_out, explicit_parts = explicit(x)
    fused_out, fused_parts = fused(x)

    def show(row, keep=8):
        return "[" + ", ".join(f"{v:7.3f}" for v in row[:keep]) + "]"

    def grid(rows, labels):
        print("               " + "".join(f"{w:>10}" for w in labels))
        for i, row in enumerate(rows):
            print(f"    {labels[i]:>9} " + "".join(f"{v:9.4f} " for v in row))

    d_head = D_MODEL // N_HEADS

    print("=" * 74)
    print("THE PROBLEM - WHAT ONE HEAD CANNOT DO")
    print("=" * 74)
    print()
    print("  L4 built one attention head and it worked: every position ended up")
    print("  holding a blend of what came before it. So why add more?")
    print()
    print("  Look at how one head builds position 3's answer. These are our own")
    print("  numbers, from head 0:")
    print()
    head_zero = explicit_parts["heads"][0]
    row_weights = head_zero["weights"][3]
    head_values = head_zero["values"]
    print(f"        weights:  " + "  ".join(f"{words[j]}={row_weights[j]:.3f}" for j in range(T)))
    print()
    for d in range(d_head):
        terms = " + ".join(
            f"{row_weights[j]:.3f}*{head_values[j][d]:6.3f}" for j in range(T)
        )
        answer = sum(row_weights[j] * head_values[j][d] for j in range(T))
        print(f"        out[{d}] = {terms} = {answer:7.3f}")
    print()
    print("  The same four weights appear in every single line.")
    print()
    print("  That is the limitation. A head's weights do not depend on which")
    print("  output number is being produced, so EVERY output number attends to")
    print("  exactly the same places. One head has one opinion and applies it to")
    print("  everything it produces - and making the head wider does not help,")
    print("  because a wider head still has one row of weights per position.")
    print()

    print("=" * 74)
    print("WHAT SEVERAL HEADS DO INSTEAD")
    print("=" * 74)
    print()
    other = explicit_parts["heads"][1]["weights"][3]
    print("  Same position, now with two heads:")
    print()
    print(f"        out[0..{d_head - 1}] used  " + "  ".join(f"{row_weights[j]:.3f}" for j in range(T)))
    print(f"        out[{d_head}..{D_MODEL - 1}] used  " + "  ".join(f"{other[j]:.3f}" for j in range(T)))
    print()
    print(f"  The first half leans on {words[int(row_weights.argmax())]!r}. The second half leans on")
    print(f"  {words[int(other.argmax())]!r}. Different parts of the same answer looked at")
    print("  different words.")
    print()
    print("  That is what a single head cannot do, however wide you make it, and")
    print("  it is the whole reason for more than one.")
    print()
    print("  And it costs nothing. Count the weights:")
    print()
    for n in (1, 2, 4, 8):
        sample = MultiHeadExplicit(n_heads=n)
        qkv = sum(p.numel() for h in sample.heads for p in h.parameters())
        projection = sample.to_output.weight.numel()
        marker = "   <- ours" if n == N_HEADS else ""
        print(f"        {n} head(s) of {D_MODEL // n}:  Q/K/V {qkv:4} + W_O {projection} = "
              f"{qkv + projection} weights{marker}")
    print()
    print(f"  Identical every time. One head of {D_MODEL} uses three {D_MODEL}x{D_MODEL} matrices;")
    print(f"  {N_HEADS} heads of {d_head} use six {D_MODEL}x{d_head} matrices. Same total, same")
    print("  arithmetic. So the question is never whether extra heads are worth")
    print("  the cost - it is what to do with a fixed budget.")
    print()
    print("  The catch is that each head gets narrower, and narrow heads are")
    print("  cruder. From the width measurement in attention_plain.py:")
    print()
    print(f"        1 head  of {D_MODEL}   1 view,  each head ~3.6 distinct orders of {T}")
    print(f"        {N_HEADS} heads of {d_head}   {N_HEADS} views, each head ~3.5 - nearly as good")
    print(f"        {D_MODEL} heads of 1   {D_MODEL} views, each head 2.0 - almost useless")
    print()
    print("  Somewhere in the middle wins, which is why GPT-2 uses 12 heads of")
    print("  64 rather than 1 of 768 or 768 of 1.")
    print()

    print("=" * 74)
    print("SO THE HEADS HAVE TO GET NARROWER")
    print("=" * 74)
    print()
    print(f"  A word row is {D_MODEL} numbers. That is what this layer receives, and")
    print("  the answer it hands back has to be usable in the same places the")
    print("  input was - fed to another layer, or combined with the original")
    print(f"  row. So it has to come back out as {D_MODEL} numbers too.")
    print()
    print(f"  Watch what happens if each head keeps the full {D_MODEL}:")
    print()
    wide = [CausalSelfAttention(D_MODEL, D_MODEL, T) for _ in range(N_HEADS)]
    wide_joined = torch.cat([head(x)[0] for head in wide], dim=-1)
    print(f"        {N_HEADS} heads x {D_MODEL} numbers each, stuck side by side")
    print(f"        -> {wide_joined.shape[-1]} numbers per word")
    print()
    print(f"        the layer received {D_MODEL} and handed back {wide_joined.shape[-1]}.")
    print()
    print("  Try to use that answer again - feed it to another such layer:")
    print()
    try:
        CausalSelfAttention(D_MODEL, D_MODEL, T)(wide_joined)
    except RuntimeError as error:
        print(f"        RuntimeError: {str(error).splitlines()[0][:60]}")
    print()
    print(f"  Now narrow each head to {d_head}:")
    print()
    print(f"        {N_HEADS} heads x {d_head} numbers each, stuck side by side")
    print(f"        -> {N_HEADS * d_head} numbers per word, which is what came in")
    print()
    print("  Feed THAT to another layer and it works:")
    print()
    print(f"        n_heads x d_head = d_model")
    print(f"        {N_HEADS} x {d_head} = {N_HEADS * d_head}")
    print()
    print("  These are not two different kinds of head. It is the same class")
    print("  from L4 with a narrower setting - a head's output width simply IS")
    print("  its d_head:")
    print()
    for n in (1, 2, 4, 8):
        width = D_MODEL // n
        marker = "   <- attention_plain.py" if n == 1 else ("   <- here" if n == N_HEADS else "")
        print(f"        n_heads={n}  d_head={width}  each head gives ({T} x {width}),"
              f"  {n} of them = ({T} x {n * width}){marker}")
    print()
    print(f"  attention_plain.py is the n_heads=1 case: with nothing to share the")
    print(f"  width with, its head kept all {D_MODEL}.")
    print()

    print("=" * 74)
    print("THE HEADS DISAGREE, WHICH IS THE POINT")
    print("=" * 74)
    print()
    for h, parts in enumerate(explicit_parts["heads"]):
        print(f"  head {h} - who each position looked at:")
        print()
        grid(parts["weights"].tolist(), words)
        print()
    print("  Same input, same mask, different weights - so different opinions.")
    print("  One head can follow one kind of relationship while another follows")
    print("  something else entirely.")
    print()

    print("=" * 74)
    print("WHY THERE IS AN OUTPUT PROJECTION")
    print("=" * 74)
    print()
    joined = explicit_parts["joined"]
    print(f"  Concatenating gives {tuple(joined.shape)} - back to d_model. But look at")
    print("  what that row actually is, for position 3:")
    print()
    print(f"        {show(joined[3].tolist())}")
    print(f"         <------- head 0 -------> <------- head 1 ------->")
    print()
    print(f"  Numbers 0-{d_head - 1} came ONLY from head 0. Numbers {d_head}-{D_MODEL - 1} ONLY from")
    print("  head 1. The heads sat next to each other and never spoke.")
    print()
    print(f"  W_O is a ({D_MODEL} x {D_MODEL}) matrix, so every output number is a blend of")
    print("  every head:")
    print()
    print(f"        {show(explicit_out[3].tolist())}")
    print()
    print("  Without it, a later layer would see two separate halves rather than")
    print("  one combined answer.")
    print()
    print("  ---")
    print()
    print("  The whole chain, for one sequence with no batch:")
    print()
    print(f"        input                 ({T} x {D_MODEL})")
    for h in range(N_HEADS):
        print(f"        head {h} output         ({T} x {d_head})     <- narrow, not {D_MODEL}")
    print(f"        concatenated          ({T} x {D_MODEL})")
    print(f"        after W_O             ({T} x {D_MODEL})")
    print()
    print(f"  In short: ({T}x{D_MODEL}) -> {N_HEADS} x ({T}x{d_head}) -> ({T}x{D_MODEL}) -> ({T}x{D_MODEL})")
    print()
    print(f"  Each head produces {d_head} numbers, NOT {D_MODEL}. If they each produced {D_MODEL} the")
    print(f"  concatenation would be {N_HEADS * D_MODEL} and the layer would break, as shown above.")
    print()

    print("=" * 74)
    print("STAGE 2 - THE FUSED VERSION")
    print("=" * 74)
    print()
    print("  The explicit version runs each head separately. Real implementations")
    print("  do one big multiply and carve the heads out of the result:")
    print()
    print(f"        to_query(x)                  ({T} x {D_MODEL}) - all heads at once")
    print(f"        reshape into heads           ({T} x {N_HEADS} x {d_head})")
    print(f"        move heads in front of T     ({N_HEADS} x {T} x {d_head})")
    print()
    print("  Head h owns numbers h*d_head onward, which is exactly the slice the")
    print("  explicit version gives its separate heads. Two reshapes replace a")
    print(f"  Python loop over {N_HEADS} modules.")
    print()
    print(f"  parameter count is identical either way:")
    print(f"        explicit  {sum(p.numel() for p in explicit.parameters()):4}")
    print(f"        fused     {sum(p.numel() for p in fused.parameters()):4}")
    print()

    print("=" * 74)
    print("STAGE 3 - DO THE TWO AGREE?")
    print("=" * 74)
    print()
    print("  Same weights in both, so the numbers must match. This one is worth")
    print("  checking carefully: a reshape that splits the wrong dimension, or")
    print("  forgets to move the head axis, does not crash - it quietly mixes")
    print("  the wrong numbers together.")
    print()
    difference = (explicit_out - fused_out).abs().max().item()
    print(f"        joined rows   biggest difference "
          f"{(explicit_parts['joined'] - fused_parts['joined']).abs().max().item():.2e}")
    print(f"        final output  biggest difference {difference:.2e}")
    print(f"        agree: {torch.allclose(explicit_out, fused_out, atol=1e-6)}")
    print()

    print("=" * 74)
    print("SHAPES, AND A BATCH")
    print("=" * 74)
    print()
    batch = torch.stack([x, x, x])
    batch_out, _ = fused(batch)
    print(f"        one sequence   {tuple(x.shape)}      -> {tuple(fused_out.shape)}")
    print(f"        three stacked  {tuple(batch.shape)}   -> {tuple(batch_out.shape)}")
    print()
    print(f"  Still {D_MODEL} numbers per position, whatever the head count. That is")
    print("  the constraint doing its job: the layer hands back exactly the")
    print("  shape it was given, so it can be stacked or added to.")
    print()
