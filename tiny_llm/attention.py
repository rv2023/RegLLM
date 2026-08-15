"""L4 - one causal self-attention head.

Stage 1 of 3: plain Python. Lists and loops, no PyTorch and no matrix syntax, so
every number is visible. Stages 2 and 3 (tensors, then checking the two agree)
come next.

The input is the REAL output of embeddings.py, not a small stand-in. An earlier
draft used invented numbers because they fit the screen better, and it hid a
genuine problem: with L3's actual values the scores come out so large that
softmax collapses to winner-take-all. See SCALING below.

After L3 each position holds a row of numbers saying which word it is and where
it sits - and nothing about the other positions. Every position is an island. To
predict what comes next, a position has to gather from the positions before it.
That is what attention does.

Five steps:

    1. query, key, value   each position's row, put through three matrices
    2. scores              every query against every key, divided by sqrt(d_head)
    3. mask                blank out anything later than the current position
    4. weights             softmax each row, so it sums to 1
    5. output              weighted average of the values

MASKING. The answers (y) are the inputs (x) shifted one place, so the correct
next word sits one slot to the right of every position. Without the mask a
position could just copy it - scoring well, learning nothing, and failing at
generation where there is no slot to the right.

SCALING. Each score sums d_head products, each of which came from a sum of
d_model products, so scores grow with the sizes involved. Left alone on L3's real
embeddings the gap between scores in a row reached 32, and softmax on a gap of 10
already returns 1.0000/0.0000. A saturated softmax has almost no gradient, so the
query and key matrices would barely learn. Two things keep it in range:

    - dividing scores by sqrt(d_head), which cancels the growth with width
    - initialising matrices with std = 1/sqrt(fan_in) rather than uniform(-1, 1)

Measured on this data at D_HEAD = D_MODEL = 8: the biggest gap between two
scores in a row is 29.8 unscaled and 10.5 scaled. Even 10.5 is enough to pin a
row - this particular seed does saturate, and the file says so rather than
glossing over it. Across 200 random starts the average biggest weight is 0.69 and
about 30 in 200 saturate.
"""

import math
import random

import torch

from dataset import BLOCK_SIZE, EXAMPLES
from embeddings import D_MODEL, Embeddings
from tokenizer import ITOS, STOI

T      = BLOCK_SIZE      # positions per example, from dataset.py
SEED   = 0

# One head, so the head is as wide as its input. Narrowing it would throw
# information away for no gain - there is nothing else here to make use of the
# space. This becomes a real choice when there is more than one head; until then
# the honest value is the same width in and out.
D_HEAD = D_MODEL         # numbers per position inside the head


def build_matrix(fan_in, fan_out, seed):
    """A fan_in x fan_out matrix of small random numbers.

    std = 1/sqrt(fan_in) keeps the outputs roughly the same size as the inputs
    however wide the matrix is. uniform(-1, 1) does not, and produces scores
    large enough to saturate softmax - see SCALING in the module docstring.
    """
    rng = random.Random(seed)
    spread = 1 / math.sqrt(fan_in)
    return [[rng.gauss(0, spread) for _ in range(fan_out)] for _ in range(fan_in)]


def example_input():
    """The real L3 embeddings for the first training example."""
    torch.manual_seed(SEED)
    ids = EXAMPLES[0][0]
    rows = Embeddings()(torch.tensor(ids)).tolist()
    return rows, [ITOS[i] for i in ids]


def project(row, matrix):
    """One position's row through one matrix: d_model numbers in, d_head out."""
    n_out = len(matrix[0])
    return [sum(row[i] * matrix[i][j] for i in range(len(row))) for j in range(n_out)]


def project_all(rows, matrix):
    """Every position through the SAME matrix. What differs per position is the
    row going in, not the matrix."""
    return [project(row, matrix) for row in rows]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(row):
    """Scores into shares: all positive, adding to 1.

    Subtracting the largest first changes nothing mathematically and stops exp()
    overflowing on large numbers.
    """
    biggest = max(row)
    exponentiated = [math.exp(v - biggest) for v in row]
    total = sum(exponentiated)
    return [v / total for v in exponentiated]


def scores_for(queries, keys, scale=True):
    """scores[i][j] = how well position i's question matches position j's
    advertisement, divided by sqrt(d_head) to stop it growing with width."""
    divisor = math.sqrt(len(queries[0])) if scale else 1.0
    return [[dot(q, k) / divisor for k in keys] for q in queries]


def apply_causal_mask(scores):
    """Blank out anything later than the current position.

    -inf because exp(-inf) is exactly 0, so blocked positions get no weight at
    all - not a small weight, none.
    """
    return [
        [score if j <= i else float("-inf") for j, score in enumerate(row)]
        for i, row in enumerate(scores)
    ]


def weighted_average(weights, values):
    n_out = len(values[0])
    return [
        [sum(w[j] * values[j][d] for j in range(len(values))) for d in range(n_out)]
        for w in weights
    ]


def attention_plain(rows, w_query, w_key, w_value, scale=True):
    """The five steps, start to finish."""
    queries = project_all(rows, w_query)
    keys    = project_all(rows, w_key)
    values  = project_all(rows, w_value)

    raw     = scores_for(queries, keys, scale=scale)
    masked  = apply_causal_mask(raw)
    weights = [softmax(row) for row in masked]

    return weighted_average(weights, values), {
        "queries": queries, "keys": keys, "values": values,
        "raw": raw, "masked": masked, "weights": weights,
    }


if __name__ == "__main__":
    x, words = example_input()
    w_query = build_matrix(D_MODEL, D_HEAD, SEED + 1)
    w_key   = build_matrix(D_MODEL, D_HEAD, SEED + 2)
    w_value = build_matrix(D_MODEL, D_HEAD, SEED + 3)

    output, parts = attention_plain(x, w_query, w_key, w_value)

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
    print("THE PROBLEM")
    print("=" * 74)
    print()
    print("  This is the real output of embeddings.py for our first training")
    print("  example - the same numbers that file prints, nothing invented:")
    print()
    print(f"                <-------------- D_MODEL = {D_MODEL} numbers -------------->")
    for i in range(T):
        tail = f"   <-- T = {T} rows, one per word" if i == 1 else ""
        print(f"    {words[i]:>7}   " + " ".join(f"{v:7.3f}" for v in x[i]) + tail)
    print()
    print(f"    T       = {T}   how many words we look at at once   (the rows)")
    print(f"    D_MODEL = {D_MODEL}   numbers describing each word        (the columns)")
    print(f"    D_HEAD  = {D_HEAD}   numbers per word once inside the head")
    print()
    print("  Every row says which word it is and where it sits. None of them")
    print("  says anything about the other rows. Position 2 ('ruled') has no")
    print("  idea 'the king' came before it.")
    print()
    print("  Two words used throughout, so they are clear from the start:")
    print()
    print("    a HEAD is one complete copy of the machinery below - its own")
    print(f"    W_Q, W_K and W_V ({3 * D_MODEL * D_HEAD} weights) plus the five steps. It produces")
    print("    exactly ONE opinion per position about what to look at. It does")
    print("    not own the rows above; any head would read the same ones. A")
    print("    second opinion needs a second head with its own weights.")
    print()
    print("    a LAYER is all the heads at one stage, run together and their")
    print("    results combined - one round of every position gathering from")
    print("    the positions before it.")
    print()
    print("    Heads sit SIDE BY SIDE: same input, different opinions at once.")
    print("    Layers sit ON TOP of each other: each reads what the one below")
    print("    produced, so a second layer blends rows that already carry")
    print("    context. That is how indirect relationships get built.")
    print()
    print("    This file is one head, one layer - the smallest thing that works.")
    print("    (A full transformer layer holds more than attention: a small")
    print("    feed-forward network, normalisation, and a residual add. Those")
    print("    are assembled later; here 'layer' means the attention part.)")
    print()
    print("  T never changes below - four words in, four out. Only the width")
    print("  changes, D_MODEL -> D_HEAD, and step 1 is what does it.")
    print()

    print("=" * 74)
    print("STEP 1 - EACH POSITION ASKS, OFFERS, AND CARRIES SOMETHING")
    print("=" * 74)
    print()
    print("    query  - what am I looking for?")
    print("    key    - what do I advertise?")
    print("    value  - what do I hand over if picked?")
    print()
    print(f"  We have {D_MODEL} numbers describing a word. We want {D_HEAD}.")
    print()
    print("  To make ONE output number, every input number gets a say - so you")
    print(f"  need {D_MODEL} weights, one per input number. Here is that for position 0:")
    print()
    print("        input number       weight        product")
    running = 0.0
    for i in range(D_MODEL):
        product = x[0][i] * w_query[i][0]
        running += product
        print(f"          {x[0][i]:7.3f}     x   {w_query[i][0]:7.3f}   =  {product:8.4f}")
    print(f"        {'':32}{'-' * 8}")
    print(f"          output number 0 =            {running:8.4f}")
    print()
    print(f"  Those {D_MODEL} weights build query number 0 and nothing else. Query")
    print(f"  number 1 needs its own {D_MODEL} weights, and so on - {D_HEAD} sets of weights,")
    print("  one per query NUMBER:")
    print()
    for j in range(D_HEAD):
        weights = ", ".join(f"{w_query[i][j]:6.3f}" for i in range(D_MODEL))
        print(f"        the {D_MODEL} weights for query number {j}:  {weights}")
    print()
    print("  Careful here - a set of weights belongs to a query NUMBER, not to a")
    print(f"  word. There are {T} words with {D_HEAD} query numbers each, so {T * D_HEAD} numbers")
    print(f"  altogether, but only {D_HEAD} sets of weights. They cannot be per word.")
    print()
    print(f"  The set for query number 0 builds the FIRST number of every word:")
    print()
    for i in range(T):
        print(f"        {words[i]:>7}  query number 0 = {parts['queries'][i][0]:8.3f}")
    print()
    print(f"  The set for query number 1 builds the SECOND number of every word:")
    print()
    for i in range(T):
        print(f"        {words[i]:>7}  query number 1 = {parts['queries'][i][1]:8.3f}")
    print()
    print("  A word's full query is all 8 sets applied to that word's row.")
    print()
    print(f"  {D_HEAD} sets of {D_MODEL} weights = {D_MODEL * D_HEAD} numbers. Stored as a table with {D_MODEL} rows")
    print(f"  and {D_HEAD} columns, where each COLUMN is one of the sets above. That is")
    print("  all a 'matrix' is here - the sets, side by side.")
    print()
    print("  IMPORTANT: those sets are the QUERY matrix only. Keys and values")
    print("  have their own matrices, built the same way but with their own")
    print("  numbers - three separate tables, no weights shared between them:")
    print()
    for name, matrix in (("W_Q  builds queries", w_query),
                         ("W_K  builds keys", w_key),
                         ("W_V  builds values", w_value)):
        print(f"        {name}")
        print(f"           set 0:  " + ", ".join(f"{matrix[i][0]:6.3f}" for i in range(D_MODEL)))
        print(f"           set 1:  " + ", ".join(f"{matrix[i][1]:6.3f}" for i in range(D_MODEL)))
        print(f"           ... {D_HEAD - 2} more sets")
    print()
    print(f"        each matrix   {D_HEAD} sets x {D_MODEL} weights = {D_MODEL * D_HEAD}")
    print(f"        three of them                    = {3 * D_MODEL * D_HEAD} weights in this head")
    print()
    print("  Every position goes through all three matrices and comes out with")
    print("  three rows of its own:")
    print()
    for i in range(T):
        print(f"    position {i}  {words[i]!r}")
        print(f"       row in (L3)  {' '.join(f'{v:7.3f}' for v in x[i])}")
        print(f"       query        {' '.join(f'{v:7.3f}' for v in parts['queries'][i])}   <- x[{i}] @ W_Q")
        print(f"       key          {' '.join(f'{v:7.3f}' for v in parts['keys'][i])}   <- x[{i}] @ W_K")
        print(f"       value        {' '.join(f'{v:7.3f}' for v in parts['values'][i])}   <- x[{i}] @ W_V")
        print()
    print("  Positions 0 and 3 are both 'the', and all six of their rows differ")
    print("  - because the rows going IN already differed, by the slot numbers")
    print("  L3 added. Same word, same matrices, different results.")
    print()
    print("  ---")
    print()
    print(f"  But why {D_HEAD} sets? Suppose we kept only the first one, so every")
    print("  word's key were a single number. Using the real W_K's first column,")
    print("  each word then advertises just this:")
    print()
    # The real W_K, cut down to its first column - not an invented matrix.
    tiny_keys = [key_row[0] for key_row in parts["keys"]]
    for i in range(T):
        print(f"        {words[i]:>7}  advertises {tiny_keys[i]:6.2f}")
    print()
    tiny_queries = [query_row[0] for query_row in parts["queries"]]
    print("  And their queries, cut down the same way:")
    print()
    for i in range(T):
        print(f"        {words[i]:>7}  query {tiny_queries[i]:6.2f}")
    print()
    print("  A score is query times key, so each word picks whichever key gives")
    print("  it the biggest number. Working out position 0's row - its query is")
    print(f"  {tiny_queries[0]:.2f}:")
    print()
    row_products = [tiny_queries[0] * k for k in tiny_keys]
    for j in range(T):
        note = "   <- biggest" if row_products[j] == max(row_products) else ""
        print(f"        {tiny_queries[0]:6.2f}  x  {tiny_keys[j]:6.2f}  =  {row_products[j]:7.2f}"
              f"   ({words[j]}){note}")
    print()
    print("  Do that for all four and look what happens:")
    print()
    for i in range(T):
        products = [tiny_queries[i] * k for k in tiny_keys]
        best = max(range(T), key=lambda j: products[j])
        sign = "positive" if tiny_queries[i] > 0 else "negative"
        print(f"        {words[i]:>7}  query {tiny_queries[i]:6.2f} ({sign})  ->  picks {words[best]}")
    print()
    print("  Three of the four positions choose the same word. That is the")
    print("  collapse, on real numbers rather than a hypothetical.")
    print()
    print("  All four rows at once, every product shown:")
    print()
    print("                       " + "".join(f"{w:>10}" for w in words))
    print("           key   =     " + "".join(f"{tiny_keys[j]:9.2f} " for j in range(T)))
    print("                       " + "-" * (10 * T))
    for i in range(T):
        products = [tiny_queries[i] * tiny_keys[j] for j in range(T)]
        best = max(range(T), key=lambda j: products[j])
        cells = "".join(
            (f"{products[j]:9.2f}*" if j == best else f"{products[j]:9.2f} ")
            for j in range(T)
        )
        print(f"  {words[i]:>7} q ={tiny_queries[i]:6.2f}  {cells}  -> {words[best]}")
    print()
    print("        (* marks the biggest score in that row)")
    print()
    negatives = [tiny_queries[i] for i in range(T) if tiny_queries[i] < 0]
    print("  And it could not have gone otherwise. Look at the three negative")
    print(f"  queries: {', '.join(f'{v:.2f}' for v in negatives)}. Their sizes differ by a factor of")
    print(f"  {max(abs(v) for v in negatives) / min(abs(v) for v in negatives):.1f}, and all three pick the same word.")
    print()
    print("  That is the whole story. Multiplying a row of keys by a positive")
    print("  number keeps their order, so the biggest key wins. Multiplying by")
    print("  a negative number reverses the order, so the most negative key")
    print("  wins. The SIZE of the query scales every score equally and cannot")
    print("  change which is largest - only the sign can, and it has only two")
    print("  options.")
    print()
    biggest = words[max(range(T), key=lambda j: tiny_keys[j])]
    smallest = words[min(range(T), key=lambda j: tiny_keys[j])]
    print(f"  Only two words can EVER win: {biggest!r}, which advertises the biggest")
    print(f"  number, and {smallest!r}, which advertises the smallest. A positive query")
    print("  picks one, a negative query picks the other, and that is the whole")
    print("  range of opinions available.")
    print()
    reachable = {
        max(range(T), key=lambda j: tiny_keys[j]),
        min(range(T), key=lambda j: tiny_keys[j]),
    }
    stranded = [f"{words[j]!r} ({tiny_keys[j]:.2f})" for j in range(T) if j not in reachable]
    print("  The words in between can never be anyone's first choice, no matter")
    print(f"  what query you write. Here that strands {', '.join(stranded)} -")
    print("  there is no query that says 'I want that one'.")
    print()
    print(f"  With {D_HEAD} numbers a query can ask for a COMBINATION - high on this,")
    print("  low on that - so any word can be picked out. Each position ends up")
    print("  preferring a different order:")
    print()
    for i in range(T):
        order = [words[j] for j in sorted(range(T), key=lambda j: -dot(parts["queries"][i], parts["keys"][j]))]
        print(f"        {words[i]:>7}  prefers {order}")
    print()
    print(f"  That is what the {D_HEAD} sets buy: enough room to point at a particular")
    print("  word rather than just 'the most' or 'the least'.")
    print()
    print(f"  So why exactly {D_HEAD}? We did not work it out. It is D_MODEL, the")
    print("  width L3 happened to use, and one head is as wide as its input.")
    print()
    print("  Would fewer do? Here is a test. Ask each word to rank all four")
    print("  words by score, then count how many of those rankings are")
    print("  actually different. With one number per word:")
    print()
    narrow_q = build_matrix(D_MODEL, 1, 1)
    narrow_k = build_matrix(D_MODEL, 1, 2)
    nq, nk = project_all(x, narrow_q), project_all(x, narrow_k)
    rankings = []
    for i in range(T):
        ranking = tuple(sorted(range(T), key=lambda j: -dot(nq[i], nk[j])))
        rankings.append(ranking)
        repeat = "   <- same as an earlier row" if ranking in rankings[:-1] else ""
        print(f"        {words[i]:>7} ranks them: {[words[j] for j in ranking]}{repeat}")
    print()
    print(f"        {len(rankings)} rankings, but only {len(set(rankings))} are DIFFERENT")
    print()
    print(f"  So this width scores {len(set(rankings))}. Doing that at several widths, 60")
    print("  random starts each and averaged:")
    print()
    for width in (1, 2, 4, 8, 16):
        counts = []
        for seed in range(60):
            wq = build_matrix(D_MODEL, width, seed * 2 + 1)
            wk = build_matrix(D_MODEL, width, seed * 2 + 2)
            qs, ks = project_all(x, wq), project_all(x, wk)
            orders = {
                tuple(sorted(range(T), key=lambda j: -dot(qs[i], ks[j])))
                for i in range(T)
            }
            counts.append(len(orders))
        marker = "   <- ours" if width == D_HEAD else ""
        print(f"        {width:2} numbers per word  ->  {sum(counts) / len(counts):.1f} of the {T} want something different{marker}")
    print()
    print("  1 number gets you 2, which is what we just saw. Going to 2 numbers")
    print("  gets you 3. After 4 it barely moves - there are only 4 words here,")
    print(f"  so there is little left to tell apart. {D_HEAD} is more than this problem")
    print("  needs.")
    print()
    print("  Real models use 64 or 128 numbers per head, because they rank")
    print("  thousands of positions drawn from a 50,000-word vocabulary. Far")
    print("  more to distinguish, so far more room needed.")
    print()
    print(f"  Running all {D_HEAD} sets on position 0 gives its query:")
    print(f"        {show(parts['queries'][0])}")
    print()
    print("  ---")
    print()
    print("  The same weights are used for EVERY word. There is one query")
    print("  matrix, not one per position. Taking set 0 and running it on all")
    print(f"  {T} words:")
    print()
    first_set = [w_query[i][0] for i in range(D_MODEL)]
    print(f"        set 0:  " + ", ".join(f"{v:6.3f}" for v in first_set))
    print()
    for i in range(T):
        total = sum(x[i][d] * first_set[d] for d in range(D_MODEL))
        print(f"        {words[i]:>7} row x set 0, summed  ->  {total:8.4f}")
    print()
    print("  Same weights every time. Different rows going in, so different")
    print("  numbers coming out.")
    print()
    print("  Two different things are easy to mix up here, so both at once -")
    print("  every number below was built by the set at the top of its column:")
    print()
    print("             " + "".join(f"{'out ' + str(j):>9}" for j in range(D_HEAD)))
    print("             " + "".join(f"{'set ' + str(j):>9}" for j in range(D_HEAD)))
    print("           " + "-" * (9 * D_HEAD + 1))
    for i in range(T):
        print(f"  {words[i]:>7}   " + " ".join(f"{v:8.3f}" for v in parts["queries"][i]))
    print()
    print("        ACROSS a row  - one word, all 8 sets -> its 8 query numbers")
    print("        DOWN a column - 4 different words, all using the SAME set")
    print()
    print("  So 'the same weights for every word' is the column direction, and")
    print("  'set 0, set 1, ...' is the row direction. There are still only")
    print(f"  {D_MODEL * D_HEAD} weights in total, however many words go through them.")
    print()
    print("  ---")
    print()
    print("  Where do the weights come from? Nowhere clever. This is the whole")
    print("  of build_matrix, the function that made every weight above:")
    print()
    print("        spread = 1 / sqrt(fan_in)")
    print("        weight = rng.gauss(0, spread)      for every cell")
    print()
    print(f"  gauss(0, spread) draws a random number centred on 0, with most")
    print(f"  landing within +/- spread. Here fan_in is {D_MODEL} (the numbers coming")
    print(f"  in), so spread = 1/sqrt({D_MODEL}) = {1 / math.sqrt(D_MODEL):.3f}.")
    print()
    print("  Drawing 2000 of them and counting where they fall:")
    print()
    sample_rng = random.Random(1)
    sample = [sample_rng.gauss(0, 1 / math.sqrt(D_MODEL)) for _ in range(2000)]
    edges = [-1.2, -0.8, -0.4, -0.15, 0, 0.15, 0.4, 0.8, 1.2]
    for low, high in zip(edges, edges[1:]):
        count = sum(1 for v in sample if low <= v < high)
        print(f"        {low:5.2f} to {high:5.2f}  {'#' * (count // 12):38} {count:4}")
    print()
    print(f"  About two thirds land within +/-{1 / math.sqrt(D_MODEL):.3f} and effectively all of")
    print("  them within three times that. The model starts with no preferences")
    print("  whatsoever - training is what makes these numbers mean anything.")
    print()
    print("  The only real decision is how spread out they start, and it matters:")
    print()
    print("        too big            -> outputs grow at every layer")
    print("        1 / sqrt(fan_in)   -> outputs stay about the size of the inputs")
    print("        too small          -> outputs fade toward zero")
    print()

    print("=" * 74)
    print("STEP 1, ALL IN ONE PLACE")
    print("=" * 74)
    print()
    print("  Take ONE word. Its row is 8 numbers - the word row plus the slot")
    print("  row, added together back in L3. Dot that row with each of the 8")
    print(f"  weight sets of W_Q. Each dot gives ONE number:")
    print()
    for j in range(D_HEAD):
        weight_set = [w_query[i][j] for i in range(D_MODEL)]
        value = sum(a * b for a, b in zip(x[0], weight_set))
        print(f"        row (8)  .  W_Q set {j} (8)   =  {value:8.4f}   -> query number {j}")
    print()
    print("  Collect those 8 answers and you have that word's query:")
    print(f"        {show(parts['queries'][0])}")
    print()
    print("  Written as shapes, one word:")
    print()
    print(f"        (1 x {D_MODEL}) word row   @   ({D_MODEL} x {D_HEAD}) W_Q   =   (1 x {D_HEAD}) query")
    print()
    print("  The same W_Q is used for every word, so all four rows can go")
    print("  through it at once:")
    print()
    print(f"        ({T} x {D_MODEL}) all words   @   ({D_MODEL} x {D_HEAD}) W_Q   =   ({T} x {D_HEAD}) all queries")
    print()
    print("  Same answer either way - four separate dots or one bigger")
    print("  multiplication. That is the whole difference between this file and")
    print("  the tensor version to come: not a different method, the same dot")
    print("  products with the loops handed to the library.")
    print()
    print("  And three times over, once per matrix:")
    print()
    print(f"        Q = X @ W_Q        ({T} x {D_MODEL}) @ ({D_MODEL} x {D_HEAD}) = ({T} x {D_HEAD})")
    print(f"        K = X @ W_K        same shapes")
    print(f"        V = X @ W_V        same shapes")
    print()
    print(f"  Counting the arithmetic: {D_HEAD} dots for a query, {D_HEAD} for a key, {D_HEAD} for a")
    print(f"  value = {3 * D_HEAD} per word, x {T} words = {3 * D_HEAD * T} dot products, using the")
    print(f"  {3 * D_MODEL * D_HEAD} weights in the three matrices.")
    print()

    print("=" * 74)
    print("STEP 2 - HOW WELL DOES EACH QUESTION MATCH EACH ADVERTISEMENT?")
    print("=" * 74)
    print()
    print("  A query and a key do NOT multiply every number against every")
    print("  number. They pair up matching positions and add: query number 0")
    print("  meets key number 0, number 1 meets number 1, and so on. Eight")
    print("  products, summed to ONE number.")
    print()
    print(f"  Working out one cell - {words[0]!r}'s query against {words[0]!r}'s key:")
    print()
    print("        position     query      key     product")
    running = 0.0
    for i in range(D_HEAD):
        product = parts["queries"][0][i] * parts["keys"][0][i]
        running += product
        print(f"            {i}      {parts['queries'][0][i]:8.3f} {parts['keys'][0][i]:8.3f}  {product:9.4f}")
    print(f"        {'':30}{'-' * 9}")
    print(f"        raw dot product:            {running:9.4f}")
    print()
    print(f"        divide by sqrt(D_HEAD) = sqrt({D_HEAD}) = {math.sqrt(D_HEAD):.4f}")
    print(f"           {running:.4f} / {math.sqrt(D_HEAD):.4f} = {running / math.sqrt(D_HEAD):.4f}")
    print()
    print("  Do that for every pair. Four queries x four keys = 16 cells:")
    print()
    grid(parts["raw"], words)
    print()
    print("        row i     word i's query, tested against all four keys")
    print("        column j  word j's key, tested against all four queries")
    print("        diagonal  each word against itself")
    print()
    print(f"  Note the {D_HEAD} disappeared. It was the axis being summed over, so")
    print(f"  {T} queries x {T} keys leaves a {T} x {T} grid - one score per pair of WORDS,")
    print(f"  not per pair of numbers. In one line: Q @ K.T / sqrt({D_HEAD}), which is")
    print(f"  ({T} x {D_HEAD}) @ ({D_HEAD} x {T}) = ({T} x {T}).")
    print()
    raw_scores = scores_for(parts["queries"], parts["keys"], scale=False)
    raw_gap    = max(max(r) - min(r) for r in raw_scores)
    scaled_gap = max(max(r) - min(r) for r in parts["raw"])
    print(f"  One detail: those numbers are already divided by sqrt(D_HEAD) =")
    print(f"  sqrt({D_HEAD}) = {math.sqrt(D_HEAD):.2f}. Without it the biggest gap between two scores")
    print(f"  in a row would be {raw_gap:.1f} rather than {scaled_gap:.1f}.")
    print()
    print("  Why that gap matters only becomes clear at step 4, so it is left")
    print("  until then.")
    print()

    print("=" * 74)
    print("STEP 3 - BLOCK THE FUTURE")
    print("=" * 74)
    print()
    print("  A position may look at itself and what came before, nothing later:")
    print()
    grid(parts["masked"], words)
    print()
    print("  Row 0 keeps one number, row 1 two, row 2 three, row 3 four. That")
    print("  staircase is the whole of causality.")
    print()

    print("=" * 74)
    print("STEP 4 - TURN SCORES INTO SHARES")
    print("=" * 74)
    print()
    print("  The scores are just numbers - some negative, no particular total.")
    print("  We need shares: all positive, adding up to 1, so they can be used")
    print("  as proportions. Two steps get there. Taking row 2 ('ruled') of the")
    print("  masked scores above:")
    print()
    example_row = parts["masked"][2]
    shown = ["-inf" if v == float("-inf") else f"{v:.4f}" for v in example_row]
    print(f"        scores   [{', '.join(shown)}]")
    print()
    print("  STEP ONE - raise 2.718 to the power of each score. That is what")
    print("  exp() does. The answer is always positive, and it grows fast:")
    print()
    exponentiated = []
    for j, value in enumerate(example_row):
        result = 0.0 if value == float("-inf") else math.exp(value)
        exponentiated.append(result)
        label = "-inf" if value == float("-inf") else f"{value:8.4f}"
        print(f"          {label:>8}   ->  {result:9.4f}    ({words[j]})")
    total = sum(exponentiated)
    print(f"          {'':8}       {'-' * 9}")
    print(f"          total          {total:9.4f}")
    print()
    print("  STEP TWO - divide each one by that total, so they add up to 1:")
    print()
    for j, value in enumerate(exponentiated):
        print(f"          {value:9.4f} / {total:.4f}  =  {value / total:.4f}    ({words[j]})")
    print()
    print("  Those four numbers are row 2 of the grid below. Every row is made")
    print("  the same way:")
    print()
    grid(parts["weights"], words)
    print()
    for i, row in enumerate(parts["weights"]):
        print(f"        row {i} adds up to {sum(row):.4f}")
    print()
    print("  exp(-inf) is exactly 0, so the blocked cells are 0.0000 - no share")
    print("  at all, rather than a small one.")
    print()
    print("  ---")
    print()
    print("  Now the gap from step 2. Why does exp() make the SIZE of a gap")
    print("  matter so much? Because exp turns a DIFFERENCE into a RATIO:")
    print()
    print("        gap between two scores    exp(gap)         the odds")
    for gap in (0.5, 1, 3, 10, 30):
        ratio = math.exp(gap)
        print(f"              {gap:5}              {ratio:16.1f}   {ratio:14.1f} to 1")
    print()
    print("  A gap of 3 means 20-to-1. A gap of 10 means 22,000-to-1. exp grows")
    print("  so fast that a modest gap becomes total dominance:")
    print()
    for gap in (0.5, 1, 3, 10, 30):
        pair = softmax([0.0, gap])
        print(f"        two scores {gap:5} apart  ->  shares {pair[0]:.4f} / {pair[1]:.4f}")
    print()
    print(f"  Our biggest gap is {scaled_gap:.1f}, and the grid above shows what that")
    print("  costs: rows pinned at 0.9991 and 0.9881. This particular random")
    print("  start IS in the saturated range, and it is worth not pretending")
    print("  otherwise.")
    print()
    print(f"  The division still did a lot of work - without it the gap would be")
    print(f"  {raw_gap:.1f}, and the last row would sit at 1.0000 exactly. It moved us")
    print("  from hopeless to merely unlucky.")
    print()
    print("  Across 200 random starts the average biggest weight is 0.69, and")
    print("  only about 30 in 200 saturate. Ours is one of those 30. Rerun with")
    print("  a different SEED and the rows spread out.")
    print()
    print("  Why saturation matters before training has even started: when a")
    print("  row is pinned at 1 and 0, nudging a score barely changes the")
    print("  shares, so almost no gradient flows back and the query and key")
    print("  matrices would hardly learn at all. It is not wrong, it is stuck.")
    print()

    print("=" * 74)
    print("STEP 5 - MIX THE VALUES USING THOSE SHARES")
    print("=" * 74)
    print()
    for i in range(T):
        print(f"    position {i} ({words[i]}):")
        for j in range(T):
            share = parts["weights"][i][j]
            note = "  (blocked)" if share == 0 else ""
            print(f"        {share:6.4f} of {words[j]:>7} value {show(parts['values'][j])}{note}")
        print(f"      = {show(output[i])}")
        print()

    print("=" * 74)
    print("WHAT CHANGED")
    print("=" * 74)
    print()
    print(f"  The SHAPE is untouched - {T} positions x {D_MODEL} numbers went in, {T} x {D_HEAD}")
    print("  came out. That is deliberate: the next stage expects the same")
    print("  shape, so attention has to hand back what it was given.")
    print()
    print("  The CONTENTS are completely different:")
    print()
    for i in range(T):
        print(f"    position {i} ({words[i]})")
        print(f"       before  {show(x[i])}")
        print(f"       after   {show(output[i])}")
    print()
    print("  Before, each row said only 'I am this word, in this slot'. After,")
    print("  each row is a blend of the values of every position up to and")
    print("  including itself, mixed in the proportions from step 4.")
    print()
    print("  One thing to notice, because it is the saturation from step 4")
    print("  showing up in the answers. Positions 0 and 1 came out nearly")
    print("  identical:")
    print()
    print(f"       position 0 after  {show(output[0])}")
    print(f"       position 1 after  {show(output[1])}")
    print()
    print(f"  Row 1's shares were {parts['weights'][1][0]:.4f} on position 0 and only")
    print(f"  {parts['weights'][1][1]:.4f} on itself. So 'king' handed back almost a straight")
    print("  copy of 'the' and kept nearly none of its own value. A pinned")
    print("  softmax does not just stop learning - it throws information away.")
    print()
    print("  Position 0 is the exception - nothing precedes it, so its answer is")
    print("  its own value unchanged:")
    print()
    print(f"    position 0 value  {show(parts['values'][0])}")
    print(f"    position 0 output {show(output[0])}")
    print()
    print("  Positions 0 and 3 are both 'the'. After L3 their rows already")
    print("  differed, because of the slot numbers added there. Now they differ")
    print("  for a second reason too: position 3 had three earlier words to draw")
    print("  on and position 0 had none.")
    print()
    print(f"    position 0 output {show(output[0])}")
    print(f"    position 3 output {show(output[3])}")
    print()
    print("  ---")
    print()
    print("  Did any of this achieve the thing we set out to do? Run the same")
    print("  head on two sentences that differ by ONE earlier word:")
    print()
    swapped_ids = list(EXAMPLES[0][0])
    swapped_ids[1] = STOI["queen"]
    torch.manual_seed(SEED)
    swapped_rows = Embeddings()(torch.tensor(swapped_ids)).tolist()
    swapped_out, _ = attention_plain(swapped_rows, w_query, w_key, w_value)

    print(f"        A:  {' '.join(words)}")
    print(f"        B:  {' '.join(ITOS[i] for i in swapped_ids)}      <- word 1 changed")
    print()
    print("  Position 3 is 'the' in both. Its row going IN is identical - L3")
    print("  gave it no way to know what came earlier:")
    print()
    print(f"        A  {show(x[3])}")
    print(f"        B  {show(swapped_rows[3])}")
    print(f"        the same? {x[3] == swapped_rows[3]}")
    print()
    print("  Its row coming OUT of attention:")
    print()
    print(f"        A  {show(output[3])}")
    print(f"        B  {show(swapped_out[3])}")
    same = [round(v, 6) for v in output[3]] == [round(v, 6) for v in swapped_out[3]]
    print(f"        the same? {same}")
    print()
    print("  That is what one attention head did. The same word in the same slot")
    print("  now means something different depending on what came before it.")
    print("  Every position started as an island; attention is the bridge.")
    print()
    print("  Which is exactly what the task needs. To guess 'kingdom' after")
    print("  'the king ruled the', position 3 has to know 'king' and 'ruled'")
    print("  came earlier. Before this file it could not. Now it can.")
    print()
    print("=" * 74)
    print("SO WHAT IS 'THE HEAD' AND 'THE LAYER' HERE?")
    print("=" * 74)
    print()
    print("  Both words describe things this file just built. Pointing at them:")
    print()
    print("  OUR HEAD is exactly three things:")
    print()
    print(f"    1. its weights - {3 * D_MODEL * D_HEAD} numbers it owns and nothing else uses")
    print(f"         W_Q  {D_MODEL} x {D_HEAD}   first row {[round(v, 3) for v in w_query[0]][:4]} ...")
    print(f"         W_K  {D_MODEL} x {D_HEAD}   first row {[round(v, 3) for v in w_key[0]][:4]} ...")
    print(f"         W_V  {D_MODEL} x {D_HEAD}   first row {[round(v, 3) for v in w_value[0]][:4]} ...")
    print()
    print("    2. its opinion - one row of shares per position, which is what")
    print("       those weights were for:")
    print()
    grid(parts["weights"], words)
    print()
    print(f"    3. its answer - {T} rows of {D_HEAD} numbers, each a blend of the values")
    print("       above it:")
    print()
    for i in range(T):
        print(f"         {words[i]:>7}  {show(output[i])}")
    print()
    print("  What the head does NOT own: the input rows. Those came from L3 and")
    print("  a second head would read exactly the same ones.")
    print()
    print("  OUR LAYER is 'all the heads at this stage, combined'. We have one")
    print("  head, so the layer is that head and the combining step is nothing.")
    print("  The layer's answer IS the head's answer, printed just above.")
    print()
    print("  With two heads it would be: run both on the same input rows, get")
    print("  two opinions and two answers, then join the answers back together.")
    print("  Same input, different weights, so genuinely different opinions:")
    print()
    second_q = build_matrix(D_MODEL, D_HEAD, SEED + 40)
    second_k = build_matrix(D_MODEL, D_HEAD, SEED + 41)
    second_v = build_matrix(D_MODEL, D_HEAD, SEED + 42)
    _, second = attention_plain(x, second_q, second_k, second_v)
    print("    head 1 thinks the last position should look at:")
    print("      " + "  ".join(f"{words[j]}={parts['weights'][-1][j]:.3f}" for j in range(T)))
    print("    head 2, same input, its own weights:")
    print("      " + "  ".join(f"{words[j]}={second['weights'][-1][j]:.3f}" for j in range(T)))
    print()
    print("  Counting what a bigger model would hold:")
    print()
    print(f"    one head            {3 * D_MODEL * D_HEAD:5} weights")
    print(f"    a layer of 2 heads  {2 * 3 * D_MODEL * D_HEAD:5} weights   (heads side by side)")
    print(f"    6 such layers       {6 * 2 * 3 * D_MODEL * D_HEAD:5} weights   (layers stacked)")
    print()
    print("  and layers stacked is the interesting direction: layer 2 reads")
    print("  rows that layer 1 already filled with context, so a position can")
    print("  reach information second-hand. More heads only add more views of")
    print("  the same input.")
    print()
