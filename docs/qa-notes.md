# Q&A Notes

Questions asked during the build, and the answers, kept so they don't have to be
re-derived. Organised by topic rather than by date. Milestone references point
at [MILESTONES.md](../MILESTONES.md).

Every number here was produced by running the project's own code — none are
illustrative.

Covers R1–R8 complete, plus the R9 optimizer concepts worked through so far.

---

## Part 1 — Gradients

### Why does one loss produce two gradients?

A partial derivative differentiates with respect to one parameter while holding
the other fixed. The same loss gets differentiated twice:

$$\frac{\partial L}{\partial m} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial m} \qquad \frac{\partial L}{\partial c} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial c}$$

The left factor is **identical** in both — it comes from differentiating the
square in the MSE. The two gradients differ only in the right factor:

- `ŷ = mx + c` differentiated with respect to `m` gives **`x`**
- the same expression differentiated with respect to `c` gives **`1`**

That is the entire difference. One gradient weights each residual by its input;
the other doesn't. It is one derivation with two different tails, not two
separate derivations.

### Why are there five values of `dL/dŷ` but only one `dL/dm`?

**The count follows the parameters, not the data.**

- `dL/dŷ` — one per observation. Five predictions, five values.
- `dL/dm` and `dL/dc` — one each. There is only one `m` and one `c`, shared by
  every observation.

Change `ŷ₃` and only observation 3's error moves. Change `m` and **all five
predictions move at once**, so `m` needs a single slope accounting for its
effect everywhere.

### Where does the sum come from?

`m` reaches the loss through five separate routes, one per data point. When a
variable affects the output through multiple paths, the chain rule says to add
up the contribution from every path:

$$\frac{\partial L}{\partial m} = \sum_{i=1}^{n} \frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial m}$$

The `total_m += residual * x` line **is** that multi-path chain rule. Worked
through with real values at `m=0, c=0`:

```
dL/dy` = [-20, -26, -32, -38, -44]
inputs = [  1,   2,   3,   4,   5]

dL/dm:  (-20·1 + -26·2 + -32·3 + -38·4 + -44·5) / 5  =  -540 / 5  =  -108
dL/dc:  (-20   + -26   + -32   + -38   + -44  ) / 5  =  -160 / 5  =   -32
```

Five numbers in, one number out. Twice, with different weights.

### Does this generalise?

Yes: **you get exactly as many gradients as you have parameters**, regardless of
how much data there is. Ten thousand observations still produce one `dL/dm` —
the sum just has ten thousand terms.

- **R5+**, with more features: `m₁, m₂, m₃, c` → four gradients, still `n` values
  of `dL/dŷ`.
- **L7**, the transformer: `dL/dlogits` has shape `[batch, sequence, vocab]` — a
  huge number of per-position derivatives — collapsing into one gradient per
  weight, shaped like the weight itself. Batch size changes the number of
  paths, never the number of gradients.

### The consequence to remember for L9

Because gradients accumulate over paths, PyTorch makes `.grad` something you
**add into**, not overwrite. Hence every training loop containing
`optimizer.zero_grad()`. Forget it and this iteration's gradients pile onto last
iteration's — the same accumulation done deliberately across observations,
happening accidentally across time steps.

It is also the machinery behind **gradient accumulation** (L12): deliberately
*not* zeroing across several small batches to simulate one large one.

### Why compute both gradients in one pass?

**Correctness.** Two separate functions each recompute the residual, so the sign
convention lives in two places and can drift — [it did](#the-sign-inversion).
One residual computed once cannot disagree with itself.

**Performance.** Two functions mean two full traversals per training step.
Irrelevant at five observations; at L9 it is the difference between a run that
finishes and one that doesn't. Backpropagation exists specifically to compute
all gradients in a single backward sweep, reusing shared intermediates.

---

## Part 2 — The finite-difference check

### What is it for?

The gradient formulas were derived by hand. Nothing about writing them down
proves they are right. A finite-difference check measures the same slope a
completely different way, **using no calculus at all**: nudge a parameter up a
little, nudge it down a little, see how much the loss actually moved, divide by
the distance travelled. Rise over run.

It must call the project's real `mse` and `predict`. If it reimplemented them it
could share a bug with the thing it is checking and agree for the wrong reason.

### Why two separate lines?

```python
dm = (loss(m + h, c) - loss(m - h, c)) / (2 * h)
dc = (loss(m, c + h) - loss(m, c - h)) / (2 * h)
```

The first moves only `m`; the second moves only `c`. Holding the other fixed is
what makes each a *partial* derivative.

### What is `h` called?

**Step size** — also *perturbation size*, *finite-difference step*, or
*interval*. In ML code it is very often named **epsilon** (`eps`, `ε`). The `h`
notation comes from the limit definition of a derivative:

$$f'(p) = \lim_{h \to 0} \frac{f(p+h) - f(p-h)}{2h}$$

The check computes exactly this, minus the limit.

**Name collision:** "step size" is also a common synonym for the **learning
rate** `η`. Two unrelated quantities:

| | `h` / epsilon | `η` / learning rate |
| --- | --- | --- |
| Purpose | measure a slope | move along a slope |
| Wanted | as small as precision allows | tuned; neither tiny nor huge |
| Changes parameters? | no — temporary probe, discarded | yes — this is the update |
| Typical value | `1e-5` to `1e-7` | `0.001` to `0.1` |

### Why did `h = 1` pass when `1e-5` was specified?

Because MSE is **exactly quadratic** in `m` and `c`, and a central difference is
exact for any polynomial up to degree two.

Take `f(p) = ap² + bp + d`, true derivative `2ap + b`:

```
f(p+h) = a(p² + 2ph + h²) + b(p+h) + d
f(p−h) = a(p² − 2ph + h²) + b(p−h) + d
```

Subtracting, the `p²` and `d` terms cancel — and critically the `h²` terms are
**identical on both sides**, so they cancel too:

$$\frac{f(p+h) - f(p-h)}{2h} = \frac{4aph + 2bh}{2h} = 2ap + b$$

`h` has vanished entirely. The symmetry does it: stepping equally in both
directions makes the even-powered terms identical on each side.

Expanding `L = (1/n)Σ(mx + c − y)²` in `m` and `c` gives `m²`, `c²`, `mc`, `m`,
`c`, constants — **degree 2, nothing higher**.

### Where it stops being exact

`f(p) = p³`, true derivative `3p²`:

$$\frac{f(p+h) - f(p-h)}{2h} = \frac{6p^2h + 2h^3}{2h} = 3p^2 + h^2$$

The `h³` doesn't cancel — cubing breaks the symmetry — leaving an error of
exactly `h²`. Measured at `p = 2`, true slope `12`:

| `h` | predicted error `h²` | measured |
| --- | --- | --- |
| `1` | `1` | `13.000000` |
| `0.01` | `0.0001` | `12.000100` |
| `1e-5` | `1e-10` | `12.000000` |

### So why keep `1e-5`?

Two failure modes squeeze `h` from both sides:

- **Too large:** the `h²` truncation error. Zero for this MSE, *not* zero for
  cross-entropy at L8 or anything with a nonlinearity.
- **Too small:** floating-point cancellation. `f(p+h)` and `f(p−h)` become
  nearly identical; subtracting destroys the significant digits, and that
  wreckage is divided by a tiny `2h`. `h = 1e-15` fails *harder* than `h = 1`.

### Comparing the two results

```python
abs(analytical - numerical) < tolerance     # never ==
```

They reach the same answer by different routes and differ in the final digits.
Observed differences here were around `1e-10`.

---

## Part 3 — NumPy: arrays, shapes, and reductions

### What holds what?

| Thing | Type | Why |
| --- | --- | --- |
| `x`, `y` | arrays | the data — one number per observation |
| `m`, `c` | plain floats | **one number each**, shared by all observations |
| predictions, residuals | arrays, made by NumPy | one per observation |
| `dm`, `dc`, `loss` | plain floats | one per parameter / one total |

Only the **data** is arrays. Parameters are scalars and stay scalars.

Counting them: before the split, two named arrays (`x`, `y`). After the split,
four (`x_train`, `y_train`, `x_test`, `y_test`). Everything else is either a
scalar or a temporary that NumPy builds and frees inside one expression.

**Arrays scale with the data; scalars don't.** Go from 50 points to 50,000 and
`x`, `y` grow to `(50000,)` while `m`, `c`, `dm`, `dc`, `loss` stay one number
each. Nothing in the training loop changes.

### Broadcasting

```
x         [1. 2. 3. 4. 5.]  shape (5,)
m          3.0              <- a single number, not an array
m*x + c   [10. 13. 16. 19. 22.]  shape (5,)
```

`m * x` needs no loop: NumPy **broadcasts** the scalar across every element.
Two arrays of equal length subtract elementwise, position by position.

### The three stages

```
at m=0,c=0  residual [-10. -13. -16. -19. -22.]      shape (5,)
residual * x         [-10. -26. -48. -76. -110.]     shape (5,)  still 5
2*np.mean(residual*x) = -108.0                       ONE number
2*np.mean(residual)   =  -32.0                       ONE number
```

1. **Elementwise** — `m*x+c`, `preds-y`, `residual*x`. Array in, same-length
   array out. This replaces the loop *body*.
2. **Reduction** — `np.mean`. Array in, one number out. This replaces the
   *accumulator and the divide by n*.
3. Scalars come out the far end.

The array-to-scalar collapse and the many-paths-to-one-gradient collapse are the
same operation. `residual` **is** the `dL/dŷ` values (up to the factor of 2), and
`np.mean` is the summation over paths.

Note `(2/n) * sum` **is** `2 * mean` — the `/n` folds into the reduction.

### The approach to avoid

```python
np.array([m * j + c for j in inputs])       # don't
```

Still one element at a time in Python, just with an array-shaped result. Almost
none of the speed, none of the clarity.

**Test for a proper conversion:** no `for` loop and no `.append` left in
`predict`, `mse`, or `gradients`. The only remaining loop is the training loop,
which iterates over *time steps*, not data points — and that one stays.

That distinction carries into PyTorch at L3: tensor operations handle the data
dimensions, your Python loop handles training steps. At L9 the batch dimension
is handled by the tensor too, never by a loop you write.

### Keeping the list version during conversion

An independent second implementation is a real test, right up until it is
removed. A broken NumPy conversion usually produces a *plausible* number rather
than a crash — a `np.sum` where `np.mean` was needed makes the gradient `n`×
too large, which just looks like a wrong learning rate.

So: write the NumPy versions alongside the old ones, feed both the original five
clean points where the answer is known (`274.0`, `-108.0`, `-32.0`), confirm they
agree **within a tolerance** (NumPy uses pairwise summation, so the last bits can
differ), then delete the list versions.

---

## Part 4 — Noise, the floor, and generalisation

### What is the noise floor?

The lowest loss any model of this shape can reach on this data. Not zero.

The data is `y = 3x + 7 + noise`. The model can only produce a straight line, so
even with **perfect** parameters the noise stays unexplained — nothing in `x`
predicts it.

Proof, on the 40-point training set:

```
                          train MSE   test MSE
at TRUE m=3, c=7            2.6457     1.4650
at LEARNED m=3.07,c=6.27    2.4766     1.6489

mean of noise^2 on train = 2.6457   <- exactly the train MSE at the true parameters
```

At the true parameters, train MSE equals the mean squared noise **exactly**. With
the correct line, every remaining error *is* noise. That is what "irreducible"
means.

R5's `9e-15` was misleading: clean data has no noise term, so its floor was
genuinely zero.

### Why the learned parameters beat the true ones on training data

`2.4766 < 2.6457`. Gradient descent minimises loss on the 40 points it was
shown, and some of what it fit was that sample's particular noise. On the test
set the ordering reverses (`1.6489 > 1.4650`) — that noise doesn't recur.

This is the overfitting signature in miniature. With two parameters the
magnitude is negligible and a 10-point test set can't distinguish it from
chance, but the mechanism is present. It becomes a problem when capacity grows;
a degree-10 polynomial through 50 noisy points would show it dramatically.

### Names for it

- **Irreducible error** — the standard term in the bias–variance decomposition
- **Aleatoric uncertainty** — randomness inherent in the data, as opposed to
  *epistemic* uncertainty, which is the model's ignorance and can be reduced

At **L11**, perplexity will not reach 1 for the same reason: given
`"the king ruled the"` more than one continuation is legitimate, and no model
drives that uncertainty to zero.

### How do you calculate the noise?

```
1. sigma^2 you chose when generating       = 4.0000
2. mean(true_noise^2) in this sample       = 2.6457
3. mean(residual^2)  = train MSE           = 2.4766
4. estimate  SSE/(n-p) = MSE * n/(n-p)     = 2.6069    n=40, p=2
```

**In this project** you know it exactly, because you wrote it: `σ = 2` was
chosen, and `noise = y - (TRUE_M*x + TRUE_C)` recovers each draw. That is a
luxury of synthetic data.

**In real life** there is no `TRUE_M`. You estimate from **residuals**,
`y - predict(x, m, c)`. But the raw mean of squared residuals is **biased low**
— the model bent toward this sample's noise. The correction divides by `n - p`
instead of `n`, where `p` is the number of fitted parameters:

$$\hat{\sigma}^2 = \frac{\sum (y_i - \hat{y}_i)^2}{n - p}$$

Fitting `p` parameters costs `p` **degrees of freedom**. With 2 points and 2
parameters you fit perfectly and get zero residuals — which says nothing about
noise, because you had no freedom left to be wrong.

Here the correction is ~5% (`40/38`). With 10 points and 9 parameters it is a
factor of 10.

### Why the parameters aren't exactly 3 and 7

`m=3.0707, c=6.2686`. With noiseless data the parameters were determined
exactly; now you estimate them from a noisy sample, and the estimate carries
error.

The errors are also **correlated**: slope slightly high, intercept slightly low.
A line pivots — raise the slope and the intercept must drop to keep passing
through the middle of the data. This is the same coupling that made `c` converge
so much more slowly than `m` in R4.

### Sampling variation is not a bug

```
population noise variance (sigma^2) = 4.0
sample variance, all 50   = 2.2984
sample variance, train 40 = 2.5120
sample variance, test 10  = 1.4231
```

`σ² = 4` describes the distribution drawn *from*; the values actually drawn have
their own variance. Test MSE came out **below** train MSE simply because those
10 points happened to be quieter. Change `SEED` and every number moves. The true
relationship never changes; the estimate of it does.

---

## Part 5 — Regularization (L1 and L2)

### What the penalties are

A second term added to the loss, so training minimises a **compromise** between
fitting the data and keeping parameters small:

$$L_{2}: \quad L + \lambda m^2 \qquad\qquad L_{1}: \quad L + \lambda |m|$$

This is the first time the loss contains something that is not about accuracy.

### The derivative is where they differ

| | penalty | gradient contribution | behaviour |
| --- | --- | --- | --- |
| **L2** (ridge, weight decay) | `λm²` | `2λm` — proportional to `m` | shrinks large values hard, small ones barely; never quite reaches zero |
| **L1** (lasso) | `λ\|m\|` | `λ·sign(m)` — **constant** | pushes just as hard at `m=0.001` as at `m=1000`; drives small values exactly to zero |

L1's gradient does not exist at `m = 0` (the absolute value has a corner).
`np.sign(0)` returns `0`, which is a valid subgradient choice — and since
training starts at `m = 0.0`, the very first L1 step carries no penalty at all.

### Why the intercept is never penalized

`c` is not a measure of complexity — it is where the data happens to sit. Add
1,000 to every `y` and the correct `c` becomes 1,007 while the relationship is
unchanged; penalising it would fight an arbitrary property of the units.

Watch `c` climb as `λ` grows: `6.27 → 8.23 → 15.56`. It is compensating for the
shrinking slope, the line pivoting to keep passing through the middle of the
data — the same `m`/`c` correlation seen in R6, now driven deliberately. Penalise
`c` too and it cannot compensate, so the fit collapses much faster.

Every real implementation excludes the bias for this reason. At L9, weight decay
is applied to weight matrices but conventionally **not** to biases or LayerNorm
parameters.

### What regularization does to the numbers

`lr=0.01`, 20,000 iterations, penalty on `m` only:

```
penalty   lambda           m         c    train MSE  test MSE
none         0.0      3.0707    6.2686       2.4766    1.6489
l2           0.1      3.0280    6.4894       2.4895    1.5159
l2           1.0      2.6914    8.2303       3.4975    1.7920
l2          10.0      1.2745   15.5581      25.3693   28.7412
l1           0.1      3.0637    6.3051       2.4769    1.6243
l1           1.0      3.0002    6.6331       2.5118    1.4496
l1          10.0      2.3660    9.9130       6.0000    4.2944
```

- **Train MSE always worsens** as `λ` grows. It must — the objective is no longer
  train MSE, and the unregularised fit was already optimal on that metric. A
  regularised model beating the unregularised one on *training* error means a
  bug.
- **Test MSE can improve** at small `λ` — the bias/variance trade, accepting a
  little bias to reduce sensitivity to this sample's noise. (With 10 test points
  these differences are within sampling noise; the direction is right, the
  magnitude is not evidence.)
- **Large `λ` underfits.** At `λ=10` L2 gives `m=1.27` against a true `3`.
- Both curves pass *through* the true `m = 3.0` on their way down.

### Which shrinks harder?

At `λ=10`, L2 reaches `m=1.27` and L1 only `2.37` — **L2 shrinks harder here**,
which contradicts the common half-memory that "L1 is the aggressive one".

Both claims are regime-dependent:

- **At large `m`:** L2's push is `2λm` while L1's is `λ`. At `m ≈ 3` that is six
  times stronger. L2 wins.
- **Near `m = 0`:** L2's push fades to nothing, L1's stays at full strength `λ`.
  L1 wins, and that is what drives parameters *exactly* to zero.

L1's sparsity reputation comes entirely from the second regime.

### When to use which

- **L2 by default**, especially for neural networks. Differentiable everywhere,
  well-behaved with correlated features (it shares weight among them), and it is
  what "weight decay" means in AdamW at L9.
- **L1 when sparsity is the goal** — many candidate features, most expected to be
  irrelevant, and a model that names the few that matter is worth more than a
  slightly more accurate dense one. A model with 12 non-zero coefficients out of
  800 is something a human can read; a dense one with 800 small weights is not.
  This is feature selection as a side effect of training.
- **Neither, when the model already has the right capacity.** With two parameters
  on genuinely linear data there is almost nothing to over-fit, which is why the
  gains above are marginal at best.

**The decision rule is not really "which penalty".** It is *"is there anything
here to over-fit?"* Regularization pays when capacity exceeds what the truth
requires. Choosing between L1 and L2 only matters once the answer to the first
question is yes.

L1's weakness with correlated features: given two near-identical inputs it picks
one arbitrarily and zeros the other, so small data changes can flip which
survives. L2 keeps both at half weight, which is more stable. **Elastic net**
uses both penalties at once for exactly this reason.

In deep learning L1 is rare on network weights; sparsity there is normally
pursued afterwards through pruning or quantisation instead.

### Evaluate without the penalty

Train with the regularised objective, but **report plain `mse`**. The penalty is
a training device, not a measure of prediction quality — scoring a model partly
on its own penalty says nothing about how well it predicts.

---

## Part 6 — Batching and SGD

*R8 — concepts worked through before implementation. Numbers below come from
running the R6 training set with a batched loop.*

### What is SGD?

**Stochastic Gradient Descent.** *Stochastic* means random, and the randomness
is in **which data you look at** before each step.

Everything up to R7 was **not** stochastic: every update used all 40 training
points, in the same order, every time. That is **full-batch gradient descent**.

SGD replaces "use all the data" with "use a randomly chosen piece of it".
Shuffle, take a slice, compute the gradient from just that slice, step. The
gradient becomes an *estimate* of the true gradient — a noisy one, drawn at
random. The update rule itself is unchanged:

$$m \leftarrow m - \eta \frac{\partial L}{\partial m}$$

Only the data behind `∂L/∂m` changed.

| name | points per update | gradient |
| --- | --- | --- |
| **full-batch** (batch GD) | all 40 | exact |
| **mini-batch** | 8 | noisy estimate |
| **SGD** (strict sense) | 1 | very noisy estimate |

### The terminology trap

"SGD" is used two ways:

- **Strictly** — batch size 1, one sample per update. The sense used in the table
  above, to keep it distinct from mini-batch.
- **In practice** — any mini-batch training, whatever the size. "Trained with
  SGD, batch size 256" is not a contradiction; it is the common usage.

`torch.optim.SGD` follows the loose sense. It neither knows nor cares about batch
size — it is just the plain update rule `param -= lr * grad`, as opposed to Adam
or RMSprop. So **SGD names the optimizer**, and separately describes estimating
gradients from random subsets. Batch size 1 is the extreme case, not the
definition.

### Why mini-batch beats full-batch per epoch

`lr=0.01`, 200 epochs each — equal passes over the data:

```
mode         batch  updates          m        c  train MSE  test MSE
full-batch      40      200     3.4538   3.7774     3.7780    4.5414
mini-batch       8     1000     3.0751   6.1869     2.4802    1.6792
mini-batch       6     1400     3.0345   6.2602     2.5242    1.6228
SGD              1     8000     2.5854   6.2440    10.5731   11.5070
```

Full-batch spends every point on one very accurate gradient and takes one step.
200 epochs buys 200 updates, and `m=3.4538, c=3.7774` is nowhere near converged.
Mini-batch-8 saw **exactly the same data** and reached `3.0751, 6.1869` —
essentially the R6 answer — because those same passes bought it 1,000 updates.

**A less accurate gradient computed five times beats an exact gradient computed
once.** That is the whole argument for mini-batching, and why every large model
is trained this way. At L9 the full-batch option will not exist — the dataset
will not fit in memory.

### Why SGD is worse despite 8,000 updates

More updates is not better if each is based on a bad estimate.

```
FULL-BATCH gradient at the minimum:  dm= -0.0017  dc= -0.0003   <- essentially zero

SINGLE-POINT gradients at the SAME parameters:
   point 0: x= 1.30  dm=   4.5528
   point 1: x= 9.68  dm=  11.0552
   point 4: x= 6.70  dm=  -3.7889
   point 5: x= 4.76  dm= -13.0250

   across all 40 points: dm ranges  -37.307 to   48.656, mean -0.0017
   one SGD step at lr=0.01 moves m by up to 0.4866
```

**At the minimum the full-batch gradient is zero. The individual point gradients
are not.** They span `-37` to `+49` and cancel only *on average* — their mean is
the full-batch gradient. SGD never sees the mean; it steps on one point at a
time. Point 1 says "increase `m`", point 5 says "decrease `m`, hard". Both are
right about their own point and wrong about the dataset.

This breaks the self-slowing seen in R4, where shrinking gradients
(`-108 → -82 → -62`) made steps smaller automatically near the minimum.
Single-point gradients **do not shrink** as the minimum is approached — no
individual point is ever predicted perfectly, because of the noise — so the steps
stay large forever. At `lr=0.01` each step moves `m` by up to `0.4866` against a
true value of `3.07`.

SGD therefore arrives in the neighbourhood and orbits. Update 8,000 is as
unsettled as update 800. Train MSE of `10.57` does not mean the parameters were
wrong on average; it means they were never anywhere in particular when training
stopped.

### Batch size and learning rate are coupled

Averaging `k` samples cuts the noise in the estimate by roughly `√k`. Eight
points reduce the wobble about 2.8×; forty about 6.3×.

| batch | gradient quality | updates per epoch |
| --- | --- | --- |
| 1 | terrible | 40 |
| 8 | decent | 5 |
| 40 | exact | 1 |

So halving the batch generally means reducing the learning rate too. A step size
that is fine averaged over 40 points is far too large applied to one.

Two fixes:

- **Lower the learning rate.** `lr=0.001` makes SGD stable on this data.
- **Decay the learning rate over time.** Big noisy steps early to cover ground,
  then shrink them so the noise cannot throw you around once you have arrived.
  This is the real answer, and precisely why **learning-rate schedules** exist —
  the warmup-and-decay added at L9. SGD without a schedule does not converge; it
  orbits.

That is also why R8 precedes R9: Momentum, RMSprop and Adam are all, in part,
machinery for coping with noisy mini-batch gradients.

### Batch size and epochs are independent knobs

A frequent tangle. They set different things:

- **`batch_size`** — how much data goes into *one* update. Controls gradient
  quality.
- **`epochs`** — how many times the whole dataset is swept. Controls how long
  training lasts.

Batch size says nothing about how long you train. What it *does* determine is
**batches per epoch**, and that is the mechanism behind mini-batch extracting
more updates from the same data:

```
batches per epoch = ceil(n / batch_size)
total updates     = batches per epoch × epochs
```

For `n = 40`:

| batch_size | batches per epoch | updates at 200 epochs | points processed |
| --- | --- | --- | --- |
| 40 | 1 | 200 | 8,000 |
| 8 | 5 | 1,000 | 8,000 |
| 6 | 7 (six of 6, one of 4) | 1,400 | 8,000 |
| 5 | 8 | 1,600 | 8,000 |
| 1 | 40 | 8,000 | 8,000 |

**The last column is identical for all rows** — that is what "equal epochs"
means. `40 × 200 = 8,000` points touched, whatever the batch size. What differs
is how many updates each run extracts from that same data. (`8,000` appearing
twice in the last row is a coincidence of batch size 1, where points and updates
coincide.)

Note `batch_size = 6` gives **7** batches, not 6.67 — six full batches plus a
short one of 4.

### What updates when

**Per batch** (five times per epoch at `batch_size=8`): `m`, `c`, the update
counter.

**Per epoch** (once): the shuffle order, and one loss measurement over the
**full** training set.

**Never:** the data. Only the order it is visited in changes.

So "per epoch" does not describe when parameters move — they move once per
batch. It describes the bookkeeping around them.

```
--- epoch 0 --- (order reshuffled)
   batch 0  (8 pts)  dm=-293.439  ->  m= 2.9344  c= 0.4560
   batch 1  (8 pts)  dm= -73.947  ->  m= 3.6739  c= 0.5879
   batch 2  (8 pts)  dm= -21.146  ->  m= 3.8853  c= 0.6325
   batch 3  (8 pts)  dm= -20.115  ->  m= 4.0865  c= 0.6995
   batch 4  (8 pts)  dm=  11.681  ->  m= 3.9697  c= 0.7002
   end of epoch: train MSE over ALL 40 points = 9.0557
--- epoch 1 --- (order reshuffled)
   ...
   end of epoch: train MSE over ALL 40 points = 11.2967
```

The gradients **disagree with each other**: `-293, -74, -21, -20, +11.7`. Batch 4
wants to move `m` in the opposite direction from batches 0–3. Neither is wrong —
they are different subsets, each honestly reporting what its own points want.
Full-batch averaged all of that into one number before stepping.

Record the epoch's loss on the **full** training set, not on the last batch and
not as an average of batch losses. Batch losses are computed on different subsets
at different parameter values, so averaging them mixes several things together.

### Loss can go up between epochs

Epoch 0 ends at `9.0557`, epoch 1 at `11.2967` — worse, despite five more
updates. **This is not a bug.**

With full-batch at a sane learning rate, loss decreased monotonically, so any
increase meant something was broken. That rule does not survive batching: each
step improves *its own batch* while possibly worsening the full-set average. The
curve trends down over many epochs while wobbling locally.

The diagnostic therefore changes from "did loss go up?" to "is the trend
descending over tens of epochs?" — the same judgement needed when reading a noisy
training curve at L9.

### R6 and R7 in this vocabulary

Those milestones had no batching concept at all — no `batch_size`, no shuffling,
no inner loop. Every iteration predicted all 40 points, computed one loss and one
gradient pair from all 40, and applied one update.

| R6 / R7 | R8 equivalent |
| --- | --- |
| `ITERATIONS = 20000` | `EPOCHS = 20000`, `batch_size = 40` |
| 20,000 iterations | 20,000 epochs, 1 batch each |
| 20,000 updates | 20,000 updates |

So `batch_size = 40` must reproduce R6/R7 exactly at the same epoch count — the
checkpoint that proves a batching loop is correct before trying anything
interesting.

**R6 and R7 never shuffled**, and did not need to: with one batch containing
everything, order cannot change a mean over 40 numbers. Shuffling only starts to
matter once the data is split, because then order decides *which points share a
batch*. That is why `rng.permutation` appears in R8 and nowhere earlier.

R8 adds exactly two things: an inner loop over batch slices, and a per-epoch
shuffle. `predict`, `mse`, `gradients` and `update_parameters` are unchanged.

### Choosing the epoch budget

Cost is not the constraint — the full sweep at 20,000 epochs takes about 15
seconds (`0.3s` for full-batch, `~10.6s` for SGD).

The argument against a large budget is that it makes the comparison **boring**.
At 20,000 epochs full-batch gets 20,000 updates and converges to `m=3.0707`,
mini-batch gets 100,000 and converges to the same place, and the final-value
table shows three identical rows. The difference between batch sizes lives
entirely in *how fast they arrived*.

Running the sweep at two budgets costs seconds and shows both facts:

- **Short budget** (e.g. 200 epochs) — endpoints differ; mini-batch is ahead on
  equal data.
- **Long budget** (e.g. 20,000) — see below. The result is not what you would
  guess.

### Only full-batch actually converges

Measured, 20,000 epochs:

```
mode        batch  updates           m         c   train MSE  test MSE
full-batch     40    20000      3.0707    6.2686      2.4766    1.6489
mini-batch      8   100000      2.9579    6.2479      2.9323    1.9855
mini-batch      6   140000      2.9518    6.2454      2.9841    2.0407
SGD             1   800000      2.6882    6.3497      7.1148    7.0862
```

Mini-batch-8 ends **worse** than full-batch, and worse than its own 200-epoch
result (`2.4802`). Training five times longer made it slightly worse.

The reason is the one from "Why SGD is worse despite 8,000 updates", applied at
every batch size below full: **at the minimum the full-batch gradient is zero,
but a batch gradient is not.** So full-batch's steps shrink to nothing and it
settles; every smaller batch keeps taking finite steps forever and orbits. Batch
8's orbit is roughly `√8 ≈ 2.8×` tighter than SGD's — smaller, not absent.

Running longer does not shrink the orbit. It just returns a different random
point on it.

So the accurate summary is **not** "mini-batch is faster to the same answer". It
is:

- mini-batch reaches the neighbourhood far faster, and never settles
- full-batch is much slower, and does settle
- **learning-rate decay is what closes the gap** — shrinking `η` shrinks the
  orbit, which is why schedules are not optional for batched training

Counted per update rather than per epoch, all four runs trace nearly the same
path, with full-batch as the smooth lower envelope and the smaller batches
wobbling around it. Per update, noise buys nothing.

### Regularization interacts with batch size only through speed

The penalty gradient (`2λm` or `λ·sign(m)`) is applied on **every update**, so
smaller batches apply it far more often — 8,000 times versus 200 at these
settings. That does not make the penalty stronger.

The equilibrium sits where the data gradient and the penalty gradient cancel,
and that balance point does not depend on how often you step. Measured shrinkage
in `m` from adding L2 `λ=0.1` over 200 epochs:

| batch | without penalty | with L2 λ=0.1 | shrinkage |
| --- | --- | --- | --- |
| 40 | 3.4538 | 3.4214 | −0.0324 |
| 8 | 3.0751 | 3.0298 | −0.0453 |
| 6 | 3.0345 | 2.9914 | −0.0431 |
| 1 | 2.5854 | 2.5508 | −0.0346 |

Comparable across a 40× range of update counts — not scaled by it. Batch size
changes how fast the equilibrium is reached, not where it is.

### Compare by epoch, not by update

Full-batch at 200 updates and SGD at 200 updates are not comparable — SGD has
seen 200 points and full-batch has seen 8,000. **Equal epochs means equal data
seen**, which is the fair comparison and the one the tables above use.

Both framings are legitimate and they rank the methods **oppositely**:

- **Per epoch** (equal data): mini-batch wins decisively.
- **Per update** (equal steps): full-batch wins, since each of its steps uses an
  exact gradient.

The ML convention is to count epochs because **data is the expensive resource** —
reading and processing points costs time, while the parameter update itself is
trivial. Mini-batch wins on the axis that reflects real cost.

Whichever is used, say which one the table is holding constant, since the ranking
depends on the answer.

---

## Part 7 — Optimizers (Momentum, RMSprop, Adam)

*R9 — concepts worked through before implementation. Numbers measured on the R6
training set, full-batch, `lr=0.01`, starting from `m = c = 0`.*

### The four update rules

Everything else in the loop is unchanged — same `predict`, `mse`, `gradients`.
Only the line turning `g` into a step differs.

```
plain GD                          state: none
    p = p - lr * g

momentum                          state: v (one per parameter)
    v = beta * v + g
    p = p - lr * v

RMSprop                           state: s (one per parameter)
    s = beta * s + (1 - beta) * g**2
    p = p - lr * g / (sqrt(s) + eps)

Adam                              state: v, s (two per parameter) + step count t
    v     = b1 * v + (1 - b1) * g
    s     = b2 * s + (1 - b2) * g**2
    v_hat = v / (1 - b1**t)
    s_hat = s / (1 - b2**t)
    p     = p - lr * v_hat / (sqrt(s_hat) + eps)
```

Defaults: `beta = 0.9`; `b1 = 0.9`, `b2 = 0.999`, `eps = 1e-8`. `t` starts at 1.

Two momentum conventions exist — `v = beta*v + g` (PyTorch's, and the one these
numbers use) and `v = beta*v + (1-beta)*g`. The second scales `v` down 10×, so a
learning rate tuned for one is wrong for the other.

### What each update actually consumes

```
GD         p = p - lr * g                          uses g
momentum   p = p - lr * v                          uses v
RMSprop    p = p - lr * g / (sqrt(s) + eps)        uses g AND s
Adam       p = p - lr * v_hat/(sqrt(s_hat) + eps)  uses v AND s, not g
```

**`g` does not appear in Adam's update line.** It enters only through `v` and
`s`. RMSprop keeps raw `g` in the numerator; Adam swaps it for the smoothed
`v_hat` — that substitution *is* the difference between them. Adam is RMSprop
with momentum in the numerator.

### Written out for two parameters

```
state carried:  v_m, s_m, v_c, s_c, t          (5 numbers)

each update:
    t = t + 1                                   <- shared, counts UPDATES

    g_m, g_c = gradients(x_batch, y_batch, predict(...))

    v_m = b1 * v_m + (1 - b1) * g_m
    s_m = b2 * s_m + (1 - b2) * g_m**2
    m   = m - lr * (v_m/(1 - b1**t)) / (sqrt(s_m/(1 - b2**t)) + eps)

    v_c = b1 * v_c + (1 - b1) * g_c
    s_c = b2 * s_c + (1 - b2) * g_c**2
    c   = c - lr * (v_c/(1 - b1**t)) / (sqrt(s_c/(1 - b2**t)) + eps)
```

- **Separate state per parameter.** `v_m` and `v_c` never mix. Sharing them
  defeats the purpose — each parameter must be scaled by its *own* gradient
  history.
- **`t` is shared**, incremented once per update, never reset.
- **`v` and `s` persist across updates**, initialised to zero once before
  training. Re-initialising inside the loop reduces Adam to a fixed `g/|g|` sign
  step. It will not crash; it will just train oddly — the same silent failure
  mode as the R2 accumulation bug.

### Everything updates once per batch

The optimizer never sees epochs, only updates. With `batch_size=8` over 40
points, `v`, `s` and `t` all advance five times per epoch; with full-batch, once.

Adam's `t` therefore counts **updates**, not epochs, and must never reset — reset
it per epoch and bias correction fires again every epoch, inflating each epoch's
first step.

### What it costs, measured

```
optimizer   iters          m        c  train MSE
plain GD      200     3.4538   3.7774     3.7780
momentum      200     3.0707   6.2684     2.4766     <- converged
RMSprop       200     2.0197   2.0226   104.0524
Adam          200     1.7358   1.7446   145.7151

plain GD     2000     3.0710   6.2671     2.4766
momentum     2000     3.0707   6.2686     2.4766
RMSprop      2000     3.0657   6.2636     2.4777
Adam         2000     3.3327   4.6100     3.0559

plain GD    20000     3.0707   6.2686     2.4766
momentum    20000     3.0707   6.2686     2.4766
RMSprop     20000     3.0657   6.2636     2.4777
Adam        20000     3.0708   6.2687     2.4766
```

**Momentum reaches in 200 iterations what plain GD needs 20,000 for** — same
data, same learning rate, same gradients, a hundredfold difference from three
extra lines.

**RMSprop and Adam are *slower* here**, which is not a bug. See below.

### Momentum: velocity builds, and overshoots

```
  t        g          v = 0.9v+g     step=lr*v        m
  1   -272.6719       -272.6719      -2.7267     2.7267
  2    -83.5390       -328.9437      -3.2894     6.0162
  3    144.7949       -151.2544      -1.5125     7.5287
  4    250.2348        114.1059       1.1411     6.3876
  5    172.0263        274.7216       2.7472     3.6404
  8   -196.0462       -167.7014      -1.6770     2.7029
```

`m` swings `2.7 → 6.0 → 7.5 → 6.4 → 3.6 → 2.7` and still lands exactly on
`3.0707` by iteration 200.

With `beta = 0.9` a sustained gradient accumulates until `v ≈ g/(1-beta) = 10g`,
so **momentum's effective learning rate is about `lr/(1-beta)`, i.e. 10× `lr`**.
That is where the speedup comes from, and why early steps look wild — `0.1` is
above the `0.0845` divergence threshold plain GD had in R4. Momentum has
different stability conditions, so it survives and damps out.

### RMSprop: the step shrinks as `s` warms up

```
  t        g       s = .9s+.1g^2    sqrt(s)   g/sqrt(s)   step        m
  1   -272.672          7435.0     86.226    -3.1623   -0.0316    0.0316
  2   -270.205         13992.5    118.290    -2.2843   -0.0228    0.0545
  4   -266.934         24943.9    157.936    -1.6901   -0.0169    0.0904
  8   -262.216         40286.0    200.714    -1.3064   -0.0131    0.1471
```

`s` starts at 0 and climbs toward the typical squared gradient (`g**2 ≈ 72,000`).
While it is small, `sqrt(s)` is small, so early steps are inflated. Once `s`
settles at `g**2`, `sqrt(s) = |g|`, the ratio becomes exactly ±1, and the step
converges to `lr` regardless of gradient size.

### Adam: a constant step of exactly `lr`

```
  t        g          v        v_hat          s        s_hat    step        m
  1   -272.672    -27.267    -272.672         74.3      74350.0  -0.0100    0.0100
  2   -271.892    -51.730    -272.261        148.2      74137.4  -0.0100    0.0200
  4   -270.331    -93.334    -271.399        294.4      73713.3  -0.0100    0.0400
  8   -267.213   -153.498    -269.515        580.9      72869.5  -0.0100    0.0800
```

**The step is `-0.0100` every iteration — exactly `lr`.** `v_hat` tracks `g` and
`s_hat` tracks `g**2`, so `v_hat/sqrt(s_hat) = g/|g| = -1`.

That is Adam's defining property: while the gradient sign is consistent, each
parameter moves by exactly the learning rate per step, whatever the gradient's
magnitude.

It also explains the "slow" result. Moving `m` from 0 to `3.07` at `0.01` per
step needs ~307 steps; `c` needs ~627. At 200 iterations Adam has `m = 1.74` —
on schedule, not broken.

**So `lr` means something different for Adam.** For GD it scales the gradient;
for Adam it *is* the step size. Choose it by asking "how far should a parameter
move per step?" **Learning rates are not transferable between optimizers.**

### Bias correction, visible

With a constant gradient of `100`:

```
 t     v (raw)   1-b1^t     v_hat
 1      10.000    0.1000    100.000
 2      19.000    0.1900    100.000
 3      27.100    0.2710    100.000
 6      46.856    0.4686    100.000
```

The raw average reads `10` at step 1 — a tenth of the truth, because it started
from zero. Dividing by `1 - b1**t` recovers exactly `100.000` every step.

In the Adam trace above: at `t=1`, raw `v = -27.267` but `v_hat = -272.672`,
exactly `g`; raw `s = 74.3` but `s_hat = 74,350`, exactly `g**2`.

Without correction the first steps would be far too small, and `s` with
`b2 = 0.999` would need ~1,000 steps to warm up. `1 - b1**t → 1` as `t` grows, so
the correction fades — it only matters early, which is exactly why `t` must be
the true update count.

### How momentum cancels swings

Expanding `v = beta*v + g` gives a weighted sum of every past gradient with
exponentially fading weights:

```
v_t  =  g_t  +  0.9*g_(t-1)  +  0.81*g_(t-2)  +  0.729*g_(t-3)  +  ...
```

What happens next depends entirely on whether those gradients agree. Same
magnitude `100` in both columns:

```
  t   consistent g     v        |   alternating g     v
   1        100.0    100.00      |        100.0    100.00
   2        100.0    190.00      |       -100.0    -10.00
   4        100.0    343.90      |       -100.0    -18.10
   8        100.0    569.53      |       -100.0    -29.98
  12        100.0    717.57      |       -100.0    -37.77

  steady state: consistent  -> g/(1-b) = 1000.00   (10.0x amplified)
                alternating -> g/(1+b) =   52.63   ( 0.53x damped)
  ratio: 19.0x
```

Terms with the same sign add; opposite signs subtract. A **19× difference in
effective step** between a direction that keeps pointing the same way and one
that keeps reversing — not a decision, just a consequence of the weighted sum.

### Correction: on this problem both parameters swing together

The natural story is "`c`'s gradient stays consistent while `m`'s alternates, so
momentum speeds up the slow one". **Measurement says otherwise** — `g_m` and
`g_c` flip sign in lockstep, every iteration:

```
  t    g_m       sign   |    g_c       sign
   1    -272.67   -    |    -44.30   -
   3     144.79   +    |     19.92   +
   5     172.03   +    |     24.16   +
   7    -175.47   -    |    -29.18   -
  10      81.84   +    |     10.54   +
```

The oscillation is not along the `m` axis or the `c` axis. It is along the steep
direction of the loss surface, which is a *mixture* of both (roughly 96% `m`,
26% `c`) — the same `m`/`c` correlation seen since R6. Both coordinates project
onto it, so both swing in phase.

So momentum here is not separating a fast parameter from a slow one. It is
buying the **10× effective learning rate** on the sustained part of the gradient,
carrying both parameters down the long valley floor, while the in-phase
oscillation damps out over the first hundred or so iterations. The cancellation
mechanism is real; it just operates along the loss surface's own axes, not along
`m` and `c`.

### `v` cancels, `s` never does

RMSprop and Adam *do* treat parameters separately, and unlike momentum they do
not care about sign agreement at all — `s` accumulates `g**2`, always positive,
so nothing cancels.

- **`v` is a signed average** — disagreement across time destroys it. Handles
  **oscillation**.
- **`s` is a squared average** — always positive, never cancels. Handles
  **scale**.

`s_m` settles near `g_m**2 ≈ 72,000`; `s_c` near `g_c**2 ≈ 1,900`. Dividing each
step by `sqrt(s)` divides out that parameter's own gradient scale, so both move
at about `lr` per step.

Two different problems, two different accumulators. Adam keeps both. The formulas
look similar enough that the distinction is easy to lose.

### Practical notes

**Memory.** Adam stores two extra numbers per parameter, so optimizer state is
**twice the model size**. Irrelevant with two parameters; at L12 it is a headline
number when estimating training memory — a model that fits may not fit *with*
Adam.

**`eps` is not optional.** `s` starts at zero, so the first RMSprop or Adam step
divides by `sqrt(0)`. `eps = 1e-8` is what prevents that.

**Cheapest correctness checks.** `beta = 0` in momentum must collapse to exactly
plain GD (`v = 0*v + g = g`). `b1 = b2 = 0` in Adam reduces it to `g/(|g|+eps)`, a
pure sign step. If the degenerate cases misbehave, the general case will too.

**Naming collision.** `m` is the slope here; `m` is also the conventional name
for Adam's first moment (the original paper uses `m` and `v`). Use `v_m, v_c` and
`s_m, s_c` so the parameter letters keep their meaning.

---

## Part 8 — Modules, imports, and `__main__`

### Importing a module runs it

`import regression_from_scratch` doesn't merely "make functions available" —
Python **executes the file top to bottom**. That execution is *how* the
functions come into existence; a `def` is a statement that runs. But every other
top-level statement runs too, including `print`.

That is why importing a file with twelve top-level prints produced 15 lines of
noise before the importing script did anything.

### What `if __name__ == "__main__":` does

Python sets `__name__` in every module, and its value depends on **how the file
was loaded**:

- Run directly → `__name__` is `"__main__"`
- Imported → `__name__` is the module's own name

```
$ python3 toolbox.py
toolbox: __name__ is __main__
toolbox: this only runs when toolbox.py is run directly

$ python3 main.py
toolbox: __name__ is toolbox        <- unwrapped print still fires
main: __name__ is __main__
main: helper() = 42                 <- wrapped print did NOT fire
```

The guard reads: *"only do this when this file is the one being run."* It is an
ordinary `if`; the condition just happens to be true in one case and false in
the other. One file, two roles: a **toolbox** when imported, a **script** when
run.

### Why did the model train twice?

Because `training.py` contained `from training import ...` — the file importing
itself.

1. `python training.py` loads the file and names it `__main__`.
2. The self-import asks for a module named `training`. The loaded one is called
   `__main__`, so as far as Python is concerned **`training` has not been loaded
   yet.** It reads the same file off disk and executes it again as a second
   module.
3. That second copy runs with `__name__ == "training"`, so its guarded block is
   skipped — which is why the five-point check printed once but the training run
   printed twice.
4. Its own self-import succeeds without recursing, because `sys.modules` already
   holds the partially-built module and the names it wants are defined above the
   import line.
5. The second copy trains and plots. Control returns to `__main__`, which trains
   and plots **again**.

The file existed twice in memory under two names, with two copies of every
array and function. Only the seeding made them identical.

**Rule: a file should never import itself.** If you write
`from <this file> import ...`, either the import is unnecessary or the code
below it belongs in a different file.

---

## Part 9 — Bugs hit, and what caused them

### The sign inversion

Comments said `-2(y - ŷ)`; code said `2 * (actual - predicted)`. The minus was
dropped when substituting `dL/dŷ` into the chain rule, in two places.

Nothing crashed. Every line was well-formed. The only symptom was a number
moving the wrong way:

```
                sign wrong              sign fixed
start   loss =    274.0                   274.0000
step 1  loss =    415.9                   162.1264
step 2  loss =    632.8                    96.9154
step 3  loss =    964.6                    58.8973
step 4  loss =  1,471.9                    36.7258
step 5  loss =  2,247.7                    23.7892
```

**Convention, fixed permanently:** `residual = predicted - actual`. Squaring
hides the sign in the loss; the gradient formulas carry it directly.

### Two ways loss explodes, and how to tell them apart

| | sign error | learning rate too large |
| --- | --- | --- |
| shape | steady climb | geometric explosion |
| numbers | `274 → 416 → 633 → 965` | `274 → 1.03e13 → 3.92e23` |
| parameter sign | marches one way | **flips every step** |
| fix | the formula | the step size |

The oscillation is the giveaway. At `lr=0.09`, `m` overshoots the minimum, lands
further out on the far side where the gradient is *bigger*, and swings back
harder. Sampling the pre-update value shows the sign flipping every iteration;
sampling post-update hides it.

The finite-difference check distinguishes the two definitively in one run.

### Gradients returned as lists (twice)

First in the list version, then again after the NumPy conversion. Both times the
**reduction** was missing — the per-observation values were returned instead of
being summed and divided by `n`. In the NumPy case, `total_m = 0.0` followed by
`+= residual * x` silently turns `total_m` into an array.

**Rule:** a gradient is **one number per parameter**. `update_parameters` should
do nothing but `m - learning_rate * dm`.

### MSE as a sum instead of a mean

`sum(...)` with no `/n` gives SSE, not MSE — `1124862.0` instead of a sensible
number. Knock-on effects: gradients `n` times too large (equivalent to an `n`×
learning rate), and every loss value incomparable with earlier milestones.

### A function mutating a list it didn't own

```python
model_outputs = []              # module level

def linear_regression(inputs, m, c):
    for j in inputs:
        model_outputs.append(...)   # appends to the OUTER list
    return model_outputs
```

Assigning to a name inside a function creates a local. **Mutating** an object
found in the enclosing scope changes it for everyone — `.append()` is mutation.
So the function accumulated permanently across calls:

```
after 1st call: [10, 13, 16, 19, 22, 10, 13, 16, 19, 22]
after 2nd call: [10, 13, 16, 19, 22, 10, 13, 16, 19, 22, 10, 13, 16, 19, 22]
```

Running the file once hid it completely. The same shape recurred later when
`train` was handed a `history_list` created at module level — a second call
would have concatenated onto the first run's history.

### `append` takes one argument and returns `None`

```
append(loss, m, c)  ->  TypeError: list.append() takes exactly one argument (3 given)
h2.append((a,b,c)) returns -> None    and h2 is now [(1.0, 2.0, 3.0)]
```

Three values go in as **one tuple** — double parentheses. And
`history_list = history_list.append(...)` sets the list to `None`; the crash then
surfaces on the *next* iteration, one line away from the actual bug.

### Shadowing a builtin, and one-shot iterators

```python
zip = zip(inputs, actual_outputs, model_outputs)
```

```
first  list(zip): [(1, 3), (2, 4)]
second list(zip): []                      <- silently empty, no error
calling zip() again -> TypeError: 'zip' object is not callable
```

Two bugs: the builtin name is destroyed, and a zip object is a **one-shot
iterator** that yields nothing on a second pass. At R8, iterating mini-batches
across epochs, this trains on epoch 1 and silently trains on nothing after.

The mirror image happened later: naming a parameter `input` and passing the
builtin `input` function as data, producing
`TypeError: 'builtin_function_or_method' object is not iterable`. Both come from
names that collide with Python's own.

### A typo that never ends

```python
initail = initial + 1        # note the spelling
```

Python creates a brand new variable rather than objecting. `initial` stayed `1`,
`while True` never broke.

The real lesson isn't the typo — it is that the counter was hand-maintained at
all. `for i in range(iterations)` counts and stops on its own: no `initial`, no
increment, no `break`, nothing to misspell. It also removed an off-by-one, since
starting at `1` and breaking on `>= 1000` ran 999 iterations.

Same shape as the manual index `a = 0` / `a += 1` that `zip` removed in R1.

### `zip` stops at the shortest input

It truncates silently. Validating only two of three zipped collections leaves a
hole; a short third list quietly produces a gradient over fewer observations.
Chained comparison closes it:

```python
if not (n == len(model_outputs) == len(inputs)):
```

### Default arguments and parameter shadowing

Changing `h=1e-5` to bare `h` made the argument **required** and broke the
existing four-argument call. And a parameter named `h` **shadows** any
module-level `h`, so moving the constant into another file and importing it has
no effect inside the function.

### Fake noise

`(i % 5) * 2` produces `[0, 2, 4, 6, 8, 0, 2, 4, ...]` — periodic, not random,
and with mean **+4** rather than 0. That is a systematic shift, so the data
followed `y = 3x + 11` on average and the learned `c` would have come out near
11, looking like a model failure rather than a generator bug.

Real noise is symmetric and zero-mean: `rng.normal(0, sigma, n)`.

---

## Part 10 — Process questions

### Where were those five training steps running?

Not in the project file — in a throwaway snippet that imported the module and
looped around its functions. The project file at that point performed exactly
one update. Done deliberately: wrapping the step in a loop was the R4 task.

**Every number reported about this code comes from running this code.** If one
looks wrong, ask what was run.

### What is the training-loop flow?

Four steps, repeated:

1. **Predict** with the current `m` and `c`
2. **Measure** the loss
3. **Compute gradients**
4. **Update** `m` and `c`

**Only `m` and `c` survive between iterations.** Everything else is rebuilt.
Predictions must be recomputed *inside* the loop — computing them once before it
would give the same stale loss and gradients forever.

The loop counter does nothing but count; `i` never appears in the arithmetic.
That is genuinely different from R1–R3's loops, where the loop variable *was*
the data.

Watch the gradients shrink (`-108 → -82.32 → -62.72`): as parameters approach
the truth the slope flattens, so steps get smaller **automatically** at a fixed
learning rate.

### Iterations vs data points — are we training on the same data repeatedly?

Yes. 40 training points, 20,000 iterations: **the same 40 points, 20,000 times
over.** Two completely different axes:

| | count | what it means |
| --- | --- | --- |
| **data points** | 40 | how much evidence each update is based on |
| **iterations** | 20,000 | how many updates you perform |

Every iteration uses **all 40 points**: predict 40 values, reduce to one loss,
reduce to two gradients, apply one update. In total 20,000 × 40 = 800,000
predictions were computed.

Repeating is not pointless because **the data stays the same but the parameters
change.** Iteration 1 predicts with `m=0, c=0`; iteration 2 with `m=0.11,
c=0.02`. Same `x`, different `m`, so different predictions, residuals, and
gradients. The gradient gives a direction and never a distance, so the method is
necessarily: small step, look again, small step. One pass could never find the
answer however many points it had.

### Epoch

An **epoch** is one complete pass through the training set.

Because each iteration here consumes all 40 points, **1 iteration = 1 epoch** —
this is **full-batch** gradient descent, and the two words are interchangeable
for now.

That equivalence breaks at **R8**, which is what that milestone is about. With
mini-batches of 8, one epoch contains 5 iterations — 5 updates per pass, each
based on 8 points instead of 40. From then on "iterations" and "epochs" are
different numbers and you have to say which you mean. At L9 both get counted,
for a real reason: more frequent updates on less evidence each, versus fewer
updates on more.

### Why 20,000 iterations?

Overkill, deliberately. Iterations 4,000 through 16,000 are identical to six
decimal places — it converged around iteration 1,000 and the remaining 19,000
changed nothing measurable. 20,000 was chosen to be safely past convergence for
the reference numbers.

Knowing when to stop is its own topic. At L9 it is called **early stopping**, and
it is driven by validation loss rather than by picking a large number and hoping.

### Reading the print counter

`print_every=4000` over `range(20000)` prints at `i = 0, 4000, 8000, 12000,
16000` — five lines. `i = 20000` never occurs, since the loop stops at 19,999.

### Should there be graphs?

Yes, at **R5**. But R4 came first because it produces what R5 plots, and because
divergence in a terminal looks like `loss=274.0 loss=415.9 loss=632.8 ...` —
recognising that in a column of numbers matters, since at L9 the training loop
emits loss to a terminal and there isn't always a chart.

**Log axes are not optional for loss curves.** On a linear axis the fall from
274 to 3 happens in the first few dozen iterations and the remaining 950 look
like a flat line at zero, as though training stopped. The same data on a log
axis is a clean descending slope with structure throughout.

Straight lines on a log axis mean **geometric convergence** — each iteration
multiplies the loss by a constant factor, and the learning rate sets that
factor.

A diverging run spanning 110 decades will squash every converging curve into an
unreadable band; plot it separately.

### When is a function obsolete?

Three cases came up with three different answers:

- **`dL_dm` / `dL_dc`** — *delete.* Fully superseded. Two implementations of one
  formula is how the sign drift survived.
- **`dL_dy`** — *delete, or wire it in.* Worth keeping only if something calls
  it; an uncalled function drifts out of sync with its replacement.
- **`update_parameters`** — *keep, change the signature.* The function is the
  gradient-descent step itself; only its list-taking interface was obsolete.

Before deleting a superseded implementation, run it against its replacement
once.

---

## Part 11 — Reference numbers

### Clean five-point dataset

`x = [1,2,3,4,5]`, `y = [10,13,16,19,22]`, true `m = 3`, `c = 7`.

| `m` | `c` | `L` | `∂L/∂m` | `∂L/∂c` |
| --- | --- | --- | --- | --- |
| `3` | `7` | `0.0` | `0.0` | `0.0` |
| `0` | `0` | `274.0` | `-108.0` | `-32.0` |
| `2` | `5` | `27.0` | `-34.0` | `-10.0` |
| `5` | `9` | `72.0` | `+56.0` | `+16.0` |

At the true parameters everything is zero — the bottom of the bowl. **A formula
with a sign error also returns zero there**, so it is useless for verification.
Test at wrong parameters. At `m=5, c=9`, past the true values, both gradients
are **positive** — that is the sign convention's proof.

### Learning rates, from `m=0, c=0`, 1000 iterations

| learning rate | final `m` | final `c` | final loss |
| --- | --- | --- | --- |
| `0.001` | `4.13165` | `2.91439` | `3.03828` |
| `0.01` | `3.05367` | `6.80623` | `0.00683442` |
| `0.05` | `3.0000` | `7.0000` | `9.27867e-15` |
| `0.09` | `-3.76003e+53` | `-1.04147e+53` | `1.80097e+108` |

- `0.001` hasn't failed, it hasn't **finished** — loss is still falling. Around
  20,000 iterations gets there.
- `0.05` reaches machine precision.
- The divergence threshold is near **`0.0845`**. The boundary is sharp.

### Noisy dataset (R6)

`default_rng(42)`, 50 points, `x` uniform on `[0, 10]`, noise `normal(0, 2)`,
shuffled 80/20 split, `lr=0.01`, 20,000 iterations.

```
learned     m=3.0707  c=6.2686      (true 3.0, 7.0)
train MSE   2.4766
test  MSE   1.6489
noise floor 4.0000   (sigma^2)
```

Converged by roughly iteration 1,000; iterations 4,000 through 16,000 are
identical to six decimal places. Training past the plateau changes nothing.

### Why `c` converges slower than `m`

Inputs `1..5` scale `m`'s gradient and leave `c`'s unscaled, so the two do not
converge at the same speed. The asymmetry is real, and it is why **feature
scaling** exists.
