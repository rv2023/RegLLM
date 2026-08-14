# Q&A Notes

Questions asked during the build, and the answers, kept so they don't have to be
re-derived. Organised by topic rather than by date. Milestone references point
at [MILESTONES.md](../MILESTONES.md).

Every number here was produced by running the project's own code — none are
illustrative.

Covers R1–R7 complete, plus the R8 concepts worked through so far.

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

## Part 7 — Modules, imports, and `__main__`

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

## Part 8 — Bugs hit, and what caused them

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

## Part 9 — Process questions

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

## Part 10 — Reference numbers

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
