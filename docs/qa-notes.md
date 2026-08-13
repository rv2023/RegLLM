# Q&A Notes

Questions asked during the build, and the answers, kept so they don't have to be
re-derived. Organised by topic rather than by date. Milestone references point
at [MILESTONES.md](../MILESTONES.md).

Every number here was produced by running the project's own code — none are
illustrative.

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

- `dL/dŷ` — one per observation. Five predictions, five values. Each `ŷᵢ` is its
  own quantity that can be nudged independently.
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

Five numbers in, one number out. Twice, with different weights. That is all
`gradients()` does.

### Does this generalise?

Yes: **you get exactly as many gradients as you have parameters**, regardless of
how much data there is. Ten thousand observations still produce one `dL/dm` —
the sum just has ten thousand terms.

- **R5**, with more features: `m₁, m₂, m₃, c` → four gradients, still `n` values
  of `dL/dŷ`.
- **L7**, the transformer: `dL/dlogits` has shape `[batch, sequence, vocab]` — a
  huge number of per-position derivatives — collapsing into one gradient per
  weight, shaped like the weight itself. Batch size changes the number of
  paths, never the number of gradients.

### The consequence to remember for L9

Because gradients accumulate over paths, PyTorch makes `.grad` something you
**add into**, not overwrite. Hence every training loop containing:

```python
optimizer.zero_grad()
```

Forget it and this iteration's gradients pile onto last iteration's — the same
accumulation done deliberately across observations, happening accidentally
across time steps.

It is also the machinery behind **gradient accumulation** (L12): deliberately
*not* zeroing across several small batches to simulate one large one.

### Why compute both gradients in one pass?

**Correctness.** Two separate functions each recompute the residual, so the sign
convention lives in two places and can drift. This actually happened here — see
[the sign inversion](#the-sign-inversion). One residual computed once cannot
disagree with itself.

**Performance.** Two functions mean two full traversals per training step.
Irrelevant at five observations; at L9 it is the difference between a training
run that finishes and one that doesn't. Backpropagation exists specifically to
compute all gradients in a single backward sweep, reusing shared intermediates.

---

## Part 2 — The finite-difference check

### What is it for?

The gradient formulas were derived by hand. Nothing about writing them down
proves they are right. A finite-difference check measures the same slope a
completely different way, **using no calculus at all**: nudge a parameter up a
little, nudge it down a little, see how much the loss actually moved, divide by
the distance travelled. Rise over run.

If the hand-derived formula and the measurement agree, the derivation is right.

It must call the project's real `MSE` and `linear_regression`. If it
reimplemented them it could share a bug with the thing it is checking and agree
for the wrong reason.

### Why two separate lines?

```python
dm = (loss(m + h, c) - loss(m - h, c)) / (2 * h)
dc = (loss(m, c + h) - loss(m, c - h)) / (2 * h)
```

The first line moves only `m` and leaves `c` alone; the second moves only `c`.
Holding the other fixed is what makes each a *partial* derivative.

### What is `h` called?

**Step size** — also *perturbation size*, *finite-difference step*, or
*interval*. In ML code it is very often named **epsilon** (`eps`, `ε`) instead
of `h`. The `h` notation comes from the limit definition of a derivative:

$$f'(p) = \lim_{h \to 0} \frac{f(p+h) - f(p-h)}{2h}$$

The check computes exactly this, minus the limit — stopping at a small `h`
because zero would divide by zero.

**Name collision worth knowing:** "step size" is also a common synonym for the
**learning rate** `η`. Two unrelated quantities:

| | `h` / epsilon | `η` / learning rate |
| --- | --- | --- |
| Purpose | measure a slope | move along a slope |
| Wanted | as small as precision allows | tuned; neither tiny nor huge |
| Changes parameters? | no — temporary probe, discarded | yes — this is the update |
| Typical value | `1e-5` to `1e-7` | `0.001` to `0.1` |

In an optimizer, "step size" means the learning rate. In a numerical-differentiation
routine, it means `h`.

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

`h` has vanished from the result entirely. Any value works, because there is no
`h` left to be wrong about. The symmetry does it: stepping equally in both
directions makes the even-powered terms identical on each side.

Expanding `L = (1/n)Σ(mx + c − y)²` in terms of `m` and `c` gives `m²`, `c²`,
`mc`, `m`, `c`, constants — **degree 2, nothing higher**. The loss surface is a
perfect parabolic bowl and the central difference reads its slope exactly.

### Where it stops being exact

`f(p) = p³`, true derivative `3p²`:

$$\frac{f(p+h) - f(p-h)}{2h} = \frac{6p^2h + 2h^3}{2h} = 3p^2 + h^2$$

The `h³` doesn't cancel — cubing breaks the symmetry — leaving an error of
exactly `h²`. Measured at `p = 2`, where the true slope is `12`:

| `h` | predicted error `h²` | measured |
| --- | --- | --- |
| `1` | `1` | `13.000000` |
| `0.01` | `0.0001` | `12.000100` |
| `1e-5` | `1e-10` | `12.000000` |

Off by exactly `h²` every time.

### So why keep `1e-5`?

Two failure modes squeeze `h` from both sides:

- **Too large:** the `h²` truncation error above. Zero for this MSE, *not* zero
  for cross-entropy at L8 or anything with a nonlinearity.
- **Too small:** floating-point cancellation. `f(p+h)` and `f(p−h)` become
  nearly identical; subtracting destroys the significant digits, and that
  wreckage is then divided by a tiny `2h`. `h = 1e-15` fails *harder* than
  `h = 1`.

`1e-5` sits in the valley between them. This dataset is simply too forgiving to
punish a bad choice.

### Comparing the two results

```python
abs(analytical - numerical) < tolerance     # never ==
```

The two values reach the same answer by different computational routes and will
differ in the final digits. Observed differences here were around `1e-10` —
vanishingly small, but not zero.

---

## Part 3 — Bugs hit, and what caused them

### The sign inversion

The comments said `-2(y - ŷ)`; the code said `2 * (actual - predicted)`. The
minus was dropped when substituting `dL/dŷ` into the chain rule, in two places.

Nothing crashed. No exception, no warning. Every line was well-formed. The only
symptom was a number moving the wrong way:

```
                sign wrong              sign fixed
start   loss =    274.0                   274.0000
step 1  loss =    415.9                   162.1264
step 2  loss =    632.8                    96.9154
step 3  loss =    964.6                    58.8973
step 4  loss =  1,471.9                    36.7258
step 5  loss =  2,247.7                    23.7892
```

Same code structure. One character.

**Lesson:** a gradient sign error and a too-large learning rate produce the
*identical* symptom — loss climbing to infinity — and have completely different
fixes. The finite-difference check distinguishes them in one run. It would have
caught this immediately; three review rounds were spent instead.

**Convention, fixed permanently:** `residual = predicted - actual`. Squaring
hides the sign in the loss, but the gradient formulas carry it directly.

### Gradients returned as lists

The gradient functions returned one value per observation instead of summing and
dividing by `n`. The reduction had migrated into `update_parameters`, which meant
the update function was doing calculus that belonged to the gradient function.

**Rule:** a gradient is **one number per parameter**. `update_parameters` should
do nothing but `m - learning_rate * dm`.

### A function mutating a list it didn't own

```python
model_outputs = []              # module level

def linear_regression(inputs, m, c):
    for j in inputs:
        model_outputs.append(...)   # appends to the OUTER list
    return model_outputs
```

Assigning to a name inside a function creates a local. **Mutating** an object
found in the enclosing scope reaches out and changes it for everyone —
`.append()` is mutation. So the function accumulated permanently across calls:

```
after 1st call: [10, 13, 16, 19, 22, 10, 13, 16, 19, 22]
after 2nd call: [10, 13, 16, 19, 22, 10, 13, 16, 19, 22, 10, 13, 16, 19, 22]
```

Running the file once hid it completely. A 500-iteration training loop would
have built a 2,500-element list. **Fix:** the function creates its own list and
returns it, depending on nothing but its arguments.

### Shadowing a builtin, and one-shot iterators

```python
zip = zip(inputs, actual_outputs, model_outputs)
```

Two bugs in one line:

```
first  list(zip): [(1, 3), (2, 4)]
second list(zip): []                      <- silently empty, no error
calling zip() again -> TypeError: 'zip' object is not callable
```

1. The name `zip` no longer refers to the builtin — any later call fails.
2. A zip object is a **one-shot iterator**. Walk it twice and the second pass
   yields nothing. No error, no warning.

Survivable only because the script used it exactly once. At R8, iterating
mini-batches across epochs, this produces a loop that trains on epoch 1 and
silently trains on nothing afterward.

### Default arguments

Changing `def numerical_gradients(..., h=1e-5)` to `def numerical_gradients(..., h)`
made the argument **required**, and the existing four-argument call broke:

```
TypeError: numerical_gradients() missing 1 required positional argument: 'h'
```

`h=1e-5` in a signature makes an argument optional with a fallback; bare `h`
makes it mandatory. Also: a parameter named `h` **shadows** any module-level `h`,
so moving the constant into another file and importing it has no effect inside
the function.

### `zip` stops at the shortest input

`zip` truncates silently and reports nothing. A validation check comparing only
two of three zipped collections leaves a hole — a short third list would quietly
produce a gradient over fewer observations. Chained comparison closes it:

```python
if not (n == len(model_outputs) == len(inputs)):
```

### Module-level prints run on import

`print(...)` at module level fires when the module is imported, not only when the
file is run directly. `if __name__ == "__main__":` exists for this.

---

## Part 4 — Process questions

### Where were those five training steps running?

Not in the project file — they were run externally, in a throwaway snippet that
imported the module and looped around its functions. The project file at that
point performed exactly **one** update.

Done that way deliberately: wrapping the step in a loop is the R4 task. A single
step shows direction but not whether the loss *keeps* falling, so five steps were
needed to confirm the sign fix.

**Takeaway:** every number reported about this code comes from running this code.
If one looks wrong, ask what was run.

### Should there be graphs?

Yes — at **R5**, which is exactly "Visualization and evaluation": the
observations, the fitted line, and the loss history.

R4 comes first because it produces what R5 plots — the loss history list is the
input to the loss curve. There is also a reason to read R4's output as raw
numbers first: divergence in a terminal looks like

```
loss=274.0  loss=415.9  loss=632.8  loss=964.6 ...
```

and recognising that pattern in a column of numbers matters, because at L9 the
training loop emits loss values to a terminal and there isn't always a chart.

**Setup note:** matplotlib is the first third-party dependency; R1–R4 use only
the standard library. Create the virtualenv at R5 and confirm `which python`
points inside `RegLLM/.venv` before installing — bare `pip3` on this machine
resolves into a different project's virtualenv. `requirements.txt` is created at
that point.

### When is a function obsolete?

Three cases came up, with three different answers:

- **`dL_dm` / `dL_dc`** — *delete.* Fully superseded by `gradients()`. Two
  implementations of one formula is how the sign drift happened: one got fixed,
  the other didn't.
- **`dL_dy`** — *delete, or wire it in.* Worth keeping only if something calls
  it. An uncalled function drifts out of sync with the code that replaced it.
- **`update_parameters`** — *keep, change the signature.* The function is the
  gradient-descent step itself; only its list-taking interface was obsolete.

Before deleting a superseded implementation, run it against its replacement
once. An independent second implementation is a genuine test, right up until it
is removed.

---

## Part 5 — Reference numbers

All verified against the project code on the dataset
`x = [1,2,3,4,5]`, `y = [10,13,16,19,22]`, true parameters `m = 3`, `c = 7`.

### Loss and gradients

| `m` | `c` | `L` | `∂L/∂m` | `∂L/∂c` |
| --- | --- | --- | --- | --- |
| `3` | `7` | `0.0` | `0.0` | `0.0` |
| `0` | `0` | `274.0` | `-108.0` | `-32.0` |
| `2` | `5` | `27.0` | `-34.0` | `-10.0` |
| `5` | `9` | `72.0` | `+56.0` | `+16.0` |

At the true parameters everything is zero — the bottom of the bowl, flat in both
directions. **A formula with a sign error also returns zero there**, so it is
useless for verification. Test at wrong parameters.

At `m=5, c=9` — past the true values — both gradients are **positive**. That is
the sign convention's proof.

### Learning rates, from `m=0, c=0`, 1000 iterations

| learning rate | final `m` | final `c` | final loss |
| --- | --- | --- | --- |
| `0.001` | `4.1316` | `2.9144` | `3.04` |
| `0.01` | `3.0537` | `6.8062` | `0.0068` |
| `0.05` | `3.0000` | `7.0000` | `~9e-15` |
| `0.09` | overflow | overflow | `~1.8e108` |

- `0.001` hasn't failed, it hasn't **finished** — loss is still falling. Around
  20,000 iterations gets there.
- `0.05` reaches machine precision; `9e-15` is zero as far as floats go.
- The divergence threshold for this dataset is near **`0.0845`**. The boundary is
  sharp, not gradual.

### Why `c` converges slower than `m`

Inputs `1..5` scale `m`'s gradient and leave `c`'s unscaled, so the two do not
converge at the same speed. The asymmetry is real, and it is why **feature
scaling** exists.
