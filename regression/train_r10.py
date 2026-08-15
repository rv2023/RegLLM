"""R10 - comparison with scikit-learn.

The last regression milestone. Everything the library does here has been built by
hand in R1-R9, so this is about seeing exactly what the API replaced - and why
gradient descent exists at all when a closed-form solution is available.

scikit-learn does NOT do gradient descent for LinearRegression. It solves the
normal equation directly: one shot, no learning rate, no iterations.
"""

import time

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge

SEED        = 42
N           = 50
TRUE_M      = 3.0
TRUE_C      = 7.0
NOISE_SIGMA = 2.0

rng = np.random.default_rng(SEED)

x = rng.uniform(0, 10, N)
y = TRUE_M * x + TRUE_C + rng.normal(0, NOISE_SIGMA, N)

order = rng.permutation(N)
split = int(0.8 * N)
train_idx, test_idx = order[:split], order[split:]

x_train, y_train = x[train_idx], y[train_idx]
x_test,  y_test  = x[test_idx],  y[test_idx]


def predict(x, m, c):
    return m * x + c


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def gradients(x, y_true, y_pred):
    residual = y_pred - y_true
    dm = 2 * np.mean(residual * x)
    dc = 2 * np.mean(residual)
    return dm, dc


def train(x, y_true, m, c, learning_rate, iterations, lam=0.0, kind="none"):
    for _ in range(iterations):
        dm, dc = gradients(x, y_true, predict(x, m, c))
        if kind == "l2":
            dm += 2 * lam * m           # penalty on the slope only, never c
        elif kind == "l1":
            dm += lam * np.sign(m)
        elif kind != "none":
            raise ValueError(f"unknown kind: {kind!r}")
        m -= learning_rate * dm
        c -= learning_rate * dc
    return m, c


def r_squared(y_true, y_pred):
    """What sklearn's .score() returns: 1 - (residual SS / total SS)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


if __name__ == "__main__":
    LEARNING_RATE = 0.01
    ITERATIONS    = 20000

    # ---- 1. ours: gradient descent ----------------------------------------
    t0 = time.perf_counter()
    our_m, our_c = train(x_train, y_train, 0.0, 0.0, LEARNING_RATE, ITERATIONS)
    ours_seconds = time.perf_counter() - t0

    # ---- 2. the normal equation, solved directly --------------------------
    # Build the design matrix: one column of x, one column of ones for the
    # intercept.  lstsq returns the parameters minimising ||X.beta - y||^2.
    X_design = np.column_stack([x_train, np.ones_like(x_train)])
    beta, *_ = np.linalg.lstsq(X_design, y_train, rcond=None)
    exact_m, exact_c = beta

    # ---- 3. scikit-learn ---------------------------------------------------
    # sklearn wants X as (n_samples, n_features), so a column not a flat array.
    X_train = x_train.reshape(-1, 1)
    X_test  = x_test.reshape(-1, 1)

    model = LinearRegression()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    sklearn_seconds = time.perf_counter() - t0

    sk_m = model.coef_[0]        # trailing underscore = learned from data
    sk_c = model.intercept_

    # ---- results -----------------------------------------------------------
    print("parameters")
    print(f"  {'method':22} {'m':>16} {'c':>16}")
    print(f"  {'ours (gradient desc)':22} {our_m:16.10f} {our_c:16.10f}")
    print(f"  {'normal equation':22} {exact_m:16.10f} {exact_c:16.10f}")
    print(f"  {'scikit-learn':22} {sk_m:16.10f} {sk_c:16.10f}")
    print(f"  {'true (generating)':22} {TRUE_M:16.10f} {TRUE_C:16.10f}")
    print()

    print("agreement")
    print(f"  sklearn vs normal equation   dm={abs(sk_m - exact_m):.2e}  dc={abs(sk_c - exact_c):.2e}")
    print(f"  ours    vs normal equation   dm={abs(our_m - exact_m):.2e}  dc={abs(our_c - exact_c):.2e}")
    print()

    # sklearn's answer is exact in one shot. Ours approaches it asymptotically -
    # by 20,000 iterations the gap has closed to floating-point noise, but the
    # approach itself is visible at shorter budgets.
    print("how gradient descent closes the gap")
    print(f"  {'iterations':>10}   {'|m - exact|':>12}   {'|c - exact|':>12}")
    for iters in (20, 200, 2000, 20000):
        gm, gc = train(x_train, y_train, 0.0, 0.0, LEARNING_RATE, iters)
        print(f"  {iters:10}   {abs(gm - exact_m):12.2e}   {abs(gc - exact_c):12.2e}")
    print("  sklearn needs no such table: one solve, exact, no iteration count.")
    print()

    print("error, both models")
    for name, (m, c) in [("ours", (our_m, our_c)), ("sklearn", (sk_m, sk_c))]:
        print(f"  {name:8} train MSE={mse(y_train, predict(x_train, m, c)):.10f}"
              f"   test MSE={mse(y_test, predict(x_test, m, c)):.10f}")
    print()

    print("predict() agrees too")
    ours_pred = predict(x_test, our_m, our_c)
    sk_pred   = model.predict(X_test)
    print(f"  max |ours - sklearn| over the test set = {np.max(np.abs(ours_pred - sk_pred)):.2e}")
    print()

    print("score() is R-squared, which we never computed")
    print(f"  sklearn .score(train)  = {model.score(X_train, y_train):.10f}")
    print(f"  our r_squared(train)   = {r_squared(y_train, predict(x_train, our_m, our_c)):.10f}")
    print(f"  sklearn .score(test)   = {model.score(X_test, y_test):.10f}")
    print()

    print("time")
    print(f"  ours    {ITERATIONS} iterations   {ours_seconds*1000:8.2f} ms")
    print(f"  sklearn one .fit() call      {sklearn_seconds*1000:8.2f} ms")
    print(f"  ratio                        {ours_seconds/sklearn_seconds:8.0f}x")
    print("  (not the same work done faster - different work entirely)")
    print()

    # ---- 4. the regularized pair: Ridge and Lasso -------------------------
    # Our loss uses np.mean; sklearn's uses sums, and the two estimators do not
    # even agree with each other about the scaling.  Working the factors out is
    # the exercise: it forces you to write down exactly what each is minimising.
    #
    #   ours   L2:  mean((y-yhat)^2) + lam*m^2
    #   Ridge:      ||y - Xw||^2     + alpha*||w||^2        -> alpha = n * lam
    #
    #   ours   L1:  mean((y-yhat)^2) + lam*|m|
    #   Lasso:      (1/(2n))*||y-Xw||^2 + alpha*|w|         -> alpha = lam / 2
    #
    REG_ITERS = 200000        # penalised problems converge more slowly
    print("regularized: ours vs Ridge / Lasso")
    print(f"  {'penalty':8} {'lam':>5} {'sklearn alpha':>14}   {'ours m':>12} {'their m':>12}   {'ours c':>12} {'their c':>12}")
    for lam in (0.1, 1.0):
        om, oc = train(x_train, y_train, 0.0, 0.0, LEARNING_RATE, REG_ITERS, lam, "l2")
        model_r = Ridge(alpha=len(x_train) * lam).fit(X_train, y_train)
        print(f"  {'L2':8} {lam:5.1f} {len(x_train)*lam:14.2f}   {om:12.8f} {model_r.coef_[0]:12.8f}"
              f"   {oc:12.8f} {model_r.intercept_:12.8f}")
    for lam in (0.1, 1.0):
        om, oc = train(x_train, y_train, 0.0, 0.0, LEARNING_RATE, REG_ITERS, lam, "l1")
        model_l = Lasso(alpha=lam / 2).fit(X_train, y_train)
        print(f"  {'L1':8} {lam:5.1f} {lam/2:14.2f}   {om:12.8f} {model_l.coef_[0]:12.8f}"
              f"   {oc:12.8f} {model_l.intercept_:12.8f}")
    print("  alpha != lam. Ridge scales by n, Lasso by 1/2 - they disagree with")
    print("  each other, so read the docs rather than assuming a shared convention.")
    print()

    print("what each API call replaced")
    print("  LinearRegression()   our m, c and their initialisation")
    print("  Ridge(alpha=n*lam)   our L2 penalty and its gradient")
    print("  Lasso(alpha=lam/2)   our L1 penalty and its gradient")
    print("  .fit(X, y)           the whole training loop: gradients, optimizer,")
    print("                       learning rate, iteration count, convergence")
    print("  .coef_, .intercept_  m, c")
    print("  .predict(X)          our predict()")
    print("  .score(X, y)         R-squared, which we never wrote")
