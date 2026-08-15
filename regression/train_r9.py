"""R8 - SGD and mini-batch training, carrying R7's L1/L2 penalties forward.

Self-contained: generates its own data with the same seed and call order as
training.py, so `batch_size = len(x_train)` reproduces the R6/R7 results exactly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED          = 42
N             = 50
TRUE_M        = 3.0
TRUE_C        = 7.0
NOISE_SIGMA   = 2.0
LEARNING_RATE = 0.01

EPOCHS_SHORT  = 200        # endpoints still differ - shows mini-batch ahead
EPOCHS_LONG   = 20000      # everyone has converged - shows the advantage is speed
SHUFFLE_SEED  = 0          # separate from SEED so the data never moves

rng = np.random.default_rng(SEED)

x = rng.uniform(0, 10, N)
y = TRUE_M * x + TRUE_C + rng.normal(0, NOISE_SIGMA, N)

order = rng.permutation(N)
split = int(0.8 * N)
train_idx, test_idx = order[:split], order[split:]

x_train, y_train = x[train_idx], y[train_idx]
x_test,  y_test  = x[test_idx],  y[test_idx]

FULL = len(x_train)        # 40 - "full batch" means the whole training set


def predict(x, m, c):
    return m * x + c


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def gradients(x, y_true, y_pred):
    residual = y_pred - y_true
    dm = 2 * np.mean(residual * x)
    dc = 2 * np.mean(residual)
    return dm, dc


def penalty(m, lam, kind):
    if kind == "none":
        return 0
    elif kind == "l1":
        return lam * np.abs(m)
    elif kind == "l2":
        return lam * m ** 2
    else:
        raise ValueError(f"unknown kind: {kind!r} (expected 'none', 'l1' or 'l2')")


def penalty_gradient(m, lam, kind):
    if kind == "none":
        return 0
    elif kind == "l1":
        return lam * np.sign(m)
    elif kind == "l2":
        return 2 * lam * m
    else:
        raise ValueError(f"unknown kind: {kind!r} (expected 'none', 'l1' or 'l2')")


def total_loss(y_true, y_pred, m, lam, kind):
    return mse(y_true, y_pred) + penalty(m, lam, kind)


def total_gradients(x, y_true, y_pred, m, lam, kind):
    dm, dc = gradients(x, y_true, y_pred)
    dm += penalty_gradient(m, lam, kind)      # penalty applies to m only, never c
    return dm, dc


def numerical_gradients(x, y_true, m, c, lam, kind, h=1e-5):
    def loss(mm, cc):
        return total_loss(y_true, predict(x, mm, cc), mm, lam, kind)

    dm = (loss(m + h, c) - loss(m - h, c)) / (2 * h)
    dc = (loss(m, c + h) - loss(m, c - h)) / (2 * h)
    return dm, dc


def check_gradients(x, y_true, m, c, lam, kind, tolerance=1e-4):
    gm, gc = total_gradients(x, y_true, predict(x, m, c), m, lam, kind)
    nm, nc = numerical_gradients(x, y_true, m, c, lam, kind)
    print(f"  m={m}, c={c}, lam={lam}, kind={kind}")
    print(f"    dL/dm  analytical={gm:10.4f}  numerical={nm:10.4f}  ok={abs(gm - nm) < tolerance}")
    print(f"    dL/dc  analytical={gc:10.4f}  numerical={nc:10.4f}  ok={abs(gc - nc) < tolerance}")


def update_parameters(m, c, dm, dc, learning_rate):
    return m - learning_rate * dm, c - learning_rate * dc


def train(x, y_true, m, c, learning_rate, epochs, batch_size,
          lam=0.0, kind="none", seed=SHUFFLE_SEED):
    """Mini-batch gradient descent.

    batch_size == len(x) is full-batch; batch_size == 1 is strict SGD.
    History holds one entry per EPOCH, measured over the whole training set,
    so runs with different batch sizes share an x-axis.
    """
    rng = np.random.default_rng(seed)          # own generator; data never moves
    n = len(x)
    history = []
    updates = 0

    for _ in range(epochs):
        order = rng.permutation(n)             # reshuffled every epoch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]   # short final batch is fine
            xb, yb = x[idx], y_true[idx]

            dm, dc = total_gradients(xb, yb, predict(xb, m, c), m, lam, kind)
            m, c = update_parameters(m, c, dm, dc, learning_rate)
            updates += 1

        history.append(mse(y_true, predict(x, m, c)))   # full set, not the batch

    return m, c, history, updates


def label(batch_size):
    if batch_size == FULL:
        return "full-batch"
    return "SGD" if batch_size == 1 else "mini-batch"


def sweep(batch_sizes, epochs, learning_rate=LEARNING_RATE, lam=0.0, kind="none"):
    results = {}
    header = f"{'mode':11} {'batch':>5} {'updates':>8}   {'m':>9} {'c':>9}  {'train MSE':>10} {'test MSE':>9}"
    print(header)
    for bs in batch_sizes:
        m, c, history, updates = train(x_train, y_train, 0.0, 0.0,
                                       learning_rate, epochs, bs, lam, kind)
        tr = mse(y_train, predict(x_train, m, c))    # evaluate WITHOUT the penalty
        te = mse(y_test,  predict(x_test,  m, c))
        results[bs] = (m, c, history, updates, tr, te)
        print(f"{label(bs):11} {bs:5} {updates:8}   {m:9.4f} {c:9.4f}  {tr:10.4f} {te:9.4f}")
    return results


if __name__ == "__main__":
    BATCH_SIZES = [FULL, 8, 6, 1]

    # ---- 1. the five-point reference from R1-R5 ---------------------------
    cx = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cy = np.array([10.0, 13.0, 16.0, 19.0, 22.0])
    cp = predict(cx, 0.0, 0.0)
    print("five-point check")
    print("  mse       ", mse(cy, cp), "   expect 274.0")
    print("  gradients ", gradients(cx, cy, cp), "   expect (-108.0, -32.0)")
    print()

    # ---- 2. full-batch must reproduce R6 ----------------------------------
    print("full-batch baseline (must reproduce R6)")
    bm, bc, _, bu = train(x_train, y_train, 0.0, 0.0, LEARNING_RATE, EPOCHS_LONG, FULL)
    print(f"  m={bm:.4f}  c={bc:.4f}  updates={bu}     expect m=3.0707  c=6.2686")
    print()

    # ---- 3. gradients still check out, penalty included -------------------
    print("gradient checks")
    check_gradients(x_train, y_train, 0.0, 0.0, 0.0, "none")
    check_gradients(x_train, y_train, 2.0, 5.0, 1.0, "l2")
    check_gradients(x_train, y_train, 2.0, 5.0, 1.0, "l1")
    print()

    # ---- 4. batch size, short budget: endpoints differ --------------------
    print(f"batch-size sweep, {EPOCHS_SHORT} epochs (equal data seen)")
    short = sweep(BATCH_SIZES, EPOCHS_SHORT)
    print()

    # ---- 5. batch size, long budget: everyone converges but SGD -----------
    print(f"batch-size sweep, {EPOCHS_LONG} epochs")
    long = sweep(BATCH_SIZES, EPOCHS_LONG)
    print()

    # ---- 6. SGD is fine once the learning rate matches the batch ----------
    print("SGD at a smaller learning rate")
    sweep([1], EPOCHS_SHORT, learning_rate=0.001)
    print()

    # ---- 7. the penalty across batch sizes --------------------------------
    # The equilibrium sits where the data gradient and the penalty gradient
    # cancel, which does not depend on how often you step - so batch size
    # changes how fast that point is reached, not where it is.
    print(f"L2 lambda=0.1, {EPOCHS_SHORT} epochs - same lambda, different batch sizes")
    sweep(BATCH_SIZES, EPOCHS_SHORT, lam=0.1, kind="l2")
    print()
    print(f"L1 lambda=0.1, {EPOCHS_SHORT} epochs")
    sweep(BATCH_SIZES, EPOCHS_SHORT, lam=0.1, kind="l1")
    print()

    # ---- plots ------------------------------------------------------------
    plt.figure(figsize=(7.5, 4.5))
    for bs in BATCH_SIZES:
        plt.plot(short[bs][2], label=f"batch {bs} ({label(bs)})")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("train MSE (log scale)")
    plt.title(f"Loss per epoch by batch size ({EPOCHS_SHORT} epochs, equal data seen)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("r8_by_epoch.png", dpi=120)
    plt.close()

    # the same curves against cumulative updates - the ranking flips
    plt.figure(figsize=(7.5, 4.5))
    for bs in BATCH_SIZES:
        history = short[bs][2]
        per_epoch = short[bs][3] // EPOCHS_SHORT
        updates_axis = [(e + 1) * per_epoch for e in range(len(history))]
        plt.plot(updates_axis, history, label=f"batch {bs} ({label(bs)})")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("cumulative updates (log scale)")
    plt.ylabel("train MSE (log scale)")
    plt.title("The same runs, counted by update instead of by epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("r8_by_update.png", dpi=120)
    plt.close()

    print("wrote r8_by_epoch.png, r8_by_update.png")
