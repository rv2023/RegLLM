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
ITERATIONS    = 20000

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
    dm += penalty_gradient(m, lam, kind)
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
    print(f"m={m}, c={c}, lam={lam}, kind={kind}")
    print(f"  dL/dm  analytical={gm:10.4f}  numerical={nm:10.4f}  ok={abs(gm - nm) < tolerance}")
    print(f"  dL/dc  analytical={gc:10.4f}  numerical={nc:10.4f}  ok={abs(gc - nc) < tolerance}")
    
def update_parameters(m, c, dm, dc, learning_rate):
    return m - learning_rate * dm, c - learning_rate * dc


def train(x, y_true, m, c, learning_rate, iterations, lam=0.0, kind="none", print_every=None):
    history = []
    for i in range(iterations):
        y_pred = predict(x, m, c)
        loss   = total_loss(y_true, y_pred, m, lam, kind)
        dm, dc = total_gradients(x, y_true, y_pred, m, lam, kind)

        history.append((loss, m, c))

        if print_every and i % print_every == 0:
            print(f"  iter {i:6}  loss={loss:10.6f}  m={m:9.6f}  c={c:9.6f}")

        m, c = update_parameters(m, c, dm, dc, learning_rate)
        
    return m, c, history

if __name__ == "__main__":
    # cross-check against the known five-point answers from R1-R5
    cx = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cy = np.array([10.0, 13.0, 16.0, 19.0, 22.0])
    cp = predict(cx, 0.0, 0.0)
    print("five-point check")
    print("  mse       ", mse(cy, cp), "   expect 274.0")
    print("  gradients ", gradients(cx, cy, cp), "   expect (-108.0, -32.0)")
    print()

    # the penalty must not disturb anything when it is switched off
    print("lambda=0 baseline (must reproduce R6)")
    base_m, base_c, _ = train(x_train, y_train, 0.0, 0.0, LEARNING_RATE, ITERATIONS)
    print(f"  m={base_m:.4f}  c={base_c:.4f}     expect m=3.0707  c=6.2686")
    print()

    # analytical vs numerical, with the penalty active
    print("gradient checks")
    check_gradients(x_train, y_train, 0.0, 0.0, 0.0, "none")
    check_gradients(x_train, y_train, 2.0, 5.0, 1.0, "l2")
    check_gradients(x_train, y_train, 2.0, 5.0, 1.0, "l1")
    print()

    # sweep
    LAMBDAS = [0.1, 1.0, 10.0]
    results = {}

    print(f"{'penalty':8} {'lambda':>7}   {'m':>9} {'c':>9}   {'train MSE':>10} {'test MSE':>9}")
    for kind, lam in [("none", 0.0)] + [(k, l) for k in ("l2", "l1") for l in LAMBDAS]:
        m, c, _ = train(x_train, y_train, 0.0, 0.0, LEARNING_RATE, ITERATIONS, lam, kind)
        tr = mse(y_train, predict(x_train, m, c))   # evaluate WITHOUT the penalty
        te = mse(y_test,  predict(x_test,  m, c))
        results[(kind, lam)] = (m, c, tr, te)
        print(f"{kind:8} {lam:7.1f}   {m:9.4f} {c:9.4f}   {tr:10.4f} {te:9.4f}")

    # --- how the slope shrinks as the penalty strengthens ------------------
    plt.figure(figsize=(7, 4.5))
    for kind, marker in (("l2", "o"), ("l1", "s")):
        ms = [results[(kind, lam)][0] for lam in LAMBDAS]
        plt.plot(LAMBDAS, ms, marker=marker, label=f"{kind.upper()} penalty")

    plt.axhline(TRUE_M, color="grey", linestyle="--", label=f"true m = {TRUE_M}")
    plt.axhline(results[("none", 0.0)][0], color="crimson", linestyle=":",
                label=f"unregularised m = {results[('none', 0.0)][0]:.4f}")
    plt.xscale("log")
    plt.xlabel("lambda (log scale)")
    plt.ylabel("learned m")
    plt.title("Both penalties shrink the slope toward zero")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("r7_shrinkage.png", dpi=120)
    plt.close()
    print("\nwrote r7_shrinkage.png")