"""R9 - optimizer progression: plain GD, Momentum, RMSprop, Adam.

Self-contained, and deliberately full-batch: R8 already varies batch size, and
varying both at once would make results impossible to attribute.

Each optimizer is a small class with the same interface:

    opt = Adam(lr=0.01)
    m, c = opt.step(m, c, dm, dc)

which is close to torch.optim, so L9 becomes recognition rather than learning.
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

BETA = 0.9        # momentum / RMSprop decay
B1   = 0.9        # Adam first moment
B2   = 0.999      # Adam second moment
EPS  = 1e-8

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


# --------------------------------------------------------------------------
# Optimizers.  Each carries its own state, zeroed in __init__ and persisting
# across every call to step().  state() exists only so the traces can print it.
# --------------------------------------------------------------------------

class PlainGD:
    name = "plain GD"

    def __init__(self, lr=LEARNING_RATE):
        self.lr = lr

    def step(self, m, c, dm, dc):
        return m - self.lr * dm, c - self.lr * dc

    def state(self):
        return {}


class Momentum:
    name = "momentum"

    def __init__(self, lr=LEARNING_RATE, beta=BETA):
        self.lr = lr
        self.beta = beta
        self.v_m = 0.0
        self.v_c = 0.0

    def step(self, m, c, dm, dc):
        self.v_m = self.beta * self.v_m + dm
        self.v_c = self.beta * self.v_c + dc
        return m - self.lr * self.v_m, c - self.lr * self.v_c

    def state(self):
        return {"v_m": self.v_m, "v_c": self.v_c}


class RMSprop:
    name = "RMSprop"

    def __init__(self, lr=LEARNING_RATE, beta=BETA, eps=EPS):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s_m = 0.0
        self.s_c = 0.0

    def step(self, m, c, dm, dc):
        self.s_m = self.beta * self.s_m + (1 - self.beta) * dm ** 2
        self.s_c = self.beta * self.s_c + (1 - self.beta) * dc ** 2
        m -= self.lr * dm / (np.sqrt(self.s_m) + self.eps)
        c -= self.lr * dc / (np.sqrt(self.s_c) + self.eps)
        return m, c

    def state(self):
        return {"s_m": self.s_m, "s_c": self.s_c}


class Adam:
    name = "Adam"

    def __init__(self, lr=LEARNING_RATE, b1=B1, b2=B2, eps=EPS):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.v_m = 0.0
        self.v_c = 0.0
        self.s_m = 0.0
        self.s_c = 0.0
        self.t = 0            # counts UPDATES, never reset

    def step(self, m, c, dm, dc):
        self.t += 1           # before the bias correction: 1 - b1**0 would be 0

        self.v_m = self.b1 * self.v_m + (1 - self.b1) * dm
        self.v_c = self.b1 * self.v_c + (1 - self.b1) * dc
        self.s_m = self.b2 * self.s_m + (1 - self.b2) * dm ** 2
        self.s_c = self.b2 * self.s_c + (1 - self.b2) * dc ** 2

        vm_hat = self.v_m / (1 - self.b1 ** self.t)
        vc_hat = self.v_c / (1 - self.b1 ** self.t)
        sm_hat = self.s_m / (1 - self.b2 ** self.t)
        sc_hat = self.s_c / (1 - self.b2 ** self.t)

        m -= self.lr * vm_hat / (np.sqrt(sm_hat) + self.eps)
        c -= self.lr * vc_hat / (np.sqrt(sc_hat) + self.eps)
        return m, c

    def state(self):
        return {"v_m": self.v_m, "s_m": self.s_m, "t": self.t}


def train(x, y_true, m, c, optimizer, iterations):
    """Full-batch training. The optimizer must be FRESH - reusing one carries
    its state into the next run and produces numbers you cannot explain."""
    history = []
    for _ in range(iterations):
        y_pred = predict(x, m, c)
        history.append(mse(y_true, y_pred))
        dm, dc = gradients(x, y_true, y_pred)
        m, c = optimizer.step(m, c, dm, dc)
    return m, c, history


def trace(make_optimizer, steps=8):
    """Print the optimizer's internal state for the first few updates."""
    opt = make_optimizer()
    m = c = 0.0
    print(f"  {opt.name}")
    for _ in range(steps):
        dm, dc = gradients(x_train, y_train, predict(x_train, m, c))
        before = m
        m, c = opt.step(m, c, dm, dc)
        bits = "  ".join(f"{k}={v:11.4f}" for k, v in opt.state().items())
        print(f"    g_m={dm:10.4f}  {bits}  step={m - before:8.4f}  m={m:8.4f}")
    print()


def sweep(make_optimizers, iteration_counts):
    print(f"  {'optimizer':11} {'iters':>6}   {'m':>8} {'c':>8}  {'train MSE':>10}")
    for iters in iteration_counts:
        for make in make_optimizers:
            m, c, _ = train(x_train, y_train, 0.0, 0.0, make(), iters)
            L = mse(y_train, predict(x_train, m, c))
            print(f"  {make().name:11} {iters:6}   {m:8.4f} {c:8.4f}  {L:10.4f}")
        print()


if __name__ == "__main__":
    MAKERS = [PlainGD, Momentum, RMSprop, Adam]

    # ---- 1. the five-point reference from R1-R5 ---------------------------
    cx = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cy = np.array([10.0, 13.0, 16.0, 19.0, 22.0])
    cp = predict(cx, 0.0, 0.0)
    print("five-point check")
    print("  mse       ", mse(cy, cp), "   expect 274.0")
    print("  gradients ", gradients(cx, cy, cp), "   expect (-108.0, -32.0)")
    print()

    # ---- 2. the class wrapper must not change plain GD --------------------
    print("plain GD baseline (must reproduce R6)")
    bm, bc, _ = train(x_train, y_train, 0.0, 0.0, PlainGD(), 20000)
    print(f"  m={bm:.4f}  c={bc:.4f}     expect m=3.0707  c=6.2686")
    print()

    # ---- 3. degenerate cases: the cheapest correctness checks -------------
    print("degenerate checks")
    gm, gc, _ = train(x_train, y_train, 0.0, 0.0, PlainGD(), 2000)
    zm, zc, _ = train(x_train, y_train, 0.0, 0.0, Momentum(beta=0.0), 2000)
    print(f"  momentum beta=0 vs plain GD:  m {zm:.10f} vs {gm:.10f}  same={np.isclose(zm, gm, atol=1e-12)}")
    am, ac, _ = train(x_train, y_train, 0.0, 0.0, Adam(b1=0.0, b2=0.0), 200)
    print(f"  Adam b1=b2=0 (pure sign step): m={am:.4f}  c={ac:.4f}   expect ~lr*200 = 2.0")
    print()

    # ---- 4. Adam's bias correction at t=1 ---------------------------------
    print("Adam bias correction at t=1")
    a = Adam()
    dm0, dc0 = gradients(x_train, y_train, predict(x_train, 0.0, 0.0))
    a.step(0.0, 0.0, dm0, dc0)
    print(f"  g_m   = {dm0:12.4f}")
    print(f"  v_m   = {a.v_m:12.4f}   (raw, biased toward zero)")
    print(f"  v_hat = {a.v_m / (1 - a.b1):12.4f}   must equal g_m")
    print()

    # ---- 5. the comparison sweep ------------------------------------------
    print("optimizer comparison")
    sweep(MAKERS, [200, 2000, 20000])

    # ---- 6. what the state actually does ----------------------------------
    print("state traces, first 8 updates")
    for make in MAKERS:
        trace(make)

    # ---- 7. both parameters swing in phase --------------------------------
    print("momentum gradient signs (g_m and g_c flip together)")
    opt = Momentum()
    m = c = 0.0
    print(f"    {'t':>2}  {'g_m':>10} {'sign':>5}  |  {'g_c':>9} {'sign':>5}")
    for t in range(1, 11):
        dm, dc = gradients(x_train, y_train, predict(x_train, m, c))
        m, c = opt.step(m, c, dm, dc)
        print(f"    {t:2}  {dm:10.2f} {'+' if dm > 0 else '-':>5}  |  {dc:9.2f} {'+' if dc > 0 else '-':>5}")
    print()

    # ---- 8. Adam was not worse, it was mismatched -------------------------
    print("Adam at a learning rate chosen for Adam")
    for lr in (0.01, 0.1):
        m, c, _ = train(x_train, y_train, 0.0, 0.0, Adam(lr=lr), 200)
        print(f"  Adam lr={lr:<5} 200 iters   m={m:8.4f} c={c:8.4f}  train MSE={mse(y_train, predict(x_train, m, c)):9.4f}")
    print()

    # ---- plots -------------------------------------------------------------
    plt.figure(figsize=(7.5, 4.5))
    for make in MAKERS:
        _, _, history = train(x_train, y_train, 0.0, 0.0, make(), 2000)
        plt.plot(history, label=make().name)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("train MSE (log scale)")
    plt.title("Optimizers at lr=0.01, full batch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("r9_optimizers.png", dpi=120)
    plt.close()

    # the parameter-space path: momentum's overshoot is worth seeing once
    plt.figure(figsize=(6.5, 5))
    for make in MAKERS:
        opt = make()
        m = c = 0.0
        path = [(m, c)]
        for _ in range(300):
            dm, dc = gradients(x_train, y_train, predict(x_train, m, c))
            m, c = opt.step(m, c, dm, dc)
            path.append((m, c))
        plt.plot([p[0] for p in path], [p[1] for p in path], label=make().name)
    plt.scatter([TRUE_M], [TRUE_C], marker="*", s=150, color="black", zorder=5, label="true (3, 7)")
    plt.xlabel("m")
    plt.ylabel("c")
    plt.title("Path through parameter space, 300 iterations")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("r9_paths.png", dpi=120)
    plt.close()

    print("wrote r9_optimizers.png, r9_paths.png")
