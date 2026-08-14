import numpy as np

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


def numerical_gradients(x, y_true, m, c, h=1e-5):
    def loss(mm, cc):
        return mse(y_true, predict(x, mm, cc))

    dm = (loss(m + h, c) - loss(m - h, c)) / (2 * h)
    dc = (loss(m, c + h) - loss(m, c - h)) / (2 * h)
    return dm, dc


def check_gradients(x, y_true, m, c, tolerance=1e-4):
    gm, gc = gradients(x, y_true, predict(x, m, c))
    nm, nc = numerical_gradients(x, y_true, m, c)
    print(f"m={m}, c={c}")
    print(f"  dL/dm  analytical={gm:10.4f}  numerical={nm:10.4f}  ok={abs(gm - nm) < tolerance}")
    print(f"  dL/dc  analytical={gc:10.4f}  numerical={nc:10.4f}  ok={abs(gc - nc) < tolerance}")

if __name__ == "__main__":
    # cross-check against the known five-point answers from R1-R5
    cx = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cy = np.array([10.0, 13.0, 16.0, 19.0, 22.0])
    cp = predict(cx, 0.0, 0.0)
    print("five-point check")
    print("  mse       ", mse(cy, cp), "   expect 274.0")
    print("  gradients ", gradients(cx, cy, cp), "   expect (-108.0, -32.0)")
    print()

    print("shapes:", "x", x.shape, "x_train", x_train.shape, "x_test", x_test.shape)
    print("noise mean:", f"{(y - (TRUE_M*x + TRUE_C)).mean():.4f}", " (should be near 0)")
    print()
    check_gradients(x_train, y_train, 0.0, 0.0)