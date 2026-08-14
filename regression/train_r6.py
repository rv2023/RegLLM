import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

from training import (x, y, x_train, y_train, x_test, y_test,
                      predict, mse, gradients,
                      TRUE_M, TRUE_C, NOISE_SIGMA)

LEARNING_RATE = 0.01
ITERATIONS    = 20000


def update_parameters(m, c, dm, dc, learning_rate):
    return m - learning_rate * dm, c - learning_rate * dc


def train(x, y_true, m, c, learning_rate, iterations, print_every=None):
    history = []
    for i in range(iterations):
        y_pred = predict(x, m, c)
        loss   = mse(y_true, y_pred)
        dm, dc = gradients(x, y_true, y_pred)

        history.append((loss, m, c))

        if print_every and i % print_every == 0:
            print(f"  iter {i:6}  loss={loss:10.6f}  m={m:9.6f}  c={c:9.6f}")

        m, c = update_parameters(m, c, dm, dc, learning_rate)

    return m, c, history

final_m, final_c, history = train(
    x_train, y_train, 0.0, 0.0, LEARNING_RATE, ITERATIONS, print_every=4000
)

train_mse = mse(y_train, predict(x_train, final_m, final_c))
test_mse  = mse(y_test,  predict(x_test,  final_m, final_c))

print()
print(f"learned     m={final_m:.4f}  c={final_c:.4f}      (true {TRUE_M}, {TRUE_C})")
print(f"train MSE   {train_mse:.4f}")
print(f"test  MSE   {test_mse:.4f}")
print(f"noise floor {NOISE_SIGMA ** 2:.4f}   (sigma^2)")

# --- fitted line over the noisy data ---------------------------------------
edges = [x.min(), x.max()]
plt.figure(figsize=(6.5, 4.5))
plt.scatter(x_train, y_train, s=28, alpha=0.8, label="train")
plt.scatter(x_test,  y_test,  s=28, alpha=0.8, marker="s", label="test")
plt.plot(edges, [predict(e, TRUE_M, TRUE_C) for e in edges],
         "--", color="grey", label=f"true:   y = {TRUE_M}x + {TRUE_C}")
plt.plot(edges, [predict(e, final_m, final_c) for e in edges],
         color="crimson", label=f"fitted: y = {final_m:.3f}x + {final_c:.3f}")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Noisy data and fitted line")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("r6_fit.png", dpi=120); plt.close()

# --- loss curve with the noise floor marked --------------------------------
losses = [entry[0] for entry in history]
plt.figure(figsize=(7, 4.5))
plt.plot(losses, label="training loss")
plt.axhline(NOISE_SIGMA ** 2, color="grey", linestyle="--",
            label=f"noise variance = {NOISE_SIGMA ** 2:.0f}")
plt.axhline(train_mse, color="crimson", linestyle=":",
            label=f"final train MSE = {train_mse:.3f}")
plt.yscale("log"); plt.xlabel("iteration"); plt.ylabel("loss (log scale)")
plt.title("Loss stops at a floor, not at zero")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("r6_loss.png", dpi=120); plt.close()