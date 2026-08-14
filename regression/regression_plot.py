import matplotlib
matplotlib.use("Agg")               # write files instead of opening a window
import matplotlib.pyplot as plt

from regression_inputs import inputs, m, c, interations as iterations
from regression_from_scratch import predict
from regression_loop import train, actual_outputs

RATES = [0.001, 0.01, 0.05, 0.09]

results = {}
for lr in RATES:
    final_m, final_c, history = train(
        inputs, actual_outputs, m, c, lr, iterations, print_every=iterations
    )
    results[lr] = (final_m, final_c, history)

# --- Plot 1: observations and the fitted line ------------------------------
best_m, best_c, _ = results[0.05]
fitted = predict(inputs, best_m, best_c)

plt.figure(figsize=(6, 4))
plt.scatter(inputs, actual_outputs, s=60, zorder=3, label="observations")
plt.plot(inputs, fitted, label=f"fitted:  y = {best_m:.4f}x + {best_c:.4f}")
plt.xlabel("advertising spend")
plt.ylabel("sales")
plt.title("Fitted line (learning rate 0.05)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_fit.png", dpi=120)
plt.close()

# --- Plot 2: one loss curve, linear axis vs log axis -----------------------
losses = [entry[0] for entry in results[0.01][2]]

fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 4))
ax_lin.plot(losses)
ax_lin.set_title("linear y-axis")
ax_lin.set_xlabel("iteration"); ax_lin.set_ylabel("loss")
ax_lin.grid(alpha=0.3)

ax_log.plot(losses)
ax_log.set_yscale("log")
ax_log.set_title("log y-axis")
ax_log.set_xlabel("iteration"); ax_log.set_ylabel("loss")
ax_log.grid(alpha=0.3)

fig.suptitle("Same data, two axes (learning rate 0.01)")
fig.tight_layout()
fig.savefig("plot_loss_axes.png", dpi=120)
plt.close(fig)

# --- Plot 3: all four learning rates on one figure -------------------------
plt.figure(figsize=(7, 4.5))
for lr in RATES:
    losses = [entry[0] for entry in results[lr][2]]
    plt.plot(losses, label=f"lr = {lr}")

plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("loss (log scale)")
plt.title("Learning rate comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_learning_rates.png", dpi=120)
plt.close()

print("wrote plot_fit.png, plot_loss_axes.png, plot_learning_rates.png")

# --- Plot 4: the three converging rates, on a readable axis ----------------
plt.figure(figsize=(7, 4.5))
for lr in [0.001, 0.01, 0.05]:
    losses = [entry[0] for entry in results[lr][2]]
    plt.plot(losses, label=f"lr = {lr}")

plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("loss (log scale)")
plt.title("Converging learning rates (0.09 excluded)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_learning_rates_converging.png", dpi=120)
plt.close()