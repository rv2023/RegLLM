from regression_inputs import inputs, m, c, interations as iterations
from regression_from_scratch import predict, MSE, gradients, update_parameters

actual_outputs = [10, 13, 16, 19, 22]


def train(inputs, actual_outputs, m, c, learning_rate, iterations, print_every=100):
    history = []
    for i in range(iterations):
        predictions = predict(inputs, m, c)
        loss        = MSE(actual_outputs, predictions)
        dm, dc      = gradients(actual_outputs, predictions, inputs)

        history.append((loss, m, c))

        if i % print_every == 0:
            print(f"  iter {i:4}  loss={loss:12.6g}  m={m:12.6g}  c={c:12.6g}")

        m, c = update_parameters(m, c, dm, dc, learning_rate)

    return m, c, history
if __name__ == "__main__":
    results = {}

    for lr in [0.001, 0.01, 0.05, 0.09]:
        print(f"learning_rate = {lr}")
        final_m, final_c, history = train(inputs, actual_outputs, m, c, lr, iterations)
        final_loss = MSE(actual_outputs, predict(inputs, final_m, final_c))
        results[lr] = (final_m, final_c, final_loss, history)
        print(f"  final     loss={final_loss:12.6g}  m={final_m:12.6g}  c={final_c:12.6g}\n")

    print("summary")
    for lr, (final_m, final_c, final_loss, history) in results.items():
        print(f"  lr={lr:<6}  m={final_m:12.6g}  c={final_c:12.6g}  loss={final_loss:12.6g}")
