from regression_inputs import inputs, m, c, h, tolerance, learning_rate
actual_outputs = [10, 13, 16, 19, 22]

def predict(inputs, m, c):
    model_outputs = []
    for j in inputs:
        y = m * j + c
        model_outputs.append(y)
    
    return model_outputs

def MSE (actual_outputs, model_outputs):
    n = len(actual_outputs)
    n2 = len(model_outputs)
    
    if n != n2:
        raise ValueError("The lengths of actual_outputs and model_outputs must be the same.", n, n2)
    
    # L = ((y - y`)^2)
    mse = sum((predicted - actual) ** 2 for actual, predicted in zip(actual_outputs, model_outputs)) / n
    return mse


# print("MSE", MSE(actual_outputs, linear_regression(inputs, m, c)))

## G Rules #######
# dL/dy` = -2(y - y`)
def dL_dy(actual_outputs, model_outputs):
    n = len(actual_outputs)
    n2 = len(model_outputs)
    
    if n != n2:
        raise ValueError("The lengths of actual_outputs and model_outputs must be the same.", n, n2)
    
    dL_dy_values = []
    for actual, predicted in zip(actual_outputs, model_outputs):
        dL_dy = -2 * (actual - predicted)
        dL_dy_values.append(dL_dy)
    
    return dL_dy_values

# dL/dm = (dL/dy`)*(dy`/dm) = -2(y - y`) * x
# dL/dc = (dL/dy`)*(dy`/dc) = -2(y - y`) * 1

def gradients(actual_outputs, model_outputs, inputs):
    n = len(actual_outputs)
    if not (n == len(model_outputs) == len(inputs)):
        raise ValueError(f"length mismatch: {n}, {len(model_outputs)}, {len(inputs)}")

    total_m = 0.0
    total_c = 0.0
    for actual, predicted, x in zip(actual_outputs, model_outputs, inputs):
        residual = predicted - actual      # convention, fixed once, used twice
        total_m += residual * x            # dŷ/dm = x
        total_c += residual                # dŷ/dc = 1
    return (2 / n) * total_m, (2 / n) * total_c


def update_parameters(m, c, gm, gc, learning_rate):
    m -= learning_rate * gm
    c -= learning_rate * gc
    return m, c

def numerical_gradients(actual_outputs, inputs, m, c, h):
    def loss(mm, cc):
        return MSE(actual_outputs, predict(inputs, mm, cc))

    dm = (loss(m + h, c) - loss(m - h, c)) / (2 * h)
    dc = (loss(m, c + h) - loss(m, c - h)) / (2 * h)
    return dm, dc


def check_gradients(actual_outputs, inputs, m, c, tolerance):
    gm, gc = gradients(actual_outputs, predict(inputs, m, c), inputs)
    nm, nc = numerical_gradients(actual_outputs, inputs, m, c, h)

    print(f"m={m}, c={c}, h={h}")
    print(f"  dL/dm  analytical={gm:9.4f}  numerical={nm:9.4f}  ok={abs(gm - nm) < tolerance}")
    print(f"  dL/dc  analytical={gc:9.4f}  numerical={nc:9.4f}  ok={abs(gc - nc) < tolerance}")

if __name__ == "__main__":
    print("Gradiant validation")
    check_gradients(actual_outputs, inputs, 0, 0, tolerance)
    check_gradients(actual_outputs, inputs, 2, 5, tolerance)

    print("Gradiant run")
    gm, gc = gradients(actual_outputs, predict(inputs, m, c), inputs)
    nm, nc = numerical_gradients(actual_outputs, inputs, m, c, h)
    print(f"m={m}, c={c}, h={h}")
    print(f"  dL/dm  analytical={gm:9.4f}  numerical={nm:9.4f}  ok={abs(gm - nm) < tolerance}")
    print(f"  dL/dc  analytical={gc:9.4f}  numerical={nc:9.4f}  ok={abs(gc - nc) < tolerance}")

    print("dL/dy`", dL_dy(actual_outputs, predict(inputs, m, c)))
    print("dL/dm,", "dL/dc", gradients(actual_outputs, predict(inputs, m, c), inputs))
    updated_m, updated_c = update_parameters(m, c, gm, gc, learning_rate)
    print("Updated parameters", updated_m, updated_c)