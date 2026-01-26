from value import Value

x = Value(2.0)
weight = Value(0.0)
bias = Value(0.0)

target_y = Value(1.0)

lr = 0.1
for step in range(100):
    out = (x*weight + bias).tanh()
    out_loss = Value.mse(out, target_y)

    out_loss.backward()

    weight.data -= lr * weight.grad
    bias.data -= lr * bias.grad

    weight.grad = 0
    bias.grad = 0
    x.grad = 0

    if step % 10 == 0:
        print(step, out_loss.data, out.data)