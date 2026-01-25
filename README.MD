# DCSLM — Darko Cvetkovski Small Language Model

DCSLM is a long-term personal project focused on building a **language model from first principles**.

Instead of starting with high-level frameworks like PyTorch or TensorFlow, this project begins by implementing the **foundational learning machinery** that modern language models rely on internally, and incrementally builds upward toward token-based sequence modeling.

The goal is not speed or scale, but **deep understanding** of how language models actually work under the hood.

---

## Project Philosophy

Most language model implementations start at the top of the stack:
- import an autodiff framework
- define a Transformer
- train on text

DCSLM intentionally does the opposite.

This project builds the stack **bottom-up**, implementing each core component manually in order to understand:
- how gradients propagate
- how optimization behaves
- how numerical stability issues arise
- how learning dynamics emerge from simple rules

This mirrors how early ML frameworks and research systems were developed.

---

## Current State (v0.3)

The project currently implements a **scalar reverse-mode automatic differentiation engine**, including:

- Dynamic computational graphs
- Reverse-mode backpropagation
- Topological graph traversal
- Gradient accumulation
- Operator overloading for clean mathematical syntax
- Numerically stable division (epsilon handling)
- Mean Squared Error (MSE) loss
- Gradient descent training loop
- Successful training of a linear regression model from scratch

All gradients are computed automatically via the graph structure — no symbolic differentiation, no external libraries.

---

## Example (Current)

```python
w = Value(0.0)
x = Value(2.0)
y_true = Value(8.0)

for step in range(50):
    y_pred = w * x
    loss = (y_pred - y_true) ** 2

    loss.backward()
    w.data -= 0.1 * w.grad
    w.grad = 0

    print(step, loss.data, w.data)
```
This loop trains w toward the correct value using only the custom autodiff engine.