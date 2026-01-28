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

## Current State (v0.5)

The project currently implements a **scalar reverse-mode automatic differentiation engine** and the first neural network primitives, including:

### Autodiff Core
- Dynamic computational graphs
- Reverse-mode backpropagation
- Topological graph traversal
- Gradient accumulation
- Operator overloading for clean mathematical syntax
- Numerically stable division (epsilon handling)

### Learning Components
- Mean Squared Error (MSE) loss
- Gradient descent training loop
- Successful training of a linear regression model from scratch

### Neural Network Primitives
- A fully differentiable `Neuron` abstraction
- Trainable weights and bias represented as autodiff nodes
- Weighted sum + optional `tanh` nonlinearity
- Forward passes that dynamically construct computation graphs
- Automatic gradient propagation through neuron parameters

All gradients are computed automatically via the graph structure — no symbolic differentiation, no external libraries.

---

## Planned Next Steps

- Multi-neuron layers
- Multi-layer perceptron (MLP)
- Nonlinear function approximation (e.g. XOR)
- Extension toward vectorized operations
- Sequence modeling foundations

Each step will be implemented manually and validated before moving upward in abstraction.
