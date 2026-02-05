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

## Current State (v0.1.0)

The project has moved beyond basic MLPs and is now implementing the architectural guts of a Small Language Model. All operations are grounded in a custom scalar reverse-mode automatic differentiation engine.

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
- Multi-neuron layers
- MLP abstraction

Fully connected Layer abstraction where there is support for dimensionality transformation (N inputs, M outputs), activations are created. And the layer abstraction returns a flattened list of parameters provided by each neuron within the layer. 

Output squeezing is also implemented where the final output is returned as a pure value instead of a list. 

All gradients are computed automatically via the graph structure — no symbolic differentiation, no external libraries.


### Data & Tokenization

- Character-Level Tokenizer: A custom encoder/decoder class that maps the "alphabet" of a dataset into a learnable format.
- Vocabulary Mapping (stoi / itos): Bi-directional lookup tables (string-to-integer and integer-to-string) generated dynamically from training data.
- Dataset Integration: Successfully integrated a 4.8MB subset of the TinyStories dataset, allowing for character-based sequence modeling.

### Sequence Modeling Foundations

- Token Embedding Table: A trainable lookup matrix where character identities are mapped to high-dimensional coordinates (D-dimensions). This allows the model to learn semantic similarity between characters.
- Positional Embedding Table: A learnable "address" system that assigns unique vectors to every index in the sequence (up to max_len), solving the permutation-invariance problem.
- Vector Fusion: Implementation of element-wise addition (Char + Pos) to create a unified representation of a character's identity and its location in time.

### Causal Self-Attention
- Q/K/V Projection: Triple-head linear transformation using the custom Layer abstraction.
- Scaled Dot-Product Attention: Manual implementation of the attention formula: $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
- Causal Masking: A strict "look-ahead" constraint implemented via nested loops, ensuring tokens only attend to the past and present.
- Micro-Softmax: A scalar-based implementation of the softmax function that preserves gradient flow through the exponentiation and normalization phases.
