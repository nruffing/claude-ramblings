# How Neural Networks Work: Build One, Train It, Understand Why

*A from-scratch walkthrough of the feedforward neural network and the algorithm that trains it. Written for an experienced engineer who wants to understand the machine well enough to build one — and to see exactly what a framework like PyTorch is doing for you.*

---

## How to read this

This is the prequel to the companion document *How LLMs Work*. That one deliberately skipped the single most important idea in deep learning — **how the weights are learned** — and treated every matrix as given. This document is about exactly that. A transformer is just one particular differentiable function; everything here is how *any* such function gets trained.

The spine of the whole thing is one idea: **a neural network is a differentiable function built by composition, and "learning" is following gradients downhill.** The mechanism that makes that tractable — **backpropagation** — is just the chain rule applied systematically over a computational graph. Internalize the graph-and-local-gradients picture and the rest is detail. Backprop is the centerpiece here the way attention was the centerpiece there; spend your attention on section 6.

I assume you're comfortable with partial derivatives and the chain rule, matrix multiplication, basic probability, and reading Python/NumPy. I will not re-teach those. Code is in NumPy (for the from-scratch parts) and PyTorch (to show the real-world idiom), and it's meant to be read and run.

Rough budget: ~1 hour.

---

## 0. The one-sentence model

A neural network is a parameterized function:

```
f_θ(x) -> ŷ
```

It maps an input `x` to a prediction `ŷ`, and its behavior is controlled by a big bag of parameters `θ` (the weights and biases). "Learning" is the search for the `θ` that makes `f` fit your data well. That's it.

We make that search concrete in four moves, and the entire field is variations on these:

1. **Define the function** `f_θ` (the architecture — layers, nonlinearities).
2. **Define a loss** `L(θ)` — a single number measuring how wrong `f_θ` is on the data.
3. **Compute the gradient** `∇_θ L` — which way is uphill, for every parameter at once. *(This is backprop.)*
4. **Step downhill**: `θ ← θ − η ∇_θ L`, and repeat. *(This is gradient descent.)*

The data-flow, which we'll fill in piece by piece:

```
x ──► [ layer 1 ] ──► [ layer 2 ] ──► ... ──► ŷ ──► L (a scalar)
                                                      │
        ◄──────────  gradients flow backward  ───────┘
        (∂L/∂θ for every weight, in ONE pass: backprop)
                                                      │
                              θ ← θ − η·∇θL  (update)
```

Notation: `x` input, `y` true target, `ŷ` prediction, `W`/`b` a layer's weight matrix and bias, `z` a pre-activation (`Wx + b`), `a` a post-activation (`σ(z)`), `σ` an activation function, `η` the learning rate, `L` the loss. Superscripts `^(l)` index layers.

---

## 1. The neuron and the layer

The atomic unit is the **artificial neuron**: take an input vector, form a weighted sum, add a bias, and pass the result through a nonlinear function.

```
a = σ(w · x + b)
```

The loose biological story — a cell that "fires" when its weighted inputs exceed a threshold — is where the name comes from, and it's worth exactly one sentence. Drop it now. An artificial neuron is a dot product followed by a squashing function, nothing more, and reading too much biology into it will only mislead you.

A **layer** is just many neurons computed in parallel, which is one matrix multiply:

```
h = σ(W x + b)
```

If the layer has `H` neurons and the input has `D` features, then `W` is `(H, D)`, `b` is `(H,)`, and `h` is `(H,)`. Stacking these layers — feeding each layer's output as the next layer's input — gives the **multi-layer perceptron (MLP)**, also called a fully-connected or dense network. It is the canonical neural network, and (perhaps surprisingly) it's exactly the feed-forward block inside a transformer.

In code, a two-layer MLP forward pass is almost nothing (batched, so `X` is `(N, D)` for `N` examples at once):

```python
import numpy as np

def mlp_forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1          # (N, H)   linear layer
    a1 = np.maximum(0, z1)    # (N, H)   ReLU nonlinearity
    z2 = a1 @ W2 + b2         # (N, C)   linear layer -> C outputs
    return z2
```

Two layers, two matmuls, one nonlinearity in between. The depth and width of this stack are the network's main capacity dials.

---

## 2. Why nonlinearity is the whole game

Here is the single most important conceptual point about why neural networks work, and it's a one-liner. Suppose you drop the activation functions and stack two *linear* layers:

```
h = W₂ (W₁ x) = (W₂ W₁) x = W x
```

The composition of linear maps is just another linear map. Stack a thousand linear layers and you still have a single linear function — capacity zero beyond a straight-line fit. **Depth buys you nothing without a nonlinearity between the layers.** The activation function is not a minor implementation detail; it is the reason deep networks can represent anything interesting. It lets the network bend and fold input space into the curved decision boundaries that real data needs.

The common activation functions:

- **Sigmoid** `σ(z) = 1/(1+e⁻ᶻ)` and **tanh**: smooth, squash to a bounded range. Historically dominant, now mostly avoided in hidden layers because they **saturate** — for large |z| the curve flattens and its derivative goes to ~0, which (section 7) chokes gradients in deep networks.
- **ReLU** `max(0, z)`: the modern default for hidden layers. Cheap, and crucially its derivative is exactly 1 on the positive side, so it doesn't shrink gradients. Its quirks (dead neurons when stuck at zero) are patched by variants like **Leaky ReLU** and **GELU** (the smooth version you saw in the transformer doc).

A famous result, the **Universal Approximation Theorem**, says a feedforward net with a single hidden layer and enough neurons can approximate any continuous function on a bounded domain to arbitrary accuracy. It's reassuring but weaker than it sounds: "enough neurons" can be astronomically large, and the theorem says nothing about whether gradient descent will actually *find* those weights. In practice, **depth** (many layers) is far more parameter-efficient than width (one enormous layer) — which is why we build deep networks rather than shallow-but-vast ones.

---

## 3. The forward pass

Generalize to `L` layers. We track two quantities per layer: the pre-activation `z` and the activation `a`.

```
a⁽⁰⁾ = x
for l = 1 … L:
    z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾        # linear
    a⁽ˡ⁾ = σ(z⁽ˡ⁾)                    # nonlinear (often identity on the last layer)
ŷ = a⁽ᴸ⁾
```

That's the **forward pass**: evaluate the function. Note we'll need to *remember* the `z` and `a` values at each layer — the backward pass reuses them. (Holding onto these activations is exactly why training a network costs much more memory than running it for inference.)

---

## 4. The loss: turning "fit the data" into a number to minimize

To optimize, we need a single scalar that measures wrongness. Two cases cover most of it:

**Regression** (predict a continuous value) → **Mean Squared Error**:

```
L = (1/N) Σ (ŷ − y)²
```

**Classification** (predict a category) → **softmax + cross-entropy**. Softmax turns the raw output scores (logits) into a probability distribution; cross-entropy then measures how much probability mass you put on the correct class:

```
probs = softmax(z⁽ᴸ⁾)            # logits -> distribution over classes
L = − (1/N) Σ log(probs[correct class])
```

Cross-entropy is the maximum-likelihood loss: minimizing it is the same as maximizing the probability the model assigns to the true labels — equivalently, minimizing the model's "surprise." This is the exact same objective that trains an LLM (where each "class" is the next token over the whole vocabulary). The loss converts the vague goal "fit the data" into a precise one: **find θ that minimizes this number.**

---

## 5. Gradient descent: how we minimize it

The loss `L(θ)` is a scalar function of millions of parameters. Picture (loosely) a landscape over that high-dimensional parameter space, with height = loss. We want a low point.

The gradient `∇_θ L` is the vector of partial derivatives `∂L/∂θᵢ`; it points in the direction of steepest *increase*. So to go downhill we step the other way:

```
θ ← θ − η · ∇_θ L
```

The **learning rate** `η` is the step size, and it's the most consequential hyperparameter you'll tune. Too large and you overshoot the valley and diverge (loss explodes to NaN); too small and training crawls. 

We rarely use the full dataset to compute each gradient. Variants:

- **Batch GD**: gradient over all data per step. Accurate, expensive, and the smooth gradient can get stuck.
- **Stochastic GD (SGD)**: gradient from one example at a time. Noisy but cheap.
- **Mini-batch GD**: gradient over a small batch (32–1024 examples). The practical default everywhere — it fits hardware (GPUs love batched matmuls) and the gradient noise actually *helps* escape bad spots in the landscape.

A blunt truth worth stating: this landscape is wildly non-convex, full of local minima and saddle points, and there is no theorem promising gradient descent reaches a good solution. It just reliably *does*, in practice, for the over-parameterized networks we use. That it works as well as it does is one of the genuine empirical surprises of the field.

The one missing piece is step 3: how do we get `∇_θ L` — the gradient with respect to *every* weight — efficiently? That's backprop.

---

## 6. Backpropagation: the heart of the machine

### 6.1 The problem

A modern network has millions to billions of parameters. We need `∂L/∂θ` for every single one. The naive approach — nudge each parameter a little and re-measure the loss (finite differences) — would require a full forward pass *per parameter*. For a million parameters that's a million forward passes per training step. Completely hopeless.

We need **all** the gradients from a single backward pass that costs about the same as one forward pass. Backpropagation delivers exactly that. It is not a new kind of math; it is the **chain rule, applied systematically and reusing shared work.**

### 6.2 The key idea: a computational graph with local gradients

A network is a composition of simple operations: `L = f_L ∘ f_{L−1} ∘ … ∘ f_1`. The chain rule says the derivative of a composition is the **product of the local derivatives** along the chain. Backprop organizes this as a graph: each operation is a node that knows how to compute (a) its output from its inputs (forward) and (b) how to convert a gradient on its output into gradients on its inputs (backward, using only *local* information).

Start with a tiny scalar example to see the mechanism nakedly. Let `e = (a · b) + c`, and suppose we want `∂e/∂a`, `∂e/∂b`, `∂e/∂c`:

```
a ──┐
    (·) ──► d=a·b ──┐
b ──┘               (+) ──► e
c ───────────────────┘
```

Forward gives the values. Backward starts with `∂e/∂e = 1` at the output and pushes gradients back through each node using its local rule:

- The `+` node: `e = d + c`, so locally `∂e/∂d = 1` and `∂e/∂c = 1`. It passes the incoming gradient (1) unchanged to both `d` and `c`.
- The `·` node: `d = a·b`, so locally `∂d/∂a = b` and `∂d/∂b = a`. It multiplies the gradient arriving at `d` by these: `∂e/∂a = 1·b`, `∂e/∂b = 1·a`.

Each node needed only the gradient flowing in from above and its own local derivative. **That locality is the whole trick** — it means we can compose arbitrary graphs and gradients just flow backward through them, no global derivation required. This is precisely what an autodiff framework automates.

### 6.3 The layer-wise matrix form

For an MLP we don't track scalars; we batch the chain rule into matrix operations. Define the **error signal** at layer `l` as `δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾` — the gradient of the loss with respect to that layer's pre-activations. Backprop computes these from the last layer to the first:

```
# at the output layer (for softmax + cross-entropy, the gradient is beautifully simple):
δ⁽ᴸ⁾ = ŷ − y

# propagate the error backward through each hidden layer:
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ σ′(z⁽ˡ⁾)

# once you have δ at a layer, its parameter gradients fall out directly:
∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ
∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

Read the middle line: to get the error at a layer, take the error from the layer above, pull it back through that layer's weights (the transpose `W⁽ˡ⁺¹⁾ᵀ`), and gate it by the local activation slope `σ′`. It's the same "incoming gradient × local derivative" pattern as the scalar graph, just vectorized. The `⊙` is elementwise multiplication. Note why we cached the forward values: `σ′(z⁽ˡ⁾)` needs the stored `z`, and the weight gradient needs the stored input activation `a⁽ˡ⁻¹⁾`.

### 6.4 The code: a complete net, trained from scratch

Here's a full two-layer classifier — forward, manual backward, gradient-descent update — on a genuinely nonlinear toy task. This is the centerpiece; read it line by line against the equations above.

```python
import numpy as np
rng = np.random.default_rng(0)

# Toy task: is a 2D point inside the unit circle? (A nonlinear boundary —
# no linear model can solve it, so the hidden layer has to earn its keep.)
N, D, H, C = 1000, 2, 32, 2
X = rng.normal(size=(N, D))
y = (np.sum(X**2, axis=1) < 1.0).astype(int)     # 1 = inside circle
Y = np.eye(C)[y]                                  # one-hot targets, (N, C)

# He initialization (scale by sqrt(2/fan_in)) — see section 7 for why this matters.
W1 = rng.normal(size=(D, H)) * np.sqrt(2.0 / D); b1 = np.zeros(H)
W2 = rng.normal(size=(H, C)) * np.sqrt(2.0 / H); b2 = np.zeros(C)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

lr = 0.5
for step in range(2001):
    # ---------- forward ----------
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)                         # ReLU
    z2 = a1 @ W2 + b2
    probs = softmax(z2)
    loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))   # cross-entropy

    # ---------- backward (the chain rule, layer by layer) ----------
    dz2 = (probs - Y) / N                          # δ at output: ŷ − y  (averaged over batch)
    dW2 = a1.T @ dz2                               # ∂L/∂W2 = δ · aᵀ
    db2 = dz2.sum(axis=0)                          # ∂L/∂b2 = δ
    da1 = dz2 @ W2.T                               # pull error back through W2
    dz1 = da1 * (z1 > 0)                           # gate by ReLU′ (1 where z>0, else 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    # ---------- update (gradient descent) ----------
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if step % 500 == 0:
        acc = (probs.argmax(1) == y).mean()
        print(f"step {step:4d}   loss {loss:.4f}   acc {acc:.3f}")
```

Run it and the loss drops while accuracy climbs toward ~1.0 — the network discovers a circular boundary that no linear model could. That is a neural network learning, with nothing hidden. Every line of "deep learning" is a scaled-up version of this loop.

### 6.5 How frameworks generalize this: a 40-line autograd engine

Writing the backward pass by hand for every architecture would be miserable and error-prone. Frameworks (PyTorch, JAX, TensorFlow) instead build the computational graph automatically during the forward pass and replay it in reverse. Here's the entire idea, stripped to its essence (a tiny version of Andrej Karpathy's *micrograd*):

```python
class Value:
    """A scalar that remembers how it was computed, so it can backprop."""
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None     # local rule: push our grad to our inputs
        self._prev = set(_children)       # the nodes that produced us

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other))
        def _backward():                  # ∂(a+b)/∂a = ∂(a+b)/∂b = 1
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))
        def _backward():                  # product rule
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,))
        def _backward():                  # ReLU′ = 1 if data>0 else 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # topologically sort the graph, then apply the chain rule in reverse order
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0                   # seed: ∂L/∂L = 1
        for v in reversed(topo):
            v._backward()                 # each node pushes its grad to its inputs
```

(Simplified — a real version wraps raw numbers as `Value`s and adds more ops.) With this, `loss.backward()` populates `.grad` on every parameter automatically, for *any* expression you build. The manual `dz2 = …; dW2 = …` block from 6.4 becomes one call. **This is conceptually all PyTorch's autograd is**: record local gradient rules during forward, then walk the graph backward multiplying them together. Backprop, once you see it as graph + locality, is general and mechanical.

---

## 7. Making it actually train

The math above will silently fail to learn if a few practical things are wrong. These are the issues that occupied the field for years and the fixes that unlocked deep networks.

**Initialization.** Don't initialize weights to zero — every neuron in a layer would then compute the same thing and receive the same gradient forever (a symmetry that never breaks). Initialize randomly, but *scaled*: **Xavier/Glorot** (for tanh) and **He** (for ReLU, scale `√(2/fan_in)`) keep the variance of activations and gradients roughly stable as signals pass through many layers. Bad scaling makes signals shrink or blow up exponentially with depth.

**Vanishing and exploding gradients.** Because backprop *multiplies* local derivatives across layers, deep networks are fragile. With saturating activations (sigmoid's derivative peaks at 0.25), the product of many small numbers vanishes — early layers get almost no gradient and don't learn. The opposite (repeated multiplication by >1) explodes. The fixes are exactly the ingredients you met in the transformer doc: **ReLU-family activations** (local derivative 1, not <1), careful **initialization**, **normalization layers** (batch norm / layer norm, which re-center and re-scale activations to keep them sane), and **residual connections** (`x + f(x)`, which give gradients a clean identity path straight back through the network). Residuals and normalization are not transformer-specific tricks — they're general deep-learning fixes for this multiplication problem.

**Better optimizers.** Plain SGD works but is slow and sensitive. Two upgrades dominate:
- **Momentum**: accumulate a running "velocity" of past gradients so updates build speed in consistent directions and damp oscillation — like a ball rolling downhill rather than teleporting.
- **Adam (AdamW)**: maintains a per-parameter adaptive learning rate from running estimates of the gradient's mean and variance. It's robust and the de-facto default for training large models — it's the optimizer the LLM doc mentioned was nudging all those billions of parameters.

**Learning-rate schedules.** A fixed `η` is rarely optimal. A short **warmup** (ramp up from near-zero) followed by gradual **decay** is standard for large models — it stabilizes the chaotic early phase, then lets the model settle.

---

## 8. Generalization: the goal we actually care about

A crucial reframe: **we don't care about the training loss.** A network that merely memorizes the training set is useless. We care about performance on *unseen* data — generalization. So we split data into **train** (fit on this), **validation** (tune hyperparameters against
