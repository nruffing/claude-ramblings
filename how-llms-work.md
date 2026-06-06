# How LLMs Work: A Forward Pass, End to End

*A mechanistic walkthrough of the transformer, written for an experienced engineer who wants to understand the machine well enough to build one — not just use one.*

---

## How to read this

This document follows a single token-sequence on its journey through a decoder-only transformer (the GPT family), from raw text to a sampled next token. That path — **tokens → output** — is the whole inference story, and once you see it clearly, almost everything else about LLMs is a variation on this theme.

The frame throughout is the **residual stream**: a running vector representation, one per token position, that every layer reads from and writes back to. If you internalize that single idea, the architecture stops being a pile of boxes and becomes one coherent data-flow.

I assume you're fluent with vectors and matrices, gradients, softmax, and reading PyTorch. I will *not* re-derive backprop or explain what a dot product is. Code is in nanoGPT idiom — correct, minimal, and meant to be read.

Rough budget: ~1 hour. The attention section is the centerpiece; spend your attention there (so to speak).

---

## 0. The one-sentence model

An LLM is a function:

```
f(token_1, ..., token_n) -> probability distribution over the next token
```

That's it. It consumes a sequence of integers and emits a vector of `vocab_size` numbers that sum to 1 — the model's belief about what comes next. To generate text, you sample one token from that distribution, append it to the input, and call `f` again. This loop is called **autoregressive generation**, and it is the only thing a base LLM does.

Everything below is the internals of `f`. The shape of the data as it flows:

```
ids        (B, T)              integer token IDs
  │  embedding lookup + positional info
  ▼
x          (B, T, d_model)     the residual stream — one vector per token
  │  × N transformer blocks (attention + MLP), each rewriting x in place
  ▼
x          (B, T, d_model)
  │  final norm + projection to vocabulary
  ▼
logits     (B, T, vocab_size)  one distribution per position
  │  take last position, sample
  ▼
next_id    (B, 1)
```

Notation I'll use consistently: `B` = batch, `T` = sequence length (number of tokens, a.k.a. context), `d_model` = the width of the residual stream (also called `n_embd` or `d`), `n_head` = number of attention heads, `d_head = d_model / n_head`, `n_layer` = number of stacked blocks, `vocab_size` (≈ 50K–130K in practice).

For concreteness, keep GPT-2 small in mind as a running example: `d_model=768`, `n_head=12`, `n_layer=12`, `vocab_size=50257`, max context `1024`, ~124M parameters.

---

## 1. Tokenization: text becomes integers

The model never sees characters. It sees integers indexing a fixed vocabulary. The mapping is learned once, before training, and frozen.

**Why not just use characters or words?** Characters give tiny vocabularies but very long sequences (and attention cost grows with sequence length squared — section 4). Whole words give short sequences but an unbounded, sparse vocabulary that can't handle typos, new words, or code. The winning compromise is **subword tokenization**: common words become single tokens, rare words split into pieces, and you can always fall back to bytes so *nothing* is out-of-vocabulary.

The dominant algorithm is **Byte-Pair Encoding (BPE)**. The idea is greedy and almost embarrassingly simple:

1. Start with a vocabulary of all individual bytes (256 of them — so any UTF-8 text is representable).
2. Scan a large corpus and find the most frequent *adjacent pair* of tokens.
3. Merge that pair into a new single token; add it to the vocabulary.
4. Repeat until you hit your target vocabulary size.

The result is a list of merge rules. At inference time you apply them to split text into the longest known pieces. So `tokenization` might become `token` + `ization`, while a common word like ` the` is a single token (note the leading space — byte-level BPE treats spaces as part of tokens, which is why token counts and word counts never quite match).

A few consequences worth holding onto:

- **One token ≈ 0.75 words** in English on average; ~4 characters. This is why context windows and pricing are quoted in tokens.
- **Tokenization is lossy-looking but reversible.** Decode = concatenate the byte-pieces back together.
- **Many model quirks are tokenizer quirks.** The classic "how many *r*'s in strawberry" failures, weird arithmetic, and sensitivity to spacing all trace back to the fact that the model reasons over chunks like ` straw` + `berry`, not letters. The model literally cannot see the characters inside a token unless that information happens to be encoded in the token's learned vector.

You don't need to implement BPE to understand LLMs, but the mental model matters: **by the time the neural network sees anything, the input is a list of integers like `[15496, 11, 995]`, and the output it's trained to produce is the next integer.**

---

## 2. Embeddings: integers become vectors

A token ID is just an index. The first thing the network does is look it up in an **embedding table** — a learned matrix `W_e` of shape `(vocab_size, d_model)`. Row `i` is the `d_model`-dimensional vector for token `i`.

```python
self.tok_emb = nn.Embedding(vocab_size, d_model)   # the lookup table
x = self.tok_emb(ids)                               # (B, T) -> (B, T, d_model)
```

That's the entire operation: an indexed read. No matmul, just a gather. For GPT-2 small this table alone is `50257 × 768 ≈ 38.6M` parameters — about a third of the whole model.

**What is in these vectors?** They're learned, so they encode whatever helps predict the next token. Empirically they organize semantically: similar tokens land near each other, and directions in the space carry meaning (the famous, slightly overstated `king - man + woman ≈ queen` geometry). But don't over-romanticize a single token's vector. Its job is to be a *useful starting point* in the residual stream — a packet of features that later layers will refine using context. The word "bank" gets one vector regardless of whether you mean a river or a vault; disambiguating that is attention's job, not the embedding's.

This vector is the token's first entry into the **residual stream**. From here on, every layer's job is to *add information* to this stream.

---

## 3. Positional information: telling the model about order

Here's a fact that surprises people: **self-attention is permutation-invariant.** If you shuffle the input tokens, the core attention computation produces the same set of outputs, just shuffled. Attention sees a *set* of vectors, not a *sequence*. "The dog bit the man" and "the man bit the dog" would be indistinguishable.

So we must inject position explicitly. Three approaches, in historical order:

**Learned absolute positions (GPT-2).** Have a second lookup table `W_p` of shape `(max_T, d_model)` — one learned vector per position index — and add it to the token embedding.

```python
self.pos_emb = nn.Embedding(max_T, d_model)
pos = torch.arange(T, device=ids.device)            # [0, 1, 2, ..., T-1]
x = self.tok_emb(ids) + self.pos_emb(pos)           # (B, T, d_model)
```

Simple, but it hard-caps the context length at `max_T` and generalizes poorly beyond positions seen in training.

**Sinusoidal (original "Attention Is All You Need").** Use fixed sine/cosine waves of different frequencies as position signals. No parameters, and in principle extrapolates, but it's largely been superseded.

**Rotary Position Embedding — RoPE (most modern models: LLaMA, Qwen, etc.).** Instead of *adding* a position vector, RoPE *rotates* the query and key vectors (section 4) by an angle proportional to their position, applied inside each attention head. The elegant property: the dot product between a query at position `m` and a key at position `n` ends up depending only on their **relative** distance `m − n`, not their absolute positions. This is exactly what you want — language cares about "how far apart" more than "where from the start" — and it extrapolates to longer contexts far more gracefully. RoPE is why "context window extension" tricks (like adjusting the rotation base frequency) are even possible.

The takeaway for conceptual mastery: **position is not free; it's a design choice that's been bolted onto an order-blind core, and the move from absolute to relative (RoPE) is one of the quiet reasons modern models handle long contexts better.**

---

## 4. Self-attention: the heart of the machine

This is the only mechanism in the transformer that moves information *between* token positions. Everything else (the MLP, the norms) operates on each position independently. If you understand attention, you understand the part that makes transformers special.

### 4.1 The problem attention solves

After embedding, each position holds a context-free vector. But meaning is contextual: in "the trophy didn't fit in the suitcase because **it** was too big," resolving "it" requires looking back at "trophy." We need a mechanism where each position can **gather information from other positions**, weighted by relevance.

Attention is a differentiable, content-based lookup. Think of it as a soft dictionary:

- Each position emits a **query**: "here's what I'm looking for."
- Each position also emits a **key**: "here's what I contain."
- And a **value**: "here's what I'll hand over if you attend to me."

A position's query is compared against every key (via dot product). High match → high attention weight. The output for that position is a weighted average of all the values, weighted by those match scores. The pronoun "it" learns to emit a query that matches the key of "trophy," and so pulls trophy's value into its own representation.

The crucial design property: queries, keys, and values are all **linear projections of the same input** (hence *self*-attention). The model learns three weight matrices that decide what each token asks for, advertises, and offers.

### 4.2 The computation

Given input `X` of shape `(T, d_model)` (ignore batch for a moment), project to queries, keys, values:

```
Q = X W_q        K = X W_k        V = X W_v        # each (T, d_head)
```

Compute a score for every (query, key) pair, scale, softmax over the keys, and take the weighted sum of values:

```
                 ⎛ Q Kᵀ ⎞
Attention(Q,K,V) = softmax⎜ ───── + mask ⎟ V
                 ⎝ √d_head ⎠
```

Walking through it:

- `Q Kᵀ` is a `(T, T)` matrix. Entry `(i, j)` is the dot product of query `i` with key `j` — how much position `i` wants to attend to position `j`.
- **Divide by `√d_head`.** Without this, dot products grow with dimension, pushing softmax into saturated regions where gradients vanish. The square root keeps the variance of the scores roughly constant regardless of head size. This is a small detail with outsized importance for trainability.
- **`softmax` over the last dimension** turns each row into a probability distribution: position `i`'s attention weights over all positions sum to 1.
- **Multiply by `V`** `(T, d_head)`: each position's output is the attention-weighted average of all value vectors.

### 4.3 Causal masking: why a chatbot can't see the future

For a language model trained to predict the *next* token, position `i` must only attend to positions `≤ i`. If it could see future tokens, next-token prediction would be trivial (and useless) — the model would just copy the answer. We enforce this with a **causal mask**: set the scores for all future positions to `−∞` before the softmax, so their weights become exactly 0.

This single constraint is what makes the architecture **autoregressive** and is the difference between a GPT-style decoder (causal, left-to-right) and a BERT-style encoder (bidirectional, sees everything, used for understanding rather than generation).

It also enables a beautiful training efficiency: because of the mask, the prediction at every position depends only on earlier positions. So a single forward pass over a length-`T` sequence computes `T` next-token predictions *simultaneously*, each with the correct restricted view. You get `T` training signals for the price of one pass. (More on this in section 9.)

### 4.4 Multi-head attention: several conversations at once

One attention operation can only represent one "kind" of relationship at a time — its single softmax has to commit to one weighting. Language has many relationships running in parallel: syntactic agreement, coreference, positional patterns, topic. So we run several attention operations — **heads** — in parallel, each with its own `W_q, W_k, W_v` and its own slice of the dimension (`d_head = d_model / n_head`). Their outputs are concatenated and passed through one more projection `W_o`.

Different heads empirically specialize: some track the previous token, some do coreference, some attend to syntactically related words, some implement copying patterns ("induction heads," a key ingredient in in-context learning). Multi-head attention is, in effect, an **ensemble of relationship-detectors** sharing the same residual stream.

### 4.5 The code

Here is a complete, correct causal multi-head self-attention module. This is the piece worth reading line by line.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        # Fuse Q, K, V into one matmul for efficiency: (d_model) -> (3 * d_model)
        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)   # the W_o output projection

    def forward(self, x):
        B, T, C = x.shape                          # batch, time (tokens), channels (=d_model)

        # Project and split into Q, K, V, each (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=2)

        # Reshape each into heads: (B, T, C) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Scaled dot-product scores: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Causal mask: position i may attend only to j <= i
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))

        att = F.softmax(att, dim=-1)               # each row sums to 1
        y = att @ v                                # (B, n_head, T, d_head)

        # Concatenate heads back together: -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)                        # mix head outputs
```

> In production, that explicit score matrix and softmax are replaced by a fused kernel — **FlashAttention** — which computes the same result without ever materializing the full `(T, T)` matrix in memory. It's a numerically-equivalent IO optimization, not a different algorithm, but it's most of why long contexts are tractable today.

### 4.6 The cost that shapes everything

Notice `att` is `(T, T)`: attention is **O(T²)** in time and memory. Double the context, quadruple the attention cost. This single fact drives an enormous amount of LLM engineering — FlashAttention, sliding-window attention, sparse and linear attention variants, and the entire field of "long-context" research all exist to fight this quadratic. When you read about a model's context window being expensive, this is why.

---

## 5. The MLP: where computation (and arguably knowledge) lives

After attention has mixed information across positions, each position is processed independently by a **feed-forward network** (also called the MLP block). It's just two linear layers with a nonlinearity between them:

```python
class MLP(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.fc   = nn.Linear(d_model, mult * d_model)   # expand: 768 -> 3072
        self.proj = nn.Linear(mult * d_model, d_model)   # contract: 3072 -> 768

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))             # GELU is a smooth ReLU
```

Two things to understand:

**The 4× expansion is the convention.** The hidden layer is typically four times wider than `d_model`. This is where a large share of the parameters live — for GPT-2 small, the two MLP matrices per block (`768×3072` + `3072×768`) dwarf the attention matrices. The expansion gives the network room to compute richer nonlinear features before compressing back down.

**This is plausibly where facts are stored.** A productive interpretation (from mechanistic interpretability) views the MLP as a large **key-value memory**: the first matrix's rows act as pattern detectors ("keys") that fire on certain residual-stream directions, and the second matrix's columns ("values") write associated information back into the stream. Under this lens, attention *moves* information around and the MLP *looks things up and transforms* it. So "what is the capital of France" might be retrieved by an MLP neuron that fires on a France-feature and writes a Paris-feature. This is a model, not gospel — but it's a far more useful intuition than "it's just a nonlinearity."

The nonlinearity itself (GELU, or SwiGLU in newer models) matters only in that *some* nonlinearity is essential — stack linear layers without one and they collapse into a single linear map, no matter how many you stack.

---

## 6. The block: residual stream + normalization

Now we assemble attention and the MLP into a repeatable **transformer block**. Two structural ideas make deep stacks of these trainable: **residual connections** and **normalization**.

### 6.1 The residual stream (the central abstraction)

Each sub-layer doesn't *replace* the representation; it *adds* to it:

```python
class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # attention reads the stream, writes a delta
        x = x + self.mlp(self.ln2(x))     # MLP reads the stream, writes a delta
        return x
```

That `x = x + sublayer(...)` pattern is the **residual connection**, and it deserves more weight than it usually gets. The right mental model is a **residual stream**: a `d_model`-wide vector (per position) that flows straight through the entire network from embedding to output. Each sub-layer *reads* a copy of the stream, computes something, and *adds* its result back. Nothing overwrites; everything accumulates.

This reframes the whole architecture:

- The stream is a shared communication channel. Layer 3 can write a feature that layer 9 reads. Earlier work isn't destroyed.
- "Depth" means *successive refinement* of one representation, not a sequence of distinct transformations.
- Gradients flow cleanly backward through the additive identity path, which is why we can train 50, 100, even hundreds of layers without the signal vanishing. (This is the same insight that made ResNets work for vision; it's arguably *the* trick that made deep learning deep.)
- It explains why you can sometimes ablate or reorder layers and still get coherent output — layers communicate through a common medium rather than a rigid pipeline.

If you remember one diagram from this whole document, remember this: a wide arrow going straight from input to output, with attention and MLP blocks reaching in to read and add along the way.

### 6.2 Normalization: keeping activations sane

**LayerNorm** normalizes each token's vector to zero mean and unit variance (across the `d_model` features), then applies a learned scale and shift. It keeps activations in a stable range across the depth of the network, which keeps gradients well-behaved and training stable.

Note in the code above that the norm is applied *inside* the residual branch, before each sub-layer — this is **pre-norm**, and it's what virtually all modern models use because it's dramatically more stable to train than the original "post-norm" design (which placed the norm after the addition). Modern models also often swap LayerNorm for **RMSNorm**, a slightly cheaper variant that skips the mean-centering and just normalizes by the root-mean-square. Same purpose, fewer operations.

You don't need the exact formula memorized. The concept: **normalization is the maintenance crew that keeps the residual stream's numbers in a workable range so the rest of the machine functions.**

---

## 7. The full forward pass

Stack `n_layer` blocks, add a final norm, and project back to vocabulary size. Here's the complete model:

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_T):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)   # token lookup
        self.pos_emb = nn.Embedding(max_T, d_model)        # position lookup
        self.blocks  = nn.ModuleList(
            [Block(d_model, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)                  # final norm
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # unembedding

        # Weight tying: reuse the embedding matrix as the output projection.
        # Saves ~vocab_size * d_model params and tends to help.
        self.head.weight = self.tok_emb.weight

    def forward(self, ids):                 # ids: (B, T) integer token IDs
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)   # (B, T, d_model) — enter the stream
        for block in self.blocks:
            x = block(x)                            # refine the stream, N times
        x = self.ln_f(x)
        logits = self.head(x)                       # (B, T, vocab_size)
