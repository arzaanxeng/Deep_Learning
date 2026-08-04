# The Vanishing Gradient Problem

## What It Is

In deep networks trained via backpropagation, gradients of the loss with respect to weights in early layers can shrink exponentially as they propagate backward through many layers — approaching zero. The result: early layers receive almost no learning signal and barely update, while later layers train normally. The network effectively becomes "shallow" in practice even though it's deep on paper.

## The Math

Backprop is repeated application of the chain rule. For a network with activations `a₁, a₂, ..., aₙ`, the gradient flowing back to layer 1 looks like:

```
∂L/∂W₁ = (∂L/∂aₙ) · (∂aₙ/∂aₙ₋₁) · ... · (∂a₂/∂a₁) · (∂a₁/∂W₁)
```

Each term `∂aᵢ/∂aᵢ₋₁ ≈ Wᵢᵀ · f'(zᵢ)` — a weight matrix multiplied by the activation function's derivative. This is a **long product of matrices**. If the "effective magnitude" of each term is consistently less than 1, the product shrinks exponentially with depth.

Two things drive this:

1. **Saturating activations** — sigmoid's derivative maxes out at 0.25 and → 0 for large |z|; tanh's derivative maxes at 1 but also saturates at the extremes. Neurons operating in the saturated region kill gradient flow.
2. **Weight scale** — if the singular values of `Wᵢ` are consistently < 1, repeated multiplication compounds the shrinkage.

## Quick Intuition

Picture the gradient as a signal passing through 50 "leaky pipes," each attenuating it by a factor of 0.9:

```
0.9^50 ≈ 0.005
```

By the time the signal reaches the first layer, it's essentially gone.

## Where It Bites Hardest

- Deep MLPs/CNNs with sigmoid or tanh activations (pre-ReLU era networks)
- **Vanilla RNNs** especially — the same weight matrix is reused at every timestep, so a long sequence is like an extremely deep network. Gradients from step 50 back to step 1 vanish fast, which is why plain RNNs struggle with long-range dependencies.

## Symptoms

- Early layers' weight histograms barely change across epochs
- Training loss plateaus early despite a deep architecture
- Deeper networks perform *worse* than shallower ones (a classic pre-ResNet finding)

## Solutions

| Fix | Why it helps |
|---|---|
| **ReLU / Leaky ReLU / GELU** | Derivative is 1 (not saturating) for the active region — no automatic shrinkage |
| **Xavier/Glorot init** (for tanh) or **He init** (for ReLU) | Scales initial weight variance so activations/gradients stay roughly constant in magnitude across layers |
| **Batch Normalization** | Keeps layer inputs in a well-conditioned range, away from saturation |
| **Residual/skip connections (ResNets)** | `∂(x + F(x))/∂x = 1 + ∂F/∂x` — the "+1" gives gradient a direct highway around the vanishing path |
| **LSTM / GRU gating** | Additive cell-state updates ("constant error carousel") let gradient flow across many timesteps without repeated multiplicative shrinkage |
| **Layer Normalization** | Similar stabilizing effect, common in RNNs/Transformers |

## Related
See `exploding_gradient_problem.md` for the opposite failure mode — same root cause (long chain-rule products), opposite direction.
