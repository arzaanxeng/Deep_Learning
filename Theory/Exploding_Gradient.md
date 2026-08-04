# The Exploding Gradient Problem

## What It Is

The mirror image of the vanishing gradient problem: during backpropagation, gradients grow exponentially large as they propagate backward through many layers. This causes huge, unstable weight updates — training diverges, loss spikes to `NaN`/`Inf`, and the model effectively breaks.

## The Math

Same chain-rule product as the vanishing case:

```
∂L/∂W₁ = (∂L/∂aₙ) · (∂aₙ/∂aₙ₋₁) · ... · (∂a₂/∂a₁) · (∂a₁/∂W₁)
```

Here, each term `∂aᵢ/∂aᵢ₋₁ ≈ Wᵢᵀ · f'(zᵢ)` has magnitude **greater than 1** on average. A long product of numbers > 1 blows up exponentially with depth instead of shrinking.

Common causes:

- **Large weight initialization** — weight matrices with large singular values amplify the signal at every layer
- **RNNs with spectral radius > 1** — if the largest eigenvalue of the recurrent weight matrix exceeds 1, repeated multiplication across timesteps compounds the amplification
- **High learning rates** — turn an already-large gradient into a destructively large parameter update

## Quick Intuition

Same leaky-pipe picture as before, but now each of 50 stages *amplifies* the signal by 1.1×:

```
1.1^50 ≈ 117
```

The gradient reaching the first layer is over 100× its original size — enough to send weights flying and destabilize training in a single step.

## Symptoms

- Loss suddenly jumps to `NaN` or a huge value mid-training
- Training loss curve is spiky and erratic rather than smoothly decreasing
- Weight norms grow unboundedly across epochs
- Easiest detection: log the L2 norm of the gradient every step — a sudden multi-order-of-magnitude spike is the tell

## Solutions

| Fix | Why it helps |
|---|---|
| **Gradient clipping** | The standard fix. Clip by norm: if `‖g‖ > threshold`, rescale `g ← g · threshold/‖g‖` (preserves direction, bounds magnitude). Clip-by-value is a simpler alternative but distorts direction. |
| **Careful initialization** (Xavier/He) | Keeps the effective spectral radius near 1 from the start |
| **Lower learning rate** | Directly shrinks the size of each update, even if the gradient itself is large |
| **Batch/Layer Normalization** | Stabilizes activation scale layer-to-layer, preventing runaway amplification |
| **L2 weight regularization** | Discourages weights from growing large in the first place |
| **LSTM/GRU gating (for RNNs)** | Gated, additive updates are less prone to the repeated-matrix-multiplication blowup than vanilla RNNs (though not fully immune — clipping is still standard practice) |

## Related
See `vanishing_gradient_problem.md` for the opposite failure mode — same underlying cause (long chain-rule products through depth or time), opposite direction of instability.
