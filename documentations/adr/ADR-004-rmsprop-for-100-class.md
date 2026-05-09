# ADR-004: Why RMSprop for 100-Class Cold-Start Training

**Status:** Accepted  
**Date:** 2026-05  
**Context:** 100-class digit recognition is a challenging task with noisy gradients from class imbalance and synthetic augmentation data.

## Decision

Use **RMSprop** as the default optimizer for cold-start 100-class training, with a dynamic scheduler that switches to AdamW/Adam for fine-tuning near the ceiling.

## Rationale

1. **Gradient noise robustness:** RMSprop's per-parameter LR adaptation (ρ=0.9) handles the noisy gradients from fake-quantization (QAT) and class imbalance better than Adam.
2. **Proven track record:** All successful 100-class QAT runs used RMSprop. Adam consistently underperformed in cold-start scenarios.
3. **No weight decay interference:** Unlike Adam (which has incorrect weight decay), RMSprop has no built-in decay — L2 regularization works as expected.
4. **Dynamic scheduler compatibility:** RMSprop for initial climb → AdamW for fine-tuning gives the best of both worlds.

## Trade-offs

- RMSprop converges slower than Adam on well-balanced datasets.
- No built-in weight decay — must use explicit L2_REGULARIZATION.
- AdamW is better for fine-tuning once the model is near its ceiling.

## Consequences

- Default `OPTIMIZER_TYPE = "rmsprop"` for cold-start.
- `OPTIMIZER_SEQUENCE = ["rmsprop", "rmsprop", "adamw"]` for dynamic switching.
- Tuner searches across RMSprop, Adam, AdamW, SGD, and Nadam.
