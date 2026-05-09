# ADR-005: Why Data Augmentation is Done Offline (Static) vs Online

**Status:** Accepted  
**Date:** 2026-05  
**Context:** Training data for water meter digits is limited (~30K real images). Augmentation is essential but can be done online (during training) or offline (pre-generated).

## Decision

Use **offline (static) augmentation** — pre-generate augmented images using `datasets/tools/generate_augmented_dataset.py` and save them to disk.

## Rationale

1. **Deterministic reproducibility:** Static augmentation produces the same dataset every time — critical for debugging accuracy regressions.
2. **No training-time overhead:** Online augmentation slows each epoch by 2–5×. Static augmentation adds a one-time cost.
3. **Multi-source compatibility:** Static augmented images are loaded through the same `MultiSourceDataLoader` as real data — no special pipeline needed.
4. **Weighted sampling:** Static augmented datasets can be down-weighted (weight=0.5) to prevent synthetic data from dominating real data.

## Trade-offs

- Static augmentation uses ~2GB of disk space for the generated dataset.
- Cannot adapt augmentation parameters mid-training (e.g., increase rotation range if model plateaus).
- Online augmentation is still available via `USE_DATA_AUGMENTATION = True` for fine-tuning.

## Consequences

- `USE_DATA_AUGMENTATION = False` by default — augmentation is done offline.
- Static augmented datasets are stored in `datasets/static_augmentation/` and `datasets/static_augmentation_mixup/`.
- Synthetic datasets have `is_synthetic: True` flag and `weight: 0.5` to prevent over-dominance.
