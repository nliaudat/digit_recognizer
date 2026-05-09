# ADR-002: Why GhostNet Family for the IoT Digit Recognition Task

**Status:** Accepted  
**Date:** 2026-05  
**Context:** Need a CNN architecture that achieves >90% accuracy on 10-class rotating digit recognition while fitting in <100KB INT8 for ESP32-S3.

## Decision

Use **GhostNet-inspired architectures** (v18, v19, v27, v28, v29) as the primary model family.

## Rationale

1. **Ghost convolutions:** Generate "ghost" feature maps via cheap linear operations, reducing parameters by 2–4× vs standard convolutions at the same FLOP count.
2. **Proven IoT performance:** v18 achieves >90% accuracy at <100KB INT8 — the only family to cross this threshold.
3. **Grayscale-first design:** GhostNet's efficiency compounds with 1-channel input, making it ideal for water meter digit recognition.
4. **Progressive improvements:** v23→v29 show a clear evolution path (adaptive contrast, soft binarization, 2-channel hybrid processing).

## Trade-offs

- GhostNet is less well-known than MobileNetV2 — fewer community resources.
- Some GhostNet variants use custom ops that may not be TFLite Micro compatible (vetted in v18+).
- Larger models (v19 for 100-class) approach 1.5MB — borderline for ESP32 flash.

## Consequences

- All new model development should start from the GhostNet family.
- Legacy models (v3–v17) are preserved for backward compatibility but not actively developed.
- Teacher models for distillation use EfficientNetB0/ResNet50 backbones (not GhostNet).
