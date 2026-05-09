# ADR-003: Why Grayscale over RGB Input

**Status:** Accepted  
**Date:** 2026-05  
**Context:** Water meter digit images are inherently grayscale (single-channel). The model must decide between 1-channel and 3-channel input.

## Decision

Use **grayscale (1-channel) input** as the default, with RGB support preserved for models that need it.

## Rationale

1. **3× fewer parameters:** First conv layer with 32 filters: 1×32×3×3 = 288 params (grayscale) vs 3×32×3×3 = 864 params (RGB) — 3× savings.
2. **No information loss:** Water meter digits are grayscale by nature. RGB adds no discriminative information.
3. **Faster inference:** 3× less data to move through the pipeline — critical for ESP32's limited memory bandwidth.
4. **Smaller model size:** ~30% smaller TFLite file for the same architecture.

## Trade-offs

- Some models (v23, v24) use luminance-based processing that requires RGB input internally but converts to grayscale via learned weights.
- RGB support is preserved for future use cases (e.g., colored digits, LED displays).

## Consequences

- Default `INPUT_CHANNELS = 1` (grayscale).
- `USE_GRAYSCALE` flag controls preprocessing pipeline.
- Models that need RGB internally (v23+) handle the conversion in their first layer.
