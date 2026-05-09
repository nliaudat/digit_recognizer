# ADR-001: Why TQT over QAT for ESP32 Deployment

**Status:** Accepted  
**Date:** 2026-05  
**Context:** ESP32-S3 deployment requires INT8 quantized TFLite models with minimal accuracy loss.

## Decision

Use **Trainable Quantization Thresholds (TQT)** via `esp-ppq` instead of standard Quantization-Aware Training (QAT) for ESP32 deployment.

## Rationale

1. **Higher accuracy:** TQT fine-tunes quantization scales post-training, recovering 1–3% accuracy vs standard PTQ/QAT.
2. **ESP-DL compatibility:** TQT exports directly to ESP-DL's INT8 format without manual scale extraction.
3. **No training-time overhead:** QAT adds fake-quantization nodes during training, slowing each epoch. TQT runs after training.
4. **Multi-target support:** TQT can export for ESP32, ESP32-S3, and ESP32-P4 from a single float model.

## Trade-offs

- TQT requires `esp-ppq` which has GPU stability issues (CUDA segfaults) — forced to CPU.
- TQT adds a post-training step (~200 steps) that must be run after each training cycle.
- QAT is simpler for teams without access to `esp-ppq`.

## Consequences

- Default `QUANTIZATION_MODE = "tqt"` in `parameters.py`.
- TQT pipeline runs automatically after training in `train.py`.
- CPU-only TQT execution to avoid GPU segfaults.
