# ADR-006: Why 10 Classes is the Primary Target with 100-Class Helper

**Status:** Accepted  
**Date:** 2026-05  
**Context:** Water meter digit recognition needs to classify digits 0–9 (10 classes). However, the project also supports 100-class training.

## Decision

**10 classes** is the primary deployment target. **100 classes** is a helper mode for training and classification refinement.

## Rationale

1. **Physical constraint:** Water meters display digits 0–9. There are only 10 possible values per digit position.
2. **100-class as auxiliary:** The 100-class mode maps each digit to 10 sub-classes (e.g., digit "3" with 10 rotation variants). This helps the model learn rotation invariance by explicitly classifying rotation buckets.
3. **Model size:** 10-class models are significantly smaller (<100KB INT8) than 100-class models (~1.5MB) — critical for ESP32 flash constraints.
4. **Accuracy ceiling:** 10-class models achieve >99% accuracy. 100-class models plateau around 92% due to the harder task.

## Trade-offs

- 100-class training produces better feature representations that can be transferred to 10-class via fine-tuning.
- 100-class inference requires post-processing to map 100 outputs → 10 digits (summing sub-class probabilities).
- Most development effort goes into 10-class optimization since that's what runs on the ESP32.

## Consequences

- Default `NB_CLASSES = 10` for deployment, `NB_CLASSES = 100` for training refinement.
- `DIGIT_NB_CLASSES` env var controls the mode at runtime.
- Benchmark results are reported separately for 10-class and 100-class.
- 100-class models are not deployed to ESP32 — only used for PC-based analysis.
