# Digit Recognizer Quantization Guide: TQT vs QAT

This document explains the two primary quantization strategies available in the Digit Recognizer pipeline. Understanding the difference is critical, as one happens **during training** and the other happens **after training**.

---

## 🚀 Quantization Strategies

### 1. **QAT (Quantization-Aware Training)** 
> [!IMPORTANT]
> **Must be configured BEFORE training.**

*   **When to set**: In `parameters.py`, set `QUANTIZATION_MODE = "qat"` before running `train.py`.
*   **Mechanism**: Modifies the model graph to insert "fake quantization" nodes during training. The model learns to compensate for quantization noise over many epochs.
*   **Pros**: Generally offers the highest mathematical accuracy for standard TFLite targets.
*   **Cons**: Requires a full training run from scratch; if you have already trained a model in "none" or "tqt" mode, you cannot retroactively apply QAT to it.
*   **Workflow**: 
    1.  Edit `parameters.py` → `QUANTIZATION_MODE = "qat"`.
    2.  Run `python train.py`.
    3.  The result is a `.tflite` model that was "born" quantized.

### 2. **TQT (Training-aware Quantization Thresholds)** — *Recommended for ESP32*
> [!NOTE]
> **Executed AFTER normal training.**

*   **When to set**: You can use the default `QUANTIZATION_MODE = "tqt"` or `"none"` in `parameters.py`.
*   **Mechanism**: A specialized post-training refinement phase managed by `quantize_espdl.py`. It takes a standard pre-trained FP32 model and performs a very fast (minutes) "calibration" to find optimal hardware-specific bit-shift scales for the ESP32.
*   **Pros**: Extremely fast; optimized specifically for the ESP-DL hardware backend; doesn't require retraining the entire model from scratch.
*   **Workflow**:
    1.  Run `python train.py` (with standard settings).
    2.  Run `python quantize_espdl.py --keras ...` to refine and export for ESP32.
    3.  The result is an optimized `.espdl` and a suite of TFLite variants.

---

## 📦 TFLite Variants Comparison

Regardless of the strategy used, the export pipeline generates these standard formats:

| Model Variant (`.tflite`) | Quantization | Recommended Use Case |
| :--- | :--- | :--- |
| **`*_float.tflite`** | FP32 | Debugging and baseline baseline accuracy. |
| **`*_float16.tflite`** | FP16 | GPU-accelerated devices (Mobile/Web). |
| **`*_dynamic_range_quant.tflite`** | W: INT8 / Act: FP32 | Simple CPU speedup on Mobile/PC. |
| **`*_full_integer_quant.tflite`** | **INT8 Everything** | **✅ Primary target for ESP32 and MCUs**. |
| **`*_full_integer_quant_with_int16_act.tflite`** | W: INT8 / Act: INT16 | High-precision signal requirements. |

---

## 📁 Artifact Management Summary

*   **Root Folder**: Contains only deployment-ready files (`.espdl` for Native ESP, `_full_integer_quant.tflite` for TFLite Micro).
*   **`full_models/` Folder**: Contains all other 20+ variants for archiving and debugging.

## 💡 Summary
If you want to use **QAT**, you **must** commit to it in `parameters.py` before you start training. If you have a pre-trained model and want to deploy it to ESP32 efficiently, use the **TQT** pipeline via `quantize_espdl.py`.
