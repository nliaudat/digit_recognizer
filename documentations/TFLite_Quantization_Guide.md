# Digit Recognizer Quantization Guide: TQT, QAT & PTQ

This document explains the quantization strategies available in the Digit Recognizer pipeline. All strategies are controlled by a single master switch in `config/quantization.py`.

---

## 🎮 Quantization Mode — Master Switch

```python
# config/quantization.py
QUANTIZATION_MODE = "tqt"  # Options: "none", "ptq", "qat", "tqt", "auto"
```

When `QUANTIZATION_MODE` is set (not `None`), it automatically configures all underlying flags:

| Mode     | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| `none`   | Float32 training & inference. Reference baseline.                           |
| `ptq`    | Standard Post-Training Quantization (TFLite full-integer uint8).            |
| `qat`    | Quantization-Aware Training — must be set **before** training starts.       |
| `tqt`    | **Recommended** — Trainable Quantization Thresholds via ESP-DL pipeline.    |
| `auto`   | Automatically picks TQT for deployable models, float32 for teacher models.  |

To manually override individual flags without the master switch, set `QUANTIZATION_MODE = None` and set each flag directly.

---

## 🚀 Quantization Strategies

### 1. QAT (Quantization-Aware Training)

> [!IMPORTANT]
> **Must be configured BEFORE training.**

- **When to set**: In `config/quantization.py`, set `QUANTIZATION_MODE = "qat"` before running `train.py`.
- **Mechanism**: Modifies the model graph to insert "fake quantization" nodes during training. The model learns to compensate for quantization noise over many epochs.
- **Pros**: Generally offers the highest mathematical accuracy for standard TFLite targets.
- **Cons**: Requires a full training run from scratch; if you have already trained a model in `"none"` or `"tqt"` mode, you cannot retroactively apply QAT to it.
- **Workflow**:
  1. Edit `config/quantization.py` → `QUANTIZATION_MODE = "qat"`.
  2. Run `python train.py`.
  3. The result is a `.tflite` model that was "born" quantized.

### 2. TQT (Trainable Quantization Thresholds) — Recommended for ESP32

> [!NOTE]
> **Executed AFTER normal training.**

- **When to set**: Use the default `QUANTIZATION_MODE = "tqt"` or `"none"` in `config/quantization.py`.
- **Mechanism**: A specialized post-training refinement phase managed by `quantize_espdl.py`. It takes a standard pre-trained FP32 model and performs a fast (minutes) "calibration" to learn optimal per-channel quantization thresholds via the ESP-DL toolchain.
- **Pros**: Extremely fast; optimized specifically for each ESP chip; doesn't require retraining the entire model from scratch.
- **Workflow**:
  1. Train normally: `python train.py --model digit_recognizer_v4 --classes 10 --color gray`
  2. Refine and export: `python quantize_espdl.py --keras path/to/model.keras --target esp32`
  3. The result is an optimized `.espdl` and a suite of TFLite variants.

#### Per-Chip Hyperparameters

Each target chip has dedicated settings tuned for its hardware characteristics:

| Chip      | Steps | Learning Rate | Block Size | Integer Lambda |
|-----------|-------|---------------|------------|----------------|
| `esp32`   | 200   | 1e-6          | 2          | 0.10           |
| `esp32s3` | 200   | 1e-6          | 2          | 0.05           |
| `esp32p4` | 200   | 1e-6          | 2          | 0.0            |

- **Integer Lambda** controls the regularization strength on integerization error. Higher values force harder quantization at the cost of accuracy. ESP32-P4 needs no regularization (0.0) since it has superior DSP/NEON-like support.
- **Block Size** controls the granularity of threshold learning across channel groups.
- **Multi-chip export**: Set `TQT_EXPORT_ALL_TARGETS = True` in `config/quantization.py` to produce optimized models for all three targets in a single `quantize_espdl.py` run.

#### Calibration

The TQT pipeline uses 300 calibration steps at batch size 1 over 22,000 training samples to estimate activation ranges and refine thresholds. Controlled via:

```python
TQT_CALIB_STEPS      = 300   # config/quantization.py
TQT_CALIB_BATCH_SIZE = 1
```

#### Output Artifacts

The TQT pipeline produces 3 TFLite files per model:

| File | Purpose |
|------|---------|
| `*_quantized_integer_quant_float32.tflite` | Internal intermediate — creates a saved_model/ for uint8 conversion |
| `*_quantized_integer_quant_uint8.tflite`   | **ESP32 TFLite Micro** — uint8 I/O, TQT-quality weights |
| `*_quantized_float32.tflite`               | PC benchmarking — float32 I/O |

**File organization:**

- **Root directory** keeps only: `*_integer_quant_uint8.tflite`, `*_float32.tflite`, `.keras`, `.onnx`.
- **`full_models/` directory** stores all intermediate pipeline artifacts (`.espdl`, `*_integer_quant_float32`, per-chip variants, legacy names).

### 3. PTQ (Post-Training Quantization)

- Standard TensorFlow Lite full-integer quantization applied post-training.
- Uses representative calibration data to compute min/max ranges.
- No training or refinement needed — just convert.
- Controlled via `QUANTIZATION_MODE = "ptq"`.

---

## 📦 TFLite I/O Data Type: uint8 vs int8

This is critical for ESP32 deployment. The pipeline must produce models that match what the inference runtime expects.

```python
USE_TFLITE_BUILTINS_UINT8_ONLY = True   # Default: TFLite Micro path
USE_TFLITE_BUILTINS_INT8_ONLY  = False  # Set True for ESP-DL SDK
```

Only one of the two should be `True` at any time.

### uint8 Path (Default) — TFLite Micro on ESP32

- Uses `TFLITE_BUILTINS` ops with no forced I/O type override.
- The exported model accepts **uint8 [0, 255]** — raw camera bytes directly.
- ✅ Works with TFLite Micro on ESP32 without any pixel conversion.
- ❌ **NOT** compatible with ESP-DL SDK (needs int8).

### int8 Path — ESP-DL SDK

- Forces `TFLITE_BUILTINS_INT8` ops + `inference_input_type=tf.int8`.
- The exported model expects **int8 [-128, 127]**.
- ✅ Correct for ESP-DL SDK.
- ❌ Wrong for TFLite Micro — bytes get sign-misinterpreted.
- C++ code **must** convert: `int8_t pixel = (uint8_t)pixel_raw - 128;`

### History

- **Before April 2026**: `train_modelmanager.py` always produced uint8 models → `digit_recognizer_v23` and `v17` from 04.04.2026 work on ESP32.
- **April 2026**: `USE_TFLITE_BUILTINS_INT8_ONLY` was added for ESP-DL support → models exported after April 5 showed "low confidence" on ESP32.
- **June 2026**: `USE_TFLITE_BUILTINS_UINT8_ONLY` added as the default → reverts to the uint8 contract that matches TFLite Micro / raw camera bytes.

The model architectures themselves never changed — only these flags.

---

## 🔧 TFLite Export: Key Settings

| Setting | Default | Notes |
|---------|---------|-------|
| `DISABLE_XNNPACK` | `True` | Required for TFLite Micro compatibility; XNNPACK delegates ops away from the Micro interpreter. |
| `QUANTIZE_NUM_SAMPLES` | 22000 | Number of calibration samples used for range estimation. |
| `TF_DATA_SHUFFLE_BUFFER` | 1000 | Shuffle buffer size for the TF data pipeline. |

---

## 🔍 Auto Mode

`QUANTIZATION_MODE = "auto"` automatically selects the best strategy:

- **Deployable models** (models not ending in `_teacher`): TQT pipeline with ESP-DL target.
- **Teacher models** (ending in `_teacher`): Float32 (no quantization) since teachers only run on PC.

```python
def _auto_configure_quantization():
    """From config/quantization.py — automatically selects based on model name."""
    esp_compatible_models = [m for m in AVAILABLE_MODELS if not m.endswith("_teacher")]
    if MODEL_ARCHITECTURE in esp_compatible_models:
        # Enable TQT pipeline
        ...
    else:
        # Float32 — teacher models
```

---

## 💡 Summary

| Goal | Strategy | Configure When | Command |
|------|----------|---------------|---------|
| Float32 reference | `"none"` | Anytime | `QUANTIZATION_MODE = "none"` |
| Quick TFLite export | `"ptq"` | After training | `QUANTIZATION_MODE = "ptq"` |
| Best accuracy (retrain) | `"qat"` | **Before** training | `QUANTIZATION_MODE = "qat"` |
| **ESP32 deployment** | **`"tqt"`** | **After training** | **`QUANTIZATION_MODE = "tqt"`** (default) |
| Let the pipeline decide | `"auto"` | Anytime | `QUANTIZATION_MODE = "auto"` |

If you have a pre-trained model and want to deploy it to ESP32 efficiently, use the **TQT** pipeline via `quantize_espdl.py`. No retraining needed.

For detailed workflow steps, see the [Training Guide](training_guide.md) and [Benchmarking Guide](benchmarking_and_prediction.md).