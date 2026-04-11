# TQT Quantization Pipeline — Implementation Plan

> **Goal**: Export digit recognizer models to `.espdl` format for ESP32-P4/S3 targets using
> ESP-PPQ's **Trained Quantization Thresholds (TQT)** pass, achieving better INT8 accuracy
> than vanilla post-training quantization (PTQ). This replaces / complements the existing
> TFLite + `train_qat_helper.py` path with an ONNX-based ESP-DL pipeline.

---

## Background: Why TQT fits this project

| What we have | TQT relevance |
|---|---|
| Keras models trained with `USE_QAT=True` (fake-quant) | TQT only needs **calibration data**, not labels → reuse existing dataset loaders |
| 20×32 grayscale/RGB images, 10 or 100 classes | Small inputs → calibration is fast even on CPU |
| `QUANTIZE_NUM_SAMPLES = 22 000` representative set | Ready-made calibration pool |
| `exported_models/` directory per class/colour config | Natural home for `.espdl` output |
| `parameters.py` central config | New TQT flags fit existing pattern |
| `utils/quantization_analysis.py` | Extend for TQT error reporting |

TQT jointly fine-tunes **quantization thresholds (scale)** and **weights** using MSE loss
between float and quantized outputs — no labels needed, fewer compute requirements than
full QAT, and directly targets ESP-DL's Power-of-2 + Symmetric + Per-Tensor constraint.

---

## Prerequisites

### 1. Install esp-ppq (≥ 1.2.7)

**In Docker (GPU available — `tensorflow:2.20.0-gpu` base image):**
```bash
pip install esp-ppq>=1.2.7 tf2onnx>=1.16.0 onnx>=1.16.0 onnxsim>=0.4.36 onnxruntime-gpu>=1.18.0
```

> ✅ `docker/requirements.txt` already updated with these packages.
> CUDA is confirmed available in the Docker image — `TQT_COLLECTING_DEVICE` defaults to `"cuda"`
> for 3–5× faster TQT fine-tuning (esp-ppq uses torch CUDA kernels for the backprop pass).

**Outside Docker (local/CPU fallback):**
```bash
pip install esp-ppq>=1.2.7 tf2onnx>=1.16.0 onnx>=1.16.0 onnxsim>=0.4.36 onnxruntime
```
Add to `requirements_py3.12.3.txt` as well.

### 2. Verify ONNX export works

The existing Keras→TFLite path doesn't produce ONNX. We need an intermediate step.
- **`tf2onnx`** (recommended): already in Docker requirements → `python -m tf2onnx.convert`

---

## Phase 0 — ONNX Export Helper

**New file**: `utils/export_onnx.py`

```python
"""Export a trained Keras model to ONNX for use with esp-ppq / TQT."""
import subprocess, pathlib, onnx

def export_keras_to_onnx(keras_model_path: str, onnx_path: str,
                          input_shape: tuple, simplify: bool = True) -> str:
    """
    Run tf2onnx to convert a saved Keras model (.keras or SavedModel) to ONNX.
    Returns the path of the produced ONNX file.
    """
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", keras_model_path,
        "--output", onnx_path,
        "--opset", "13",
    ]
    subprocess.run(cmd, check=True)

    if simplify:
        import onnxsim
        model = onnx.load(onnx_path)
        model_simplified, check = onnxsim.simplify(model)
        assert check
        model_simplified = onnx.shape_inference.infer_shapes(model_simplified)
        onnx.save(model_simplified, onnx_path)

    return onnx_path
```

**Integration point**: call from `train.py` post-training export block (or standalone
`export_onnx.py` script triggered by a new CLI flag).

---

## Phase 1 — New Parameters (parameters.py)

Add a new `# TQT / ESP-DL QUANTIZATION` section:

> **QAT and TQT are mutually exclusive pipeline paths.**
> - `USE_QAT = True` → TensorFlow fake-quant during training → `.tflite` export
> - `USE_TQT_PIPELINE = True` → converged **float** model → ONNX → TQT fine-tune → **both** `.espdl` + `.tflite` export
>
> Running both on the same model is redundant and counter-productive: QAT bakes fake-quant
> into the Keras graph, which then exports to ONNX with distorted weight distributions
> that confuse TQT's MSE-based scale optimisation.
> The code guard in `quantize_espdl.py` (Phase 2) will warn if `USE_QAT=True` is detected
> in the loaded checkpoint.

```python
# ==============================================================================
# TQT / ESP-DL QUANTIZATION
# ==============================================================================
# Mutually exclusive with USE_QAT:
#   USE_QAT=True  → TFLite path (fake-quant during training)
#   USE_TQT_PIPELINE=True → ONNX → .espdl path (post-training TQT)
# Both can coexist in parameters.py to allow running each independently,
# but quantize_espdl.py will warn if USE_QAT=True is set.

# Enable the ONNX → .espdl TQT pipeline (requires esp-ppq ≥ 1.2.7)
USE_TQT_PIPELINE       = False          # master switch

# Hardware target — determines rounding strategy and operator support.
# Valid values and notes:
#   "c"       → ESP32 (original)  ROUND_HALF_UP  — note: must use "c", NOT "esp32"
#   "esp32s3" → ESP32-S3          ROUND_HALF_UP
#   "esp32p4" → ESP32-P4          ROUND_HALF_EVEN  (most accurate INT8 hardware)
# ⚠ DO NOT mix: the .espdl file is target-specific and won't run correctly on another chip.
TQT_TARGET             = "esp32p4"      # change to "c" for original ESP32, "esp32s3" for S3
TQT_NUM_BITS           = 8              # 8 or 16

# TQT fine-tuning hypers (see esp-ppq TQTSetting docs)
TQT_STEPS              = 500            # steps per block (200 for fast, 500 for best)
TQT_LR                 = 1e-5           # learning rate
TQT_BLOCK_SIZE         = 4              # larger = faster, less stable
TQT_INT_LAMBDA         = 0.25           # pull alpha toward integer (0 = off)
TQT_IS_SCALE_TRAINABLE = True           # also fine-tune scale, not just weights
TQT_COLLECTING_DEVICE  = "cuda"          # "cuda" in Docker (GPU confirmed); "cpu" fallback

# Calibration
TQT_CALIB_STEPS        = 32            # calibration batches passed before TQT
TQT_CALIB_BATCH_SIZE   = 32
```

---

## Phase 2 — Core Script: `quantize_espdl.py`

**New top-level file** (mirrors `retrain.py` in style).

### Structure

```
quantize_espdl.py
  ├── parse_args()            — --model, --onnx, --output, --target, --classes, ...
  ├── build_calibration_dataloader()   — wraps existing DATA_SOURCES / preprocess
  ├── run_tqt_quantization()  — main quantization + export
  └── evaluate_quantized()    — compare float vs quantized outputs (MSE / accuracy)
```

### Skeleton

```python
"""
quantize_espdl.py
Export a trained digit-recognizer model to .espdl using esp-ppq TQT.

Usage:
    python quantize_espdl.py \
        --onnx exported_models/100cls_RGB/digit_recognizer_v16.onnx \
        --output exported_models/100cls_RGB/digit_recognizer_v16.espdl \
        --target esp32p4
"""
import argparse, torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

import parameters as params
from utils.preprocess import preprocess_for_inference

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx


class DigitCaliDataset(Dataset):
    """Torch Dataset wrapping the project's numpy calibration arrays."""
    def __init__(self, x_np: np.ndarray):
        # x_np: uint8 [N, H, W, C] → float32 NCHW
        x = preprocess_for_inference(x_np).astype("float32")
        # HWC → CHW (esp-ppq / ONNX expect NCHW)
        self.data = torch.from_numpy(x.transpose(0, 3, 1, 2))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def build_calibration_dataloader() -> DataLoader:
    from utils import get_data_splits
    (x_train, _), _, _ = get_data_splits()
    n = min(params.TQT_CALIB_STEPS * params.TQT_CALIB_BATCH_SIZE, len(x_train))
    dataset = DigitCaliDataset(x_train[:n])
    return DataLoader(dataset, batch_size=params.TQT_CALIB_BATCH_SIZE, shuffle=False)


def run_tqt_quantization(onnx_path: str, espdl_path: str, target: str):
    dataloader = build_calibration_dataloader()

    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.tqt_optimization = True
    s = quant_setting.tqt_optimization_setting
    s.steps              = params.TQT_STEPS
    s.lr                 = params.TQT_LR
    s.block_size         = params.TQT_BLOCK_SIZE
    s.int_lambda         = params.TQT_INT_LAMBDA
    s.is_scale_trainable = params.TQT_IS_SCALE_TRAINABLE
    s.collecting_device  = params.TQT_COLLECTING_DEVICE

    collate_fn = lambda batch: torch.stack(batch).to(params.TQT_COLLECTING_DEVICE)

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file  = onnx_path,
        espdl_export_file = espdl_path,
        calib_dataloader  = dataloader,
        calib_steps       = params.TQT_CALIB_STEPS,
        input_shape       = [1, params.INPUT_CHANNELS,
                               params.INPUT_HEIGHT, params.INPUT_WIDTH],
        target            = target,
        num_of_bits       = params.TQT_NUM_BITS,
        collate_fn        = collate_fn,
        setting           = quant_setting,
        device            = params.TQT_COLLECTING_DEVICE,
        error_report      = True,
        skip_export       = False,
        export_test_values= True,
        verbose           = 1,
    )
    return quant_ppq_graph


if __name__ == "__main__":
    # ... argparse, then call run_tqt_quantization(...)
    pass
```

> **Dual output**: `quantize_espdl.py` produces **both** `.espdl` and `.tflite` from the
> same float ONNX in a single run:
>
> ```
> float .keras
>   └─ export_keras_to_onnx() ──► float .onnx
>                                     ├─ espdl_quantize_onnx() + TQT ──► .espdl   (for ESPHome)
>                                     └─ onnx2tf + PTQ calib data   ──► .tflite   (for TFLite runtime)
> ```
>
> The TFLite is produced via **`onnx2tf`** (PTQ with the same calibration representative
> dataset). It does not carry TQT-optimized scales, but uses the same float weights and
> avoids re-training. Add to `docker/requirements.txt`:
> ```
> onnx2tf>=1.20.0
> flatbuffers>=2.0
> ```
>
> ```python
> def export_tflite_from_onnx(onnx_path: str, tflite_path: str,
>                              calib_dataloader) -> str:
>     """Convert float ONNX → TFLite with INT8 PTQ via onnx2tf."""
>     import subprocess
>     tf_dir = tflite_path.replace(".tflite", "_tf")
>     subprocess.run([
>         "onnx2tf", "-i", onnx_path, "-o", tf_dir,
>         "--output_integer_quantized_tflite",
>         "--quant_type", "per_tensor",
>     ], check=True)
>     # onnx2tf writes the quantized .tflite into tf_dir automatically
>     return tflite_path
> ```

---

## Phase 3 — Input Shape Convention (HWC → CHW)

> **Critical**: TFLite models use **NHWC** (TensorFlow default).  
> ONNX / PyTorch / esp-ppq expect **NCHW**.

The `DigitCaliDataset` above handles the transpose.  
The ONNX export via tf2onnx also needs `--inputs_as_nchw` flag or a reshape layer:

```bash
python -m tf2onnx.convert \
    --saved-model checkpoints/digit_recognizer_v16 \
    --output exported_models/.../digit_recognizer_v16.onnx \
    --opset 13 \
    --inputs_as_nchw input_1
```

---

## Phase 4 — Integration with Existing Export Flow

### Option A (lightweight): standalone script
Run `quantize_espdl.py` **after** normal training finishes. No changes to `train.py`.

```
train.py  →  best_model.keras
          →  digit_recognizer_v16.tflite    (existing)
quantize_espdl.py
          →  digit_recognizer_v16.onnx      (new)
          →  digit_recognizer_v16.espdl     (new)
```

### Option B (integrated): extend `train.py` export block
Add an `if params.USE_TQT_PIPELINE:` branch in the post-training export section of
`train.py` / `retrain.py`, calling `run_tqt_quantization()` from `quantize_espdl.py`.
This is cleaner for automated runs via `train_all.py` / `retrain_all.py`.

**Recommended: Option A first, promote to Option B once validated.**

---

## Phase 5 — Benchmark & Evaluation

### 5a. Extend `utils/quantization_analysis.py`

Add `compare_float_vs_tqt(onnx_path, quant_ppq_graph, x_val)` that:
1. Runs float ONNX inference via `onnxruntime`
2. Runs quantized graph inference via esp-ppq's `PPQTorchExecutor`
3. Reports per-class accuracy delta and mean output MSE

### 5b. Extend `predict.py` with ONNX inference

`predict.py` currently only handles `.tflite` via `TFLiteDigitPredictor`. Add an
`OnnxDigitPredictor` class (or extend via duck typing) so the same `--model` CLI
can load `.onnx` files for float validation — and compare against the TQT quantized graph.

```python
class OnnxDigitPredictor:
    """Inference using onnxruntime — for float and TQT-quantized ONNX models."""
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image):
        from utils.preprocess import preprocess_for_inference
        # NHWC float32 [0,1] → NCHW (ONNX convention)
        x = preprocess_for_inference(image).astype("float32")  # [H,W,C]
        x = x.transpose(2, 0, 1)[np.newaxis]                  # [1,C,H,W]
        logits = self.session.run(None, {self.input_name: x})[0][0]
        # same logits/softmax auto-detection as TFLiteDigitPredictor
        ...
        return prediction, confidence, output_vector
```

**`find_model_path()` extension**: recognise `.onnx` and `.espdl` extensions alongside
`.tflite` so `--model digit_recognizer_v16` resolves to the right file automatically.

### 5c. Extend `bench_predict.py`

Add an `--espdl` flag reporting `.espdl` file size and embedded quantization metadata
(scale, zero-point per layer via esp-ppq graph inspection).

---

## Phase 6 — config_runner.py Integration

Add a new `config_runner` group for TQT sweeps:

```python
TQT_SWEEP = [
    {"TQT_STEPS": 200, "TQT_LR": 1e-5, "TQT_BLOCK_SIZE": 2},
    {"TQT_STEPS": 500, "TQT_LR": 1e-5, "TQT_BLOCK_SIZE": 4},
    {"TQT_STEPS": 500, "TQT_LR": 1e-4, "TQT_INT_LAMBDA": 0.25},
]
```

This maps directly to the existing `config_runner.py` override mechanism.

---

## File Change Summary

| File | Change |
|---|---|
| `parameters.py` | Add `TQT_*` flags section |
| `utils/export_onnx.py` | **New** — Keras → ONNX via tf2onnx |
| `quantize_espdl.py` | **New** — TQT → `.espdl` + `onnx2tf` → `.tflite` (dual output) |
| `utils/quantization_analysis.py` | Extend with `compare_float_vs_tqt()` |
| `predict.py` | Add `OnnxDigitPredictor` class + `.onnx` model discovery |
| `bench_predict.py` | Add `--espdl` size/metadata reporting |
| `docker/requirements.txt` | ✅ Already updated — add `onnx2tf>=1.20.0` too |
| `requirements_py3.12.3.txt` | Add same (CPU versions) for local use |
| `train.py` / `retrain.py` | (Phase 4B) Optional `USE_TQT_PIPELINE` branch |
| `config_runner.py` | (Phase 6) TQT sweep group |

---

## Suggested Implementation Order

1. **Install deps** + verify `tf2onnx` export on one model (v16 or v18)
2. **`parameters.py`** — add TQT flags
3. **`utils/export_onnx.py`** — Keras → ONNX helper
4. **`quantize_espdl.py`** — core TQT script (MVP: hardcoded model path)
5. **Test end-to-end**: float accuracy vs TQT quantized accuracy on val set
6. **`utils/quantization_analysis.py`** — add error reporting
7. **`bench_predict.py`** — add `.espdl` metadata reporting
8. **`config_runner.py`** — TQT sweep group
9. **`train.py` integration** (Option B) if sweeps show value

---

## Open Questions

> [!IMPORTANT]
> One key decision remaining:

1. **Target hardware**: Which chip? Sets `TQT_TARGET` and affects which `.espdl` binary
   is produced for [esphome_ai_component](https://github.com/nliaudat/esphome_ai_component/):
   - Original ESP32 → `"c"` (`ROUND_HALF_UP`)
   - ESP32-S3 → `"esp32s3"` (`ROUND_HALF_UP`)
   - ESP32-P4 → `"esp32p4"` (`ROUND_HALF_EVEN`, best INT8 hw)

> [!NOTE]
> **Deployment path confirmed**: `.espdl` → ESPHome via
> [esphome_ai_component](https://github.com/nliaudat/esphome_ai_component/).
> On-device validation will be handled independently — no idf.py toolchain needed here.

> [!NOTE]
> **NCHW convention**: models v23–v29 have fixed depthwise conv weights for luminance
> conversion. Verify these export cleanly to ONNX with `--inputs_as_nchw` before
> running TQT on them.
