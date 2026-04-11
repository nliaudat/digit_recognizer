# TQT Pipeline — Task Tracker

> Reference plan: [tqt_implementation_plan.md](tqt_implementation_plan.md)
> Target: Export digit-recognizer models to `.espdl` (ESPHome) + `.tflite` (TFLite) via esp-ppq TQT.

---

## Status Legend
- ✅ Done
- 🔧 Partial / needs fix
- ❌ Not started
- ➖ Explicitly deferred / N-A

---

## Prerequisites

| # | Task | Status | Notes |
|---|---|---|---|
| P1 | Add TQT deps to `docker/requirements.txt` | ✅ | `esp-ppq`, `tf2onnx`, `onnx`, `onnxsim`, `onnxruntime-gpu`, `onnx2tf` |
| P2 | Add TQT deps to `requirements_py3.12.3.txt` (local/CPU) | ✅ | CPU variants; no `onnxruntime-gpu` |
| P3 | Verify `tf2onnx` export on a real model | ❌ | **Needs Docker run** — cannot test locally without esp-ppq |

---

## Phase 0 — ONNX Export Helper

| # | Task | Status | Notes |
|---|---|---|---|
| 0.1 | Create `utils/export_onnx.py` | ✅ | Keras → ONNX via tf2onnx + onnxsim |
| 0.2 | `--inputs_as_nchw` flag (NHWC→NCHW) | ✅ | Default `True`; configurable |
| 0.3 | Verify luminance models (v23–v29) export cleanly | ❌ | **Needs Docker**: depthwise conv weight convention check |

---

## Phase 1 — Parameters

| # | Task | Status | Notes |
|---|---|---|---|
| 1.1 | Add `TQT_*` flags to `parameters.py` | ✅ | Full block with dual-output + PTQ note |
| 1.2 | Per-target auto-configuration (`_TQT_DEFAULTS` dict) | ✅ | Change only `TQT_TARGET`; all hypers auto-derived |
| 1.3 | QAT / TQT mutual exclusivity documented | ✅ | Comment block + `quantize_espdl.py` warning |

---

## Phase 2 — Core Script

| # | Task | Status | Notes |
|---|---|---|---|
| 2.1 | Create `quantize_espdl.py` | ✅ | Full pipeline: ONNX → .espdl (TQT) + .tflite (PTQ) |
| 2.2 | `DigitCaliDataset` (HWC→CHW, same preprocess) | ✅ | Reuses `preprocess_for_inference` |
| 2.3 | `run_tqt_quantization()` with all hyper-params | ✅ | Reads from `params.TQT_*` |
| 2.4 | `export_tflite_from_onnx()` via onnx2tf | ✅ | PTQ note embedded in output |
| 2.5 | `_warn_if_qat_checkpoint()` guard | ✅ | Warns if `USE_QAT=True` |
| 2.6 | CLI: `--model`, `--target`, `--steps`, `--skip_tflite`, etc. | ✅ | |
| 2.7 | Auto-resolve paths from `params.OUTPUT_DIR` | ✅ | |
| 2.8 | End-to-end test run in Docker | ❌ | **Needs Docker** |

---

## Phase 3 — Input Shape Convention

| # | Task | Status | Notes |
|---|---|---|---|
| 3.1 | NHWC→NCHW transpose in `DigitCaliDataset` | ✅ | |
| 3.2 | `--inputs_as_nchw` flag in `export_keras_to_onnx()` | ✅ | |

---

## Phase 4 — Integration with Existing Export Flow

| # | Task | Status | Notes |
|---|---|---|---|
| 4.1 | Option A: standalone `quantize_espdl.py` (run after training) | ✅ | Primary path, works now |
| 4.2 | Option B: `USE_TQT_PIPELINE` branch in `train.py` export block | ❌ | **Not started** — add after first successful Docker run validates Option A |
| 4.3 | Option B: `USE_TQT_PIPELINE` branch in `retrain.py` | ❌ | Low priority; retrain output is already a float `.keras` |

---

## Phase 5 — Benchmark & Evaluation

| # | Task | Status | Notes |
|---|---|---|---|
| 5a | Add `compare_float_vs_tqt()` to `utils/quantization_analysis.py` | ✅ | float ONNX vs TQT graph; MSE + accuracy delta |
| 5b | Add `OnnxDigitPredictor` to `predict.py` | ✅ | NCHW, CUDA/CPU fallback, same softmax heuristic |
| 5b | Extend `find_model_path()` to resolve `.onnx` / `.espdl` | ✅ | |
| 5b | Route `predict.py main()` to `OnnxDigitPredictor` for `.onnx` | ✅ | |
| 5c | Add `inspect_espdl()` + `--espdl` CLI to `bench_predict.py` | ✅ | File size + header hex dump |
| 5d | Run actual accuracy comparison (float vs TQT on val set) | ❌ | **Needs Docker** |

---

## Phase 6 — config_runner.py Integration

| # | Task | Status | Notes |
|---|---|---|---|
| 6.1 | Add Group E (TQT sweep) to `EXPERIMENTS` in `config_runner.py` | ✅ | E1/E2/E3 sweep steps/lr/block_size via `quantize_espdl.py` |
| 6.2 | Update `_build_cmd()` to route Group E to `quantize_espdl.py` | ✅ | |

---

## requirements_py3.12.3.txt

| # | Task | Status | Notes |
|---|---|---|---|
| R1 | Add CPU TQT deps (local, no GPU) | ✅ | `esp-ppq`, `tf2onnx`, `onnx`, `onnxsim`, `onnxruntime`, `onnx2tf` |

---

## Remaining Work (Docker)

These cannot be completed without a Docker run:

| Priority | Task |
|---|---|
| 🔴 High | Run `quantize_espdl.py` on one model (`digit_recognizer_v16`) — confirm end-to-end |
| 🔴 High | Check that `tf2onnx` export + `--inputs_as_nchw` works for all input channel counts |
| 🟡 Medium | Verify luminance models (v23–v29) export cleanly (depthwise conv layer) |
| 🟡 Medium | Validate `.tflite` from `onnx2tf` loads in `TFLiteDigitPredictor` |
| 🟢 Low | Integrate Phase 4B into `train.py` once Option A is validated |
| 🟢 Low | Run Group E sweep in `config_runner.py` to tune TQT hypers |

---

## Files Changed

| File | Change | Status |
|---|---|---|
| `parameters.py` | `TQT_*` block with per-target auto-config | ✅ |
| `utils/export_onnx.py` | **New** — Keras → ONNX | ✅ |
| `quantize_espdl.py` | **New** — TQT → `.espdl` + PTQ → `.tflite` | ✅ |
| `utils/quantization_analysis.py` | `compare_float_vs_tqt()` appended | ✅ |
| `predict.py` | `OnnxDigitPredictor` + `.onnx` routing in `main()` | ✅ |
| `bench_predict.py` | `inspect_espdl()` + `--espdl` flag | ✅ |
| `docker/requirements.txt` | TQT deps added | ✅ |
| `requirements_py3.12.3.txt` | Local CPU TQT deps | ✅ |
| `config_runner.py` | Group E TQT sweep | ✅ |
| `train.py` / `retrain.py` | Phase 4B integration | ❌ deferred |
| `tqt_implementation_plan.md` | Living plan (project root) | ✅ |
| `tqt_task.md` | This file | ✅ |
