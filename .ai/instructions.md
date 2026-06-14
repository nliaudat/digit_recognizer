# Digit Recognizer Project Instructions

## Project Overview
This project focuses on training deep learning models for water meter digit recognition (both 10-class and 100-class). The models are specifically tailored and optimized for edge deployment on ESP32 microcontrollers using ESP-DL and TFLite Micro constraints. 

## Key Technologies
- **Python 3.12**
- **TensorFlow / Keras**
- **TensorFlow Lite (TFLite)** with a strong focus on **Quantization Aware Training (QAT)** and INT8 quantization.

## Architecture and Core Files
- `parameters.py`: The central source of truth for all global variables, hyperparameters, dataset paths, and training configs. **Always update or reference this file instead of hardcoding values.**
- `train.py`: The primary entry point for training models.
- `models/`: Directory containing Keras model definitions. Each model (e.g., `digit_recognizer_vX`) explores different size/accuracy tradeoffs.
- `utils/`: Helper modules (e.g., image generation, result plotting, and system diagnostics).
- `tuner.py`: Script for hyperparameter optimization using Keras Tuner.
- `predict.py` / `bench_predict.py`: Utilities for testing exported models.

## AI Assistant Guidelines

### 1. Edge Device & ESP-DL Compatibility
- **Operation Constraints**: When designing or modifying models, remember they must run on TFLite Micro and ESP-DL. Avoid operations that are not supported by these runtime environments.
- **Activations**: Be extremely careful with custom activations. Stick to standard layers (Conv2D, DepthwiseConv2D, Dense, MaxPooling2D, AveragePooling2D, ReLU, Softmax).
- **Memory Limits**: The primary goal is finding the Pareto-optimal frontier between model size (in KB) and accuracy. When proposing new architectures, minimizing parameters and memory footprint is critical. 

### 2. Quantization Aware Training (QAT)
- Most viable models are quantized to INT8 or UINT8. Ensure that changes in the model structure (e.g., new layers, normalization layers, binarization steps) interact correctly with `tensorflow-model-optimization` (tfmot) tools.

### 3. Model Parameters and Dimensions
- **Input Shape**: Default input dimensions are strictly `Height=32, Width=20`. 
- **Channels**: The pipeline supports both Grayscale (`INPUT_CHANNELS=1`) and RGB (`INPUT_CHANNELS=3`). Code should dynamically adapt to both via `parameters.INPUT_CHANNELS`.
- **Classes**: Models must scale seamlessly between 10-class (0-9) and 100-class (0-99) problems, reading the target from `parameters.NB_CLASSES`.

### 4. Code Style and Conventions
- Follow existing clear formatting patterns. `parameters.py` relies on well-commented uppercase global constants grouped by domain (e.g., GENERAL PARAMETERS, GRADIENT & TRAINING HYPERPARAMETERS).
- Use `snake_case` for variables and functions, `CamelCase` for classes.
- Log important script outputs explicitly (e.g., using `print` with emojis for visibility, as seen across the codebase).

### 5. **NO HARDCODED VALUES** — Centralized Configuration
- Every tunable value must live in its dedicated `config/` module, NOT in Python code.
  - `config/distillation.py` — distillation hyperparameters
  - `config/training.py` — training hyperparameters
  - `config/augmentation.py` — augmentation parameters
  - `config/models.py` — model architecture choices
  - `config/quantization.py` — quantization parameters
- Function default arguments should reference config constants (e.g., `temperature=dist_cfg.DISTILLATION_TEMPERATURE`), not literal numbers.
- Ratios, thresholds, and clipping boundaries must be config constants.
- See `.clinerules` (project root) for the definitive rule and a checklist of common violations.

### 6. TFLite Export & ESP32 Deployment

#### Pipeline Output Files
The TQT pipeline in `quantize_espdl.py` produces exactly 3 TFLite files:
- `*_quantized_integer_quant_uint8.tflite` — **ESP32 deployment** (uint8 I/O, TQT-quality weights)
- `*_quantized_integer_quant_float32.tflite` — internal intermediate (required to create saved_model/)
- `*_quantized_float32.tflite` — PC benchmarking (float32 I/O)

No float16, dynamic_range, or int16_act variants are generated (no production use).

#### Root Directory Organization
After each training run, `organize_output_folder()` leaves only these files at root:
- `*_integer_quant_uint8.tflite` — deploy this to ESP32
- `*_float32.tflite` — use this for PC benchmarking
- `.keras`, `.onnx` — source artifacts for retraining

Everything else (`.espdl`, `*_integer_quant_float32.tflite`, per-chip variants, legacy names) is moved to `full_models/`.

#### Generation Order (tflite_suite_export)
1. Run integer_quant_float32 FIRST (this creates saved_model/ via onnx2tf)
2. **Immediately** convert saved_model/ → integer_quant_uint8 with `TFLiteConverter.from_saved_model()` + `inference_input_type=tf.uint8`
3. Run the float32 variant (saved_model/ may be overwritten, uint8 is done)
4. `organize_output_folder()` moves non-essential files to full_models/
Calibration uses 500 samples.

#### Benchmarking
- `bench_predict.py` defaults to `--simulate-esp32` ON (real uint8 I/O + sensor noise)
- Model priority in `benchmark/data.py`: `*_integer_quant_uint8.tflite` > `*_integer_quant_float32.tflite` > `*_full_integer_quant.tflite`
- Use `--no-simulate-esp32` for faster PC-only benchmarking

#### Existing Exports
`fix_existing_outputs.py` renames legacy ambiguous filenames and generates uint8 variants on existing exports. Run it once after pulling updates if you have old model directories.

### 7. Advanced Training Routines
- The codebase uses custom loss controllers (`IntelligentFocalLossController`) and dynamic learning rate schedulers to tackle difficult datasets. Updates to the loss or fitting loop should respect these adaptive training routines.
- Data augmentation is configured centrally in `parameters.py`. When adding new augmentation layers, ensure they can be bypassed for standard validation or evaluation flows.
