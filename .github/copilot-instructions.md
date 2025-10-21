# Copilot Instructions for Digit Recognizer

## Project Architecture
- **Purpose:** Deep learning for rotating digit recognition (10 or 100 classes) using multiple neural network architectures. Focus on model size, accuracy, and deployment for resource-constrained environments (IoT, ESP-DL, TensorFlow Lite Micro).
- **Major Components:**
  - `models/`: Individual model architectures. Use `model_factory.py` for dynamic model creation based on `parameters.py`.
  - `parameters.py`: Centralized hyperparameter and configuration management. Controls model selection, input shape, quantization, etc.
  - `utils/`: Data pipeline (`data_pipeline.py`), preprocessing (`preprocess.py`), and multi-source loading.
  - `datasets/`: Contains raw and augmented digit datasets (10/100 classes, RGB/grayscale).
  - Main scripts: `train.py`, `predict.py`, `fine-tune.py`, `bench_predict.py`, etc.

## Key Workflows
- **Training:**
  - Run `train.py` with arguments (see `parse_arguments`).
  - Model selection and hyperparameters are set in `parameters.py`.
  - Quantization-aware training (QAT) is supported if `tensorflow-model-optimization` is installed.
  - Data is preprocessed and split using `utils/preprocess.py` and `utils/data_pipeline.py`.
- **Model Creation:**
  - Use `model_factory.create_model()`; model name must be in `parameters.AVAILABLE_MODELS`.
  - Models are dynamically imported from `models/`.
- **Data Pipeline:**
  - Use `create_tf_dataset_from_arrays` for preprocessed data; no extra normalization in pipeline.
  - Data splits are managed via `get_tf_data_splits_from_arrays`.
- **Quantization & Deployment:**
  - QAT and post-training quantization are controlled by flags in `parameters.py`.
  - ESP-DL compatibility disables some operators (e.g., relu6).

## Project-Specific Patterns
- **Parameter-driven:** All major settings (model, input shape, quantization) are set in `parameters.py` and imported everywhere.
- **Dynamic Model Import:** Models are loaded by name; add new models to `models/` and `AVAILABLE_MODELS`.
- **Preprocessing:** Images are normalized to `[0, 1]` (float32) or `[0, 255]` (uint8) depending on quantization settings.
- **No Redundant Preprocessing:** Data pipeline expects already preprocessed arrays.
- **Debugging:** Use `--debug` flag in scripts for verbose TensorFlow logs.

## External Dependencies
- **TensorFlow** (core and model optimization)
- **tqdm, numpy, matplotlib, cv2**
- Optional: `onnx`, `tf2onnx` for ONNX export

## Examples
- To train with QAT: Set `USE_QAT=True` and `QUANTIZE_MODEL=True` in `parameters.py`, ensure `tensorflow-model-optimization` is installed.
- To switch models: Change `MODEL_ARCHITECTURE` in `parameters.py` to a valid name from `AVAILABLE_MODELS`.
- To preprocess new data: Use `preprocess_images` in `utils/preprocess.py`.

## Conventions
- All new models must have a `create_<modelname>()` function in their file.
- All scripts import parameters from `parameters.py`.
- Data pipeline expects arrays already normalized and shaped.

## References
- See `README.md` for project overview and dataset details.
- See `parameters.py` for all configuration options.
- See `models/model_factory.py` for model loading logic.
- See `utils/preprocess.py` for preprocessing and quantization logic.

---
**Feedback:** If any section is unclear or missing, please specify which workflows, conventions, or architectural details need more coverage.
