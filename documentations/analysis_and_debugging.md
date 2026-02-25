# Analysis and Debugging Guide

This repository contains dedicated scripts to analyze model weights, evaluate quantization integrity, diagnose dataset discrepancies, and debug layer activations.

## 1. Quantization Analysis (`run_quant_analysis.py`)
This script executes the comprehensive `QuantizationAnalyzer` over the latest trained model. It's meant to bridge the gap between Keras float precision and the resulting integer distributions.

### What it does
- Automatically locates the most recent training directory in `exported_models/`.
- Loads *both* the unquantized Keras `best_model.keras` weights and the quantized `.tflite` model.
- Passes a sample of the test dataset through both architectures simultaneously.

### Outputs
It calculates the absolute differences and produces an HTML/Markdown report documenting:
- Weight distributions (how weights were clamped).
- Activation shifts (how intermediate values shifted post-quantization).
- Signal-to-Noise Ratio (SNR) degradation.
- Confusion matrix differences between Float and INT8.

### Usage
```bash
python run_quant_analysis.py
```
*(The `--debug` flag prints additional verbose checks).*

## 2. Training Diagnostics (`diagnose_training.py`)
If your model is diverging or suddenly predicting only a single class, `diagnose_training.py` provides targeted analytics on your data pipeline and early-stop logic.

It checks for:
- **Dataset Structure:** Verifies folder structures, image counts per class, and reports critical class imbalances.
- **Data Loading:** Hooks into `load_combined_dataset()` to ensure the arrays are genuinely containing unique labels and varied matrices.
- **Preprocessing Checks:** Creates synthetic test images and passes them through `preprocess_single_image()` to track bounding-box or thresholding manipulations.

### Usage
*(Intended to be executed directly or imported periodically within Jupyter environments).*
```bash
python diagnose_training.py
```

## 3. General Architecture Debugging (`debug.py`)
A comprehensive playground script for invoking specific diagnostic utilities found in `utils.train_analyse` without needing to trigger a full training epoch. It provides functions to hook directly into layers and observe outputs, acting as a sandbox for inspecting Keras graphs.
