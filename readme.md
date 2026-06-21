# Digit Recognizer

A deep learning project for rotating digit recognition using various neural network architectures.

## Overview

This project implements and compares different neural network models for recognizing rotating digits from the haverland dataset (repository no longer available as of 20.06.2026). 

The goal is to explore the trade-offs between model complexity, size, and accuracy.

It's actually focused on 10 classes recognitions [0-9] but can works for 100 classes [0-99] And grayscale as it performs the same on test datasets

Grayscale or RGB runs the same on test datasets, but RGB need more resources at image processing level.

## Model Performance

The project explores the relationship between model accuracy and model size across different neural network architectures, demonstrating how different model complexities affect recognition performance.

## Directory Organization

The exported models in the `exported_models/` directory are organized by classification complexity and color space:
- `100cls_GRAY` and `100cls_RGB`: Stress-test models trained on 100 classes (0-99).
- `10cls_GRAY` and `10cls_RGB`: Standard digits models trained on 10 classes (0-9).

## Benchmark on 10 Classes (Standard Digits [0-9]) — TQT RGB Models

The RGB models now use **Trainable Quantization Thresholds (TQT)** for ESP32 deployment. Only three architectures proved significant for RGB: `v16` (maximum accuracy), `v24` (best balance), and `v3` (maximum throughput).

The primary use case remains 10-class recognition. The models perform exceptionally well on grayscale images, offering high accuracy for IoT deployment.

| Model                          | Parameters | Size (KB) | Accuracy (TQT) | Inferences/sec |
| ------------------------------ | ---------- | --------- | -------------- | -------------- |
| digit_recognizer_v16_10cls_RGB | 262500     | 128.1     | 0.995          | 1087           |
| digit_recognizer_v24_10cls_RGB | 158100     | 69.0      | 0.989          | 3404           |
| digit_recognizer_v3_10cls_RGB  | 95300      | 37.4      | 0.980          | 4557           |

### Pareto-Optimal Choices: `v16`, `v24` and `v3`

**For Maximum Accuracy (`v16`):**
1.  **Robust Accuracy**: `v16` achieves **99.5%** TQT accuracy, making it the most accurate quantized model in the repository.
2.  **Stable Quantization**: The gap between float32 (0.995) and uint8 (0.995) is negligible — excellent for real hardware deployment.
3.  **Trade-off**: At **128.1 KB** and **1087 inf/s**, it's the largest and slowest of the three, suited for accuracy-critical applications where memory permits.

**For Best All-Round Balance (`v24`):**
1.  **High Accuracy**: `v24` achieves **98.9%** TQT accuracy — within 0.6% of `v16` while being nearly half the size.
2.  **Compact Footprint**: At only **69.0 KB**, it fits comfortably in constrained ESP32-S3 boards.
3.  **Fast Inference**: Processes **3404 inferences/second**, making it an excellent default choice for most deployments.

**For Maximum Throughput (`v3`):**
1.  **Smallest & Fastest**: At just **37.4 KB** and **4557 inf/s**, `v3` is the speed king.
2.  **Solid Accuracy**: **98.0%** TQT accuracy is still well above the original baseline (96.0%).
3.  **Ultra-Light IoT**: Perfect for memory-constrained ESP32 boards where every KB counts.

## Usage

### Training
To train a model, run:
```bash
python train.py --model digit_recognizer_v4 --classes 10 --color gray
```

#### Configuration Overrides
The training pipeline supports multiple ways to configure hyperparameters without interactive prompts:

1.  **CLI Arguments**: (Highest priority)
    - `--classes {10,100}`: Number of classification classes.
    - `--color {rgb,gray}`: Input image color mode.
    - `--model {architecture}`: Choose from `AVAILABLE_MODELS` in `parameters.py`.
    - `--focal-loss`: Enables Intelligent Focal Loss.

2.  **Environment Variables**:
    - `DIGIT_NB_CLASSES`: Sets the number of classes.
    - `DIGIT_INPUT_CHANNELS`: Sets to `1` (Gray) or `3` (RGB).
    - `DATASET_CACHE_DIR`: Directory for image caching.

3.  **Manual Overrides (parameters.py)**:
    - Set `MANUAL_NB_CLASSES` or `MANUAL_INPUT_CHANNELS` to a fixed value.

#### Non-Interactive Mode
If no configuration is provided and the terminal is detected as non-interactive (e.g., in a CI/CD pipeline or some IDEs), the script will automatically default to **10 classes** and **Grayscale** to avoid blocking.

## Advanced Training Features
- **Intelligent Focal Loss**: Replaces standard Cross-Entropy once the model masters the basics (default: >0.80 accuracy). Controlled via `LOSS_TYPE = "IntelligentFocalLossController"`.
- **Dynamic Alpha Scaling**: Scaling factor $\alpha$ for Focal Loss is automatically calculated based on class count to maintain class balance.
- **Adaptive Per-Class Weighting**: Dynamically adjusts importance of specific difficult digits during training.
- **Dual-Layer Data Augmentation**: Heavy static augmentation (~82k pre-generated images covering rotation, zoom, shift, shear, perspective, flashlight, and more) combined with probability-gated inline augmentation (30% of images re-randomized per epoch to prevent memorization of fixed variants). See [Augmentation Strategy](documentations/augmentation_strategy.md).

## Knowledge Distillation
The project includes a robust distillation framework in `utils/distiller.py`.

- **Accuracy Boost**: Distilled models often exceed their teacher's performance. For example, `distilled_many_to_v16` achieved **99.34%** accuracy, surpassing the standard `v16` (**99.18%**).
- **Logit/Softmax Autodetection**: Automatically handles diverse model output types.
- **Mixed Input Support**: Distill from RGB teachers into efficient Grayscale students.

## Quantization & Deployment

The project supports multiple quantization strategies, all controlled by a single master switch in `config/quantization.py`:

```python
QUANTIZATION_MODE = "tqt"  # Options: "none", "ptq", "qat", "tqt", "auto"
```

Each mode automatically configures all underlying flags (QAT, TQT, ESP-DL, TFLite I/O types, XNNPACK) — no manual flag flipping needed.

| Mode     | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| `none`   | Float32 training & inference. Reference baseline.                           |
| `ptq`    | Standard Post-Training Quantization (TFLite full-integer uint8).            |
| `qat`    | Quantization-Aware Training — must be set **before** training starts.       |
| `tqt`    | **Recommended** — Trainable Quantization Thresholds via ESP-DL pipeline.    |
| `auto`   | Automatically picks TQT for deployable models, float32 for teacher models.  |

### TQT Pipeline (Trainable Quantization Thresholds) — Recommended for ESP32

TQT is a post-training refinement that learns optimal per-channel quantization thresholds instead of using heuristic min/max ranges. It runs after normal training and takes minutes, not hours.

- **No retraining needed**: Tune any pre-trained FP32 model for deployment.
- **Per-chip hyperparameters**: Each target chip has dedicated settings in `config/quantization.py`:

| Chip      | Steps | Learning Rate | Block Size | Integer Lambda |
|-----------|-------|---------------|------------|----------------|
| `esp32`   | 200   | 1e-6          | 2          | 0.10           |
| `esp32s3` | 200   | 1e-6          | 2          | 0.05           |
| `esp32p4` | 200   | 1e-6          | 2          | 0.0            |

- **Calibration**: 300 steps at batch size 1 over 22,000 samples to estimate activation ranges.
- **Multi-chip export**: Set `TQT_EXPORT_ALL_TARGETS = True` to produce optimized models for all three targets in a single run.

#### Output Artifacts

The TQT pipeline produces 3 TFLite files per model:

| File | Purpose |
|------|---------|
| `*_quantized_integer_quant_float32.tflite` | Internal intermediate (creates saved_model/ for uint8 conversion) |
| `*_quantized_integer_quant_uint8.tflite`   | **ESP32 TFLite Micro** — uint8 I/O, TQT-quality weights |
| `*_quantized_float32.tflite`               | PC benchmarking — float32 I/O |

Root directory stores the two deployment-ready files: `*_integer_quant_uint8.tflite` and `*_float32.tflite`. All other pipeline artifacts go to `full_models/`.

#### uint8 I/O Contract

TFLite Micro on ESP32 expects raw camera bytes in `[0, 255]` (uint8). The pipeline defaults to `USE_TFLITE_BUILTINS_UINT8_ONLY = True`. If targeting the ESP-DL SDK (which expects int8), set `USE_TFLITE_BUILTINS_INT8_ONLY = True` instead — the model will require pixel conversion in C++ code (`pixel_int8 = pixel_uint8 - 128`).

#### Quick Workflow

1. Set `QUANTIZATION_MODE = "tqt"` in `config/quantization.py` (already the default).
2. Train normally: `python train.py --model digit_recognizer_v4 --classes 10 --color gray`
3. Quantize & export: `python quantize_espdl.py --keras path/to/model.keras --target esp32`
4. Deploy the resulting `*_integer_quant_uint8.tflite` to your ESP32.

### Notes on TFLite Micro & ESP-DL Compatibility

- Certain ops (e.g., `relu6`) are suppressed for ESP-DL compatibilty.
- TFLite output size roughly doubles the tensor arena needed in inference RAM.
- XNNPACK delegate is automatically disabled (`DISABLE_XNNPACK = True`) for TFLite Micro compatibility.
- RGB vs grayscale: Conv2D flattens all input channels, so the kernel weights are comparable. However, RGB triples the pre-processing time and input tensor memory on IoT devices. Grayscale is strongly recommended for constrained microcontrollers.

## Documentation

For detailed guides analyzing how to train, benchmark, and debug the models within this repository, refer to the guides in the [`documentations/`](documentations/) folder:
- [Training Guide](documentations/training_guide.md)
- [Benchmarking & Prediction](documentations/benchmarking_and_prediction.md)
- [Analysis & Debugging](documentations/analysis_and_debugging.md)
- [Augmentation Strategy](documentations/augmentation_strategy.md)
- [TFLite Quantization Guide](documentations/TFLite_Quantization_Guide.md)

## RGB vs Grayscale Model Comparison

 If the model uses Conv2D (first layer), it flattens all channels, which is why the theoretical benchmark results (accuracy and raw inference speed) are very similar between RGB and grayscale variants. 

 However, **RGB increases the pre-processing time in IoT devices by a significant factor**. The image processing pipeline must handle 3 times as much data (e.g., resizing and normalizing channels). Furthermore, the input tensor memory footprint (`H, W, 3`) takes substantially more RAM before inference even begins. Therefore, Grayscale (`10cls_GRAY`) is highly recommended for constrained microcontrollers.
 
 ##### How Conv2D Actually Works with Channels:
-   **Input**: `(height, width, channels)` - e.g., `(32, 20, 1)` for grayscale or `(32, 20, 3)` for RGB
-   **Conv2D with 32 filters**: Each filter has shape `(3, 3, channels)` and produces 1 output channel
-   **Output**: `(height, width, 32)` - 32 feature maps, each combining information from all input channels

## Summary of Model Choice and Application

The table below summarizes the trade-offs between accuracy and model size across different classification complexities (10 vs 100 classes) and color spaces (Grayscale vs RGB). It highlights the most notable models to help you choose the best fit for your specific IoT application.

> **RGB columns now reflect TQT-quantized models** — only v3, v16, and v24 proved significant for RGB deployment. Other architectures are not shown for RGB.

| Model Name | 10 cls Gray | 10 cls RGB (TQT) | 100 cls Gray | 100 cls RGB | Application / Comment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v6** | 96.5% / 36.5KB | — | 84.0% / 132.5KB | 89.5% / 160.8KB | Best balanced IoT model for 10cls Gray, extremely small memory footprint. |
| **v3** | 98.0% / 69.4KB | **98.0% / 37.4KB** | 76.5% / 74.6KB | 82.5% / 45.1KB | Fastest TQT RGB model (4557 inf/s) and smallest TQT footprint. |
| **v7** | 96.6% / 46.7KB | — | 75.4% / 55.5KB | 79.7% / 56.0KB | Fastest inference speed under 100KB, optimal for speed-critical IoT. |
| **v4** | 98.5% / 61.4KB | — | 82.9% / 69.5KB | 90.7% / 87.1KB | Excellent accuracy while remaining under 100KB, great all-rounder. |
| **v15** | 99.1% / 79.3KB | — | 85.4% / 86.0KB | 90.5% / 107.4KB | Best accuracy for models under 100KB in 10-class scenarios. |
| **v17** | 99.0% / 70.7KB | — | 82.6% / 80.2KB | 87.5% / 80.5KB | Ultra-efficient GhostNet-inspired alternative with solid accuracy. |
| **v16** | 99.2% / 128.6KB | **99.5% / 128.1KB** | 88.4% / 139.5KB | 93.4% / 139.7KB | Highest accuracy TQT RGB model, excellent under stress. |
| **v18** | 98.9% / 97.1KB | — | 90.2% / 109.4KB | 89.4% / 109.7KB | New variant with very strong performance hovering around 100KB. |
| **v19** | 98.9% / 131.9KB | — | 91.6% / 145.6KB | 91.4% / 146.0KB | New high-capacity variant built for challenging 100-class scenarios. |
| **v24** | — | **98.9% / 69.0KB** | — | — | Best all-round TQT RGB model: near-v16 accuracy at half the size. |
| **original_haverland** | 98.2% / 203.3KB | — | 81.7% / 228.2KB | 83.8% / 228.8KB | Legacy baseline, superseded by v16 and newer variants. |

## Related Projects

This work contributes to improved digit recognition research, including the Tenth-of-step-of-a-meter-digit project (repository no longer available as of 20.06.2026) for enhanced meter digit analysis.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions.

## License
This project is licensed under the Apache-2.0 license