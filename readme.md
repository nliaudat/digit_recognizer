# Digit Recognizer

A deep learning project for rotating digit recognition using various neural network architectures.

## Overview

This project implements and compares different neural network models for recognizing rotating digits from [haverland dataset](https://github.com/haverland/Tenth-of-step-of-a-meter-digit). 

The goal is to explore the trade-offs between model complexity, size, and accuracy.

It's actually focused on 10 classes recognitions [0-9] but can works for 100 classes [0-99] And grayscale as it performs the same on test datasets

Grayscale or RGB runs the same on test datasets, but RGB need more resources at image processing level.

## Model Performance

![Accuracy vs Model Size](imgs/accuracy_vs_size.png)

The graph above shows the relationship between model accuracy and model size across different neural network architectures. As demonstrated, the project explores how different model complexities affect recognition performance.

## Directory Organization

The exported models in the `exported_models/` directory are organized by classification complexity and color space:
- `100cls_GRAY` and `100cls_RGB`: Stress-test models trained on 100 classes (0-99).
- `10cls_GRAY` and `10cls_RGB`: Standard digits models trained on 10 classes (0-9).

## Benchmark on 10 Classes (Standard Digits [0-9])

The primary use case is 10-class recognition. The models perform exceptionally well on grayscale images, offering high accuracy for IoT deployment.

| Model                          | Parameters | Size (KB) | Accuracy | Inferences/sec |
| ------------------------------ | ---------- | --------- | -------- | -------------- |
| digit_recognizer_v12_10cls_GRAY| 490000     | 406.7     | 0.993    | 1570           |
| original_haverland_10cls_GRAY  | 234500     | 203.3     | 0.982    | 4513           |
| digit_recognizer_v16_10cls_GRAY| 244000     | 128.6     | 0.992    | 3690           |
| digit_recognizer_v15_10cls_GRAY| 109100     | 79.3      | 0.991    | 5184           |
| digit_recognizer_v17_10cls_GRAY| 172900     | 70.7      | 0.990    | 4971           |
| digit_recognizer_v3_10cls_GRAY | 118400     | 69.4      | 0.980    | 6359           |
| mnist_quantization_10cls_GRAY  | 98700      | 63.6      | 0.970    | 4719           |
| digit_recognizer_v4_10cls_GRAY | 79700      | 61.4      | 0.985    | 5834           |
| digit_recognizer_v7_10cls_GRAY | 75600      | 46.7      | 0.966    | 6485           |
| digit_recognizer_v6_10cls_GRAY | 61500      | 36.5      | 0.965    | 5167           |

### Pareto-Optimal Choice: `v4` vs `original_haverland`

When comparing the `original_haverland` baseline to `digit_recognizer_v4` on the 10-class dataset, `v4` demonstrates strict superiority across all key edge-deployment metrics:

1.  **Higher Accuracy**: `v4` achieves **98.5%** accuracy compared to the original's **98.2%**.
2.  **Dramatically Smaller Memory Footprint**: `v4` is only **61.4 KB**, making it roughly **3.3x smaller** than the original's **203.3 KB**, saving critical flash memory on ESP32 devices.
3.  **Faster Inference**: `v4` processes **5834 inferences/second**, making it nearly **30% faster** than the original's **4513 inferences/second**.
4.  **Fewer Parameters**: `v4` utilizes nearly 3 times fewer parameters (79k vs 234k), directly reducing the required RAM tensor arena size.

## Benchmark on 24351 real images (100 Classes [0-99])

To ensure a fair and comprehensive comparison between architectures under stress, the following benchmark utilizes the 100 classes (`0` to `99`) dataset, which represents a significantly harder classification task than the simple 0-9 digits.

| Model                          | Parameters | Size (KB) | Accuracy | Inferences/sec |
| ------------------------------ | ---------- | --------- | -------- | -------------- |
| digit_recognizer_v12_100cls_GRAY| 496100     | 414.8     | 0.892    | 1459           |
| original_haverland_100cls_GRAY  | 257899     | 228.2     | 0.817    | 4395           |
| digit_recognizer_v16_100cls_GRAY| 253000     | 139.5     | 0.884    | 3507           |
| digit_recognizer_v6_100cls_GRAY | 171800     | 132.5     | 0.840    | 2437           |
| digit_recognizer_v15_100cls_GRAY| 113700     | 86.0      | 0.854    | 4595           |
| digit_recognizer_v17_100cls_GRAY| 180400     | 80.2      | 0.826    | 4730           |
| digit_recognizer_v3_100cls_GRAY | 121600     | 74.6      | 0.765    | 6215           |
| mnist_quantization_100cls_GRAY  | 104800     | 71.7      | 0.792    | 4497           |
| digit_recognizer_v4_100cls_GRAY | 85800      | 69.5      | 0.829    | 5759           |
| digit_recognizer_v7_100cls_GRAY | 82400      | 55.5      | 0.754    | 6692           |

### Performance under Stress: `v4` vs `original_haverland` (100-Class)

Even on the harder dataset, `v4` maintains its sheer edge over `original_haverland`:

1.  **Higher Accuracy**: `v4` achieves **82.9%** accuracy compared to the original's **81.7%**.
2.  **Dramatically Smaller Memory Footprint**: `v4` drops to **69.5 KB** while the original is **228.2 KB**.
3.  **Faster Inference**: `v4` processes **5759 inferences/second**, beating the original's **4395 inferences/second**.
4.  **Fewer Parameters**: `v4` utilizes nearly 3 times fewer parameters (85k vs 257k).

## Benchmarking

To run the benchmarking suite across all available models (excluding large PC-only validators by default):
```bash
python bench_predict.py --test_all
```

You can exclude specific models from the benchmark:
```bash
python bench_predict.py --exclude_model some_other_model_name
```

## Documentation

For detailed guides analyzing how to train, benchmark, and debug the models within this repository, refer to the guides in the [`documentations/`](documentations/) folder:
- [Training Guide](documentations/training_guide.md)
- [Benchmarking & Prediction](documentations/benchmarking_and_prediction.md)
- [Analysis & Debugging](documentations/analysis_and_debugging.md)

## Features

- Multiple neural network architectures
- Model size vs accuracy analysis
- Training and evaluation pipelines
- Visualization tools
- Pre-trained models

## Results

The project demonstrates that :

 - Using training [Quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training) can lead to better results
 - tlite-micro or esp-dl compabilities suppress model operators (like relu6)
 - There is major differences between training efficiency and real tests
 - Model size "double" the tensor arena needed in memory
 - CPU operations must be also taken into parameters for IOT
 - RGB or grayscale has very same benchmark results, but the processing is not the same as in parameters needed. It also needs a lot of more cpu and memory to process
 
 ## RGB vs Grayscale Model Comparison

 If the model uses Conv2D (first layer), it flattens all channels, which is why the theoretical benchmark results (accuracy and raw inference speed) are very similar between RGB and grayscale variants. 

 However, **RGB increases the pre-processing time in IoT devices by a significant factor**. The image processing pipeline must handle 3 times as much data (e.g., resizing and normalizing channels). Furthermore, the input tensor memory footprint (`H, W, 3`) takes substantially more RAM before inference even begins. Therefore, Grayscale (`10cls_GRAY`) is highly recommended for constrained microcontrollers.
 
 ##### How Conv2D Actually Works with Channels:
-   **Input**: `(height, width, channels)` - e.g., `(32, 20, 1)` for grayscale or `(32, 20, 3)` for RGB
-   **Conv2D with 32 filters**: Each filter has shape `(3, 3, channels)` and produces 1 output channel
-   **Output**: `(height, width, 32)` - 32 feature maps, each combining information from all input channels

## Summary of Model Choice and Application

The table below summarizes the trade-offs between accuracy and model size across different classification complexities (10 vs 100 classes) and color spaces (Grayscale vs RGB). It highlights the most notable models to help you choose the best fit for your specific IoT application.

| Model Name | 10 cls Gray | 10 cls RGB | 100 cls Gray | 100 cls RGB | Application / Comment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v6** | 96.5% / 36.5KB | 96.1% / 46.9KB | 84.0% / 132.5KB | 85.7% / 160.8KB | Best balanced IoT model for 10cls Gray, extremely small memory footprint. |
| **v3** | 98.0% / 69.4KB | 97.7% / 38.4KB | 76.5% / 74.6KB | 78.9% / 45.1KB | Fastest overall inference speed and best balanced for 100cls RGB. |
| **v7** | 96.6% / 46.7KB | 96.7% / 47.2KB | 75.4% / 55.5KB | 75.0% / 56.0KB | Fastest inference speed under 100KB, optimal for speed-critical IoT. |
| **v4** | 98.5% / 61.4KB | 98.7% / 78.3KB | 82.9% / 69.5KB | 86.1% / 87.1KB | Excellent accuracy while remaining under 100KB, great all-rounder. |
| **mnist_quantization** | 97.0% / 63.6KB | 97.0% / 64.2KB | 79.2% / 71.7KB | 81.2% / 72.2KB | Standard quantization baseline model. |
| **v15** | 99.1% / 79.3KB | 99.2% / 100.0KB | 85.4% / 86.0KB | 84.9% / 107.4KB | Best accuracy for models under 100KB in 10-class scenarios. |
| **v17** | 99.0% / 70.7KB | 98.8% / 71.0KB | 82.6% / 80.2KB | 82.2% / 80.5KB | Ultra-efficient GhostNet-inspired alternative with solid accuracy. |
| **v16** | 99.2% / 128.6KB | 99.2% / 128.8KB | 88.4% / 139.5KB | 89.5% / 139.7KB | High accuracy MobileNetV2-based model, but larger footprint. |
| **original_haverland** | 98.2% / 203.3KB | 98.2% / 203.8KB | 81.7% / 228.2KB | 82.0% / 228.8KB | Legacy baseline, superseded by v4 and newer variants. |
| **v12** | 99.3% / 406.7KB | 99.3% / 407.3KB | 89.2% / 414.8KB | 89.3% / 415.4KB | Best overall absolute accuracy under 1MB. Not suitable for constrained IoT. |
| **high_accuracy_validator** | N/A | N/A | N/A | 91.9% / 3149.7KB | PC-only large model validator, highest absolute accuracy found. |

## Related Projects

This work contributes to improved digit recognition research, including the [Tenth-of-step-of-a-meter-digit](https://github.com/haverland/Tenth-of-step-of-a-meter-digit) project for enhanced meter digit analysis.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions.

## License
This project is licensed under the Apache-2.0 license
