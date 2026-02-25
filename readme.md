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

## Benchmark on 24351 real images (100 Classes [0-99])

To ensure a fair and comprehensive comparison between architectures, the following benchmark utilizes the 100 classes (`0` to `99`) dataset, which represents a significantly harder classification task than the simple 0-9 digits.

| Model                          | Parameters | Size (KB) | Accuracy | Inferences/sec |
| ------------------------------ | ---------- | --------- | -------- | -------------- |
| digit_recognizer_v12_100cls_GRAY| 496100     | 414.8     | 0.9149   | 2039           |
| digit_recognizer_v9_100cls_GRAY | 911500     | 159.5     | 0.8734   | 3145           |
| digit_recognizer_v6_100cls_GRAY | 171800     | 132.5     | 0.8622   | 3343           |
| digit_recognizer_v4_100cls_GRAY | 85800      | 69.5      | 0.8558   | 8003           |
| original_haverland_100cls_GRAY  | 257899     | 228.2     | 0.8435   | 6220           |
| mnist_quantization_100cls_GRAY  | 104800     | 71.7      | 0.8159   | 6400           |
| digit_recognizer_v3_100cls_GRAY | 121600     | 74.6      | 0.7873   | 8331           |
| digit_recognizer_v7_100cls_GRAY | 82400      | 55.5      | 0.7807   | 9270           |

### Pareto-Optimal Choice: `v4` vs `original_haverland`

When comparing the `original_haverland` baseline to `digit_recognizer_v4` on the 100 classes dataset, `v4` demonstrates strict superiority across all key edge-deployment metrics:

1.  **Higher Accuracy**: `v4` achieves **85.58%** accuracy compared to the original's **84.35%**.
2.  **Dramatically Smaller Memory Footprint**: `v4` is only **69.5 KB**, making it roughly **3.3x smaller** than the original's **228.2 KB**, saving critical flash memory on ESP32 devices.
3.  **Faster Inference**: `v4` processes **8003 inferences/second**, making it nearly **30% faster** than the original's **6220 inferences/second**.
4.  **Fewer Parameters**: `v4` utilizes nearly 3 times fewer parameters (85k vs 257k), directly reducing the required RAM tensor arena size.

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
 
 ## RGB - Grayscale model comparison : 
 if the model use Conv2D (fast all), it flatten all channel, that's why the result are very the same between RGB and grayscale
 
 ##### How Conv2D Actually Works with Channels:
-   **Input**: `(height, width, channels)` - e.g., `(32, 20, 1)` for grayscale or `(32, 20, 3)` for RGB
-   **Conv2D with 32 filters**: Each filter has shape `(3, 3, channels)` and produces 1 output channel
-   **Output**: `(height, width, 32)` - 32 feature maps, each combining information from all input channels

## Related Projects

This work contributes to improved digit recognition research, including the [Tenth-of-step-of-a-meter-digit](https://github.com/haverland/Tenth-of-step-of-a-meter-digit) project for enhanced meter digit analysis.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions.

## License
This project is licensed under the Apache-2.0 license


