# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-07 20:47:48
- **Models Tested**: 11 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (45.1KB, 0.850 acc, 5751 inf/s)
- **Best Accuracy**: **high_accuracy_validator.tflite** (0.939)
- **Fastest Model**: **digit_recognizer_v7.tflite** (8483 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (45.1 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 45.1 | 0.850 | 5751 | 75900 | 0.888 |
| digit_recognizer_v7.tflite | 56.0 | 0.831 | 8483 | 85500 | 0.884 |
| digit_recognizer_v17.tflite | 80.5 | 0.896 | 6742 | 183300 | 0.804 |
| mnist_quantization.tflite | 72.2 | 0.867 | 5705 | 108000 | 0.783 |
| digit_recognizer_v4.tflite | 87.1 | 0.897 | 6201 | 111500 | 0.779 |
| digit_recognizer_v15.tflite | 107.4 | 0.896 | 5685 | 145400 | 0.737 |
| digit_recognizer_v16.tflite | 139.7 | 0.933 | 4670 | 255800 | 0.704 |
| original_haverland.tflite | 228.8 | 0.883 | 5716 | 263600 | 0.664 |
| digit_recognizer_v6.tflite | 160.8 | 0.896 | 2441 | 209300 | 0.619 |
| digit_recognizer_v12.tflite | 415.4 | 0.927 | 1610 | 499300 | 0.564 |
| high_accuracy_validator.tflite | 3149.7 | 0.939 | 801 | 3300000 | 0.523 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.888
- **Accuracy**: 0.850
- **Size**: 45.1 KB
- **Speed**: 5751 inf/s
- **Efficiency**: 0.0188 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.850 | 45.1KB | 5751/s | 0.888 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.831 | 56.0KB | 8483/s | 0.884 | ⚡ Fastest |
| digit_recognizer_v17.tflite | 0.896 | 80.5KB | 6742/s | 0.804 | Alternative |
| mnist_quantization.tflite | 0.867 | 72.2KB | 5705/s | 0.783 | Alternative |
| digit_recognizer_v4.tflite | 0.897 | 87.1KB | 6201/s | 0.779 | 🎯 Best Accuracy |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.897 (best under 100KB)
- **Trade-off**: 87.1KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 8483 inf/s (fastest under 100KB)
- **Trade-off**: 0.831 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v16.tflite
- **Size**: 139.7KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.933 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0188 | 11.194993412384717 | 75900 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0148 | 9.722807017543861 | 85500 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0111 | 4.889252591380251 | 183300 | ⚖️ Good |
| mnist_quantization.tflite | 0.0120 | 8.024074074074074 | 108000 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0103 | 8.047533632286996 | 111500 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
