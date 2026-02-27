# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-02-27 14:55:20
- **Models Tested**: 12 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (45.1KB, 0.789 acc, 1796 inf/s)
- **Best Accuracy**: **high_accuracy_validator.tflite** (0.919)
- **Fastest Model**: **mnist_quantization.tflite** (3385 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (45.1 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 45.1 | 0.789 | 1796 | 75900 | 0.835 |
| mnist_quantization.tflite | 72.2 | 0.812 | 3385 | 108000 | 0.829 |
| digit_recognizer_v7.tflite | 56.0 | 0.750 | 2279 | 85500 | 0.784 |
| digit_recognizer_v4.tflite | 87.1 | 0.861 | 2120 | 111500 | 0.749 |
| digit_recognizer_v17.tflite | 80.5 | 0.822 | 1693 | 183300 | 0.715 |
| original_haverland.tflite | 228.8 | 0.820 | 3215 | 263600 | 0.695 |
| digit_recognizer_v15.tflite | 107.4 | 0.849 | 1589 | 145400 | 0.682 |
| digit_recognizer_v16.tflite | 139.7 | 0.895 | 1478 | 255800 | 0.671 |
| digit_recognizer_v9.tflite | 160.0 | 0.862 | 905 | 914600 | 0.607 |
| digit_recognizer_v6.tflite | 160.8 | 0.857 | 736 | 209300 | 0.594 |
| digit_recognizer_v12.tflite | 415.4 | 0.893 | 583 | 499300 | 0.553 |
| high_accuracy_validator.tflite | 3149.7 | 0.919 | 255 | 3300000 | 0.519 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.835
- **Accuracy**: 0.789
- **Size**: 45.1 KB
- **Speed**: 1796 inf/s
- **Efficiency**: 0.0175 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.789 | 45.1KB | 1796/s | 0.835 | 🏆 **BEST BALANCED** |
| mnist_quantization.tflite | 0.812 | 72.2KB | 3385/s | 0.829 | ⚡ Fastest |
| digit_recognizer_v7.tflite | 0.750 | 56.0KB | 2279/s | 0.784 | Alternative |
| digit_recognizer_v4.tflite | 0.861 | 87.1KB | 2120/s | 0.749 | 🎯 Best Accuracy |
| digit_recognizer_v17.tflite | 0.822 | 80.5KB | 1693/s | 0.715 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.861 (best under 100KB)
- **Trade-off**: 87.1KB size

**For Speed-Critical IoT:**
- **Choice**: mnist_quantization.tflite
- **Speed**: 3385 inf/s (fastest under 100KB)
- **Trade-off**: 0.812 accuracy

**For Memory-Constrained IoT:**
- **Choice**: high_accuracy_validator.tflite
- **Size**: 3149.7KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.919 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0175 | 10.392621870882738 | 75900 | 🎯 **OPTIMAL** |
| mnist_quantization.tflite | 0.0112 | 7.516666666666666 | 108000 | ⚖️ Good |
| digit_recognizer_v7.tflite | 0.0134 | 8.76608187134503 | 85500 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0099 | 7.7192825112107615 | 111500 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0102 | 4.486633933442444 | 183300 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
