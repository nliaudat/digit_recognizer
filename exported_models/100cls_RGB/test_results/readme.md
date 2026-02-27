# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-02-27 08:34:09
- **Models Tested**: 11 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (45.1KB, 0.790 acc, 5187 inf/s)
- **Best Accuracy**: **digit_recognizer_v16.tflite** (0.894)
- **Fastest Model**: **digit_recognizer_v7.tflite** (6888 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (45.1 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 45.1 | 0.790 | 5187 | 75900 | 0.893 |
| digit_recognizer_v7.tflite | 56.0 | 0.752 | 6888 | 85500 | 0.862 |
| digit_recognizer_v4.tflite | 87.1 | 0.861 | 6649 | 111500 | 0.830 |
| mnist_quantization.tflite | 72.2 | 0.814 | 4635 | 108000 | 0.777 |
| digit_recognizer_v15.tflite | 107.4 | 0.848 | 3846 | 145400 | 0.712 |
| digit_recognizer_v16.tflite | 139.7 | 0.894 | 3831 | 255800 | 0.708 |
| original_haverland.tflite | 228.8 | 0.821 | 4331 | 263600 | 0.644 |
| digit_recognizer_v9.tflite | 160.0 | 0.863 | 2404 | 914600 | 0.637 |
| digit_recognizer_v6.tflite | 160.8 | 0.857 | 2093 | 209300 | 0.624 |
| digit_recognizer_v12.tflite | 415.4 | 0.893 | 1405 | 499300 | 0.573 |
| digit_recognizer_v17.tflite | 80.5 | 0.071 | 5264 | 183300 | 0.361 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.893
- **Accuracy**: 0.790
- **Size**: 45.1 KB
- **Speed**: 5187 inf/s
- **Efficiency**: 0.0175 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.790 | 45.1KB | 5187/s | 0.893 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.752 | 56.0KB | 6888/s | 0.862 | ⚡ Fastest |
| digit_recognizer_v4.tflite | 0.861 | 87.1KB | 6649/s | 0.830 | 🎯 Best Accuracy |
| mnist_quantization.tflite | 0.814 | 72.2KB | 4635/s | 0.777 | Alternative |
| digit_recognizer_v17.tflite | 0.071 | 80.5KB | 5264/s | 0.361 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.861 (best under 100KB)
- **Trade-off**: 87.1KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 6888 inf/s (fastest under 100KB)
- **Trade-off**: 0.752 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v16.tflite
- **Size**: 139.7KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.894 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0175 | 10.412384716732543 | 75900 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0134 | 8.790643274853801 | 85500 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0099 | 7.721973094170403 | 111500 | ⚖️ Good |
| mnist_quantization.tflite | 0.0113 | 7.532407407407407 | 108000 | ⚖️ Good |
| digit_recognizer_v15.tflite | 0.0079 | 5.834938101788171 | 145400 | ❌ Too large |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
