# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-07 20:14:01
- **Models Tested**: 10 quantized models
- **Best IoT Model**: **digit_recognizer_v7.tflite** (47.2KB, 0.956 acc, 8184 inf/s)
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.989)
- **Fastest Model**: **digit_recognizer_v7.tflite** (8184 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (38.4 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v7.tflite | 47.2 | 0.956 | 8184 | 78600 | 0.927 |
| digit_recognizer_v3.tflite | 38.4 | 0.958 | 5280 | 71200 | 0.914 |
| digit_recognizer_v6.tflite | 46.9 | 0.950 | 4433 | 79500 | 0.834 |
| mnist_quantization.tflite | 64.2 | 0.960 | 5783 | 101900 | 0.806 |
| digit_recognizer_v4.tflite | 78.3 | 0.968 | 6777 | 104700 | 0.802 |
| digit_recognizer_v17.tflite | 71.0 | 0.988 | 5277 | 175700 | 0.791 |
| digit_recognizer_v15.tflite | 100.0 | 0.986 | 4787 | 140000 | 0.731 |
| digit_recognizer_v16.tflite | 128.8 | 0.986 | 4169 | 246800 | 0.690 |
| original_haverland.tflite | 203.8 | 0.975 | 5609 | 240200 | 0.687 |
| digit_recognizer_v12.tflite | 407.3 | 0.989 | 1764 | 493100 | 0.571 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v7.tflite**
- **IoT Score**: 0.927
- **Accuracy**: 0.956
- **Size**: 47.2 KB
- **Speed**: 8184 inf/s
- **Efficiency**: 0.0203 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v7.tflite | 0.956 | 47.2KB | 8184/s | 0.927 | 🏆 **BEST BALANCED** |
| digit_recognizer_v3.tflite | 0.958 | 38.4KB | 5280/s | 0.914 | 💾 Smallest Adequate |
| digit_recognizer_v6.tflite | 0.950 | 46.9KB | 4433/s | 0.834 | Alternative |
| mnist_quantization.tflite | 0.960 | 64.2KB | 5783/s | 0.806 | Alternative |
| digit_recognizer_v4.tflite | 0.968 | 78.3KB | 6777/s | 0.802 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v17.tflite
- **Accuracy**: 0.988 (best under 100KB)
- **Trade-off**: 71.0KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 8184 inf/s (fastest under 100KB)
- **Trade-off**: 0.956 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v3.tflite
- **Size**: 38.4KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.958 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v7.tflite | 0.0203 | 12.162849872773537 | 78600 | 🎯 **OPTIMAL** |
| digit_recognizer_v3.tflite | 0.0250 | 13.456460674157304 | 71200 | ⚖️ Good |
| digit_recognizer_v6.tflite | 0.0203 | 11.948427672955976 | 79500 | ⚖️ Good |
| mnist_quantization.tflite | 0.0149 | 9.418056918547595 | 101900 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0124 | 9.245463228271252 | 104700 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
