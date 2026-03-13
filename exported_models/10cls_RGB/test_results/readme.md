# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-13 14:39:48
- **Models Tested**: 11 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (38.4KB, 0.973 acc, 4695 inf/s)
- **Best Accuracy**: **digit_recognizer_v16.tflite** (0.992)
- **Fastest Model**: **digit_recognizer_v7.tflite** (6423 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (38.4 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 38.4 | 0.973 | 4695 | 71200 | 0.937 |
| digit_recognizer_v7.tflite | 47.2 | 0.970 | 6423 | 78600 | 0.933 |
| digit_recognizer_v6.tflite | 46.9 | 0.963 | 4047 | 79500 | 0.857 |
| digit_recognizer_v17.tflite | 71.0 | 0.987 | 6256 | 175700 | 0.854 |
| digit_recognizer_v4.tflite | 78.3 | 0.988 | 6121 | 104700 | 0.836 |
| digit_recognizer_v15.tflite | 100.0 | 0.989 | 5506 | 140000 | 0.785 |
| digit_recognizer_v18.tflite | 97.4 | 0.990 | 4131 | 213000 | 0.746 |
| digit_recognizer_v16.tflite | 128.8 | 0.992 | 4960 | 246800 | 0.744 |
| digit_recognizer_v19.tflite | 132.2 | 0.988 | 3286 | 287200 | 0.687 |
| original_haverland.tflite | 203.8 | 0.966 | 4286 | 240200 | 0.677 |
| digit_recognizer_v12.tflite | 407.3 | 0.983 | 1942 | 493100 | 0.584 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.937
- **Accuracy**: 0.973
- **Size**: 38.4 KB
- **Speed**: 4695 inf/s
- **Efficiency**: 0.0253 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.973 | 38.4KB | 4695/s | 0.937 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.970 | 47.2KB | 6423/s | 0.933 | ⚡ Fastest |
| digit_recognizer_v6.tflite | 0.963 | 46.9KB | 4047/s | 0.857 | Alternative |
| digit_recognizer_v17.tflite | 0.987 | 71.0KB | 6256/s | 0.854 | Alternative |
| digit_recognizer_v4.tflite | 0.988 | 78.3KB | 6121/s | 0.836 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v18.tflite
- **Accuracy**: 0.990 (best under 100KB)
- **Trade-off**: 97.4KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 6423 inf/s (fastest under 100KB)
- **Trade-off**: 0.970 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v3.tflite
- **Size**: 38.4KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.973 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0253 | 13.664325842696629 | 71200 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0205 | 12.338422391857506 | 78600 | ⚖️ Good |
| digit_recognizer_v6.tflite | 0.0205 | 12.11194968553459 | 79500 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0139 | 5.615822424587366 | 175700 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0126 | 9.432664756446991 | 104700 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
