# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-13 15:08:02
- **Models Tested**: 10 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (45.1KB, 0.835 acc, 4533 inf/s)
- **Best Accuracy**: **digit_recognizer_v16.tflite** (0.942)
- **Fastest Model**: **digit_recognizer_v7.tflite** (6258 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (45.1 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 45.1 | 0.835 | 4533 | 75900 | 0.888 |
| digit_recognizer_v7.tflite | 56.0 | 0.806 | 6258 | 85500 | 0.869 |
| digit_recognizer_v4.tflite | 87.1 | 0.905 | 5856 | 111500 | 0.823 |
| digit_recognizer_v17.tflite | 80.5 | 0.885 | 4860 | 183300 | 0.793 |
| digit_recognizer_v15.tflite | 107.4 | 0.914 | 4961 | 145400 | 0.770 |
| digit_recognizer_v18.tflite | 109.7 | 0.904 | 4309 | 223400 | 0.741 |
| digit_recognizer_v16.tflite | 139.7 | 0.942 | 3565 | 255800 | 0.711 |
| digit_recognizer_v19.tflite | 146.0 | 0.924 | 3311 | 299100 | 0.689 |
| original_haverland.tflite | 228.8 | 0.847 | 4666 | 263600 | 0.658 |
| digit_recognizer_v6.tflite | 160.8 | 0.905 | 1954 | 209300 | 0.627 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.888
- **Accuracy**: 0.835
- **Size**: 45.1 KB
- **Speed**: 4533 inf/s
- **Efficiency**: 0.0185 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.835 | 45.1KB | 4533/s | 0.888 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.806 | 56.0KB | 6258/s | 0.869 | ⚡ Fastest |
| digit_recognizer_v4.tflite | 0.905 | 87.1KB | 5856/s | 0.823 | 🎯 Best Accuracy |
| digit_recognizer_v17.tflite | 0.885 | 80.5KB | 4860/s | 0.793 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.905 (best under 100KB)
- **Trade-off**: 87.1KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 6258 inf/s (fastest under 100KB)
- **Trade-off**: 0.806 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Size**: 87.1KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.905 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0185 | 10.994729907773387 | 75900 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0144 | 9.423391812865498 | 85500 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0104 | 8.120179372197308 | 111500 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0110 | 4.82924168030551 | 183300 | ⚖️ Good |
| digit_recognizer_v15.tflite | 0.0085 | 6.285419532324622 | 145400 | ❌ Too large |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
