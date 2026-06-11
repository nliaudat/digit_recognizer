# Digit Recognition Benchmark Report

## 🔌 ESP32 Hardware Simulation

> **What this is**: Each model was also tested through an ESP32-simulated inference pipeline that adds quantization noise to simulate the integer-only arithmetic of TFLite Micro on ESP32. Models with a smaller gap between PC and ESP32 accuracy are more robust for real hardware deployment.

| Model | PC Accuracy | ESP32 Sim. | Gap | Verdict |
|-------|-------------|------------|-----|---------|
| digit_recognizer_v16_full_integer_quant.tflite | 0.995 | 0.995 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v19_full_integer_quant.tflite | 0.994 | 0.994 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v18_full_integer_quant.tflite | 0.992 | 0.992 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v15_full_integer_quant.tflite | 0.992 | 0.992 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v24.tflite | 0.989 | 0.989 | -0.0% | ✅ Excellent (robust) |
| digit_recognizer_v24_full_integer_quant.tflite | 0.989 | 0.989 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v4_full_integer_quant.tflite | 0.987 | 0.987 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v3_full_integer_quant.tflite | 0.980 | 0.980 | -0.0% | ✅ Excellent (robust) |
| digit_recognizer_v23_full_integer_quant.tflite | 0.979 | 0.979 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v23_full_integer_quant.tflite | 0.971 | 0.971 | +0.0% | ✅ Excellent (robust) |

### 📊 ESP32 Simulation Graphs

![PC vs ESP32 Accuracy](graphs/esp32/pc_vs_esp32_accuracy_quantized_full.png)

![ESP32 Accuracy Gap](graphs/esp32/esp32_accuracy_gap_quantized_full.png)

![ESP32 Accuracy vs Size](graphs/esp32/esp32_accuracy_vs_size_quantized_full.png)

---

## 📊 Executive Summary

- **Test Date**: 2026-06-11 17:00:49
- **Models Tested**: 10 quantized models
- **Best IoT Model**: **digit_recognizer_v3_full_integer_quant.tflite** (37.0KB, 0.980 acc, 5269 inf/s)
- **Best Accuracy**: **digit_recognizer_v16_full_integer_quant.tflite** (0.995)
- **Fastest Model**: **digit_recognizer_v3_full_integer_quant.tflite** (5269 inf/s)
- **Smallest Model**: **digit_recognizer_v3_full_integer_quant.tflite** (37.0 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3_full_integer_quant.tflite | 37.0 | 0.980 | 5269 | 93400 | 0.992 |
| digit_recognizer_v23_full_integer_quant.tflite | 61.4 | 0.979 | 3817 | 143900 | 0.818 |
| digit_recognizer_v23_full_integer_quant.tflite | 61.4 | 0.971 | 3805 | 143900 | 0.813 |
| digit_recognizer_v24_full_integer_quant.tflite | 68.6 | 0.989 | 3213 | 156200 | 0.781 |
| digit_recognizer_v24.tflite | 67.7 | 0.989 | 3057 | 152200 | 0.777 |
| digit_recognizer_v4_full_integer_quant.tflite | 78.2 | 0.987 | 3586 | 170200 | 0.774 |
| digit_recognizer_v18_full_integer_quant.tflite | 93.1 | 0.992 | 2513 | 226700 | 0.713 |
| digit_recognizer_v15_full_integer_quant.tflite | 99.2 | 0.992 | 2383 | 252900 | 0.701 |
| digit_recognizer_v16_full_integer_quant.tflite | 127.7 | 0.995 | 2207 | 260600 | 0.671 |
| digit_recognizer_v19_full_integer_quant.tflite | 127.8 | 0.994 | 1808 | 300900 | 0.655 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3_full_integer_quant.tflite**
- **IoT Score**: 0.992
- **Accuracy**: 0.980
- **Size**: 37.0 KB
- **Speed**: 5269 inf/s
- **Efficiency**: 0.0265 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3_full_integer_quant.tflite | 0.980 | 37.0KB | 5269/s | 0.992 | 🏆 **BEST BALANCED** |
| digit_recognizer_v23_full_integer_quant.tflite | 0.979 | 61.4KB | 3817/s | 0.818 | Alternative |
| digit_recognizer_v23_full_integer_quant.tflite | 0.971 | 61.4KB | 3805/s | 0.813 | Alternative |
| digit_recognizer_v24_full_integer_quant.tflite | 0.989 | 68.6KB | 3213/s | 0.781 | Alternative |
| digit_recognizer_v24.tflite | 0.989 | 67.7KB | 3057/s | 0.777 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v18_full_integer_quant.tflite
- **Accuracy**: 0.992 (best under 100KB)
- **Trade-off**: 93.1KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v3_full_integer_quant.tflite
- **Speed**: 5269 inf/s (fastest under 100KB)
- **Trade-off**: 0.980 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v3_full_integer_quant.tflite
- **Size**: 37.0KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.980 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3_full_integer_quant.tflite | 0.0265 | 10.487152034261243 | 93400 | 🎯 **OPTIMAL** |
| digit_recognizer_v23_full_integer_quant.tflite | 0.0159 | 6.801250868658791 | 143900 | ⚖️ Good |
| digit_recognizer_v23_full_integer_quant.tflite | 0.0158 | 6.747046560111189 | 143900 | ⚖️ Good |
| digit_recognizer_v24_full_integer_quant.tflite | 0.0144 | 6.331626120358515 | 156200 | ⚖️ Good |
| digit_recognizer_v24.tflite | 0.0146 | 6.495400788436268 | 152200 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
