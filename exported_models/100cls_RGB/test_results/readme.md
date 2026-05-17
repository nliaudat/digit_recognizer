# Digit Recognition Benchmark Report

## 🔌 ESP32 Hardware Simulation

> **What this is**: Each model was also tested through an ESP32-simulated inference pipeline that adds quantization noise to simulate the integer-only arithmetic of TFLite Micro on ESP32. Models with a smaller gap between PC and ESP32 accuracy are more robust for real hardware deployment.

| Model | PC Accuracy | ESP32 Sim. | Gap | Verdict |
|-------|-------------|------------|-----|---------|
| digit_recognizer_v24_quantized_full_integer_quant.tflite | 0.954 | 0.955 | -0.0% | ✅ Excellent (robust) |
| digit_recognizer_v16.tflite | 0.946 | 0.946 | +0.1% | ✅ Excellent (robust) |
| digit_recognizer_v16.tflite | 0.943 | 0.942 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v23_quantized_full_integer_quant.tflite | 0.922 | 0.923 | -0.0% | ✅ Excellent (robust) |
| digit_recognizer_v16_quantized_full_integer_quant.tflite | 0.902 | 0.902 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v6_quantized_full_integer_quant.tflite | 0.899 | 0.899 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v27_quantized_full_integer_quant.tflite | 0.888 | 0.888 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v19_quantized_full_integer_quant.tflite | 0.884 | 0.884 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v15_quantized_full_integer_quant.tflite | 0.869 | 0.868 | +0.1% | ✅ Excellent (robust) |
| digit_recognizer_v18_quantized_full_integer_quant.tflite | 0.862 | 0.861 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v4_quantized_full_integer_quant.tflite | 0.861 | 0.861 | +0.0% | ✅ Excellent (robust) |
| digit_recognizer_v17_quantized_full_integer_quant.tflite | 0.830 | 0.831 | -0.1% | ✅ Excellent (robust) |
| digit_recognizer_v3_quantized_full_integer_quant.tflite | 0.791 | 0.791 | +0.1% | ✅ Excellent (robust) |
| digit_recognizer_v7_quantized_full_integer_quant.tflite | 0.763 | 0.771 | -0.8% | ✅ Excellent (robust) |
| v31_full_integer.tflite | 0.023 | 0.022 | +0.1% | ✅ Excellent (robust) |
| v16_esp32_quantized_full_integer_quant.tflite | 0.023 | 0.021 | +0.1% | ✅ Excellent (robust) |
| v31_full_integer.tflite | 0.022 | 0.018 | +0.4% | ✅ Excellent (robust) |

### 📊 ESP32 Simulation Graphs

![PC vs ESP32 Accuracy](graphs/esp32/pc_vs_esp32_accuracy_quantized_full.png)

![ESP32 Accuracy Gap](graphs/esp32/esp32_accuracy_gap_quantized_full.png)

![ESP32 Accuracy vs Size](graphs/esp32/esp32_accuracy_vs_size_quantized_full.png)

---

## 📊 Executive Summary

- **Test Date**: 2026-05-16 21:41:49
- **Models Tested**: 17 quantized models
- **Best IoT Model**: **digit_recognizer_v3_quantized_full_integer_quant.tflite** (43.7KB, 0.791 acc, 2086 inf/s)
- **Best Accuracy**: **digit_recognizer_v24_quantized_full_integer_quant.tflite** (0.954)
- **Fastest Model**: **digit_recognizer_v3_quantized_full_integer_quant.tflite** (2086 inf/s)
- **Smallest Model**: **digit_recognizer_v3_quantized_full_integer_quant.tflite** (43.7 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3_quantized_full_integer_quant.tflite | 43.7 | 0.791 | 2086 | 98000 | 0.915 |
| digit_recognizer_v7_quantized_full_integer_quant.tflite | 55.3 | 0.763 | 1922 | 116200 | 0.821 |
| digit_recognizer_v4_quantized_full_integer_quant.tflite | 87.0 | 0.861 | 1510 | 176900 | 0.747 |
| digit_recognizer_v17_quantized_full_integer_quant.tflite | 77.0 | 0.830 | 1033 | 196800 | 0.704 |
| digit_recognizer_v15_quantized_full_integer_quant.tflite | 106.5 | 0.869 | 1098 | 258300 | 0.683 |
| digit_recognizer_v18_quantized_full_integer_quant.tflite | 105.6 | 0.862 | 1055 | 237000 | 0.677 |
| digit_recognizer_v16_quantized_full_integer_quant.tflite | 138.6 | 0.902 | 865 | 269500 | 0.650 |
| digit_recognizer_v19_quantized_full_integer_quant.tflite | 141.7 | 0.884 | 800 | 312700 | 0.632 |
| digit_recognizer_v6_quantized_full_integer_quant.tflite | 160.1 | 0.899 | 578 | 459000 | 0.608 |
| digit_recognizer_v16.tflite | 437.2 | 0.946 | 393 | 687400 | 0.564 |
| digit_recognizer_v23_quantized_full_integer_quant.tflite | 427.0 | 0.922 | 486 | 661400 | 0.561 |
| digit_recognizer_v24_quantized_full_integer_quant.tflite | 434.3 | 0.954 | 284 | 673700 | 0.557 |
| digit_recognizer_v16.tflite | 432.5 | 0.943 | 196 | 671500 | 0.543 |
| digit_recognizer_v27_quantized_full_integer_quant.tflite | 433.9 | 0.888 | 172 | 674300 | 0.512 |
| v16_esp32_quantized_full_integer_quant.tflite | 163.8 | 0.023 | 1314 | 269700 | 0.218 |
| v31_full_integer.tflite | 26306.5 | 0.023 | 28 | 28800000 | 0.015 |
| v31_full_integer.tflite | 26306.5 | 0.022 | 27 | 28800000 | 0.015 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3_quantized_full_integer_quant.tflite**
- **IoT Score**: 0.915
- **Accuracy**: 0.791
- **Size**: 43.7 KB
- **Speed**: 2086 inf/s
- **Efficiency**: 0.0181 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3_quantized_full_integer_quant.tflite | 0.791 | 43.7KB | 2086/s | 0.915 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7_quantized_full_integer_quant.tflite | 0.763 | 55.3KB | 1922/s | 0.821 | Alternative |
| digit_recognizer_v4_quantized_full_integer_quant.tflite | 0.861 | 87.0KB | 1510/s | 0.747 | 🎯 Best Accuracy |
| digit_recognizer_v17_quantized_full_integer_quant.tflite | 0.830 | 77.0KB | 1033/s | 0.704 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4_quantized_full_integer_quant.tflite
- **Accuracy**: 0.861 (best under 100KB)
- **Trade-off**: 87.0KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v3_quantized_full_integer_quant.tflite
- **Speed**: 2086 inf/s (fastest under 100KB)
- **Trade-off**: 0.791 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v16_quantized_full_integer_quant.tflite
- **Size**: 138.6KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.902 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3_quantized_full_integer_quant.tflite | 0.0181 | 8.073469387755102 | 98000 | 🎯 **OPTIMAL** |
| digit_recognizer_v7_quantized_full_integer_quant.tflite | 0.0138 | 6.566265060240964 | 116200 | ⚖️ Good |
| digit_recognizer_v4_quantized_full_integer_quant.tflite | 0.0099 | 4.866026003391747 | 176900 | ⚖️ Good |
| digit_recognizer_v17_quantized_full_integer_quant.tflite | 0.0108 | 4.217479674796747 | 196800 | ⚖️ Good |
| digit_recognizer_v15_quantized_full_integer_quant.tflite | 0.0082 | 3.362369337979094 | 258300 | ❌ Too large |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
