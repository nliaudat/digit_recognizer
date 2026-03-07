# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-07 20:33:47
- **Models Tested**: 10 quantized models
- **Best IoT Model**: **digit_recognizer_v7.tflite** (55.5KB, 0.835 acc, 6201 inf/s)
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.928)
- **Fastest Model**: **digit_recognizer_v7.tflite** (6201 inf/s)
- **Smallest Model**: **digit_recognizer_v7.tflite** (55.5 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v7.tflite | 55.5 | 0.835 | 6201 | 82400 | 0.950 |
| digit_recognizer_v4.tflite | 69.5 | 0.883 | 5301 | 85800 | 0.887 |
| digit_recognizer_v3.tflite | 74.6 | 0.851 | 5476 | 121600 | 0.858 |
| mnist_quantization.tflite | 71.7 | 0.855 | 5051 | 104800 | 0.856 |
| digit_recognizer_v15.tflite | 86.0 | 0.898 | 5364 | 113700 | 0.851 |
| digit_recognizer_v17.tflite | 80.2 | 0.883 | 4573 | 180400 | 0.831 |
| digit_recognizer_v16.tflite | 139.5 | 0.917 | 3594 | 253000 | 0.730 |
| original_haverland.tflite | 228.2 | 0.881 | 4390 | 257899 | 0.689 |
| digit_recognizer_v6.tflite | 132.5 | 0.888 | 2580 | 171800 | 0.687 |
| digit_recognizer_v12.tflite | 414.8 | 0.928 | 1502 | 496100 | 0.589 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v7.tflite**
- **IoT Score**: 0.950
- **Accuracy**: 0.835
- **Size**: 55.5 KB
- **Speed**: 6201 inf/s
- **Efficiency**: 0.0150 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v7.tflite | 0.835 | 55.5KB | 6201/s | 0.950 | 🏆 **BEST BALANCED** |
| digit_recognizer_v4.tflite | 0.883 | 69.5KB | 5301/s | 0.887 | Alternative |
| digit_recognizer_v3.tflite | 0.851 | 74.6KB | 5476/s | 0.858 | Alternative |
| mnist_quantization.tflite | 0.855 | 71.7KB | 5051/s | 0.856 | Alternative |
| digit_recognizer_v15.tflite | 0.898 | 86.0KB | 5364/s | 0.851 | 🎯 Best Accuracy |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v15.tflite
- **Accuracy**: 0.898 (best under 100KB)
- **Trade-off**: 86.0KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 6201 inf/s (fastest under 100KB)
- **Trade-off**: 0.835 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v16.tflite
- **Size**: 139.5KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.917 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v7.tflite | 0.0150 | 10.135922330097088 | 82400 | 🎯 **OPTIMAL** |
| digit_recognizer_v4.tflite | 0.0127 | 10.297202797202797 | 85800 | ⚖️ Good |
| digit_recognizer_v3.tflite | 0.0114 | 6.999177631578947 | 121600 | ⚖️ Good |
| mnist_quantization.tflite | 0.0119 | 8.153625954198475 | 104800 | ⚖️ Good |
| digit_recognizer_v15.tflite | 0.0104 | 7.901495162708883 | 113700 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
