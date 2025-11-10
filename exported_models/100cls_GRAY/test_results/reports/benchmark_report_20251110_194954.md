# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2025-11-10 19:49:54
- **Models Tested**: 8 quantized models
- **Best IoT Model**: **digit_recognizer_v7.tflite** (55.5KB, 0.781 acc, 9270 inf/s)
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.915)
- **Fastest Model**: **digit_recognizer_v7.tflite** (9270 inf/s)
- **Smallest Model**: **digit_recognizer_v7.tflite** (55.5 KB)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v7.tflite | 55.5 | 0.781 | 9270 | 82400 | 0.927 |
| digit_recognizer_v4.tflite | 69.5 | 0.856 | 8003 | 85800 | 0.880 |
| digit_recognizer_v3.tflite | 74.6 | 0.787 | 8331 | 121600 | 0.833 |
| mnist_quantization.tflite | 71.7 | 0.816 | 6400 | 104800 | 0.816 |
| digit_recognizer_v6.tflite | 132.5 | 0.862 | 3343 | 171800 | 0.669 |
| original_haverland.tflite | 228.2 | 0.844 | 6220 | 257899 | 0.668 |
| digit_recognizer_v9.tflite | 159.5 | 0.873 | 3145 | 911500 | 0.650 |
| digit_recognizer_v12.tflite | 414.8 | 0.915 | 2039 | 496100 | 0.584 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v7.tflite**
- **IoT Score**: 0.927
- **Accuracy**: 0.781
- **Size**: 55.5 KB
- **Speed**: 9270 inf/s
- **Efficiency**: 0.0141 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v7.tflite | 0.781 | 55.5KB | 9270/s | 0.927 | 🏆 **BEST BALANCED** |
| digit_recognizer_v4.tflite | 0.856 | 69.5KB | 8003/s | 0.880 | 🎯 Best Accuracy |
| digit_recognizer_v3.tflite | 0.787 | 74.6KB | 8331/s | 0.833 | Alternative |
| mnist_quantization.tflite | 0.816 | 71.7KB | 6400/s | 0.816 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.856 (best under 100KB)
- **Trade-off**: 69.5KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 9270 inf/s (fastest under 100KB)
- **Trade-off**: 0.781 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v12.tflite
- **Size**: 414.8KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.915 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v7.tflite | 0.0141 | 9.474514563106796 | 82400 | 🎯 **OPTIMAL** |
| digit_recognizer_v4.tflite | 0.0123 | 9.974358974358974 | 85800 | ⚖️ Good |
| digit_recognizer_v3.tflite | 0.0106 | 6.474506578947369 | 121600 | ⚖️ Good |
| mnist_quantization.tflite | 0.0114 | 7.7853053435114505 | 104800 | ⚖️ Good |
| digit_recognizer_v6.tflite | 0.0065 | 5.018626309662397 | 171800 | ❌ Too large |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
