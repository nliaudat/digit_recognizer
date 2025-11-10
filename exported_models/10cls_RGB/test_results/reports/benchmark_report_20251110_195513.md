# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2025-11-10 19:55:13
- **Models Tested**: 8 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (38.4KB, 0.983 acc, 7022 inf/s)
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.996)
- **Fastest Model**: **digit_recognizer_v7.tflite** (9015 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (38.4 KB)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 38.4 | 0.983 | 7022 | 71200 | 0.950 |
| digit_recognizer_v7.tflite | 47.2 | 0.974 | 9015 | 78600 | 0.933 |
| digit_recognizer_v6.tflite | 46.9 | 0.968 | 5354 | 79500 | 0.851 |
| digit_recognizer_v4.tflite | 78.3 | 0.992 | 8817 | 104700 | 0.841 |
| mnist_quantization.tflite | 64.2 | 0.976 | 6252 | 101900 | 0.808 |
| original_haverland.tflite | 203.8 | 0.988 | 5894 | 240200 | 0.683 |
| digit_recognizer_v9.tflite | 149.1 | 0.992 | 3141 | 905600 | 0.645 |
| digit_recognizer_v12.tflite | 407.3 | 0.996 | 2032 | 493100 | 0.573 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.950
- **Accuracy**: 0.983
- **Size**: 38.4 KB
- **Speed**: 7022 inf/s
- **Efficiency**: 0.0256 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.983 | 38.4KB | 7022/s | 0.950 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.974 | 47.2KB | 9015/s | 0.933 | ⚡ Fastest |
| digit_recognizer_v6.tflite | 0.968 | 46.9KB | 5354/s | 0.851 | Alternative |
| digit_recognizer_v4.tflite | 0.992 | 78.3KB | 8817/s | 0.841 | 🎯 Best Accuracy |
| mnist_quantization.tflite | 0.976 | 64.2KB | 6252/s | 0.808 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.992 (best under 100KB)
- **Trade-off**: 78.3KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 9015 inf/s (fastest under 100KB)
- **Trade-off**: 0.974 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v3.tflite
- **Size**: 38.4KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.983 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0256 | 13.811797752808989 | 71200 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0206 | 12.39440203562341 | 78600 | ⚖️ Good |
| digit_recognizer_v6.tflite | 0.0206 | 12.177358490566037 | 79500 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0127 | 9.470869149952245 | 104700 | ⚖️ Good |
| mnist_quantization.tflite | 0.0152 | 9.580961727183514 | 101900 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
