# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-07 20:42:01
- **Models Tested**: 10 quantized models
- **Best IoT Model**: **digit_recognizer_v6.tflite** (36.5KB, 0.946 acc, 6475 inf/s)
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.988)
- **Fastest Model**: **digit_recognizer_v3.tflite** (8401 inf/s)
- **Smallest Model**: **digit_recognizer_v6.tflite** (36.5 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v6.tflite | 36.5 | 0.946 | 6475 | 61500 | 0.933 |
| digit_recognizer_v7.tflite | 46.7 | 0.955 | 7898 | 75600 | 0.906 |
| digit_recognizer_v3.tflite | 69.4 | 0.968 | 8401 | 118400 | 0.848 |
| digit_recognizer_v4.tflite | 61.4 | 0.970 | 7127 | 79700 | 0.839 |
| digit_recognizer_v17.tflite | 70.7 | 0.981 | 6706 | 172900 | 0.811 |
| digit_recognizer_v15.tflite | 79.3 | 0.983 | 6423 | 109100 | 0.789 |
| mnist_quantization.tflite | 63.6 | 0.953 | 4626 | 98700 | 0.765 |
| digit_recognizer_v16.tflite | 128.6 | 0.986 | 4718 | 244000 | 0.696 |
| original_haverland.tflite | 203.3 | 0.975 | 4503 | 234500 | 0.654 |
| digit_recognizer_v12.tflite | 406.7 | 0.988 | 1957 | 490000 | 0.574 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v6.tflite**
- **IoT Score**: 0.933
- **Accuracy**: 0.946
- **Size**: 36.5 KB
- **Speed**: 6475 inf/s
- **Efficiency**: 0.0259 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v6.tflite | 0.946 | 36.5KB | 6475/s | 0.933 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.955 | 46.7KB | 7898/s | 0.906 | Alternative |
| digit_recognizer_v3.tflite | 0.968 | 69.4KB | 8401/s | 0.848 | ⚡ Fastest |
| digit_recognizer_v4.tflite | 0.970 | 61.4KB | 7127/s | 0.839 | Alternative |
| digit_recognizer_v17.tflite | 0.981 | 70.7KB | 6706/s | 0.811 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v15.tflite
- **Accuracy**: 0.983 (best under 100KB)
- **Trade-off**: 79.3KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v3.tflite
- **Speed**: 8401 inf/s (fastest under 100KB)
- **Trade-off**: 0.968 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v6.tflite
- **Size**: 36.5KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.946 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v6.tflite | 0.0259 | 15.3739837398374 | 61500 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0204 | 12.62962962962963 | 75600 | ⚖️ Good |
| digit_recognizer_v3.tflite | 0.0139 | 8.173141891891891 | 118400 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0158 | 12.165621079046424 | 79700 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0139 | 5.673221515326778 | 172900 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
