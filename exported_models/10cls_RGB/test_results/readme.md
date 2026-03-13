# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-13 15:04:47
- **Models Tested**: 10 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (38.4KB, 0.973 acc, 5172 inf/s)
- **Best Accuracy**: **digit_recognizer_v16.tflite** (0.992)
- **Fastest Model**: **digit_recognizer_v4.tflite** (6380 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (38.4 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 38.4 | 0.973 | 5172 | 71200 | 0.953 |
| digit_recognizer_v7.tflite | 47.2 | 0.970 | 6264 | 78600 | 0.929 |
| digit_recognizer_v6.tflite | 46.9 | 0.963 | 3824 | 79500 | 0.851 |
| digit_recognizer_v4.tflite | 78.3 | 0.988 | 6380 | 104700 | 0.845 |
| digit_recognizer_v17.tflite | 71.0 | 0.987 | 4810 | 175700 | 0.810 |
| digit_recognizer_v18.tflite | 97.4 | 0.990 | 4272 | 213000 | 0.751 |
| digit_recognizer_v15.tflite | 100.0 | 0.989 | 4206 | 140000 | 0.745 |
| digit_recognizer_v16.tflite | 128.8 | 0.992 | 3907 | 246800 | 0.712 |
| digit_recognizer_v19.tflite | 132.2 | 0.988 | 3453 | 287200 | 0.693 |
| original_haverland.tflite | 203.8 | 0.966 | 4661 | 240200 | 0.690 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.953
- **Accuracy**: 0.973
- **Size**: 38.4 KB
- **Speed**: 5172 inf/s
- **Efficiency**: 0.0253 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.973 | 38.4KB | 5172/s | 0.953 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.970 | 47.2KB | 6264/s | 0.929 | Alternative |
| digit_recognizer_v6.tflite | 0.963 | 46.9KB | 3824/s | 0.851 | Alternative |
| digit_recognizer_v4.tflite | 0.988 | 78.3KB | 6380/s | 0.845 | ⚡ Fastest |
| digit_recognizer_v17.tflite | 0.987 | 71.0KB | 4810/s | 0.810 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v18.tflite
- **Accuracy**: 0.990 (best under 100KB)
- **Trade-off**: 97.4KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Speed**: 6380 inf/s (fastest under 100KB)
- **Trade-off**: 0.988 accuracy

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
| digit_recognizer_v4.tflite | 0.0126 | 9.432664756446991 | 104700 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0139 | 5.615822424587366 | 175700 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
