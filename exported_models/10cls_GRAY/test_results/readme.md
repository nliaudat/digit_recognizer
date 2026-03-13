# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-13 14:40:30
- **Models Tested**: 11 quantized models
- **Best IoT Model**: **digit_recognizer_v7.tflite** (46.7KB, 0.973 acc, 8690 inf/s)
- **Best Accuracy**: **digit_recognizer_v15.tflite** (0.991)
- **Fastest Model**: **digit_recognizer_v7.tflite** (8690 inf/s)
- **Smallest Model**: **digit_recognizer_v6.tflite** (36.5 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v7.tflite | 46.7 | 0.973 | 8690 | 75600 | 0.926 |
| digit_recognizer_v6.tflite | 36.5 | 0.939 | 6534 | 61500 | 0.924 |
| digit_recognizer_v4.tflite | 61.4 | 0.986 | 7213 | 79700 | 0.842 |
| digit_recognizer_v3.tflite | 69.4 | 0.975 | 6812 | 118400 | 0.806 |
| digit_recognizer_v17.tflite | 70.7 | 0.985 | 4928 | 172900 | 0.765 |
| digit_recognizer_v15.tflite | 79.3 | 0.991 | 5211 | 109100 | 0.758 |
| digit_recognizer_v18.tflite | 97.1 | 0.989 | 4443 | 210200 | 0.714 |
| original_haverland.tflite | 203.3 | 0.966 | 6012 | 234500 | 0.680 |
| digit_recognizer_v16.tflite | 128.6 | 0.990 | 3730 | 244000 | 0.671 |
| digit_recognizer_v19.tflite | 131.9 | 0.989 | 3494 | 284300 | 0.663 |
| digit_recognizer_v12.tflite | 406.7 | 0.982 | 1634 | 490000 | 0.560 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v7.tflite**
- **IoT Score**: 0.926
- **Accuracy**: 0.973
- **Size**: 46.7 KB
- **Speed**: 8690 inf/s
- **Efficiency**: 0.0208 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v7.tflite | 0.973 | 46.7KB | 8690/s | 0.926 | 🏆 **BEST BALANCED** |
| digit_recognizer_v6.tflite | 0.939 | 36.5KB | 6534/s | 0.924 | 💾 Smallest Adequate |
| digit_recognizer_v4.tflite | 0.986 | 61.4KB | 7213/s | 0.842 | Alternative |
| digit_recognizer_v3.tflite | 0.975 | 69.4KB | 6812/s | 0.806 | Alternative |
| digit_recognizer_v17.tflite | 0.985 | 70.7KB | 4928/s | 0.765 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v15.tflite
- **Accuracy**: 0.991 (best under 100KB)
- **Trade-off**: 79.3KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 8690 inf/s (fastest under 100KB)
- **Trade-off**: 0.973 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v6.tflite
- **Size**: 36.5KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.939 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v7.tflite | 0.0208 | 12.871693121693122 | 75600 | 🎯 **OPTIMAL** |
| digit_recognizer_v6.tflite | 0.0257 | 15.260162601626018 | 61500 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0161 | 12.376411543287327 | 79700 | ⚖️ Good |
| digit_recognizer_v3.tflite | 0.0140 | 8.23141891891892 | 118400 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0139 | 5.698091382301909 | 172900 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
