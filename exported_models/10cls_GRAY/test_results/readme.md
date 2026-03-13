# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-13 15:04:32
- **Models Tested**: 10 quantized models
- **Best IoT Model**: **digit_recognizer_v6.tflite** (36.5KB, 0.939 acc, 5556 inf/s)
- **Best Accuracy**: **digit_recognizer_v15.tflite** (0.991)
- **Fastest Model**: **digit_recognizer_v7.tflite** (6383 inf/s)
- **Smallest Model**: **digit_recognizer_v6.tflite** (36.5 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v6.tflite | 36.5 | 0.939 | 5556 | 61500 | 0.948 |
| digit_recognizer_v7.tflite | 46.7 | 0.973 | 6383 | 75600 | 0.926 |
| digit_recognizer_v4.tflite | 61.4 | 0.986 | 6231 | 79700 | 0.871 |
| digit_recognizer_v3.tflite | 69.4 | 0.975 | 6295 | 118400 | 0.847 |
| digit_recognizer_v15.tflite | 79.3 | 0.991 | 6026 | 109100 | 0.827 |
| digit_recognizer_v17.tflite | 70.7 | 0.985 | 5200 | 172900 | 0.815 |
| digit_recognizer_v18.tflite | 97.1 | 0.989 | 4435 | 210200 | 0.751 |
| digit_recognizer_v16.tflite | 128.6 | 0.990 | 3700 | 244000 | 0.701 |
| digit_recognizer_v19.tflite | 131.9 | 0.989 | 3447 | 284300 | 0.690 |
| original_haverland.tflite | 203.3 | 0.966 | 4485 | 234500 | 0.682 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v6.tflite**
- **IoT Score**: 0.948
- **Accuracy**: 0.939
- **Size**: 36.5 KB
- **Speed**: 5556 inf/s
- **Efficiency**: 0.0257 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v6.tflite | 0.939 | 36.5KB | 5556/s | 0.948 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.973 | 46.7KB | 6383/s | 0.926 | ⚡ Fastest |
| digit_recognizer_v4.tflite | 0.986 | 61.4KB | 6231/s | 0.871 | Alternative |
| digit_recognizer_v3.tflite | 0.975 | 69.4KB | 6295/s | 0.847 | Alternative |
| digit_recognizer_v15.tflite | 0.991 | 79.3KB | 6026/s | 0.827 | 🎯 Best Accuracy |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v15.tflite
- **Accuracy**: 0.991 (best under 100KB)
- **Trade-off**: 79.3KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 6383 inf/s (fastest under 100KB)
- **Trade-off**: 0.973 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v6.tflite
- **Size**: 36.5KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.939 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v6.tflite | 0.0257 | 15.260162601626018 | 61500 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0208 | 12.871693121693122 | 75600 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0161 | 12.376411543287327 | 79700 | ⚖️ Good |
| digit_recognizer_v3.tflite | 0.0140 | 8.23141891891892 | 118400 | ⚖️ Good |
| digit_recognizer_v15.tflite | 0.0125 | 9.081576535288725 | 109100 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
