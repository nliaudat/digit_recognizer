# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-04-10 11:22:02
- **Models Tested**: 21 quantized models
- **Best IoT Model**: **v3.tflite** (38.4KB, 0.979 acc, 3378 inf/s)
- **Best Accuracy**: **v16.tflite** (0.993)
- **Fastest Model**: **digit_recognizer_v7.tflite** (4139 inf/s)
- **Smallest Model**: **v3.tflite** (38.4 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| v3.tflite | 38.4 | 0.979 | 3378 | 112600 | 0.956 |
| digit_recognizer_v3.tflite | 38.4 | 0.966 | 3456 | 112600 | 0.953 |
| digit_recognizer_v7.tflite | 47.2 | 0.966 | 4139 | 111300 | 0.930 |
| digit_recognizer_v23.tflite | 61.8 | 0.987 | 3692 | 145800 | 0.862 |
| v23.tflite | 61.8 | 0.993 | 3632 | 145800 | 0.862 |
| v23.tflite | 61.8 | 0.991 | 3649 | 145800 | 0.862 |
| v23.tflite | 61.8 | 0.991 | 3622 | 145800 | 0.860 |
| digit_recognizer_v6.tflite | 46.9 | 0.954 | 2329 | 195800 | 0.838 |
| digit_recognizer_v28.tflite | 67.2 | 0.969 | 3245 | 156100 | 0.816 |
| digit_recognizer_v24.tflite | 68.0 | 0.979 | 3025 | 154200 | 0.808 |
| v4.tflite | 79.4 | 0.992 | 3346 | 172000 | 0.806 |
| digit_recognizer_v27.tflite | 67.8 | 0.974 | 2980 | 156100 | 0.804 |
| digit_recognizer_v4.tflite | 78.3 | 0.985 | 3231 | 171900 | 0.799 |
| digit_recognizer_v17.tflite | 71.0 | 0.980 | 2656 | 191200 | 0.784 |
| digit_recognizer_v29.tflite | 71.6 | 0.977 | 2607 | 171600 | 0.778 |
| digit_recognizer_v18.tflite | 97.4 | 0.984 | 2374 | 228500 | 0.728 |
| digit_recognizer_v15.tflite | 100.0 | 0.984 | 2429 | 254800 | 0.728 |
| digit_recognizer_v16.tflite | 128.8 | 0.992 | 2140 | 262300 | 0.692 |
| v16.tflite | 129.8 | 0.993 | 2108 | 262400 | 0.691 |
| original_haverland.tflite | 203.8 | 0.960 | 2751 | 324700 | 0.673 |
| digit_recognizer_v19.tflite | 132.2 | 0.983 | 1779 | 302700 | 0.668 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **v3.tflite**
- **IoT Score**: 0.956
- **Accuracy**: 0.979
- **Size**: 38.4 KB
- **Speed**: 3378 inf/s
- **Efficiency**: 0.0255 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| v3.tflite | 0.979 | 38.4KB | 3378/s | 0.956 | 🏆 **BEST BALANCED** |
| digit_recognizer_v3.tflite | 0.966 | 38.4KB | 3456/s | 0.953 | Alternative |
| digit_recognizer_v7.tflite | 0.966 | 47.2KB | 4139/s | 0.930 | ⚡ Fastest |
| digit_recognizer_v23.tflite | 0.987 | 61.8KB | 3692/s | 0.862 | Alternative |
| v23.tflite | 0.993 | 61.8KB | 3632/s | 0.862 | 🎯 Best Accuracy |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: v23.tflite
- **Accuracy**: 0.993 (best under 100KB)
- **Trade-off**: 61.8KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 4139 inf/s (fastest under 100KB)
- **Trade-off**: 0.966 accuracy

**For Memory-Constrained IoT:**
- **Choice**: v3.tflite
- **Size**: 38.4KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.979 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| v3.tflite | 0.0255 | 8.692717584369449 | 112600 | 🎯 **OPTIMAL** |
| digit_recognizer_v3.tflite | 0.0252 | 8.579928952042627 | 112600 | ⚖️ Good |
| digit_recognizer_v7.tflite | 0.0205 | 8.681940700808626 | 111300 | ⚖️ Good |
| digit_recognizer_v23.tflite | 0.0160 | 6.772290809327847 | 145800 | ⚖️ Good |
| v23.tflite | 0.0161 | 6.810013717421125 | 145800 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
