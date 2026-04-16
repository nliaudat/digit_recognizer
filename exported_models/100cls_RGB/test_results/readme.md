# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-04-10 11:47:24
- **Models Tested**: 16 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (45.1KB, 0.825 acc, 3359 inf/s)
- **Best Accuracy**: **v32_xl.tflite** (0.964)
- **Fastest Model**: **digit_recognizer_v7.tflite** (4159 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (45.1 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 45.1 | 0.825 | 3359 | 117300 | 0.890 |
| digit_recognizer_v7.tflite | 56.0 | 0.797 | 4159 | 118200 | 0.855 |
| digit_recognizer_v4.tflite | 87.1 | 0.907 | 3227 | 178800 | 0.781 |
| digit_recognizer_v17.tflite | 80.5 | 0.875 | 2421 | 198700 | 0.738 |
| digit_recognizer_v15.tflite | 107.4 | 0.904 | 2324 | 260200 | 0.707 |
| digit_recognizer_v18.tflite | 109.7 | 0.894 | 1884 | 238900 | 0.678 |
| digit_recognizer_v16.tflite | 139.7 | 0.934 | 1999 | 271300 | 0.678 |
| digit_recognizer_v19.tflite | 146.0 | 0.914 | 1641 | 314600 | 0.646 |
| original_haverland.tflite | 228.8 | 0.838 | 2863 | 348100 | 0.632 |
| digit_recognizer_v6.tflite | 160.8 | 0.895 | 1080 | 460900 | 0.600 |
| digit_recognizer_v24.tflite | 433.6 | 0.939 | 918 | 671600 | 0.562 |
| digit_recognizer_v29.tflite | 437.5 | 0.939 | 859 | 689400 | 0.559 |
| digit_recognizer_v28.tflite | 432.8 | 0.924 | 924 | 673600 | 0.555 |
| digit_recognizer_v27.tflite | 433.4 | 0.917 | 914 | 673600 | 0.551 |
| digit_recognizer_v23.tflite | 427.4 | 0.906 | 958 | 663300 | 0.548 |
| v32_xl.tflite | 2110.5 | 0.964 | 369 | 2600000 | 0.524 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.890
- **Accuracy**: 0.825
- **Size**: 45.1 KB
- **Speed**: 3359 inf/s
- **Efficiency**: 0.0183 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.825 | 45.1KB | 3359/s | 0.890 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.797 | 56.0KB | 4159/s | 0.855 | ⚡ Fastest |
| digit_recognizer_v4.tflite | 0.907 | 87.1KB | 3227/s | 0.781 | 🎯 Best Accuracy |
| digit_recognizer_v17.tflite | 0.875 | 80.5KB | 2421/s | 0.738 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.907 (best under 100KB)
- **Trade-off**: 87.1KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 4159 inf/s (fastest under 100KB)
- **Trade-off**: 0.797 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Size**: 87.1KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.907 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0183 | 7.034100596760442 | 117300 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0142 | 6.7428087986463625 | 118200 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0104 | 5.070469798657718 | 178800 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0109 | 4.4031202818319075 | 198700 | ⚖️ Good |
| digit_recognizer_v15.tflite | 0.0084 | 3.4761721752498076 | 260200 | ❌ Too large |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
