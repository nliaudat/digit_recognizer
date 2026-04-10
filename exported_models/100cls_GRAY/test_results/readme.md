# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-04-10 10:58:07
- **Models Tested**: 15 quantized models
- **Best IoT Model**: **digit_recognizer_v7.tflite** (55.5KB, 0.806 acc, 4037 inf/s)
- **Best Accuracy**: **digit_recognizer_v24.tflite** (0.945)
- **Fastest Model**: **digit_recognizer_v7.tflite** (4037 inf/s)
- **Smallest Model**: **digit_recognizer_v7.tflite** (55.5 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v7.tflite | 55.5 | 0.806 | 4037 | 104900 | 0.927 |
| digit_recognizer_v4.tflite | 70.5 | 0.853 | 3694 | 150100 | 0.871 |
| digit_recognizer_v3.tflite | 74.6 | 0.808 | 3161 | 178600 | 0.807 |
| digit_recognizer_v15.tflite | 86.0 | 0.883 | 2869 | 219700 | 0.803 |
| digit_recognizer_v17.tflite | 80.2 | 0.866 | 2436 | 185600 | 0.787 |
| digit_recognizer_v18.tflite | 109.2 | 0.876 | 2181 | 225700 | 0.724 |
| digit_recognizer_v16.tflite | 140.5 | 0.908 | 2051 | 258300 | 0.701 |
| digit_recognizer_v19.tflite | 145.3 | 0.898 | 1605 | 301300 | 0.670 |
| original_haverland.tflite | 228.2 | 0.836 | 2888 | 332100 | 0.659 |
| digit_recognizer_v6.tflite | 132.5 | 0.847 | 1304 | 384400 | 0.638 |
| digit_recognizer_v24.tflite | 430.6 | 0.945 | 881 | 665100 | 0.582 |
| digit_recognizer_v29.tflite | 435.0 | 0.936 | 839 | 682900 | 0.575 |
| digit_recognizer_v27.tflite | 430.5 | 0.924 | 884 | 667000 | 0.572 |
| digit_recognizer_v28.tflite | 429.9 | 0.925 | 870 | 667000 | 0.571 |
| digit_recognizer_v23.tflite | 426.8 | 0.901 | 924 | 661200 | 0.562 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v7.tflite**
- **IoT Score**: 0.927
- **Accuracy**: 0.806
- **Size**: 55.5 KB
- **Speed**: 4037 inf/s
- **Efficiency**: 0.0145 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v7.tflite | 0.806 | 55.5KB | 4037/s | 0.927 | 🏆 **BEST BALANCED** |
| digit_recognizer_v4.tflite | 0.853 | 70.5KB | 3694/s | 0.871 | Alternative |
| digit_recognizer_v3.tflite | 0.808 | 74.6KB | 3161/s | 0.807 | Alternative |
| digit_recognizer_v15.tflite | 0.883 | 86.0KB | 2869/s | 0.803 | 🎯 Best Accuracy |
| digit_recognizer_v17.tflite | 0.866 | 80.2KB | 2436/s | 0.787 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v15.tflite
- **Accuracy**: 0.883 (best under 100KB)
- **Trade-off**: 86.0KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 4037 inf/s (fastest under 100KB)
- **Trade-off**: 0.806 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v16.tflite
- **Size**: 140.5KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.908 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v7.tflite | 0.0145 | 7.686367969494758 | 104900 | 🎯 **OPTIMAL** |
| digit_recognizer_v4.tflite | 0.0121 | 5.685542971352432 | 150100 | ⚖️ Good |
| digit_recognizer_v3.tflite | 0.0108 | 4.522956326987682 | 178600 | ⚖️ Good |
| digit_recognizer_v15.tflite | 0.0103 | 4.018206645425581 | 219700 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0108 | 4.666487068965517 | 185600 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
