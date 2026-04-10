# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-04-10 11:21:20
- **Models Tested**: 10 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (45.1KB, 0.825 acc, 3330 inf/s)
- **Best Accuracy**: **digit_recognizer_v16.tflite** (0.934)
- **Fastest Model**: **digit_recognizer_v7.tflite** (4151 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (45.1 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 45.1 | 0.825 | 3330 | 117300 | 0.902 |
| digit_recognizer_v7.tflite | 56.0 | 0.797 | 4151 | 118200 | 0.868 |
| digit_recognizer_v4.tflite | 87.1 | 0.907 | 3247 | 178800 | 0.797 |
| digit_recognizer_v17.tflite | 80.5 | 0.875 | 2487 | 198700 | 0.756 |
| digit_recognizer_v15.tflite | 107.4 | 0.904 | 2343 | 260200 | 0.723 |
| digit_recognizer_v18.tflite | 109.7 | 0.894 | 2278 | 238900 | 0.712 |
| digit_recognizer_v16.tflite | 139.7 | 0.934 | 2026 | 271300 | 0.694 |
| digit_recognizer_v19.tflite | 146.0 | 0.914 | 1753 | 314600 | 0.666 |
| original_haverland.tflite | 228.8 | 0.838 | 2714 | 348100 | 0.639 |
| digit_recognizer_v6.tflite | 160.8 | 0.895 | 1076 | 460900 | 0.615 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.902
- **Accuracy**: 0.825
- **Size**: 45.1 KB
- **Speed**: 3330 inf/s
- **Efficiency**: 0.0183 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.825 | 45.1KB | 3330/s | 0.902 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.797 | 56.0KB | 4151/s | 0.868 | ⚡ Fastest |
| digit_recognizer_v4.tflite | 0.907 | 87.1KB | 3247/s | 0.797 | 🎯 Best Accuracy |
| digit_recognizer_v17.tflite | 0.875 | 80.5KB | 2487/s | 0.756 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.907 (best under 100KB)
- **Trade-off**: 87.1KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 4151 inf/s (fastest under 100KB)
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
