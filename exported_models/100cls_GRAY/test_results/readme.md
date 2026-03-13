# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-13 14:33:57
- **Models Tested**: 11 quantized models
- **Best IoT Model**: **digit_recognizer_v7.tflite** (55.5KB, 0.802 acc, 6018 inf/s)
- **Best Accuracy**: **digit_recognizer_v16.tflite** (0.926)
- **Fastest Model**: **digit_recognizer_v7.tflite** (6018 inf/s)
- **Smallest Model**: **digit_recognizer_v7.tflite** (55.5 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v7.tflite | 55.5 | 0.802 | 6018 | 82400 | 0.933 |
| digit_recognizer_v4.tflite | 69.5 | 0.896 | 5465 | 85800 | 0.905 |
| digit_recognizer_v3.tflite | 74.6 | 0.825 | 5921 | 121600 | 0.866 |
| digit_recognizer_v17.tflite | 80.2 | 0.890 | 4740 | 180400 | 0.846 |
| digit_recognizer_v15.tflite | 86.0 | 0.902 | 4788 | 113700 | 0.840 |
| digit_recognizer_v18.tflite | 109.4 | 0.902 | 4263 | 220600 | 0.781 |
| digit_recognizer_v16.tflite | 139.5 | 0.926 | 3558 | 253000 | 0.738 |
| digit_recognizer_v19.tflite | 145.6 | 0.916 | 3371 | 296200 | 0.721 |
| digit_recognizer_v6.tflite | 132.5 | 0.909 | 2312 | 171800 | 0.693 |
| original_haverland.tflite | 228.2 | 0.845 | 4308 | 257899 | 0.672 |
| digit_recognizer_v12.tflite | 414.8 | 0.901 | 1434 | 496100 | 0.574 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v7.tflite**
- **IoT Score**: 0.933
- **Accuracy**: 0.802
- **Size**: 55.5 KB
- **Speed**: 6018 inf/s
- **Efficiency**: 0.0145 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v7.tflite | 0.802 | 55.5KB | 6018/s | 0.933 | 🏆 **BEST BALANCED** |
| digit_recognizer_v4.tflite | 0.896 | 69.5KB | 5465/s | 0.905 | Alternative |
| digit_recognizer_v3.tflite | 0.825 | 74.6KB | 5921/s | 0.866 | Alternative |
| digit_recognizer_v17.tflite | 0.890 | 80.2KB | 4740/s | 0.846 | Alternative |
| digit_recognizer_v15.tflite | 0.902 | 86.0KB | 4788/s | 0.840 | 🎯 Best Accuracy |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v15.tflite
- **Accuracy**: 0.902 (best under 100KB)
- **Trade-off**: 86.0KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 6018 inf/s (fastest under 100KB)
- **Trade-off**: 0.802 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v15.tflite
- **Size**: 86.0KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.902 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v7.tflite | 0.0145 | 9.737864077669903 | 82400 | 🎯 **OPTIMAL** |
| digit_recognizer_v4.tflite | 0.0129 | 10.442890442890443 | 85800 | ⚖️ Good |
| digit_recognizer_v3.tflite | 0.0111 | 6.786184210526316 | 121600 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0111 | 4.93569844789357 | 180400 | ⚖️ Good |
| digit_recognizer_v15.tflite | 0.0105 | 7.931398416886545 | 113700 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
