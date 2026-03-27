# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-27 13:35:57
- **Models Tested**: 15 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (38.4KB, 0.966 acc, 6503 inf/s)
- **Best Accuracy**: **digit_recognizer_v16.tflite** (0.992)
- **Fastest Model**: **digit_recognizer_v7.tflite** (8868 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (38.4 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 38.4 | 0.966 | 6503 | 71200 | 0.933 |
| digit_recognizer_v7.tflite | 47.2 | 0.966 | 8868 | 78600 | 0.931 |
| digit_recognizer_v23.tflite | 61.8 | 0.987 | 7585 | 82900 | 0.855 |
| digit_recognizer_v6.tflite | 46.9 | 0.954 | 5228 | 79500 | 0.844 |
| digit_recognizer_v4.tflite | 78.3 | 0.985 | 8545 | 104700 | 0.836 |
| digit_recognizer_v17.tflite | 71.0 | 0.981 | 7194 | 175700 | 0.819 |
| digit_recognizer_v28.tflite | 67.2 | 0.969 | 6840 | 93100 | 0.814 |
| digit_recognizer_v24.tflite | 68.0 | 0.978 | 5825 | 91200 | 0.794 |
| digit_recognizer_v27.tflite | 67.8 | 0.974 | 5649 | 93100 | 0.788 |
| digit_recognizer_v29.tflite | 71.6 | 0.977 | 5135 | 102900 | 0.769 |
| digit_recognizer_v18.tflite | 97.4 | 0.984 | 6524 | 213000 | 0.761 |
| digit_recognizer_v15.tflite | 100.0 | 0.984 | 5750 | 140000 | 0.741 |
| digit_recognizer_v16.tflite | 128.8 | 0.992 | 5385 | 246800 | 0.711 |
| digit_recognizer_v19.tflite | 132.2 | 0.983 | 5084 | 287200 | 0.697 |
| original_haverland.tflite | 203.8 | 0.960 | 5754 | 240200 | 0.670 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.933
- **Accuracy**: 0.966
- **Size**: 38.4 KB
- **Speed**: 6503 inf/s
- **Efficiency**: 0.0252 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.966 | 38.4KB | 6503/s | 0.933 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.966 | 47.2KB | 8868/s | 0.931 | ⚡ Fastest |
| digit_recognizer_v23.tflite | 0.987 | 61.8KB | 7585/s | 0.855 | 🎯 Best Accuracy |
| digit_recognizer_v6.tflite | 0.954 | 46.9KB | 5228/s | 0.844 | Alternative |
| digit_recognizer_v4.tflite | 0.985 | 78.3KB | 8545/s | 0.836 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v23.tflite
- **Accuracy**: 0.987 (best under 100KB)
- **Trade-off**: 61.8KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 8868 inf/s (fastest under 100KB)
- **Trade-off**: 0.966 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v3.tflite
- **Size**: 38.4KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.966 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0252 | 13.564606741573034 | 71200 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0205 | 12.288804071246819 | 78600 | ⚖️ Good |
| digit_recognizer_v23.tflite | 0.0160 | 11.910735826296744 | 82900 | ⚖️ Good |
| digit_recognizer_v6.tflite | 0.0203 | 11.99748427672956 | 79500 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0126 | 9.404966571155683 | 104700 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
