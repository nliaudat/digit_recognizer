# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-02-27 20:34:40
- **Models Tested**: 12 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (38.4KB, 1.000 acc, 5498 inf/s)
- **Best Accuracy**: **digit_recognizer_v12.tflite** (1.000)
- **Fastest Model**: **digit_recognizer_v4.tflite** (6707 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (38.4 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 38.4 | 1.000 | 5498 | 71200 | 0.964 |
| digit_recognizer_v7.tflite | 47.2 | 1.000 | 6510 | 78600 | 0.938 |
| digit_recognizer_v6.tflite | 46.9 | 1.000 | 4490 | 79500 | 0.880 |
| digit_recognizer_v4.tflite | 78.3 | 1.000 | 6707 | 104700 | 0.847 |
| digit_recognizer_v17.tflite | 71.0 | 1.000 | 5744 | 175700 | 0.834 |
| digit_recognizer_v17.tflite | 71.0 | 1.000 | 5727 | 175700 | 0.833 |
| digit_recognizer_v17.tflite | 71.0 | 1.000 | 5695 | 175700 | 0.832 |
| mnist_quantization.tflite | 64.2 | 1.000 | 5086 | 101900 | 0.831 |
| digit_recognizer_v15.tflite | 100.0 | 1.000 | 2933 | 140000 | 0.703 |
| digit_recognizer_v16.tflite | 128.8 | 1.000 | 3403 | 246800 | 0.691 |
| original_haverland.tflite | 203.8 | 1.000 | 3906 | 240200 | 0.673 |
| digit_recognizer_v12.tflite | 407.3 | 1.000 | 1739 | 493100 | 0.580 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.964
- **Accuracy**: 1.000
- **Size**: 38.4 KB
- **Speed**: 5498 inf/s
- **Efficiency**: 0.0260 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 1.000 | 38.4KB | 5498/s | 0.964 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 1.000 | 47.2KB | 6510/s | 0.938 | Alternative |
| digit_recognizer_v6.tflite | 1.000 | 46.9KB | 4490/s | 0.880 | Alternative |
| digit_recognizer_v4.tflite | 1.000 | 78.3KB | 6707/s | 0.847 | ⚡ Fastest |
| digit_recognizer_v17.tflite | 1.000 | 71.0KB | 5744/s | 0.834 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v15.tflite
- **Accuracy**: 1.000 (best under 100KB)
- **Trade-off**: 100.0KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Speed**: 6707 inf/s (fastest under 100KB)
- **Trade-off**: 1.000 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v3.tflite
- **Size**: 38.4KB (smallest with ≥85% accuracy)
- **Trade-off**: 1.000 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0260 | 14.04494382022472 | 71200 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0212 | 12.72264631043257 | 78600 | ⚖️ Good |
| digit_recognizer_v6.tflite | 0.0213 | 12.578616352201259 | 79500 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0128 | 9.551098376313275 | 104700 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0141 | 5.691519635742743 | 175700 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
