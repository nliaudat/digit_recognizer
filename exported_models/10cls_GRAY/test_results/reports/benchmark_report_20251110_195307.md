# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2025-11-10 19:53:07
- **Models Tested**: 10 quantized models
- **Best IoT Model**: **digit_recognizer_v6.tflite** (36.5KB, 0.971 acc, 6643 inf/s)
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.996)
- **Fastest Model**: **digit_recognizer_v7.tflite** (8884 inf/s)
- **Smallest Model**: **digit_recognizer_v6.tflite** (36.5 KB)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v6.tflite | 36.5 | 0.971 | 6643 | 61500 | 0.937 |
| digit_recognizer_v7.tflite | 46.7 | 0.974 | 8884 | 75600 | 0.924 |
| digit_recognizer_v4.tflite | 61.4 | 0.990 | 7617 | 79700 | 0.847 |
| digit_recognizer_v3.tflite | 69.4 | 0.985 | 8420 | 118400 | 0.842 |
| mnist_quantization.tflite | 63.6 | 0.977 | 6045 | 98700 | 0.799 |
| original_haverland.tflite | 203.3 | 0.987 | 5873 | 234500 | 0.682 |
| digit_recognizer_v9.tflite | 148.6 | 0.993 | 3079 | 902500 | 0.642 |
| digit_recognizer_v12.tflite | 406.7 | 0.996 | 1917 | 490000 | 0.570 |
| digit_recognizer_v8.tflite | 396.4 | 0.995 | 1485 | 602700 | 0.561 |
| digit_recognizer_v11.tflite | 1370.8 | 0.993 | 1405 | 2800000 | 0.538 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v6.tflite**
- **IoT Score**: 0.937
- **Accuracy**: 0.971
- **Size**: 36.5 KB
- **Speed**: 6643 inf/s
- **Efficiency**: 0.0266 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v6.tflite | 0.971 | 36.5KB | 6643/s | 0.937 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.974 | 46.7KB | 8884/s | 0.924 | ⚡ Fastest |
| digit_recognizer_v4.tflite | 0.990 | 61.4KB | 7617/s | 0.847 | 🎯 Best Accuracy |
| digit_recognizer_v3.tflite | 0.985 | 69.4KB | 8420/s | 0.842 | Alternative |
| mnist_quantization.tflite | 0.977 | 63.6KB | 6045/s | 0.799 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.990 (best under 100KB)
- **Trade-off**: 61.4KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 8884 inf/s (fastest under 100KB)
- **Trade-off**: 0.974 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v6.tflite
- **Size**: 36.5KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.971 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v6.tflite | 0.0266 | 15.788617886178862 | 61500 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0209 | 12.880952380952381 | 75600 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0161 | 12.424090338770387 | 79700 | ⚖️ Good |
| digit_recognizer_v3.tflite | 0.0142 | 8.319256756756756 | 118400 | ⚖️ Good |
| mnist_quantization.tflite | 0.0154 | 9.901722391084093 | 98700 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
