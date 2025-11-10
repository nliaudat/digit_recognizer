# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2025-11-10 19:59:59
- **Models Tested**: 8 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (45.1KB, 0.817 acc, 6584 inf/s)
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.916)
- **Fastest Model**: **digit_recognizer_v7.tflite** (9076 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (45.1 KB)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 45.1 | 0.817 | 6584 | 75900 | 0.891 |
| digit_recognizer_v7.tflite | 56.0 | 0.776 | 9076 | 85500 | 0.865 |
| digit_recognizer_v4.tflite | 87.1 | 0.886 | 8600 | 111500 | 0.829 |
| mnist_quantization.tflite | 72.2 | 0.838 | 6358 | 108000 | 0.785 |
| original_haverland.tflite | 228.8 | 0.848 | 5983 | 263600 | 0.654 |
| digit_recognizer_v9.tflite | 160.0 | 0.882 | 3033 | 914600 | 0.633 |
| digit_recognizer_v6.tflite | 160.8 | 0.882 | 2684 | 209300 | 0.625 |
| digit_recognizer_v12.tflite | 415.4 | 0.916 | 2042 | 499300 | 0.578 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.891
- **Accuracy**: 0.817
- **Size**: 45.1 KB
- **Speed**: 6584 inf/s
- **Efficiency**: 0.0181 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.817 | 45.1KB | 6584/s | 0.891 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.776 | 56.0KB | 9076/s | 0.865 | ⚡ Fastest |
| digit_recognizer_v4.tflite | 0.886 | 87.1KB | 8600/s | 0.829 | 🎯 Best Accuracy |
| mnist_quantization.tflite | 0.838 | 72.2KB | 6358/s | 0.785 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.886 (best under 100KB)
- **Trade-off**: 87.1KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 9076 inf/s (fastest under 100KB)
- **Trade-off**: 0.776 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v12.tflite
- **Size**: 415.4KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.916 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0181 | 10.766798418972332 | 75900 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0138 | 9.071345029239765 | 85500 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0102 | 7.9479820627802695 | 111500 | ⚖️ Good |
| mnist_quantization.tflite | 0.0116 | 7.756481481481481 | 108000 | ⚖️ Good |
| original_haverland.tflite | 0.0037 | 3.2166160849772383 | 263600 | ❌ Too large |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
