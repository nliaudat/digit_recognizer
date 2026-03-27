# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-03-27 13:47:35
- **Models Tested**: 15 quantized models
- **Best IoT Model**: **digit_recognizer_v3.tflite** (45.1KB, 0.824 acc, 6174 inf/s)
- **Best Accuracy**: **digit_recognizer_v24.tflite** (0.939)
- **Fastest Model**: **digit_recognizer_v7.tflite** (8824 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (45.1 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| digit_recognizer_v3.tflite | 45.1 | 0.824 | 6174 | 75900 | 0.879 |
| digit_recognizer_v7.tflite | 56.0 | 0.797 | 8824 | 85500 | 0.866 |
| digit_recognizer_v4.tflite | 87.1 | 0.906 | 8448 | 111500 | 0.830 |
| digit_recognizer_v17.tflite | 80.5 | 0.875 | 7122 | 183300 | 0.796 |
| digit_recognizer_v18.tflite | 109.7 | 0.894 | 6310 | 223400 | 0.742 |
| digit_recognizer_v15.tflite | 107.4 | 0.905 | 5680 | 145400 | 0.737 |
| digit_recognizer_v16.tflite | 139.7 | 0.934 | 5296 | 255800 | 0.714 |
| digit_recognizer_v19.tflite | 146.0 | 0.914 | 4971 | 299100 | 0.692 |
| original_haverland.tflite | 228.8 | 0.838 | 5745 | 263600 | 0.636 |
| digit_recognizer_v6.tflite | 160.8 | 0.894 | 2609 | 209300 | 0.620 |
| digit_recognizer_v24.tflite | 433.6 | 0.939 | 1991 | 502000 | 0.576 |
| digit_recognizer_v29.tflite | 437.5 | 0.939 | 1888 | 514000 | 0.574 |
| digit_recognizer_v28.tflite | 432.8 | 0.924 | 2070 | 503900 | 0.570 |
| digit_recognizer_v27.tflite | 433.4 | 0.916 | 1976 | 503900 | 0.564 |
| digit_recognizer_v23.tflite | 427.4 | 0.906 | 2131 | 493700 | 0.563 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **digit_recognizer_v3.tflite**
- **IoT Score**: 0.879
- **Accuracy**: 0.824
- **Size**: 45.1 KB
- **Speed**: 6174 inf/s
- **Efficiency**: 0.0183 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| digit_recognizer_v3.tflite | 0.824 | 45.1KB | 6174/s | 0.879 | 🏆 **BEST BALANCED** |
| digit_recognizer_v7.tflite | 0.797 | 56.0KB | 8824/s | 0.866 | ⚡ Fastest |
| digit_recognizer_v4.tflite | 0.906 | 87.1KB | 8448/s | 0.830 | 🎯 Best Accuracy |
| digit_recognizer_v17.tflite | 0.875 | 80.5KB | 7122/s | 0.796 | Alternative |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Accuracy**: 0.906 (best under 100KB)
- **Trade-off**: 87.1KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 8824 inf/s (fastest under 100KB)
- **Trade-off**: 0.797 accuracy

**For Memory-Constrained IoT:**
- **Choice**: digit_recognizer_v4.tflite
- **Size**: 87.1KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.906 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| digit_recognizer_v3.tflite | 0.0183 | 10.86034255599473 | 75900 | 🎯 **OPTIMAL** |
| digit_recognizer_v7.tflite | 0.0142 | 9.31812865497076 | 85500 | ⚖️ Good |
| digit_recognizer_v4.tflite | 0.0104 | 8.130044843049326 | 111500 | ⚖️ Good |
| digit_recognizer_v17.tflite | 0.0109 | 4.775231860338243 | 183300 | ⚖️ Good |
| digit_recognizer_v18.tflite | 0.0081 | 4.001790510295434 | 223400 | ❌ Too large |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
