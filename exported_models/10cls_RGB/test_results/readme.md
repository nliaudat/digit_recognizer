# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2026-04-04 16:07:01
- **Models Tested**: 21 quantized models
- **Best IoT Model**: **v3.tflite** (38.4KB, 0.979 acc, 4073 inf/s)
- **Best Accuracy**: **v32_xl.tflite** (0.994)
- **Fastest Model**: **digit_recognizer_v7.tflite** (4946 inf/s)
- **Smallest Model**: **v3.tflite** (38.4 KB)

## 📈 Performance vs Size

![Accuracy vs Size](graphs/accuracy_vs_size_quantized_full.png)

## 📋 Detailed Results

| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |
|-------|-----------|----------|-------|------------|-----------|
| v3.tflite | 38.4 | 0.979 | 4073 | 114500 | 0.957 |
| digit_recognizer_v3.tflite | 38.4 | 0.966 | 4096 | 114500 | 0.951 |
| digit_recognizer_v7.tflite | 47.2 | 0.966 | 4946 | 113300 | 0.930 |
| v23.tflite | 61.8 | 0.991 | 4497 | 147700 | 0.867 |
| v23.tflite | 61.8 | 0.993 | 4289 | 147700 | 0.859 |
| digit_recognizer_v23.tflite | 61.8 | 0.987 | 4229 | 147700 | 0.854 |
| digit_recognizer_v6.tflite | 46.9 | 0.954 | 2569 | 197700 | 0.829 |
| digit_recognizer_v28.tflite | 67.2 | 0.969 | 3742 | 158000 | 0.810 |
| digit_recognizer_v24.tflite | 68.0 | 0.979 | 3466 | 156100 | 0.802 |
| v4.tflite | 79.4 | 0.992 | 3812 | 173900 | 0.798 |
| digit_recognizer_v27.tflite | 67.8 | 0.974 | 3265 | 158000 | 0.792 |
| digit_recognizer_v4.tflite | 78.3 | 0.985 | 3293 | 173900 | 0.776 |
| digit_recognizer_v17.tflite | 71.0 | 0.980 | 2844 | 193100 | 0.770 |
| digit_recognizer_v29.tflite | 71.6 | 0.977 | 2887 | 173600 | 0.769 |
| digit_recognizer_v15.tflite | 100.0 | 0.984 | 2631 | 256700 | 0.716 |
| digit_recognizer_v18.tflite | 97.4 | 0.984 | 2446 | 230400 | 0.712 |
| v16.tflite | 129.8 | 0.993 | 2445 | 264300 | 0.687 |
| digit_recognizer_v16.tflite | 128.8 | 0.992 | 2235 | 264200 | 0.679 |
| original_haverland.tflite | 203.8 | 0.960 | 3077 | 326600 | 0.664 |
| digit_recognizer_v19.tflite | 132.2 | 0.983 | 1921 | 304600 | 0.659 |
| v32_xl.tflite | 2074.3 | 0.994 | 358 | 2500000 | 0.520 |

## 💡 IoT-Specific Recommendations

### 🏆 Dynamic IoT Model Selection

#### 🎯 Best Overall for ESP32
- **Model**: **v3.tflite**
- **IoT Score**: 0.957
- **Accuracy**: 0.979
- **Size**: 38.4 KB
- **Speed**: 4073 inf/s
- **Efficiency**: 0.0255 accuracy per KB

#### 📊 IoT Model Comparison (Under 100KB)
| Model | Accuracy | Size | Speed | IoT Score | Use Case |
|-------|----------|------|-------|-----------|----------|
| v3.tflite | 0.979 | 38.4KB | 4073/s | 0.957 | 🏆 **BEST BALANCED** |
| digit_recognizer_v3.tflite | 0.966 | 38.4KB | 4096/s | 0.951 | Alternative |
| digit_recognizer_v7.tflite | 0.966 | 47.2KB | 4946/s | 0.930 | ⚡ Fastest |
| v23.tflite | 0.991 | 61.8KB | 4497/s | 0.867 | 🎯 Best Accuracy |
| v23.tflite | 0.993 | 61.8KB | 4289/s | 0.859 | 🎯 Best Accuracy |

#### 🔧 Alternative IoT Scenarios

**For Accuracy-Critical IoT:**
- **Choice**: v23.tflite
- **Accuracy**: 0.993 (best under 100KB)
- **Trade-off**: 61.8KB size

**For Speed-Critical IoT:**
- **Choice**: digit_recognizer_v7.tflite
- **Speed**: 4946 inf/s (fastest under 100KB)
- **Trade-off**: 0.966 accuracy

**For Memory-Constrained IoT:**
- **Choice**: v3.tflite
- **Size**: 38.4KB (smallest with ≥85% accuracy)
- **Trade-off**: 0.979 accuracy

#### 📈 Efficiency Analysis
| Model | Acc/KB | Acc/Param | Parameters | Verdict |
|-------|--------|-----------|------------|---------|
| v3.tflite | 0.0255 | 8.548471615720524 | 114500 | 🎯 **OPTIMAL** |
| digit_recognizer_v3.tflite | 0.0252 | 8.437554585152839 | 114500 | ⚖️ Good |
| digit_recognizer_v7.tflite | 0.0205 | 8.528684907325683 | 113300 | ⚖️ Good |
| v23.tflite | 0.0160 | 6.70886932972241 | 147700 | ⚖️ Good |
| v23.tflite | 0.0161 | 6.722410291130671 | 147700 | ⚖️ Good |

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
