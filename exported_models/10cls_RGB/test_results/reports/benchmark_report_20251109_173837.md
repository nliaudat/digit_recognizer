# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2025-11-09 17:38:37
- **Models Tested**: 8 quantized models
- **Test Images**: All available
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.996)
- **Fastest Model**: **digit_recognizer_v7.tflite** (8688 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (38.4 KB)

## 📋 Detailed Results

| Model | Directory | Type | Parameters | Size (KB) | Accuracy | Inf/s |
|-------|-----------|------|------------|-----------|----------|-------|
| digit_recognizer_v12.tflite | digit_recognizer_v12_10cls_QAT_QUANT_RGB | quantized | 493,100 | 407.3 | 0.996 | 1171 |
| digit_recognizer_v9.tflite | digit_recognizer_v9_10cls_QAT_QUANT_RGB | quantized | 905,600 | 149.1 | 0.992 | 3122 |
| digit_recognizer_v4.tflite | digit_recognizer_v4_10cls_QAT_QUANT_RGB | quantized | 104,700 | 78.3 | 0.991 | 8464 |
| original_haverland.tflite | original_haverland_10cls_QAT_QUANT_RGB | quantized | 240,200 | 203.8 | 0.988 | 5606 |
| digit_recognizer_v3.tflite | digit_recognizer_v3_10cls_QAT_QUANT_RGB | quantized | 71,200 | 38.4 | 0.983 | 6871 |
| mnist_quantization.tflite | mnist_quantization_10cls_QAT_QUANT_RGB | quantized | 101,900 | 64.2 | 0.976 | 5936 |
| digit_recognizer_v7.tflite | digit_recognizer_v7_10cls_QAT_QUANT_RGB | quantized | 78,600 | 47.2 | 0.974 | 8688 |
| digit_recognizer_v6.tflite | digit_recognizer_v6_10cls_QAT_QUANT_RGB | quantized | 79,500 | 46.9 | 0.968 | 5054 |

## 🏆 Performance Analysis

### Best by Accuracy
- **Model**: digit_recognizer_v12.tflite
- **Directory**: digit_recognizer_v12_10cls_QAT_QUANT_RGB
- **Accuracy**: 0.996
- **Speed**: 1171 inf/s
- **Size**: 407.3 KB

### Fastest Model
- **Model**: digit_recognizer_v7.tflite
- **Directory**: digit_recognizer_v7_10cls_QAT_QUANT_RGB
- **Speed**: 8688 inf/s
- **Accuracy**: 0.974
- **Size**: 47.2 KB

### Most Efficient (Smallest)
- **Model**: digit_recognizer_v3.tflite
- **Directory**: digit_recognizer_v3_10cls_QAT_QUANT_RGB
- **Size**: 38.4 KB
- **Accuracy**: 0.983
- **Speed**: 6871 inf/s

## 💡 Recommendations

### For High Accuracy Applications
- Use **digit_recognizer_v12.tflite** from digit_recognizer_v12_10cls_QAT_QUANT_RGB
- Accuracy: 0.996
- Trade-off: 1171 inf/s

### For Real-time Applications
- Use **digit_recognizer_v7.tflite** from digit_recognizer_v7_10cls_QAT_QUANT_RGB
- Speed: 8688 inf/s
- Trade-off: 0.974 accuracy

### For Resource-Constrained Environments
- Use **digit_recognizer_v3.tflite** from digit_recognizer_v3_10cls_QAT_QUANT_RGB
- Size: 38.4 KB
- Trade-off: 0.983 accuracy

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
