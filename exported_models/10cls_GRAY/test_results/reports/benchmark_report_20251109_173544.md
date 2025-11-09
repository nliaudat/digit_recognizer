# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2025-11-09 17:35:44
- **Models Tested**: 10 quantized models
- **Test Images**: All available
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.996)
- **Fastest Model**: **digit_recognizer_v3.tflite** (8706 inf/s)
- **Smallest Model**: **digit_recognizer_v6.tflite** (36.5 KB)

## 📋 Detailed Results

| Model | Directory | Type | Parameters | Size (KB) | Accuracy | Inf/s |
|-------|-----------|------|------------|-----------|----------|-------|
| digit_recognizer_v12.tflite | digit_recognizer_v12_10cls_QAT_QUANT_GRAY | quantized | 490,000 | 406.7 | 0.996 | 2048 |
| digit_recognizer_v8.tflite | digit_recognizer_v8_10cls_QAT_QUANT_GRAY | quantized | 602,700 | 396.4 | 0.994 | 956 |
| digit_recognizer_v9.tflite | digit_recognizer_v9_10cls_QAT_QUANT_GRAY | quantized | 902,500 | 148.6 | 0.993 | 2136 |
| digit_recognizer_v11.tflite | digit_recognizer_v11_10cls_QAT_QUANT_GRAY | quantized | 2,800,000 | 1370.8 | 0.993 | 1384 |
| digit_recognizer_v4.tflite | digit_recognizer_v4_10cls_QAT_QUANT_GRAY | quantized | 79,700 | 61.4 | 0.990 | 6768 |
| original_haverland.tflite | original_haverland_10cls_QAT_QUANT_GRAY | quantized | 234,500 | 203.3 | 0.987 | 6256 |
| digit_recognizer_v3.tflite | digit_recognizer_v3_10cls_QAT_QUANT_GRAY | quantized | 118,400 | 69.4 | 0.985 | 8706 |
| mnist_quantization.tflite | mnist_quantization_10cls_QAT_QUANT_GRAY | quantized | 98,700 | 63.6 | 0.977 | 6431 |
| digit_recognizer_v7.tflite | digit_recognizer_v7_10cls_QAT_QUANT_GRAY | quantized | 75,600 | 46.7 | 0.974 | 4907 |
| digit_recognizer_v6.tflite | digit_recognizer_v6_10cls_QAT_QUANT_GRAY | quantized | 61,500 | 36.5 | 0.971 | 3666 |

## 🏆 Performance Analysis

### Best by Accuracy
- **Model**: digit_recognizer_v12.tflite
- **Directory**: digit_recognizer_v12_10cls_QAT_QUANT_GRAY
- **Accuracy**: 0.996
- **Speed**: 2048 inf/s
- **Size**: 406.7 KB

### Fastest Model
- **Model**: digit_recognizer_v3.tflite
- **Directory**: digit_recognizer_v3_10cls_QAT_QUANT_GRAY
- **Speed**: 8706 inf/s
- **Accuracy**: 0.985
- **Size**: 69.4 KB

### Most Efficient (Smallest)
- **Model**: digit_recognizer_v6.tflite
- **Directory**: digit_recognizer_v6_10cls_QAT_QUANT_GRAY
- **Size**: 36.5 KB
- **Accuracy**: 0.971
- **Speed**: 3666 inf/s

## 💡 Recommendations

### For High Accuracy Applications
- Use **digit_recognizer_v12.tflite** from digit_recognizer_v12_10cls_QAT_QUANT_GRAY
- Accuracy: 0.996
- Trade-off: 2048 inf/s

### For Real-time Applications
- Use **digit_recognizer_v3.tflite** from digit_recognizer_v3_10cls_QAT_QUANT_GRAY
- Speed: 8706 inf/s
- Trade-off: 0.985 accuracy

### For Resource-Constrained Environments
- Use **digit_recognizer_v6.tflite** from digit_recognizer_v6_10cls_QAT_QUANT_GRAY
- Size: 36.5 KB
- Trade-off: 0.971 accuracy

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
