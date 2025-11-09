# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2025-11-09 17:00:49
- **Models Tested**: 8 quantized models
- **Test Images**: All available
- **Best Accuracy**: **digit_recognizer_v12.tflite** (0.993)
- **Fastest Model**: **digit_recognizer_v4.tflite** (7356 inf/s)
- **Smallest Model**: **digit_recognizer_v3.tflite** (38.4 KB)

## 📋 Detailed Results

| Model | Directory | Type | Parameters | Size (KB) | Accuracy | Inf/s |
|-------|-----------|------|------------|-----------|----------|-------|
| digit_recognizer_v12.tflite | digit_recognizer_v12_10cls_QAT_QUANT_RGB | quantized | 493,100 | 407.3 | 0.993 | 1028 |
| digit_recognizer_v9.tflite | digit_recognizer_v9_10cls_QAT_QUANT_RGB | quantized | 905,600 | 149.1 | 0.990 | 2512 |
| digit_recognizer_v4.tflite | digit_recognizer_v4_10cls_QAT_QUANT_RGB | quantized | 104,700 | 78.3 | 0.988 | 7356 |
| original_haverland.tflite | original_haverland_10cls_QAT_QUANT_RGB | quantized | 240,200 | 203.8 | 0.983 | 4932 |
| digit_recognizer_v3.tflite | digit_recognizer_v3_10cls_QAT_QUANT_RGB | quantized | 71,200 | 38.4 | 0.977 | 4146 |
| mnist_quantization.tflite | mnist_quantization_10cls_QAT_QUANT_RGB | quantized | 101,900 | 64.2 | 0.970 | 5300 |
| digit_recognizer_v7.tflite | digit_recognizer_v7_10cls_QAT_QUANT_RGB | quantized | 78,600 | 47.2 | 0.968 | 7334 |
| digit_recognizer_v6.tflite | digit_recognizer_v6_10cls_QAT_QUANT_RGB | quantized | 79,500 | 46.9 | 0.961 | 4406 |

## 🏆 Performance Analysis

### Best by Accuracy
- **Model**: digit_recognizer_v12.tflite
- **Directory**: digit_recognizer_v12_10cls_QAT_QUANT_RGB
- **Accuracy**: 0.993
- **Speed**: 1028 inf/s
- **Size**: 407.3 KB

### Fastest Model
- **Model**: digit_recognizer_v4.tflite
- **Directory**: digit_recognizer_v4_10cls_QAT_QUANT_RGB
- **Speed**: 7356 inf/s
- **Accuracy**: 0.988
- **Size**: 78.3 KB

### Most Efficient (Smallest)
- **Model**: digit_recognizer_v3.tflite
- **Directory**: digit_recognizer_v3_10cls_QAT_QUANT_RGB
- **Size**: 38.4 KB
- **Accuracy**: 0.977
- **Speed**: 4146 inf/s

## 📈 Visualizations

### Accuracy Vs Speed 20251109 170044 Quantized Full

![Accuracy Vs Speed 20251109 170044 Quantized Full](..\graphs\accuracy_vs_speed_20251109_170044_quantized_full.png)

*Click image to view full resolution*

### Accuracy Vs Size 20251109 170044 Quantized Full

![Accuracy Vs Size 20251109 170044 Quantized Full](..\graphs\accuracy_vs_size_20251109_170044_quantized_full.png)

*Click image to view full resolution*

### Speed Vs Complexity 20251109 170044 Quantized Full

![Speed Vs Complexity 20251109 170044 Quantized Full](..\graphs\speed_vs_complexity_20251109_170044_quantized_full.png)

*Click image to view full resolution*

### Speed Comparison 20251109 170044 Quantized Full

![Speed Comparison 20251109 170044 Quantized Full](..\graphs\speed_comparison_20251109_170044_quantized_full.png)

*Click image to view full resolution*

### Accuracy Comparison 20251109 170044 Quantized Full

![Accuracy Comparison 20251109 170044 Quantized Full](..\graphs\accuracy_comparison_20251109_170044_quantized_full.png)

*Click image to view full resolution*

## 💡 Recommendations

### For High Accuracy Applications
- Use **digit_recognizer_v12.tflite** from digit_recognizer_v12_10cls_QAT_QUANT_RGB
- Accuracy: 0.993
- Trade-off: 1028 inf/s

### For Real-time Applications
- Use **digit_recognizer_v4.tflite** from digit_recognizer_v4_10cls_QAT_QUANT_RGB
- Speed: 7356 inf/s
- Trade-off: 0.988 accuracy

### For Resource-Constrained Environments
- Use **digit_recognizer_v3.tflite** from digit_recognizer_v3_10cls_QAT_QUANT_RGB
- Size: 38.4 KB
- Trade-off: 0.977 accuracy

## 🔧 Technical Details

### Test Configuration
- Quantized Models Only: True
- Use All Datasets: True
- Test Images Count: All available
- Total Models Tested: 8
- Average Accuracy: 0.979
- Average Speed: 4627 inf/s
- Average Model Size: 129.4 KB

### Files Generated
- **CSV Results**: `model_comparison_20251109_170049_quantized_full.csv`
- **Graphs**: 5 visualization files
- **This Report**: `benchmark_report_20251109_170049.md`

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
