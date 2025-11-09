# Digit Recognition Benchmark Report

## 📊 Executive Summary

- **Test Date**: 2025-11-09 17:16:09
- **Models Tested**: 10 quantized models
- **Test Images**: All available
- **Best Accuracy**: **digit_recognizer_v11.tflite** (0.000)
- **Fastest Model**: **digit_recognizer_v3.tflite** (25412 inf/s)
- **Smallest Model**: **digit_recognizer_v6.tflite** (36.5 KB)

## 📋 Detailed Results

| Model | Directory | Type | Parameters | Size (KB) | Accuracy | Inf/s |
|-------|-----------|------|------------|-----------|----------|-------|
| digit_recognizer_v11.tflite | digit_recognizer_v11_10cls_QAT_QUANT_GRAY | quantized | 2,800,000 | 1370.8 | 0.000 | 21946 |
| digit_recognizer_v12.tflite | digit_recognizer_v12_10cls_QAT_QUANT_GRAY | quantized | 490,000 | 406.7 | 0.000 | 24767 |
| digit_recognizer_v3.tflite | digit_recognizer_v3_10cls_QAT_QUANT_GRAY | quantized | 118,400 | 69.4 | 0.000 | 25412 |
| digit_recognizer_v4.tflite | digit_recognizer_v4_10cls_QAT_QUANT_GRAY | quantized | 79,700 | 61.4 | 0.000 | 24912 |
| digit_recognizer_v6.tflite | digit_recognizer_v6_10cls_QAT_QUANT_GRAY | quantized | 61,500 | 36.5 | 0.000 | 24962 |
| digit_recognizer_v7.tflite | digit_recognizer_v7_10cls_QAT_QUANT_GRAY | quantized | 75,600 | 46.7 | 0.000 | 25114 |
| digit_recognizer_v8.tflite | digit_recognizer_v8_10cls_QAT_QUANT_GRAY | quantized | 602,700 | 396.4 | 0.000 | 22503 |
| digit_recognizer_v9.tflite | digit_recognizer_v9_10cls_QAT_QUANT_GRAY | quantized | 902,500 | 148.6 | 0.000 | 24471 |
| mnist_quantization.tflite | mnist_quantization_10cls_QAT_QUANT_GRAY | quantized | 98,700 | 63.6 | 0.000 | 18737 |
| original_haverland.tflite | original_haverland_10cls_QAT_QUANT_GRAY | quantized | 234,500 | 203.3 | 0.000 | 24232 |

## 🏆 Performance Analysis

### Best by Accuracy
- **Model**: digit_recognizer_v11.tflite
- **Directory**: digit_recognizer_v11_10cls_QAT_QUANT_GRAY
- **Accuracy**: 0.000
- **Speed**: 21946 inf/s
- **Size**: 1370.8 KB

### Fastest Model
- **Model**: digit_recognizer_v3.tflite
- **Directory**: digit_recognizer_v3_10cls_QAT_QUANT_GRAY
- **Speed**: 25412 inf/s
- **Accuracy**: 0.000
- **Size**: 69.4 KB

### Most Efficient (Smallest)
- **Model**: digit_recognizer_v6.tflite
- **Directory**: digit_recognizer_v6_10cls_QAT_QUANT_GRAY
- **Size**: 36.5 KB
- **Accuracy**: 0.000
- **Speed**: 24962 inf/s

## 📈 Visualizations

### Accuracy Vs Speed 20251109 171559 Quantized Full

![Accuracy Vs Speed 20251109 171559 Quantized Full](..\graphs\accuracy_vs_speed_20251109_171559_quantized_full.png)

*Click image to view full resolution*

### Accuracy Vs Size 20251109 171559 Quantized Full

![Accuracy Vs Size 20251109 171559 Quantized Full](..\graphs\accuracy_vs_size_20251109_171559_quantized_full.png)

*Click image to view full resolution*

### Speed Vs Complexity 20251109 171559 Quantized Full

![Speed Vs Complexity 20251109 171559 Quantized Full](..\graphs\speed_vs_complexity_20251109_171559_quantized_full.png)

*Click image to view full resolution*

### Speed Comparison 20251109 171559 Quantized Full

![Speed Comparison 20251109 171559 Quantized Full](..\graphs\speed_comparison_20251109_171559_quantized_full.png)

*Click image to view full resolution*

### Accuracy Comparison 20251109 171559 Quantized Full

![Accuracy Comparison 20251109 171559 Quantized Full](..\graphs\accuracy_comparison_20251109_171559_quantized_full.png)

*Click image to view full resolution*

## 💡 Recommendations

### For High Accuracy Applications
- Use **digit_recognizer_v11.tflite** from digit_recognizer_v11_10cls_QAT_QUANT_GRAY
- Accuracy: 0.000
- Trade-off: 21946 inf/s

### For Real-time Applications
- Use **digit_recognizer_v3.tflite** from digit_recognizer_v3_10cls_QAT_QUANT_GRAY
- Speed: 25412 inf/s
- Trade-off: 0.000 accuracy

### For Resource-Constrained Environments
- Use **digit_recognizer_v6.tflite** from digit_recognizer_v6_10cls_QAT_QUANT_GRAY
- Size: 36.5 KB
- Trade-off: 0.000 accuracy

## 🔧 Technical Details

### Test Configuration
- Quantized Models Only: True
- Use All Datasets: True
- Test Images Count: All available
- Total Models Tested: 10
- Average Accuracy: 0.000
- Average Speed: 23706 inf/s
- Average Model Size: 280.3 KB

### Files Generated
- **CSV Results**: `model_comparison_20251109_171609_quantized_full.csv`
- **Graphs**: 5 visualization files
- **This Report**: `benchmark_report_20251109_171609.md`

---
*Report generated automatically by Digit Recognition Benchmarking Tool*
