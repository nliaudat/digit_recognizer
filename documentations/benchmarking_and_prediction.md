# Benchmarking and Prediction Guide

This guide outlines how to use the prediction and testing scripts to benchmark model accuracy, visualize predictions, and measure inference latency on PC/validation hardware.

## 1. Individual Prediction (`predict.py`)
The `predict.py` script is designed to run a single inference on a given image using a trained TFLite model. It acts as a lightweight sanity check for your quantization pipeline and preprocessing logic.

### Usage
To predict using the *latest* trained model from `exported_models/`:
```bash
python predict.py --img path/to/image.jpg
```

### Features
- **Auto-Model Selection:** If no model is specified, it scans `exported_models/` and selects the most recently compiled TFLite model, prioritizing quantized versions.
- **Random Dataset Test:** If `--img` is omitted, it randomly selects and loads an image from your primary dataset listed in `params.DATA_SOURCES`.
- **Known-Pattern Test:** The `--test` flag generates a synthetic image array (a bright vertical line mimicking a "1") to verify the TFLite interpreter and preprocessing bounds are functioning correctly before testing real images.
- **Detailed Preprocessing Debugging:** The `--debug` flag prints specific array ranges, shape transformations, and quantization scale details.

## 2. Mass Benchmarking (`bench_predict.py`)
When you want to evaluate and compare the comprehensive performance of *multiple* models across your entire test dataset, use `bench_predict.py`.

### Features
This script performs mass sweeps across the `exported_models/` folder. For each model found, it calculates:
- Accuracy (percentage of correct classes).
- Inference latency (measured in milliseconds per frame).
- Inferences per second.
- Total model size (KB) and Parameter counting.

### Usage
```bash
python bench_predict.py
```

It outputs a highly detailed Summary Table directly in your terminal comparing all architectures.

### Output Visualizations
`bench_predict.py` automatically generates a suite of Matplotlib comparison charts in `exported_models/test_results/graphs/`, including:
1. Accuracy vs Speed (Inferences per Second).
2. Accuracy vs Model Size.
3. Speed vs Model Complexity (Parameters).
4. Horizontal Bar Charts for simple ranking.

These graphs make it extremely easy to identify the Pareto frontier (the best tradeoff between memory limits of an ESP32, speed, and accuracy).

## 3. Training Results Aggregation (`exported_models/benchmark.py`)
While `bench_predict.py` runs active inferences on the models, `exported_models/benchmark.py` serves as an offline parser that dynamically generates performance reports without needing to run the dataset.

### Features
This script scans through all sub-directories within `exported_models/` looking for `training_results.csv` files generated during the training phases. It aggregates them into a single file and estimates memory properties:
- **Tensor Arena Estimation:** Calculates how much active RAM an ESP32 would require to run the TFLite model.
- **CPU Operations (OPS):** Estimates the computational cost required per inference. 
- **Efficiency Plotting:** Automatically generates comprehensive scatter plots comparing Model Size versus Accuracy, Memory Efficiency, and Computational Costs.

### Usage
Run the script from inside the `exported_models/` directory:
```bash
cd exported_models/
python benchmark.py
```

This updates the respective `training_results.csv` lines with the newly calculated `tensor_arena_kb` and `cpu_ops_millions`, giving you a global `benchmark_results.csv` summary of your entire experimentation history.
