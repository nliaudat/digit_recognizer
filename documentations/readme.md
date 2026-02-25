# Digit Recognizer Documentations

Welcome to the documentation folder. This directory contains detailed guides on how to utilize the various executable scripts and modules provided in this repository. 

Whether you are looking to train a new model from scratch, benchmark compiled TFLite models against one another, or deep-dive into the mathematical impacts of INT8 quantization, refer to the guides listed below.

## Available Guides

1. **[Training and Fine-Tuning Guide](training_guide.md)**
   - Understanding `parameters.py`
   - Training models from scratch (`train.py`)
   - Fine-tuning existing checkpoints (`fine-tune.py`)
   - Retraining specifically on bad predictions (`retrain.py`)
   - Hyperparameter Optimization (`tuner.py`)

2. **[Benchmarking and Prediction Guide](benchmarking_and_prediction.md)**
   - Running isolated image inferences (`predict.py`)
   - Mass evaluating models arrays (`bench_predict.py`)
   - Visualizing accuracy vs. parameter size comparisons

3. **[Analysis and Debugging Guide](analysis_and_debugging.md)**
   - Evaluating Quantization SNR and Error Distributions (`run_quant_analysis.py`)
   - Diagnosing Dataset imbalances & Preprocessing Flow (`diagnose_training.py`)
   - Inspecting underlying Model Architecture outputs (`debug.py`)
