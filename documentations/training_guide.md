# Training and Fine-Tuning Guide

This guide covers the core training processes in the `digit_recognizer` project, including training from scratch, hyperparameter tuning, and fine-tuning existing models.

## 1. Top-Level Configurations (`parameters.py`)
All scripts rely on `parameters.py` for their global configuration. Before running any training workflow, ensure you have correctly set up:
- `MODEL_ARCHITECTURE`: Select the model you want to train (e.g., `"digit_recognizer_v12"`, `"cnn32"`).
- `BATCH_SIZE` & `EPOCHS`: Standard training loop parameters.
- `LEARNING_RATE` & `OPTIMIZER`: The default configurations for gradient descent.
- `DATA_SOURCES`: Paths to your datasets (augmented, real-world, etc.) and their respective weights in the pipeline.
- `QUANTIZE_MODEL`, `USE_QAT`, `ESP_DL_QUANTIZE`: The quantization flags for post-training quantization and quantization-aware training. Automatic overrides exist for certain architectures like `high_accuracy_validator`.

## 2. Standard Training (`train.py`)
The `train.py` script is the central entry point for training a model from scratch.

### Basic Usage
To train the architecture currently selected in `parameters.py`:
```bash
python train.py
```

### Command-Line Arguments
You can override the `MODEL_ARCHITECTURE` dynamically using arguments, which allows you to run multiple instances concurrently in separate terminals without editing `parameters.py` each time.

- `--train [list of models]`: Train specific architectures sequentially or concurrently.
  ```bash
  python train.py --train digit_recognizer_v4
  python train.py --train cnn32 digit_recognizer_v12
  ```
- `--test_all_models`: Run a quick 1-epoch sanity check on every available architecture.
- `--train_all`: Train every architecture sequentially.
- `--use_tuner`: Run hyperparameter tuning before initiating the main training loop.
- `--debug`: Enable verbose TensorFlow logging and extra prints.
- `--no_cleanup`: Skip the automatic cleanup of checkpoint files post-training.

### Output
Models are heavily processed and exported to the `exported_models/` directory, saving:
- `best_model.keras`
- `<model>_quantized.tflite`
- `<model>_float.tflite`
- A visual training history plot.

## 3. Fine-Tuning Models (`fine-tune.py`)
If you have a pre-trained model and want to adapt it to new data (e.g., augmented datasets or correcting bad predictions) without starting from scratch, use `fine-tune.py`. By default, this script handles loading original datasets combined with heavily augmented data to avoid *catastrophic forgetting*.

### Usage
```bash
python fine-tune.py --model_path exported_models/<your_model_dir>/best_model.keras
```

### Command-Line Arguments
- `--strategy`: Strategy for fine-tuning (`full`, `last_layer`, `feature_extractor`).
- `--data_ratio`: Ratio of the original dataset to retain during training (default: 0.1).
- `--augmented_ratio`: Ratio of the augmented dataset to mix in (default: 0.9).
- `--learning_rate_multiplier`: Multiplier for `params.LEARNING_RATE` to ensure smaller weight updates (default: 0.5).

## 4. Retraining Specifically on Bad Predictions (`retrain.py`)
`retrain.py` is specifically built to adapt models to failed predictions dynamically. It differs from `fine-tune.py` by targeting a specific folder of difficult images and giving them a heavy weight multiplier in the data loader.

### Usage
```bash
python retrain.py --model_path exported_models/<your_model_dir>/best_model.keras --bad_images_dir datasets/real_integra_bad_predictions
```

### Key Differences
- Automatically appends the bad predictions folder to `params.DATA_SOURCES` if it isn't listed, or overrides its weight if it is.
- Implements a very low learning rate by default (`lr=0.0001`).
- Exports a new model labeled `_finetuned` using standard PTQ INT8 quantization.

## 5. Hyperparameter Tuning (`tuner.py`)
You can optimize the hyperparameters via KerasTuner. The module utilizes a "guaranteed unique" search strategy to iterate over different optimizers, learning rates, and batch sizes efficiently.

You can trigger this automatically from `train.py` using `--use_tuner` or it handles fallback execution manually within the codebase. The best parameters are summarized in a corresponding `best_hyperparameters.json` and a `.py` file for easy copying into `parameters.py`.
