#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py – Central training script with robust quantisation handling.

Features
--------
* Deterministic seeding and optional GPU configuration.
* Automatic validation / correction of the three quantisation flags.
* Unified loss selection (categorical vs sparse).
* Proper PTQ representative data (float32 [0, 1]).
* QAT aware representative data that uses the *training* preprocessing.
* QAT models skip the representative dataset (they already embed scales).
* Explicit model.build() before any training / conversion.
* Detailed callbacks (early stop, LR scheduler, CSV logger, TFLite checkpoint,
  tqdm progress bar, optional augmentation safety monitor).
* Comprehensive final reporting (TXT + CSV + model summary file).
* Optional hyper parameter tuning (via `tuner.py`).
* Integrated training analysis and automatic cleanup.
"""

# --------------------------------------------------------------------------- #
# Standard library imports
# --------------------------------------------------------------------------- #
import os
import sys

# Initial delay: configuration will be handled by interactive_digit_config in main()

import argparse
import logging
import warnings
import random
import json
import shutil
import gc
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Third party imports
# --------------------------------------------------------------------------- #
import numpy as np
import tensorflow as tf
from utils.keras_helper import keras
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
#  Project imports (core utilities)
# --------------------------------------------------------------------------- #
from models import create_model, compile_model, model_summary
from utils import get_data_splits,  preprocess_for_training, preprocess_for_inference, get_calibration_data, suppress_all_output
from utils.preprocess import (
    validate_preprocessing_consistency,
    get_qat_training_format,
    get_preprocessing_info,
)
from utils.data_pipeline import create_tf_dataset_from_arrays
from utils.custom_logger import log_print
from utils.multi_source_loader import clear_cache

from utils.augmentation import (
    create_augmentation_pipeline,
    apply_augmentation_to_dataset,
    test_augmentation_pipeline,
    print_augmentation_summary,
    create_augmentation_safety_monitor,
    setup_augmentation_for_training,
)

# Parameter handling: import here for module-level defaults (e.g. function signatures).
# main() will do a deferred re-import *after* env-vars are set by interactive_digit_config,
# ensuring NB_CLASSES / INPUT_CHANNELS are correct before any model is created.
import parameters as params

# --------------------------------------------------------------------------- #
#  Optional QAT import
# --------------------------------------------------------------------------- #
try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except Exception:  # pragma: no cover
    log_print(
        "⚠️  tensorflow-model-optimization not available. Install with: pip install tensorflow-model-optimization",
        level=1,
    )
    QAT_AVAILABLE = False
    tfmot = None

# --------------------------------------------------------------------------- #
#  Import the refactored helper classes
# --------------------------------------------------------------------------- #
from utils.train_modelmanager   import TFLiteModelManager
from utils.train_trainingmonitor import TrainingMonitor
from utils.train_checkpoint     import TFLiteCheckpoint
from utils.train_progressbar    import TQDMProgressBar

from utils.train_qat_helper     import (
    create_qat_model,
    create_qat_representative_dataset,
    validate_qat_data_flow,
    _is_qat_model,
    check_qat_compatibility,
    debug_preprocessing_flow,
    diagnose_quantization_settings,
    validate_quantization_combination,
    validate_qat_data_consistency,
    validate_complete_qat_setup,
    verify_qat_model,
    debug_qat_layers,
    check_qat_gradient_flow,
    diagnose_qat_output_behavior
)


from utils.train_callbacks import create_callbacks
from utils.train_helpers import (
    print_training_summary, 
    save_model_summary_to_file,
    save_training_config,
    save_training_csv
)

from utils.train_cleaning import cleanup_training_directory, cleanup_multiple_training_runs

# --------------------------------------------------------------------------- #
#  Import analysis functions
# --------------------------------------------------------------------------- #
from utils.train_analyse import (
    evaluate_keras_model,
    evaluate_tflite_model,
    get_keras_model_size,
    get_tflite_model_size,
    measure_keras_inference_time,
    measure_tflite_inference_time,
    analyze_quantization_impact,
    training_diagnostics,
    debug_model_architecture,
    verify_model_predictions,
    analyze_confusion_matrix,
    analyze_training_history,
    model_size_analysis
)


from utils.quantization_analysis import QuantizationAnalyzer

# --------------------------------------------------------------------------- #
#  Suppress TF logs unless --debug is requested
# --------------------------------------------------------------------------- #
def suppress_tf_logs_if_needed():
    if "--debug" not in sys.argv:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


suppress_tf_logs_if_needed()

# --------------------------------------------------------------------------- #
#  TensorFlow logging configuration
# --------------------------------------------------------------------------- #
def setup_tensorflow_logging(debug: bool = False):
    if debug:
        tf.get_logger().setLevel("INFO")
        tf.autograph.set_verbosity(3)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        logging.getLogger().setLevel(logging.INFO)
    else:
        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(0)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "0"
        try:
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
        except Exception:
            pass
        warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
        warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("h5py").setLevel(logging.ERROR)
        logging.getLogger("numexpr").setLevel(logging.ERROR)

        # Hide the deprecation warning about tf.lite.Interpreter
        warnings.filterwarnings(
            "ignore",
            message=r".*tf\.lite\.Interpreter is deprecated.*",
            category=UserWarning,
        )


# --------------------------------------------------------------------------- #
#  Deterministic seeding
# --------------------------------------------------------------------------- #
def set_all_seeds(seed=params.SHUFFLE_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    tf.config.experimental.enable_op_determinism()  # still in experimental as of TF 2.x


# --------------------------------------------------------------------------- #
#  Argument parsing
# --------------------------------------------------------------------------- #
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Digit Recognition Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Logging & Debugging ---
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose TensorFlow logging and extra debug prints.\n"
             "Useful for troubleshooting model graph issues or tracking detailed execution flow."
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Custom prefix for the output directory name. Useful for organizing experiments."
    )
    
    # --- Training Modes ---
    parser.add_argument(
        "--train_all",
        action="store_true",
        help="Train every architecture listed in params.AVAILABLE_MODELS sequentially.\n"
             "Ideal for running large benchmarking sweeps overnight."
    )
    parser.add_argument(
        "--train",
        nargs="+",
        choices=params.AVAILABLE_MODELS,
        help="Train only the explicitly listed architectures.\n"
             "Example: --train digit_recognizer_v4 digit_recognizer_v17"
    )
    parser.add_argument(
        "--test_all_models",
        action="store_true",
        help="Run a rapid sanity check on every architecture.\n"
             "Compiles each model and trains for exactly 1 epoch using a tiny subset of data "
             "to ensure there are no syntax errors or shape mismatches."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=params.AVAILABLE_MODELS,
        help="Specify a single architecture to train (overrides parameters.py).\n"
             "Example: --model digit_recognizer_v17"
    )
    
    # --- Hyperparameter Tuning ---
    parser.add_argument(
        "--use_tuner",
        action="store_true",
        help="Run Keras Tuner hyperparameter search before training.\n"
             "Automatically searches for optimal learning rates, batch sizes, and optimizers."
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Maximum number of tuner configurations to test (default: 100).\n"
             "Higher values explore more combinations but take significantly longer."
    )
    
    # --- Training Enhancements ---
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Enable experimental training features (if implemented in parameters.py).\n"
             "E.g., Mixed precision, gradient accumulation, or learning rate cyclers."
    )
    
    # --- Post-Training Options ---
    parser.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Skip deletion of intermediate checkpoint files after training.\n"
             "Keeps the saved_model checkpoints on disk (requires more storage space)."
    )
    parser.add_argument(
        "--no_analysis",
        action="store_true",
        help="Skip the comprehensive evaluation phase after training completes.\n"
             "Disables TFLite conversion, confusion matrix generation, and inference timing."
    )
    
    # --- Hyperparameter Overrides (from legacy train_super_validator) ---
    parser.add_argument("--epochs", type=int, default=None, help="Override maximum number of training epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate.")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Override learning rate warmup epochs.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay parameter.")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Override label smoothing factor.")
    parser.add_argument("--loss_type", type=str, default=None, 
                        help="Override the loss function type.\n"
                             "Possibilities:\n"
                             "  - IntelligentFocalLossController: Adaptive Focal Loss (Default)\n"
                             "  - focal_loss: Standard Focal Loss with fixed gamma\n"
                             "  - sparse_categorical_crossentropy: Standard CrossEntropy for integer labels\n"
                             "  - categorical_crossentropy: Standard CrossEntropy for one-hot labels")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Override Focal Loss Gamma.")
    parser.add_argument("--rotation-range", type=float, default=None, help="Override maximum rotation augmentation in degrees.")
    parser.add_argument("--no-mixup", action="store_true", help="Disable Mixup augmentation.")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--focal-loss", action="store_true", help="Explicitly enable Focal Loss.")
    parser.add_argument("--no-focal-loss", action="store_true", help="Explicitly disable Focal Loss.")
    
    # --- Advanced Features ---
    parser.add_argument("--cutmix", action="store_true", help="Enable CutMix augmentation.")
    parser.add_argument("--no-random-erasing", action="store_true", help="Disable Random Erasing augmentation.")
    parser.add_argument("--optimizer", type=str, default=None, help="Override the optimizer (e.g. adamw).")
    parser.add_argument("--lr-scheduler", type=str, default=None, help="Override the LR scheduler (e.g. cosine).")
    parser.add_argument("--no-dynamic-weights", action="store_true", help="Disable dynamic per-class weighting.")

    # --- Dynamic Scheduler / Optimizer ---
    parser.add_argument("--dynamic-scheduler", action="store_true", default=None,
                        help="Enable DynamicSchedulerController (USE_DYNAMIC_SCHEDULER=True).")
    parser.add_argument("--no-dynamic-scheduler", action="store_true",
                        help="Disable DynamicSchedulerController (USE_DYNAMIC_SCHEDULER=False).")
    parser.add_argument("--dynamic-optimizer", action="store_true",
                        help="Enable mid-training optimizer switch (USE_DYNAMIC_OPTIMIZER=True). Experimental.")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Disable LR warm-up (USE_LR_WARMUP=False).")
    parser.add_argument("--lr-reset-fraction", type=float, default=None,
                        help="Override LR_SCHEDULER_RESET_FRACTION (fraction of base LR restored on scheduler switch).")

    # --- Resume Training ---
    parser.add_argument("--resume", type=str, default="",
        help="Path to a best_model.keras file to resume from.")
    parser.add_argument("--initial-epoch", type=int, default=0,
        help="Epoch to resume from (auto-filled by retrain_all.py from training_log.csv).")

    # --- Dataset Configuration Overrides ---
    parser.add_argument("--classes", type=int, choices=[10, 100], help="Number of classes (10 or 100).")
    parser.add_argument("--color", type=str, choices=["rgb", "gray"], help="Color mode (rgb or gray).")

    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  GPU configuration
# --------------------------------------------------------------------------- #
def setup_gpu():
    """Detect GPUs, enable memory growth / limits, and optionally MirroredStrategy."""
    print("🔧 Configuring hardware …")
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    print(f"📋 Devices – CPUs: {len(cpus)}  GPUs: {len(gpus)}")

    if not params.USE_GPU:
        print("🔧 GPU usage disabled in parameters – falling back to CPU")
        return None

    if not gpus:
        print("❌ No GPUs detected – training will run on CPU")
        return None

    try:
        for gpu in gpus:
            if params.GPU_MEMORY_GROWTH:
                tf.config.set_memory_growth(gpu, True)
                print("   ✅ Memory growth enabled")
            if params.GPU_MEMORY_LIMIT is not None:
                tf.config.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.VirtualDeviceConfiguration(
                            memory_limit=params.GPU_MEMORY_LIMIT
                        )
                    ],
                )
                print(f"   ✅ Memory limit set to {params.GPU_MEMORY_LIMIT} MiB")
        # Test a tiny op
        with tf.device("/GPU:0"):
            _ = tf.constant([1.0, 2.0, 3.0]) + 1.0
        if len(gpus) > 1:
            print(f"   🚀 Using {len(gpus)} GPUs with MirroredStrategy")
            return tf.distribute.MirroredStrategy()
        else:
            print("   ✅ Single GPU configuration OK")
            return None
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  GPU configuration failed: {exc}")
        return None


# --------------------------------------------------------------------------- #
#  Training Analysis Functions
# --------------------------------------------------------------------------- #
def run_comprehensive_analysis(model, history, training_dir, x_test, y_test, debug=False):
    """
    Run comprehensive analysis on training results
    """
    print("\n📊 Running comprehensive analysis...")
    
    try:
        # Create analysis directory
        analysis_dir = os.path.join(training_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 1. Training History Analysis
        if history and hasattr(history, 'history'):
            print("📈 Analyzing training history...")
            analyze_training_history(os.path.join(training_dir, "training_log.csv"), analysis_dir)
        
        # 2. Model Performance Analysis
        print("🧪 Analyzing model performance...")
        analyze_confusion_matrix(model, x_test, y_test, analysis_dir)
        
        # 3. Verify model predictions
        print("✅ Verifying model predictions...")
        sample_accuracy = verify_model_predictions(model, x_test[:100], y_test[:100])
        
        # 4. Model size analysis
        print("📦 Analyzing model sizes...")
        model_size_analysis(training_dir)
        
        # 5. Training diagnostics
        print("🔍 Running training diagnostics...")
        training_diagnostics(model, x_test[:100], y_test[:100], x_test[:50], y_test[:50], debug)
        
        print(f"✅ Comprehensive analysis saved to: {analysis_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


# --------------------------------------------------------------------------- #
#  Helper: quick sanity check of every architecture (used by --test_all_models)
# --------------------------------------------------------------------------- #
def test_all_models(debug: bool = False):
    """Instantiate each architecture, train a few epochs and report accuracy."""
    original = params.MODEL_ARCHITECTURE
    results = {}

    print("\n🧪 Testing all available model architectures")
    print("=" * 60)

    for arch in params.AVAILABLE_MODELS:
        print(f"\n🔍 Architecture: {arch}")
        params.MODEL_ARCHITECTURE = arch
        try:
            model, hist, out_dir = train_model(debug=debug)
            if model is None:
                raise RuntimeError("Training returned None")
            # Load test data for a quick evaluation
            (_, _), (_, _), (x_test_raw, y_test_raw) = get_data_splits()
            x_test = preprocess_for_inference(x_test_raw)
            if arch == "original_haverland":
                y_test = keras.utils.to_categorical(
                    y_test_raw, params.NB_CLASSES
                )
            else:
                y_test = y_test_raw
            test_acc = model.evaluate(x_test, y_test, verbose=0)[1]
            results[arch] = {"accuracy": test_acc, "dir": out_dir}
            print(f"✅ {arch} – test accuracy: {test_acc:.4f}")
        except Exception as exc:
            print(f"❌ {arch} failed: {exc}")
            results[arch] = {"accuracy": 0.0, "dir": None}
        finally:
            keras.backend.clear_session()

    # Restore original architecture
    params.MODEL_ARCHITECTURE = original

    # Summary table
    print("\n" + "=" * 60)
    print("🏆 MODEL COMPARISON")
    print("=" * 60)
    sorted_res = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for i, (arch, info) in enumerate(sorted_res, 1):
        print(f"{i:2d}. {arch:30} → {info['accuracy']:.4f}")

    return results


# --------------------------------------------------------------------------- #
#  Helper: train a *specific* list of architectures (used by --train)
# --------------------------------------------------------------------------- #
def train_specific_models(models_to_train, debug: bool = False, no_cleanup: bool = False, full_analysis: bool = True):
    """Train the supplied architectures sequentially."""
    original = params.MODEL_ARCHITECTURE
    results = {}

    print(f"\n🚀 Training {len(models_to_train)} architectures")
    print("=" * 60)

    for arch in models_to_train:
        print(f"\n🎯 Architecture: {arch}")
        params.MODEL_ARCHITECTURE = arch
        try:
            model, hist, out_dir = train_model(debug=debug, no_cleanup=no_cleanup, full_analysis=full_analysis)
            if model is None:
                raise RuntimeError("Training returned None")
            # Quick test accuracy
            (_, _), (_, _), (x_test_raw, y_test_raw) = get_data_splits()
            x_test = preprocess_for_inference(x_test_raw)
            if arch == "original_haverland":
                y_test = keras.utils.to_categorical(
                    y_test_raw, params.NB_CLASSES
                )
            else:
                y_test = y_test_raw
            test_acc = model.evaluate(x_test, y_test, verbose=0)[1]
            results[arch] = {"accuracy": test_acc, "dir": out_dir}
            print(f"✅ {arch} – test accuracy: {test_acc:.4f}")
        except Exception as exc:
            print(f"❌ {arch} failed: {exc}")
            results[arch] = {"accuracy": 0.0, "dir": None}
        finally:
            keras.backend.clear_session()

    params.MODEL_ARCHITECTURE = original

    # -----------------------------------------------------------------
    #  Summary of the specific model run
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🏁 TRAINING SUMMARY")
    print("=" * 60)
    sorted_res = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for i, (arch, info) in enumerate(sorted_res, 1):
        print(f"{i:2d}. {arch:30} → {info['accuracy']:.4f}")

    return results


# --------------------------------------------------------------------------- #
#  Entry point – parse arguments and dispatch to the appropriate mode
# --------------------------------------------------------------------------- #
def main():
    global params  # declare before any use of params in this scope
    # Now parse arguments
    args = parse_arguments()

    # Apply overrides to params module
    if args.model is not None:
        params.MODEL_ARCHITECTURE = args.model
    if args.epochs is not None:
        params.EPOCHS = args.epochs
    if args.batch is not None:
        params.BATCH_SIZE = args.batch
    if args.lr is not None:
        params.LEARNING_RATE = args.lr
    if args.focal_gamma is not None:
        params.FOCAL_GAMMA = args.focal_gamma
    if args.rotation_range is not None:
        params.AUGMENTATION_ROTATION_RANGE = args.rotation_range
    
    if args.no_mixup:
        params.USE_MIXUP = False
    if args.no_mixed_precision:
        params.USE_MIXED_PRECISION = False
    
    # Loss type overrides - CLI --loss_type takes precedence
    if args.loss_type:
        params.LOSS_TYPE = args.loss_type
    elif args.focal_loss:
        params.LOSS_TYPE = "focal_loss"
    elif args.no_focal_loss:
        params.LOSS_TYPE = "sparse_categorical_crossentropy"

    # NEW: Centralized interactive configuration handling
    from utils.input_helper import interactive_digit_config
    nb_classes, channels = interactive_digit_config(override_classes=args.classes, override_color=args.color)
    
    # Sync with parameters module after it gets imported
    # (The actual import happens after this in the original structure, 
    # but we need to ensure the env vars are set FIRST)
    
    # Re-import parameters now that env vars are set
    import parameters as p
    params = p
    
    if args.classes is not None:
        params.NB_CLASSES = args.classes
    if args.color is not None:
        params.INPUT_CHANNELS = 1 if args.color == "gray" else 3
    
    # Update derived parameters after potential CLI overrides
    if args.classes is not None or args.color is not None:
        params.update_derived_parameters()

    if args.warmup_epochs is not None:
        params.LR_WARMUP_EPOCHS = args.warmup_epochs
    if args.weight_decay is not None:
        params.ADAMW_WEIGHT_DECAY = args.weight_decay
    if args.cutmix:
        params.USE_CUTMIX = True
    if args.no_random_erasing:
        params.USE_RANDOM_ERASING = False
    if args.optimizer is not None:
        params.OPTIMIZER_TYPE = args.optimizer
    if args.lr_scheduler is not None:
        params.LR_SCHEDULER_TYPE = args.lr_scheduler
    if args.task_name is not None:
        params.TASK_NAME = args.task_name
    if args.no_dynamic_weights:
        params.USE_DYNAMIC_WEIGHTS = False
    if args.label_smoothing is not None:
        params.LABEL_SMOOTHING = args.label_smoothing
    if args.dynamic_scheduler:
        params.USE_DYNAMIC_SCHEDULER = True
    if args.no_dynamic_scheduler:
        params.USE_DYNAMIC_SCHEDULER = False
    if args.dynamic_optimizer:
        params.USE_DYNAMIC_OPTIMIZER = True
    if args.no_warmup:
        params.USE_LR_WARMUP = False
    if args.lr_reset_fraction is not None:
        params.LR_SCHEDULER_RESET_FRACTION = args.lr_reset_fraction
    if args.resume:
        params.RESUME_MODEL_PATH = args.resume
    if args.initial_epoch:
        params.INITIAL_EPOCH = args.initial_epoch

    # -----------------------------------------------------------------
    #  Hyper parameter tuning mode
    # -----------------------------------------------------------------
    if args.use_tuner:
        from tuner import run_architecture_tuning

        print("🚀 Running hyper parameter tuning …")
        # Load a *small* subset for faster tuning
        (x_train_raw, y_train_raw), (x_val_raw, y_val_raw), _ = get_data_splits()
        x_train = preprocess_for_training(x_train_raw)
        x_val   = preprocess_for_training(x_val_raw)

        # Only one-hot encode if specifically using categorical_crossentropy
        # (original_haverland default or manual setting)
        if params.LOSS_TYPE == "categorical_crossentropy":
            y_train = keras.utils.to_categorical(y_train_raw, params.NB_CLASSES)
            y_val   = keras.utils.to_categorical(y_val_raw,   params.NB_CLASSES)
            print("🔧 One-hot encoding labels (for categorical_crossentropy)")
        else:
            y_train = y_train_raw
            y_val   = y_val_raw
            print(f"🔧 Using sparse labels (for {params.LOSS_TYPE})")

        tuning_res = run_architecture_tuning(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            num_trials=args.num_trials,
            debug=args.debug,
        )
        if tuning_res:
            # Apply the best hyper parameters for the subsequent training run
            best = tuning_res
            print("\n🎯 Best hyper parameters discovered:")
            print(f"   Optimizer   : {best['optimizer']}")
            print(f"   Learning LR : {best['learning_rate']}")
            print(f"   Batch size  : {best['batch_size']}")
            print(f"   Val acc     : {best['val_accuracy']:.4f}")

            # Continue with a normal training run using the tuned values
            model, hist, out_dir = train_model(debug=args.debug, best_hps=best, 
                                             no_cleanup=args.no_cleanup, 
                                             full_analysis=not args.no_analysis)
            print(f"\n✅ Training finished – results in {out_dir}")
        else:
            print("❌ Tuning failed – exiting.")
        return

    # -----------------------------------------------------------------
    #  Test all architectures (quick sanity check)
    # -----------------------------------------------------------------
    if args.test_all_models:
        test_all_models(debug=args.debug)
        return

    # -----------------------------------------------------------------
    #  Train a specific list of architectures
    # -----------------------------------------------------------------
    if args.train:
        train_specific_models(args.train, debug=args.debug, 
                            no_cleanup=args.no_cleanup, 
                            full_analysis=not args.no_analysis)
        return

    # -----------------------------------------------------------------
    #  Train *every* architecture sequentially
    # -----------------------------------------------------------------
    if args.train_all:
        train_specific_models(params.AVAILABLE_MODELS, debug=args.debug,
                            no_cleanup=args.no_cleanup,
                            full_analysis=not args.no_analysis)
        return

    # -----------------------------------------------------------------
    #  Default – train the single architecture defined in parameters.py
    # -----------------------------------------------------------------
    print(f"🚀 Training architecture: {params.MODEL_ARCHITECTURE}")
    result = train_model(debug=args.debug, 
                                     no_cleanup=args.no_cleanup,
                                     full_analysis=not args.no_analysis)
    
    if result is not None and len(result) == 3 and result[0] is not None:
        model, hist, out_dir = result
        print(f"\n✅ Training completed - results stored in {out_dir}")
    else:
        print("\n❌ Training failed or returned early.")
        if result is None:
            print("   Reason: train_model returned None (check logs above for validation errors)")
        elif result[0] is None:
            print("   Reason: model object is None (check logs above for architecture/compilation errors)")


def train_model(debug: bool = False, best_hps=None, no_cleanup: bool = False, full_analysis: bool = True):
    """Main training function with comprehensive quantization handling"""
    try:
        setup_tensorflow_logging(debug)
        set_all_seeds(params.SHUFFLE_SEED)
        
        try:
            import mlflow
            MLFLOW_AVAILABLE = True
        except ImportError:
            MLFLOW_AVAILABLE = False
            print("ℹ️ MLflow not installed. Skipping experiment tracking. (Run `pip install mlflow` to enable)")
        
        # ==============================================================================
        # INITIALIZATION
        # ==============================================================================

        # Note: We NO LONGER validate or print on import to allow external 
        # configuration management (e.g., interactive_digit_config)
        print("🎯 VALIDATING QUANTIZATION PARAMETERS...")
        try:
            is_valid, corrected_params, message = params.validate_quantization_parameters()
            print(message)
        except Exception as e:
            print(f"❌ Quantization parameter validation failed: {e}")
            return None, None, None
        
        if not is_valid:
            print("🔄 Applying parameter corrections...")
            params.QUANTIZE_MODEL = corrected_params['QUANTIZE_MODEL']
            params.USE_QAT = corrected_params['USE_QAT']
            params.ESP_DL_QUANTIZE = corrected_params['ESP_DL_QUANTIZE']
            print("✅ Corrected parameters applied")
        
        # Validate quantization combination
        try:
            is_valid, msg = validate_quantization_combination()
            if not is_valid:
                print(f"❌ Quantization combination invalid: {msg}")
                return None, None, None
            print(f"✅ {msg}")
        except Exception as e:
            print(f"❌ Quantization combination validation crashed: {e}")
            return None, None, None
        
        # Setup hardware
        strategy = setup_gpu()
        
        # Create output directory
        color_mode = "GRAY" if params.USE_GRAYSCALE else "RGB"
        quantization_mode = ""
        if params.USE_QAT:
            quantization_mode = "_QAT"
        if params.ESP_DL_QUANTIZE:
            quantization_mode += "_ESP-DL"
        elif params.QUANTIZE_MODEL:
            quantization_mode += "_QUANT"
        
        # Use task name if provided
        task_prefix = f"{params.TASK_NAME}_" if getattr(params, 'TASK_NAME', None) else ""
        
        training_dir = os.path.join(
            params.OUTPUT_DIR, 
            f"{task_prefix}{params.MODEL_ARCHITECTURE}_{params.NB_CLASSES}cls{quantization_mode}_{color_mode}_{datetime.now().strftime('%m%d_%H%M')}"
        )
        os.makedirs(training_dir, exist_ok=True)
        print(f"📁 Output directory: {training_dir}")
        
        # Load and preprocess data
        print("\nLoading dataset from multiple sources...")
        (x_train_raw, y_train_raw), (x_val_raw, y_val_raw), (x_test_raw, y_test_raw) = get_data_splits()
        
        print("🔄 Preprocessing images...")
        x_train = preprocess_for_training(x_train_raw)
        x_val   = preprocess_for_training(x_val_raw)
        x_test  = preprocess_for_training(x_test_raw)
        
        # Handle labels based on model type
        if params.MODEL_ARCHITECTURE == "original_haverland":
            y_train_final = keras.utils.to_categorical(y_train_raw, params.NB_CLASSES)
            y_val_final = keras.utils.to_categorical(y_val_raw, params.NB_CLASSES) 
            y_test_final = keras.utils.to_categorical(y_test_raw, params.NB_CLASSES)
        else:
            y_train_final = y_train_raw.copy()
            y_val_final = y_val_raw.copy()
            y_test_final = y_test_raw.copy()
            
        # -----------------------------------------------------------------
        # MEMORY OPTIMIZATION: PUSH DATA TO GPU VRAM
        # -----------------------------------------------------------------
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("🚀 Memory Optimization: Attempting to push float32 data to GPU VRAM...")
            try:
                # Dynamically get the device name of the first GPU
                # This works even if the system maps GPUs differently than /GPU:0
                gpu_name = gpus[0].name.replace('physical_device:', '')
                with tf.device(gpu_name):
                    x_train = tf.constant(x_train)
                    x_val   = tf.constant(x_val)
                    x_test  = tf.constant(x_test)
                
                # Immediately aggressively clean up system memory
                gc.collect()
                print("   ✅ Datasets successfully moved to GPU VRAM. System DDR4 RAM freed.")
            except Exception as e:
                print(f"⚠️ Could not push to GPU VRAM (perhaps Out-Of-Memory or fragmentation): {e}")
                print("   Falling back to DDR4 system memory.")
        else:
            print("💡 No GPU detected for memory optimization. Using system DDR4 RAM.")
        
        # Create model with QAT if enabled
        use_qat = params.QUANTIZE_MODEL and params.USE_QAT and QAT_AVAILABLE

        if use_qat:
            print("Creating model with Quantization Aware Training...")
            model = create_qat_model()
        else:
            model = create_model()

        # Compile – loss depends on the architecture
        if params.MODEL_ARCHITECTURE == "original_haverland":
            loss_type = "categorical"
        else:
            loss_type = "sparse"

        model = compile_model(model, loss_type=loss_type) 
        
        # -----------------------------------------------------------------
        #  Resume Training Logic
        # -----------------------------------------------------------------
        if hasattr(params, 'RESUME_MODEL_PATH') and params.RESUME_MODEL_PATH:
            print(f"🔄 Resuming from checkpoint: {params.RESUME_MODEL_PATH}")
            try:
                model.load_weights(params.RESUME_MODEL_PATH)
                print("   ✅ Weights loaded successfully")
            except Exception as e:
                print(f"   ❌ Failed to load weights: {e}")
                print("   ⚠️  Starting training from scratch instead.")
        
        # Build model with explicit input shape (required for TF2.x eager mode)
        print("🔧 Building model with explicit input shape...")
        model.build(input_shape=(None,) + params.INPUT_SHAPE)
        
        # ✅ MOVE QAT VALIDATION HERE - AFTER MODEL IS CREATED
        print("🎯 VALIDATING QAT DATA CONSISTENCY...")
        if params.USE_QAT and params.QUANTIZE_MODEL:
            qat_consistent, qat_msg = validate_qat_data_consistency()
            if not qat_consistent:
                print(f"🚨 CRITICAL: {qat_msg}")
                print("   QAT training may produce incorrect results!")
            else:
                print(f"✅ {qat_msg}")
        
        # ✅ MOVE COMPREHENSIVE QAT VALIDATION HERE
        if params.USE_QAT and params.QUANTIZE_MODEL:
            print("🎯 RUNNING COMPREHENSIVE QAT VALIDATION...")
            from tqdm import tqdm
            with tqdm(total=1, desc="QAT Validation", leave=False) as pbar:
                qat_valid, qat_summary = validate_complete_qat_setup(model, debug=debug)
                pbar.update(1)
            
            if not qat_valid:
                print("🚨 QAT validation failed - consider reviewing quantization settings")
            else:
                print("✅ QAT validation passed - ready for quantization-aware training")
                
        if params.USE_QAT and params.QUANTIZE_MODEL and params.MODEL_ARCHITECTURE != "original_haverland":
            gradient_ok = check_qat_gradient_flow(model, x_train, y_train_final)
            output_ok = diagnose_qat_output_behavior(model, x_train, y_train_final)
            if not gradient_ok:
                print("🔄 QAT gradient issue - falling back to standard model")
                params.USE_QAT = False
                model = create_model()
            
        # Validate QAT data flow if using QAT
        if use_qat:
            try:
                qat_valid, qat_msg = validate_qat_data_flow(model, x_train, debug=debug)
                if not qat_valid:
                    print(f"❌ QAT data flow validation failed: {qat_msg}")
                    return None, None, None
            except Exception as e:
                print(f"❌ QAT data flow validation crashed: {e}")
                return None, None, None
                
        # Test if the model behaves like a quantized model
        print("\n🧪 QUICK QAT VERIFICATION TEST:")
        test_input = tf.convert_to_tensor(np.random.randint(0, 255, (1,) + params.INPUT_SHAPE, dtype=np.uint8))
        output = model(test_input)
        print(f"   Input dtype: {test_input.dtype}")
        print(f"   Output range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")
        print(f"   Output sum: {np.sum(output.numpy(), axis=1)}")
    
        # Continue with the rest of training...
        # Setup training components
        tflite_manager = TFLiteModelManager(training_dir, debug)
        tflite_manager.set_data(x_train=x_train_raw, x_test=x_test_raw, y_test=y_test_raw)

        # This will automatically test strategies and use the best one
        print("📦 Generating optimized TFLite model for initial verification...")
        with tqdm(total=1, desc="TFLite Conversion", leave=False) as pbar:
            tflite_blob, size = tflite_manager.save_as_tflite_enhanced(
                model, 
                params.get_tflite_filename(),
                quantize=params.QUANTIZE_MODEL
            )
            pbar.update(1)
        
        monitor = TrainingMonitor(training_dir, debug)
        monitor.set_model(model)
        
        # Print training summary
        print_training_summary(model, x_train, x_val, x_test, debug)
        save_model_summary_to_file(model, training_dir)
        
        # # Create representative dataset
        # if params.USE_QAT and params.QUANTIZE_MODEL:
            # # QAT models already carry quantisation information – no calibration needed.
            # representative_data = None
        # else:
            # representative_data = create_qat_representative_dataset(x_train_raw)
            
        # -----------------------------------------------------------------
        #  Representative dataset for PTQ
        # -----------------------------------------------------------------
        # If we are doing QAT we **do not** need a calibration set – the model
        # already contains fake quant nodes.  For pure PTQ we must supply
        # float32 data; the helper below now uses the *training* preprocessing
        # (float32) to avoid the previous uint8 → float conversion.
        # if params.USE_QAT and params.QUANTIZE_MODEL:
            # representative_data = None
        # else:
            # representative_data = create_qat_representative_dataset(x_train_raw)
            
        if params.USE_QAT and params.QUANTIZE_MODEL:
            # QAT models already carry quantization information – no calibration needed for TFLite conversion
            representative_data = None
            print("🎯 QAT model: No representative dataset needed (fake quant layers provide scales)")
        else:
            # PTQ needs calibration data
            print("🎯 PTQ: Creating representative dataset for calibration...")
            with tqdm(total=1, desc="Calibration Data", leave=False) as pbar:
                representative_data = create_qat_representative_dataset(x_train_raw)
                pbar.update(1)
            print("   ✅ Representative dataset ready")
        
        # Create callbacks
        callbacks = create_callbacks(
            training_dir, 
            tflite_manager, 
            representative_data, 
            params.EPOCHS, 
            monitor, 
            debug, 
            validation_data=(x_val, y_val_final),
            x_train_raw=x_train_raw
        )
        
        # Start training
        print("\n🎯 Starting training...")
        start_time = datetime.now()
        
        active_run = None
        if MLFLOW_AVAILABLE:
            try:
                import mlflow
                mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
                mlflow.set_experiment("Digit_Recognizer")
                active_run = mlflow.start_run(run_name=f"{params.MODEL_ARCHITECTURE}_{color_mode}")
                mlflow.tensorflow.autolog(log_models=False, log_datasets=False)
                mlflow.log_param("architecture", params.MODEL_ARCHITECTURE)
                mlflow.log_param("qat_enabled", params.USE_QAT)
                mlflow.log_param("epochs", params.EPOCHS)
                mlflow.log_param("batch_size", params.BATCH_SIZE)
            except Exception as e:
                print(f"⚠️ MLflow tracking initialization failed: {e}")
                MLFLOW_AVAILABLE = False

        if params.USE_DATA_AUGMENTATION:
            train_dataset, val_dataset, _ = setup_augmentation_for_training(
                x_train, y_train_final, x_val, y_val_final, debug=debug
            )
            print(f"\n🚀 Initiating model.fit (Phase: {'QAT' if params.USE_QAT else 'Standard'})")
            print(f"   Epochs: {params.INITIAL_EPOCH if hasattr(params, 'INITIAL_EPOCH') else 0} to {params.EPOCHS}")
            
            print("\n⏳ Loading dataset and optimizing computation graph...")
            print("   (This may take a minute for the first epoch)")
            from tqdm import tqdm
            with tqdm(total=1, desc="Graph Optimization", leave=False) as pbar:
                history = model.fit(
                    train_dataset,
                    epochs=params.EPOCHS,
                    initial_epoch=params.INITIAL_EPOCH if hasattr(params, 'INITIAL_EPOCH') else 0,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    verbose=0
                )
                pbar.update(1)
        else:
            # Compute class weights to handle imbalanced datasets
            try:
                from sklearn.utils.class_weight import compute_class_weight
                unique_classes = np.unique(y_train_final)
                weights = compute_class_weight(
                    class_weight='balanced',
                    classes=unique_classes,
                    y=y_train_final
                )
                class_weight_dict = dict(zip(unique_classes, weights))
                print(f"⚖️  Using class weights for {len(unique_classes)} classes (max ratio: {max(weights)/min(weights):.2f}x)")
            except Exception as e:
                print(f"⚠️  Could not compute class weights: {e}. Training without class weighting.")
                class_weight_dict = None

            print(f"\n🚀 Initiating model.fit (Phase: {'QAT' if params.USE_QAT else 'Standard'})")
            print(f"   Epochs: {params.INITIAL_EPOCH if hasattr(params, 'INITIAL_EPOCH') else 0} to {params.EPOCHS}")
            
            history = model.fit(
                x_train, y_train_final,
                batch_size=params.BATCH_SIZE,
                epochs=params.EPOCHS,
                initial_epoch=params.INITIAL_EPOCH if hasattr(params, 'INITIAL_EPOCH') else 0,
                validation_data=(x_val, y_val_final),
                callbacks=callbacks,
                verbose=0,
                shuffle=True,
                class_weight=class_weight_dict
            )
        
        training_time = datetime.now() - start_time
        
        # Final model evaluation and quantization analysis
        print("\n📈 Evaluating models...")
        
        # Evaluate Keras model
        train_accuracy = model.evaluate(x_train, y_train_final, verbose=0)[1]
        val_accuracy = model.evaluate(x_val, y_val_final, verbose=0)[1]
        test_accuracy = model.evaluate(x_test, y_test_final, verbose=0)[1]
        
        print(f"✅ Keras Model Evaluation:")
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Val Accuracy: {val_accuracy:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        
        # TFLite model evaluation and quantization analysis
        tflite_accuracy = 0.0
        # quantized_tflite_path = os.path.join(training_dir, params.TFLITE_FILENAME)
        quantized_tflite_path = os.path.join(training_dir, params.get_tflite_filename())
        
        quantization_results = {
            'tflite_accuracy': 0.0,
            'keras_accuracy': test_accuracy,
            'tflite_size': 0,
            'keras_size': 0,
            'accuracy_drop': 0.0,
            'size_reduction': 0.0
        }

        if os.path.exists(quantized_tflite_path) and params.QUANTIZE_MODEL:
            try:
                print("🔍 Running quantization analysis...")
                # Use the analysis function with correct parameter order
                analysis_result = analyze_quantization_impact(
                    model, x_test, y_test_final, quantized_tflite_path, debug=debug
                )
                
                if analysis_result is not None:
                    quantization_results.update(analysis_result)
                    tflite_accuracy = quantization_results.get('tflite_accuracy', 0.0)
                    
                    # Generate detailed report
                    analyzer = QuantizationAnalyzer(debug=debug)
                    analyzer.analysis_results = quantization_results
                    analyzer.generate_quantization_report(training_dir)
                else:
                    print("⚠️  Quantization analysis returned no results, using fallback values")
                    # Fallback: basic size measurements with better error handling
                    try:
                        quantization_results['tflite_size'] = get_tflite_model_size(quantized_tflite_path)
                        quantization_results['keras_size'] = get_keras_model_size(model)
                        if quantization_results['keras_size'] > 0:
                            quantization_results['size_reduction'] = (
                                (quantization_results['keras_size'] - quantization_results['tflite_size']) / 
                                quantization_results['keras_size'] * 100
                            )
                    except Exception as size_error:
                        print(f"⚠️  Fallback size measurement failed: {size_error}")
                        
            except Exception as e:
                print(f"❌ Quantization analysis failed: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                # Fallback measurements with error handling
                try:
                    quantization_results['tflite_size'] = get_tflite_model_size(quantized_tflite_path)
                except:
                    quantization_results['tflite_size'] = 0
                    
                try:
                    quantization_results['keras_size'] = get_keras_model_size(model)
                except:
                    quantization_results['keras_size'] = 0
        
        # Run comprehensive analysis if requested
        if full_analysis:
            run_comprehensive_analysis(model, history, training_dir, x_test, y_test_final, debug)
        else:
            print("⏭️  Skipping comprehensive analysis (--no_analysis flag used)")
        
        # Save training plots and configuration
        monitor.save_training_plots()
        save_training_config(training_dir, 
                            quantization_results['tflite_size'],
                            quantization_results['keras_size'],
                            tflite_manager,
                            test_accuracy, tflite_accuracy, training_time, debug, model=model)
        
        # Final summary
        print("\n" + "="*60)
        print("🏁 TRAINING COMPLETED")
        print("="*60)
        print(f"⏱️  Training time: {training_time}")
        print(f"📊 Final Results:")
        print(f"   Best Validation Accuracy: {tflite_manager.best_accuracy:.4f}")
        print(f"   Test Accuracy - Keras: {test_accuracy:.4f}")
        print(f"   Test Accuracy - TFLite: {tflite_accuracy:.4f}")
        
        if params.QUANTIZE_MODEL:
            print(f"   Quantized Model Size: {quantization_results['tflite_size']:.1f} KB")
            print(f"   Float Model Size: {quantization_results['keras_size']:.1f} KB")
            if quantization_results['keras_size'] > 0:
                size_reduction = (
                    (quantization_results['keras_size'] - quantization_results['tflite_size']) / 
                    quantization_results['keras_size'] * 100
                )
                print(f"   Size Reduction: {size_reduction:.1f}%")
        
        # -----------------------------------------------------------------
        #  CLEANUP: Delete checkpoints if not in debug mode and cleanup not disabled
        # -----------------------------------------------------------------
        if not debug and not no_cleanup:
            print("\n🧹 Cleaning up training checkpoints...")
            try:
                files_deleted, space_freed = cleanup_training_directory(training_dir, debug=False)
                if files_deleted > 0:
                    print(f"✅ Cleanup completed: {files_deleted} files deleted, {space_freed/1024/1024:.1f} MB freed")
                else:
                    print("💡 No checkpoints found to clean up")
            except Exception as e:
                print(f"⚠️  Cleanup failed: {e}")
        else:
            if debug:
                print("🔍 Debug mode - skipping cleanup")
            if no_cleanup:
                print("🚫 Cleanup disabled - keeping checkpoints")
            
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("tflite_accuracy", tflite_accuracy)
                
                # Log the optimized TFLite model as an artifact
                for root, dirs, files in os.walk(training_dir):
                    for file in files:
                        if file.endswith(".tflite"):
                            mlflow.log_artifact(os.path.join(root, file))
                
                if active_run:
                    mlflow.end_run()

        return model, history, training_dir

    except Exception as e:
        print(f"\n💥 CRITICAL TRAINING ERROR: {e}")
        import traceback
        traceback.print_exc()
        # End any active MLflow run if it exists
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
        except:
            pass
        return None, None, None

if __name__ == "__main__":
    main()
    clear_cache()
    
# py train.py --train_all