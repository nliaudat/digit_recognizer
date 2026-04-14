#!/usr/bin/env python3
"""
Retrain existing edge models (v4, v16, etc.) using a teacher model.
This does not modify models/__init__.py - imports directly from model files.
"""

import argparse
import datetime
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

# Add project root to path before other imports
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import parameters as params
from utils.distiller import (
    DistillationProgressCallback, Distiller, MixedInputDistiller,
    ProgressiveDistiller
)
from utils.losses import (
    DynamicFocalLoss, DynamicSparseFocalLoss, focal_loss, sparse_focal_loss
)
from utils.model_distiller_utils import (
    evaluate_distilled_model, export_student_for_edge, freeze_teacher_model,
    get_model_size_kb
)
from utils.train_analyse import (
    analyze_confusion_matrix, analyze_training_history, verify_tflite_full_qat
)
from utils.train_distill_helper import load_distillation_data
from utils.train_helpers import save_model_summary_to_file
from utils.train_trainingmonitor import TrainingMonitor

# Optional/Delayed imports to avoid circularities
try:
    from models.model_factory import create_model_by_name
except ImportError:
    create_model_by_name = None

try:
    from utils.ensemble_teacher import EnsembleTeacher
except ImportError:
    EnsembleTeacher = None

try:
    from utils.train_qat_helper import create_qat_model
except ImportError:
    create_qat_model = None

# Registry for existing edge models (now dynamic metadata)
EXISTING_EDGE_MODELS = {
    m.replace("digit_recognizer_", "").replace("_teacher", ""): {
        "available": True,
        "full_name": m
    } for m in params.AVAILABLE_MODELS
}

# Add full names too
for m in params.AVAILABLE_MODELS:
    EXISTING_EDGE_MODELS[m] = {"available": True, "full_name": m}

TEACHERS = EXISTING_EDGE_MODELS # Any model can be a teacher

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain existing edge models with teacher supervision"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=params.get_available_model_names(),
        help="Existing edge model to retrain."
    )
    parser.add_argument(
        "--teacher", "--teachers",
        dest="teachers",
        type=str,
        nargs="+",
        default=["v16"],
        help="Teacher model(s) for distillation"
    )
    parser.add_argument(
        "--teacher-color",
        type=str,
        default=None,
        choices=["gray", "rgb"],
        help="Teacher color mode (defaults to student color)"
    )
    
    # Dataset
    parser.add_argument(
        "--classes",
        type=int,
        default=10,
        choices=[10, 100],
        help="Number of classes"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="gray",
        choices=["gray", "rgb"],
        help="Input color mode"
    )
    
    # Distillation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Balance between hard and soft labels"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="soft",
        choices=["soft", "hard", "hybrid"],
        help="Distillation mode"
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Use progressive distillation"
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Retraining epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (lower for fine-tuning)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to existing model weights (optional)"
    )
    
    # Teacher
    parser.add_argument(
        "--teacher-checkpoint", "--teacher-checkpoints",
        dest="teacher_checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to teacher weights (optional)"
    )
    parser.add_argument(
        "--teacher-weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each teacher in the ensemble"
    )
    parser.add_argument(
        "--teacher-pretrained",
        action="store_true",
        default=True,
        help="Use pretrained teacher"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_models/",
        help="Output directory"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Export quantized TFLite"
    )
    
    return parser.parse_args()


def find_best_checkpoint(
    model_name: str, 
    num_classes: int, 
    color_mode: str
) -> Optional[str]:
    """
    Find best checkpoint with improved matching.
    """
    color_suffix = color_mode.upper()
    
    search_dirs = [
        Path(f"exported_models/{num_classes}cls_{color_suffix}"),
        Path("checkpoints")
    ]
    
    # Normalize model name
    model_name = model_name.strip("\\/")
    short_name = model_name.replace("digit_recognizer_", "").replace("_teacher", "").replace("_student", "")
    
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
            
        model_lower = model_name.lower()
        short_lower = short_name.lower()

        # 1. First check for EXACT directory match (this allows passing a full folder name as model_name)
        exact_dir = base_dir / model_name
        if exact_dir.is_dir():
            candidate_paths = [
                exact_dir / "best_model.keras",
                exact_dir / "model" / "best_model.keras",
                exact_dir / "model.keras",
                exact_dir / "model" / "model.keras"
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    logger.info(f"Found exact folder match: {candidate}")
                    return str(candidate)

        # 2. Search subdirectories
        for sub_dir in base_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            
            dir_name_lower = sub_dir.name.lower()
            
            # Strict matching for model name to avoid v3 matching v31
            pattern_a = f"digit_recognizer_{short_lower}_"
            pattern_b = f"{short_lower}_"
            
            if (dir_name_lower.startswith(pattern_a) or 
                dir_name_lower.startswith(pattern_b) or 
                dir_name_lower == model_lower or
                dir_name_lower == f"digit_recognizer_{short_lower}"):
                # Check root of the run folder and the 'model' subfolder 
                candidate_paths = [
                    sub_dir / "best_model.keras",
                    sub_dir / "model" / "best_model.keras",
                    sub_dir / "model.keras",
                    sub_dir / "model" / "model.keras"
                ]
                for candidate in candidate_paths:
                    if candidate.exists():
                        logger.info(f"Found fuzzy folder match: {candidate}")
                        return str(candidate)
        
        # 3. Search direct files (.keras files directly in search_dirs)
        for f in base_dir.glob("*.keras"):
            fname_lower = f.name.lower()
            if (fname_lower.startswith(f"{model_lower}_") or 
                fname_lower.startswith(f"{short_lower}_") or 
                fname_lower == f"{model_lower}.keras" or
                fname_lower == f"{short_lower}.keras"):
                return str(f)
    
    return None


def load_or_create_model(
    model_name: str,
    num_classes: int,
    color_mode: str,
    load_path: Optional[str] = None
) -> tf.keras.Model:
    """
    Load existing model or create new one.
    """
    
    channels = 1 if color_mode == "gray" else 3
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
    
    # Create model dynamically
    model = create_model_by_name(model_name, num_classes=num_classes, input_shape=input_shape)
    
    # Wrap for QAT if active in parameters
    if getattr(params, 'USE_QAT', False) and getattr(params, 'QUANTIZE_MODEL', False):
        logger.info(f"🎯 Wrapping student {model_name} for QAT...")
        model = create_qat_model(model)

    # Load weights if provided
    if load_path and os.path.exists(load_path):
        logger.info(f"Loading weights from {load_path}")
        model.load_weights(load_path)
    
    logger.info(f"Model: {model_name}, params: {model.count_params():,}")
    return model


def retrain_with_teacher(
    model: tf.keras.Model,
    teacher: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace
) -> Tuple[Distiller, str]:
    """
    Retrain model using teacher distillation.
    """
    # ── Verify Teacher(s) ────────────────────────────────────────────────
    teacher_name = "Ensemble" if isinstance(teacher, tf.keras.Model) and teacher.name == "ensemble_teacher" else args.teachers[0]
    
    logger.info("=" * 60)
    logger.info(f"Retraining {args.model} with teacher(s) {args.teachers}")
    logger.info(f"Mode: {args.mode}, T={args.temperature}, α={args.alpha}")
    logger.info("=" * 60)
    
    # Freeze teacher
    teacher = freeze_teacher_model(teacher)
    
    # Check for channel mismatch (e.g. Grayscale student, RGB teacher)
    # Most teachers expect 3 channels if they are standard pre-trained models
    teacher_channels = teacher.input_shape[-1]
    student_channels = model.input_shape[-1]
    
    teacher_input_fn = None
    if student_channels == 1 and teacher_channels == 3:
        logger.info("Detected mismatched channels: Grayscale student -> RGB teacher. Using MixedInputDistiller.")
        teacher_input_fn = lambda x: tf.image.grayscale_to_rgb(x)
    
    # Create distiller
    if args.progressive:
        if teacher_input_fn:
            logger.warning("ProgressiveDistiller does not natively support MixedInputDistiller logic yet. Using standard MixedInputDistiller.")
            distiller = MixedInputDistiller(
                student=model,
                teacher=teacher,
                teacher_input_fn=teacher_input_fn,
                temperature=args.temperature,
                alpha=args.alpha,
                mode=args.mode,
            )
        else:
            distiller = ProgressiveDistiller(
                student=model,
                teacher=teacher,
                initial_temperature=8.0,
                final_temperature=2.0,
                initial_alpha=0.3,
                final_alpha=0.8,
                total_epochs=args.epochs,
                mode=args.mode,
            )
    else:
        if teacher_input_fn:
            distiller = MixedInputDistiller(
                student=model,
                teacher=teacher,
                teacher_input_fn=teacher_input_fn,
                temperature=args.temperature,
                alpha=args.alpha,
                mode=args.mode,
            )
        else:
            distiller = Distiller(
                student=model,
                teacher=teacher,
                temperature=args.temperature,
                alpha=args.alpha,
                mode=args.mode,
            )
    
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        metrics=["accuracy"],
    )
    
    # Build to initialize weights and avoid warnings
    channels = model.input_shape[-1]
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
    distiller.build((None,) + input_shape)
    
    # Use the provided output directory or default
    checkpoint_dir = getattr(args, "output_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Standardize checkpoint name
    ckpt_path = os.path.join(
        checkpoint_dir,
        f"best_model.keras"
    )
    
    callbacks = [
        DistillationProgressCallback(),  # Sync current_epoch for schedules
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]
    
    # Add CSV text logger and Monitor for plots (parity with train.py)
    log_csv_path = os.path.join(checkpoint_dir, "training_log.csv")
    callbacks.append(tf.keras.callbacks.CSVLogger(log_csv_path))
    
    monitor = TrainingMonitor(checkpoint_dir)
    
    class TrainingMonitorCallbackWrapper(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            monitor.model = self.model
            monitor.on_epoch_end(epoch, logs)
            
    callbacks.append(TrainingMonitorCallbackWrapper())
    
    # Train
    history = distiller.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    
    monitor.save_training_plots()
    
    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Checkpoint saved: {ckpt_path}")
    
    return distiller, ckpt_path


def main():
    args = parse_args()
    
    # ── Param sync ──────────────────────────────────────────────────────────
    # Ensure parameters.py reflects CLI args before any models are created
    params.NB_CLASSES = args.classes
    params.INPUT_CHANNELS = 1 if args.color == "gray" else 3
    params.update_derived_parameters()
    
    # Validate model exists and is available
    if args.model not in EXISTING_EDGE_MODELS:
        logger.error(f"Model '{args.model}' not found.")
        logger.info(f"Available models: {list(EXISTING_EDGE_MODELS.keys())}")
        sys.exit(1)
    
    model_info = EXISTING_EDGE_MODELS[args.model]
    if not model_info["available"]:
        logger.error(f"Model '{args.model}' cannot be imported.")
        logger.info("Check that the model file exists in models/ directory.")
        sys.exit(1)
    
    # Determine input shape
    channels = 1 if args.color == "gray" else 3
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
    
    logger.info("=" * 60)
    logger.info("🚀 Retraining Existing Model with Teacher")
    logger.info("=" * 60)
    logger.info(f"  Model:      {args.model}")
    logger.info(f"  Teacher(s): {args.teachers}")
    logger.info(f"  Classes:    {args.classes}")
    logger.info(f"  Color:      {args.color.upper()}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Alpha:      {args.alpha}")
    logger.info(f"  Mode:       {args.mode}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  LR:         {args.lr}")
    logger.info("=" * 60)
    
    # Load dataset
    x_train, y_train, x_val, y_val, x_test, y_test = load_distillation_data(
        num_classes=args.classes,
        color_mode=args.color,
    )

    # ── Output directory logic ──────────────────────────────────────────
    color_label = args.color.upper()
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # User-requested folder structure: exported_models/10cls_RGB/retrained_v4_10cls_RGB_0328_2152
    # params.OUTPUT_DIR already includes 'exported_models/10cls_RGB'
    folder_name = f"retrained_{args.model}_{args.classes}cls_{color_label}_{timestamp}"
    export_dir = os.path.join(params.OUTPUT_DIR, folder_name)
    os.makedirs(export_dir, exist_ok=True)
    
    # Redirect intermediate checkpoints to this folder
    args.output_dir = export_dir
    
    # ── Load or create teacher(s) ─────────────────────────────────────────
    teacher_color = args.teacher_color if args.teacher_color else args.color
    teacher_channels = 1 if teacher_color == "gray" else 3
    teacher_input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, teacher_channels)
    
    loaded_teachers = []
    
    for i, t_name in enumerate(args.teachers):
        actual_teacher_checkpoint = None
        if args.teacher_checkpoints and i < len(args.teacher_checkpoints):
            actual_teacher_checkpoint = args.teacher_checkpoints[i]
        
        # Auto-find teacher if not provided
        if not actual_teacher_checkpoint:
            found = find_best_checkpoint(t_name, args.classes, teacher_color)
            if found:
                logger.info(f"Found existing teacher checkpoint for {t_name}: {found}")
                actual_teacher_checkpoint = found

        if actual_teacher_checkpoint and os.path.exists(actual_teacher_checkpoint):
            logger.info(f"Loading teacher {t_name} from {actual_teacher_checkpoint}")
            # Pass custom objects for focal loss if needed
            custom_objects = {
                "DynamicSparseFocalLoss": DynamicSparseFocalLoss,
                "DynamicFocalLoss": DynamicFocalLoss,
                "sparse_focal_loss": sparse_focal_loss,
                "focal_loss": focal_loss
            }
            # Auto-inject any custom layers from the teacher's script
            clean_name = t_name.replace('digit_recognizer_', '').replace('_teacher', '')
            for prefix in ["models.", "models.digit_recognizer_", "models._tested_but_rejected.", "models._tested_but_rejected.digit_recognizer_"]:
                try:
                    mod = importlib.import_module(f"{prefix}{clean_name}")
                    for name, obj in inspect.getmembers(mod, inspect.isclass):
                        if issubclass(obj, tf.keras.layers.Layer) and obj is not tf.keras.layers.Layer:
                            custom_objects[name] = obj
                    break # Success
                except (ModuleNotFoundError, Exception):
                    continue
                
            t_model = tf.keras.models.load_model(actual_teacher_checkpoint, custom_objects=custom_objects, safe_mode=False)
        else:
            logger.warning(f"No trained teacher found for {t_name}. Using random-weight teacher (not recommended!)")
            t_model = create_model_by_name(
                t_name,
                num_classes=args.classes,
                input_shape=teacher_input_shape,
                pretrained=args.teacher_pretrained,
            )
        
        # Freeze teacher
        t_model = freeze_teacher_model(t_model)
        loaded_teachers.append(t_model)
    
    if len(loaded_teachers) > 1:
        teacher = EnsembleTeacher(loaded_teachers, teacher_weights=args.teacher_weights)
        teacher_name_str = "+".join(args.teachers)
    else:
        teacher = loaded_teachers[0]
        teacher_name_str = args.teachers[0]
    
    # Auto-find student baseline if not provided
    actual_student_checkpoint = args.load_checkpoint
    if not actual_student_checkpoint:
        found = find_best_checkpoint(args.model, args.classes, args.color)
        if found:
            logger.info(f"Found existing student baseline: {found}")
            actual_student_checkpoint = found

    # Load or create model to retrain
    model = load_or_create_model(
        args.model,
        args.classes,
        args.color,
        actual_student_checkpoint
    )
    
    # Evaluate baseline accuracy
    logger.info("\n📊 Baseline evaluation (before retraining):")
    baseline_metrics = evaluate_distilled_model(model, (x_test, y_test))
    logger.info(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # Retrain with teacher
    distiller, checkpoint_path = retrain_with_teacher(
        model, teacher,
        x_train, y_train,
        x_val, y_val,
        args
    )
    
    # Get retrained model
    retrained_model = distiller.get_student()
    
    # Evaluate after retraining
    logger.info("\n📊 After retraining evaluation:")
    after_metrics = evaluate_distilled_model(retrained_model, (x_test, y_test))
    logger.info(f"  Accuracy: {after_metrics['accuracy']:.4f}")
    improvement = (after_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    logger.info(f"  Improvement: {improvement:+.2f}%")
    
    # Export
    if args.quantize:
        # Path where Keras/TFLite models will be saved (flat like train.py)
        export_path = os.path.join(export_dir, args.model)
        
        # Increase calibration data for PTQ (ignored if QAT was already active)
        n_calib = min(1000, len(x_test))
        tflite_path = export_student_for_edge(
            retrained_model,
            export_path,
            quantize=True,
            representative_dataset=x_test[:n_calib],
            target_hardware="esp32"
        )
        logger.info(f"Exported to: {tflite_path}")
    else:
        tflite_path = None
    
    # Save results
    results = {
        "model": args.model,
        "teacher": teacher_name_str,
        "classes": args.classes,
        "color_mode": args.color,
        "distillation": {
            "mode": args.mode,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "progressive": args.progressive,
        },
        "baseline_accuracy": baseline_metrics["accuracy"],
        "retrained_accuracy": after_metrics["accuracy"],
        "improvement": after_metrics["accuracy"] - baseline_metrics["accuracy"],
        "checkpoint": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
    }
    
    results_path = os.path.join(export_dir, f"retrain_{args.model}_{args.classes}cls.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save alias for parity with train.py
    with open(os.path.join(export_dir, "training_config.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # ── 7. Generate Training Resume (parity with train.py) ───────────────
    
    # Save standard best_model.keras in the root
    retrained_model.save(os.path.join(export_dir, "best_model.keras"))
    
    # Save model summary
    save_model_summary_to_file(retrained_model, export_dir)
    
    # Generate Confusion Matrix
    try:
        analysis_dir = os.path.join(export_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        analyze_confusion_matrix(retrained_model, x_test, y_test, save_path=analysis_dir)
    except Exception as e:
        logger.warning(f"⚠️ Could not generate confusion matrix: {e}")

    # Generate Detailed Training plot from CSV if possible
    try:
        csv_log = os.path.join(export_dir, "training_log.csv")
        if os.path.exists(csv_log):
            analyze_training_history(csv_log, save_path=analysis_dir)
    except Exception as e:
        logger.warning(f"⚠️ Could not generate training history plot: {e}")

    # ── 8. Full QAT Verification ──
    try:
        if args.quantize and tflite_path and os.path.exists(tflite_path):
            qat_report = verify_tflite_full_qat(tflite_path, debug=False)
            if qat_report:
                results['qat_verification'] = qat_report
                if qat_report['is_full_qat']:
                    logger.info("✅ TFLite QAT Verification: Model is FULL QAT (integer only)")
                else:
                    logger.warning("⚠️ TFLite QAT Verification: Model may NOT be full QAT (float ops detected)")
                    logger.warning(f"   I/O: {qat_report['input_dtype']} / {qat_report['output_dtype']}")
                    logger.warning(f"   Quantized Tensors: {qat_report['quantization_ratio']:.1%}")
            else:
                logger.warning("⚠️ Could not perform QAT verification")
    except Exception as e:
        logger.warning(f"⚠️ QAT status verification failed: {e}")

    logger.info(f"\n✅ Results saved to {results_path}")
    
    return retrained_model, results


if __name__ == "__main__":
    main()