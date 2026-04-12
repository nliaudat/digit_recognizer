#!/usr/bin/env python3
"""
utils/train_distillation.py
─────────────────────────────────────────────────────────────────────────────
Knowledge-distillation training helpers integrated with the project data
pipeline.  Called from the top-level ``train_distill.py`` entry-point.

Workflow
────────
1. Load data via the project's existing ``get_data_splits`` / preprocess stack.
2. Optionally train (or load) the teacher using the normal Keras compile/fit
   flow — same loss / callbacks used in ``train.py``.
3. Build the student (v30 or v31, any size variant, gray or RGB).
4. Wrap both in ``Distiller`` and train the student via knowledge distillation.
5. Export the trained student with TFLite quantization.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ── Project root setup ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import parameters as params
from models.model_factory import create_model_by_name
from utils import (
    get_data_splits, preprocess_for_inference, preprocess_for_training
)
from utils.distiller import (
    DistillationProgressCallback, Distiller, MixedInputDistiller,
    ProgressiveDistiller
)
from utils.ensemble_teacher import EnsembleTeacher
from utils.export_onnx import export_keras_to_onnx
from utils.losses import (
    DynamicFocalLoss, DynamicSparseFocalLoss, focal_loss, sparse_focal_loss
)
from utils.model_distiller_utils import (
    compare_teacher_student, evaluate_distilled_model, export_student_for_edge,
    freeze_teacher_model, get_model_size_kb, save_distillation_results
)
from utils.retrain_with_teacher import find_best_checkpoint
from utils.train_analyse import (
    analyze_confusion_matrix, analyze_training_history, verify_tflite_full_qat
)
from utils.train_helpers import save_model_summary_to_file
from utils.train_qat_helper import create_qat_model
from utils.train_trainingmonitor import TrainingMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry maps
# ---------------------------------------------------------------------------

TEACHERS: Dict[str, Any] = {}

STUDENTS: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Data loading — delegates to the project's existing pipeline
# ---------------------------------------------------------------------------

def load_distillation_data(
    num_classes: int,
    color_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the dataset using the project's multi-source loader.

    Temporarily overrides the global ``params.NB_CLASSES`` and
    ``params.INPUT_CHANNELS`` so the project pipeline loads the correct
    label files and resizes to the right channel count.

    Returns:
        (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    # ── Save current globals and apply overrides ──────────────────────────
    _prev_classes  = params.NB_CLASSES
    _prev_channels = params.INPUT_CHANNELS

    params.NB_CLASSES   = num_classes
    params.INPUT_CHANNELS = 1 if color_mode == "gray" else 3
    params.update_derived_parameters()

    try:
        logger.info(
            f"Loading dataset: {num_classes} classes, "
            f"color_mode={color_mode} → INPUT_CHANNELS={params.INPUT_CHANNELS}"
        )
        (x_train_raw, y_train_raw), (x_val_raw, y_val_raw), (x_test_raw, y_test_raw) = get_data_splits()

        x_train = preprocess_for_training(x_train_raw)
        x_val   = preprocess_for_training(x_val_raw)
        x_test  = preprocess_for_training(x_test_raw)  # USE TRAINING PREPROCESS FOR EVAL OF KERAS MODEL

        logger.info(
            f"Dataset loaded — train: {len(x_train)}, "
            f"val: {len(x_val)}, test: {len(x_test)}"
        )
        logger.info(f"Input shape: {x_train.shape[1:]}")

        return x_train, y_train_raw, x_val, y_val_raw, x_test, y_test_raw

    finally:
        # ── Always restore globals ────────────────────────────────────────
        params.NB_CLASSES   = _prev_classes
        params.INPUT_CHANNELS = _prev_channels
        params.update_derived_parameters()


# ---------------------------------------------------------------------------
# Teacher training
# ---------------------------------------------------------------------------

def train_teacher(
    teacher_type: str,
    num_classes: int,
    color_mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> tf.keras.Model:
    """
    Train a teacher model from scratch (or with pretrained ImageNet backbone).

    The teacher uses the *same* input shape as the student (32×20, gray or RGB)
    so that both can be evaluated on the same validation data.

    Returns:
        Trained teacher model.
    """
    tf.keras.backend.clear_session()
    channels = 1 if color_mode == "gray" else 3
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)

    logger.info("=" * 60)
    logger.info(f"Training Teacher: {teacher_type} | {num_classes} classes | {color_mode.upper()}")
    logger.info("=" * 60)

    if teacher_type in TEACHERS:
        builder = TEACHERS[teacher_type]
        teacher = builder(
            num_classes=num_classes,
            input_shape=input_shape,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        teacher = builder(
            num_classes=num_classes,
            input_shape=input_shape,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        teacher = create_model_by_name(
            teacher_type,
            num_classes=num_classes,
            input_shape=input_shape,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    teacher.summary(print_fn=logger.info)

    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # Softmax if params.USE_LOGITS is False, otherwise logits
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=params.USE_LOGITS),
        metrics=["accuracy"],
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Standardize checkpoint path to follow project pattern: exported_models/xxx/model/
    ckpt_path = os.path.join(
        checkpoint_dir,
        f"best_model.keras"
    )
    
    # ── Parity with train.py: Logging and Monitoring ─────────────────────
    
    # Training logs and monitor
    log_csv_path = os.path.join(checkpoint_dir, "training_log.csv")
    monitor = TrainingMonitor(checkpoint_dir)

    class TrainingMonitorCallbackWrapper(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            monitor.model = self.model
            monitor.on_epoch_end(epoch, logs)

    callbacks = [
        tf.keras.callbacks.CSVLogger(log_csv_path),
        TrainingMonitorCallbackWrapper(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    history = teacher.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    
    monitor.save_training_plots()
    
    # Save model summary
    save_model_summary_to_file(teacher, checkpoint_dir)
    
    # Generate Analysis folder
    try:
        analysis_dir = os.path.join(checkpoint_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        # We use whole x_val for teacher since it's small usually
        analyze_confusion_matrix(teacher, x_val, y_val, save_path=analysis_dir)
        if os.path.exists(log_csv_path):
            analyze_training_history(log_csv_path, save_path=analysis_dir)
    except Exception as e:
        logger.warning(f"⚠️ Teacher analysis failed: {e}")

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    logger.info(f"Teacher best val_accuracy: {best_val_acc:.4f}")
    logger.info(f"Teacher checkpoint saved → {ckpt_path}")
    return teacher


# ---------------------------------------------------------------------------
# Student distillation training
# ---------------------------------------------------------------------------

def train_student_distillation(
    teacher: tf.keras.Model,
    student_variant: str,
    num_classes: int,
    color_mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 4.0,
    alpha: float = 0.7,
    mode: str = "soft",
    use_progressive: bool = False,
    checkpoint_dir: str = "checkpoints",
) -> Tuple[Distiller, Dict]:
    """
    Train a student model via knowledge distillation from a frozen teacher.

    Args:
        teacher:          Pre-trained frozen teacher model.
        student_variant:  Key from STUDENTS dict, e.g. 'v30_medium'.
        num_classes:      10 or 100.
        color_mode:       'gray' or 'rgb'.
        x_train/y_train:  Training data (preprocessed).
        x_val/y_val:      Validation data (preprocessed).
        epochs:           Max training epochs.
        batch_size:       Batch size.
        learning_rate:    Initial LR for Adam.
        temperature:      KL softening temperature.
        alpha:            Hard-label weight (0 = all teacher, 1 = all hard).
        mode:             'soft' | 'hard' | 'hybrid'.
        use_progressive:  Use ProgressiveDistiller (dynamic T and alpha).
        checkpoint_dir:   Directory to save student checkpoints.

    Returns:
        (distiller, history_dict)
    """
    channels   = 1 if color_mode == "gray" else 3
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)

    logger.info("=" * 60)
    logger.info(f"Distilling → Student: {student_variant}")
    logger.info(f"Mode: {mode} | T={temperature} | alpha={alpha} | progressive={use_progressive}")
    logger.info("=" * 60)

    # ── Build student ──────────────────────────────────────────────────────
    if student_variant in STUDENTS:
        builder = STUDENTS[student_variant]
        student = builder(num_classes=num_classes, input_shape=input_shape)
    else:
        student = create_model_by_name(student_variant, num_classes=num_classes, input_shape=input_shape)
    
    # Wrap student for QAT if enabled in parameters
    if getattr(params, 'USE_QAT', False) and getattr(params, 'QUANTIZE_MODEL', False):
        logger.info(f"🎯 Wrapping student {student_variant} for QAT...")
        student = create_qat_model(student)
    logger.info(f"Student parameters: {student.count_params():,}")
    logger.info(f"Estimated INT8 size: {get_model_size_kb(student):.1f} KB")

    # ── Build distiller ────────────────────────────────────────────────────
    
    # Check for channel mismatch (e.g. Grayscale student, RGB teacher)
    teacher_channels = teacher.input_shape[-1]
    student_channels = student.input_shape[-1]
    
    teacher_input_fn = None
    if student_channels == 1 and teacher_channels == 3:
        logger.info("Detected mismatched channels: Grayscale student -> RGB teacher. Using MixedInputDistiller.")
        teacher_input_fn = lambda x: tf.image.grayscale_to_rgb(x)
    
    if use_progressive:
        if teacher_input_fn:
            logger.warning("ProgressiveDistiller does not natively support MixedInputDistiller logic yet. Using standard MixedInputDistiller.")
            distiller = MixedInputDistiller(
                student=student,
                teacher=teacher,
                teacher_input_fn=teacher_input_fn,
                temperature=temperature,
                alpha=alpha,
                mode=mode,
            )
        else:
            distiller = ProgressiveDistiller(
                student=student,
                teacher=teacher,
                initial_temperature=8.0,
                final_temperature=2.0,
                initial_alpha=0.3,
                final_alpha=0.8,
                total_epochs=epochs,
                mode=mode,
            )
    else:
        if teacher_input_fn:
            distiller = MixedInputDistiller(
                student=student,
                teacher=teacher,
                teacher_input_fn=teacher_input_fn,
                temperature=temperature,
                alpha=alpha,
                mode=mode,
            )
        else:
            distiller = Distiller(
                student=student,
                teacher=teacher,
                temperature=temperature,
                alpha=alpha,
                mode=mode,
            )

    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    
    # Build to initialize weights and avoid warnings
    distiller.build((None,) + input_shape)

    # ── Callbacks ──────────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir,
        f"best_model.keras"
    )
    callbacks = [
        DistillationProgressCallback(),  # Sync current_epoch for schedules
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    # Add CSV text logger and Monitor for plots
    log_csv_path = os.path.join(checkpoint_dir, "training_log.csv")
    callbacks.append(tf.keras.callbacks.CSVLogger(log_csv_path))
    
    monitor = TrainingMonitor(checkpoint_dir)
    
    class TrainingMonitorCallbackWrapper(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            monitor.model = self.model
            monitor.on_epoch_end(epoch, logs)
            
    callbacks.append(TrainingMonitorCallbackWrapper())

    # ── Train ──────────────────────────────────────────────────────────────
    history = distiller.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    monitor.save_training_plots()

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    logger.info(f"Student best val_accuracy: {best_val_acc:.4f}")
    logger.info(f"Student checkpoint saved → {ckpt_path}")

    return distiller, history.history


# ---------------------------------------------------------------------------
# Full distillation pipeline
# ---------------------------------------------------------------------------

def run_distillation_pipeline(
    teacher_types: List[str] = ["v16"],
    student_variant: str = "v30_medium",
    num_classes: int = 10,
    color_mode: str = "gray",
    teacher_color: Optional[str] = None,
    # Teacher training
    teacher_epochs: int = 50,
    teacher_lr: float = 1e-3,
    teacher_pretrained: bool = True,
    teacher_freeze_backbone: bool = False,
    teacher_checkpoints: Optional[List[str]] = None,
    teacher_weights: Optional[List[float]] = None,
    # Student distillation
    student_epochs: int = 50,
    student_lr: float = 1e-3,
    temperature: float = 4.0,
    alpha: float = 0.7,
    mode: str = "soft",
    use_progressive: bool = False,
    # Infrastructure
    batch_size: int = 32,
    checkpoint_dir: str = "checkpoints",
    output_dir: Optional[str] = None,
    export_quantized: bool = True,
    use_tqt: bool = False,
    target_hardware: str = "esp32",
) -> Dict[str, Any]:
    """
    End-to-end distillation pipeline.

    1. Load data via the project's multi-source loader.
    2. Load / train teacher.
    3. Distill student.
    4. Evaluate and compare.
    5. Export student as TFLite (quantized).
    6. Save JSON results.

    Returns:
        results dict with teacher + student metrics and paths.
    """
    # ── Output directory ───────────────────────────────────────────────────
    teacher_type_str = "+".join(teacher_types)

    if output_dir is None:
        color_label = color_mode.upper()
        timestamp   = datetime.now().strftime("%m%d_%H%M")
        
        # Match train.py naming convention: exported_models/10cls_RGB/distilled_v30_to_v4_10cls_RGB_0328_1910
        run_folder = f"distilled_{teacher_type_str}_to_{student_variant}_{num_classes}cls_{color_label}_{timestamp}"
        output_dir = os.path.join(params.OUTPUT_DIR, run_folder)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use output_dir for checkpoints to keep things contained
    actual_checkpoint_dir = output_dir
    os.makedirs(actual_checkpoint_dir, exist_ok=True)
    
    # Path where Keras/TFLite models will be saved (flat like train.py)
    export_path = os.path.join(output_dir, student_variant)

    logger.info("=" * 60)
    logger.info("🚀 Starting Distillation Pipeline")
    logger.info(f"   Teachers: {teacher_type_str}  →  Student: {student_variant}")
    logger.info(f"   Classes:  {num_classes}   |  Color: {color_mode.upper()}")
    logger.info(f"   Mode:     {mode}          |  T={temperature}  α={alpha}")
    logger.info("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────────────
    x_train, y_train, x_val, y_val, x_test, y_test = load_distillation_data(
        num_classes=num_classes,
        color_mode=color_mode,
    )

    # ── 2. Teacher ────────────────────────────────────────────────────────
    teacher_color = teacher_color or color_mode
    teacher_channels = 1 if teacher_color == "gray" else 3
    teacher_input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, teacher_channels)

    loaded_teachers = []
    
    # Flags and caching for dataset loading
    teacher_data_loaded = False
    x_train_teacher, y_train_teacher, x_val_teacher, y_val_teacher = None, None, None, None
    
    for i, t_type in enumerate(teacher_types):
        t_checkpoint = teacher_checkpoints[i] if teacher_checkpoints and i < len(teacher_checkpoints) else None
        
        # Auto-find checkpoint if not provided
        if not t_checkpoint:
            t_checkpoint = find_best_checkpoint(t_type, num_classes, teacher_color)
        
        if t_checkpoint and os.path.exists(t_checkpoint):
            logger.info(f"Loading teacher {t_type} from: {t_checkpoint}")
            custom_objects = {
                "DynamicSparseFocalLoss": DynamicSparseFocalLoss,
                "DynamicFocalLoss": DynamicFocalLoss,
                "sparse_focal_loss": sparse_focal_loss,
                "focal_loss": focal_loss
            }
            # Auto-inject any custom layers from the teacher's script
            clean_name = t_type.replace('digit_recognizer_', '').replace('_teacher', '')
            for prefix in ["models.", "models.digit_recognizer_", "models._tested_but_rejected.", "models._tested_but_rejected.digit_recognizer_"]:
                try:
                    mod = importlib.import_module(f"{prefix}{clean_name}")
                    for name, obj in inspect.getmembers(mod, inspect.isclass):
                        if issubclass(obj, tf.keras.layers.Layer) and obj is not tf.keras.layers.Layer:
                            custom_objects[name] = obj
                    break # Success
                except (ModuleNotFoundError, Exception):
                    continue

            t_model = tf.keras.models.load_model(t_checkpoint, custom_objects=custom_objects, safe_mode=False)
        else:
            # CRITICAL FIX: Load proper data for teacher training
            logger.info(f"Training teacher {t_type} from scratch...")
            
            # Load data in teacher's color mode only once if needed
            if not teacher_data_loaded:
                x_train_teacher, y_train_teacher, x_val_teacher, y_val_teacher, _, _ = load_distillation_data(
                    num_classes=num_classes,
                    color_mode=teacher_color,  # Use teacher's color mode
                )
                teacher_data_loaded = True
            
            t_model = train_teacher(
                teacher_type=t_type,
                num_classes=num_classes,
                color_mode=teacher_color,  # Use teacher's color mode
                x_train=x_train_teacher,
                y_train=y_train_teacher,
                x_val=x_val_teacher,
                y_val=y_val_teacher,
                epochs=teacher_epochs,
                batch_size=batch_size,
                learning_rate=teacher_lr,
                checkpoint_dir=checkpoint_dir,
                pretrained=teacher_pretrained,
                freeze_backbone=teacher_freeze_backbone,
            )

        # Freeze teacher for distillation
        t_model = freeze_teacher_model(t_model)
        loaded_teachers.append(t_model)

    if len(loaded_teachers) > 1:
        teacher = EnsembleTeacher(loaded_teachers, teacher_weights=teacher_weights)
        teacher_size = sum(get_model_size_kb(t) for t in loaded_teachers)
    else:
        teacher = loaded_teachers[0]
        teacher_size = get_model_size_kb(teacher)

    # CRITICAL: Verify teacher is working
    logger.info("Verifying teacher ensemble...")
    dummy_input = tf.zeros((1, params.INPUT_HEIGHT, params.INPUT_WIDTH, 1 if teacher_color == "gray" else 3))
    dummy_output = teacher(dummy_input)
    logger.info(f"Teacher output shape: {dummy_output.shape}")
    
    # Quick sanity check on a small batch
    teacher_preds = teacher.predict(x_test[:32], verbose=0)
    teacher_acc = np.mean(np.argmax(teacher_preds, axis=1) == y_test[:32])
    logger.info(f"Teacher sanity check accuracy (32 samples): {teacher_acc:.4f}")
    if teacher_acc < 0.5:
        logger.error("=" * 60)
        logger.error("❌ TEACHER ENSEMBLE IS NOT WORKING CORRECTLY!")
        logger.error(f"Teacher accuracy: {teacher_acc:.4f} (random for {num_classes} classes is ~{1.0/num_classes:.1%})")
        logger.error("=" * 60)
        logger.error("Possible causes:")
        logger.error("  1. Teacher checkpoints not found or wrong")
        logger.error("  2. Teacher color mode mismatch")
        logger.error("  3. EnsembleTeacher weights not properly loaded")
        logger.error("=" * 60)
        
        # List all loaded teachers for debugging
        for i, t in enumerate(loaded_teachers):
            logger.info(f"Teacher {i}: {t.name}")
            if hasattr(t, 'input_shape'):
                logger.info(f"  Input shape: {t.input_shape}")
        
        logger.warning("Continuing with broken teacher - student will likely fail!")

    # Quick teacher evaluation
    teacher_metrics = evaluate_distilled_model(teacher, (x_test, y_test))
    logger.info(f"Teacher test accuracy: {teacher_metrics['accuracy']:.4f}")

    # ── 3. Distill student ────────────────────────────────────────────────
    distiller, hist = train_student_distillation(
        teacher=teacher,
        student_variant=student_variant,
        num_classes=num_classes,
        color_mode=color_mode,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=student_epochs,
        batch_size=batch_size,
        learning_rate=student_lr,
        temperature=temperature,
        alpha=alpha,
        mode=mode,
        use_progressive=use_progressive,
        checkpoint_dir=actual_checkpoint_dir,
    )

    student = distiller.get_student()

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    student_metrics = evaluate_distilled_model(student, (x_test, y_test))
    comparison = compare_teacher_student(teacher, student, (x_test, y_test))

    logger.info("\n" + "=" * 60)
    logger.info("📊 DISTILLATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Teacher accuracy:  {teacher_metrics['accuracy']:.4f}")
    logger.info(f"Student accuracy:  {student_metrics['accuracy']:.4f}")
    logger.info(f"Accuracy retention: {comparison['comparison']['accuracy_retention']:.2%}")
    logger.info(f"Compression ratio:  {comparison['comparison']['compression_ratio']:.1f}×")

    # ── 5. Export ─────────────────────────────────────────────────────────
    # Save student artifacts into the output_dir
    export_path = os.path.join(output_dir, student_variant)
    if export_quantized:
        # Increase calibration data for PTQ
        # (This is a fallback if QAT scales aren't present)
        n_calib = min(1000, len(x_test))
        tflite_path = export_student_for_edge(
            student,
            export_path,
            quantize=True,
            representative_dataset=x_test[:n_calib],
            target_hardware=target_hardware,
        )
        logger.info(f"Student TFLite → {tflite_path}")

        # ── TQT / ESP-DL Quantitative Pipeline ──
        if use_tqt:
            # Loop through all targets if requested
            targets = getattr(params, 'TQT_ALL_TARGETS', [params.TQT_TARGET]) if getattr(params, 'TQT_EXPORT_ALL_TARGETS', False) else [params.TQT_TARGET]
            
            logger.info(f"\n🚀 STARTING TQT PIPELINE FOR TARGETS: {', '.join(targets)}")
            
            onnx_path = os.path.join(output_dir, f"{student_variant}.onnx")
            
            # CRITICAL: Simplify MUST be True to avoid -11 crashes in esp-ppq
            if export_keras_to_onnx(student, onnx_path, simplify=True):
                for target_soc in targets:
                    logger.info(f"\n⚙️ Quantizing for target: {target_soc}...")
                    cmd = [
                        sys.executable, "quantize_espdl.py",
                        "--model", student_variant,
                        "--onnx", onnx_path,
                        "--output", output_dir,
                        "--bits", str(params.TQT_NUM_BITS),
                        "--target", target_soc,
                        "--classes", str(params.NB_CLASSES),
                        "--color", color_mode.lower(),
                        "--steps", str(params.TQT_STEPS),
                        "--lr", str(params.TQT_LR),
                        "--device", params.TQT_COLLECTING_DEVICE,
                        "--calib_steps", str(params.TQT_CALIB_STEPS),
                        "--skip_onnx_export",
                        "--tflite" # Student models ALWAYS want the TFLite scale-preserved version
                    ]

                    if getattr(params, 'TQT_IS_WEIGHT_TRAINABLE', False):
                        cmd.append("--tune_weights")
                    
                    logger.info(f"   Executing TQT command: {' '.join(cmd)}")
                    try:
                        # CRITICAL FIX for exit code -11: Hide GPU from child if running on CPU
                        env = os.environ.copy()
                        if params.TQT_COLLECTING_DEVICE == "cpu":
                            env["CUDA_VISIBLE_DEVICES"] = ""

                        subprocess.run(cmd, check=True, env=env)
                        logger.info(f"✅ TQT Pipeline finished successfully for {target_soc}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"❌ TQT Pipeline failed for {target_soc} with exit code {e.returncode}")
                    except Exception as e:
                        logger.error(f"❌ TQT Pipeline error for {target_soc}: {e}")
            else:
                logger.error("❌ TQT Pipeline aborted: Student ONNX export failed")

    else:
        student.save(f"{export_path}.keras")
        tflite_path = None

    # ── 6. Save results ───────────────────────────────────────────────────
    results = {
        "teacher": {
            "type":     teacher_type_str,
            "accuracy": teacher_metrics["accuracy"],
            "size_kb":  teacher_size,
        },
        "student": {
            "type":        student_variant,
            "accuracy":    student_metrics["accuracy"],
            "size_kb":     get_model_size_kb(student),
            "tflite_path": tflite_path,
        },
        "distillation": {
            "mode":             mode,
            "temperature":      temperature,
            "alpha":            alpha,
            "use_progressive":  use_progressive,
        },
        "comparison": comparison["comparison"],
        "training": {
            "num_classes":  num_classes,
            "color_mode":   color_mode,
            "batch_size":   batch_size,
            "teacher_lr":   teacher_lr,
            "student_lr":   student_lr,
            "input_shape":  list(params.INPUT_SHAPE),
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save training report / summary into output_dir (like train.py)
    results_path = os.path.join(
        output_dir,
        f"distillation_{teacher_type_str}_to_{student_variant}_{num_classes}cls.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save alias for parity with train.py
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # ── 7. Generate Training Resume (parity with train.py) ───────────────
    
    # Save standard best_model.keras in the root
    student.save(os.path.join(output_dir, "best_model.keras"))
    
    # Save model summary
    save_model_summary_to_file(student, output_dir)
    
    # Copy logs/plots from checkpoint_dir IF they are not already in output_dir
    if checkpoint_dir != output_dir:
        for file in ["training_history.png", "training_log.csv"]:
            src = os.path.join(checkpoint_dir, file)
            dst = os.path.join(output_dir, file)
            if os.path.exists(src) and src != dst:
                shutil.copy(src, dst)

    # Generate Confusion Matrix (wrap in try/except for safety)
    try:
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        analyze_confusion_matrix(student, x_test, y_test, save_path=analysis_dir)
    except Exception as e:
        logger.warning(f"⚠️ Could not generate confusion matrix: {e}")

    # Generate Detailed Training plot from CSV if possible
    try:
        csv_log = os.path.join(output_dir, "training_log.csv")
        if os.path.exists(csv_log):
            analyze_training_history(csv_log, save_path=analysis_dir)
    except Exception as e:
        logger.warning(f"⚠️ Could not generate training history plot: {e}")

    # ── 8. Full QAT Verification ──
    try:
        if export_quantized:
            tflite_path = results['student'].get('tflite_path')
            if tflite_path and os.path.exists(tflite_path):
                qat_report = verify_tflite_full_qat(tflite_path, debug=True)
                if qat_report:
                    results['student']['qat_verification'] = qat_report
                    if qat_report['is_full_qat']:
                        logger.info("✅ TFLite QAT Verification: Model is FULL QAT (integer only)")
                    else:
                        logger.warning("⚠️ TFLite QAT Verification: Model may NOT be full QAT (float ops detected)")
                        logger.warning(f"   I/O: {qat_report['input_dtype']} / {qat_report['output_dtype']}")
                        logger.warning(f"   Quantized tensors: {qat_report['quantization_ratio']:.1%} of total")
                else:
                    logger.warning("⚠️ Could not perform QAT verification")
            else:
                 logger.warning(f"⚠️ TFLite path (to verify) not found: {tflite_path}")

    except Exception as e:
        logger.warning(f"⚠️ QAT status verification failed: {e}")
        
    logger.info(f"Results saved → {results_path}")
    
    return results