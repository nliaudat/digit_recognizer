#!/usr/bin/env python3
"""
train_super_student.py  —  Multi-Teacher Ensemble Distillation for Super Students.

Trains the V33 (10cls) or V34 (100cls) ConvNeXt super student by absorbing
knowledge from ALL previously trained models via ensemble distillation.

Strategy
────────
1. **Auto-discover all trained teachers** from ``exported_models/``:
   - Scans ``<N>cls_RGB/`` for .keras files from any model
   - Loads each into an ``EnsembleTeacher`` (weighted by accuracy if available)

2. **Build the super student** (V33 for 10cls, V34 for 100cls)

3. **Train via progressive distillation** from the ensemble:
   - ProgressiveDistiller with temperature 8→2, alpha 0.3→0.8
   - Warmup cosine LR schedule (AdamW)
   - Per-class focal loss for hard classes
   - 150-300 epochs depending on convergence

4. **Save** as a standard .keras in ``exported_models/<N>cls_RGB/v3{N}_super_student/``

Usage
─────
    # Train 10-class super student
    python train_super_student.py --classes 10 --color rgb

    # Train 100-class super student
    python train_super_student.py --classes 100 --color rgb

    # Dry run — show what teachers would be used without training
    python train_super_student.py --classes 100 --color rgb --dry-run

    # Override epochs / learning rate
    python train_super_student.py --classes 100 --color rgb --epochs 300 --lr 0.0005
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ── Project root ──────────────────────────────────────────────────────────
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Set env vars BEFORE importing parameters
os.environ.setdefault("DIGIT_NB_CLASSES", "100")
os.environ.setdefault("DIGIT_INPUT_CHANNELS", "3")

import numpy as np
import tensorflow as tf

import config as params
from config.validation import validate_full_config
from models.model_factory import create_model_by_name, resolve_model_name
from utils import get_data_splits
from utils.preprocess import preprocess_for_training
from utils.distiller import ProgressiveDistiller, DistillationProgressCallback
from utils.ensemble_teacher import EnsembleTeacher

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Teacher discovery — scan exported_models for .keras files
# ═══════════════════════════════════════════════════════════════════════════

def discover_teachers(
    num_classes: int,
    color_mode: str = "rgb",
    min_accuracy: float = 0.0,
) -> List[Dict]:
    """
    Scan ``exported_models/<N>cls_<COLOR>/`` for .keras files and return
    a list of teacher info dicts.

    Each dict contains:
        name:        Model name (e.g. "digit_recognizer_v24")
        keras_path:  Path to the .keras file
        directory:   Export subdirectory
        accuracy:    Accuracy if available (from model_comparison.csv), else None
    """
    color_label = color_mode.upper()
    export_base = os.path.join("exported_models", f"{num_classes}cls_{color_label}")

    if not os.path.isdir(export_base):
        logger.warning(f"⚠️  Export directory not found: {export_base}")
        return []

    # First, try to load accuracy from model_comparison.csv
    csv_path = os.path.join(export_base, "test_results", "model_comparison.csv")
    csv_accuracies = {}
    if os.path.isfile(csv_path):
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get("Model", "").split("_quantized")[0].split("_full")[0]
                try:
                    csv_accuracies[model_name] = float(row["Accuracy"])
                except (ValueError, KeyError):
                    pass

    teachers = []
    seen_models = set()

    for d in sorted(os.listdir(export_base)):
        dir_path = os.path.join(export_base, d)
        if not os.path.isdir(dir_path):
            continue

        # Find .keras files
        keras_files = list(Path(dir_path).rglob("*.keras"))
        if not keras_files:
            continue

        # Pick the best .keras file (prefer non-distilled, non-extracted)
        # Priority: best.keras > final.keras > *_extracted.keras > any other
        def _priority(p: Path) -> int:
            name = p.name.lower()
            if "best" in name:
                return 0
            if "final" in name:
                return 1
            if "extracted" in name:
                return 2
            return 3

        keras_files.sort(key=_priority)
        keras_path = str(keras_files[0])

        # Infer model name from directory or filename
        model_name = d.split("_DISTILL")[0].split("_distilled")[0]
        # Clean up common prefixes
        for prefix in ["distilled_", "distill_"]:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):]

        # Skip if we've already seen this model
        if model_name in seen_models:
            continue
        seen_models.add(model_name)

        accuracy = csv_accuracies.get(model_name, None)

        teachers.append({
            "name": model_name,
            "keras_path": keras_path,
            "directory": d,
            "accuracy": accuracy,
        })

        logger.info(
            f"  📦 Found teacher: {model_name} "
            f"(acc={f'{accuracy:.4f}' if accuracy else 'N/A'})"
        )

    # Sort by accuracy descending (known accuracies first, then unknown)
    teachers.sort(
        key=lambda t: (
            t["accuracy"] is None,  # None sorts last
            -(t["accuracy"] or 0.0),
        )
    )

    # Filter by minimum accuracy
    if min_accuracy > 0:
        teachers = [
            t for t in teachers
            if t["accuracy"] is not None and t["accuracy"] >= min_accuracy
        ]

    logger.info(f"  → {len(teachers)} teachers discovered")
    return teachers


def load_teacher_model(keras_path: str) -> Optional[tf.keras.Model]:
    """
    Load a teacher model from a .keras file.
    Returns None if loading fails.
    """
    logger.info(f"  📦 Loading teacher from: {keras_path}")

    if not os.path.isfile(keras_path):
        logger.warning(f"  ⚠️  File not found: {keras_path}")
        return None

    try:
        model = tf.keras.models.load_model(keras_path, compile=False)
        logger.info(f"     ✅ Loaded: {model.name} ({model.count_params():,} params)")
        return model
    except Exception as e:
        logger.warning(f"  ⚠️  Failed to load {keras_path}: {e}")
        return None


def build_ensemble_teacher(
    teachers_info: List[Dict],
    num_classes: int,
    input_shape: Tuple[int, int, int],
) -> Optional[EnsembleTeacher]:
    """
    Build an EnsembleTeacher from discovered teachers.

    Teachers are weighted by their accuracy (if available), otherwise equal weight.
    """
    loaded_teachers = []
    weights = []

    for info in teachers_info:
        model = load_teacher_model(info["keras_path"])
        if model is not None:
            loaded_teachers.append(model)
            # Weight by accuracy if available, else 0.5
            w = info["accuracy"] if info["accuracy"] is not None else 0.5
            weights.append(w)

    if not loaded_teachers:
        logger.error("❌ No teachers could be loaded!")
        return None

    # Normalize weights
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    logger.info(f"\n🎯 Building EnsembleTeacher with {len(loaded_teachers)} teachers:")
    for i, (t, w) in enumerate(zip(loaded_teachers, weights)):
        logger.info(f"   Teacher {i}: {t.name} (weight={w:.4f})")

    ensemble = EnsembleTeacher(
        teachers=loaded_teachers,
        teacher_weights=weights.tolist(),
        temperature=1.0,
        use_logits=False,
        name="super_ensemble",
    )

    # Build the ensemble with a dummy input
    dummy = tf.zeros((1, *input_shape))
    _ = ensemble(dummy, training=False)
    logger.info(f"   ✅ Ensemble built: {ensemble.count_params():,} total params")

    return ensemble


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading with augmentation
# ═══════════════════════════════════════════════════════════════════════════

def load_data(
    num_classes: int,
    color_mode: str,
    batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    """
    Load and prepare datasets for super student training.

    Returns:
        train_ds, val_ds, test_ds, x_train_raw
    """
    logger.info(f"📦 Loading data ({num_classes} classes, {color_mode})...")

    params.NB_CLASSES = num_classes
    params.INPUT_CHANNELS = 1 if color_mode == "gray" else 3
    params.update_derived_parameters()

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data_splits()

    # Keep raw training data for potential calibration
    x_train_raw = x_train.copy()

    # Preprocess
    x_train = preprocess_for_training(x_train).astype("float32")
    x_val = preprocess_for_training(x_val).astype("float32")
    x_test = preprocess_for_training(x_test).astype("float32")

    # Convert labels to int32 for sparse CE
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val = np.argmax(y_val, axis=1)
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    y_train = y_train.astype("int32")
    y_val = y_val.astype("int32")
    y_test = y_test.astype("int32")

    # Create tf.data datasets
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    logger.info(
        f"   Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}"
    )
    return train_ds, val_ds, test_ds, x_train_raw


# ═══════════════════════════════════════════════════════════════════════════
#  Super student training
# ═══════════════════════════════════════════════════════════════════════════

def build_super_student(
    num_classes: int,
    color_mode: str,
) -> tf.keras.Model:
    """
    Build the appropriate super student model based on num_classes.
    """
    if num_classes == 10:
        model_name = "digit_recognizer_v33_super_student_10"
    else:
        model_name = "digit_recognizer_v34_super_student_100"

    channels = 1 if color_mode == "gray" else 3
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)

    logger.info(f"🏗️  Building super student: {model_name}")
    logger.info(f"   Input shape: {input_shape}")
    logger.info(f"   Classes: {num_classes}")

    student = create_model_by_name(
        model_name,
        num_classes=num_classes,
        input_shape=input_shape,
    )

    logger.info(f"   Params: {student.count_params():,}")
    return student


def run_super_student_training(
    ensemble: EnsembleTeacher,
    student: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_classes: int,
    epochs: int = 200,
    learning_rate: float = 0.001,
    output_dir: str = "super_student_output",
) -> Tuple[ProgressiveDistiller, bool]:
    """
    Train the super student via progressive distillation from the ensemble.

    Uses:
    - ProgressiveDistiller with temperature 8→2, alpha 0.3→0.8
    - AdamW optimizer with warmup cosine decay
    - ReduceLROnPlateau + EarlyStopping
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀  Starting Super Student Training")
    logger.info(f"{'='*60}")
    logger.info(f"   Epochs:      {epochs}")
    logger.info(f"   LR:          {learning_rate}")
    logger.info(f"   Output:      {output_dir}")
    logger.info(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # ── Build the ProgressiveDistiller ────────────────────────────────────
    if num_classes > 10:
        # 100-class task is harder: use a lower starting temperature (4)
        # so teacher soft targets are sharper / more informative, and a
        # higher alpha (0.5) to give more weight to ground-truth labels
        # from the start, preventing the student from getting stuck in
        # the random-guess regime.
        initial_temperature = 4.0
        initial_alpha = 0.5
        logger.info(f"   [100cls mode] temperature={initial_temperature}, alpha={initial_alpha}")
    else:
        # 10-class: high temperature (8) softens teacher signals,
        # alpha=0.3 (more teacher influence early on) — works well.
        initial_temperature = 8.0
        initial_alpha = 0.3

    distiller = ProgressiveDistiller(
        student=student,
        teacher=ensemble,
        temperature=initial_temperature,
        alpha=initial_alpha,
        total_epochs=epochs,
        name="super_student_distiller",
    )

    # ── Compile with AdamW ────────────────────────────────────────────────
    try:
        import tensorflow_addons as tfa
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
        )
    except ImportError:
        logger.warning("⚠️  tensorflow-addons not available, using Adam")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
        )

    distiller.compile(
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        DistillationProgressCallback(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            min_delta=5e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, "training_log.csv"),
        ),
    ]

    # Custom Callback that saves the student model when val_accuracy improves
    # (Uses plain Callback instead of ModelCheckpoint to avoid Keras 3
    #  read-only 'model' property issue.)
    class _StudentCheckpoint(tf.keras.callbacks.Callback):
        """Saves the standalone student model when val_accuracy improves."""

        def __init__(self, student: tf.keras.Model, filepath: str, verbose: int = 1):
            super().__init__()
            self._student = student
            self.filepath = filepath
            self.verbose = verbose
            self.best_val_acc = -1.0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_acc = logs.get("val_accuracy", 0.0)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._student.save(self.filepath)
                if self.verbose:
                    bn = os.path.basename(self.filepath)
                    print(f"\n  🏆 Best val_acc improved to {val_acc:.4f} — student saved to {bn}")

    student_ckpt = _StudentCheckpoint(
        student=student,
        filepath=os.path.join(output_dir, "best_student.keras"),
        verbose=1,
    )
    callbacks.append(student_ckpt)

    # ── Train ─────────────────────────────────────────────────────────────
    interrupted = False
    try:
        history = distiller.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user (KeyboardInterrupt)")
        interrupted = True
        # Keras .fit() may or may not have a history object depending on when interrupted
        history = getattr(distiller, "history", None)
        if history is None or not history.history:
            history = type("obj", (object,), {"history": {}})()

    # ── Save final model ──────────────────────────────────────────────────
    distiller.save(os.path.join(output_dir, "final_super_student.keras"))
    logger.info(f"✅ Super student training complete — model saved to {output_dir}")

    # Log best accuracy
    best_acc = max(history.history.get("val_accuracy", [0]))
    logger.info(f"   Best validation accuracy: {best_acc:.4f}")

    return distiller, interrupted


def extract_student(distiller: ProgressiveDistiller) -> tf.keras.Model:
    """Extract the trained student from the distiller wrapper."""
    logger.info("📦 Extracting student from distiller...")

    if hasattr(distiller, "student"):
        student = distiller.student
        logger.info(f"   ✅ Extracted student: {student.name}")
        logger.info(f"      Params: {student.count_params():,}")
        return student

    # Fallback: search submodels
    for layer in distiller.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 1:
            try:
                if layer.output_shape[-1] == params.NB_CLASSES:
                    logger.info(f"   ✅ Found student: {layer.name}")
                    return layer
            except (AttributeError, ValueError):
                continue

    raise RuntimeError("Could not extract student from distiller")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-teacher ensemble distillation for super students",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--classes",
        type=int,
        default=100,
        choices=[10, 100],
        help="Number of output classes (default: 100)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="rgb",
        choices=["gray", "rgb"],
        help="Color mode (default: rgb)",
    )
    parser.add_argument(
        "--epochs", "--epoch",
        type=int,
        default=200,
        help="Max training epochs (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--min-teacher-accuracy",
        type=float,
        default=0.0,
        help="Minimum teacher accuracy to include (default: 0.0 = all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not set)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually training",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    validate_full_config()

    # ── Set params ────────────────────────────────────────────────────────
    params.NB_CLASSES = args.classes
    params.INPUT_CHANNELS = 1 if args.color == "gray" else 3
    params.update_derived_parameters()

    color_label = args.color.upper()
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS)

    # ── Output directory ──────────────────────────────────────────────────
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_version = "v33" if args.classes == 10 else "v34"
        timestamp = datetime.now().strftime("%m%d_%H%M")
        run_folder = (
            f"{model_version}_super_student_{args.classes}cls_{color_label}_"
            f"ENSEMBLE_{timestamp}"
        )
        output_dir = os.path.join(
            "exported_models", f"{args.classes}cls_{color_label}", run_folder
        )

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Discover teachers ─────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("🔍 Step 1/5: Discovering trained teachers")
    logger.info(f"{'='*60}")

    teachers_info = discover_teachers(
        num_classes=args.classes,
        color_mode=args.color,
        min_accuracy=args.min_teacher_accuracy,
    )

    if not teachers_info:
        logger.error("❌ No teachers found! Train some models first.")
        sys.exit(1)

    # ── Step 2: Build ensemble ────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("🎯 Step 2/5: Building ensemble teacher")
    logger.info(f"{'='*60}")

    ensemble = build_ensemble_teacher(
        teachers_info=teachers_info,
        num_classes=args.classes,
        input_shape=input_shape,
    )

    if ensemble is None:
        logger.error("❌ Failed to build ensemble teacher!")
        sys.exit(1)

    # ── Step 3: Build super student ───────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("🏗️  Step 3/5: Building super student model")
    logger.info(f"{'='*60}")

    student = build_super_student(args.classes, args.color)

    # ── Step 4: Load data ─────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("📦 Step 4/5: Loading data")
    logger.info(f"{'='*60}")

    train_ds, val_ds, test_ds, x_train_raw = load_data(
        num_classes=args.classes,
        color_mode=args.color,
        batch_size=args.batch_size,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📋  SUPER STUDENT TRAINING PLAN")
    logger.info("=" * 60)
    logger.info(f"   Model:     {'V33 (ConvNeXt-T, 10cls)' if args.classes == 10 else 'V34 (ConvNeXt-S, 100cls)'}")
    logger.info(f"   Teachers:  {len(teachers_info)} discovered")
    logger.info(f"   Ensemble:  {len(ensemble.teachers)} loaded")
    logger.info(f"   Classes:   {args.classes}")
    logger.info(f"   Color:     {args.color.upper()}")
    logger.info(f"   Epochs:    {args.epochs}")
    logger.info(f"   LR:        {args.lr}")
    logger.info(f"   Batch:     {args.batch_size}")
    logger.info(f"   Output:    {output_dir}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("\n🏁 DRY RUN — stopping. Use without --dry-run to execute.")
        return

    # ── Step 5: Train super student ───────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("🚀 Step 5/5: Training super student via ensemble distillation")
    logger.info(f"{'='*60}")

    distiller, training_interrupted = run_super_student_training(
        ensemble=ensemble,
        student=student,
        train_ds=train_ds,
        val_ds=val_ds,
        num_classes=args.classes,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=output_dir,
    )

    # ── Handle keyboard interrupt ─────────────────────────────────────────
    if training_interrupted:
        print()  # blank line for readability
        while True:
            choice = input(
                "⚡ Training interrupted by user.\n"
                "  [F] Finish gracefully — save best student, evaluate, generate summary\n"
                "  [C] Continue training (resume)\n"
                "  [A] Abort — exit immediately (no save)\n"
                "  Choose [F/C/A]: "
            ).strip().upper()

            if choice == "F":
                logger.info("→ Finishing gracefully...")
                break
            elif choice == "C":
                logger.info("→ Resuming training...")
                # Re-run training with the same distiller (it picks up from current state)
                distiller, training_interrupted = run_super_student_training(
                    ensemble=ensemble,
                    student=student,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    num_classes=args.classes,
                    epochs=args.epochs,
                    learning_rate=args.lr,
                    output_dir=output_dir,
                )
                if not training_interrupted:
                    break  # training completed normally
                # If interrupted again, loop back to the prompt
                continue
            elif choice == "A":
                logger.warning("→ Aborting. No model saved.")
                sys.exit(1)
            else:
                print("  ❌ Invalid choice. Please enter F, C, or A.")

    # ── Extract and save standalone student ───────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("📦 Extracting and saving standalone super student")
    logger.info(f"{'='*60}")

    trained_student = extract_student(distiller)

    # Save standalone student .keras
    model_version = "v33" if args.classes == 10 else "v34"
    student_keras_path = os.path.join(
        output_dir, f"{model_version}_super_student_extracted.keras"
    )
    trained_student.save(student_keras_path)
    logger.info(f"   ✅ Standalone student saved → {student_keras_path}")

    # Also save a copy with the full model name for discoverability
    full_name = (
        "digit_recognizer_v33_super_student_10"
        if args.classes == 10
        else "digit_recognizer_v34_super_student_100"
    )
    full_keras_path = os.path.join(output_dir, f"{full_name}.keras")
    trained_student.save(full_keras_path)
    logger.info(f"   ✅ Full-name copy saved → {full_keras_path}")

    # ── Evaluate on test set ──────────────────────────────────────────────
    logger.info("\n📊 Evaluating super student on test set...")
    # Compile the extracted student before evaluation (Keras 3 requires it)
    trained_student.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    test_loss, test_acc = trained_student.evaluate(test_ds, verbose=0)
    logger.info(f"   Test accuracy: {test_acc:.4f}")

    # ── Per-class accuracy ────────────────────────────────────────────────
    logger.info("\n📊 Per-class accuracy:")
    all_preds = []
    all_labels = []
    for x_batch, y_batch in test_ds:
        preds = trained_student.predict(x_batch, verbose=0)
        all_preds.append(preds)
        all_labels.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pred_classes = np.argmax(all_preds, axis=1)

    per_class_acc = {}
    for cls in range(args.classes):
        mask = all_labels == cls
        if mask.sum() > 0:
            acc = (pred_classes[mask] == all_labels[mask]).mean()
            per_class_acc[int(cls)] = float(acc)

    # Log classes with lowest accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
    logger.info(f"   Top 10 hardest classes:")
    for cls, acc in sorted_classes[:10]:
        logger.info(f"     Class {cls}: {acc:.4f}")

    # Save per-class accuracy
    acc_path = os.path.join(output_dir, "per_class_accuracy.json")
    with open(acc_path, "w") as f:
        json.dump(per_class_acc, f, indent=2)
    logger.info(f"   ✅ Per-class accuracy saved → {acc_path}")

    # ── Save training summary ─────────────────────────────────────────────
    summary = {
        "model": full_name,
        "num_classes": args.classes,
        "color_mode": args.color,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "num_teachers": len(teachers_info),
        "num_ensemble_teachers": len(ensemble.teachers),
        "epochs_trained": len(distiller.history.history.get("loss", [])),
        "best_val_accuracy": float(max(distiller.history.history.get("val_accuracy", [0]))),
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"   ✅ Training summary saved → {summary_path}")

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✅  SUPER STUDENT TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Model:    {full_name}")
    logger.info(f"   Test acc: {test_acc:.4f}")
    logger.info(f"   Teachers: {len(teachers_info)} discovered, {len(ensemble.teachers)} loaded")
    logger.info(f"   Output:   {output_dir}")
    logger.info("=" * 60)
    logger.info(f"\n💡 Now use distill_best.py with this super student as teacher:")
    logger.info(f"   python distill_best.py --teacher {model_version} "
                f"--student v16 --classes {args.classes} --color {args.color}")


if __name__ == "__main__":
    validate_full_config()
    main()
