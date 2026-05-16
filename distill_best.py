#!/usr/bin/env python3
"""
distill_best.py  —  Best-practice knowledge distillation script.

Reads ``exported_models/100cls_RGB/test_results/model_comparison.csv`` to
identify the top-performing teacher, then distills its knowledge into a
compact student model for edge deployment.

Strategy
────────
1. **Teacher selection** — picks the highest-accuracy model from the CSV.
   The teacher is a full-precision (non-quantized) .keras file in the same
   export directory.  If no float .keras is found, the quantized TFLite is
   used as a fallback (less ideal but still works).

2. **Student selection** — you choose a lightweight architecture (v4, v15,
   v16, v23, v31_student_*, etc.).  The script will suggest the best
   student based on the accuracy/size trade-off from the CSV.

3. **Distillation** — uses the ``ProgressiveDistiller`` wrapper with:
   - Temperature = 4.0 (soften teacher logits)
   - Alpha = 0.5 (balance teacher vs. hard labels)
   - Beta = 0.3 (intermediate feature loss)
   - ReduceLROnPlateau + EarlyStopping
   - 150 epochs max

4. **Export** — saves the distilled student as:
   - Standalone .keras (extracted from the distiller wrapper)
   - Float32 TFLite
   - Float16 TFLite (optional)
   - Full integer TFLite (optional, requires calibration data)

Usage
─────
    # Auto-select best teacher, distill into v4
    python distill_best.py --student v4 --classes 100 --color rgb

    # Specify teacher explicitly
    python distill_best.py --teacher v24 --student v15 --classes 100 --color rgb

    # Dry run — show what would be done without training
    python distill_best.py --student v4 --classes 100 --color rgb --dry-run

    # With TFLite export
    python distill_best.py --student v4 --classes 100 --color rgb --export-tflite

    # Full pipeline with TQT (ESP-DL quantization)
    python distill_best.py --student v4 --classes 100 --color rgb --tqt
"""

import argparse
import csv
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

import parameters as params
from config.validation import validate_full_config
from config.distillation import (
    DISTILLATION_TEMPERATURE,
    DISTILLATION_ALPHA,
    DISTILLATION_BETA,
    DISTILLATION_EPOCHS,
    DISTILLATION_LEARNING_RATE,
    DISTILLATION_BATCH_SIZE,
    DISTILLATION_EARLY_STOPPING_PATIENCE,
    DISTILLATION_LR_SCHEDULER_PATIENCE,
)
from models.model_factory import create_model_by_name, resolve_model_name
from utils import get_data_splits
from utils.preprocess import preprocess_for_training
from utils.distiller import ProgressiveDistiller, DistillationProgressCallback
from utils.ensemble_teacher import EnsembleTeacher
from utils.tflite_converter import (
    Float16Strategy,
    FullIntegerStrategy,
    build_representative_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  CSV parsing — find the best teacher
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH = "exported_models/100cls_RGB/test_results/model_comparison.csv"
EXPORT_BASE = "exported_models/100cls_RGB"


def get_csv_path(num_classes: int, color_mode: str = "rgb") -> str:
    """Get the CSV path for the given class count and color mode."""
    color_label = color_mode.upper()
    return f"exported_models/{num_classes}cls_{color_label}/test_results/model_comparison.csv"


def get_export_base(num_classes: int, color_mode: str = "rgb") -> str:
    """Get the export base directory for the given class count and color mode."""
    color_label = color_mode.upper()
    return f"exported_models/{num_classes}cls_{color_label}"


def parse_model_csv(csv_path: str = CSV_PATH) -> List[Dict]:
    """Parse the model comparison CSV into a list of dicts."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["Accuracy"] = float(row["Accuracy"])
            row["Accuracy_RealOnly"] = float(row["Accuracy_RealOnly"])
            row["Parameters"] = int(row["Parameters"])
            row["Size_KB"] = float(row["Size_KB"])
            row["Inference_Time_ms"] = float(row["Inference_Time_ms"])
            records.append(row)
    return records


def find_best_teacher(
    records: List[Dict],
    min_accuracy: float = 0.90,
    export_base: str = EXPORT_BASE,
) -> Optional[Dict]:
    """
    Find the best teacher from the CSV.

    Criteria:
    1. Highest accuracy overall
    2. Must be >= min_accuracy
    3. Prefer models with a corresponding .keras file in the export dir
    """
    # Sort by accuracy descending
    sorted_records = sorted(records, key=lambda r: r["Accuracy"], reverse=True)

    for rec in sorted_records:
        if rec["Accuracy"] < min_accuracy:
            continue

        # Check if there's a .keras file in the export directory
        export_dir = os.path.join(export_base, rec["Directory"])
        keras_files = list(Path(export_dir).rglob("*.keras"))
        if keras_files:
            rec["_keras_path"] = str(keras_files[0])
            return rec

    # Fallback: return the best even without .keras
    if sorted_records:
        rec = sorted_records[0]
        export_dir = os.path.join(export_base, rec["Directory"])
        keras_files = list(Path(export_dir).rglob("*.keras"))
        if keras_files:
            rec["_keras_path"] = str(keras_files[0])
        else:
            rec["_keras_path"] = None
        return rec

    return None


def suggest_best_student(records: List[Dict]) -> str:
    """
    Suggest the best student model based on accuracy/size trade-off.

    Looks for models that are small (< 300 KB) but have good accuracy.
    """
    # Filter to small models
    small_models = [
        r for r in records
        if r["Size_KB"] < 300 and r["Accuracy"] > 0.80
    ]
    small_models.sort(key=lambda r: r["Accuracy"], reverse=True)

    if small_models:
        best = small_models[0]
        # Extract model name from the Model column
        model_name = best["Model"].split("_quantized")[0].split("_full")[0]
        logger.info(
            f"  Suggested student: {model_name} "
            f"(acc={best['Accuracy']:.2%}, size={best['Size_KB']:.0f}KB)"
        )
        return model_name
    return "v4"


def find_teacher_keras(teacher_name: str, num_classes: int = 100, color_mode: str = "rgb") -> Optional[str]:
    """
    Find the .keras file for a named teacher by scanning export directories.
    """
    export_dir = get_export_base(num_classes, color_mode)
    if not os.path.isdir(export_dir):
        return None

    for d in os.listdir(export_dir):
        if teacher_name.lower() in d.lower():
            keras_files = list(
                Path(os.path.join(export_dir, d)).rglob("*.keras")
            )
            if keras_files:
                return str(keras_files[0])
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Teacher loading
# ═══════════════════════════════════════════════════════════════════════════

def load_teacher_model(keras_path: str) -> tf.keras.Model:
    """
    Load a teacher model from a .keras file.

    Handles:
    - Standard Keras models
    - ProgressiveDistiller wrappers (extracts the teacher submodel)
    - TFLite files (fallback — creates a dummy teacher)
    """
    logger.info(f"📦 Loading teacher from: {keras_path}")

    if not os.path.isfile(keras_path):
        raise FileNotFoundError(f"Teacher not found: {keras_path}")

    # Try loading as a standard Keras model
    try:
        model = tf.keras.models.load_model(keras_path, compile=False)
        logger.info(f"   ✅ Loaded teacher: {model.name}")
        logger.info(f"      Params: {model.count_params():,}")
        return model
    except Exception as e:
        logger.warning(f"   ⚠️ Standard load failed: {e}")
        logger.info("   Trying to extract teacher from ProgressiveDistiller...")

    # Try extracting teacher from a ProgressiveDistiller wrapper
    try:
        import zipfile
        import h5py

        # Build a fresh teacher model
        # Infer architecture from filename or use v24 as default
        teacher_arch = _infer_teacher_architecture(keras_path)
        teacher = create_model_by_name(teacher_arch)

        # Load weights from the .keras zip
        with zipfile.ZipFile(keras_path, "r") as zf:
            with zf.open("model.weights.h5") as f:
                with h5py.File(f, "r") as h5:
                    weight_map = {}

                    def collect_weights(name, obj):
                        if not isinstance(obj, h5py.Dataset):
                            return
                        parts = name.split("/")
                        # Teacher weights are under:
                        # layers/functional/layers/functional/layers/<layer_name>/vars/<idx>
                        if (
                            len(parts) >= 7
                            and parts[0] == "layers"
                            and parts[1] == "functional"
                            and parts[2] == "layers"
                            and parts[3] == "functional"
                            and parts[4] == "layers"
                            and parts[6] == "vars"
                        ):
                            layer_name = parts[5]
                            idx = int(parts[7])
                            if layer_name not in weight_map:
                                weight_map[layer_name] = {}
                            weight_map[layer_name][idx] = obj[()]

                    h5.visititems(collect_weights)

        # Set weights
        set_count = 0
        for layer in teacher.layers:
            if layer.name in weight_map:
                sorted_w = [
                    weight_map[layer.name][i]
                    for i in sorted(weight_map[layer.name].keys())
                ]
                try:
                    layer.set_weights(sorted_w)
                    set_count += 1
                except Exception as e:
                    logger.debug(f"   ⚠️ Could not set '{layer.name}': {e}")

        logger.info(f"   ✅ Extracted teacher from distiller ({set_count} layers)")
        return teacher

    except Exception as e:
        logger.error(f"   ❌ Failed to extract teacher: {e}")
        raise


def _infer_teacher_architecture(keras_path: str) -> str:
    """Infer the teacher architecture from the filename or path."""
    path_lower = keras_path.lower()
    # Check super students first (v33, v34 have longer names)
    if "v33" in path_lower or "super_student_10" in path_lower:
        return "digit_recognizer_v33_super_student_10"
    if "v34" in path_lower or "super_student_100" in path_lower:
        return "digit_recognizer_v34_super_student_100"
    for arch in ["v32", "v31", "v30", "v29", "v28", "v27", "v24", "v23", "v19", "v18", "v17", "v16"]:
        if arch in path_lower:
            return f"digit_recognizer_{arch}"
    return "digit_recognizer_v24"


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data(
    num_classes: int,
    color_mode: str,
    batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load and prepare datasets for distillation."""
    logger.info(f"📦 Loading data ({num_classes} classes, {color_mode})...")

    params.NB_CLASSES = num_classes
    params.INPUT_CHANNELS = 1 if color_mode == "gray" else 3
    params.update_derived_parameters()

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data_splits()

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
    return train_ds, val_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════
#  Distillation training
# ═══════════════════════════════════════════════════════════════════════════

def build_student_model(
    student_name: str,
    num_classes: int,
    color_mode: str,
) -> tf.keras.Model:
    """Build a fresh student model."""
    channels = 1 if color_mode == "gray" else 3
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)

    logger.info(f"🏗️  Building student: {student_name}")
    logger.info(f"   Input shape: {input_shape}")
    logger.info(f"   Classes: {num_classes}")

    student = create_model_by_name(
        student_name,
        num_classes=num_classes,
        input_shape=input_shape,
    )

    logger.info(f"   Params: {student.count_params():,}")
    return student


def run_distillation(
    teacher: tf.keras.Model,
    student: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_classes: int,
    temperature: float = 4.0,
    alpha: float = 0.5,
    beta: float = 0.3,
    epochs: int = 150,
    learning_rate: float = 0.001,
    output_dir: str = "distill_output",
) -> ProgressiveDistiller:
    """
    Run ProgressiveDistillation training.

    Uses the ProgressiveDistiller which dynamically adjusts temperature
    and alpha during training for better knowledge transfer.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀  Starting Progressive Distillation")
    logger.info(f"{'='*60}")
    logger.info(f"   Temperature: {temperature}")
    logger.info(f"   Alpha:       {alpha}")
    logger.info(f"   Beta:        {beta}")
    logger.info(f"   Epochs:      {epochs}")
    logger.info(f"   LR:          {learning_rate}")
    logger.info(f"   Output:      {output_dir}")
    logger.info(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # ── Build the ProgressiveDistiller ────────────────────────────────────
    distiller = ProgressiveDistiller(
        student=student,
        teacher=teacher,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        num_classes=num_classes,
        name="progressive_distiller",
    )

    # ── Compile ───────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
            patience=DISTILLATION_LR_SCHEDULER_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=DISTILLATION_EARLY_STOPPING_PATIENCE,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_distill.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, "training_log.csv"),
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    history = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save final model ──────────────────────────────────────────────────
    distiller.save(os.path.join(output_dir, "final_distill.keras"))
    logger.info(f"✅ Distillation complete — model saved to {output_dir}")

    # Log best accuracy
    best_acc = max(history.history.get("val_accuracy", [0]))
    logger.info(f"   Best validation accuracy: {best_acc:.4f}")

    return distiller


# ═══════════════════════════════════════════════════════════════════════════
#  Student extraction from distiller wrapper
# ═══════════════════════════════════════════════════════════════════════════

def extract_student(distiller: ProgressiveDistiller) -> tf.keras.Model:
    """Extract the trained student from the distiller wrapper."""
    logger.info("📦 Extracting student from distiller...")

    # The student is stored as a submodel attribute
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
#  TFLite export
# ═══════════════════════════════════════════════════════════════════════════

def export_tflite(
    student: tf.keras.Model,
    output_dir: str,
    model_name: str,
    x_train_raw: Optional[np.ndarray] = None,
    export_float16: bool = True,
    export_int8: bool = True,
):
    """Export the student model to TFLite formats."""
    logger.info(f"\n🔄 Exporting TFLite models...")

    # ── Float32 ───────────────────────────────────────────────────────────
    logger.info("   Converting to Float32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(student)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_f32 = converter.convert()
    f32_path = os.path.join(output_dir, f"{model_name}_float32.tflite")
    with open(f32_path, "wb") as f:
        f.write(tflite_f32)
    logger.info(f"   ✅ Float32: {f32_path} ({len(tflite_f32)/1024:.1f} KB)")

    # ── Float16 ───────────────────────────────────────────────────────────
    if export_float16:
        logger.info("   Converting to Float16 TFLite...")
        strategy = Float16Strategy()
        tflite_f16 = strategy.convert(student)
        f16_path = os.path.join(output_dir, f"{model_name}_float16.tflite")
        with open(f16_path, "wb") as f:
            f.write(tflite_f16)
        logger.info(f"   ✅ Float16: {f16_path} ({len(tflite_f16)/1024:.1f} KB)")

    # ── Full Integer (Int8) ───────────────────────────────────────────────
    if export_int8 and x_train_raw is not None:
        logger.info("   Converting to Full Integer (Int8) TFLite...")
        try:
            rep_dataset = build_representative_dataset(
                x_train_raw,
                num_samples=200,
            )
            strategy = FullIntegerStrategy(representative_dataset=rep_dataset)
            tflite_int8 = strategy.convert(student)
            int8_path = os.path.join(output_dir, f"{model_name}_full_integer_quant.tflite")
            with open(int8_path, "wb") as f:
                f.write(tflite_int8)
            logger.info(f"   ✅ Int8: {int8_path} ({len(tflite_int8)/1024:.1f} KB)")
        except Exception as e:
            logger.warning(f"   ⚠️ Int8 conversion failed: {e}")

    logger.info(f"✅ TFLite export complete")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Best-practice knowledge distillation for digit recognizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="Teacher model name (e.g. v24). Auto-selected from CSV if not set.",
    )
    parser.add_argument(
        "--teacher-keras",
        type=str,
        default=None,
        help="Path to teacher .keras file (overrides auto-detection).",
    )
    parser.add_argument(
        "--student",
        type=str,
        default=None,
        help="Student model architecture (e.g. v4, v15, v16, v23, v31_student_medium). "
             "Auto-suggested from CSV if not set.",
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
        "--temperature",
        type=float,
        default=DISTILLATION_TEMPERATURE,
        help=f"Distillation temperature (default: {DISTILLATION_TEMPERATURE})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DISTILLATION_ALPHA,
        help=f"Balance between teacher and hard labels (default: {DISTILLATION_ALPHA})",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DISTILLATION_BETA,
        help=f"Intermediate feature loss weight (default: {DISTILLATION_BETA})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DISTILLATION_EPOCHS,
        help=f"Max distillation epochs (default: {DISTILLATION_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DISTILLATION_LEARNING_RATE,
        help=f"Learning rate (default: {DISTILLATION_LEARNING_RATE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DISTILLATION_BATCH_SIZE,
        help=f"Batch size (default: {DISTILLATION_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not set)",
    )
    parser.add_argument(
        "--export-tflite",
        action="store_true",
        help="Export TFLite models after distillation",
    )
    parser.add_argument(
        "--tqt",
        action="store_true",
        help="Run TQT/ESP-DL quantization pipeline after distillation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually training",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=CSV_PATH,
        help=f"Path to model comparison CSV (default: {CSV_PATH})",
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

    # ── Parse CSV for teacher/student suggestions ─────────────────────────
    csv_path = get_csv_path(args.classes, args.color)
    export_base = get_export_base(args.classes, args.color)

    if not os.path.isfile(csv_path):
        logger.warning(f"⚠️  CSV not found at {csv_path} — using explicit teacher/student args")
        records = []
    else:
        records = parse_model_csv(csv_path)

    # ── Teacher selection ─────────────────────────────────────────────────
    teacher_keras_path = args.teacher_keras

    if teacher_keras_path is None:
        if args.teacher:
            # Find teacher by name
            teacher_keras_path = find_teacher_keras(args.teacher, args.classes, args.color)
            if teacher_keras_path is None:
                logger.error(
                    f"❌ Could not find .keras file for teacher '{args.teacher}' "
                    f"under {export_base}"
                )
                sys.exit(1)
        else:
            # Auto-select best teacher from CSV
            best = find_best_teacher(records, export_base=export_base)
            if best is None:
                logger.error("❌ No suitable teacher found in CSV")
                sys.exit(1)

            teacher_name = best["Model"].split("_quantized")[0]
            logger.info(f"🏆 Auto-selected teacher: {teacher_name}")
            logger.info(f"   Accuracy: {best['Accuracy']:.2%}")
            logger.info(f"   Size: {best['Size_KB']:.0f} KB")
            logger.info(f"   Directory: {best['Directory']}")

            teacher_keras_path = best.get("_keras_path")
            if teacher_keras_path is None:
                logger.error(
                    f"❌ No .keras file found for best teacher in {best['Directory']}"
                )
                sys.exit(1)

    logger.info(f"   Teacher .keras: {teacher_keras_path}")

    # ── Student selection ─────────────────────────────────────────────────
    student_name = args.student
    if student_name is None:
        student_name = suggest_best_student(records)
        logger.info(f"💡 Suggested student: {student_name}")
        logger.info(f"   (use --student to override)")

    # Resolve model name
    resolved_student = resolve_model_name(student_name)
    logger.info(f"   Resolved student: {resolved_student}")

    # ── Output directory ──────────────────────────────────────────────────
    if args.output_dir:
        output_dir = args.output_dir
    else:
        color_label = args.color.upper()
        timestamp = datetime.now().strftime("%m%d_%H%M")
        run_folder = (
            f"distilled_{resolved_student}_{args.classes}cls_{color_label}_"
            f"DISTILL_{timestamp}"
        )
        output_dir = os.path.join(
            "exported_models", f"{args.classes}cls_{color_label}", run_folder
        )

    os.makedirs(output_dir, exist_ok=True)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📋  DISTILLATION PLAN")
    logger.info("=" * 60)
    logger.info(f"   Teacher:  {teacher_keras_path}")
    logger.info(f"   Student:  {resolved_student}")
    logger.info(f"   Classes:  {args.classes}")
    logger.info(f"   Color:    {args.color.upper()}")
    logger.info(f"   Temp:     {args.temperature}")
    logger.info(f"   Alpha:    {args.alpha}")
    logger.info(f"   Beta:     {args.beta}")
    logger.info(f"   Epochs:   {args.epochs}")
    logger.info(f"   LR:       {args.lr}")
    logger.info(f"   Batch:    {args.batch_size}")
    logger.info(f"   Output:   {output_dir}")
    logger.info(f"   TFLite:   {'Yes' if args.export_tflite else 'No'}")
    logger.info(f"   TQT:      {'Yes' if args.tqt else 'No'}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("\n🏁 DRY RUN — stopping. Use without --dry-run to execute.")
        return

    # ── Step 1: Load teacher ──────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("📦 Step 1/5: Loading teacher model")
    logger.info(f"{'='*60}")
    teacher = load_teacher_model(teacher_keras_path)

    # ── Step 2: Build student ─────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("🏗️  Step 2/5: Building student model")
    logger.info(f"{'='*60}")
    student = build_student_model(resolved_student, args.classes, args.color)

    # ── Step 3: Load data ─────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("📦 Step 3/5: Loading data")
    logger.info(f"{'='*60}")
    train_ds, val_ds, test_ds = load_data(
        num_classes=args.classes,
        color_mode=args.color,
        batch_size=args.batch_size,
    )

    # ── Step 4: Run distillation ──────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("🚀 Step 4/5: Running distillation")
    logger.info(f"{'='*60}")
    distiller = run_distillation(
        teacher=teacher,
        student=student,
        train_ds=train_ds,
        val_ds=val_ds,
        num_classes=args.classes,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=output_dir,
    )

    # ── Step 5: Extract student & export ──────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("📦 Step 5/5: Extracting student and exporting")
    logger.info(f"{'='*60}")

    # Extract the trained student
    trained_student = extract_student(distiller)

    # Save standalone student .keras
    student_keras_path = os.path.join(output_dir, f"{resolved_student}_extracted.keras")
    trained_student.save(student_keras_path)
    logger.info(f"   ✅ Standalone student saved → {student_keras_path}")

    # Evaluate on test set
    logger.info("\n📊 Evaluating student on test set...")
    test_loss, test_acc = trained_student.evaluate(test_ds, verbose=0)
    logger.info(f"   Test accuracy: {test_acc:.4f}")

    # TFLite export
    if args.export_tflite:
        # Load raw training data for int8 calibration
        (x_train_raw, _), _, _ = get_data_splits()
        export_tflite(
            student=trained_student,
            output_dir=output_dir,
            model_name=resolved_student,
            x_train_raw=x_train_raw,
            export_float16=True,
            export_int8=True,
        )

    # TQT pipeline (optional)
    if args.tqt:
        logger.info(f"\n{'='*60}")
        logger.info("🔧 Running TQT/ESP-DL quantization pipeline")
        logger.info(f"{'='*60}")
        try:
            from extract_keras_tqt import (
                load_calibration_data,
                run_tqt_pipeline,
            )

            calib_data = load_calibration_data(
                num_classes=args.classes,
                color_mode=args.color,
                calib_steps=250,
            )

            run_tqt_pipeline(
                student=trained_student,
                model_name=resolved_student,
                output_dir=output_dir,
                calib_data=calib_data,
                target="esp32",
                num_classes=args.classes,
                color_mode=args.color,
                device="cpu",
                timeout=600,
            )
        except Exception as e:
            logger.warning(f"   ⚠️ TQT pipeline failed: {e}")
            logger.info("   (TQT requires esp_ppq to be installed)")

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✅  DISTILLATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Teacher:  {os.path.basename(teacher_keras_path)}")
    logger.info(f"   Student:  {resolved_student}")
    logger.info(f"   Test acc: {test_acc:.4f}")
    logger.info(f"   Output:   {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    validate_full_config()
    main()
