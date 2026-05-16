#!/usr/bin/env python3
"""
convert_v31_to_tflite.py
========================
Convert standalone v31_extracted.keras models to TFLite suite
(dynamic range + float16 + full int8).

Usage:
    python convert_v31_to_tflite.py

The script processes two models:
    exported_models/100cls_RGB/distilled_v31_100cls_RGB_TQT_SOFTMAX_0515_1859/v31_extracted.keras
    exported_models/100cls_RGB/distilled_v31_100cls_RGB_TQT_SOFTMAX_0515_2144/v31_extracted.keras
"""

import logging
import os
import sys
from pathlib import Path

# ── Set env vars BEFORE any project imports (avoids interactive prompts) ─
os.environ.setdefault("DIGIT_NB_CLASSES", "100")
os.environ.setdefault("DIGIT_INPUT_CHANNELS", "3")

# ── Project root setup ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import tensorflow as tf

import parameters as params
from config.validation import validate_full_config
from utils import get_data_splits
from utils.preprocess import preprocess_for_inference
from utils.tflite_converter import (
    DynamicRangeStrategy,
    Float16Strategy,
    FullIntegerStrategy,
    build_representative_dataset,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Model paths
# ---------------------------------------------------------------------------

MODEL_DIRS = [
    r"exported_models\100cls_RGB\distilled_v31_100cls_RGB_TQT_SOFTMAX_0515_1859",
    r"exported_models\100cls_RGB\distilled_v31_100cls_RGB_TQT_SOFTMAX_0515_2144",
]

KERAS_FILENAME = "v31_extracted.keras"

# Output filenames
OUTPUT_NAMES = {
    "dynamic_range": "v31_dynamic_range.tflite",
    "float16":       "v31_float16.tflite",
    "full_integer":  "v31_full_integer.tflite",
}


# ---------------------------------------------------------------------------
#  Conversion helpers
# ---------------------------------------------------------------------------

def load_keras_model(keras_path: str) -> tf.keras.Model:
    """Load a standalone .keras model with fallback for custom objects."""
    logger.info(f"📦 Loading model: {keras_path}")
    try:
        model = tf.keras.models.load_model(keras_path, compile=False)
        logger.info(f"   ✅ Loaded: {model.name}  |  params: {model.count_params():,}")
        return model
    except Exception as e:
        logger.warning(f"   ⚠️  Direct load failed: {e}")
        logger.info("   Trying with custom_objects (distiller classes)...")
        try:
            from utils.distiller import (
                Distiller, MixedInputDistiller, ProgressiveDistiller,
                DistillationProgressCallback
            )
            from utils.ensemble_teacher import EnsembleTeacher
            custom_objects = {
                'Distiller': Distiller,
                'MixedInputDistiller': MixedInputDistiller,
                'ProgressiveDistiller': ProgressiveDistiller,
                'DistillationProgressCallback': DistillationProgressCallback,
                'EnsembleTeacher': EnsembleTeacher,
            }
            model = tf.keras.models.load_model(
                keras_path, custom_objects=custom_objects, compile=False
            )
            logger.info(f"   ✅ Loaded with custom_objects: {model.name}")
            return model
        except Exception as e2:
            logger.error(f"   ❌ Failed with custom_objects too: {e2}")
            raise


def convert_model(
    model: tf.keras.Model,
    output_dir: str,
    rep_dataset_fn,
):
    """Convert a Keras model to 3 TFLite variants and save them."""
    results = {}

    # ── 1. Dynamic range (weight-only) ──────────────────────────────────
    logger.info("   🔄 Dynamic range (weight-only)...")
    try:
        strategy = DynamicRangeStrategy(debug=False)
        tflite_bytes = strategy.convert(model)
        out_path = os.path.join(output_dir, OUTPUT_NAMES["dynamic_range"])
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)
        size_kb = len(tflite_bytes) / 1024
        logger.info(f"      ✅ {OUTPUT_NAMES['dynamic_range']}  ({size_kb:.1f} KB)")
        results["dynamic_range"] = out_path
    except Exception as e:
        logger.error(f"      ❌ Dynamic range failed: {e}")

    # ── 2. Float16 ──────────────────────────────────────────────────────
    logger.info("   🔄 Float16...")
    try:
        strategy = Float16Strategy(debug=False)
        tflite_bytes = strategy.convert(model)
        out_path = os.path.join(output_dir, OUTPUT_NAMES["float16"])
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)
        size_kb = len(tflite_bytes) / 1024
        logger.info(f"      ✅ {OUTPUT_NAMES['float16']}  ({size_kb:.1f} KB)")
        results["float16"] = out_path
    except Exception as e:
        logger.error(f"      ❌ Float16 failed: {e}")

    # ── 3. Full integer (int8) with representative dataset ──────────────
    logger.info("   🔄 Full integer (int8)...")
    try:
        strategy = FullIntegerStrategy(representative_dataset=rep_dataset_fn, debug=False)
        tflite_bytes = strategy.convert(model)
        out_path = os.path.join(output_dir, OUTPUT_NAMES["full_integer"])
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)
        size_kb = len(tflite_bytes) / 1024
        logger.info(f"      ✅ {OUTPUT_NAMES['full_integer']}  ({size_kb:.1f} KB)")
        results["full_integer"] = out_path
    except Exception as e:
        logger.error(f"      ❌ Full integer failed: {e}")

    return results


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    validate_full_config()

    logger.info("=" * 60)
    logger.info("🔧 Converting v31_extracted.keras → TFLite suite")
    logger.info(f"   Classes: {params.NB_CLASSES}")
    logger.info(f"   Input:   {params.INPUT_SHAPE}")
    logger.info("=" * 60)

    # ── Load calibration data for full int8 ──────────────────────────────
    logger.info("\n📦 Loading calibration data...")
    (x_train, _), _, _ = get_data_splits()
    logger.info(f"   Loaded {len(x_train)} training samples")

    rep_dataset_fn = build_representative_dataset(
        x_train,
        num_samples=200,
        preprocess_fn=preprocess_for_inference,
    )
    logger.info("   ✅ Representative dataset ready (200 samples)")

    # ── Process each model directory ─────────────────────────────────────
    for model_dir_rel in MODEL_DIRS:
        model_dir = os.path.join(_PROJECT_ROOT, model_dir_rel)
        keras_path = os.path.join(model_dir, KERAS_FILENAME)

        if not os.path.isfile(keras_path):
            logger.warning(f"⚠️  Skipping — not found: {keras_path}")
            continue

        logger.info(f"\n{'─' * 60}")
        logger.info(f"📁 {model_dir_rel}")

        # Load model
        try:
            model = load_keras_model(keras_path)
        except Exception as e:
            logger.error(f"   ❌ Skipping — could not load model: {e}")
            continue

        # Convert
        results = convert_model(model, model_dir, rep_dataset_fn)

        # Summary
        logger.info(f"\n   📊 Results for {model_dir_rel}:")
        for variant, path in results.items():
            size_kb = os.path.getsize(path) / 1024
            logger.info(f"      ✅ {variant}: {os.path.basename(path)} ({size_kb:.1f} KB)")

    logger.info(f"\n{'=' * 60}")
    logger.info("✅ All conversions complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
