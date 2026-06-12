"""
Export TFLite from a .keras model (supports any architecture).

Usage:
    python export_tflite.py --keras path/to/model.keras --model v16 --classes 100 --color rgb
    python export_tflite.py --keras path/to/model.keras --model v33_super_student_10 --classes 10 --color rgb --dry-run
"""
import argparse
import os
import sys
import traceback

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DIGIT_NB_CLASSES", "10")
os.environ.setdefault("DIGIT_INPUT_CHANNELS", "3")

from utils import get_data_splits
from utils.preprocess import preprocess_for_training
from utils.tflite_converter import Float16Strategy, FullIntegerStrategy, build_representative_dataset
from utils.distiller import Distiller, MixedInputDistiller, ProgressiveDistiller, DistillationProgressCallback
from utils.ensemble_teacher import EnsembleTeacher
from models.convnext_blocks import DropPath

import zipfile
import json
import h5py

from models.model_factory import create_model_by_name

CUSTOM_OBJECTS = {
    "DropPath": DropPath,
    "Distiller": Distiller,
    "MixedInputDistiller": MixedInputDistiller,
    "ProgressiveDistiller": ProgressiveDistiller,
    "DistillationProgressCallback": DistillationProgressCallback,
    "EnsembleTeacher": EnsembleTeacher,
}


def is_distiller_wrapper(keras_path: str) -> bool:
    """Check if the .keras file is a Distiller/ProgressiveDistiller wrapper."""
    with zipfile.ZipFile(keras_path) as z:
        cfg = json.loads(z.read("config.json"))
    class_name = cfg.get("class_name", "")
    return "istiller" in class_name


def extract_student_weights(keras_path: str, student_model: tf.keras.Model) -> tf.keras.Model:
    """
    Manually extract student weights from a ProgressiveDistiller .keras file.
    Uses h5py to dig into the H5 weight file.
    """
    with zipfile.ZipFile(keras_path, 'r') as zf:
        with zf.open('model.weights.h5') as f:
            with h5py.File(f, 'r') as h5:
                weight_map = {}
                def collect_weights(name, obj):
                    if not isinstance(obj, h5py.Dataset):
                        return
                    parts = name.split('/')
                    # Student weights are under:
                    # layers/functional/layers/<layer_name>/vars/<idx>
                    # Teacher is under layers/functional/layers/functional/...
                    if (len(parts) >= 5 and
                        parts[0] == 'layers' and
                        parts[1] == 'functional' and
                        parts[2] == 'layers' and
                        parts[4] == 'vars'):
                        # Skip teacher submodel
                        if len(parts) > 6 and parts[3] == 'functional':
                            return
                        layer_name = parts[3]
                        idx = int(parts[5])
                        if layer_name not in weight_map:
                            weight_map[layer_name] = {}
                        weight_map[layer_name][idx] = obj[()]
                h5.visititems(collect_weights)

    # Sort weights by index and apply
    set_count = 0
    for layer in student_model.layers:
        if layer.name in weight_map:
            sorted_w = [weight_map[layer.name][i] for i in sorted(weight_map[layer.name].keys())]
            try:
                layer.set_weights(sorted_w)
                set_count += 1
            except Exception:
                pass
    print(f"   Set weights for {set_count} layers")
    return student_model


def load_model_safe(keras_path: str) -> tf.keras.Model:
    """
    Try loading with Keras first. Falls back to manual extraction
    from the distiller wrapper's H5 file.
    """
    # Try standard Keras load
    try:
        m = tf.keras.models.load_model(keras_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        print(f"   ✅ Loaded via Keras: {m.name} ({m.count_params():,} params)")

        # If distiller, try to extract student
        if hasattr(m, 'layers'):
            candidates = []
            def _search(mod, depth=0):
                if depth > 10:
                    return
                if hasattr(mod, 'layers') and len(mod.layers) > 1:
                    try:
                        if mod.output_shape is not None and len(mod.output_shape) == 2:
                            candidates.append(mod)
                    except (AttributeError, ValueError):
                        pass
                for layer in mod.layers:
                    if hasattr(layer, 'layers') and len(layer.layers) > 1:
                        _search(layer, depth + 1)
            _search(m)
            if candidates:
                candidates.sort(key=lambda c: c.count_params())
                student = candidates[0]
                print(f"   🔍 Extracted student: {student.name} ({student.count_params():,} params)")
                return student
        return m
    except Exception as e:
        print(f"   ⚠️ Keras load failed: {e}")

    # Fallback: manual extraction
    print("   🔧 Falling back to manual weight extraction from H5...")
    output_dir = os.path.dirname(keras_path)
    model_name = os.path.splitext(os.path.basename(keras_path))[0]

    # Check config for model info
    with zipfile.ZipFile(keras_path) as z:
        cfg = json.loads(z.read("config.json"))

    # Build a generic student model
    student = create_model_by_name(
        f"digit_recognizer_v16",
        num_classes=100,
        input_shape=(32, 20, 3),
    )
    student.build((None, 32, 20, 3))
    student = extract_student_weights(keras_path, student)
    print(f"   ✅ Manual extraction complete: {student.name} ({student.count_params():,} params)")
    return student


def main():
    parser = argparse.ArgumentParser(description="Export TFLite from a .keras model")
    parser.add_argument("--keras", required=True, help="Path to .keras file")
    parser.add_argument("--model", default=None, help="Model name for filenames (inferred from filename if not set)")
    parser.add_argument("--classes", type=int, default=100, choices=[10, 100], help="Number of classes")
    parser.add_argument("--color", default="rgb", choices=["gray", "rgb"], help="Color mode")
    parser.add_argument("--dry-run", action="store_true", help="Only load model, don't export")
    args = parser.parse_args()

    os.environ["DIGIT_NB_CLASSES"] = str(args.classes)
    os.environ["DIGIT_INPUT_CHANNELS"] = "3" if args.color == "rgb" else "1"

    import config as params

    keras_path = os.path.abspath(args.keras)
    if not os.path.isfile(keras_path):
        print(f"❌ File not found: {keras_path}")
        sys.exit(1)

    output_dir = os.path.dirname(keras_path)
    model_name = args.model or os.path.splitext(os.path.basename(keras_path))[0]
    model_name = model_name.replace("best_model", "model").replace(" ", "_")

    print(f"📦 Loading {keras_path}...")
    m = load_model_safe(keras_path)

    if args.dry_run:
        print("\n🏁 DRY RUN — stopping. Remove --dry-run to export.")
        return

    # ── Float32 ──────────────────────────────────────────────────────────
    print(f"\n⏳ Float32 TFLite...")
    try:
        c = tf.lite.TFLiteConverter.from_keras_model(m)
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        f32 = c.convert()
        path = os.path.join(output_dir, f"{model_name}_float32.tflite")
        open(path, "wb").write(f32)
        print(f"   ✅ {model_name}_float32.tflite ({len(f32)/1024:.1f} KB)")
    except Exception as e:
        print(f"   ❌ Float32 failed: {e}")

    # ── Float16 ──────────────────────────────────────────────────────────
    print(f"⏳ Float16 TFLite...")
    try:
        f16 = Float16Strategy().convert(m)
        path = os.path.join(output_dir, f"{model_name}_float16.tflite")
        open(path, "wb").write(f16)
        print(f"   ✅ {model_name}_float16.tflite ({len(f16)/1024:.1f} KB)")
    except Exception as e:
        print(f"   ❌ Float16 failed: {e}")

    # ── Int8 ─────────────────────────────────────────────────────────────
    print(f"⏳ Int8 TFLite (calibration may take a moment)...")
    try:
        (x_train, _), _, _ = get_data_splits()
        n = min(200, len(x_train))
        x = preprocess_for_training(x_train[:n]).astype("float32")
        rep_ds = build_representative_dataset(x, n)
        int8 = FullIntegerStrategy(representative_dataset=rep_ds).convert(m)
        path = os.path.join(output_dir, f"{model_name}_full_integer_quant.tflite")
        open(path, "wb").write(int8)
        print(f"   ✅ {model_name}_full_integer_quant.tflite ({len(int8)/1024:.1f} KB)")
    except Exception as e:
        print(f"   ❌ Int8 failed: {e}")
        traceback.print_exc()

    print(f"\n🎉 TFLite files saved to: {output_dir}")


if __name__ == "__main__":
    main()