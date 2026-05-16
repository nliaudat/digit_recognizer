#!/usr/bin/env python3
"""
extract_keras_tqt.py  —  Extract student model from a ProgressiveDistiller .keras
                          file and run the full TQT pipeline (ONNX → TQT → TFLite).

Usage:
    # 10 classes, RGB, v23 student
    python extract_keras_tqt.py --keras path/to/best_model.keras --model v23 --classes 10 --color rgb

    # 100 classes, grayscale, v16 student
    python extract_keras_tqt.py --keras path/to/best_model.keras --model v16 --classes 100 --color gray

    # With custom target hardware
    python extract_keras_tqt.py --keras path/to/best_model.keras --model v23 --classes 10 --color rgb --target esp32s3

The script:
    1. Loads the ProgressiveDistiller .keras with Keras (using custom_objects)
    2. Extracts the student submodel from the distiller
    3. Exports to ONNX
    4. Runs TQT quantization via quantize_espdl.py
    5. Generates TFLite suite
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

# ── Project root setup ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import parameters as params
from config.validation import validate_full_config
from models.model_factory import create_model_by_name, resolve_model_name
from utils import get_data_splits
from utils.preprocess import preprocess_for_training
from utils.export_onnx import export_keras_to_onnx

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract student from ProgressiveDistiller .keras and run TQT pipeline"
    )
    parser.add_argument(
        "--keras", required=True,
        help="Path to best_model.keras (ProgressiveDistiller wrapper)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Student model architecture name (e.g. v23, v16, v30_medium, v31_student_medium)"
    )
    parser.add_argument(
        "--classes", type=int, default=10, choices=[10, 100],
        help="Number of output classes (default: 10)"
    )
    parser.add_argument(
        "--color", default="rgb", choices=["gray", "rgb"],
        help="Color mode (default: rgb)"
    )
    parser.add_argument(
        "--target", default="esp32",
        choices=["esp32", "esp32s3", "esp32c3", "esp32p4"],
        help="Target ESP32 SoC (default: esp32)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same as --keras parent)"
    )
    parser.add_argument(
        "--calib-steps", type=int, default=250,
        help="Number of calibration samples (default: 250)"
    )
    parser.add_argument(
        "--tqt-steps", type=int, default=None,
        help="TQT optimization steps (default: from parameters.py)"
    )
    parser.add_argument(
        "--tqt-lr", type=float, default=None,
        help="TQT learning rate (default: from parameters.py)"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device for TQT calibration (default: cpu)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only extract student and save .keras, don't run TQT"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="TQT worker timeout in seconds (default: 600)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Student extraction from ProgressiveDistiller .keras
# ---------------------------------------------------------------------------

def _import_distiller_classes():
    """Import the custom distiller classes so Keras can load the model."""
    try:
        from utils.distiller import (
            Distiller, MixedInputDistiller, ProgressiveDistiller,
            DistillationProgressCallback
        )
        from utils.ensemble_teacher import EnsembleTeacher
        return {
            'Distiller': Distiller,
            'MixedInputDistiller': MixedInputDistiller,
            'ProgressiveDistiller': ProgressiveDistiller,
            'DistillationProgressCallback': DistillationProgressCallback,
            'EnsembleTeacher': EnsembleTeacher,
        }
    except ImportError as e:
        logger.warning(f"Could not import distiller classes: {e}")
        return {}


def extract_student_from_distiller(keras_path: str) -> tf.keras.Model:
    """
    Load the ProgressiveDistiller .keras and extract the student submodel.

    Strategy:
    1. Load the full distiller model with Keras (using custom_objects)
    2. The student is stored as a submodel — find it by looking for
       the model that has the expected student architecture (small, with
       a softmax output of NB_CLASSES)
    3. Return the student model with its trained weights
    """
    logger.info(f"📦 Loading ProgressiveDistiller from: {keras_path}")

    # ── Step 1: Load the full distiller ──────────────────────────────────
    custom_objects = _import_distiller_classes()
    
    # Read config.json to understand the model structure
    with zipfile.ZipFile(keras_path, 'r') as zf:
        config = json.loads(zf.read('config.json'))
    
    class_name = config.get('class_name', '')
    logger.info(f"   Detected model type: {class_name}")

    # Load the full model
    try:
        full_model = tf.keras.models.load_model(
            keras_path,
            custom_objects=custom_objects,
            compile=False,
        )
        logger.info(f"   ✅ Loaded full model: {full_model.name}")
    except Exception as e:
        logger.error(f"   ❌ Failed to load model: {e}")
        logger.info("   Trying manual weight extraction as fallback...")
        return _extract_student_manual(keras_path)

    # ── Step 2: Find the student submodel ────────────────────────────────
    student = _find_student_submodel(full_model)
    
    if student is None:
        logger.warning("   Could not find student submodel via Keras API.")
        logger.info("   Falling back to manual weight extraction...")
        return _extract_student_manual(keras_path)

    logger.info(f"   ✅ Found student submodel: {student.name}")
    logger.info(f"      Layers: {len(student.layers)}")
    logger.info(f"      Params: {student.count_params():,}")
    
    # Verify with forward pass
    dummy = tf.zeros((1,) + tuple(student.input_shape[1:]))
    out = student(dummy, training=False)
    logger.info(f"      Output shape: {out.shape}")
    
    return student


def _find_student_submodel(model: tf.keras.Model) -> tf.keras.Model:
    """
    Recursively search for the student submodel inside a distiller wrapper.

    The student is typically the smaller model (fewer params) that has
    a softmax output. The teacher is the larger model.
    """
    candidates = []
    
    def _search(m, depth=0):
        if depth > 10:  # Safety limit
            return
        # Check if this model looks like a student
        if hasattr(m, 'layers') and len(m.layers) > 1:
            # Check output: should be softmax with NB_CLASSES
            try:
                out_shape = m.output_shape
                if out_shape is not None and len(out_shape) == 2:
                    candidates.append(m)
            except (AttributeError, ValueError):
                pass
        
        # Recurse into submodels
        for layer in m.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 1:
                _search(layer, depth + 1)
    
    _search(model)
    
    if not candidates:
        return None
    
    # The student is the one with the smallest param count
    # (teacher is always larger)
    candidates.sort(key=lambda m: m.count_params())
    
    # The smallest model is likely the student
    student = candidates[0]
    
    # Verify it has a softmax output with the right number of classes
    try:
        out = student.output
        if out.shape[-1] == params.NB_CLASSES:
            logger.info(f"   Student candidate: {student.name} ({student.count_params():,} params)")
            return student
    except (AttributeError, ValueError):
        pass
    
    # If verification fails, try the next smallest
    for m in candidates:
        try:
            if m.output_shape[-1] == params.NB_CLASSES:
                logger.info(f"   Student candidate: {m.name} ({m.count_params():,} params)")
                return m
        except (AttributeError, ValueError):
            continue
    
    return candidates[0]  # Best guess


def _extract_student_manual(keras_path: str) -> tf.keras.Model:
    """
    Manual fallback: extract student weights from the H5 file and build
    a fresh student model.

    The student weights are under `layers/functional/` in the H5 file.
    The teacher is nested under `layers/functional/layers/functional/`.
    We only take weights from `layers/functional/layers/` that are NOT
    under `layers/functional/layers/functional/`.
    """
    logger.info("   Manual extraction: reading weights from H5...")
    
    weight_map = {}
    with zipfile.ZipFile(keras_path, 'r') as zf:
        with zf.open('model.weights.h5') as f:
            with h5py.File(f, 'r') as h5:
                def collect_weights(name, obj):
                    if not isinstance(obj, h5py.Dataset):
                        return
                    parts = name.split('/')
                    # We want: layers/functional/layers/<layer_name>/vars/<idx>
                    # But NOT: layers/functional/layers/functional/... (teacher)
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
    
    # Sort by index
    result = {}
    for layer_name, var_dict in weight_map.items():
        sorted_indices = sorted(var_dict.keys())
        result[layer_name] = [var_dict[i] for i in sorted_indices]
    
    logger.info(f"   Found {len(result)} student layers with weights")
    for layer_name, weights in result.items():
        shapes = [str(w.shape) for w in weights]
        logger.info(f"     {layer_name}: {', '.join(shapes)}")
    
    return result  # Return weight map for later use


def build_student_and_set_weights(
    model_name: str,
    num_classes: int,
    color_mode: str,
    weight_map: dict,
) -> tf.keras.Model:
    """
    Build a fresh student model and set its weights from the extracted map.
    """
    channels = 1 if color_mode == "gray" else 3
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
    
    logger.info(f"Building student model: {model_name}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Classes: {num_classes}")
    
    student = create_model_by_name(
        model_name,
        num_classes=num_classes,
        input_shape=input_shape,
    )
    
    # Build the model to initialize weights
    student.build((None,) + input_shape)
    
    # Set weights layer by layer
    set_count = 0
    skip_count = 0
    for layer in student.layers:
        if layer.name in weight_map:
            try:
                layer.set_weights(weight_map[layer.name])
                set_count += 1
            except Exception as e:
                logger.warning(f"  ⚠️ Could not set weights for layer '{layer.name}': {e}")
                skip_count += 1
        else:
            if layer.weights:
                logger.debug(f"  ℹ️ No weights found for layer '{layer.name}' in checkpoint")
                skip_count += 1
    
    logger.info(f"  ✅ Set weights for {set_count} layers ({skip_count} skipped)")
    
    # Verify with a forward pass
    dummy = tf.zeros((1,) + input_shape)
    out = student(dummy, training=False)
    logger.info(f"  ✅ Forward pass OK — output shape: {out.shape}")
    
    return student


# ---------------------------------------------------------------------------
# Calibration data loading
# ---------------------------------------------------------------------------

def load_calibration_data(num_classes: int, color_mode: str, calib_steps: int):
    """Load calibration data for TQT."""
    logger.info(f"📦 Loading calibration data ({num_classes} classes, {color_mode})...")
    
    # Set params for data loading
    params.NB_CLASSES = num_classes
    params.INPUT_CHANNELS = 1 if color_mode == "gray" else 3
    params.update_derived_parameters()
    
    (x_train, _), _, _ = get_data_splits()
    n = min(calib_steps, len(x_train))
    
    # Use the same progressive calibration as quantize_espdl
    from quantize_espdl import get_progressive_calibration_data
    x = get_progressive_calibration_data(x_train, n)
    x = preprocess_for_training(x).astype('float32')
    
    import torch
    calib_data = [torch.from_numpy(x[i].transpose(2, 0, 1)).unsqueeze(0) for i in range(n)]
    
    logger.info(f"✅ Loaded {len(calib_data)} calibration samples")
    return calib_data


# ---------------------------------------------------------------------------
# TQT pipeline
# ---------------------------------------------------------------------------

def run_tqt_pipeline(
    student: tf.keras.Model,
    model_name: str,
    output_dir: str,
    calib_data: list,
    target: str,
    num_classes: int,
    color_mode: str,
    tqt_steps: int = None,
    tqt_lr: float = None,
    device: str = "cpu",
    timeout: int = 600,
):
    """Run ONNX export → TQT quantization → TFLite suite."""
    from quantize_espdl import tflite_suite_export, organize_output_folder
    
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    espdl_path = os.path.join(output_dir, f"{model_name}_{target}.espdl")
    
    # ── Step 1: ONNX export ──────────────────────────────────────────────
    logger.info(f"\n🔄 Step 1/3: Exporting student → ONNX")
    result = export_keras_to_onnx(student, onnx_path, simplify=True)
    if result is None or not os.path.exists(onnx_path):
        logger.error("❌ ONNX export failed")
        return False
    
    # ── Step 2: TQT quantization ─────────────────────────────────────────
    logger.info(f"\n🔄 Step 2/3: Running TQT quantization → {os.path.basename(espdl_path)}")
    
    color_channels = 1 if color_mode == "gray" else 3
    input_height = params.INPUT_HEIGHT
    input_width = params.INPUT_WIDTH
    
    # Prepare calibration data as NCHW numpy array
    import torch
    calib_concat = torch.cat(calib_data, dim=0).numpy()
    calib_npy_path = os.path.join(output_dir, "calib_data_nchw.npy")
    np.save(calib_npy_path, calib_concat)
    
    # Get TQT config from params
    tqt_cfg = params._TQT_DEFAULTS.get(target, params._TQT_DEFAULTS.get("esp32"))
    actual_tqt_steps = tqt_steps or tqt_cfg.get("TQT_STEPS", 200)
    actual_tqt_lr = tqt_lr or tqt_cfg.get("TQT_LR", 1e-6)
    
    logger.info(f"   TQT steps: {actual_tqt_steps}, LR: {actual_tqt_lr}, timeout: {timeout}s")
    
    # Write worker script
    worker_script = os.path.join(output_dir, "_tqt_worker.py")
    worker_script_content = f'''#!/usr/bin/env python3
import sys, os, torch, numpy as np, onnx.helper, onnx.mapping
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import QuantizationSettingFactory
import faulthandler

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
faulthandler.enable()

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Compatibility wrapper for onnx helper
class OnnxCompatibilityWrapper:
    def __init__(self, func, mapping):
        self.func = func
        self.mapping = mapping
    def __getitem__(self, key): return self.mapping[key]
    def __call__(self, *args, **kwargs): return self.func(*args, **kwargs)
    def __iter__(self): return iter(self.mapping)
    def __len__(self): return len(self.mapping)
    def items(self): return self.mapping.items()
    def keys(self): return self.mapping.keys()
    def values(self): return self.mapping.values()

if callable(onnx.helper.tensor_dtype_to_np_dtype):
    onnx.helper.tensor_dtype_to_np_dtype = OnnxCompatibilityWrapper(
        onnx.helper.tensor_dtype_to_np_dtype,
        onnx.mapping.TENSOR_TYPE_TO_NP_TYPE
    )

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

quant_setting = QuantizationSettingFactory.espdl_setting()
quant_setting.tqt_optimization = True
if quant_setting.tqt_optimization:
    s = quant_setting.tqt_optimization_setting
    s.steps = {actual_tqt_steps}
    s.lr = {actual_tqt_lr}
    s.collecting_device = "{device}"
    s.int_lambda = {tqt_cfg.get("TQT_INT_LAMBDA", 0.1)}
    s.block_size = {tqt_cfg.get("TQT_BLOCK_SIZE", 2)}
    s.gamma = 0.01
    s.is_scale_trainable = True
    s.is_weight_trainable = False

calib_np = np.load(r"{os.path.abspath(calib_npy_path)}")
calib_data = [torch.from_numpy(calib_np[i:i+1]) for i in range(calib_np.shape[0])]

print("[Worker] Starting espdl_quantize_onnx...")
graph = espdl_quantize_onnx(
    onnx_import_file=r"{os.path.abspath(onnx_path)}",
    espdl_export_file=r"{os.path.abspath(espdl_path)}",
    calib_dataloader=calib_data,
    calib_steps={len(calib_data)},
    input_shape=[1, {color_channels}, {input_height}, {input_width}],
    target="{target}",
    num_of_bits=8,
    setting=quant_setting,
    device="{device}"
)

from esp_ppq.api.interface import export_ppq_graph
from esp_ppq import TargetPlatform
quant_onnx_path = r"{os.path.abspath(espdl_path)}".replace(".espdl", "_quantized.onnx")
print(f"[Worker] Exporting Scale-Preserving ONNX -> {{quant_onnx_path}}")
export_ppq_graph(graph, platform=TargetPlatform.ONNXRUNTIME, graph_save_to=quant_onnx_path)
print("[Worker] ESP-DL Quantization finished successfully!")
'''
    with open(worker_script, 'w', encoding='utf-8') as f:
        f.write(worker_script_content)
    
    logger.info("   Spawning isolated TQT worker...")
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.check_call([sys.executable, worker_script], env=env, timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            logger.warning(f"   ⚠️ TQT worker timed out (attempt {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                logger.error(f"❌ TQT worker timed out after {max_retries} attempts ({timeout}s each)")
                return False
        except subprocess.CalledProcessError as e:
            logger.warning(f"   ⚠️ TQT worker failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"❌ TQT worker failed after {max_retries} attempts")
                return False
            time.sleep(2)
    
    # Cleanup temp files
    for tmp in [calib_npy_path, worker_script]:
        if os.path.exists(tmp):
            os.remove(tmp)
    
    if not os.path.exists(espdl_path):
        logger.error(f"❌ TQT quantization failed: {espdl_path} not found")
        return False
    
    # ── Step 3: TFLite suite ─────────────────────────────────────────────
    logger.info(f"\n🔄 Step 3/3: Generating TFLite suite...")
    
    class MiniArgs:
        tflite = True
    mini_args = MiniArgs()
    mini_args.target = target
    
    tflite_suite_export(onnx_path, calib_data, mini_args, espdl_path)
    
    # Organize output
    logger.info(f"\n🧹 Organizing output folder...")
    organize_output_folder(output_dir)
    
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_full_config()
    
    keras_path = os.path.abspath(args.keras)
    if not os.path.isfile(keras_path):
        logger.error(f"❌ Keras model not found: {keras_path}")
        sys.exit(1)
    
    # Set params
    params.NB_CLASSES = args.classes
    params.INPUT_CHANNELS = 1 if args.color == "gray" else 3
    params.update_derived_parameters()
    
    # Output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.dirname(keras_path)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info(f"🔧 Extracting student from ProgressiveDistiller")
    logger.info(f"   Keras:  {keras_path}")
    logger.info(f"   Model:  {args.model}")
    logger.info(f"   Classes: {args.classes}")
    logger.info(f"   Color:  {args.color.upper()}")
    logger.info(f"   Target: {args.target}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Timeout: {args.timeout}s")
    logger.info("=" * 60)
    
    # ── Step 1: Extract student from distiller ───────────────────────────
    logger.info(f"\n📦 Extracting student from ProgressiveDistiller...")
    student = extract_student_from_distiller(keras_path)
    
    # If manual extraction returned a weight map (dict), build the model
    if isinstance(student, dict):
        weight_map = student
        student = build_student_and_set_weights(
            model_name=args.model,
            num_classes=args.classes,
            color_mode=args.color,
            weight_map=weight_map,
        )
    
    # Save the standalone student .keras
    student_keras_path = os.path.join(output_dir, f"{args.model}_extracted.keras")
    student.save(student_keras_path)
    logger.info(f"✅ Standalone student saved → {student_keras_path}")
    
    if args.dry_run:
        logger.info(f"\n🏁 DRY RUN — stopping after extraction")
        logger.info(f"   Run without --dry-run to continue with TQT pipeline")
        return
    
    # ── Step 2: Load calibration data ────────────────────────────────────
    calib_data = load_calibration_data(
        num_classes=args.classes,
        color_mode=args.color,
        calib_steps=args.calib_steps,
    )
    
    # ── Step 3: Run TQT pipeline ─────────────────────────────────────────
    success = run_tqt_pipeline(
        student=student,
        model_name=args.model,
        output_dir=output_dir,
        calib_data=calib_data,
        target=args.target,
        num_classes=args.classes,
        color_mode=args.color,
        tqt_steps=args.tqt_steps,
        tqt_lr=args.tqt_lr,
        device=args.device,
        timeout=args.timeout,
    )
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ FULL PIPELINE COMPLETE")
        logger.info(f"   Output: {output_dir}")
        logger.info("=" * 60)
    else:
        logger.error("\n❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
