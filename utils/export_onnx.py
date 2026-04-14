# utils/export_onnx.py
"""
Keras → ONNX export helper for the TQT / ESP-DL quantization pipeline.

Converts a saved Keras model (.keras or SavedModel directory) to an ONNX file
suitable for input to esp-ppq's TQT pass.  Optionally runs onnxsim simplification
and shape inference to produce a cleaner graph.

Usage (standalone):
    python -c "from utils.export_onnx import export_keras_to_onnx; \
               export_keras_to_onnx('checkpoints/best_model.keras', 'model.onnx')"

Typical integration via quantize_espdl.py:
    onnx_path = export_keras_to_onnx(keras_path, onnx_path)
"""

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import tensorflow as tf

import parameters as params

# Optional/Third party imports
try:
    import tf2onnx
except ImportError:
    tf2onnx = None

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxsim
except ImportError:
    onnxsim = None


def export_keras_to_onnx(
    model_or_path: str | object,
    onnx_path: str,
    opset: int = 13,
    simplify: bool = False,
    inputs_as_nchw: bool = True,
    input_name: str = "input_1",
) -> str:
    """
    Convert a saved Keras model or live model object to ONNX via tf2onnx.

    Parameters
    ----------
    model_or_path : str | tf.keras.Model
        Path to a .keras file / SavedModel directory, or a live Keras model object.
    onnx_path : str
        Output path for the .onnx file.
    opset : int
        ONNX opset version (13 recommended for esp-ppq compatibility).
    simplify : bool
        Run onnxsim + shape inference after conversion (recommended).
    inputs_as_nchw : bool
        Transpose the named input from NHWC (TF default) to NCHW (ONNX/esp-ppq).
    input_name : str
        Name of the input tensor to transpose when inputs_as_nchw=True.
    """
    onnx_path = str(Path(onnx_path).resolve())
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    # ── Handle model object vs path ───────────────────────────────────────────
    if isinstance(model_or_path, (str, bytes, os.PathLike)):
        keras_model_path = str(Path(model_or_path).resolve())
        is_keras_file = keras_model_path.endswith('.keras') or keras_model_path.endswith('.h5')
        print(f"🔄 Exporting path {keras_model_path} → {onnx_path}")
        
        if is_keras_file:
            print(f"   (Workaround) Loading Keras 3 model and exporting to SavedModel format for tf2onnx...")
            model = tf.keras.models.load_model(keras_model_path, compile=False)
        else:
            # It's already a SavedModel directory (raw path used by tf2onnx)
            model = None 
            tf2onnx_input = keras_model_path
    else:
        # It's a live model object
        model = model_or_path
        print(f"🔄 Exporting live model object → {onnx_path}")

    actual_input_name = input_name
    
    # ── Keras 3 SavedModel Workaround ─────────────────────────────────────────
    if model is not None:
        
        actual_input_name = model.inputs[0].name.split(':')[0]
        temp_sm_dir = tempfile.mkdtemp(prefix="tf2onnx_sm_")
        
        # Keras 3 export() method writes a SavedModel
        model.export(temp_sm_dir)
        tf2onnx_input = temp_sm_dir
        
        # Cleanup later
        atexit.register(lambda: shutil.rmtree(temp_sm_dir, ignore_errors=True))

    # ── ONNX Export Execution ────────────────────────────────────────────────
    print(f"   Converting to ONNX (opset {opset})...")
    
    try:
        if tf2onnx is None:
            raise ImportError("tf2onnx not found")
        
        if model is not None:
            # ── Case 1: Live Model Object (Direct via Python API) ─────
            # Robust path for Keras 3, improved to handle potential name issues
            input_spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name=actual_input_name),)
            
            model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
                model=model,
                input_signature=input_spec,
                opset=opset,
                output_path=onnx_path,
                inputs_as_nchw=[actual_input_name] if inputs_as_nchw else None
            )
            print(f"✅ ONNX export done (Python API): {onnx_path}")
            
        else:
            # ── Case 2: SavedModel Directory (via subprocess) ──────────
            # Fallback for static paths
            cmd = [
                sys.executable, "-m", "tf2onnx.convert",
                "--saved-model", tf2onnx_input,
                "--output", onnx_path,
                "--opset", str(opset),
            ]
            if inputs_as_nchw:
                cmd += ["--inputs-as-nchw", actual_input_name]

            print(f"   Running subprocess: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"✅ ONNX export done (Subprocess): {onnx_path}")
                
    except ImportError:
        print(f"❌ tf2onnx not found in the current environment ({sys.executable}).")
        print(f"   Please run: pip install tf2onnx")
        return None
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        # traceback.print_exc()
        return None

    # ── Optional simplification ───────────────────────────────────────────────
    if simplify:
        try:
            if onnx is None or onnxsim is None:
                raise ImportError("onnx/onnxsim not found")

            print("🔧 Running onnxsim simplification…")
            
            # Use subprocess to isolate onnxsim from potential Keras/TF library conflicts
            # and force CPU execution to prevent -11 crashes.
            cmd = [sys.executable, "-m", "onnxsim", onnx_path, onnx_path]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            
            try:
                subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
                print("✅ onnxsim simplification done")
                
                # Perform shape inference to ensure clean graph
                model_proto = onnx.load(onnx_path)
                model_proto = onnx.shape_inference.infer_shapes(model_proto)
                onnx.save(model_proto, onnx_path)
            except subprocess.CalledProcessError as e:
                print(f"⚠️  onnxsim subprocess failed (code {e.returncode}): {e.stderr}")
                print("   (Keeping unsimplified ONNX)")
        except ImportError:
            print("⚠️  onnx / onnxsim not installed — skipping simplification")
        except AssertionError as exc:
            print(f"⚠️  onnxsim check failed ({exc}) — keeping unsimplified ONNX")

    return onnx_path


def get_default_onnx_path(model_name: str | None = None) -> str:
    """
    Return the default ONNX output path for a given model name,
    mirroring the project's OUTPUT_DIR convention.
    """
    name = model_name or params.MODEL_ARCHITECTURE
    return os.path.join(params.OUTPUT_DIR, name, f"{name}.onnx")
