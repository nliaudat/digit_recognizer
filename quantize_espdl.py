#!/usr/bin/env python3
import sys
import os

# === DOCKER SEGFAULT PREVENTION ===
# Must be set BEFORE any torch/numpy imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Prevent TorchInductor /tmp issues
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/pytorch_inductor_cache"
os.makedirs("/tmp/pytorch_inductor_cache", exist_ok=True)

# Use spawn instead of fork for multiprocessing
os.environ["PYTORCH_MULTIPROCESSING_START_METHOD"] = "spawn"

import argparse
import numpy as np
import torch
from pathlib import Path

# Stability: Limit high-parallelism threading which can cause -11 segfaults in some environments
# We MUST do this before any other torch operations
torch.set_num_threads(1)
torch.set_num_interop_threads(1)  # Also limit interop threads

def check_threading_safety():
    """Verify threading config for Docker safety"""
    print("🧵 Threading Configuration:")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'not set')}")
    print(f"  torch.get_num_threads()={torch.get_num_threads()}")
    print(f"  torch.get_num_interop_threads()={torch.get_num_interop_threads()}")

check_threading_safety()

def parse_args():
    parser = argparse.ArgumentParser(description="TQT/ESP-DL Minimal Worker")
    parser.add_argument("--model",    help="Model name to resolve paths automatically")
    parser.add_argument("--keras",    help="Path to .keras model")
    parser.add_argument("--onnx",     help="Path to .onnx model")
    parser.add_argument("--output",   help="Explicit output .espdl path")
    parser.add_argument("--target",   default="esp32", help="Target chip (esp32, esp32s3, etc.)")
    parser.add_argument("--bits",     type=int,   default=8)
    parser.add_argument("--steps",    type=int,   default=300)
    parser.add_argument("--lr",       type=float, default=1e-6)
    parser.add_argument("--tune_weights", action="store_true", help="Fine-tune weights during TQT (default is scale-only)")
    def detect_device():
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    parser.add_argument("--device",   default=detect_device())
    parser.add_argument("--classes",  type=int,   default=10)
    parser.add_argument("--color",    choices=["gray", "rgb"], default="rgb")
    parser.add_argument("--calib_steps", type=int, default=32)
    parser.add_argument("--no_simplify", action="store_true")
    parser.add_argument("--skip_onnx_export", action="store_true")
    parser.add_argument("--tflite", action="store_true", help="Also export TFLite via onnx2tf using TQT scales")
    
    return parser.parse_args()

def resolve_paths(args):
    # Delayed import to avoid top-level conflict
    import parameters as params
    
    model_name = args.model or params.MODEL_ARCHITECTURE
    out_dir = os.path.join(params.OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    keras_path = args.keras  or os.path.join(out_dir, "best_model.keras")
    onnx_path  = args.onnx   or os.path.join(out_dir, f"{model_name}.onnx")
    
    if args.output and os.path.isdir(args.output):
        espdl_path = os.path.join(args.output, f"{model_name}_{args.target}.espdl")
    else:
        espdl_path = args.output or os.path.join(out_dir, f"{model_name}_{args.target}.espdl")

    return keras_path, onnx_path, espdl_path

def main():
    args = parse_args()

    # Pre-set environments for project imports
    os.environ["DIGIT_NB_CLASSES"] = str(args.classes)
    os.environ["DIGIT_INPUT_CHANNELS"] = "1" if args.color == "gray" else "3"

    # Delayed imports
    import parameters as params
    from esp_ppq.api import espdl_quantize_onnx, export_ppq_graph
    from esp_ppq import QuantizationSettingFactory, TargetPlatform
    from utils import get_data_splits
    from utils.preprocess import preprocess_for_inference
    from utils.export_onnx import export_keras_to_onnx

    keras_path, onnx_path, espdl_path = resolve_paths(args)

    # 1. Export ONNX if needed
    if not args.skip_onnx_export:
        print(f"🔄 Exporting {keras_path} -> {onnx_path}")
        export_keras_to_onnx(keras_path, onnx_path, simplify=not args.no_simplify)

    # 2. Prepare data
    print(f"📦 Loading calibration data ({args.classes} classes, {args.color})...")
    from utils.preprocess import preprocess_for_training
    (x_train, _), _, _ = get_data_splits()
    n = min(args.calib_steps, len(x_train))
    
    # CRITICAL: Use preprocess_for_training to ensure Float32 [0, 1] 
    # TQT needs calibration data in the same range as training data
    x = preprocess_for_training(x_train[:n]).astype('float32')
    
    # Add debug code right before TQT
    print("🔍 Debug: Checking calibration data statistics")
    sample = x[0]
    print(f"  Shape: {sample.shape}")
    print(f"  Value range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Mean: {x.mean():.3f}, Std: {x.std():.3f}")

    # Compare with training data distribution (informational)
    print(f"  Training sample range: [{x.min():.3f}, {x.max():.3f}]")

    # NHWC -> NCHW list
    calib_data = [torch.from_numpy(x[i].transpose(2, 0, 1)).unsqueeze(0) for i in range(n)]

    # 3. Target mapping
    target_backend = args.target # Use literal target name (e.g., 'esp32', 'esp32s3')

    # 4. Configure TQT with target-specific settings from parameters.py
    # This allows the worker to pick up the correct LR/Steps for S3 vs Generic vs P4
    tqt_cfg = params._TQT_DEFAULTS.get(args.target, params._TQT_DEFAULTS.get("esp32"))
    
    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.tqt_optimization = (args.steps > 0)
    if quant_setting.tqt_optimization:
        s = quant_setting.tqt_optimization_setting
        
        # Priority: CLI argument > target-specific default > global default
        s.steps = args.steps if args.steps != 300 else tqt_cfg.get("TQT_STEPS", args.steps)
        s.lr = args.lr if args.lr != 1e-6 else tqt_cfg.get("TQT_LR", args.lr)
        s.collecting_device = args.device
        
        s.int_lambda = tqt_cfg.get("TQT_INT_LAMBDA", 0.1)
        s.block_size = tqt_cfg.get("TQT_BLOCK_SIZE", 2)
        s.gamma = 0.01             
        
        s.is_scale_trainable = True
        s.is_weight_trainable = args.tune_weights
        
        if args.tune_weights:
             print("🧠 TQT: Tuning BOTH scales and weights (High risk of overfitting calibration set)")
        else:
             print("⚖️ TQT: Tuning SCALES ONLY (Safer for small calibration sets)")

    # 5. Run Quantization
    print(f"🚀 Starting TQT: {onnx_path} -> {espdl_path}")
    print(f"   target={target_backend} bits={args.bits} steps={args.steps} device={args.device}")
    
    try:
        # 5. Run Quantization
        # Defensive check: Can we load and check the ONNX graph?
        import onnx
        import gc
        gc.collect() # Free memory before heavy Torch lifting
        
        try:
            _m = onnx.load(onnx_path)
            onnx.checker.check_model(_m)
            print("✅ ONNX graph check passed")
        except Exception as e:
            print(f"⚠️ ONNX graph check failed: {e}")

        # --- ISOLATION WORKAROUND FOR TENSORFLOW / ONNXSIM SEGFAULT ---
        # Instead of calling espdl_quantize_onnx in the same process where TensorFlow
        # is loaded (which causes an onnxsim segmentation fault), we write the arguments
        # to a temporary worker script and execute it in a clean subprocess.
        import subprocess
        import tempfile
        import pickle
        import numpy as np
        
        calib_npy_path = os.path.join(os.path.dirname(espdl_path), "calib_data_nchw.npy")
        # calib_data is a list of NCHW tensors. Save them as a single concatenated numpy array
        calib_concat = torch.cat(calib_data, dim=0).numpy()
        np.save(calib_npy_path, calib_concat)
        
        worker_script = os.path.join(os.path.dirname(espdl_path), "_tqt_worker.py")
        color_channels = 1 if args.color == "gray" else 3
        with open(worker_script, 'w', encoding='utf-8') as f:
            f.write(f'''#!/usr/bin/env python3
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
import numpy as np
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import QuantizationSettingFactory
import faulthandler
faulthandler.enable()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

quant_setting = QuantizationSettingFactory.espdl_setting()
quant_setting.tqt_optimization = {quant_setting.tqt_optimization}
if quant_setting.tqt_optimization:
    s = quant_setting.tqt_optimization_setting
    s.steps = {s.steps if quant_setting.tqt_optimization else 0}
    s.lr = {s.lr if quant_setting.tqt_optimization else 0.0}
    s.collecting_device = "{args.device}"
    s.int_lambda = {s.int_lambda if quant_setting.tqt_optimization else 0.0}
    s.block_size = {s.block_size if quant_setting.tqt_optimization else 0}
    s.gamma = {s.gamma if quant_setting.tqt_optimization else 0.0}
    s.is_scale_trainable = True
    s.is_weight_trainable = {args.tune_weights}

calib_np = np.load("{calib_npy_path}")
calib_data = [torch.from_numpy(calib_np[i:i+1]) for i in range(calib_np.shape[0])]

print("🚀 [Worker] Starting espdl_quantize_onnx...")
graph = espdl_quantize_onnx(
    onnx_import_file="{onnx_path}",
    espdl_export_file="{espdl_path}",
    calib_dataloader=calib_data,
    calib_steps={n},
    input_shape=[1, {color_channels}, 32, 20],
    target="{target_backend}",
    num_of_bits={args.bits},
    setting=quant_setting,
    device="{args.device}"
)

# Export scale-preserving ONNX directly in worker
from esp_ppq.api.interface import export_ppq_graph
from esp_ppq import TargetPlatform
quant_onnx_path = "{espdl_path}".replace(".espdl", "_quantized.onnx")
print(f"🔄 [Worker] Exporting Scale-Preserving ONNX -> {{quant_onnx_path}}")
export_ppq_graph(graph, platform=TargetPlatform.ONNXRUNTIME, graph_save_to=quant_onnx_path)
print("✅ [Worker] ESP-DL Quantization finished successfully!")
''')

        print("🔄 Spawning isolated TQT worker to prevent onnxsim segfault...")
        try:
            subprocess.check_call([sys.executable, worker_script])
        except subprocess.CalledProcessError as e:
            print(f"❌ Worker failed with exit code {e.returncode}")
            sys.exit(1)
        finally:
            if os.path.exists(calib_npy_path): os.remove(calib_npy_path)
            if os.path.exists(worker_script): os.remove(worker_script)
            
        print("✅ ESP-DL Quantization finished successfully in worker!")

        # 6. Export Quantized ONNX (Scale-Preserving)
        # (This is now handled inside the isolated worker script)
        quant_onnx_path = espdl_path.replace(".espdl", "_quantized.onnx")
        
        # 7. Optional TFLite Export via onnx2tf
        if args.tflite:
            tflite_path = espdl_path.replace(".espdl", ".tflite")
            output_dir = os.path.dirname(tflite_path)
            try:
                import onnx2tf
                import numpy as np
                import io
                
                # 1. Monkeypatch numpy.load (Pickle Fix)
                orig_load = np.load
                def patched_load(f, *args, **kwargs):
                    try:
                        if 'allow_pickle' not in kwargs: kwargs['allow_pickle'] = True
                        return orig_load(f, *args, **kwargs)
                    except:
                        if isinstance(f, (io.BytesIO, io.BufferedReader)):
                             return np.zeros((1, 3, 32, 20), dtype=np.float32)
                        raise
                np.load = patched_load
                
                # 2. Prepare Calibration data for onnx2tf (Avoids download/pickle/shape errors)
                calib_npy_path = os.path.join(output_dir, "onnx2tf_calib_temp.npy")
                # Prepare a small subset of real calib data in NHWC (as expected by onnx2tf for broadcast)
                # calib_data is a list of [1, 3, 32, 20]
                onnx2tf_calib = []
                for i in range(min(len(calib_data), 100)):
                    sample = calib_data[i].numpy().transpose(0, 2, 3, 1) # NCHW -> NHWC
                    onnx2tf_calib.append(sample)
                np.save(calib_npy_path, np.concatenate(onnx2tf_calib, axis=0))
                
                print(f"   (Converting Scale-Preserving ONNX -> TFLite INT8)")
                onnx2tf.convert(
                    input_onnx_file_path=quant_onnx_path,
                    output_folder_path=output_dir,
                    # Pass the custom NHWC calibration data
                    custom_input_op_name_np_data_path=[["input", calib_npy_path, [0,0,0], [1,1,1]]],
                    output_integer_quantized_tflite=True,
                    not_use_onnxsim=True,
                    non_verbose=True
                )
                
                # Restore original numpy.load
                np.load = orig_load
                
                # Cleanup temp calib
                if os.path.exists(calib_npy_path): os.remove(calib_npy_path)
                
                # Find and rename result
                found_tflite = None
                search_paths = [
                    os.path.join(output_dir, "model_integer_only.tflite"),
                    os.path.join(output_dir, "saved_model", "model_integer_only.tflite"),
                    os.path.join(output_dir, "model.tflite"),
                ]
                
                for p in search_paths:
                    if os.path.exists(p):
                        found_tflite = p
                        break
                        
                if not found_tflite:
                    for root, _, files in os.walk(output_dir):
                        for f in files:
                            if f.endswith(".tflite") and "quant" not in f and "float" not in f:
                                found_tflite = os.path.join(root, f)
                                break
                        if found_tflite: break

                if found_tflite:
                    import shutil
                    if os.path.abspath(found_tflite) != os.path.abspath(tflite_path):
                        shutil.move(found_tflite, tflite_path)
                    print(f"✅ Scale-Preserved TFLite export complete: {tflite_path}")
                else:
                    print(f"⚠️ onnx2tf finished but no .tflite file was found in {output_dir}")

            except Exception as e:
                print(f"⚠️ Scale-Preserved TFLite conversion error: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"❌ espdl_quantize_onnx raised an exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if os.path.exists(espdl_path):
        size_kb = os.path.getsize(espdl_path) / 1024
        print(f"✅ Export complete: {espdl_path} ({size_kb:.1f} KB)")
    else:
        print(f"❌ Export FAILED: {espdl_path} not found")
        sys.exit(1)

if __name__ == "__main__":
    main()
