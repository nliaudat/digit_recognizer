#!/usr/bin/env python3
import sys
import os

# Set defaults BEFORE any imports that might depend on them
os.environ.setdefault("DIGIT_NB_CLASSES", "10")
os.environ.setdefault("DIGIT_INPUT_CHANNELS", "3")

import argparse
import numpy as np
import torch
import shutil
import subprocess
import io
import gc
import onnx
import onnx2tf
import faulthandler
from pathlib import Path

# Project-specific imports
import parameters as params
from utils import get_data_splits
from utils.preprocess import preprocess_for_training
from utils.export_onnx import export_keras_to_onnx

# Force UTF-8 output on Windows to support emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# === MONKEYPATCH NP.LOAD FOR ONNX2TF PICKLE SAFETY ===
orig_load = np.load
def patched_load(f, *args, **kwargs):
    try:
        if 'allow_pickle' not in kwargs:
            kwargs['allow_pickle'] = True
        return orig_load(f, *args, **kwargs)
    except Exception:
        if isinstance(f, (io.BytesIO, io.BufferedReader)):
             return np.zeros((1, 3, 32, 20), dtype=np.float32)
        raise
np.load = patched_load

# === DOCKER SEGFAULT PREVENTION ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/pytorch_inductor_cache"
os.makedirs("/tmp/pytorch_inductor_cache", exist_ok=True)
os.environ["PYTORCH_MULTIPROCESSING_START_METHOD"] = "spawn"

# Stability
torch.set_num_threads(1)
torch.set_num_interop_threads(1) 

def check_threading_safety():
    """Verify threading config for Docker safety"""
    print("🧵 Thread Configuration:")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'not set')}")
    print(f"  torch.get_num_threads()={torch.get_num_threads()}")
    print(f"  torch.get_num_interop_threads()={torch.get_num_interop_threads()}")

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
    parser.add_argument("--tune_weights", action="store_true")
    
    def detect_device():
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"

    parser.add_argument("--device",   default=detect_device())
    parser.add_argument("--classes",  type=int,   default=10)
    parser.add_argument("--color",    choices=["gray", "rgb"], default="rgb")
    parser.add_argument("--calib_steps", type=int, default=250)
    parser.add_argument("--no_simplify", action="store_true")
    parser.add_argument("--skip_onnx_export", action="store_true")
    parser.add_argument("--tflite", action="store_true")
    parser.add_argument("--export_all_variants", action="store_true", default=True)
    parser.add_argument("--organize", action="store_true", default=True)
    
    return parser.parse_args()

def resolve_paths(args):
    model_name = args.model or params.MODEL_ARCHITECTURE
    
    if args.output:
        if os.path.isdir(args.output):
            out_dir = args.output
        elif "." in os.path.basename(args.output):
            out_dir = os.path.dirname(args.output) or "."
        else:
            out_dir = args.output
            os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.join(params.OUTPUT_DIR, model_name)
        os.makedirs(out_dir, exist_ok=True)

    keras_path = args.keras  or os.path.join(out_dir, "best_model.keras")
    onnx_path  = args.onnx   or os.path.join(out_dir, f"{model_name}.onnx")
    
    if args.output and not os.path.isdir(args.output):
        espdl_path = args.output
    else:
        espdl_path = os.path.join(out_dir, f"{model_name}_{args.target}.espdl")

    return keras_path, onnx_path, espdl_path

def main():
    check_threading_safety()
    args = parse_args()
    
    os.environ["DIGIT_NB_CLASSES"] = str(args.classes)
    os.environ["DIGIT_INPUT_CHANNELS"] = "1" if args.color == "gray" else "3"

    keras_path, onnx_path, espdl_path = resolve_paths(args)

    if not args.skip_onnx_export:
        print(f"🔄 Exporting {keras_path} -> {onnx_path}")
        export_keras_to_onnx(keras_path, onnx_path, simplify=not args.no_simplify)

    print(f"📦 Loading calibration data ({args.classes} classes, {args.color})...")
    (x_train, _), _, _ = get_data_splits()
    n = min(args.calib_steps, len(x_train))
    x = preprocess_for_training(x_train[:n]).astype('float32')
    calib_data = [torch.from_numpy(x[i].transpose(2, 0, 1)).unsqueeze(0) for i in range(n)]

    target_backend = args.target 
    tqt_cfg = params._TQT_DEFAULTS.get(args.target, params._TQT_DEFAULTS.get("esp32"))
    
    try:
        calib_npy_path = os.path.join(os.path.dirname(espdl_path), "calib_data_nchw.npy")
        calib_concat = torch.cat(calib_data, dim=0).numpy()
        np.save(calib_npy_path, calib_concat)
        
        worker_script = os.path.join(os.path.dirname(espdl_path), "_tqt_worker.py")
        color_channels = 1 if args.color == "gray" else 3
        
        worker_script_content = f'''#!/usr/bin/env python3
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
quant_setting.tqt_optimization = True
if quant_setting.tqt_optimization:
    s = quant_setting.tqt_optimization_setting
    s.steps = {args.steps if args.steps != 300 else tqt_cfg.get("TQT_STEPS", args.steps)}
    s.lr = {args.lr if args.lr != 1e-6 else tqt_cfg.get("TQT_LR", args.lr)}
    s.collecting_device = "{args.device}"
    s.int_lambda = {tqt_cfg.get("TQT_INT_LAMBDA", 0.1)}
    s.block_size = {tqt_cfg.get("TQT_BLOCK_SIZE", 2)}
    s.gamma = 0.01
    s.is_scale_trainable = True
    s.is_weight_trainable = {args.tune_weights}

calib_np = np.load(r"{os.path.abspath(calib_npy_path)}")
calib_data = [torch.from_numpy(calib_np[i:i+1]) for i in range(calib_np.shape[0])]

print("🚀 [Worker] Starting espdl_quantize_onnx...")
graph = espdl_quantize_onnx(
    onnx_import_file=r"{os.path.abspath(onnx_path)}",
    espdl_export_file=r"{os.path.abspath(espdl_path)}",
    calib_dataloader=calib_data,
    calib_steps={n},
    input_shape=[1, {color_channels}, 32, 20],
    target="{target_backend}",
    num_of_bits={args.bits},
    setting=quant_setting,
    device="{args.device}"
)

from esp_ppq.api.interface import export_ppq_graph
from esp_ppq import TargetPlatform
quant_onnx_path = r"{os.path.abspath(espdl_path)}".replace(".espdl", "_quantized.onnx")
print(f"🔄 [Worker] Exporting Scale-Preserving ONNX -> {{quant_onnx_path}}")
export_ppq_graph(graph, platform=TargetPlatform.ONNXRUNTIME, graph_save_to=quant_onnx_path)
print("✅ [Worker] ESP-DL Quantization finished successfully!")
'''
        with open(worker_script, 'w', encoding='utf-8') as f:
            f.write(worker_script_content)

        print("🔄 Spawning isolated TQT worker...")
        env = os.environ.copy()
        if args.device == "cpu": env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.check_call([sys.executable, worker_script], env=env)
        
        if os.path.exists(calib_npy_path): os.remove(calib_npy_path)
        if os.path.exists(worker_script): os.remove(worker_script)

        # IMPORTANT: We use the ORIGINAL (full) onnx_path for TFLite conversion
        # to ensure onnx2tf can always produce a full integer quantization.
        if args.tflite:
            tflite_suite_export(onnx_path, calib_data, args, espdl_path)

    except Exception as e:
        print(f"❌ Quantization error: {e}")
        import traceback
        traceback.print_exc()

    if os.path.exists(espdl_path):
        print(f"✅ Export complete: {espdl_path}")
        if args.organize:
            organize_output_folder(os.path.dirname(espdl_path))
    else:
        print(f"❌ Export FAILED: {espdl_path} not found")
        sys.exit(1)

def tflite_suite_export(onnx_path, calib_data, args, espdl_path):
    """Generates the full suite of TFLite variants using the FP32 ONNX model"""
    output_dir = os.path.dirname(espdl_path)
    model_base_name = os.path.basename(espdl_path).replace(".espdl", "")
    
    calib_npy_path = os.path.join(output_dir, "onnx2tf_calib_temp.npy")
    onnx2tf_calib = []
    for i in range(min(len(calib_data), 100)):
        sample = calib_data[i].numpy().transpose(0, 2, 3, 1) # NCHW -> NHWC
        onnx2tf_calib.append(sample)
    np.save(calib_npy_path, np.concatenate(onnx2tf_calib, axis=0))
    
    variants = [
        ("_float32",            {"output_integer_quantized_tflite": False}),
        ("_float16",            {"output_float16_quantized_tflite": True}),
        ("_dynamic_range_quant", {"output_dynamic_range_quantized_tflite": True}),
        ("_full_integer_quant",  {"output_integer_quantized_tflite": True, "input_quant_dtype": "int8", "output_quant_dtype": "int8"}), 
    ]
    
    print(f"📦 Generating TFLite variant suite for {args.target} in {output_dir}...")

    for suffix, flags in variants:
        target_path = os.path.join(output_dir, f"{model_base_name}_quantized{suffix}.tflite")
        print(f"   🚀 Exporting: {os.path.basename(target_path)}")
        try:
            onnx2tf.convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=output_dir,
                custom_input_op_name_np_data_path=[["input", calib_npy_path, [0,0,0], [1,1,1]]],
                not_use_onnxsim=True,
                non_verbose=True,
                **flags
            )
            
            patterns = ["model_integer_only.tflite", "model_float16.tflite", "model_dynamic_range_quant.tflite", "model.tflite"]
            found = False
            for p in patterns:
                src = os.path.join(output_dir, p)
                if os.path.exists(src):
                    if os.path.abspath(src) != os.path.abspath(target_path):
                        shutil.move(src, target_path)
                    found = True; break
            
            if not found:
                for root, _, files in os.walk(output_dir):
                    for f in files:
                        if f.endswith(".tflite") and all(x not in f for x in ["quantized", "espdl"]):
                            shutil.move(os.path.join(root, f), target_path)
                            found = True; break
                    if found: break
            if found: print(f"   ✅ Saved: {os.path.basename(target_path)}")
        except Exception as e:
            print(f"   ⚠️ Variant {suffix} failed: {e}")

    if os.path.exists(calib_npy_path): os.remove(calib_npy_path)
    for folder in ["saved_model", "tf_layers"]:
        p = os.path.join(output_dir, folder)
        if os.path.exists(p): shutil.rmtree(p)

def organize_output_folder(output_dir):
    """Clean root directory keeping ONLY essential artifacts"""
    full_models_dir = os.path.join(output_dir, "full_models")
    if not os.path.exists(full_models_dir): os.makedirs(full_models_dir)
    print(f"🧹 Organizing output folder: {output_dir}")
    
    to_move_patterns = [".onnx", "_float32.tflite", "_float16.tflite", "_dynamic_range_quant.tflite", ".keras"]
    
    for f in os.listdir(output_dir):
        src = os.path.join(output_dir, f)
        if os.path.isdir(src): continue
        
        move_needed = False
        for pattern in to_move_patterns:
            if f.endswith(pattern):
                move_needed = True; break
        
        # USER RULE: Keep ONLY esp32 _full_integer_quant.tflite in root. All other TFLites go to full_models.
        if f.endswith(".tflite"):
            is_main_esp32_quant = "esp32_quantized_full_integer_quant.tflite" in f
            if not is_main_esp32_quant:
                move_needed = True
        
        # ALSO: preserve espdl, info, json in root
        if any(f.endswith(ext) for ext in [".espdl", ".info", ".json"]):
            move_needed = False
            
        if move_needed:
            try: shutil.move(src, os.path.join(full_models_dir, f))
            except: pass

if __name__ == "__main__":
    main()
