#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="TQT/ESP-DL Minimal Worker")
    parser.add_argument("--model",    help="Model name to resolve paths automatically")
    parser.add_argument("--keras",    help="Path to .keras model")
    parser.add_argument("--onnx",     help="Path to .onnx model")
    parser.add_argument("--output",   help="Explicit output .espdl path")
    parser.add_argument("--target",   default="esp32", help="Target chip (esp32, esp32s3, etc.)")
    parser.add_argument("--bits",     type=int,   default=8)
    parser.add_argument("--steps",    type=int,   default=500)
    parser.add_argument("--lr",       type=float, default=1e-5)
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
        espdl_path = os.path.join(args.output, f"{model_name}.espdl")
    else:
        espdl_path = args.output or os.path.join(out_dir, f"{model_name}.espdl")

    return keras_path, onnx_path, espdl_path

def main():
    args = parse_args()

    # Pre-set environments for project imports
    os.environ["DIGIT_NB_CLASSES"] = str(args.classes)
    os.environ["DIGIT_INPUT_CHANNELS"] = "1" if args.color == "gray" else "3"

    # Delayed imports
    import parameters as params
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq import QuantizationSettingFactory
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
    (x_train, _), _, _ = get_data_splits()
    n = min(args.calib_steps, len(x_train))
    x = preprocess_for_inference(x_train[:n]).astype('float32')
    # NHWC -> NCHW list
    calib_data = [torch.from_numpy(x[i].transpose(2, 0, 1)).unsqueeze(0) for i in range(n)]

    # 3. Target mapping
    target_backend = 'c' if args.target == 'esp32' else args.target

    # 4. Configure TQT
    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.tqt_optimization = (args.steps > 0)
    if quant_setting.tqt_optimization:
        s = quant_setting.tqt_optimization_setting
        s.steps = args.steps
        s.lr = args.lr
        s.collecting_device = args.device
        s.int_lambda = 0.5
        s.is_scale_trainable = True

    # 5. Run Quantization
    print(f"🚀 Starting TQT: {onnx_path} -> {espdl_path}")
    print(f"   target={target_backend} bits={args.bits} steps={args.steps} device={args.device}")
    
    try:
        espdl_quantize_onnx(
            onnx_import_file   = onnx_path,
            espdl_export_file  = espdl_path,
            calib_dataloader   = calib_data,
            calib_steps        = n,
            input_shape        = [1, (1 if args.color == "gray" else 3), 32, 20],
            target             = target_backend,
            num_of_bits        = args.bits,
            setting            = quant_setting,
            device             = args.device
        )
        print("DEBUG: espdl_quantize_onnx finished successfully!")
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
