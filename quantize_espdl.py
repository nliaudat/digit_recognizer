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
        # 5. Run Quantization
        # espdl_quantize_onnx returns the quantized graph
        graph = espdl_quantize_onnx(
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
        print("✅ ESP-DL Quantization finished successfully!")

        # 6. Export Quantized ONNX (Scale-Preserving)
        # This ONNX contains QDQ nodes (QuantizeLinear/DequantizeLinear) with TQT-optimized scales
        quant_onnx_path = onnx_path.replace(".onnx", "_quantized.onnx")
        print(f"🔄 Exporting Scale-Preserving ONNX -> {quant_onnx_path}")
        export_ppq_graph(graph, platform=TargetPlatform.ONNXRUNTIME, graph_save_to=quant_onnx_path)
        
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
