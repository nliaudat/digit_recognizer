import onnx
import numpy as np
import tensorflow as tf
import os
import argparse

def extract_scales(onnx_model):
    """Extract quantization scales from ONNX model"""
    scales = []
    for node in onnx_model.graph.node:
        if node.op_type == "QuantizeLinear":
            # Extract scale value from constant input
            scale_name = node.input[1]
            for init in onnx_model.graph.initializer:
                if init.name == scale_name:
                    scale_val = np.frombuffer(init.raw_data, dtype=np.float32)
                    if len(scale_val) > 0:
                        scales.append(scale_val[0])
    return scales

def inspect_tflite(model_path):
    """Check if TFLite model is valid and contains int8 ops"""
    if not os.path.exists(model_path):
        print(f"  [!] File not found: {model_path}")
        return
        
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        details = interpreter.get_tensor_details()
        op_types = set()
        
        has_quant = False
        for detail in details:
            if 'quantization' in detail and detail['quantization'][0] is not None:
                scale = detail['quantization'][0]
                if isinstance(scale, (list, np.ndarray)):
                    if len(scale) > 0:
                        has_quant = True
                elif scale > 0:
                    has_quant = True
                
                op_types.add(detail['dtype'])
        
        print(f"  {os.path.basename(model_path)}:")
        print(f"    Data types: {op_types}")
        if np.int8 in op_types or np.uint8 in op_types:
            print("    [+] Model has integer quantization")
        else:
            print("    [-] Model is float32 - quantization lost!")
        return True
    except Exception as e:
        print(f"  [!] Invalid TFLite: {e}")
        return False

def compare_models(ptq_onnx, tqt_onnx, ptq_tflite, tqt_tflite):
    """Compare quantization parameters between PTQ and TQT models"""
    
    print("\n[1] Comparing ONNX QDQ nodes")
    print("-" * 30)
    
    ptq_scales = []
    if os.path.exists(ptq_onnx) and ptq_onnx != "":
        ptq_model = onnx.load(ptq_onnx)
        ptq_scales = extract_scales(ptq_model)
        print(f"  PTQ Scales found: {len(ptq_scales)}")
    elif ptq_onnx != "":
        print(f"  [!] PTQ ONNX not found: {ptq_onnx}")
        
    tqt_scales = []
    if os.path.exists(tqt_onnx) and tqt_onnx != "":
        tqt_model = onnx.load(tqt_onnx)
        tqt_scales = extract_scales(tqt_model)
        print(f"  TQT Scales found: {len(tqt_scales)}")
    elif tqt_onnx != "":
        print(f"  [!] TQT ONNX not found: {tqt_onnx}")

    if ptq_scales and tqt_scales:
        print("\nScale comparison (PTQ vs TQT):")
        # Compare as many as we can (they might have different number of nodes if one was simplified)
        limit = min(len(ptq_scales), len(tqt_scales), 10)
        for i in range(limit):
            ps = ptq_scales[i]
            ts = tqt_scales[i]
            ratio = ts/ps if ps > 0 else 0
            print(f"  Layer {i}: PTQ={ps:.6f}, TQT={ts:.6f}, ratio={ratio:.2f}")
            if ratio > 2.0 or ratio < 0.5:
                print(f"    [!] Large scale difference - suggests possible range mismatch in calibration")
    
    print("\n[2] Checking TFLite Integrity")
    print("-" * 30)
    for name, path in [("PTQ", ptq_tflite), ("TQT", tqt_tflite)]:
        if path:
            inspect_tflite(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Target folder containing TQT artifacts")
    parser.add_argument("--ptq_folder", help="Baseline folder containing PTQ artifacts")
    parser.add_argument("--ptq_onnx", help="Path to baseline PTQ ONNX", default="")
    parser.add_argument("--tqt_onnx", help="Path to TQT quantized ONNX", default="")
    parser.add_argument("--ptq_tflite", help="Path to baseline PTQ TFLite", default="")
    parser.add_argument("--tqt_tflite", help="Path to TQT converted TFLite", default="")
    args = parser.parse_args()
    
    tqt_onnx = args.tqt_onnx
    tqt_tflite = args.tqt_tflite
    ptq_onnx = args.ptq_onnx
    ptq_tflite = args.ptq_tflite

    if args.folder:
        # Try to find TQT files in the folder
        for f in os.listdir(args.folder):
            if f.endswith("_quantized.onnx"):
                tqt_onnx = os.path.join(args.folder, f)
            if f.endswith(".tflite") and "float" not in f and "dynamic" not in f:
                tqt_tflite = os.path.join(args.folder, f)
    
    if args.ptq_folder:
        # Try to find PTQ files in the folder
        for f in os.listdir(args.ptq_folder):
            if f.endswith(".onnx") and "quantized" not in f:
                ptq_onnx = os.path.join(args.ptq_folder, f)
            if f.endswith(".tflite") and "float" not in f:
                ptq_tflite = os.path.join(args.ptq_folder, f)

    compare_models(ptq_onnx, tqt_onnx, ptq_tflite, tqt_tflite)
