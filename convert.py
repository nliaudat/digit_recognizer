import tensorflow as tf
import numpy as np
import os
from utils.dataset import load_digit_dataset
from utils.preprocess import preprocess_images, validate_quantization_combination
import parameters as params

def representative_dataset():
    """Generate enhanced representative dataset for better quantization"""
    (x_train, _), _ = load_digit_dataset()
    
    # CRITICAL: Always use the SAME preprocessing as training but for inference context
    # This ensures consistency between training and conversion
    x_train = preprocess_images(x_train, for_training=False)
    
    # Use more diverse samples for better quantization
    num_samples = min(params.QUANTIZE_NUM_SAMPLES, len(x_train))
    
    # Select samples from different parts of the dataset
    indices = np.linspace(0, len(x_train)-1, num_samples, dtype=int)
    
    for idx in indices:
        yield [x_train[idx:idx+1]]

def convert_to_tflite_micro():
    """Main conversion function with enhanced quantization for all 9 cases"""
    print("🎯 TFLite Conversion Configuration:")
    print(f"   QUANTIZE_MODEL: {params.QUANTIZE_MODEL}")
    print(f"   USE_QAT: {params.USE_QAT}") 
    print(f"   ESP_DL_QUANTIZE: {params.ESP_DL_QUANTIZE}")
    
    # Validate quantization combination first
    is_valid, msg = validate_quantization_combination()
    if not is_valid:
        print(f"❌ {msg}")
        return None
    
    print(f"✅ {msg}")
    
    model_path = os.path.join(params.OUTPUT_DIR, f"{params.MODEL_FILENAME}.h5")
    output_path = os.path.join(params.OUTPUT_DIR, params.TFLITE_FILENAME)
    
    # Load model
    print(f"📦 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
        
    model = tf.keras.models.load_model(model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if params.QUANTIZE_MODEL:
        print("🔧 Configuring post-training quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        
        if params.ESP_DL_QUANTIZE:
            print("🎯 Using INT8 quantization for ESP-DL [-128, 127]")
            # Full integer quantization for ESP-DL
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Enable experimental quantizer for better results
            converter.experimental_new_quantizer = True
            converter._experimental_disable_per_channel = False
            
        else:
            print("🔧 Using standard UINT8 quantization [0, 255]")
            # Standard uint8 quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        converter.allow_custom_ops = False
        
    else:
        print("🔧 Converting without quantization (float32)")
        # No quantization - keep as float32
    
    # Convert model
    try:
        print("🔄 Converting model to TFLite...")
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved to: {output_path}")
        
        # Print detailed quantization info
        print_quantization_info(output_path, params.QUANTIZE_MODEL, params.ESP_DL_QUANTIZE)
        
        # Verify ESP-DL compatibility if applicable
        if params.QUANTIZE_MODEL and params.ESP_DL_QUANTIZE:
            compatible, issues = check_esp_dl_compatibility(output_path)
            if compatible:
                print("✅ Model is ESP-DL compatible")
            else:
                print("❌ Model may have ESP-DL compatibility issues:")
                for issue in issues:
                    print(f"   - {issue}")
        
        return tflite_model
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        # Fallback: try without specific type constraints
        if params.QUANTIZE_MODEL:
            print("🔄 Trying fallback conversion...")
            try:
                converter.inference_input_type = None
                converter.inference_output_type = None
                tflite_model = converter.convert()
                
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)
                
                print("⚠️  Fallback conversion completed without strict type enforcement")
                print(f"✅ TFLite model saved to: {output_path}")
                return tflite_model
            except Exception as fallback_error:
                print(f"❌ Fallback conversion also failed: {fallback_error}")
                return None
        else:
            raise
            
def print_quantization_info(model_path, quantized, esp_dl_quantize):
    """Print detailed information about the quantized model"""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
        
    # Load the model to inspect quantization parameters
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n📊 QUANTIZATION INFORMATION:")
    print("=" * 50)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Quantized: {quantized}")
    print(f"ESP-DL Mode: {esp_dl_quantize}")
    
    if quantized:
        print(f"\nInput Details:")
        print(f"  dtype: {input_details[0]['dtype']}")
        print(f"  scale: {input_details[0]['scale']:.6f}")
        print(f"  zero_point: {input_details[0]['zero_point']}")
        print(f"  shape: {input_details[0]['shape']}")
        
        print(f"\nOutput Details:")
        print(f"  dtype: {output_details[0]['dtype']}")
        print(f"  scale: {output_details[0]['scale']:.6f}")
        print(f"  zero_point: {output_details[0]['zero_point']}")
        print(f"  shape: {output_details[0]['shape']}")
    
    model_size_kb = os.path.getsize(model_path) / 1024
    print(f"\nModel size: {model_size_kb:.1f} KB")
    
    # Print quantization ranges
    if quantized:
        if esp_dl_quantize:
            print("Quantization range: INT8 [-128, 127]")
        else:
            print("Quantization range: UINT8 [0, 255]")
    else:
        print("Quantization range: FLOAT32")
    
def check_esp_dl_compatibility(tflite_model_path):
    """Check if the quantized model is compatible with ESP-DL"""
    if not os.path.exists(tflite_model_path):
        return False, ["Model file not found"]
        
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    compatible = True
    issues = []
    
    # Check input type
    if input_details[0]['dtype'] != np.int8:
        compatible = False
        issues.append("Input type should be int8 for ESP-DL")
    
    # Check output type  
    if output_details[0]['dtype'] != np.int8:
        compatible = False
        issues.append("Output type should be int8 for ESP-DL")
    
    # Check for unsupported ops
    try:
        model = tf.lite.Interpreter(model_path=tflite_model_path)
        ops = model._get_ops_details()
        supported_ops = ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED', 
                        'RESHAPE', 'SOFTMAX', 'MAX_POOL_2D', 'AVERAGE_POOL_2D']
        
        for op in ops:
            if op['op_name'] not in supported_ops:
                issues.append(f"Unsupported op: {op['op_name']}")
                compatible = False
                
    except Exception as e:
        issues.append(f"Unsupported op check failed: {e}")
        compatible = False
    
    return compatible, issues

def convert_float_model():
    """Convert to float32 TFLite model regardless of quantization settings"""
    print("🔧 Converting to float32 TFLite model...")
    
    model_path = os.path.join(params.OUTPUT_DIR, f"{params.MODEL_FILENAME}.h5")
    output_path = os.path.join(params.OUTPUT_DIR, params.FLOAT_TFLITE_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
        
    model = tf.keras.models.load_model(model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # No quantization for float model
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size_kb = len(tflite_model) / 1024
    print(f"✅ Float32 TFLite model saved: {output_path} ({model_size_kb:.1f} KB)")
    
    return tflite_model

def batch_convert_all_combinations():
    """Batch convert for all valid quantization combinations (for testing)"""
    original_quantize = params.QUANTIZE_MODEL
    original_qat = params.USE_QAT
    original_esp_dl = params.ESP_DL_QUANTIZE
    
    test_combinations = [
        (False, False, False, "float32"),
        (True, False, False, "uint8_standard"),
        (True, False, True, "int8_esp_dl"),
        (True, True, False, "qat_uint8"),
        (True, True, True, "qat_int8_esp_dl"),
    ]
    
    results = {}
    
    for quantize, qat, esp_dl, name in test_combinations:
        print(f"\n{'='*60}")
        print(f"Testing combination: {name}")
        print(f"{'='*60}")
        
        # Skip invalid combinations
        if not quantize and esp_dl:
            print("❌ Skipping invalid combination: ESP-DL without quantization")
            continue
            
        params.QUANTIZE_MODEL = quantize
        params.USE_QAT = qat
        params.ESP_DL_QUANTIZE = esp_dl
        
        # Update filename for this combination
        original_tflite = params.TFLITE_FILENAME
        params.TFLITE_FILENAME = f"{params.MODEL_FILENAME}_{name}.tflite"
        
        try:
            result = convert_to_tflite_micro()
            if result is not None:
                results[name] = "SUCCESS"
            else:
                results[name] = "FAILED"
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            results[name] = "FAILED"
        
        # Restore original filename
        params.TFLITE_FILENAME = original_tflite
    
    # Restore original parameters
    params.QUANTIZE_MODEL = original_quantize
    params.USE_QAT = original_qat
    params.ESP_DL_QUANTIZE = original_esp_dl
    
    print(f"\n{'='*60}")
    print("BATCH CONVERSION RESULTS:")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"  {name:20} : {result}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert model to TFLite')
    parser.add_argument('--batch', action='store_true', help='Test all quantization combinations')
    parser.add_argument('--float', action='store_true', help='Convert to float32 model only')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_convert_all_combinations()
    elif args.float:
        convert_float_model()
    else:
        convert_to_tflite_micro()