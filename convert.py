import tensorflow as tf
import numpy as np
import os
from utils.dataset import load_digit_dataset
from utils.preprocess import preprocess_images
import parameters as params

def representative_dataset():
    """Generate enhanced representative dataset for better quantization"""
    (x_train, _), _ = load_digit_dataset()
    
    # Use ESP-DL specific preprocessing for quantization calibration
    if params.ESP_DL_QUANTIZE:
        x_train = preprocess_images_esp_dl(x_train)
    else:
        x_train = preprocess_images(x_train)
    
    # Use more diverse samples for better quantization
    num_samples = min(params.QUANTIZE_NUM_SAMPLES, len(x_train))
    
    # Select samples from different parts of the dataset
    indices = np.linspace(0, len(x_train)-1, num_samples, dtype=int)
    
    for idx in indices:
        yield [x_train[idx:idx+1]]

def convert_to_tflite_micro():
    """Main conversion function with enhanced quantization"""
    if params.QUANTIZE_MODEL:
        converter.representative_dataset = representative_dataset_enhanced
    
    model_path = os.path.join(params.OUTPUT_DIR, f"{params.MODEL_FILENAME}.h5")
    output_path = os.path.join(params.OUTPUT_DIR, params.TFLITE_FILENAME)
    
    # Load model
    print(f"Loading model from: {model_path}")
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
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Print detailed quantization info
        print_quantization_info(output_path, params.QUANTIZE_MODEL, params.ESP_DL_QUANTIZE)
        
        return tflite_model
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        # Fallback: try without specific type constraints
        if params.QUANTIZE_MODEL:
            print("🔄 Trying fallback conversion...")
            converter.inference_input_type = None
            converter.inference_output_type = None
            tflite_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print("⚠️  Fallback conversion completed without strict type enforcement")
            return tflite_model
        else:
            raise
            
def print_quantization_info(model_path, quantized, esp_dl_quantize):
    """Print detailed information about the quantized model"""
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
        
        print(f"\nOutput Details:")
        print(f"  dtype: {output_details[0]['dtype']}")
        print(f"  scale: {output_details[0]['scale']:.6f}")
        print(f"  zero_point: {output_details[0]['zero_point']}")
    
    model_size_kb = os.path.getsize(model_path) / 1024
    print(f"\nModel size: {model_size_kb:.1f} KB")
    
def check_esp_dl_compatibility(tflite_model_path):
    """Check if the quantized model is compatible with ESP-DL"""
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
    except:
        pass
    
    return compatible, issues

if __name__ == "__main__":
    convert_to_tflite_micro()