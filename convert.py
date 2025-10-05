import tensorflow as tf
import numpy as np
import os
from utils.dataset import load_digit_dataset
from utils.preprocess import preprocess_images
import parameters as params

def representative_dataset():
    """Generate representative dataset for quantization"""
    (x_train, _), _ = load_digit_dataset()
    x_train = preprocess_images(x_train)
    
    for i in range(min(params.QUANTIZE_NUM_SAMPLES, len(x_train))):
        yield [x_train[i:i+1]]

def convert_to_tflite_micro():
    """Convert model to TFLite with ESP-DL compatible quantization"""
    
    model_path = os.path.join(params.OUTPUT_DIR, f"{params.MODEL_FILENAME}.h5")
    output_path = os.path.join(params.OUTPUT_DIR, params.TFLITE_FILENAME)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    if params.QUANTIZE_MODEL:
        # Convert to TFLite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        
        # ESP-DL specific quantization settings
        if params.ESP_DL_QUANTIZE:
            print("🔧 Converting with INT8 quantization for ESP-DL [-128, 127]...")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            print("🔧 Converting with UINT8 quantization [0, 255]...")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        converter.allow_custom_ops = False
        
        # Enable experimental features for better quantization
        converter.experimental_new_quantizer = True
        converter._experimental_disable_per_channel = False
        
    else:
        # Convert without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        print("Converting without quantization...")
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print model info
    model_size_kb = len(tflite_model) / 1024
    print(f"TFLite model saved: {output_path}")
    print(f"Model size: {model_size_kb:.1f} KB")
    print(f"Input shape: {params.INPUT_SHAPE}")
    print(f"Quantized: {params.QUANTIZE_MODEL}")
    if params.QUANTIZE_MODEL:
        print(f"ESP-DL Quantization: {params.ESP_DL_QUANTIZE}")
        print(f"Data type: {'int8 [-128,127]' if params.ESP_DL_QUANTIZE else 'uint8 [0,255]'}")
    
    return tflite_model

if __name__ == "__main__":
    convert_to_tflite_micro()