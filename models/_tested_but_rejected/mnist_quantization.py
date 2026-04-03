# models/mnist_quantization.py
"""
mnist_quantization – TF QAT Guide Reference Implementation
===========================================================
Design goal: Replicate the MNIST model from the TensorFlow Model Optimization
quantization training guide. Used as a reference/teaching example for QAT
practices in this project (ReLU6, no BN, GAP, explicit bias).

Reference:
  https://www.tensorflow.org/model_optimization/guide/quantization/training_example

Architecture (mnist_quantization — quantization-friendly):
  - Conv2D(32) + ReLU6 + MaxPool    → Block 1
  - Conv2D(64) + ReLU6 + MaxPool    → Block 2
  - Conv2D(64) + ReLU6              → Block 3
  - GlobalAveragePooling2D → Dense(NB_CLASSES) Softmax  (no hidden Dense)

Architecture (mnist_baseline — for comparison):
  - Conv2D(32,64,64) + ReLU + MaxPool × 2 → Flatten → Dense(64) → Softmax

Utilities also provided:
  - apply_qat_to_mnist(model): wraps with tfmot.quantize_model()
  - convert_to_tflite(model, rep_dataset): full INT8 or dynamic-range convert
  - create_representative_dataset(x_data): calibration generator

Notes:
  - ReLU6 via keras.layers.ReLU(max_value=6) — cleanest QAT pattern
  - No QAT wrapper on the function itself; call apply_qat_to_mnist() separately
  - Explicit use_bias=True on all conv layers for clean quantizer behavior

Estimated: ~60K parameters → ~60 KB after INT8 quantization.
"""

import tensorflow as tf
import parameters as params
from utils.keras_helper import keras, tfmot

def create_mnist_quantization():
    """
    MNIST model optimized for quantization - reference implementation
    Based on: https://www.tensorflow.org/model_optimization/guide/quantization/training_example
    
    Features:
    - Simple architecture for easy quantization
    - No BatchNormalization layers
    - ReLU6 activations for better quantization
    - GlobalAveragePooling instead of Flatten
    """
    model = keras.Sequential([
        keras.layers.Input(shape=params.INPUT_SHAPE, name='input'),
        
        # First convolutional block
        keras.layers.Conv2D(
            32, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv1'
        ),
        keras.layers.ReLU(max_value=6, name='relu6_1'),  # ReLU6 for better quantization
        keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second convolutional block
        keras.layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv2'
        ),
        keras.layers.ReLU(max_value=6, name='relu6_2'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third convolutional block
        keras.layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv3'
        ),
        keras.layers.ReLU(max_value=6, name='relu6_3'),
        
        # Global average pooling (quantization friendly)
        keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Final classification layer
        keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model

def create_mnist_baseline():
    """
    Baseline MNIST model without quantization optimizations
    For comparison with quantized version
    """
    model = keras.Sequential([
        keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Convolutional layers
        keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
        keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
        
        # Classification head
        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(64, activation='relu', name='dense1'),
        keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model

def get_model_info():
    """Return model information for reference"""
    return {
        "name": "MNIST Quantization Reference",
        "source": "https://www.tensorflow.org/model_optimization/guide/quantization/training_example",
        "description": "MNIST model optimized for quantization with ReLU6 and no BatchNorm",
        "quantization_friendly": True,
        "features": [
            "ReLU6 activations for better quantization ranges",
            "No BatchNormalization layers",
            "GlobalAveragePooling instead of Flatten",
            "Explicit bias usage",
            "He normal initialization"
        ],
        "input_shape": params.INPUT_SHAPE,
        "nb_classes": params.NB_CLASSES
    }

# Quantization helper functions
def apply_qat_to_mnist(model):
    """
    Apply Quantization Aware Training to MNIST model
    Following TF official example
    """
    if tfmot is None:
        print("❌ tensorflow-model-optimization not available")
        return model
    
    try:
        # Apply quantization to the entire model
        qat_model = tfmot.quantization.keras.quantize_model(model)
        
        print("✅ QAT applied to MNIST model")
        return qat_model
        
    except Exception as e:
        print(f"❌ QAT failed: {e}")
        return model

def convert_to_tflite(model, x_test=None):
    """
    Convert MNIST model to TFLite with quantization
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if x_test is not None:
        def representative_data_gen():
            for i in range(min(100, len(x_test))):
                yield [x_test[i:i+1].astype("float32")]
        
        converter.representative_dataset = representative_data_gen
        # Full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        # Dynamic range quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tflite_model = converter.convert()
    return tflite_model

# Training configuration for MNIST
def get_mnist_training_config():
    """
    Return recommended training configuration for MNIST models
    """
    return {
        "batch_size": 128,
        "epochs": 10,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"]
    }