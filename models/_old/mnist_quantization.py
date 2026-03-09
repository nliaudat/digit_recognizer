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
  - ReLU6 via tf.keras.layers.ReLU(max_value=6) — cleanest QAT pattern
  - No QAT wrapper on the function itself; call apply_qat_to_mnist() separately
  - Explicit use_bias=True on all conv layers for clean quantizer behavior

Estimated: ~60K parameters → ~60 KB after INT8 quantization.
"""

import tensorflow as tf
import parameters as params

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
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE, name='input'),
        
        # First convolutional block
        tf.keras.layers.Conv2D(
            32, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv1'
        ),
        tf.keras.layers.ReLU(max_value=6, name='relu6_1'),  # ReLU6 for better quantization
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv2'
        ),
        tf.keras.layers.ReLU(max_value=6, name='relu6_2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv3'
        ),
        tf.keras.layers.ReLU(max_value=6, name='relu6_3'),
        
        # Global average pooling (quantization friendly)
        tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Final classification layer
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model

def create_mnist_baseline():
    """
    Baseline MNIST model without quantization optimizations
    For comparison with quantized version
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
        
        # Classification head
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(64, activation='relu', name='dense1'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
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
    try:
        import tensorflow_model_optimization as tfmot
        
        # Apply quantization to the entire model
        qat_model = tfmot.quantization.keras.quantize_model(model)
        
        print("✅ QAT applied to MNIST model")
        return qat_model
        
    except ImportError:
        print("❌ tensorflow-model-optimization not available")
        return model

def convert_to_tflite(model, representative_dataset=None):
    """
    Convert MNIST model to TFLite with quantization
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if representative_dataset is not None:
        converter.representative_dataset = representative_dataset
        # Full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        # Dynamic range quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tflite_model = converter.convert()
    return tflite_model

def create_representative_dataset(x_data, num_samples=100):
    """
    Create representative dataset for quantization calibration
    """
    def representative_data_gen():
        for i in range(min(num_samples, len(x_data))):
            yield [x_data[i:i+1].astype(np.float32)]
    return representative_data_gen

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

# Model comparison utility
def compare_models_quantization():
    """
    Compare quantization-friendly vs baseline models
    """
    quant_model = create_mnist_quantization()
    baseline_model = create_mnist_baseline()
    
    print("🔍 Model Comparison:")
    print(f"Quantization-friendly model parameters: {quant_model.count_params():,}")
    print(f"Baseline model parameters: {baseline_model.count_params():,}")
    
    # Check quantization compatibility
    quant_layers = [layer.__class__.__name__ for layer in quant_model.layers]
    baseline_layers = [layer.__class__.__name__ for layer in baseline_model.layers]
    
    print(f"Quantization-friendly layers: {quant_layers}")
    print(f"Baseline layers: {baseline_layers}")
    
    return {
        "quantization_friendly": quant_model,
        "baseline": baseline_model
    }