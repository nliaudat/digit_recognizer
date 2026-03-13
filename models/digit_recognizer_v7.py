# models/digit_recognizer_v7.py
"""
digit_recognizer_v7 – Balanced Depthwise Separable CNN (~80K params)
======================================================================
Design goal: Best balance of accuracy, model size, and speed based on
analysis of v1–v6 benchmark results. Targets ~80K parameters — between
v1's 135K and v4's 90K — using a depthwise separable middle block.

Architecture (grayscale, default):
  - Conv2D(28) + ReLU + MaxPool          → wider entry than v4(20)
  - DepthwiseConv2D(3×3) + Pointwise Conv2D(48) + ReLU + MaxPool
    (depthwise separable: ~3× cheaper than full Conv)
  - Conv2D(64) + ReLU                   → feature refinement
  - GlobalAveragePooling2D
  - Dense(72) + ReLU + Dropout(0.2) → Softmax

Variants also provided:
  - create_digit_recognizer_v7_compact()      → ~50K params, faster
  - create_digit_recognizer_v7_high_accuracy() → ~100K params, more accurate

Notes:
  - Standard ReLU throughout (ESP-DL compatible)
  - No BatchNormalization → simpler quantization graph
  - No QAT wrapper

Estimated: ~80K parameters → ~45–50 KB after INT8 quantization.
"""

import tensorflow as tf
import parameters as params

def create_digit_recognizer_v7():
    """
    Optimized digit recognizer v7 - Best balance of accuracy, size & speed
    Based on analysis of v1-v6 performance data
    
    Design principles:
    - Target: ~80K parameters (between v1's 135K and v5's 90K)
    - Optimized for 10-class grayscale (20x32)
    - QAT and quantization compatible
    - ESP-DL/TFLite Micro ready
    """
    input_channels = params.INPUT_SHAPE[-1]
    
    if input_channels == 1:
        print(f"Creating v7 optimized grayscale model for {params.NB_CLASSES} classes")
        return create_digit_recognizer_v7_grayscale()
    else:
        return create_digit_recognizer_v7_grayscale()  # Focus on grayscale optimization

def create_digit_recognizer_v7_grayscale():
    """
    Optimized grayscale version targeting 80K parameters
    - Balances v1's accuracy with v4's efficiency
    - Maintains QAT compatibility
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # ===== OPTIMIZED BACKBONE =====
    # Layer 1: Slightly more filters than v4 for better feature extraction
    x = tf.keras.layers.Conv2D(
        28, (3, 3), padding='same',  # Between v4(20) and v1(32)
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1'
    )(inputs)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2: Depthwise separable for efficiency (like v1)
    x = tf.keras.layers.DepthwiseConv2D(
        (3, 3), padding='same',
        depthwise_initializer='he_normal',
        use_bias=True,
        name='depthwise1'
    )(x)
    x = tf.keras.layers.ReLU(name='depthwise_relu1')(x)
    x = tf.keras.layers.Conv2D(
        48, (1, 1),  # Pointwise - optimized count
        kernel_initializer='he_normal',
        use_bias=True,
        name='pointwise1'
    )(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3: Feature refinement
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',  # Similar to v1 but optimized
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2'
    )(x)
    x = tf.keras.layers.ReLU(name='relu3')(x)
    
    # ===== EFFICIENT CLASSIFICATION HEAD =====
    # Global pooling (quantization friendly)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Single dense layer (optimized size)
    x = tf.keras.layers.Dense(72, activation='relu', name='dense1')(x)
    
    # Light dropout for regularization
    x = tf.keras.layers.Dropout(0.2, name='dropout')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v7_grayscale_{params.NB_CLASSES}classes")

def create_digit_recognizer_v7_compact():
    """
    More compact version (~50K parameters)
    For applications where size is critical but better accuracy than v6 needed
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Compact backbone
    x = tf.keras.layers.Conv2D(
        24, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1'
    )(inputs)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    x = tf.keras.layers.Conv2D(
        36, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2'
    )(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    x = tf.keras.layers.Conv2D(
        48, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3'
    )(x)
    x = tf.keras.layers.ReLU(name='relu3')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Compact dense
    x = tf.keras.layers.Dense(56, activation='relu', name='dense1')(x)
    
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v7_compact_{params.NB_CLASSES}classes")

def create_digit_recognizer_v7_high_accuracy():
    """
    Higher accuracy version (~100K parameters)
    When accuracy is prioritized over size
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Enhanced backbone
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1'
    )(inputs)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Depthwise separable
    x = tf.keras.layers.DepthwiseConv2D(
        (3, 3), padding='same',
        depthwise_initializer='he_normal',
        use_bias=True,
        name='depthwise1'
    )(x)
    x = tf.keras.layers.ReLU(name='depthwise_relu1')(x)
    x = tf.keras.layers.Conv2D(
        64, (1, 1),
        kernel_initializer='he_normal',
        use_bias=True,
        name='pointwise1'
    )(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Additional conv layer
    x = tf.keras.layers.Conv2D(
        80, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3'
    )(x)
    x = tf.keras.layers.ReLU(name='relu3')(x)
    
    x = tf.keras.layers.Conv2D(
        96, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv4'
    )(x)
    x = tf.keras.layers.ReLU(name='relu4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Enhanced classifier
    x = tf.keras.layers.Dense(96, activation='relu', name='dense1')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout1')(x)
    x = tf.keras.layers.Dense(64, activation='relu', name='dense2')(x)
    
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v7_high_acc_{params.NB_CLASSES}classes")

# QAT and Quantization Utilities
def apply_qat_v7(model):
    """
    Apply Quantization Aware Training to v7 model
    """
    try:
        import tensorflow_model_optimization as tfmot
        
        # Apply quantization to the entire model
        quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
        quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
        quantize_scope = tfmot.quantization.keras.quantize_scope
        
        with quantize_scope():
            # Annotate the model for QAT
            annotated_model = quantize_annotate_model(model)
            
            # Create QAT model
            qat_model = tfmot.quantization.keras.quantize_apply(
                annotated_model,
                tfmot.quantization.keras.Default8BitQuantizeScheme(),
                tfmot.quantization.keras.Default8BitQuantizeConfig()
            )
        
        print("✅ QAT applied to v7 model")
        return qat_model
        
    except ImportError:
        print("❌ tensorflow-model-optimization not available")
        return model

def convert_v7_to_tflite(model, representative_dataset=None):
    """
    Convert v7 model to TFLite with optimization
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if representative_dataset is not None:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        print("✅ Full integer quantization applied")
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        print("✅ Dynamic range quantization applied")
    
    tflite_model = converter.convert()
    return tflite_model

def get_v7_training_config():
    """
    Recommended training configuration for v7 models
    """
    return {
        "batch_size": 128,
        "epochs": 15,  # Slightly more than typical for better convergence
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    }

def analyze_v7_performance_targets():
    """
    Analyze expected performance based on v1-v6 data
    """
    print("=== V7 PERFORMANCE TARGETS ===")
    print("Based on analysis of v1-v6 models:")
    print()
    
    targets = {
        'balanced': {
            'params_target': '80,000',
            'accuracy_target': '0.989+ (between v1 0.987 and v4 0.9906)',
            'size_target': '45-50KB',
            'speed_target': '6,000+ inferences/sec',
            'rationale': 'Balances v4 accuracy with better efficiency'
        },
        'compact': {
            'params_target': '50,000',
            'accuracy_target': '0.980+ (better than v6 0.9486)',
            'size_target': '25-30KB',
            'speed_target': '8,000+ inferences/sec',
            'rationale': 'Significant improvement over v6 with minimal size increase'
        },
        'high_accuracy': {
            'params_target': '100,000',
            'accuracy_target': '0.991+ (approaching original_haverland)',
            'size_target': '60-65KB',
            'speed_target': '5,500+ inferences/sec',
            'rationale': 'Near original accuracy with modern architecture'
        }
    }
    
    for variant, target in targets.items():
        print(f"--- {variant.upper()} ---")
        for key, value in target.items():
            print(f"{key}: {value}")
        print()

if __name__ == "__main__":
    # Test the main function
    model = create_digit_recognizer_v7()
    print(f"Created model: {model.name}")
    model.summary()
    
    # Show performance targets
    analyze_v7_performance_targets()
    
    # Compare variants
    print("\n=== V7 VARIANTS COMPARISON ===")
    variants = {
        'Compact': create_digit_recognizer_v7_compact(),
        'Balanced': create_digit_recognizer_v7(),
        'High Accuracy': create_digit_recognizer_v7_high_accuracy()
    }
    
    for name, model in variants.items():
        params_count = model.count_params()
        size_kb = (params_count * 4) / 1024
        print(f"{name:15}: {params_count:6,} params, {size_kb:5.1f}KB (float32)")