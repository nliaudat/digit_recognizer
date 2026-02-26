# models/digit_recognizer_v6.py
"""
digit_recognizer_v6 – TFLite Micro & ESP-DL Optimized CNN
==========================================================
Design goal: Maximum hardware compatibility with ESP-DL by switching from
ReLU6 to standard ReLU and replacing SeparableConv2D with regular Conv2D.
Maintains the class-count adaptive capacity scaling from v5.

Architecture (grayscale):
  - Conv2D(24→36→48) + ReLU + MaxPool after first block
  - Optional Conv2D(56) for ≥50 classes
  - GlobalAveragePooling2D
  - Dense(64, optional Dense(48)) + ReLU → Softmax

Architecture (RGB):
  - Standard Conv2D(28→42→56) + ReLU (avoids SeparableConv for ESP-DL)
  - Class-adaptive Dense head

ESP-DL compatible ops: Conv2D, ReLU, MaxPool2D, GlobalAveragePool2D,
  FullyConnected, Softmax. Explicitly avoids: ReLU6, SeparableConv2D,
  BatchNormalization.

Notes:
  - No BatchNormalization (intentionally omitted for ESP-DL friendliness)
  - Dropout only applied in grayscale variant
  - No QAT wrapper

Estimated: ~50–140K parameters depending on class count and variant.
"""

import tensorflow as tf
import parameters as params

def create_digit_recognizer_v6():
    """
    Digit Recognizer v6 - Optimized for TFLite Micro & ESP-DL compatibility
    
    Key improvements over v6:
    - Standard ReLU instead of ReLU6 (better TFLite Micro support)
    - Optimized quantization-friendly architecture
    - ESP-DL compatible operations only
    - Maintains class-count optimizations (10 vs 100 classes)
    - Supports both RGB and Grayscale
    """
    input_channels = params.INPUT_SHAPE[-1]
    
    if input_channels == 1:
        print(f"Creating v6 grayscale model for {params.NB_CLASSES} classes")
        return create_digit_recognizer_v6_grayscale()
    elif input_channels == 3:
        print(f"Creating v6 RGB model for {params.NB_CLASSES} classes")
        return create_digit_recognizer_v6_rgb()
    else:
        print(f"Warning: Unsupported channel count {input_channels}. Using adaptive model.")
        return create_digit_recognizer_v6_adaptive()

def create_digit_recognizer_v6_grayscale():
    """
    Grayscale-optimized version with ESP-DL compatibility
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # ===== ESP-DL COMPATIBLE CONVOLUTIONAL BACKBONE =====
    # Layer 1: Standard Conv + ReLU (TFLite Micro compatible)
    x = tf.keras.layers.Conv2D(
        get_filters_v6(24, 32),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1'
    )(inputs)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    # After pool: 10x16xfilters
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        get_filters_v6(36, 48),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2'
    )(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        get_filters_v6(48, 64),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3'
    )(x)
    x = tf.keras.layers.ReLU(name='relu3')(x)
    
    # Layer 4: Additional capacity for 100-class discrimination
    if params.NB_CLASSES >= 50:
        x = tf.keras.layers.Conv2D(
            get_filters_v6(56, 72),
            (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv4'
        )(x)
        x = tf.keras.layers.ReLU(name='relu4')(x)
    
    # ===== QUANTIZATION-FRIENDLY FEATURE REDUCTION =====
    # GlobalAveragePooling (quantization friendly, ESP-DL compatible)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer with scaled units based on class complexity
    dense_units = get_dense_units_v6(64, 128)
    x = tf.keras.layers.Dense(dense_units, activation='relu', name='dense1')(x)
    
    # Optional: Additional dense layer for complex classification
    if params.NB_CLASSES >= 50:
        dense_units_2 = get_dense_units_v6(48, 96)
        x = tf.keras.layers.Dense(dense_units_2, activation='relu', name='dense2')(x)
    
    # Dropout for regularization (higher for more classes)
    dropout_rate = 0.1 if params.NB_CLASSES <= 10 else 0.2
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v6_grayscale_{params.NB_CLASSES}classes")

def create_digit_recognizer_v6_rgb():
    """
    RGB-optimized version with ESP-DL compatibility
    Uses standard Conv2D instead of SeparableConv for better ESP-DL support
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1: Standard Conv2D (better ESP-DL support than SeparableConv)
    x = tf.keras.layers.Conv2D(
        get_filters_v6(28, 36),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1'
    )(inputs)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        get_filters_v6(42, 56),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2'
    )(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        get_filters_v6(56, 72),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3'
    )(x)
    x = tf.keras.layers.ReLU(name='relu3')(x)
    
    # Layer 4: For 100-class discrimination
    if params.NB_CLASSES >= 50:
        x = tf.keras.layers.Conv2D(
            get_filters_v6(64, 80),
            (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv4'
        )(x)
        x = tf.keras.layers.ReLU(name='relu4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layers
    dense_units = get_dense_units_v6(72, 144)
    x = tf.keras.layers.Dense(dense_units, activation='relu', name='dense1')(x)
    
    if params.NB_CLASSES >= 50:
        x = tf.keras.layers.Dense(get_dense_units_v6(64, 96), activation='relu', name='dense2')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v6_rgb_{params.NB_CLASSES}classes")

def create_digit_recognizer_v6_compact():
    """
    Ultra-compact version optimized for microcontrollers
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Minimal convolutional backbone
    x = tf.keras.layers.Conv2D(
        get_filters_v6(16, 20),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1'
    )(inputs)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        get_filters_v6(24, 32),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2'
    )(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        get_filters_v6(32, 40),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3'
    )(x)
    x = tf.keras.layers.ReLU(name='relu3')(x)
    
    # Optional layer for 100 classes
    if params.NB_CLASSES >= 50:
        x = tf.keras.layers.Conv2D(
            get_filters_v6(40, 48),
            (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv4'
        )(x)
        x = tf.keras.layers.ReLU(name='relu4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Compact dense layer
    dense_units = get_dense_units_v6(48, 64)
    x = tf.keras.layers.Dense(dense_units, activation='relu', name='dense1')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v6_compact_{params.NB_CLASSES}classes")

def create_digit_recognizer_v6_high_accuracy():
    """
    High-accuracy version for complex classification (100 classes)
    Maintains ESP-DL compatibility
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # High-capacity backbone for 100-class discrimination
    x = tf.keras.layers.Conv2D(
        get_filters_v6(32, 48),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1'
    )(inputs)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Multiple conv layers for complex features
    x = tf.keras.layers.Conv2D(
        get_filters_v6(48, 64),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2'
    )(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    
    x = tf.keras.layers.Conv2D(
        get_filters_v6(64, 80),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3'
    )(x)
    x = tf.keras.layers.ReLU(name='relu3')(x)
    
    x = tf.keras.layers.Conv2D(
        get_filters_v6(72, 96),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv4'
    )(x)
    x = tf.keras.layers.ReLU(name='relu4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Larger dense layers for 100-class separation
    x = tf.keras.layers.Dense(get_dense_units_v6(96, 160), activation='relu', name='dense1')(x)
    x = tf.keras.layers.Dense(get_dense_units_v6(64, 112), activation='relu', name='dense2')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v6_high_acc_{params.NB_CLASSES}classes")

def create_digit_recognizer_v6_adaptive():
    """
    Adaptive version for unknown input configurations
    """
    return create_digit_recognizer_v6_grayscale()

def get_filters_v6(base_10_class, base_100_class):
    """
    Return appropriate filter count based on number of classes
    """
    if params.NB_CLASSES <= 10:
        return base_10_class
    elif params.NB_CLASSES <= 20:
        return int((base_10_class + base_100_class) / 2)
    else:
        return base_100_class

def get_dense_units_v6(base_10_class, base_100_class):
    """
    Return appropriate dense units based on number of classes
    """
    if params.NB_CLASSES <= 10:
        return base_10_class
    elif params.NB_CLASSES <= 20:
        return int((base_10_class + base_100_class) / 2)
    else:
        return base_100_class

def get_recommended_model_v6(priority='balanced'):
    """
    Get recommended v6 model based on priority and class count
    """
    if params.NB_CLASSES <= 10:
        priorities = {
            'size': create_digit_recognizer_v6_compact,
            'balanced': create_digit_recognizer_v6_grayscale,
            'accuracy': create_digit_recognizer_v6_grayscale
        }
    else:
        priorities = {
            'size': create_digit_recognizer_v6_grayscale,
            'balanced': create_digit_recognizer_v6_grayscale,
            'accuracy': create_digit_recognizer_v6_high_accuracy
        }
    
    return priorities.get(priority, create_digit_recognizer_v6_grayscale)()

def get_esp_dl_compatibility_info():
    """
    Return ESP-DL compatibility information for the model
    """
    return {
        "tflite_micro_support": "Fully compatible",
        "esp_dl_support": "Fully compatible", 
        "supported_operations": [
            "Conv2D",
            "ReLU", 
            "MaxPool2D",
            "GlobalAveragePool2D",
            "FullyConnected",
            "Softmax"
        ],
        "unsupported_operations": [
            "ReLU6",  # Removed for better compatibility
            "SeparableConv2D",  # Using standard Conv2D instead
            "BatchNormalization"  # Not used for quantization friendliness
        ],
        "quantization_ready": True,
        "recommended_quantization": "int8 post-training quantization"
    }

def analyze_v6_optimizations():
    """
    Analyze v6 specific optimizations for ESP32 deployment
    """
    print(f"=== V6 OPTIMIZATION ANALYSIS ===")
    print(f"Target: ESP32 with TFLite Micro & ESP-DL")
    print(f"Classes: {params.NB_CLASSES}")
    print(f"Input: {params.INPUT_SHAPE}")
    
    print("\n🔧 ESP-DL Optimizations:")
    print("✅ Standard ReLU (instead of ReLU6)")
    print("✅ Standard Conv2D (better support than SeparableConv)")
    print("✅ No BatchNormalization layers")
    print("✅ GlobalAveragePooling (quantization friendly)")
    print("✅ Explicit bias usage")
    
    print(f"\n🎯 Class-aware optimizations:")
    if params.NB_CLASSES <= 10:
        print("- Compact architecture for 0-9 digit recognition")
        print(f"- Filter range: 16-48")
        print(f"- Dense units: 48-64") 
    else:
        print("- Enhanced architecture for 0-99 number recognition")
        print(f"- Filter range: 20-96")
        print(f"- Dense units: 64-160")

def convert_to_tflite_v6(model, representative_dataset=None):
    """
    Convert v6 model to TFLite with ESP32-optimized quantization
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # ESP32-optimized quantization settings
    if representative_dataset is not None:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        print("✅ Full integer quantization (ESP32 optimized)")
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        print("✅ Dynamic range quantization")
    
    tflite_model = converter.convert()
    return tflite_model

if __name__ == "__main__":
    # Test the main function
    model = create_digit_recognizer_v6()
    print(f"Created model: {model.name}")
    model.summary()
    
    # Analyze optimizations
    analyze_v6_optimizations()
    
    # Show ESP-DL compatibility
    compat_info = get_esp_dl_compatibility_info()
    print(f"\n=== ESP-DL COMPATIBILITY ===")
    for key, value in compat_info.items():
        print(f"{key}: {value}")
    
    # Compare different versions
    print("\n=== V6 MODEL COMPARISON ===")
    versions = {
        'Compact': create_digit_recognizer_v6_compact(),
        'Balanced': create_digit_recognizer_v6_grayscale(),
        'High Accuracy': create_digit_recognizer_v6_high_accuracy()
    }
    
    for name, model in versions.items():
        params_count = model.count_params()
        print(f"{name:12}: {params_count:6,} parameters")
        
        # Estimate model size (approximate)
        size_kb = (params_count * 4) / 1024  # Rough estimate for float32
        quantized_size_kb = (params_count * 1) / 1024  # Rough estimate for int8
        print(f"{' ':12}  Estimated: {size_kb:5.1f}KB (float32), {quantized_size_kb:5.1f}KB (int8 quantized)")