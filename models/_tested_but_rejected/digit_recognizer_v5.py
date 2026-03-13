# models/digit_recognizer_v5.py
"""
digit_recognizer_v5 – Class-Count Adaptive CNN (10 vs 100 Classes)
===================================================================
Design goal: Dynamically scale the model's capacity based on NB_CLASSES —
compact for 10-class digit recognition, progressively larger for 100-class
two-digit recognition.

Architecture (grayscale, 10-class):
  - Conv2D(24→36→48) + ReLU6 + MaxPool after first block
  - GlobalAveragePooling2D
  - Dense(64) + ReLU6 → Softmax

Architecture (grayscale, 100-class):
  - Conv2D(32→48→64→72) + ReLU6 progression (extra layer for ≥50 classes)
  - Dense(128) + Dense(96) two-stage classification head

Architecture (RGB):
  - SeparableConv2D entry for RGB channel efficiency
  - Same class-adaptive filter/dense scaling

Helpers:
  - get_filters(base_10, base_100)     → picks filter count by NB_CLASSES
  - get_dense_units(base_10, base_100) → picks Dense size by NB_CLASSES

Notes:
  - No QAT wrapper (use v12+ for residual + QAT)
  - Dropout scaled from 0.1 (10-class) to 0.2 (100-class)

Estimated: ~55–130K parameters depending on class count and variant.
"""

import tensorflow as tf
import parameters as params

def create_digit_recognizer_v5():
    """
    Digit recognizer v5 optimized for 20x32 input and variable class counts
    Key optimizations:
    - Adjusted capacity based on NB_CLASSES (10 vs 100)
    - Optimized dense layers for classification complexity
    - Maintains 20x32 spatial optimizations
    - Key Class-Count Optimizations:
        For NB_CLASSES = 10 (0-9):
            Fewer filters: 24→36→48 instead of 32→48→64
            Smaller dense layers: 64 units instead of 128
            Simpler architecture: Fewer layers, more compact
            Lower dropout: 0.1 instead of 0.2
        For NB_CLASSES = 100 (0-99):
            More filters: 32→48→64→72 progression
            Additional conv layer: Better feature abstraction
            Larger dense layers: 128+ units for complex separation
            Multiple dense layers: Better classification hierarchy
            Higher dropout: 0.2 for regularization

    """
    input_channels = params.INPUT_SHAPE[-1]
    
    if input_channels == 1:
        print(f"Creating 20x32-optimized grayscale model for {params.NB_CLASSES} classes")
        return create_digit_recognizer_v5_grayscale()
    elif input_channels == 3:
        print(f"Creating 20x32-optimized RGB model for {params.NB_CLASSES} classes")
        return create_digit_recognizer_v5_rgb()
    else:
        print(f"Warning: Unsupported channel count {input_channels}. Using adaptive model.")
        return create_digit_recognizer_v5_adaptive()

def create_digit_recognizer_v5_grayscale():
    """
    Optimized for 20x32 grayscale with class-count awareness
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # ===== CONVOLUTIONAL BACKBONE (adjusted for 20x32) =====
    # Layer 1
    x = tf.keras.layers.Conv2D(
        get_filters(24, 32),  # Adjust based on class count
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv1'
    )(inputs)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    # After pool: 10x16xfilters
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        get_filters(36, 48),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    # Current: 10x16xfilters
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        get_filters(48, 64),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    # Current: 10x16xfilters
    
    # Layer 4: More important for 100-class discrimination
    if params.NB_CLASSES >= 50:
        x = tf.keras.layers.Conv2D(
            get_filters(56, 72),
            (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='conv5'
        )(x)
        x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_4')(x)
    
    # ===== FEATURE REDUCTION AND CLASSIFICATION =====
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer size scales with classification complexity
    dense_units = get_dense_units(64, 128)  # More units for 100 classes
    x = tf.keras.layers.Dense(dense_units, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    # Optional: Additional dense layer for complex classification
    if params.NB_CLASSES >= 50:
        dense_units_2 = get_dense_units(48, 96)
        x = tf.keras.layers.Dense(dense_units_2, activation='relu', name='feature_dense2')(x)
        x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense2')(x)
    
    # Optional dropout (higher for more classes to prevent overfitting)
    dropout_rate = 0.1 if params.NB_CLASSES <= 10 else 0.2
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v5_grayscale_{params.NB_CLASSES}classes")

def create_digit_recognizer_v5_rgb():
    """
    Optimized for 20x32 RGB with class-count awareness
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1: Depthwise separable
    x = tf.keras.layers.SeparableConv2D(
        get_filters(28, 36),
        (3, 3), padding='same',
        depthwise_initializer='he_normal',
        pointwise_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='sep_conv1'
    )(inputs)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        get_filters(42, 56),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        get_filters(56, 72),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    # Layer 4: For 100-class discrimination
    if params.NB_CLASSES >= 50:
        x = tf.keras.layers.Conv2D(
            get_filters(64, 80),
            (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='conv5'
        )(x)
        x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layers scale with class count
    dense_units = get_dense_units(72, 144)
    x = tf.keras.layers.Dense(dense_units, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    if params.NB_CLASSES >= 50:
        x = tf.keras.layers.Dense(get_dense_units(64, 96), activation='relu', name='feature_dense2')(x)
        x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense2')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v5_rgb_{params.NB_CLASSES}classes")

def create_digit_recognizer_v5_compact():
    """
    Ultra-compact version with class-aware optimization
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    input_channels = params.INPUT_SHAPE[-1]
    
    # Layer 1
    if input_channels == 1:
        x = tf.keras.layers.Conv2D(
            get_filters(16, 20),
            (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='conv1'
        )(inputs)
    else:
        x = tf.keras.layers.SeparableConv2D(
            get_filters(20, 24),
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            pointwise_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='sep_conv1'
        )(inputs)
    
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        get_filters(24, 32),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        get_filters(32, 40),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    # Optional layer for 100 classes
    if params.NB_CLASSES >= 50:
        x = tf.keras.layers.Conv2D(
            get_filters(40, 48),
            (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='conv5'
        )(x)
        x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Compact but class-aware dense layer
    dense_units = get_dense_units(48, 64)
    x = tf.keras.layers.Dense(dense_units, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v5_compact_{params.NB_CLASSES}classes")

def get_filters(base_10_class, base_100_class):
    """
    Return appropriate filter count based on number of classes
    """
    if params.NB_CLASSES <= 10:
        return base_10_class
    elif params.NB_CLASSES <= 20:
        return int((base_10_class + base_100_class) / 2)
    else:
        return base_100_class

def get_dense_units(base_10_class, base_100_class):
    """
    Return appropriate dense units based on number of classes
    """
    if params.NB_CLASSES <= 10:
        return base_10_class
    elif params.NB_CLASSES <= 20:
        return int((base_10_class + base_100_class) / 2)
    else:
        return base_100_class

def get_recommended_model(priority='balanced'):
    """
    Get recommended model based on priority and class count
    """
    if params.NB_CLASSES <= 10:
        # Smaller models sufficient for 10 classes
        priorities = {
            'size': create_digit_recognizer_v5_compact,
            'balanced': create_digit_recognizer_v5,
            'accuracy': create_digit_recognizer_v5
        }
    else:
        # Need more capacity for 100 classes
        priorities = {
            'size': create_digit_recognizer_v5,
            'balanced': create_digit_recognizer_v5,
            'accuracy': create_digit_recognizer_v5_high_accuracy
        }
    
    return priorities.get(priority, create_digit_recognizer_v5)()

def create_digit_recognizer_v5_high_accuracy():
    """
    High-accuracy version for complex classification (100 classes)
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    input_channels = params.INPUT_SHAPE[-1]
    
    # High-capacity backbone for 100-class discrimination
    if input_channels == 1:
        x = tf.keras.layers.Conv2D(
            get_filters(32, 48),
            (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='conv1'
        )(inputs)
    else:
        x = tf.keras.layers.SeparableConv2D(
            get_filters(36, 52),
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            pointwise_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='sep_conv1'
        )(inputs)
    
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Multiple conv layers for complex features
    x = tf.keras.layers.Conv2D(
        get_filters(48, 64),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    
    x = tf.keras.layers.Conv2D(
        get_filters(64, 80),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    x = tf.keras.layers.Conv2D(
        get_filters(72, 96),
        (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv5'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Larger dense layers for 100-class separation
    x = tf.keras.layers.Dense(get_dense_units(96, 160), activation='relu', name='feature_dense1')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense1')(x)
    
    x = tf.keras.layers.Dense(get_dense_units(64, 112), activation='relu', name='feature_dense2')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense2')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v5_high_acc_{params.NB_CLASSES}classes")

def analyze_class_optimization():
    """
    Analyze how the model adapts to different class counts
    """
    print(f"=== CLASS COUNT OPTIMIZATION ANALYSIS ===")
    print(f"Current NB_CLASSES: {params.NB_CLASSES}")
    print(f"Input shape: {params.INPUT_SHAPE}")
    
    if params.NB_CLASSES <= 10:
        print("Optimization: Compact architecture for simple 10-class problem")
        print("- Fewer convolutional filters")
        print("- Smaller dense layers")
        print("- Simpler feature hierarchy")
    elif params.NB_CLASSES <= 50:
        print("Optimization: Balanced architecture for medium complexity")
        print("- Moderate filter counts")
        print("- Additional convolutional layer")
        print("- Larger dense layers")
    else:
        print("Optimization: Enhanced architecture for complex 100-class problem")
        print("- More convolutional filters")
        print("- Multiple dense layers")
        print("- Higher capacity throughout")

if __name__ == "__main__":
    # Test the main function
    model = create_digit_recognizer_v5()
    print(f"Created model: {model.name}")
    model.summary()
    
    # Analyze optimizations
    analyze_class_optimization()
    
    # Compare different versions
    print("\n=== MODEL COMPARISON ===")
    versions = {
        'Compact': create_digit_recognizer_v5_compact(),
        'Balanced': create_digit_recognizer_v5(),
        'High Accuracy': create_digit_recognizer_v5_high_accuracy()
    }
    
    for name, model in versions.items():
        print(f"{name}: {model.count_params():,} parameters")