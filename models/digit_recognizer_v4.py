# models/digit_recognizer_v4.py
import tensorflow as tf
import parameters as params

def create_digit_recognizer_v4():
    """
    Balanced digit recognizer v4 - maintains optimizations while recovering accuracy
    Key improvements:
    - More gradual filter progression
    - Balanced capacity for grayscale
    - Maintains quantization compatibility
    """
    input_channels = params.INPUT_SHAPE[-1]
    
    if input_channels == 1:
        print("Creating balanced grayscale model (v4_grayscale)")
        return create_digit_recognizer_v4_grayscale()
    elif input_channels == 3:
        print("Creating balanced RGB model (v4_rgb)")
        return create_digit_recognizer_v4_rgb()
    else:
        print(f"Warning: Unsupported channel count {input_channels}. Using adaptive model.")
        return create_digit_recognizer_v4_adaptive()

def create_digit_recognizer_v4_grayscale():
    """
    Balanced grayscale model - recovers accuracy while maintaining efficiency
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1: More filters than v3 but less than original v2
    x = tf.keras.layers.Conv2D(
        20, (3, 3), padding='same',  # Balanced: 20 vs v2(32) vs v3(8)
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv1_20f'
    )(inputs)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2: Moderate increase
    x = tf.keras.layers.Conv2D(
        36, (3, 3), padding='same',  # 36 vs v2(64) vs v3(16)
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2_36f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3: Feature abstraction (similar to v2 but optimized)
    x = tf.keras.layers.Conv2D(
        48, (3, 3), padding='same',  # 48 vs v2(64) vs v3(24)
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3_48f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    # Additional small conv for better feature extraction
    x = tf.keras.layers.Conv2D(
        56, (3, 3), padding='same',  # Small increase for final features
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv4_56f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Larger dense layer for better classification
    x = tf.keras.layers.Dense(64, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    # Add small dropout for regularization (optional, can be removed for quantization)
    x = tf.keras.layers.Dropout(0.1, name='dropout')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v4_grayscale")

def create_digit_recognizer_v4_rgb():
    """
    Balanced RGB model - maintains efficiency with good accuracy
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1: Depthwise separable with balanced filter count
    x = tf.keras.layers.SeparableConv2D(
        24, (3, 3), padding='same',  # 24 vs v3(16)
        depthwise_initializer='he_normal',
        pointwise_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='sep_conv1_24f'
    )(inputs)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2: Regular convolution
    x = tf.keras.layers.Conv2D(
        40, (3, 3), padding='same',  # 40 vs v3(24)
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2_40f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3: Feature abstraction
    x = tf.keras.layers.Conv2D(
        56, (3, 3), padding='same',  # 56 vs v3(32)
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3_56f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    # Bottleneck layer
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',  # 64 vs v3(48)
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv4_bottleneck'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer
    x = tf.keras.layers.Dense(72, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v4_rgb")

def create_digit_recognizer_v4_adaptive():
    """
    Adaptive version with balanced capacity
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    input_channels = params.INPUT_SHAPE[-1]
    
    # Layer 1: Adaptive based on channel count
    if input_channels == 1:
        x = tf.keras.layers.Conv2D(
            20, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='conv1_adaptive'
        )(inputs)
    else:
        x = tf.keras.layers.SeparableConv2D(
            24, (3, 3), padding='same',
            depthwise_initializer='he_normal',
            pointwise_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='sep_conv1_adaptive'
        )(inputs)
    
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        36, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2_36f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        48, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3_48f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    # Additional conv layer
    x = tf.keras.layers.Conv2D(
        56, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv4_56f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer
    x = tf.keras.layers.Dense(64, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v4_adaptive")

def create_digit_recognizer_v4_high_accuracy():
    """
    High-accuracy version when model size is less critical
    Similar to v2 but with quantization-friendly improvements
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    input_channels = params.INPUT_SHAPE[-1]
    
    # Layer 1: Close to original v2 capacity
    if input_channels == 1:
        x = tf.keras.layers.Conv2D(
            28, (3, 3), padding='same',  # 28 vs v2(32)
            kernel_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='conv1_28f'
        )(inputs)
    else:
        x = tf.keras.layers.SeparableConv2D(
            32, (3, 3), padding='same',
            depthwise_initializer='he_normal',
            pointwise_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='sep_conv1_32f'
        )(inputs)
    
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        56, (3, 3), padding='same',  # 56 vs v2(64)
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2_56f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',  # Same as v2
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3_64f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    # Additional layer for better features
    x = tf.keras.layers.Conv2D(
        72, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv4_72f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Larger dense layer
    x = tf.keras.layers.Dense(96, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v4_high_accuracy")

def compare_v2_v3_v4():
    """Compare model capacities across versions"""
    print("=== MODEL VERSION COMPARISON ===")
    
    # Create sample models for comparison
    from models.digit_recognizer_v2 import create_digit_recognizer_v2
    
    models = {
        'V2 (Original)': create_digit_recognizer_v2(),
        'V4 (Balanced)': create_digit_recognizer_v4(),
        'V4 (High Accuracy)': create_digit_recognizer_v4_high_accuracy()
    }
    
    for name, model in models.items():
        params_count = model.count_params()
        print(f"\n--- {name} ---")
        print(f"Total parameters: {params_count:,}")
        print(f"Model name: {model.name}")
        
        # Show layer-wise filter counts
        conv_layers = [layer for layer in model.layers if 'conv' in layer.name]
        print("Conv layers:", [f"{layer.name}: {layer.output_shape[-1]}f" for layer in conv_layers[:3]])

if __name__ == "__main__":
    # Test the main function
    model = create_digit_recognizer_v4()
    print(f"Created model: {model.name}")
    model.summary()
    
    # Compare versions
    compare_v2_v3_v4()