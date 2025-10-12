# models/digit_recognizer_v5.py
import tensorflow as tf
import parameters as params

def create_digit_recognizer_v5():
    """
    Digit recognizer v5 optimized for 20x32 input dimensions
    Key optimizations for 20x32:
    - Adjusted pooling strategy for rectangular aspect ratio
    - Optimized filter counts for smaller spatial dimensions
    - Maintains feature resolution through careful layer design
    """
    input_channels = params.INPUT_SHAPE[-1]
    
    if input_channels == 1:
        print("Creating 20x32-optimized grayscale model (v5_grayscale)")
        return create_digit_recognizer_v5_grayscale()
    elif input_channels == 3:
        print("Creating 20x32-optimized RGB model (v5_rgb)")
        return create_digit_recognizer_v5_rgb()
    else:
        print(f"Warning: Unsupported channel count {input_channels}. Using adaptive model.")
        return create_digit_recognizer_v5_adaptive()

def create_digit_recognizer_v5_grayscale():
    """
    Optimized for 20x32 grayscale images
    - Fewer pooling layers to preserve spatial information
    - Adjusted filter counts for smaller input
    - Maintains good feature resolution
    """
    def relu6(x, name=None):
        return tf.clip_by_value(tf.nn.relu(x), 0.0, 6.0, name=name)

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1: Moderate filters for 20x32 input
    x = tf.keras.layers.Conv2D(
        24, (3, 3), padding='same',  # Increased from v3 for 20x32
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1_24f'
    )(inputs)
    x = relu6(x, name='relu6_1')
    # Single pooling to reduce 32→16 height while keeping reasonable width (20→10)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    # After pool: 10x16x24
    
    # Layer 2: Feature extraction at reduced resolution
    x = tf.keras.layers.Conv2D(
        36, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2_36f'
    )(x)
    x = relu6(x, name='relu6_2')
    # No pooling here to preserve spatial information for 10x16
    # Current: 10x16x36
    
    # Layer 3: Deeper feature abstraction
    x = tf.keras.layers.Conv2D(
        48, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3_48f'
    )(x)
    x = relu6(x, name='relu6_3')
    # Current: 10x16x48
    
    # Layer 4: Final feature refinement
    x = tf.keras.layers.Conv2D(
        56, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv5_56f'
    )(x)
    x = relu6(x, name='relu6_4')
    # Current: 10x16x56
    
    # Global pooling reduces 10x16 to 1x1x56
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer for classification
    x = tf.keras.layers.Dense(64, activation=None, name='feature_dense')(x)
    x = relu6(x, name='relu6_dense')
    
    # Optional dropout (comment out for final quantization)
    x = tf.keras.layers.Dropout(0.1, name='dropout')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v5_grayscale_20x32")

def create_digit_recognizer_v5_rgb():
    """
    Optimized for 20x32 RGB images
    - Depthwise separable convolutions for efficiency
    - Adjusted architecture for rectangular aspect ratio
    """
    def relu6(x, name=None):
        return tf.clip_by_value(tf.nn.relu(x), 0.0, 6.0, name=name)

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1: Depthwise separable for RGB efficiency
    x = tf.keras.layers.SeparableConv2D(
        28, (3, 3), padding='same',  # Increased for 20x32
        depthwise_initializer='he_normal',
        pointwise_initializer='he_normal',
        use_bias=True,
        name='sep_conv1_28f'
    )(inputs)
    x = relu6(x, name='relu6_1')
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    # After pool: 10x16x28
    
    # Layer 2: Regular convolution
    x = tf.keras.layers.Conv2D(
        42, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2_42f'
    )(x)
    x = relu6(x, name='relu6_2')
    # Current: 10x16x42
    
    # Layer 3: Feature abstraction
    x = tf.keras.layers.Conv2D(
        56, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3_56f'
    )(x)
    x = relu6(x, name='relu6_3')
    # Current: 10x16x56
    
    # Layer 4: Bottleneck with more filters
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv5_64f'
    )(x)
    x = relu6(x, name='relu6_4')
    # Current: 10x16x64
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer
    x = tf.keras.layers.Dense(72, activation=None, name='feature_dense')(x)
    x = relu6(x, name='relu6_dense')
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v5_rgb_20x32")

def create_digit_recognizer_v5_compact():
    """
    Ultra-compact version for 20x32 - maximum efficiency
    For deployment where model size is critical
    """
    def relu6(x, name=None):
        return tf.clip_by_value(tf.nn.relu(x), 0.0, 6.0, name=name)

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    input_channels = params.INPUT_SHAPE[-1]
    
    # Layer 1: Minimal but sufficient for 20x32
    if input_channels == 1:
        x = tf.keras.layers.Conv2D(
            16, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv1_16f'
        )(inputs)
    else:
        x = tf.keras.layers.SeparableConv2D(
            20, (3, 3), padding='same',
            depthwise_initializer='he_normal',
            pointwise_initializer='he_normal',
            use_bias=True,
            name='sep_conv1_20f'
        )(inputs)
    
    x = relu6(x, name='relu6_1')
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    # After pool: 10x16x16 or 10x16x20
    
    # Layer 2: Moderate increase
    x = tf.keras.layers.Conv2D(
        24, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2_24f'
    )(x)
    x = relu6(x, name='relu6_2')
    # Current: 10x16x24
    
    # Layer 3: Final features
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3_32f'
    )(x)
    x = relu6(x, name='relu6_3')
    # Current: 10x16x32
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Compact dense layer
    x = tf.keras.layers.Dense(48, activation=None, name='feature_dense')(x)
    x = relu6(x, name='relu6_dense')
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v5_compact_20x32")

def create_digit_recognizer_v5_high_accuracy():
    """
    High-accuracy version for 20x32 when model size is less critical
    """
    def relu6(x, name=None):
        return tf.clip_by_value(tf.nn.relu(x), 0.0, 6.0, name=name)

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    input_channels = params.INPUT_SHAPE[-1]
    
    # Layer 1: More capacity for 20x32
    if input_channels == 1:
        x = tf.keras.layers.Conv2D(
            32, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv1_32f'
        )(inputs)
    else:
        x = tf.keras.layers.SeparableConv2D(
            36, (3, 3), padding='same',
            depthwise_initializer='he_normal',
            pointwise_initializer='he_normal',
            use_bias=True,
            name='sep_conv1_36f'
        )(inputs)
    
    x = relu6(x, name='relu6_1')
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    # After pool: 10x16x32 or 10x16x36
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        48, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2_48f'
    )(x)
    x = relu6(x, name='relu6_2')
    # Current: 10x16x48
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3_64f'
    )(x)
    x = relu6(x, name='relu6_3')
    # Current: 10x16x64
    
    # Layer 4: Additional capacity
    x = tf.keras.layers.Conv2D(
        72, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv5_72f'
    )(x)
    x = relu6(x, name='relu6_4')
    # Current: 10x16x72
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Larger dense layer
    x = tf.keras.layers.Dense(96, activation=None, name='feature_dense')(x)
    x = relu6(x, name='relu6_dense')
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v5_high_acc_20x32")

def analyze_20x32_architecture():
    """
    Analyze the spatial dimensions through the network for 20x32 input
    """
    print("=== 20x32 ARCHITECTURE ANALYSIS ===")
    print("Input: 20x32 (width x height)")
    print("After Conv1 + Pool1 (2x2 stride): 10x16")
    print("Final before Global Pooling: 10x16")
    print("Global Average Pooling: 10x16 → 1x1")
    print("\nAdvantages for 20x32:")
    print("- Single pooling preserves adequate spatial resolution")
    print("- 10x16 final features provide good receptive field")
    print("- Balanced aspect ratio throughout network")

def get_recommended_model(priority='balanced'):
    """
    Get recommended model based on priority
    """
    priorities = {
        'size': create_digit_recognizer_v5_compact,
        'balanced': create_digit_recognizer_v5,
        'accuracy': create_digit_recognizer_v5_high_accuracy
    }
    
    if priority in priorities:
        return priorities[priority]()
    else:
        print(f"Unknown priority '{priority}'. Using balanced.")
        return create_digit_recognizer_v5()

if __name__ == "__main__":
    # Test the main function
    model = create_digit_recognizer_v5()
    print(f"Created model: {model.name}")
    model.summary()
    
    # Analyze architecture
    analyze_20x32_architecture()
    
    # Compare different versions
    print("\n=== MODEL COMPARISON ===")
    versions = {
        'Compact': create_digit_recognizer_v5_compact(),
        'Balanced': create_digit_recognizer_v5(),
        'High Accuracy': create_digit_recognizer_v5_high_accuracy()
    }
    
    for name, model in versions.items():
        print(f"{name}: {model.count_params():,} parameters")