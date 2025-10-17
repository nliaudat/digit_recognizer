# models/digit_recognizer_v3.py
import tensorflow as tf
import parameters as params

def create_digit_recognizer_v3():
    """
    Smart router function that automatically selects the appropriate model
    based on INPUT_SHAPE channels
    """
    input_channels = params.INPUT_SHAPE[-1]
    
    if input_channels == 1:
        print("Creating optimized grayscale model (v3_grayscale)")
        return create_digit_recognizer_v3_grayscale()
    elif input_channels == 3:
        print("Creating optimized RGB model (v3_rgb)")
        return create_digit_recognizer_v3_rgb()
    else:
        print(f"Warning: Unsupported channel count {input_channels}. Using adaptive model.")
        return create_digit_recognizer_v3_adaptive()

def create_digit_recognizer_v3_grayscale():
    """
    Highly optimized version for grayscale images (1 channel)
    - Minimal filters for simple digit features
    - Compact architecture for embedded deployment
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1: Minimal filters for grayscale
    x = tf.keras.layers.Conv2D(
        8, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv1_8f'
    )(inputs)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2: Moderate increase
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2_16f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3: Feature abstraction
    x = tf.keras.layers.Conv2D(
        24, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3_24f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    # Global pooling for robust feature reduction
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Small dense layer for better separation
    x = tf.keras.layers.Dense(32, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v3_grayscale")

def create_digit_recognizer_v3_rgb():
    """
    Optimized version for RGB images (3 channels)
    - Depthwise separable convolutions for efficiency
    - Slightly more capacity for color information
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1: Depthwise separable for RGB efficiency
    x = tf.keras.layers.SeparableConv2D(
        16, (3, 3), padding='same',
        depthwise_initializer='he_normal',
        pointwise_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='sep_conv1_16f'
    )(inputs)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2: Regular convolution
    x = tf.keras.layers.Conv2D(
        24, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2_24f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3: Feature abstraction
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3_32f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    # Optional bottleneck layer for richer color features
    x = tf.keras.layers.Conv2D(
        48, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv4_bottleneck'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer
    x = tf.keras.layers.Dense(48, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v3_rgb")

def create_digit_recognizer_v3_adaptive():
    """
    Adaptive version for any channel count
    - Uses depthwise separable for multi-channel, regular conv for single channel
    - Good for flexibility but less optimized than specialized versions
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    input_channels = params.INPUT_SHAPE[-1]
    
    # Layer 1: Adaptive based on channel count
    if input_channels == 1:
        x = tf.keras.layers.Conv2D(
            12, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            activation='relu',
            name='conv1_adaptive'
        )(inputs)
    else:
        x = tf.keras.layers.SeparableConv2D(
            16, (3, 3), padding='same',
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
        24, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2_24f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3_32f'
    )(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_3')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer
    x = tf.keras.layers.Dense(40, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.Activation(tf.nn.relu6, name='relu6_dense')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v3_adaptive")

def compare_models():
    """Compare all three model variants"""
    print("=== DIGIT RECOGNIZER V3 MODEL COMPARISON ===")
    print(f"Current INPUT_SHAPE: {params.INPUT_SHAPE}")
    print(f"Number of classes: {params.NB_CLASSES}")
    print()
    
    # Create and compare all models
    models = {
        'Grayscale': create_digit_recognizer_v3_grayscale(),
        'RGB': create_digit_recognizer_v3_rgb(),
        'Adaptive': create_digit_recognizer_v3_adaptive()
    }
    
    for name, model in models.items():
        print(f"--- {name} Model ---")
        print(f"Total parameters: {model.count_params():,}")
        print(f"Layers: {len(model.layers)}")
        print()

if __name__ == "__main__":
    # Test the main function
    model = create_digit_recognizer_v3()
    print(f"Created model: {model.name}")
    model.summary()
    
    # Compare all variants
    compare_models()