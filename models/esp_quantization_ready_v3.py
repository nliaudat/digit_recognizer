# models/esp_quantization_ready_v3.py
import tensorflow as tf
import parameters as params

def create_esp_quantization_ready_v3():
    """
    Improved version with better training characteristics
    - More filters for better feature extraction
    - L2 regularization to prevent overfitting
    - Dropout for better generalization
    - Still quantization-friendly
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv layer - increased filters
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            use_bias=True,
            name='conv1'
        ),
        tf.keras.layers.ReLU(name='relu1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second conv layer
        tf.keras.layers.Conv2D(
            128, (3, 3), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            use_bias=True,
            name='conv2'
        ),
        tf.keras.layers.ReLU(name='relu2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third conv layer - depthwise separable for efficiency
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            depthwise_regularizer=tf.keras.regularizers.l2(0.001),
            use_bias=True,
            name='depthwise1'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu1'),
        tf.keras.layers.Conv2D(
            128, (1, 1),
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            use_bias=True,
            name='pointwise1'
        ),
        tf.keras.layers.ReLU(name='relu3'),
        
        # Global average pooling
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        
        # Dropout for regularization (removed during quantization)
        tf.keras.layers.Dropout(0.2, name='dropout1'),
        
        # Output layer
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model