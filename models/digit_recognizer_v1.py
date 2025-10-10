# models/digit_recognizer_v1.py
import tensorflow as tf
import parameters as params

def create_digit_recognizer_v1():
    """
    Best ESP-DL compatible model with high accuracy and optimized for QAT
    - Regular ReLU (ESP-DL compatible)
    - BatchNormalization folds into Conv layers during QAT
    - Optimized architecture for best accuracy/size tradeoff
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE, name='input'),
        
        # Block 1: Conv + BN + ReLU
        tf.keras.layers.Conv2D(
            32, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=False,  # No bias when followed by BN (folds during QAT)
            name='conv1'
        ),
        tf.keras.layers.BatchNormalization(name='bn1'),
        tf.keras.layers.ReLU(name='relu1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Block 2: Depthwise separable + BN + ReLU
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            use_bias=False,  # No bias when followed by BN
            name='depthwise1'
        ),
        tf.keras.layers.BatchNormalization(name='bn_depthwise1'),
        tf.keras.layers.ReLU(name='relu2'),
        tf.keras.layers.Conv2D(
            64, (1, 1),
            kernel_initializer='he_normal',
            use_bias=False,  # No bias when followed by BN
            name='pointwise1'
        ),
        tf.keras.layers.BatchNormalization(name='bn_pointwise1'),
        tf.keras.layers.ReLU(name='relu3'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Block 3: Enhanced features
        tf.keras.layers.Conv2D(
            128, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=False,  # No bias when followed by BN
            name='conv2'
        ),
        tf.keras.layers.BatchNormalization(name='bn2'),
        tf.keras.layers.ReLU(name='relu4'),
        
        # Global features
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        
        # Classification head
        tf.keras.layers.Dense(64, activation='relu', name='dense1'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model