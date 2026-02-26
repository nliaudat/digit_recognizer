# models/cnn32.py
"""
cnn32 / CNN_s2 – Exact Haverland Original Reference Model
==========================================================
Design goal: Faithful replica of the `CNN_s2` model from Haverland's
`b2n.models.cnn32` notebook for the Tenth-of-step meter-digit project.
Used as the primary reference baseline to compare all custom models against.

Reference:
  https://github.com/haverland/Tenth-of-step-of-a-meter-digit/blob/master/dig-class100-s2.ipynb

Architecture:
  - BN → Conv2D(32, 3×3) + BN + ReLU + MaxPool + Dropout(0.2)   → Block 1
  - Conv2D(64, 3×3) + BN + ReLU + MaxPool + Dropout(0.2)         → Block 2
  - Conv2D(64, 3×3) + BN + ReLU + MaxPool                        → Block 3
  - Flatten → Dropout(0.4) → Dense(256) + ReLU + Dropout(0.4)
  - Dense(NB_CLASSES, softmax)

Notes:
  - Input normalization via first BN layer (replaces image preprocessing)
  - Uses Flatten + Dense(256) head — large; not quantization optimal
  - Configurable activation_top (None = logits, 'softmax' = probabilities)
  - Wrapped as create_original_haverland() for factory compatibility

Estimated: ~200–300K parameters → reference only, not intended for ESP32.
"""

import tensorflow as tf
import parameters as params

def CNN_s2(input_shape=None, nb_classes=params.NB_CLASSES, activation_top='softmax'):
    """
    Exact replica of the CNN_s2 model from Haverland's b2n.models.cnn32
    This matches the original implementation exactly.
    """
    if input_shape is None:
        input_shape = params.INPUT_SHAPE
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # First conv block
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x) # Processes all channels together, similar to convert to grayscale
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Second conv block  
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Third conv block
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Output layer
    if activation_top is None:
        outputs = tf.keras.layers.Dense(nb_classes)(x)
    else:
        outputs = tf.keras.layers.Dense(nb_classes, activation=activation_top)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_original_haverland():
    """Wrapper to match our factory pattern"""
    return CNN_s2(input_shape=params.INPUT_SHAPE, nb_classes=params.NB_CLASSES, activation_top='softmax')