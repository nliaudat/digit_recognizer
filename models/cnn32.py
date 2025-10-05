# models/cnn32.py
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
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Second conv block  
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Third conv block
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
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