# models/dig_class100_s2.py
import tensorflow as tf
import parameters as params

def create_dig_class100_s2():
    """
    Model architecture from the referenced notebook:
    https://github.com/haverland/Tenth-of-step-of-a-meter-digit/blob/master/dig-class100-s2.ipynb
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv block
        tf.keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[0], (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[0], (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(params.DROPOUT_RATE),
        
        # Second conv block
        tf.keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[1], (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[1], (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(params.DROPOUT_RATE),
        
        # Third conv block
        tf.keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[2], (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[2], (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(params.DROPOUT_RATE),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(params.DIG_CLASS100_DENSE_UNITS, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params.DROPOUT_RATE),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax')
    ])
    
    return model