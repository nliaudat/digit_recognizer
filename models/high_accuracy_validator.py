# models/high_accuracy_validator.py
import tensorflow as tf
import parameters as params

def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation block."""
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    return tf.keras.layers.Multiply()([input_tensor, se])

def res_se_block(x, filters, strides=1):
    """Residual block with Squeeze-and-Excitation."""
    shortcut = x
    
    y = tf.keras.layers.Conv2D(filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    
    y = tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    # Squeeze-and-Excitation
    y = se_block(y)
    
    # Adjust shortcut if needed
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
    y = tf.keras.layers.add([shortcut, y])
    y = tf.keras.layers.ReLU()(y)
    return y

def create_high_accuracy_validator():
    """
    High-capacity ResNet architecture with SE blocks
    designed strictly for PC/Server validation.
    Ignores IoT constraints to maximize accuracy.
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Initial Conv layer
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    # Stage 1
    x = res_se_block(x, filters=64, strides=1)
    x = res_se_block(x, filters=64, strides=1)
    
    # Stage 2
    x = res_se_block(x, filters=128, strides=2)
    x = res_se_block(x, filters=128, strides=1)
    
    # Stage 3
    x = res_se_block(x, filters=256, strides=2)
    x = res_se_block(x, filters=256, strides=1)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Classifier stage with Dropout
    x = tf.keras.layers.Dense(512, activation='relu', name='fc1')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu', name='fc2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="high_accuracy_validator")

def create_qat_model(base_model=None):
    """
    This model is expressly NOT for QAT or IoT execution. 
    It returns the unquantized float32 model to prevent accuracy loss.
    """
    print("==========================================================")
    print("⚠ WARNING: 'high_accuracy_validator' skips QAT annotations")
    print("This model is built for PC-level validation only.")
    print("==========================================================")
    
    if base_model is None:
        base_model = create_high_accuracy_validator()
        
    return base_model

if __name__ == "__main__":
    model = create_high_accuracy_validator()
    print(f"Created model: {model.name}")
    model.summary()
