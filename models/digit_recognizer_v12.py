# models/digit_recognizer_v12.py
import tensorflow as tf
import parameters as params

# Check for QAT compatibility
try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
    print(f"QAT available: TF {tf.__version__}, TFMo {tfmot.__version__}")
except ImportError as e:
    QAT_AVAILABLE = False
    print(f"QAT not available: {e}")

def residual_block(x, filters, kernel_size=(3, 3), strides=1):
    """A residual block."""
    shortcut = x
    
    y = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    
    y = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    # Adjust shortcut if needed
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
    y = tf.keras.layers.add([shortcut, y])
    y = tf.keras.layers.ReLU()(y)
    return y

# def create_digit_recognizer_v12():
    # """
    # Deeper digit recognizer v12 with residual connections and batch normalization.
    # This model is adaptive to the number of input channels.
    # """
    # inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # # Initial Conv layer
    # x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    # # Residual blocks
    # x = residual_block(x, filters=64, strides=1)
    # x = residual_block(x, filters=64, strides=1)
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)

    # x = residual_block(x, filters=128, strides=1)
    # x = residual_block(x, filters=128, strides=1)
    
    # # Global pooling
    # x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # # Dropout for regularization
    # x = tf.keras.layers.Dropout(0.5)(x)
    
    # # Dense layer
    # x = tf.keras.layers.Dense(128, activation='relu', name='feature_dense')(x)
    
    # # Output layer
    # outputs = tf.keras.layers.Dense(
        # params.NB_CLASSES, 
        # activation='softmax', 
        # name='output'
    # )(x)

    # model_name = f"digit_recognizer_v12"
    # return tf.keras.Model(inputs, outputs, name=model_name)
    
    
def create_digit_recognizer_v12():
    """
    Optimized version of V12 with better capacity balancing
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Initial Conv layer
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', 
                              kernel_initializer='he_normal', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    # Residual blocks with better capacity progression
    x = residual_block(x, filters=48, strides=1)  # Increased from 32→48
    x = residual_block(x, filters=48, strides=1)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)

    x = residual_block(x, filters=96, strides=1)   # Reduced from 128→96
    x = residual_block(x, filters=96, strides=1)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Less aggressive dropout
    x = tf.keras.layers.Dropout(0.3)(x)  # Reduced from 0.5
    
    # Better sized dense layer
    x = tf.keras.layers.Dense(64, activation='relu', name='feature_dense')(x)  # Reduced from 128
    x = tf.keras.layers.Dropout(0.2)(x)  # Additional light dropout
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v12_optimized")

def create_qat_model(base_model=None):
    """
    Create QAT model with TF 2.20 compatibility
    """
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model if base_model else create_digit_recognizer_v12()
    
    if base_model is None:
        base_model = create_digit_recognizer_v12()
    
    try:
        # For TF 2.20, use the newer QAT API
        quantize_annotate = tfmot.quantization.keras.quantize_annotate
        quantize_apply = tfmot.quantization.keras.quantize_apply
        quantize_scope = tfmot.quantization.keras.quantize_scope
        
        # Annotate the model for quantization
        with quantize_scope():
            annotated_model = quantize_annotate(base_model)
            
            # Apply quantization
            qat_model = quantize_apply(
                annotated_model,
                tfmot.experimental.combine.Default8BitClusterPreset()
            )
            
        print("Successfully created QAT model")
        return qat_model
        
    except Exception as e:
        print(f"QAT failed: {e}")
        print("Falling back to base model")
        return base_model

if __name__ == "__main__":
    # Test the main function
    model = create_digit_recognizer_v12()
    print(f"Created model: {model.name}")
    model.summary()
