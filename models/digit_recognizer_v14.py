# models/digit_recognizer_v14.py
import tensorflow as tf
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError as e:
    QAT_AVAILABLE = False
    print(f"QAT not available: {e}")

def bottleneck_block(x, filters, reduction_ratio=0.25, strides=1):
    """
    Ultra-compressed ESP32-optimized Bottleneck block.
    """
    shortcut = x
    reduced_filters = max(int(filters * reduction_ratio), 8)
    
    # Compress channels
    y = tf.keras.layers.Conv2D(reduced_filters, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    
    # Process features
    y = tf.keras.layers.Conv2D(reduced_filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    
    # Expand channels
    y = tf.keras.layers.Conv2D(filters, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    # Identity shortcut
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
    y = tf.keras.layers.add([shortcut, y])
    y = tf.keras.layers.ReLU()(y)
    return y

def create_digit_recognizer_v14():
    """
    Ultra-Tiny variant pushing the absolute Pareto frontier for ESP32 constraints.
    Focuses purely on the minimal possible parameter count utilizing global pooling.
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Initial Conv layer - minimal filters
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    # Minimal Bottleneck Blocks - Stage 1
    x = bottleneck_block(x, filters=16, reduction_ratio=0.5, strides=1)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    # Minimal Bottleneck Blocks - Stage 2
    x = bottleneck_block(x, filters=32, reduction_ratio=0.25, strides=1)
    
    # Global Average Pooling directly to logits
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # No intermediate dense layer! Maps directly to output.
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v14")

def create_qat_model(base_model=None):
    """
    Create QAT model using explicit annotations compatible with ESP-DL.
    """
    if not QAT_AVAILABLE:
        print("Warning: QAT not available for v14. Returning base model.")
        return base_model if base_model else create_digit_recognizer_v14()
    
    if base_model is None:
        base_model = create_digit_recognizer_v14()
    
    try:
        quantize_annotate = tfmot.quantization.keras.quantize_annotate
        quantize_apply = tfmot.quantization.keras.quantize_apply
        quantize_scope = tfmot.quantization.keras.quantize_scope
        
        with quantize_scope():
            annotated_model = quantize_annotate(base_model)
            qat_model = quantize_apply(
                annotated_model,
                tfmot.experimental.combine.Default8BitClusterPreset()
            )
            
        return qat_model
        
    except Exception as e:
        print(f"QAT failed on v14: {e}")
        return base_model

if __name__ == "__main__":
    model = create_digit_recognizer_v14()
    print(f"Created model: {model.name}")
    model.summary()
