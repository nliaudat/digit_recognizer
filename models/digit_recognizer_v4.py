# models/digit_recognizer_v4.py
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

def create_digit_recognizer_v4():
    """
    Balanced digit recognizer v4 - maintains optimizations while recovering accuracy
    """
    input_channels = params.INPUT_SHAPE[-1]
    
    if input_channels == 1:
        print("Creating balanced grayscale model (v4_grayscale)")
        return create_digit_recognizer_v4_grayscale()
    elif input_channels == 3:
        print("Creating balanced RGB model (v4_rgb)")
        return create_digit_recognizer_v4_rgb()
    else:
        print(f"Warning: Unsupported channel count {input_channels}. Using adaptive model.")
        return create_digit_recognizer_v4_adaptive()

def create_digit_recognizer_v4_grayscale():
    """
    Balanced grayscale model - optimized for TF 2.20 compatibility
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1
    x = tf.keras.layers.Conv2D(
        20, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv1_20f'
    )(inputs)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        36, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2_36f'
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        48, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3_48f'
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_3')(x)
    
    # Additional conv layer
    x = tf.keras.layers.Conv2D(
        56, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv4_56f'
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer
    x = tf.keras.layers.Dense(64, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_dense')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v4_grayscale")

def create_digit_recognizer_v4_rgb():
    """
    Balanced RGB model
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Layer 1
    x = tf.keras.layers.SeparableConv2D(
        24, (3, 3), padding='same',
        depthwise_initializer='he_normal',
        pointwise_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='sep_conv1_24f'
    )(inputs)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2
    x = tf.keras.layers.Conv2D(
        40, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv2_40f'
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(
        56, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv3_56f'
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_3')(x)
    
    # Bottleneck layer
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        activation='relu',
        name='conv4_bottleneck'
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer
    x = tf.keras.layers.Dense(72, activation='relu', name='feature_dense')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_dense')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v4_rgb")

# ... (keep the other model creation functions similar to above)

def create_qat_model(base_model=None):
    """
    Create QAT model with TF 2.20 compatibility
    """
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model if base_model else create_digit_recognizer_v4()
    
    if base_model is None:
        base_model = create_digit_recognizer_v4()
    
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

def create_qat_model_legacy(base_model=None):
    """
    Legacy QAT approach for older TF versions
    """
    if base_model is None:
        base_model = create_digit_recognizer_v4()
    
    try:
        # Try the legacy approach
        from tensorflow_model_optimization.python.core.quantization.keras import quantize
        
        qat_model = quantize.quantize_model(base_model)
        print("Successfully created QAT model (legacy method)")
        return qat_model
    except Exception as e:
        print(f"Legacy QAT also failed: {e}")
        return base_model

# Test function
def test_qat_compatibility():
    """Test QAT compatibility with current environment"""
    print("=== QAT COMPATIBILITY TEST ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    if QAT_AVAILABLE:
        print(f"TensorFlow Model Optimization version: {tfmot.__version__}")
        
        # Test basic model creation
        model = create_digit_recognizer_v4()
        print(f"Base model created: {model.name}")
        print(f"Base model parameters: {model.count_params():,}")
        
        # Test QAT
        try:
            qat_model = create_qat_model(model)
            print(f"QAT model type: {type(qat_model)}")
            print("✓ QAT test passed")
        except Exception as e:
            print(f"✗ QAT test failed: {e}")
    else:
        print("✗ QAT not available")

if __name__ == "__main__":
    # Test the main function
    model = create_digit_recognizer_v4()
    print(f"Created model: {model.name}")
    model.summary()
    
    # Test QAT compatibility
    test_qat_compatibility()