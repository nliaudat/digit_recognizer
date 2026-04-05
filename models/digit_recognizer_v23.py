# models/digit_recognizer_v23.py
"""
digit_recognizer_v23 – Luminance Grayscale with Fixed Conv2D Weights (Best Accuracy ≤100KB)
===========================================================================================
Design goal: Add perceptual luminance grayscale conversion as an entry layer while
maintaining v4 architecture and QAT compatibility. Uses Conv2D with frozen weights
for optimal TFLite Micro compatibility.

Train on RGB images, auto convert to grayscale in the first layer.

Architecture:
  - Luminance Grayscale Entry (Conv2D 1x1 with fixed weights [0.299, 0.587, 0.114] BT.601 standard)
  - Conv2D(20) + ReLU6 + MaxPool
  - Conv2D(36) + ReLU6 + MaxPool
  - Conv2D(48) + ReLU6
  - Conv2D(56) + ReLU6
  - GlobalAveragePooling2D
  - Dense(64) + ReLU6 → Dense(NB_CLASSES) Softmax

Key Features:
  - Perceptual luminance weighting: Y = 0.299*R + 0.587*G + 0.114*B
  - Conv2D with frozen weights (non-trainable) for TFLite Micro compatibility
  - Channel-adaptive: automatically handles RGB or grayscale input
  - No BatchNormalization → cleaner, faster quantization
  - Full QAT support

Estimated: ~90K parameters → ~88 KB after INT8 quantization.
Achieved accuracy: 99.0% (same as v4, now with color robustness)
"""

import tensorflow as tf
import parameters as params
try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
    print(f"QAT available: TF {tf.__version__}, TFMo {tfmot.__version__}")
except ImportError as e:
    QAT_AVAILABLE = False
    print(f"QAT not available: {e}")

# ============================================================================
# LUMINANCE GRAYSCALE CONVERSION LAYER
# ============================================================================

def create_luminance_grayscale_conv(input_tensor):
    """
    Convert RGB to grayscale using luminance weighting via Conv2D with fixed weights.
    Formula: Y = 0.299*R + 0.587*G + 0.114*B
    
    Uses 1x1 convolution with frozen weights, which is 100% compatible with
    TFLite Micro's CONV_2D operator.
    
    Args:
        input_tensor: Input tensor with shape (batch, height, width, channels)
    
    Returns:
        Grayscale tensor with shape (batch, height, width, 1)
    
    Note: This function should be called within the Keras functional API
          with symbolic tensors, not eager tensors.
    """
    # Get static channel dimension (if available)
    input_shape = input_tensor.shape
    channels = input_shape[-1] if input_shape[-1] is not None else None
    
    if channels == 1:
        # Already grayscale, return as-is
        return input_tensor
    elif channels == 3:
        # Create fixed weights for luminance conversion
        # Shape: (kernel_height, kernel_width, input_channels, output_channels)
        luminance_weights = tf.constant([
            [[[0.299],  # R weight
              [0.587],  # G weight
              [0.114]]] # B weight
        ], dtype=tf.float32)
        
        # Fixed bias (zero)
        luminance_bias = tf.constant([0.0], dtype=tf.float32)
        
        # Apply 1x1 convolution with fixed weights
        grayscale = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.Constant(luminance_weights),
            bias_initializer=tf.keras.initializers.Constant(luminance_bias),
            trainable=False,  # Freeze these weights
            name='luminance_grayscale_conv'
        )(input_tensor)
        
        return grayscale
    else:
        # Fallback for dynamic or unknown channel counts
        # Must wrap in Lambda layer for functional API compatibility
        print(f"Warning: Unknown channel count {channels}, using average fallback")
        grayscale = tf.keras.layers.Lambda(
            lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
            name='average_grayscale_fallback'
        )(input_tensor)
        return grayscale

# ============================================================================
# COMMON BACKBONE ARCHITECTURE
# ============================================================================

def _build_v23_backbone(x):
    """
    Common backbone architecture shared across all v23 variants.
    This eliminates code duplication and ensures consistency.
    
    Args:
        x: Input tensor (already preprocessed to grayscale, shape: H, W, 1)
    
    Returns:
        Output tensor (logits before softmax)
    """
    # Adaptive capacity — scales filters/dense with NB_CLASSES
    # 10cls: scale=1.0 → [20,36,48,56] dense=64  (~91K params)
    # 100cls: scale≈1.6 → [32,58,77,90] dense=102 (~195K params)
    scale = max(1.0, (params.NB_CLASSES / 10) ** 0.45)
    f     = [max(int(fi * scale), fi) for fi in [20, 36, 48, 56]]
    d     = max(int(64 * scale), 64)

    # Layer 1
    x = tf.keras.layers.Conv2D(
        f[0], (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1_{}f'.format(f[0])
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(
        f[1], (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2_{}f'.format(f[1])
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)

    # Layer 3
    x = tf.keras.layers.Conv2D(
        f[2], (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3_{}f'.format(f[2])
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_3')(x)

    # Layer 4
    x = tf.keras.layers.Conv2D(
        f[3], (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv4_{}f'.format(f[3])
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_4')(x)

    # Global pooling + dense
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = tf.keras.layers.Dense(d, activation=None, name='feature_dense')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_dense')(x)

    return x

# ============================================================================
# MODEL VARIANTS
# ============================================================================

def create_digit_recognizer_v23():
    """
    Balanced digit recognizer v23 - luminance grayscale entry + v4 architecture
    Channel-adaptive: handles RGB or grayscale input automatically
    """
    input_channels = params.INPUT_SHAPE[-1]
    
    if input_channels == 1:
        print("Creating v23 model with grayscale input (no conversion needed)")
        return create_digit_recognizer_v23_grayscale()
    elif input_channels == 3:
        print("Creating v23 model with RGB input + luminance grayscale conversion")
        return create_digit_recognizer_v23_rgb()
    else:
        print(f"Warning: Unsupported channel count {input_channels}. Using adaptive model.")
        return create_digit_recognizer_v23_adaptive()

def create_digit_recognizer_v23_grayscale():
    """
    v23 Grayscale model - no conversion needed, direct v4 architecture
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # No conversion needed for grayscale input
    x = inputs
    
    # Apply common backbone
    x = _build_v23_backbone(x)
    
    # Output layer
    if params.USE_LOGITS:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, 
            activation=None, 
            name='logits'
        )(x)
    else:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, 
            activation='softmax', 
            name='output'
        )(x)
    
    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v23_grayscale")

def create_digit_recognizer_v23_rgb():
    """
    v23 RGB model with luminance grayscale conversion entry layer
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Apply luminance grayscale conversion as first layer
    x = create_luminance_grayscale_conv(inputs)
    
    # Apply common backbone
    x = _build_v23_backbone(x)
    
    # Output layer
    if params.USE_LOGITS:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, 
            activation=None, 
            name='logits'
        )(x)
    else:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, 
            activation='softmax', 
            name='output'
        )(x)
    
    model = tf.keras.Model(inputs, outputs, name="digit_recognizer_v23_rgb")
    
    # Double-check luminance layer is frozen
    for layer in model.layers:
        if layer.name == 'luminance_grayscale_conv':
            layer.trainable = False
            print(f"✓ Luminance layer '{layer.name}' frozen")
    
    return model

def create_digit_recognizer_v23_adaptive():
    """
    v23 Adaptive model - dynamically handles input channels with fallback
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # create_luminance_grayscale_conv handles 1-channel, 3-channel, and fallback robustly
    x = create_luminance_grayscale_conv(inputs)
    
    # Apply common backbone
    x = _build_v23_backbone(x)
    
    # Output layer
    if params.USE_LOGITS:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, 
            activation=None, 
            name='logits'
        )(x)
    else:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, 
            activation='softmax', 
            name='output'
        )(x)
    
    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v23_adaptive")

# ============================================================================
# QUANTIZATION-AWARE TRAINING
# ============================================================================

def create_qat_model(base_model=None):
    """
    Create QAT model using the standard TF Model Optimization API.
    
    Args:
        base_model: Optional pre-created model. If None, creates a new one.
    
    Returns:
        Quantization-aware trained model ready for TFLite conversion.
    """
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model if base_model else create_digit_recognizer_v23()
    
    if base_model is None:
        base_model = create_digit_recognizer_v23()
    
    try:
        # Use the standard QAT API
        qat_model = tfmot.quantization.keras.quantize_model(base_model)
        
        # Re-freeze luminance layer if it exists (QAT may reset trainable flag)
        for layer in qat_model.layers:
            if 'luminance_grayscale' in layer.name or 'fallback' in layer.name:
                if hasattr(layer, 'trainable'):
                    layer.trainable = False
                    print(f"✓ Re-frozen '{layer.name}' after QAT")
        
        print("Successfully created QAT model using standard API")
        return qat_model
        
    except Exception as e:
        print(f"QAT creation failed: {e}")
        print("Falling back to base model without QAT")
        return base_model

def create_qat_model_legacy(base_model=None):
    """
    Legacy QAT approach kept for backward compatibility.
    Use create_qat_model() for new code.
    """
    return create_qat_model(base_model)

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_luminance_conversion():
    """
    Properly test the luminance grayscale conversion with a valid model.
    """
    import numpy as np
    
    print("\n=== Testing Luminance Grayscale Conversion ===")
    
    # Test with sample RGB values
    test_inputs = {
        'Pure Red': np.array([255, 0, 0], dtype=np.float32),
        'Pure Green': np.array([0, 255, 0], dtype=np.float32),
        'Pure Blue': np.array([0, 0, 255], dtype=np.float32),
        'Middle Gray': np.array([128, 128, 128], dtype=np.float32),
        'White': np.array([255, 255, 255], dtype=np.float32),
        'Black': np.array([0, 0, 0], dtype=np.float32),
    }
    
    # Build a proper test model using symbolic graph
    test_input = tf.keras.Input(shape=(1, 1, 3), name='test_input')
    test_output = create_luminance_grayscale_conv(test_input)
    test_model = tf.keras.Model(test_input, test_output)
    
    print("\nLuminance weights:")
    print("Red:   0.299  (29.9%)")
    print("Green: 0.587  (58.7%)")
    print("Blue:  0.114  (11.4%)")
    
    print("\nConversion results:")
    for name, rgb in test_inputs.items():
        # Create a 1x1x3 test image
        test_image = rgb.reshape(1, 1, 1, 3)
        
        # Apply the grayscale conversion using proper model
        result = test_model.predict(test_image, verbose=0)
        
        # Expected value for verification
        expected = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        
        print(f"{name:12} RGB{rgb.astype(int)} → {result[0,0,0,0]:.1f} (expected: {expected:.1f})")
        
        # Verify accuracy
        assert abs(result[0,0,0,0] - expected) < 0.1, f"Conversion error for {name}"
    
    print("\n✓ Luminance conversion working correctly")
    return test_model

def test_qat_compatibility():
    """Test QAT compatibility with current environment"""
    print("\n=== QAT COMPATIBILITY TEST ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    if QAT_AVAILABLE:
        print(f"TensorFlow Model Optimization version: {tfmot.__version__}")
        
        # Test basic model creation
        model = create_digit_recognizer_v23()
        print(f"Base model created: {model.name}")
        print(f"Base model parameters: {model.count_params():,}")
        
        # Verify luminance layer is frozen if present
        for layer in model.layers:
            if 'luminance' in layer.name:
                print(f"Layer '{layer.name}' trainable: {layer.trainable}")
        
        # Test QAT
        try:
            qat_model = create_qat_model(model)
            print(f"QAT model type: {type(qat_model)}")
            
            # Verify luminance layer remains frozen after QAT
            for layer in qat_model.layers:
                if 'luminance' in layer.name:
                    print(f"After QAT - '{layer.name}' trainable: {layer.trainable}")
                    assert layer.trainable is False, "Luminance layer should remain frozen!"
            
            print("✓ QAT test passed")
        except Exception as e:
            print(f"✗ QAT test failed: {e}")
    else:
        print("✗ QAT not available")

def test_backbone_consistency():
    """
    Verify that all model variants produce the same backbone structure
    """
    print("\n=== Testing Backbone Consistency ===")
    
    original_input_shape = params.INPUT_SHAPE
    
    # Test with RGB
    params.INPUT_SHAPE = (28, 28, 3)
    model_rgb = create_digit_recognizer_v23_rgb()
    
    # Test with grayscale
    params.INPUT_SHAPE = (28, 28, 1)
    model_gray = create_digit_recognizer_v23_grayscale()
    
    # Test with adaptive
    params.INPUT_SHAPE = (28, 28, 3)
    model_adaptive = create_digit_recognizer_v23_adaptive()
    
    # Compare layer counts (excluding input and output layers)
    layers_rgb = len([l for l in model_rgb.layers if 'conv' in l.name or 'dense' in l.name])
    layers_gray = len([l for l in model_gray.layers if 'conv' in l.name or 'dense' in l.name])
    layers_adaptive = len([l for l in model_adaptive.layers if 'conv' in l.name or 'dense' in l.name])
    
    print(f"RGB model backbone layers: {layers_rgb}")
    print(f"Grayscale model backbone layers: {layers_gray}")
    print(f"Adaptive model backbone layers: {layers_adaptive}")
    
    assert layers_rgb == layers_gray == layers_adaptive, "Backbone layer counts don't match!"
    print("✓ All variants have consistent backbone structure")
    
    # Restore original
    params.INPUT_SHAPE = original_input_shape

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=== Digit Recognizer v23 - Luminance Grayscale Entry ===")
    
    # Test with RGB input
    print("\n1. Testing with RGB input (28, 28, 3):")
    params.INPUT_SHAPE = (28, 28, 3)
    model_rgb = create_digit_recognizer_v23()
    print(f"Created model: {model_rgb.name}")
    model_rgb.summary()
    print(f"Total parameters: {model_rgb.count_params():,}")
    
    # Verify grayscale layer is frozen
    for layer in model_rgb.layers:
        if layer.name == 'luminance_grayscale_conv':
            print(f"\nGrayscale layer:")
            print(f"  Trainable: {layer.trainable}")
            print(f"  Weight shape: {layer.get_weights()[0].shape}")
            print(f"  Weights: {layer.get_weights()[0][0,0,:,0]}")
            assert layer.trainable is False, "Luminance layer should be frozen!"
    
    # Test with grayscale input
    print("\n2. Testing with grayscale input (28, 28, 1):")
    params.INPUT_SHAPE = (28, 28, 1)
    model_gray = create_digit_recognizer_v23()
    print(f"Created model: {model_gray.name}")
    model_gray.summary()
    print(f"Total parameters: {model_gray.count_params():,}")
    
    # Test luminance conversion
    test_luminance_conversion()
    
    # Test backbone consistency
    test_backbone_consistency()
    
    # Test QAT compatibility
    test_qat_compatibility()
    
    # Restore default input shape
    params.INPUT_SHAPE = (28, 28, 3)
    
    print("\n✓ v23 model ready for training and quantization")
    print("\nKey Features:")
    print("  - Luminance weighting: 0.299*R + 0.587*G + 0.114*B")
    print("  - Fixed Conv2D with frozen weights (non-trainable)")
    print("  - Single backbone architecture (no code duplication)")
    print("  - Proper Lambda fallback for dynamic channels")
    print("  - Correct QAT with re-frozen luminance layer")
    print("  - 100% TFLite Micro compatible")
    print("  - Same architecture as v4 after conversion")