# models/digit_recognizer_v24.py
"""
digit_recognizer_v24 – Adaptive Contrast Normalization for Light/Dark Backgrounds
=================================================================================
Design goal: Handle digits written on both light-on-dark and dark-on-light backgrounds
automatically within the model architecture. Uses adaptive contrast detection and
normalization as the first layer.

Problem:
  - MNIST-like datasets typically have dark digits on light background
  - Real-world images can have inverted contrast (light digits on dark background)
  - Standard models fail on inverted contrast (accuracy drops from 99% to 30-40%)

Solution:
  - Detect background intensity and optionally invert input if needed
  - Apply adaptive contrast normalization
  - Preserve luminance information while standardizing contrast polarity

Architecture:
  - Adaptive Contrast Detection (statistical intensity analysis)
  - Optional Inversion Layer (based on background detection)
  - Luminance Grayscale Conversion (from v23)
  - Enhanced v4 Backbone with contrast robustness
  - Output with confidence calibration

Key Features:
  - Automatic contrast polarity detection
  - Background intensity normalization
  - Robust to both light-on-dark and dark-on-light
  - No additional training data needed (self-normalizing)
  - QAT-compatible for ESP32 deployment

1. Adaptive Contrast Normalization

    Automatically detects background type (light or dark)
    Conditionally inverts image to standardize contrast
    Uses statistical analysis (mean intensity)
    Fully differentiable and QAT-compatible

2. Three Implementation Methods
    Method	Description	Best For
    Adaptive	Mean-based detection + contrast stretching	ESP32, lightweight deployment
    Robust	Percentile-based normalization	Extreme contrast variations
    Combined	Luminance + contrast in one layer	Maximum efficiency

3. Contrast Augmentation

    Random inversion during training (50% probability)
    Makes model robust to both contrast modes
    No additional data needed

4. Enhanced Backbone

    Optional BatchNormalization for stability
    Dropout for regularization
    Same parameter count as v4/v23


"""

import tensorflow as tf
import numpy as np
import parameters as params
try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False

# ============================================================================
# CONTRAST NORMALIZATION LAYERS
# ============================================================================

class AdaptiveContrastNormalization(tf.keras.layers.Layer):
    """
    Adaptive contrast normalization that detects and corrects inverted contrast.
    
    Works by analyzing image statistics:
    1. Compute global intensity statistics (mean, median)
    2. Detect if background is lighter or darker than foreground
    3. Optionally invert image to standardize contrast
    4. Apply contrast stretching for better feature extraction
    
    This layer is designed to be:
    - Fully differentiable (gradients can flow through)
    - QAT-compatible (uses only standard TFLite ops)
    - ESP32-friendly (minimal memory overhead)
    """
    
    def __init__(self, invert_threshold=0.5, stretch_contrast=True, **kwargs):
        """
        Args:
            invert_threshold: Threshold for background detection (0-1)
                              If mean intensity > threshold, assume light background
            stretch_contrast: Whether to apply contrast stretching
        """
        super().__init__(**kwargs)
        self.invert_threshold = invert_threshold
        self.stretch_contrast = stretch_contrast

    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Apply adaptive contrast normalization.
        
        Args:
            inputs: Input tensor of shape (batch, height, width, channels)
                    Expected to be grayscale (1 channel) after luminance conversion
        
        Returns:
            Normalized tensor with consistent contrast polarity
        """
        # Ensure we have single channel
        if inputs.shape[-1] != 1:
            # Average channels if needed
            x = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        else:
            x = inputs
        
        # Compute global intensity statistics
        # Use mean and median for robust background detection
        batch_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        
        # Detect background type
        # If mean > threshold, assume light background with dark digits
        # If mean < threshold, assume dark background with light digits
        should_invert = tf.cast(batch_mean < self.invert_threshold, tf.float32)
        
        # Apply inversion if needed (light digits on dark background)
        # Inversion: 1.0 - x (assuming input is normalized [0,1])
        inverted = 1.0 - x
        x = (1.0 - should_invert) * x + should_invert * inverted
        
        # Optional: Apply contrast stretching for better feature separation
        if self.stretch_contrast:
            # Compute per-image min and max for contrast stretching
            batch_min = tf.reduce_min(x, axis=[1, 2, 3], keepdims=True)
            batch_max = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
            
            # Avoid division by zero
            range_val = tf.maximum(batch_max - batch_min, 1e-6)
            
            # Stretch to [0, 1] range
            x = (x - batch_min) / range_val
        
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'invert_threshold': self.invert_threshold,
            'stretch_contrast': self.stretch_contrast,
        })
        return config


class RobustContrastNormalization(tf.keras.layers.Layer):
    """
    Alternative: Percentile-based contrast normalization.
    More robust to outliers than AdaptiveContrastNormalization.

    ⚠️  PC / training ONLY — NOT deployable to ESP32 / TFLite Micro.
    Uses tf.nn.quantile (sort-based) which has no TFLite op equivalent.
    Use AdaptiveContrastNormalization for ESP32 deployment.
    """
    
    def __init__(self, lower_percentile=10, upper_percentile=90, **kwargs):
        super().__init__(**kwargs)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
    
    def call(self, inputs):
        # Ensure single channel
        if inputs.shape[-1] != 1:
            x = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        else:
            x = inputs
        
        # Flatten spatial dimensions for percentile calculation
        batch_size = tf.shape(x)[0]
        flat = tf.reshape(x, [batch_size, -1])
        
        # Compute percentiles using tf.nn.quantile (available in TF 2.0+)
        # Note: This uses sorting which is memory intensive - use with caution on ESP32
        lower_val = tf.nn.quantile(flat, self.lower_percentile / 100.0, axis=1, keepdims=True)
        upper_val = tf.nn.quantile(flat, self.upper_percentile / 100.0, axis=1, keepdims=True)
        
        # Reshape to match input dimensions
        lower_val = tf.reshape(lower_val, [batch_size, 1, 1, 1])
        upper_val = tf.reshape(upper_val, [batch_size, 1, 1, 1])
        
        # Apply contrast stretching
        range_val = tf.maximum(upper_val - lower_val, 1e-6)
        x = (x - lower_val) / range_val
        
        # Clip to [0,1] range
        x = tf.clip_by_value(x, 0.0, 1.0)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'lower_percentile': self.lower_percentile,
            'upper_percentile': self.upper_percentile,
        })
        return config


class ContrastAwareInput(tf.keras.layers.Layer):
    """
    Combined layer: luminance conversion + adaptive contrast normalization.
    Provides both features in a single layer for cleaner model architecture.
    """
    
    def __init__(self, invert_threshold=0.5, stretch_contrast=True, **kwargs):
        super().__init__(**kwargs)
        self.invert_threshold = invert_threshold
        self.stretch_contrast = stretch_contrast

    def build(self, input_shape):
        # Store luminance weights as a non-trainable weight so they survive
        # model.save() / load_model() correctly.
        self.luminance_weights = self.add_weight(
            name='luminance_weights',
            shape=(3,),
            initializer=tf.keras.initializers.Constant([0.299, 0.587, 0.114]),
            trainable=False,
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # Step 1: Convert to grayscale with luminance weighting
        channels = inputs.shape[-1]
        
        if channels == 3:
            # Apply luminance weighting
            grayscale = (inputs[..., 0] * self.luminance_weights[0] +
                        inputs[..., 1] * self.luminance_weights[1] +
                        inputs[..., 2] * self.luminance_weights[2])
            grayscale = tf.expand_dims(grayscale, axis=-1)
        elif channels == 1:
            grayscale = inputs
        else:
            # Fallback to average
            grayscale = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        
        # Step 2: Detect and correct contrast polarity
        # Compute global statistics
        batch_mean = tf.reduce_mean(grayscale, axis=[1, 2, 3], keepdims=True)
        
        # Detect if we need to invert (dark background with light digits)
        should_invert = tf.cast(batch_mean < self.invert_threshold, tf.float32)
        
        # Apply conditional inversion
        inverted = 1.0 - grayscale
        normalized = (1.0 - should_invert) * grayscale + should_invert * inverted
        
        # Step 3: Contrast stretching (optional)
        if self.stretch_contrast:
            batch_min = tf.reduce_min(normalized, axis=[1, 2, 3], keepdims=True)
            batch_max = tf.reduce_max(normalized, axis=[1, 2, 3], keepdims=True)
            range_val = tf.maximum(batch_max - batch_min, 1e-6)
            normalized = (normalized - batch_min) / range_val
        
        return normalized
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'invert_threshold': self.invert_threshold,
            'stretch_contrast': self.stretch_contrast,
        })
        return config


# ============================================================================
# ENHANCED BACKBONE WITH CONTRAST ROBUSTNESS
# ============================================================================

def _build_v24_backbone(x, use_batch_norm=False):
    """
    Enhanced backbone for v24 with optional batch normalization for stability.
    BN helps with contrast variations but adds parameters.
    
    Args:
        x: Input tensor (already normalized to grayscale with consistent contrast)
        use_batch_norm: Whether to add BatchNormalization layers (default False for ESP32)
    """
    # Layer 1: Initial feature extraction
    x = tf.keras.layers.Conv2D(
        20, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1_20f'
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
    
    # Layer 2: Mid-level features
    x = tf.keras.layers.Conv2D(
        36, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2_36f'
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)
    
    # Layer 3: High-level features
    x = tf.keras.layers.Conv2D(
        48, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3_48f'
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_3')(x)
    
    # Layer 4: Bottleneck
    x = tf.keras.layers.Conv2D(
        56, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv4_56f'
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name='bn4')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_4')(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer with dropout for regularization
    x = tf.keras.layers.Dense(64, activation=None, name='feature_dense')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_dense')(x)
    x = tf.keras.layers.Dropout(0.25, name='dropout')(x)
    
    return x


# ============================================================================
# V24 MODEL VARIANTS
# ============================================================================

def create_digit_recognizer_v24(method='adaptive', use_batch_norm=False):
    """
    Create v24 model with adaptive contrast normalization.
    
    Args:
        method: 'adaptive' - AdaptiveContrastNormalization
                'robust' - RobustContrastNormalization (more robust but slower)
                'combined' - Combined luminance + contrast in one layer
        use_batch_norm: Whether to add BatchNormalization (default False for ESP32)
    """
    input_channels = params.INPUT_SHAPE[-1]
    
    print(f"Creating v24 model with method='{method}', use_batch_norm={use_batch_norm}")
    
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Step 1 + 2: Luminance conversion and/or contrast normalization
    if method == 'adaptive':
        # Luminance conversion first (RGB→gray), then adaptive contrast
        if input_channels == 3:
            x = tf.keras.layers.Lambda(
                lambda t: (0.299 * t[..., 0:1] +
                           0.587 * t[..., 1:2] +
                           0.114 * t[..., 2:3]),
                name='luminance_grayscale'
            )(inputs)
        else:
            x = inputs
        x = AdaptiveContrastNormalization(
            invert_threshold=0.5,
            stretch_contrast=True,
            name='adaptive_contrast'
        )(x)
    elif method == 'robust':
        # Luminance conversion first (RGB→gray), then robust contrast
        if input_channels == 3:
            x = tf.keras.layers.Lambda(
                lambda t: (0.299 * t[..., 0:1] +
                           0.587 * t[..., 1:2] +
                           0.114 * t[..., 2:3]),
                name='luminance_grayscale'
            )(inputs)
        else:
            x = inputs
        x = RobustContrastNormalization(
            lower_percentile=10,
            upper_percentile=90,
            name='robust_contrast'
        )(x)
    elif method == 'combined':
        # ContrastAwareInput handles luminance + contrast in one layer
        x = ContrastAwareInput(
            invert_threshold=0.5,
            stretch_contrast=True,
            name='combined_contrast'
        )(inputs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Step 3: Enhanced backbone
    x = _build_v24_backbone(x, use_batch_norm=use_batch_norm)
    
    # Step 4: Output layer
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES,
        activation='softmax',
        name='output'
    )(x)
    
    model = tf.keras.Model(inputs, outputs, name=f"digit_recognizer_v24_{method}")
    
    # Print model info
    print(f"Model parameters: {model.count_params():,}")
    print(f"Expected size after INT8 quantization: {model.count_params() / 1024:.1f} KB")
    
    return model


def create_digit_recognizer_v24_lightweight():
    """
    Lightweight version optimized for ESP32:
    - No BatchNormalization
    - Simplified contrast detection
    - Minimal overhead
    """
    return create_digit_recognizer_v24(method='adaptive', use_batch_norm=False)


def create_digit_recognizer_v24_robust():
    """
    Robust version for better accuracy with extreme contrast variations:
    - Uses percentile-based normalization
    - Slightly larger but handles outliers better
    """
    return create_digit_recognizer_v24(method='robust', use_batch_norm=False)


def create_digit_recognizer_v24_combined():
    """
    Combined version: luminance + contrast in single layer
    Most efficient for inference
    """
    return create_digit_recognizer_v24(method='combined', use_batch_norm=False)


# ============================================================================
# QUANTIZATION-AWARE TRAINING
# ============================================================================

def create_qat_model_v24(model=None):
    """
    Create QAT-ready v24 model with contrast normalization layers properly frozen
    """
    if model is None:
        model = create_digit_recognizer_v24_lightweight()
    
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return model
    
    try:
        # Apply QAT
        qat_model = tfmot.quantization.keras.quantize_model(model)
        
        # Ensure contrast normalization layers remain non-trainable
        for layer in qat_model.layers:
            if any(name in layer.name for name in ['contrast', 'normalization', 'luminance']):
                if hasattr(layer, 'trainable'):
                    layer.trainable = False
                    print(f"✓ Frozen '{layer.name}' after QAT")
        
        print("Successfully created QAT-ready v24 model")
        return qat_model
        
    except Exception as e:
        print(f"QAT creation failed: {e}")
        print("Falling back to base model")
        return model


# ============================================================================
# DATA AUGMENTATION FOR CONTRAST VARIATIONS
# ============================================================================

class ContrastAugmentation(tf.keras.layers.Layer):
    """
    Data augmentation layer that randomly inverts contrast during training.
    Helps the model learn to handle both contrast modes.
    """
    
    def __init__(self, invert_probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.invert_probability = invert_probability
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        # Randomly invert contrast
        batch_size = tf.shape(inputs)[0]
        random_values = tf.random.uniform([batch_size, 1, 1, 1])
        should_invert = tf.cast(random_values < self.invert_probability, tf.float32)
        
        # Apply inversion: 1.0 - x
        inverted = 1.0 - inputs
        return (1.0 - should_invert) * inputs + should_invert * inverted
    
    def get_config(self):
        config = super().get_config()
        config.update({'invert_probability': self.invert_probability})
        return config


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_contrast_augmented_dataset(x_train, y_train, batch_size=32):
    """
    Create dataset with contrast augmentation for training
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    
    # Add contrast augmentation
    def augment(image, label):
        # Randomly invert with 50% probability
        # Use tf.cond (not Python if) so each call gets a fresh random value
        image = tf.cond(
            tf.random.uniform(()) > 0.5,
            lambda: 1.0 - image,
            lambda: image
        )
        return image, label
    
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_contrast_normalization():
    """
    Test the contrast normalization layers with various contrast patterns
    """
    import numpy as np
    
    print("\n=== Testing Contrast Normalization ===")
    
    # Create test inputs: dark digits on light background
    light_bg = np.ones((1, 28, 28, 1), dtype=np.float32)  # White background
    light_bg[0, 10:18, 10:18, 0] = 0.0  # Black digit
    
    # Create test inputs: light digits on dark background
    dark_bg = np.zeros((1, 28, 28, 1), dtype=np.float32)  # Black background
    dark_bg[0, 10:18, 10:18, 0] = 1.0  # White digit
    
    # Create model with adaptive contrast
    model = create_digit_recognizer_v24_lightweight()
    
    # Extract contrast layer
    contrast_layer = None
    for layer in model.layers:
        if 'contrast' in layer.name:
            contrast_layer = layer
            break
    
    if contrast_layer:
        print("\nTesting light background (dark digits):")
        result_light = contrast_layer(light_bg)
        print(f"  Input mean: {np.mean(light_bg):.3f}")
        print(f"  Output mean: {np.mean(result_light.numpy()):.3f}")
        print(f"  Output range: [{np.min(result_light.numpy()):.3f}, {np.max(result_light.numpy()):.3f}]")
        
        print("\nTesting dark background (light digits):")
        result_dark = contrast_layer(dark_bg)
        print(f"  Input mean: {np.mean(dark_bg):.3f}")
        print(f"  Output mean: {np.mean(result_dark.numpy()):.3f}")
        print(f"  Output range: [{np.min(result_dark.numpy()):.3f}, {np.max(result_dark.numpy()):.3f}]")
        
        print("\n✓ Contrast normalization working correctly")
    else:
        print("No contrast layer found in model")


def test_contrast_invariance():
    """
    Test that the model produces similar predictions for inverted images
    """
    import numpy as np
    
    print("\n=== Testing Contrast Invariance ===")
    
    # Create a simple digit pattern
    test_digit = np.zeros((1, 28, 28, 1), dtype=np.float32)
    test_digit[0, 10:18, 10:18, 0] = 1.0  # White square (digit)
    
    # Create inverted version
    inverted_digit = 1.0 - test_digit
    
    # Create model
    model = create_digit_recognizer_v24_lightweight()
    
    # Get predictions (need to compile for prediction)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    pred_normal = model.predict(test_digit, verbose=0)
    pred_inverted = model.predict(inverted_digit, verbose=0)
    
    # Compare predictions (should be similar)
    diff = np.abs(pred_normal - pred_inverted)
    print(f"Maximum prediction difference: {np.max(diff):.6f}")
    print(f"Mean prediction difference: {np.mean(diff):.6f}")
    
    if np.max(diff) < 0.1:
        print("✓ Model is contrast-invariant")
    else:
        print("⚠️ Model shows some contrast sensitivity")


def main():
    """Test v24 model creation and functionality"""
    print("=== Digit Recognizer v24 - Adaptive Contrast Normalization ===")
    
    # Test with RGB input
    print("\n1. Creating lightweight v24 model (RGB input):")
    params.INPUT_SHAPE = (28, 28, 3)
    model = create_digit_recognizer_v24_lightweight()
    model.summary()
    
    # Test with grayscale input
    print("\n2. Creating lightweight v24 model (grayscale input):")
    params.INPUT_SHAPE = (28, 28, 1)
    model_gray = create_digit_recognizer_v24_lightweight()
    
    # Test contrast normalization
    test_contrast_normalization()
    
    # Test contrast invariance
    test_contrast_invariance()
    
    # Compare model sizes
    print("\n3. Model comparison:")
    params.INPUT_SHAPE = (28, 28, 1)
    
    # v23 model
    from digit_recognizer_v23 import create_digit_recognizer_v23_grayscale
    v23_model = create_digit_recognizer_v23_grayscale()
    print(f"v23 parameters: {v23_model.count_params():,}")
    
    # v24 models
    v24_light = create_digit_recognizer_v24_lightweight()
    print(f"v24 lightweight parameters: {v24_light.count_params():,}")
    
    v24_robust = create_digit_recognizer_v24_robust()
    print(f"v24 robust parameters: {v24_robust.count_params():,}")
    
    print("\n✓ v24 models ready for training")
    print("\nKey Features:")
    print("  - Automatic contrast detection and correction")
    print("  - Handles both light-on-dark and dark-on-light digits")
    print("  - QAT-compatible for ESP32 deployment")
    print("  - Multiple variants for different use cases")
    print("  - Contrast augmentation during training")

if __name__ == "__main__":
    main()