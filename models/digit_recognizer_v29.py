# models/digit_recognizer_v29.py
"""
digit_recognizer_v29 – Adaptive Hard Binarization + Transition Preservation
===========================================================================
The Critique of v28 & Hardware Binarization:
  1. Hardware/Global Threshold limits: Using an ESP32 camera in 1-bit B&W applies
     a single threshold across the entire image. Due to glare/shadows on water
     meters, this often erases digits entirely. v28/v29 fixes this by computing
     the threshold adaptively *per digit isolated crop*.
  2. The Transition Information Loss (The flaw in v28): While v28's hard 
     binarization successfully extracts clean shapes regardless of lighting, it 
     destroys the transitional gradient information exactly at the edge of the 
     digit. For a 100-class model, differentiating exactly where a dial sits 
     (e.g., 5.5 vs 5.6) relies entirely on these subtle shadow gradients. 
     Hard binarization turns a 50% blend block into a rigid 0 or 1, dropping 
     critical positioning data.
     
Solution (v29 Hybrid Approach):
  Combines the best of both worlds by computing the adaptive per-digit threshold, 
  but outputting TWO channels into the CNN:
    Channel 0: Hard binary image (provides clean shape/topology immunity)
    Channel 1: Continuous gradient magnitude (preserves transitional/subpixel cues)

    The 2-channel approach is mathematically sound. The CNN can now learn to:

        Use channel 0 for robust shape recognition (lighting invariant)
        Use channel 1 for precise wheel position (transitional detection)
        Combine both for optimal accuracy

Pipeline:
  Input (1 or 3 ch) 
  -> Luminance 
  -> AdaptiveHybridBinarization (outputs 2 channels: hard binary + soft gradient)
  -> PolarityNormalization2Channel (flips both if background is light)
  -> AdaptiveBackbone (Conv2D modified to accept 2 input channels)
  -> Dense(100)
"""



import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False

try:
    from .digit_recognizer_v27 import _get_adaptive_config
except ImportError:
    from digit_recognizer_v27 import _get_adaptive_config


# ============================================================================
# HYBRID BINARIZATION PIPELINE
# ============================================================================

class AdaptiveHybridBinarization(tf.keras.layers.Layer):
    """
    Hybrid approach:
    1. Compute per-digit adaptive threshold
    2. Output both binary (for shape) AND soft gradient (for transition)
    
    The CNN can learn to use:
      - Binary stream: clean shape features
      - Soft stream: transitional gradient information (distance from threshold)
      
    Forward pass STE trick on binary channel ensures differentiability.
    """
    def __init__(self, preserve_gradient=True, **kwargs):
        super().__init__(**kwargs)
        self.preserve_gradient = preserve_gradient

    def call(self, inputs):
        # Compute per-digit threshold
        mean = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
        
        # Channel 0: Hard binary stream (shape features) using STE
        binary_hard = tf.cast(inputs > mean, tf.float32)
        binary_ste  = inputs + tf.stop_gradient(binary_hard - inputs)
        
        if not self.preserve_gradient:
            return binary_ste
            
        # Channel 1: Soft gradient stream (preserves transitional info)
        # Magnitude of deviation from threshold, normalized roughly to [0,1]
        # (Using just inputs-mean preserves full continuous info, but abs() is often better for edge-like features)
        # We'll just pass the normalized continuous image centered around 0.5
        # so it's bounded similarly to the binary channel.
        std = tf.math.reduce_std(inputs, axis=[1, 2, 3], keepdims=True)
        soft_gradient = tf.sigmoid((inputs - mean) / (std + 1e-5))
        
        # Concatenate binary + soft gradient
        # Shape: [batch, H, W, 2]
        return tf.concat([binary_ste, soft_gradient], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.preserve_gradient:
            return input_shape[:-1] + (2,)
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'preserve_gradient': self.preserve_gradient})
        return config


class PolarityNormalization2Channel(tf.keras.layers.Layer):
    """
    Normalize polarity for both channels simultaneously.
    If the image is inverted (light background), flips both the binary 
    and soft gradient channels so digits are always white/positive.
    """
    def call(self, inputs):
        # inputs: [batch, H, W, 2] (binary + soft)
        binary = inputs[..., 0:1]
        soft   = inputs[..., 1:2]
        
        # Check polarity strictly on the binary channel
        mean = tf.reduce_mean(binary, axis=[1, 2, 3], keepdims=True)
        flip = tf.cast(mean > 0.5, tf.float32)
        
        # Flip both channels if mostly white background (flip=1.0)
        binary_norm = (1.0 - flip) * binary + flip * (1.0 - binary)
        soft_norm   = (1.0 - flip) * soft   + flip * (1.0 - soft)
        
        return tf.concat([binary_norm, soft_norm], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


# ============================================================================
# HYBRID BACKBONE
# ============================================================================

def _build_v29_backbone(x, filters, dense_units, use_batch_norm=False):
    """
    Exactly the same as v24/v27 adaptive backbone, but implicitly accepts 
    2 channels at conv1 instead of 1. Keras handles the input_channels automatically.
    """
    with tf.name_scope('backbone'):
        x = tf.keras.layers.Conv2D(
            filters[0], (3, 3), padding='same',
            kernel_initializer='he_normal', use_bias=True, name=f'conv1_{filters[0]}f'
        )(x)
        if use_batch_norm: x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_1')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)

        x = tf.keras.layers.Conv2D(
            filters[1], (3, 3), padding='same',
            kernel_initializer='he_normal', use_bias=True, name=f'conv2_{filters[1]}f'
        )(x)
        if use_batch_norm: x = tf.keras.layers.BatchNormalization(name='bn2')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)

        x = tf.keras.layers.Conv2D(
            filters[2], (3, 3), padding='same',
            kernel_initializer='he_normal', use_bias=True, name=f'conv3_{filters[2]}f'
        )(x)
        if use_batch_norm: x = tf.keras.layers.BatchNormalization(name='bn3')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_3')(x)

        x = tf.keras.layers.Conv2D(
            filters[3], (3, 3), padding='same',
            kernel_initializer='he_normal', use_bias=True, name=f'conv4_{filters[3]}f'
        )(x)
        if use_batch_norm: x = tf.keras.layers.BatchNormalization(name='bn4')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_4')(x)

        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = tf.keras.layers.Dense(dense_units, activation=None, name='feature_dense')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_dense')(x)
        x = tf.keras.layers.Dropout(0.25, name='dropout')(x)
    return x


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_digit_recognizer_v29(use_batch_norm=False):
    """
    Creates model with hybrid binarization:
    - Channel 0: Clean binary image (for shape)
    - Channel 1: Continuous soft image (for transitional info)
    
    This preserves the information needed for accurate wheel position
    while maintaining the benefits of a perfectly clean shape edge.
    """
    filters, dense_units, _ = _get_adaptive_config()
    print(f"v29 config: classes={params.NB_CLASSES}  filters={filters}  dense={dense_units}")

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # 1. Luminance
    input_channels = params.INPUT_SHAPE[-1]
    if input_channels is not None and input_channels == 3:
        x = tf.keras.layers.Lambda(
            lambda t: 0.299 * t[..., 0:1] + 0.587 * t[..., 1:2] + 0.114 * t[..., 2:3],
            name='luminance_grayscale'
        )(inputs)
    else:
        x = inputs

    # 2. Hybrid Binarization (outputs 2 channels)
    x = AdaptiveHybridBinarization(preserve_gradient=True, name='hybrid_binarization')(x)

    # 3. Polarity Normalization (flips both channels if inverted)
    x = PolarityNormalization2Channel(name='polarity_norm')(x)

    # 4. Backbone expects 2 channels now
    x = _build_v29_backbone(x, filters, dense_units, use_batch_norm)

    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    model = tf.keras.Model(inputs, outputs, name='digit_recognizer_v29')

    # Freeze preprocessing
    frozen = 0
    freeze_names = {'luminance_grayscale', 'hybrid_binarization', 'polarity_norm'}
    for layer in model.layers:
        if layer.name in freeze_names:
            layer.trainable = False
            frozen += 1
            print(f"✓ Locked '{layer.name}'")

    print(f"v29 params: {model.count_params():,}  frozen={frozen}")
    return model


def create_qat_model_v29(model=None):
    if model is None:
        model = create_digit_recognizer_v29()
    if not QAT_AVAILABLE:
        return model
    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        for layer in qat_model.layers:
            if layer.name in ('luminance_grayscale', 'hybrid_binarization', 'polarity_norm'):
                layer.trainable = False
        return qat_model
    except Exception as e:
        print(f"QAT creation failed: {e}")
        return model


if __name__ == "__main__":
    print("=== Digit Recognizer v29 ===\n")
    params.INPUT_SHAPE = (28, 28, 1)
    params.NB_CLASSES  = 100

    model = create_digit_recognizer_v29()
    model.summary()

    # Test dummy inference
    x = np.random.rand(2, 28, 28, 1).astype(np.float32)
    pred = model.predict(x, verbose=0)
    
    # Check intermediate
    preproc = tf.keras.Model(model.input, model.get_layer('polarity_norm').output)
    out_2ch = preproc.predict(x, verbose=0)
    
    print("\n✓ 2-Channel Preprocessing Output:")
    print(f"  Shape: {out_2ch.shape}")
    print(f"  Ch0 (binary) mean: {np.mean(out_2ch[..., 0]):.3f}")
    print(f"  Ch1 (soft) mean:   {np.mean(out_2ch[..., 1]):.3f}")
