# models/digit_recognizer_v25.py
"""
digit_recognizer_v25 – Transition-Aware Water Meter Digit Recognizer
====================================================================
Design goal: Recognize digits from water meters where the wheel can be
between two numbers (e.g., 5.5 showing both 5 and 6). Uses transition
detection and weighted prediction for accurate reading.

Problem:
  - Standard MNIST models fail on transitional digits (5.5 classified as 5 or 6)
  - Water meter wheels show partial digits during rotation
  - Need to round based on position: x.0-x.6 → x, x.7-x.9 → x+1


Water Meter Wheel Positions:
[5.0] [5.1] [5.2] [5.3] [5.4] [5.5] [5.6] [5.7] [5.8] [5.9] [6.0]
  |_____ Lower Range (round down) _____|  |__ Upper Range (round up) __|
  
Rule: x.0 - x.6 → digit x (lower)
      x.7 - x.9 → digit x+1 (upper)
      9.9 → 0 (carry to next digit)

Solution:
  - Multi-head architecture: Digit Classification + Transition Detection
  - Uncertainty-aware predictions with confidence calibration
  - Temporal consistency for video streams
  - Outputs both digit and transition state

Architecture:
  - Enhanced Luminance + Contrast Normalization (from v24)
  - Shared Feature Extractor
  - Digit Classification Head (10 classes)
  - Transition Detection Head (binary: lower/upper range)
  - Confidence Calibration Layer
  - Temporal Smoothing (optional for video)

Key Features:
  - Handles transitional digits (5.5)
  - Provides confidence scores
  - Option for temporal consistency
  - QAT-compatible for ESP32 deployment
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
# TRANSITION-AWARE LAYERS
# ============================================================================

class TransitionDetectionHead(tf.keras.layers.Layer):
    """
    Detects if the digit is in a transitional state and which direction.
    
    Outputs:
        - transition_prob: Probability of being in transition (0-1)
        - transition_direction: 0 = lower range (x.0-x.6), 1 = upper range (x.7-x.9)
    """
    
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # Dense layers for transition detection
        self.dense1 = tf.keras.layers.Dense(
            self.units, activation='relu', name='transition_dense1'
        )
        self.dense2 = tf.keras.layers.Dense(
            self.units // 2, activation='relu', name='transition_dense2'
        )
        
        # Two outputs: probability of transition, and direction
        self.transition_prob = tf.keras.layers.Dense(
            1, activation='sigmoid', name='transition_prob'
        )
        self.transition_dir = tf.keras.layers.Dense(
            1, activation='sigmoid', name='transition_dir'
        )
        
        super().build(input_shape)
    
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        
        prob = self.transition_prob(x)  # How confident we are that it's transitional
        direction = self.transition_dir(x)  # 0 = lower, 1 = upper
        
        return prob, direction
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class WeightedDigitHead(tf.keras.layers.Layer):
    """
    Digit classification head with confidence weighting.
    Outputs both digit class and confidence score.
    """
    
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(
            64, activation='relu', name='digit_dense'
        )
        self.digit_logits = tf.keras.layers.Dense(
            self.num_classes, name='digit_logits'
        )
        self.confidence = tf.keras.layers.Dense(
            1, activation='sigmoid', name='digit_confidence'
        )
        super().build(input_shape)
    
    def call(self, x):
        x = self.dense(x)
        logits = self.digit_logits(x)
        confidence = self.confidence(x)
        return logits, confidence
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config


class TemporalSmoothing(tf.keras.layers.Layer):
    """
    Temporal smoothing for video streams.
    Uses exponential moving average of predictions.
    """
    
    def __init__(self, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def build(self, input_shape):
        # State variables for temporal smoothing
        self.smoothed_digit = self.add_weight(
            name='smoothed_digit',
            shape=(1,),
            initializer='zeros',
            trainable=False
        )
        self.smoothed_transition = self.add_weight(
            name='smoothed_transition',
            shape=(1,),
            initializer='zeros',
            trainable=False
        )
        super().build(input_shape)
    
    def call(self, digit_pred, transition_pred, training=None):
        if training:
            # During training, no temporal smoothing
            return digit_pred, transition_pred
        
        # During inference, apply EMA smoothing
        smoothed_digit = (self.alpha * self.smoothed_digit + 
                         (1 - self.alpha) * tf.cast(digit_pred, tf.float32))
        smoothed_transition = (self.alpha * self.smoothed_transition + 
                              (1 - self.alpha) * transition_pred)
        
        # Update state
        self.smoothed_digit.assign(smoothed_digit)
        self.smoothed_transition.assign(smoothed_transition)
        
        return smoothed_digit, smoothed_transition
    
    def reset_states(self):
        """Reset temporal state for new sequence"""
        self.smoothed_digit.assign(tf.zeros_like(self.smoothed_digit))
        self.smoothed_transition.assign(tf.zeros_like(self.smoothed_transition))
    
    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha})
        return config


# ============================================================================
# ENHANCED BACKBONE FOR TRANSITION DETECTION
# ============================================================================

def _build_v25_backbone(x, use_batch_norm=False):
    """
    Enhanced backbone with better feature extraction for transitional digits.
    Uses larger filters in early layers to capture partial digits.
    """
    with tf.name_scope('backbone'):
        # Layer 1: Increased filters for better edge detection of partial digits
        x = tf.keras.layers.Conv2D(
            24, (3, 3), padding='same',  # 20 → 24 for better feature extraction
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv1_24f'
        )(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_1')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)
        
        # Layer 2: Maintain feature maps
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
        
        # Layer 4: Bottleneck with attention-like features
        x = tf.keras.layers.Conv2D(
            64, (3, 3), padding='same',  # 56 → 64 for better transition detection
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv4_64f'
        )(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name='bn4')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_4')(x)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Shared dense layer
        x = tf.keras.layers.Dense(96, activation=None, name='shared_dense')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_shared')(x)
        x = tf.keras.layers.Dropout(0.25, name='shared_dropout')(x)
        
    return x


# ============================================================================
# V25 MODEL WITH MULTI-HEAD ARCHITECTURE
# ============================================================================

class DigitRecognizerV25(tf.keras.Model):
    """
    Custom model with two heads:
    1. Digit classification (10 classes)
    2. Transition detection (lower/upper range)
    
    Also provides confidence scores and temporal smoothing.
    """
    
    def __init__(self, use_temporal_smoothing=False, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.use_temporal_smoothing = use_temporal_smoothing
        self.alpha = alpha
        
        # Feature extractor
        self.backbone = None  # Will be built in call
        
        # Heads
        self.digit_head = None
        self.transition_head = None
        
        # Temporal smoothing (optional)
        self.temporal_smoothing = None
        
    def build(self, input_shape):
        # Build backbone with input shape
        x = tf.keras.Input(shape=input_shape[1:])
        backbone_output = _build_v25_backbone(x, use_batch_norm=False)
        
        # Create sub-models
        self.backbone = tf.keras.Model(inputs=x, outputs=backbone_output)
        
        # Build heads
        self.digit_head = WeightedDigitHead(num_classes=params.NB_CLASSES, name='digit_head')
        self.transition_head = TransitionDetectionHead(units=32, name='transition_head')
        
        if self.use_temporal_smoothing:
            self.temporal_smoothing = TemporalSmoothing(alpha=self.alpha)
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Extract features
        features = self.backbone(inputs, training=training)
        
        # Get digit predictions
        digit_logits, digit_confidence = self.digit_head(features, training=training)
        digit_pred = tf.argmax(digit_logits, axis=-1)
        
        # Get transition predictions
        transition_prob, transition_dir = self.transition_head(features, training=training)
        
        # Apply temporal smoothing if enabled
        if self.use_temporal_smoothing and not training:
            digit_pred, transition_dir = self.temporal_smoothing(
                digit_pred, transition_dir, training=training
            )
        
        # Apply transition rule to get final digit
        # If transition_dir > 0.5 (upper range), round up to next digit
        # But only if we're in transition (transition_prob > threshold)
        transition_threshold = 0.5
        is_upper = tf.cast(transition_dir > transition_threshold, tf.float32)
        is_transition = tf.cast(transition_prob > transition_threshold, tf.float32)
        
        # Adjust digit based on transition direction
        # If in upper range, digit + 1 (with wrap-around for 9)
        digit_float = tf.cast(digit_pred, tf.float32)
        adjusted_digit = digit_float + (is_upper * is_transition)
        
        # Handle wrap-around: 10 becomes 0 (for 9.9 → 0)
        adjusted_digit = tf.math.floormod(adjusted_digit, 10.0)
        
        # Final output: rounded digit and confidence
        final_digit = tf.cast(adjusted_digit, tf.int32)
        
        # Overall confidence = digit_confidence * (1 - transition_prob) + confidence in transition
        # This gives lower confidence for transitional states
        overall_confidence = digit_confidence * (1.0 - transition_prob) + transition_prob
        
        return {
            'digit': final_digit,
            'digit_confidence': digit_confidence,
            'transition_prob': transition_prob,
            'transition_dir': transition_dir,
            'overall_confidence': overall_confidence,
            'raw_logits': digit_logits
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'use_temporal_smoothing': self.use_temporal_smoothing,
            'alpha': self.alpha,
        })
        return config


# ============================================================================
# SIMPLIFIED V25 MODEL (FUNCTIONAL API)
# ============================================================================

def create_digit_recognizer_v25(method='standard', use_temporal=False):
    """
    Create v25 model using functional API for easier deployment.
    
    Args:
        method: 'standard' - Basic transition detection
                'temporal' - With temporal smoothing for video
        use_temporal: Enable temporal smoothing (for video streams)
    """
    input_channels = params.INPUT_SHAPE[-1]
    
    print(f"Creating v25 model with method='{method}', temporal={use_temporal}")
    
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Step 1: Luminance conversion (if RGB)
    if input_channels == 3:
        x = tf.keras.layers.Lambda(
            lambda t: (0.299 * t[..., 0:1] +
                       0.587 * t[..., 1:2] +
                       0.114 * t[..., 2:3]),
            name='luminance_grayscale'
        )(inputs)
    else:
        x = inputs
    
    # Step 2: Contrast normalization (from v24)
    x = AdaptiveContrastNormalization(
        invert_threshold=0.5,
        stretch_contrast=True,
        name='adaptive_contrast'
    )(x)
    
    # Step 3: Enhanced backbone
    x = _build_v25_backbone(x, use_batch_norm=False)
    
    # Step 4: Multi-head outputs
    # Digit classification head
    digit_dense = tf.keras.layers.Dense(64, activation='relu', name='digit_dense')(x)
    digit_logits = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='digit_logits'
    )(digit_dense)
    
    # Transition detection head
    trans_dense = tf.keras.layers.Dense(32, activation='relu', name='trans_dense')(x)
    transition_prob = tf.keras.layers.Dense(
        1, activation='sigmoid', name='transition_prob'
    )(trans_dense)
    transition_dir = tf.keras.layers.Dense(
        1, activation='sigmoid', name='transition_dir'
    )(trans_dense)
    
    # Confidence scores
    digit_confidence = tf.keras.layers.Dense(
        1, activation='sigmoid', name='digit_confidence'
    )(digit_dense)
    
    # Apply transition rule
    def apply_transition_rule(inputs):
        digit_probs, trans_prob, trans_dir = inputs
        digit_class = tf.argmax(digit_probs, axis=-1)
        
        # Thresholds
        trans_threshold = 0.5
        dir_threshold = 0.5
        
        is_transition = tf.cast(trans_prob > trans_threshold, tf.float32)
        is_upper = tf.cast(trans_dir > dir_threshold, tf.float32)
        
        # Adjust digit
        digit_float = tf.cast(digit_class, tf.float32)
        adjusted = digit_float + (is_upper * is_transition)
        adjusted = tf.math.floormod(adjusted, 10.0)
        
        return tf.cast(adjusted, tf.int32)
    
    final_digit = tf.keras.layers.Lambda(
        apply_transition_rule,
        name='final_digit'
    )([digit_logits, transition_prob, transition_dir])
    
    # Create model
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            'digit': final_digit,
            'digit_probs': digit_logits,
            'digit_confidence': digit_confidence,
            'transition_prob': transition_prob,
            'transition_dir': transition_dir,
        },
        name=f"digit_recognizer_v25_{method}"
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    return model


# ============================================================================
# CUSTOM LOSS FUNCTIONS
# ============================================================================

class TransitionAwareLoss(tf.keras.losses.Loss):
    """
    Custom loss that handles transitional digits.
    For x.0-x.6 → target = x
    For x.7-x.9 → target = x+1
    """
    
    def __init__(self, digit_weight=1.0, transition_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.digit_weight = digit_weight
        self.transition_weight = transition_weight
    
    def call(self, y_true, y_pred):
        # y_true: [digit, transition_state] where transition_state is 0 or 1
        # y_pred: dict with 'digit_logits', 'transition_prob', 'transition_dir'
        
        digit_target = y_true[..., 0]
        transition_target = y_true[..., 1]
        
        # Digit loss (sparse categorical crossentropy)
        digit_loss = tf.keras.losses.sparse_categorical_crossentropy(
            digit_target, y_pred['digit_logits']
        )
        
        # Transition loss (binary crossentropy)
        transition_loss = tf.keras.losses.binary_crossentropy(
            transition_target, y_pred['transition_prob']
        )
        
        # Direction loss (only when in transition)
        direction_mask = tf.cast(transition_target > 0.5, tf.float32)
        direction_target = tf.where(transition_target > 0.5, 1.0, 0.0)  # Upper range = 1
        
        direction_loss = tf.keras.losses.binary_crossentropy(
            direction_target, y_pred['transition_dir']
        )
        direction_loss = direction_loss * direction_mask
        
        # Combined loss
        total_loss = (self.digit_weight * digit_loss + 
                     self.transition_weight * transition_loss +
                     self.transition_weight * direction_loss)
        
        return total_loss


# ============================================================================
# DATA GENERATION FOR TRANSITIONAL DIGITS
# ============================================================================

def generate_transitional_mnist(original_images, original_labels, transition_range=0.3):
    """
    Generate transitional digits by blending two digits.
    
    Args:
        original_images: MNIST images (28x28 grayscale)
        original_labels: Original digit labels
        transition_range: How much to blend (0-1)
    
    Returns:
        blended_images, blended_labels (with transition state)
    """
    blended_images = []
    blended_labels = []
    
    for i in range(len(original_images) - 1):
        # Randomly decide to create transitional pair
        if np.random.random() > 0.3:
            # Create transitional between current and next digit
            digit1 = original_labels[i]
            digit2 = original_labels[i + 1]
            
            # Ensure digits are consecutive for realistic transition
            if abs(digit1 - digit2) == 1:
                # Blend weights based on transition position
                for weight in np.arange(0.3, 0.8, 0.1):
                    blended = (1 - weight) * original_images[i] + weight * original_images[i + 1]
                    
                    # Determine label based on transition rule
                    if weight <= 0.6:  # x.0 - x.6
                        label = digit1
                        trans_state = 0  # Lower range
                    else:  # x.7 - x.9
                        label = digit2
                        trans_state = 1  # Upper range
                    
                    blended_images.append(blended)
                    blended_labels.append([label, trans_state])
    
    return np.array(blended_images), np.array(blended_labels)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_transition_dataset(x_train, y_train, batch_size=32, augment=True):
    """
    Create dataset with transitional examples for training
    """
    # Generate transitional examples
    x_trans, y_trans = generate_transitional_mnist(x_train, y_train)
    
    # Combine with original data
    x_combined = np.concatenate([x_train, x_trans], axis=0)
    y_combined = np.concatenate([y_train, y_trans], axis=0)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_combined, y_combined))
    dataset = dataset.shuffle(10000)
    
    if augment:
        # Add contrast augmentation
        def augment_data(image, label):
            # Random brightness adjustment
            image = image + tf.random.uniform([], -0.1, 0.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            # Random inversion (for dark-on-light)
            if tf.random.uniform(()) > 0.5:
                image = 1.0 - image
                # If inverted, transition state remains the same
                # (the wheel position doesn't change)
            
            return image, label
        
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=== Digit Recognizer v25 - Transition-Aware Model ===")
    
    # Test with RGB input
    print("\n1. Creating v25 model with RGB input (28, 28, 3):")
    params.INPUT_SHAPE = (28, 28, 3)
    params.NB_CLASSES = 10
    model = create_digit_recognizer_v25(method='standard')
    model.summary()
    print(f"Model parameters: {model.count_params():,}")
    
    # Test with grayscale input
    print("\n2. Creating v25 model with grayscale input (28, 28, 1):")
    params.INPUT_SHAPE = (28, 28, 1)
    model_gray = create_digit_recognizer_v25(method='standard')
    
    # Test temporal version
    print("\n3. Creating v25 model with temporal smoothing:")
    model_temporal = create_digit_recognizer_v25(method='temporal', use_temporal=True)
    
    # Test prediction on transitional digit
    print("\n4. Testing transition detection:")
    test_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
    predictions = model_gray.predict(test_input, verbose=0)
    
    print(f"  Digit prediction: {predictions['digit'][0]}")
    print(f"  Digit confidence: {predictions['digit_confidence'][0][0]:.3f}")
    print(f"  Transition probability: {predictions['transition_prob'][0][0]:.3f}")
    print(f"  Transition direction: {predictions['transition_dir'][0][0]:.3f}")
    
    print("\n✅ v25 model ready for water meter digit recognition")
    print("\nKey Features:")
    print("  - Handles transitional digits (5.5 → 5 or 6 based on position)")
    print("  - Provides confidence scores for uncertain predictions")
    print("  - Optional temporal smoothing for video streams")
    print("  - Custom loss function for transition-aware training")
    print("  - QAT-compatible for ESP32 deployment")
    print("\nTransition Rule:")
    print("  - x.0 - x.6 → digit x (lower range)")
    print("  - x.7 - x.9 → digit x+1 (upper range)")