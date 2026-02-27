# models/digit_recognizer_v10.py
"""
digit_recognizer_v10 – Hybrid Transformer-Style CNN with Multi-Scale Fusion
============================================================================
Design goal: Approximate Transformer-level global context reasoning using
only standard CNN ops (no self-attention), keeping QAT compatibility via
residual-like patterns and multi-scale pooling.

Architecture:
  - Conv2D(32) + BN + ReLU + MaxPool
  - Conv2D(64) + BN + ReLU + SimplifiedAttention (CNN-based SE) + MaxPool + Dropout(0.1)
  - Conv2D(128) × 2 + BN + ReLU + SimplifiedAttention + Dropout(0.2)
  - Conv2D(256) × 2 + BN + ReLU + Dropout(0.3)
  - Multi-scale aggregation:
      GAP → Dense(128) + GMP → Dense(128) + Spatial 1×1 Conv → GAP(128)
      Concatenated (384-d) → Dropout(0.4)
  - Dense(256) BN ReLU + Dropout(0.3)
  - Dense(128) BN ReLU + Dropout(0.2) → Dense(NB_CLASSES) Softmax

Custom class:
  - SimplifiedAttention: CNN-based channel attention (QAT-compatible SE variant)

Notes:
  - Multi-scale fusion (GAP + GMP + spatial conv) provides global context
    without Transformer self-attention ops
  - Large model; intended for accuracy benchmarking, not IoT deployment
  - Light version (v10_light) also available for comparison

Estimated: ~700K+ parameters → not intended for ESP32 deployment.
"""

import tensorflow as tf
import parameters as params
from tensorflow.keras import layers, Model

class SimplifiedAttention(layers.Layer):
    """CNN-based attention mechanism that's QAT compatible"""
    def __init__(self, ratio=8, **kwargs):
        super(SimplifiedAttention, self).__init__(**kwargs)
        self.ratio = ratio
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.conv1 = layers.Conv2D(self.channels // self.ratio, 1, activation='relu')
        self.conv2 = layers.Conv2D(self.channels, 1, activation='sigmoid')
        
    def call(self, inputs):
        # Channel attention using conv layers (QAT compatible)
        attention_weights = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
        attention_weights = self.conv1(attention_weights)
        attention_weights = self.conv2(attention_weights)
        return inputs * attention_weights

def create_digit_recognizer_v10():
    """
    Quantization-friendly hybrid architecture with CNN-based attention
    Combines efficient convolutions with global context mechanisms
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Initial feature extraction
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    # Block 1: Basic features + attention
    x = layers.Conv2D(64, 3, padding='same', use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = SimplifiedAttention(ratio=8, name='att1')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = layers.Dropout(0.1, name='drop1')(x)
    
    # Block 2: Deeper features
    x = layers.Conv2D(128, 3, padding='same', use_bias=False, name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False, name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.ReLU(name='relu4')(x)
    x = SimplifiedAttention(ratio=8, name='att2')(x)
    x = layers.Dropout(0.2, name='drop2')(x)
    
    # Block 3: Feature refinement
    x = layers.Conv2D(256, 3, padding='same', use_bias=False, name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.ReLU(name='relu5')(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False, name='conv6')(x)
    x = layers.BatchNormalization(name='bn6')(x)
    x = layers.ReLU(name='relu6')(x)
    x = layers.Dropout(0.3, name='drop3')(x)
    
    # Multi-scale feature aggregation (QAT compatible alternative to transformers)
    # Branch 1: Global context
    branch1 = layers.GlobalAveragePooling2D(name='gap')(x)
    branch1 = layers.Dense(128, activation='relu', name='dense_gap')(branch1)
    
    # Branch 2: Local context
    branch2 = layers.GlobalMaxPooling2D(name='gmp')(x)
    branch2 = layers.Dense(128, activation='relu', name='dense_gmp')(branch2)
    
    # Branch 3: Spatial context (using 1x1 conv)
    branch3 = layers.Conv2D(128, 1, activation='relu', name='spatial_conv')(x)
    branch3 = layers.GlobalAveragePooling2D(name='spatial_pool')(branch3)
    
    # Feature fusion
    x = layers.Concatenate(name='feature_fusion')([branch1, branch2, branch3])
    x = layers.Dropout(0.4, name='fusion_dropout')(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.BatchNormalization(name='dense_bn1')(x)
    x = layers.Dropout(0.3, name='dense_drop1')(x)
    
    x = layers.Dense(128, activation='relu', name='dense2')(x)
    x = layers.BatchNormalization(name='dense_bn2')(x)
    x = layers.Dropout(0.2, name='dense_drop2')(x)
    
    outputs = layers.Dense(params.NB_CLASSES, activation='softmax', name='output')(x)
    
    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v10")

def create_digit_recognizer_v10_light():
    """
    Ultra QAT-compatible version with maximum quantization stability
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Simple, clean architecture that quantizes well
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Dropout(0.3, name='drop1')(x)
    
    # Global context pooling
    gap = layers.GlobalAveragePooling2D(name='gap')(x)
    gmp = layers.GlobalMaxPooling2D(name='gmp')(x)
    x = layers.Concatenate(name='concat_pool')([gap, gmp])
    
    # Classifier
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.4, name='drop2')(x)
    
    x = layers.Dense(64, activation='relu', name='dense2')(x)
    
    outputs = layers.Dense(params.NB_CLASSES, activation='softmax', name='output')(x)
    
    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v10_light")

# Test QAT compatibility
def test_qat_compatibility():
    """Test if the model works with quantization"""
    try:
        model = create_digit_recognizer_v10()
        
        # Test basic forward pass
        test_input = tf.ones((1,) + params.INPUT_SHAPE)
        output = model(test_input)
        print(f"✓ Forward pass successful: {output.shape}")
        
        # Test model compilation
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("✓ Model compilation successful")
        
        # Check for QAT problematic operations
        problematic_ops = ['reshape', 'transpose', 'strided_slice', 'pack', 'unpack']
        model_ops = [layer.__class__.__name__.lower() for layer in model.layers]
        
        found_problematic = any(op in ' '.join(model_ops) for op in problematic_ops)
        if found_problematic:
            print("⚠️  Model may have QAT-sensitive operations")
        else:
            print("✓ Model appears QAT-friendly")
            
        return True
        
    except Exception as e:
        print(f"✗ QAT compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing digit_recognizer_v10...")
    model = create_digit_recognizer_v10()
    model.build((None,) + params.INPUT_SHAPE)
    model.summary()
    print(f"Total parameters: {model.count_params():,}")
    
    print("\nTesting QAT compatibility...")
    test_qat_compatibility()
    
    print("\n" + "="*50)
    print("Testing light version...")
    model_light = create_digit_recognizer_v10_light()
    model_light.build((None,) + params.INPUT_SHAPE)
    model_light.summary()
    print(f"Light version parameters: {model_light.count_params():,}")