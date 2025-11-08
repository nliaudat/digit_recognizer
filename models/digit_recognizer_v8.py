# models/digit_recognizer_v8.py
import tensorflow as tf
import parameters as params
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

'''    

V8 (SOTA Residual):

    Residual connections with pre-activation

    Squeeze-and-Excitation attention modules

    Multi-scale feature fusion

    Swish activation + BatchNorm
'''

class AttentionModule(layers.Layer):
    """Squeeze-and-Excitation Attention for channel-wise feature recalibration"""
    def __init__(self, ratio=16, **kwargs):
        super(AttentionModule, self).__init__(**kwargs)
        self.ratio = ratio
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.global_avg = layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = layers.Dense(self.channels // self.ratio, activation='relu')
        self.dense2 = layers.Dense(self.channels, activation='sigmoid')
        
    def call(self, inputs):
        # Squeeze
        x = self.global_avg(inputs)
        # Excitation
        x = self.dense1(x)
        x = self.dense2(x)
        # Scale
        return inputs * x

class ResidualUnit(layers.Layer):
    """Modern residual unit with pre-activation"""
    def __init__(self, filters, kernel_size=3, stride=1, **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        
    def build(self, input_shape):
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(self.filters, self.kernel_size, strides=self.stride, 
                                  padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.filters, self.kernel_size, padding='same', use_bias=False)
        
        # Shortcut connection if dimensions change
        if self.stride > 1 or input_shape[-1] != self.filters:
            self.shortcut = layers.Conv2D(self.filters, 1, strides=self.stride, 
                                        padding='same', use_bias=False)
        else:
            self.shortcut = layers.Lambda(lambda x: x)
            
    def call(self, inputs):
        # Pre-activation pattern
        x = self.bn1(inputs)
        x = layers.ReLU()(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        
        shortcut = self.shortcut(inputs)
        return x + shortcut

class SwishLayer(layers.Layer):
    """Custom layer for Swish activation"""
    def call(self, inputs):
        return tf.nn.swish(inputs)

def create_digit_recognizer_v8():
    """
    State-of-the-art digit recognizer with residual connections and attention
    Target: Maximum accuracy on 32x20 images
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Initial stem with aggressive feature extraction
    x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                     kernel_initializer='he_normal', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)
    
    # Residual blocks with attention
    # Block 1
    x = ResidualUnit(32, name='res1')(x)
    x = AttentionModule(ratio=8, name='att1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    x = layers.Dropout(0.1, name='drop1')(x)
    
    # Block 2
    x = ResidualUnit(64, stride=1, name='res2')(x)
    x = AttentionModule(ratio=8, name='att2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = layers.Dropout(0.2, name='drop2')(x)
    
    # Block 3
    x = ResidualUnit(128, stride=1, name='res3')(x)
    x = AttentionModule(ratio=8, name='att3')(x)
    x = layers.Dropout(0.3, name='drop3')(x)
    
    # Multi-scale feature aggregation
    # Branch 1: Global context
    branch1 = layers.GlobalAveragePooling2D(name='gap')(x)
    branch1 = layers.Dense(64, activation='relu', name='gap_dense')(branch1)
    
    # Branch 2: Local context
    branch2 = layers.Conv2D(64, 1, padding='same', activation='relu', name='local_conv')(x)
    branch2 = layers.GlobalMaxPooling2D(name='gmp')(branch2)
    
    # Combine branches
    x = layers.Concatenate(name='feature_fusion')([branch1, branch2])
    
    # Advanced classification head
    x = layers.Dense(128, activation=None, name='dense1')(x)
    x = layers.BatchNormalization(name='dense_bn1')(x)
    x = SwishLayer(name='swish1')(x)
    x = layers.Dropout(0.4, name='dense_drop1')(x)
    
    x = layers.Dense(64, activation=None, name='dense2')(x)
    x = layers.BatchNormalization(name='dense_bn2')(x)
    x = SwishLayer(name='swish2')(x)
    x = layers.Dropout(0.3, name='dense_drop2')(x)
    
    # Output with label smoothing support
    outputs = layers.Dense(params.NB_CLASSES, activation='softmax', 
                          name='output')(x)
    
    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v8")

if __name__ == "__main__":
    model = create_digit_recognizer_v8()
    print(f"Created model: {model.name}")
    model.build((None,) + params.INPUT_SHAPE)
    model.summary()
    print(f"Total parameters: {model.count_params():,}")