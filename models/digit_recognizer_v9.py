# models/digit_recognizer_v9.py
import tensorflow as tf
import parameters as params
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

'''
V9 (EfficientNet-style):

    Mobile inverted bottleneck blocks

    Depthwise separable convolutions

    Squeeze-and-Excitation in each block

    Optimized for efficiency/accuracy balance
'''

class SwishLayer(layers.Layer):
    """Custom layer for Swish activation"""
    def call(self, inputs):
        return tf.nn.swish(inputs)

class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block as a separate layer"""
    def __init__(self, se_ratio=0.25, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.se_ratio = se_ratio
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.se_filters = max(1, int(self.channels * self.se_ratio))
        
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(self.se_filters, activation='relu')
        self.dense2 = layers.Dense(self.channels, activation='sigmoid')
        
    def call(self, inputs):
        # Squeeze
        se = self.gap(inputs)
        # Excitation
        se = self.dense1(se)
        se = self.dense2(se)
        # Reshape for broadcasting
        se = tf.reshape(se, [-1, 1, 1, self.channels])
        # Scale
        return inputs * se

def create_digit_recognizer_v9():
    """
    EfficientNet-style architecture for optimal performance/accuracy balance
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    def mb_conv_block(x, filters, stride=1, expansion=4, se_ratio=0.25, name=""):
        """Mobile Inverted Residual Bottleneck with Squeeze-and-Excitation"""
        channels_in = K.int_shape(x)[-1]
        expanded = filters * expansion
        
        # Store input for residual connection
        input_tensor = x
        
        # Expansion phase
        if expansion > 1:
            x = layers.Conv2D(expanded, 1, padding='same', use_bias=False, 
                             name=f'{name}_expand_conv')(x)
            x = layers.BatchNormalization(name=f'{name}_expand_bn')(x)
            x = SwishLayer(name=f'{name}_swish1')(x)
        
        # Depthwise convolution
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                                  name=f'{name}_depthwise_conv')(x)
        x = layers.BatchNormalization(name=f'{name}_depthwise_bn')(x)
        x = SwishLayer(name=f'{name}_swish2')(x)
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            x = SEBlock(se_ratio=se_ratio, name=f'{name}_se')(x)
        
        # Projection
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                         name=f'{name}_project_conv')(x)
        x = layers.BatchNormalization(name=f'{name}_project_bn')(x)
        
        # Residual connection
        if stride == 1 and channels_in == filters:
            x = layers.Add(name=f'{name}_residual')([x, input_tensor])
            
        return x
    
    # Stem - adjusted for 32x20 input (use stride=1 instead of 2)
    x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                     name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = SwishLayer(name='stem_swish')(x)
    
    # MBConv blocks - optimized for 32x20 input
    x = mb_conv_block(x, 16, stride=1, expansion=1, se_ratio=0, name='block1')
    x = mb_conv_block(x, 24, stride=2, expansion=4, se_ratio=0.25, name='block2')
    x = mb_conv_block(x, 32, stride=1, expansion=4, se_ratio=0.25, name='block3')
    x = mb_conv_block(x, 48, stride=1, expansion=4, se_ratio=0.25, name='block4')
    
    # Head
    x = layers.Conv2D(128, 1, padding='same', use_bias=False, name='head_conv')(x)
    x = layers.BatchNormalization(name='head_bn')(x)
    x = SwishLayer(name='head_swish')(x)
    
    # Final pooling and classification
    x = layers.GlobalAveragePooling2D(name='final_gap')(x)
    x = layers.Dropout(0.3, name='final_dropout')(x)
    
    # Classification head
    x = layers.Dense(96, name='classifier_dense1')(x)
    x = SwishLayer(name='classifier_swish')(x)
    x = layers.BatchNormalization(name='classifier_bn')(x)
    x = layers.Dropout(0.2, name='classifier_dropout')(x)
    
    outputs = layers.Dense(params.NB_CLASSES, activation='softmax', name='output')(x)
    
    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v9")

# Alternative simpler version without SE blocks if you still have issues
def create_digit_recognizer_v9_simple():
    """
    Simplified EfficientNet-style without SE blocks for maximum compatibility
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    def mb_conv_block_simple(x, filters, stride=1, expansion=4, name=""):
        """Mobile Inverted Residual Bottleneck without SE"""
        channels_in = K.int_shape(x)[-1]
        expanded = filters * expansion
        
        # Store input for residual connection
        input_tensor = x
        
        # Expansion phase
        if expansion > 1:
            x = layers.Conv2D(expanded, 1, padding='same', use_bias=False, 
                             name=f'{name}_expand_conv')(x)
            x = layers.BatchNormalization(name=f'{name}_expand_bn')(x)
            x = SwishLayer(name=f'{name}_swish1')(x)
        
        # Depthwise convolution
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                                  name=f'{name}_depthwise_conv')(x)
        x = layers.BatchNormalization(name=f'{name}_depthwise_bn')(x)
        x = SwishLayer(name=f'{name}_swish2')(x)
        
        # Projection
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                         name=f'{name}_project_conv')(x)
        x = layers.BatchNormalization(name=f'{name}_project_bn')(x)
        
        # Residual connection
        if stride == 1 and channels_in == filters:
            x = layers.Add(name=f'{name}_residual')([x, input_tensor])
            
        return x
    
    # Stem
    x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                     name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = SwishLayer(name='stem_swish')(x)
    
    # MBConv blocks
    x = mb_conv_block_simple(x, 16, stride=1, expansion=1, name='block1')
    x = mb_conv_block_simple(x, 24, stride=2, expansion=4, name='block2')
    x = mb_conv_block_simple(x, 32, stride=1, expansion=4, name='block3')
    x = mb_conv_block_simple(x, 48, stride=1, expansion=4, name='block4')
    x = mb_conv_block_simple(x, 64, stride=1, expansion=6, name='block5')
    
    # Head
    x = layers.Conv2D(128, 1, padding='same', use_bias=False, name='head_conv')(x)
    x = layers.BatchNormalization(name='head_bn')(x)
    x = SwishLayer(name='head_swish')(x)
    
    # Final pooling and classification
    x = layers.GlobalAveragePooling2D(name='final_gap')(x)
    x = layers.Dropout(0.3, name='final_dropout')(x)
    
    # Classification head
    x = layers.Dense(96, activation='swish', name='classifier_dense1')(x)
    x = layers.BatchNormalization(name='classifier_bn')(x)
    x = layers.Dropout(0.2, name='classifier_dropout')(x)
    
    outputs = layers.Dense(params.NB_CLASSES, activation='softmax', name='output')(x)
    
    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v9_simple")

if __name__ == "__main__":
    model = create_digit_recognizer_v9()
    print(f"Created model: {model.name}")
    model.build((None,) + params.INPUT_SHAPE)
    model.summary()
    print(f"Total parameters: {model.count_params():,}")
    
    # Also test the simple version
    print("\n" + "="*50)
    model_simple = create_digit_recognizer_v9_simple()
    print(f"Created simple model: {model_simple.name}")
    model_simple.build((None,) + params.INPUT_SHAPE)
    model_simple.summary()
    print(f"Simple model parameters: {model_simple.count_params():,}")