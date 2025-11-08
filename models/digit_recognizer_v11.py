# models/digit_recognizer_v11.py
import tensorflow as tf
import parameters as params
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

'''
1. Advanced Activations:

    GELU: Gaussian Error Linear Unit (used in Transformers)

    Swish: x * sigmoid(x) (better than ReLU, used in EfficientNet)

2. Attention Mechanisms:

    Squeeze-and-Excitation (SE): Channel-wise attention

    QAT-optimized: Works with quantization

3. Modern Blocks:

    MBConv: Mobile Inverted Bottleneck (EfficientNet style)

    Stochastic Depth: Randomly skip layers during training

4. Regularization:

    Stochastic Depth: Better than Dropout for deep networks

    Multi-scale features: GAP + GMP + spatial attention

🎯 Expected Improvements:
Component	Benefit	QAT Compatible
GELU	Smoother gradients, better convergence	✅
Swish	Non-monotonic, better than ReLU	✅
SE Blocks	Channel attention, +1-2% accuracy	✅
MBConv	Better efficiency/accuracy trade-off	✅
Stochastic Depth	Regularization like Dropout	✅
'''

class QATFriendlyGELU(layers.Layer):
    """GELU activation that's quantization-friendly using approximation"""
    def call(self, inputs):
        # GELU approximation that works well with quantization
        return 0.5 * inputs * (1.0 + tf.math.tanh(
            tf.math.sqrt(2.0 / tf.constant(3.141592653589793)) * 
            (inputs + 0.044715 * tf.math.pow(inputs, 3))
        ))

class QATFriendlySwish(layers.Layer):
    """Swish activation that's quantization-friendly"""
    def call(self, inputs):
        # Swish: x * sigmoid(x)
        return inputs * tf.math.sigmoid(inputs)

class QATSEBlock(layers.Layer):
    """Squeeze-and-Excitation block optimized for QAT"""
    def __init__(self, se_ratio=16, activation='relu', **kwargs):
        super(QATSEBlock, self).__init__(**kwargs)
        self.se_ratio = se_ratio
        self.activation = activation
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.reduced_filters = max(1, int(self.channels // self.se_ratio))
        
        # Use GlobalAveragePooling2D layer instead of tf operation
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.dense_reduce = layers.Dense(self.reduced_filters, activation=self.activation)
        self.dense_expand = layers.Dense(self.channels, activation='sigmoid')
        
    def call(self, inputs):
        # Squeeze
        se = self.gap(inputs)
        # Excitation
        se = self.dense_reduce(se)
        se = self.dense_expand(se)
        # Scale (broadcasting happens automatically)
        return inputs * se

class MBConvBlock(layers.Layer):
    """Mobile Inverted Bottleneck Conv Block with modern features"""
    def __init__(self, filters, stride=1, expansion=4, se_ratio=0.25, 
                 activation='swish', use_se=True, **kwargs):
        super(MBConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.activation_type = activation
        self.use_se = use_se
        
    def build(self, input_shape):
        input_filters = input_shape[-1]
        self.expanded_filters = input_filters * self.expansion
        
        # Expansion phase
        if self.expansion > 1:
            self.expand_conv = layers.Conv2D(self.expanded_filters, 1, 
                                           padding='same', use_bias=False)
            self.expand_bn = layers.BatchNormalization()
            self.expand_act = self._get_activation()
        
        # Depthwise convolution
        self.depthwise_conv = layers.DepthwiseConv2D(3, strides=self.stride,
                                                   padding='same', use_bias=False)
        self.depthwise_bn = layers.BatchNormalization()
        self.depthwise_act = self._get_activation()
        
        # Squeeze-and-Excitation
        if self.use_se and self.se_ratio > 0:
            self.se_block = QATSEBlock(se_ratio=int(1/self.se_ratio))
        
        # Projection
        self.project_conv = layers.Conv2D(self.filters, 1, 
                                        padding='same', use_bias=False)
        self.project_bn = layers.BatchNormalization()
        
        # Skip connection
        if self.stride == 1 and input_filters == self.filters:
            self.use_skip = True
        else:
            self.use_skip = False
            
    def _get_activation(self):
        if self.activation_type == 'swish':
            return QATFriendlySwish()
        elif self.activation_type == 'gelu':
            return QATFriendlyGELU()
        else:
            return layers.ReLU()
            
    def call(self, inputs):
        x = inputs
        
        # Expansion
        if self.expansion > 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_act(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)
        
        # SE attention
        if hasattr(self, 'se_block'):
            x = self.se_block(x)
        
        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Skip connection
        if self.use_skip:
            x = layers.Add()([x, inputs])
            
        return x

class StochasticDepth(layers.Layer):
    """Stochastic Depth (DropPath) for regularization"""
    def __init__(self, drop_rate=0.1, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_rate = drop_rate
        
    def call(self, inputs, training=None):
        if not training or self.drop_rate == 0.0:
            return inputs
            
        keep_prob = 1.0 - self.drop_rate
        batch_size = tf.shape(inputs)[0]
        random_tensor = keep_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        return inputs * binary_tensor / keep_prob

def create_digit_recognizer_v11():
    """
    Modern SOTA architecture with GELU, Swish, SE blocks, and MBConv
    Optimized for both accuracy and QAT compatibility
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Stem with modern activations
    x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False, 
                     name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = QATFriendlySwish(name='stem_swish')(x)
    
    # Stage 1: MBConv blocks with SE attention
    x = MBConvBlock(32, stride=1, expansion=1, activation='swish', 
                   use_se=True, name='mbconv1')(x)
    x = StochasticDepth(0.05, name='stochastic_depth1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    x = layers.Dropout(0.1, name='drop1')(x)
    
    # Stage 2: Increased capacity
    x = MBConvBlock(64, stride=1, expansion=4, activation='swish',
                   use_se=True, name='mbconv2')(x)
    x = StochasticDepth(0.1, name='stochastic_depth2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = layers.Dropout(0.2, name='drop2')(x)
    
    # Stage 3: Feature refinement
    x = MBConvBlock(128, stride=1, expansion=4, activation='swish',
                   use_se=True, name='mbconv3')(x)
    x = StochasticDepth(0.15, name='stochastic_depth3')(x)
    x = MBConvBlock(128, stride=1, expansion=4, activation='gelu',
                   use_se=True, name='mbconv4')(x)
    x = layers.Dropout(0.3, name='drop3')(x)
    
    # Stage 4: High-level features
    x = MBConvBlock(256, stride=1, expansion=6, activation='gelu',
                   use_se=True, name='mbconv5')(x)
    x = layers.Dropout(0.4, name='drop4')(x)
    
    # Multi-scale feature aggregation
    # Global context branches
    gap = layers.GlobalAveragePooling2D(name='gap')(x)
    gmp = layers.GlobalMaxPooling2D(name='gmp')(x)
    
    # Spatial attention branch
    spatial_att = layers.Conv2D(64, 1, activation='relu', name='spatial_conv')(x)
    spatial_att = layers.GlobalAveragePooling2D(name='spatial_pool')(spatial_att)
    
    # Feature fusion
    x = layers.Concatenate(name='feature_fusion')([gap, gmp, spatial_att])
    x = layers.Dropout(0.5, name='fusion_dropout')(x)
    
    # Modern classification head with GELU
    x = layers.Dense(256, name='dense1')(x)
    x = layers.BatchNormalization(name='dense_bn1')(x)
    x = QATFriendlyGELU(name='dense_gelu1')(x)
    x = layers.Dropout(0.4, name='dense_drop1')(x)
    
    x = layers.Dense(128, name='dense2')(x)
    x = layers.BatchNormalization(name='dense_bn2')(x)
    x = QATFriendlySwish(name='dense_swish1')(x)
    x = layers.Dropout(0.3, name='dense_drop2')(x)
    
    x = layers.Dense(64, name='dense3')(x)
    x = QATFriendlyGELU(name='dense_gelu2')(x)
    
    outputs = layers.Dense(params.NB_CLASSES, activation='softmax', name='output')(x)
    
    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v11")

def create_digit_recognizer_v11_light():
    """
    Lightweight version with selective modern components
    Better QAT compatibility while keeping modern improvements
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Stem with Swish
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = QATFriendlySwish(name='stem_swish')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    # Simple MBConv blocks with SE
    x = MBConvBlock(64, stride=1, expansion=4, activation='swish', 
                   use_se=True, name='mbconv1')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = layers.Dropout(0.2, name='drop1')(x)
    
    x = MBConvBlock(128, stride=1, expansion=4, activation='swish',
                   use_se=True, name='mbconv2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)
    
    # Global context
    gap = layers.GlobalAveragePooling2D(name='gap')(x)
    gmp = layers.GlobalMaxPooling2D(name='gmp')(x)
    x = layers.Concatenate(name='concat_pool')([gap, gmp])
    x = layers.Dropout(0.4, name='fusion_dropout')(x)
    
    # Classifier with GELU
    x = layers.Dense(128, name='dense1')(x)
    x = QATFriendlyGELU(name='dense_gelu')(x)
    x = layers.Dropout(0.3, name='dense_drop')(x)
    
    x = layers.Dense(64, name='dense2')(x)
    x = QATFriendlySwish(name='dense_swish')(x)
    
    outputs = layers.Dense(params.NB_CLASSES, activation='softmax', name='output')(x)
    
    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v11_light")

# Test modern components
def test_modern_components():
    """Test that all modern components work correctly"""
    print("🧪 Testing Modern Components...")
    
    # Test activations
    test_input = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    gelu_layer = QATFriendlyGELU()
    swish_layer = QATFriendlySwish()
    
    gelu_output = gelu_layer(test_input)
    swish_output = swish_layer(test_input)
    
    print(f"✓ GELU works: {gelu_output.numpy()}")
    print(f"✓ Swish works: {swish_output.numpy()}")
    
    # Test SE Block
    se_block = QATSEBlock(se_ratio=16)
    test_feature = tf.ones((1, 8, 5, 64))  # Simulate feature map
    se_output = se_block(test_feature)
    print(f"✓ SE Block works: input {test_feature.shape} -> output {se_output.shape}")
    
    # Test MBConv
    mbconv = MBConvBlock(32, expansion=4)
    mbconv_output = mbconv(test_feature)
    print(f"✓ MBConv works: input {test_feature.shape} -> output {mbconv_output.shape}")
    
    print("🎯 All modern components tested successfully!")

if __name__ == "__main__":
    print("Testing digit_recognizer_v11 with modern components...")
    
    # Test components first
    test_modern_components()
    
    print("\n" + "="*60)
    model = create_digit_recognizer_v11()
    model.build((None,) + params.INPUT_SHAPE)
    model.summary()
    print(f"Total parameters: {model.count_params():,}")
    
    print("\n" + "="*60)
    print("Testing light version...")
    model_light = create_digit_recognizer_v11_light()
    model_light.build((None,) + params.INPUT_SHAPE)
    model_light.summary()
    print(f"Light version parameters: {model_light.count_params():,}")