# models/digit_recognizer_v38.py
"""
digit_recognizer_v38 – RepVGG-style Reparameterizable IoT Model
================================================================
Design: Multi-branch training for richer gradients, which can be
mathematically folded into a single VGG-like conv block for inference.
Zero inference overhead while benefiting from complex training dynamics.

Call `model.reparameterize()` before TFLite export!
"""

import tensorflow as tf
import numpy as np
import config as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


class RepVGGBlock(tf.keras.layers.Layer):
    """
    A RepVGG block.
    During training: uses parallel 3x3 conv, 1x1 conv, and identity branches.
    During inference (after reparameterization): uses a single 3x3 conv.
    """
    def __init__(self, in_channels, out_channels, stride=1, deploy=False, name_prefix='rep', **kwargs):
        super(RepVGGBlock, self).__init__(name=name_prefix, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy = deploy
        self.name_prefix = name_prefix

        if self.deploy:
            self.rbr_reparam = tf.keras.layers.Conv2D(
                filters=out_channels, kernel_size=3, strides=stride,
                padding='same', use_bias=True, name=f'{name_prefix}_reparam_conv'
            )
        else:
            self.rbr_dense = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters=out_channels, kernel_size=3, strides=stride,
                    padding='same', use_bias=False, name=f'{name_prefix}_dense_conv'
                ),
                tf.keras.layers.BatchNormalization(name=f'{name_prefix}_dense_bn')
            ], name=f'{name_prefix}_dense_branch')

            self.rbr_1x1 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters=out_channels, kernel_size=1, strides=stride,
                    padding='same', use_bias=False, name=f'{name_prefix}_1x1_conv'
                ),
                tf.keras.layers.BatchNormalization(name=f'{name_prefix}_1x1_bn')
            ], name=f'{name_prefix}_1x1_branch')

            if out_channels == in_channels and stride == 1:
                self.rbr_identity = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_id_bn')
            else:
                self.rbr_identity = None

        self.relu = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_relu6')

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride,
            "deploy": self.deploy,
            "name_prefix": self.name_prefix,
        })
        return config

    def call(self, inputs, training=None):
        if hasattr(self, 'rbr_reparam'):
            return self.relu(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs, training=training)

        return self.relu(self.rbr_dense(inputs, training=training) +
                         self.rbr_1x1(inputs, training=training) +
                         id_out)

    def get_equivalent_kernel_bias(self):
        """Fuses the 3 branches into a single 3x3 kernel and bias."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense.layers[0], self.rbr_dense.layers[1])
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1.layers[0], self.rbr_1x1.layers[1])
        kernelid, biasid = self._fuse_bn_tensor(None, self.rbr_identity)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(kernel1x1, [[1, 1], [1, 1], [0, 0], [0, 0]])

    def _fuse_bn_tensor(self, conv_layer, bn_layer):
        if bn_layer is None:
            return 0, 0
        if conv_layer is None:
            # Identity branch: construct a 3x3 kernel with 1s in the center
            input_dim = self.in_channels
            kernel_value = np.zeros((3, 3, input_dim, input_dim), dtype=np.float32)
            for i in range(input_dim):
                kernel_value[1, 1, i, i] = 1
            kernel = tf.constant(kernel_value)
        else:
            kernel = conv_layer.kernel

        gamma = bn_layer.gamma
        beta = bn_layer.beta
        moving_mean = bn_layer.moving_mean
        moving_variance = bn_layer.moving_variance
        eps = bn_layer.epsilon

        std = tf.sqrt(moving_variance + eps)
        t = gamma / std
        
        # Reshape t to broadcast with kernel
        t_reshaped = tf.reshape(t, (1, 1, 1, -1))
        
        fused_kernel = kernel * t_reshaped
        fused_bias = beta - moving_mean * t
        return fused_kernel, fused_bias

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = tf.keras.layers.Conv2D(
            filters=self.out_channels, kernel_size=3, strides=self.stride,
            padding='same', use_bias=True, name=f'{self.name_prefix}_reparam_conv'
        )
        # Build the layer by calling it on dummy input
        dummy_input = tf.keras.Input(shape=(None, None, self.in_channels))
        self.rbr_reparam(dummy_input)
        self.rbr_reparam.set_weights([kernel.numpy(), bias.numpy()])
        
        # Delete training branches
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, 'rbr_identity'):
            del self.rbr_identity
        self.deploy = True


class RepVGGModel(tf.keras.Model):
    def reparameterize(self):
        for layer in self.layers:
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
        print("Model reparameterized successfully for deployment!")


def create_digit_recognizer_v38(num_classes=None, input_shape=None, deploy=False, **kwargs):
    """
    RepVGG-style IoT digit recognizer.
    """
    if num_classes is None:
        num_classes = params.NB_CLASSES
    if input_shape is None:
        input_shape = params.INPUT_SHAPE

    inputs = tf.keras.Input(shape=input_shape, name='input')

    # Keep channels small to stay under 100KB INT8 when fused
    # The actual FLOPs/params during training are ~3x higher
    
    # 32x20
    x = RepVGGBlock(input_shape[-1], 24, stride=1, deploy=deploy, name_prefix='rep_entry')(inputs)
    
    # 16x10
    x = RepVGGBlock(24, 32, stride=2, deploy=deploy, name_prefix='rep_s1_b1')(x)
    x = RepVGGBlock(32, 32, stride=1, deploy=deploy, name_prefix='rep_s1_b2')(x)
    
    # 8x5
    x = RepVGGBlock(32, 48, stride=2, deploy=deploy, name_prefix='rep_s2_b1')(x)
    x = RepVGGBlock(48, 48, stride=1, deploy=deploy, name_prefix='rep_s2_b2')(x)
    
    # Deeper
    x = RepVGGBlock(48, 64, stride=1, deploy=deploy, name_prefix='rep_s3_b1')(x)
    
    # Head expansion
    x = tf.keras.layers.Conv2D(
        96, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='head_relu6')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='gap')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)

    if params.USE_LOGITS:
        outputs = tf.keras.layers.Dense(
            num_classes, activation=None, name='logits'
        )(x)
    else:
        outputs = tf.keras.layers.Dense(
            num_classes, activation='softmax', name='output'
        )(x)

    model = RepVGGModel(inputs, outputs, name='digit_recognizer_v38')
    return model


def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v38()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
        
    print("WARNING: RepVGG models should be reparameterized BEFORE QAT.")
    print("If you are running QAT directly, ensure the model is in deploy mode.")
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("QAT model created for digit_recognizer_v38")
        return qat_model
    except Exception as e:
        print(f"QAT failed ({e}) – returning base model.")
        return base_model


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    m = create_digit_recognizer_v38(deploy=False)
    m.summary()
    p_train = m.count_params()
    
    # Test reparameterization
    dummy = tf.random.uniform((1, 32, 20, 3))
    y_before = m(dummy, training=False)
    m.reparameterize()
    y_after = m(dummy, training=False)
    
    print("\nReparameterized Model Summary:")
    m.summary()
    p_deploy = m.count_params()
    
    diff = tf.reduce_max(tf.abs(y_before - y_after)).numpy()
    print(f"\nTraining parameters: {p_train:,}")
    print(f"Deploy parameters: {p_deploy:,}")
    print(f"Estimated INT8 KB (deploy): ~{p_deploy * 1.1 / 1024:.1f}")
    print(f"Max difference after reparam: {diff}")
    assert diff < 1e-4, "Reparameterization mismatch!"
    print("Reparameterization mathematical check passed.")
