# utils/losses.py
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import parameters as params

def sparse_focal_loss(gamma=2.0, alpha=0.25, from_logits=params.USE_LOGITS):
    """
    Focal Loss for sparse categorical labels.
    gamma=2.0 (standard), alpha=0.25 (balances hard/easy examples)
    
    This implementation handles sparse (integer) labels by converting them to one-hot 
    internally for the focal weight calculation.
    """
    def loss(y_true, y_pred):
        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            
        # Convert sparse labels to one-hot
        y_true_int = tf.cast(y_true, tf.int32)
        if len(y_true_int.shape) > 1 and y_true_int.shape[-1] == 1:
            y_true_int = tf.squeeze(y_true_int, axis=-1)
            
        y_true_one_hot = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[-1])
        
        # Clip predictions for numerical stability
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        # Calculate focal weights
        weights = tf.pow(1 - y_pred, gamma)
        
        # Apply alpha weighting if desired
        if alpha is not None:
            alpha_t = y_true_one_hot * alpha + (1 - y_true_one_hot) * (1 - alpha)
            cross_entropy = alpha_t * cross_entropy
        
        # Compute focal loss
        f = weights * cross_entropy
        return tf.reduce_sum(f, axis=-1)
    
    return loss

def focal_loss(gamma=2.0, alpha=0.25, from_logits=params.USE_LOGITS):
    """
    Focal Loss for one-hot encoded labels.
    Used for models like 'original_haverland' that use categorical_crossentropy.
    """
    def loss(y_true, y_pred):
        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            
        # Already one-hot labels
        y_true_one_hot = tf.cast(y_true, y_pred.dtype)
        
        # Clip predictions for numerical stability
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        # Calculate focal weights
        weights = tf.pow(1 - y_pred, gamma)
        
        # Apply alpha weighting if desired
        if alpha is not None:
            alpha_t = y_true_one_hot * alpha + (1 - y_true_one_hot) * (1 - alpha)
            cross_entropy = alpha_t * cross_entropy
        
        # Compute focal loss
        f = weights * cross_entropy
        return tf.reduce_sum(f, axis=-1)
    
    return loss

class DynamicSparseFocalLoss(tf.keras.losses.Loss):
    """
    Keras 3 compatible Focal Loss that allows updating gamma and alpha 
    during training without model recompilation.
    """
    def __init__(self, gamma=2.0, alpha=0.25, nb_classes=None,
                 label_smoothing=None, from_logits=params.USE_LOGITS, name='dynamic_sparse_focal_loss', **kwargs):
        # Keras 3 compatibility: 'auto' is not a valid reduction anymore
        if kwargs.get('reduction') == 'auto':
            kwargs['reduction'] = 'sum_over_batch_size'
            
        super().__init__(name=name, **kwargs)
        self.gamma = tf.Variable(gamma, dtype=tf.float32, trainable=False, name=f"{name}_gamma")
        self.from_logits = from_logits
        
        # We always use a vector for alpha internally to support per-class weighting smoothly
        if nb_classes is None:
            nb_classes = params.NB_CLASSES

        # Label smoothing: read from params when not supplied explicitly
        if label_smoothing is None:
            label_smoothing = getattr(params, 'LABEL_SMOOTHING', 0.0)
        self.label_smoothing = float(label_smoothing)
            
        if isinstance(alpha, (list, tuple, np.ndarray)):
            init_alpha = np.array(alpha, dtype=np.float32)
        else:
            init_alpha = np.ones(nb_classes, dtype=np.float32) * float(alpha)
            
        self.alpha = tf.Variable(init_alpha, dtype=tf.float32, trainable=False, name=f"{name}_alpha")

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            
        y_true_int = tf.cast(y_true, tf.int32)
        if len(y_true_int.shape) > 1 and y_true_int.shape[-1] == 1:
            y_true_int = tf.squeeze(y_true_int, axis=-1)

        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)

        # Label smoothing: spread ε across all classes, concentrate (1-ε) on true class
        if self.label_smoothing > 0.0:
            smooth = self.label_smoothing
            y_true_one_hot = y_true_one_hot * (1.0 - smooth) + smooth / tf.cast(num_classes, tf.float32)
        
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weights = tf.pow(1 - y_pred, self.gamma)
        
        # Match alpha vector to y_true_one_hot labels
        # self.alpha is (NB_CLASSES,)
        alpha_t = tf.reduce_sum(y_true_one_hot * self.alpha, axis=-1, keepdims=True)
        
        return tf.reduce_sum(weights * alpha_t * cross_entropy, axis=-1)

class DynamicFocalLoss(tf.keras.losses.Loss):
    """Same as DynamicSparseFocalLoss but for one-hot labels."""
    def __init__(self, gamma=2.0, alpha=0.25, nb_classes=None,
                 label_smoothing=None, from_logits=params.USE_LOGITS, name='dynamic_focal_loss', **kwargs):
        # Keras 3 compatibility: 'auto' is not a valid reduction anymore
        if kwargs.get('reduction') == 'auto':
            kwargs['reduction'] = 'sum_over_batch_size'
            
        super().__init__(name=name, **kwargs)
        self.gamma = tf.Variable(gamma, dtype=tf.float32, trainable=False, name=f"{name}_gamma")
        self.from_logits = from_logits
        
        if nb_classes is None:
            nb_classes = params.NB_CLASSES

        if label_smoothing is None:
            label_smoothing = getattr(params, 'LABEL_SMOOTHING', 0.0)
        self.label_smoothing = float(label_smoothing)
            
        if isinstance(alpha, (list, tuple, np.ndarray)):
            init_alpha = np.array(alpha, dtype=np.float32)
        else:
            init_alpha = np.ones(nb_classes, dtype=np.float32) * float(alpha)
            
        self.alpha = tf.Variable(init_alpha, dtype=tf.float32, trainable=False, name=f"{name}_alpha")

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.cast(y_true, y_pred.dtype)

        # Label smoothing
        if self.label_smoothing > 0.0:
            smooth = self.label_smoothing
            y_true_one_hot = y_true_one_hot * (1.0 - smooth) + smooth / tf.cast(num_classes, y_pred.dtype)
        
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weights = tf.pow(1 - y_pred, self.gamma)
        
        # Match alpha vector to y_true_one_hot labels
        alpha_t = tf.reduce_sum(y_true_one_hot * self.alpha, axis=-1, keepdims=True)
        
        return tf.reduce_sum(weights * alpha_t * cross_entropy, axis=-1)
