# utils/losses.py
import tensorflow as tf
from tensorflow.keras import backend as K

def sparse_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for sparse categorical labels.
    gamma=2.0 (standard), alpha=0.25 (balances hard/easy examples)
    
    This implementation handles sparse (integer) labels by converting them to one-hot 
    internally for the focal weight calculation.
    """
    def loss(y_true, y_pred):
        # Convert sparse labels to one-hot
        # Handle potential rank mismatch if labels have an extra dimension
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
        focal_loss = weights * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    
    return loss

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for one-hot encoded labels.
    Used for models like 'original_haverland' that use categorical_crossentropy.
    """
    def loss(y_true, y_pred):
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
        focal_loss = weights * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    
    return loss
