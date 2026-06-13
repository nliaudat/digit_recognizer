import tensorflow as tf
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

EPS = 1e-7


def _detect_output_format(model: tf.keras.Model) -> str:
    """
    Detect whether a model outputs raw logits or softmax probabilities.

    Inspects ALL output tensors. For multi-output models (e.g. both
    'logits' and 'output' heads), prefers the softmax head.

    Returns:
        'softmax' if a softmax output is found,
        'logits'  if only raw logits are found,
        'unknown' otherwise (will default to softmax after softmax()).
    """
    import config as params

    try:
        if "EnsembleTeacher" in str(type(model)):
            return 'softmax'

        outputs = getattr(model, 'outputs', None)
        if outputs is None or not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        # 1. Walk all output tensors, try to find a softmax head
        has_logits = False
        for out_tensor in outputs:
            node_layer = None
            if hasattr(out_tensor, 'node') and hasattr(out_tensor.node, 'layer'):
                node_layer = out_tensor.node.layer
            elif hasattr(out_tensor, '_keras_history'):
                node_layer = out_tensor._keras_history[0]

            if node_layer is None:
                continue

            cls_name = node_layer.__class__.__name__
            if "Softmax" in cls_name:
                return 'softmax'

            if hasattr(node_layer, 'activation'):
                act = node_layer.activation
                if act is None:
                    has_logits = True
                    continue
                name = act if isinstance(act, str) else getattr(act, '__name__', '').lower()
                if name == 'softmax':
                    return 'softmax'
                if name == 'linear' or act == tf.keras.activations.linear:
                    has_logits = True
                    continue

        if has_logits:
            return 'logits'

        # Fallback: check output name
        for out_tensor in outputs:
            name = getattr(out_tensor, 'name', '').lower()
            if 'softmax' in name or 'output' in name:
                return 'softmax'

    except Exception:
        pass

    return 'unknown'


def _normalize_to_softmax(model: tf.keras.Model, outputs: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
    """
    Given a model and its raw output, return softmax probabilities.
    
    - If the model has multiple outputs (e.g. logits + softmax), select
      the softmax one.  If none is found, apply softmax to the first.
    - If the single output is softmax already, return as-is.
    - If the single output is logits, apply softmax.
    """
    fmt = _detect_output_format(model)

    if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
        # Multi-output model: try to pick the softmax head
        for out_tensor in outputs:
            name = getattr(out_tensor, 'name', '').lower()
            if 'softmax' in name or 'output' in name:
                return tf.convert_to_tensor(out_tensor)
            # Check if this tensor is already in [0,1] range (probabilities)
            t = tf.convert_to_tensor(out_tensor)
            mean_val = tf.reduce_mean(t).numpy()
            if 0.0 <= mean_val <= 1.0 and tf.abs(tf.reduce_sum(t[0]) - 1.0) < 0.1:
                return t
        # Fallback: softmax the first output
        return tf.nn.softmax(tf.convert_to_tensor(outputs[0]))
    else:
        t = tf.convert_to_tensor(outputs if not isinstance(outputs, (list, tuple)) else outputs[0])
        if fmt == 'softmax':
            return t
        else:
            # Assume logits or unknown → apply softmax
            return tf.nn.softmax(t)


class EnsembleTeacher(tf.keras.Model):
    """
    Combine multiple teachers into a single ensemble for distillation.
    
    **CRITICAL FIX**: Each teacher's output is independently normalised to
    softmax probabilities before being averaged.  This handles:
    - Teachers that output raw logits mixed with those that output softmax
    - Multi-output teachers (logits + softmax heads)
    - All teachers in the ensemble produce comparable probability
      distributions.
    """
    
    def __init__(
        self,
        teachers: List[tf.keras.Model],
        teacher_weights: Optional[List[float]] = None,
        temperature: float = 1.0,
        use_logits: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.teachers = teachers
        self.num_teachers = len(teachers)
        self.temperature = temperature
        self.use_logits = use_logits
        
        if teacher_weights is None:
            self.teacher_weights = [1.0 / self.num_teachers] * self.num_teachers
        else:
            self.teacher_weights = teacher_weights
            
        # Freeze teachers
        for teacher in self.teachers:
            teacher.trainable = False
            
        # Detect output format for each teacher
        self._teacher_formats = []
        for i, teacher in enumerate(self.teachers):
            fmt = _detect_output_format(teacher)
            self._teacher_formats.append(fmt)
            logger.info(f"  Teacher {i} ({teacher.name}): output format = {fmt}")
            
        logger.info(f"EnsembleTeacher created with {self.num_teachers} teachers")
        logger.info(f"Weights: {self.teacher_weights}")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        logger.debug(f"EnsembleTeacher.call() - inputs shape: {inputs.shape}")
        
        # Step 1: collect and normalise each teacher's output to softmax
        normalized_outputs = []
        for i, teacher in enumerate(self.teachers):
            raw = teacher(inputs, training=training)
            probs = _normalize_to_softmax(teacher, raw)
            normalized_outputs.append(probs)
            # NOTE: use tf.shape for graph-mode symbolic tensors; avoid
            # format-specs (.4f) on symbolic tensors since Python's str.format
            # cannot handle them.  tf.print or assignment to a concrete eager
            # tensor would be needed for numeric logging at runtime.
            logger.debug(f"  Teacher {i}: raw type={type(raw).__name__} → probs shape={probs.shape}")

        # Step 2: weighted average of softmax probabilities
        weighted = tf.zeros_like(normalized_outputs[0])
        for probs, weight in zip(normalized_outputs, self.teacher_weights):
            weighted += weight * probs

        logger.debug(f"Ensemble output shape: {weighted.shape}")
        return weighted

    @property
    def input_shape(self):
        return self.teachers[0].input_shape
        
    @property
    def output_shape(self):
        return self.teachers[0].output_shape
        
    def count_params(self):
        return sum(t.count_params() for t in self.teachers)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'num_teachers': self.num_teachers,
            'teacher_weights': self.teacher_weights,
            'temperature': self.temperature,
            'use_logits': self.use_logits,
        })
        return config

    def verify(self, test_data, num_samples=100):
        """Verify teacher ensemble is working properly."""
        x_test, y_test = test_data
        predictions = self.predict(x_test[:num_samples])
        acc = np.mean(np.argmax(predictions, axis=1) == y_test[:num_samples])
        logger.info(f"Ensemble teacher verification accuracy: {acc:.4f}")
        return acc
