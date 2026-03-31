import tensorflow as tf
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class EnsembleTeacher(tf.keras.Model):
    """
    Combine multiple teachers into a single ensemble for distillation.
    
    Supports:
    - Weighted average of probabilties (use_logits=False) or logits (use_logits=True)
    - Automatically freezes all teachers
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
            
        logger.info(f"EnsembleTeacher created with {self.num_teachers} teachers")
        logger.info(f"Weights: {self.teacher_weights}")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        all_outputs = []
        for teacher in self.teachers:
            all_outputs.append(teacher(inputs, training=training))
            
        if self.use_logits:
            weighted_logits = tf.zeros_like(all_outputs[0])
            for output, weight in zip(all_outputs, self.teacher_weights):
                weighted_logits += weight * output
            return tf.nn.softmax(weighted_logits / self.temperature)
        else:
            weighted_probs = tf.zeros_like(all_outputs[0])
            for output, weight in zip(all_outputs, self.teacher_weights):
                weighted_probs += weight * output
            return weighted_probs

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
