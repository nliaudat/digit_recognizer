"""
Knowledge distillation core implementation.

Models output softmax probabilities (activation='softmax'), so:
  - Loss     : SparseCategoricalCrossentropy(from_logits=False)
  - Distill  : KL( teacher_T || student_T ) where
                prob_T = softmax( log(prob) / T )  ← temperature scaling
                This is equivalent to softmax(logits/T) when you don't
                have access to the raw logits.

Classes
-------
Distiller         – single teacher, fixed temperature / alpha.
ProgressiveDistiller – dynamic temperature + alpha schedule.
EnsembleDistiller – weighted combination of multiple teachers.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistillationProgressCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to update the distiller's current epoch.
    Crucial for ProgressiveDistiller temperature and alpha scheduling.
    """
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, "current_epoch"):
            self.model.current_epoch = epoch
            logger.info(f"Distiller epoch updated to {epoch}")


class Distiller(tf.keras.Model):
    """
    Knowledge distillation wrapper with multiple distillation modes.
    
    Supports:
    - Soft distillation (KL divergence with temperature)
    - Hard distillation (pseudo-labels from teacher)
    - Hybrid distillation (combination of soft and hard)
    - Temperature scheduling
    - Attention transfer (experimental)
    
    Usage:
        distiller = Distiller(student, teacher, temperature=4.0, alpha=0.7)
        distiller.compile(optimizer='adam', metrics=['accuracy'])
        distiller.fit(train_data, validation_data=val_data)
    """
    
    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        temperature: float = 4.0,
        alpha: float = 0.7,
        mode: str = 'soft',
        use_attention_transfer: bool = False,
        attention_layer_names: Optional[list] = None,
        name: str = "distiller",
        **kwargs
    ):
        """
        Initialize the distiller.
        
        Args:
            student: Student model to train
            teacher: Teacher model (will be frozen)
            temperature: Softening temperature for distillation
            alpha: Balance between hard labels and teacher (0=only teacher, 1=only hard)
            mode: Distillation mode ('soft', 'hard', 'hybrid')
            use_attention_transfer: Enable attention transfer between teacher and student
            attention_layer_names: Names of layers for attention transfer
            name: Name of the distiller model
        """
        super().__init__(name=name, **kwargs)
        
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.mode = mode
        self.use_attention_transfer = use_attention_transfer
        self.attention_layer_names = attention_layer_names or []
        
        # Freeze teacher
        self.teacher.trainable = False
        
        # For temperature scheduling
        self.current_epoch = 0
        self.temperature_schedule: Optional[Callable] = None
        
        # For attention transfer
        self.teacher_attention_maps = None
        self.student_attention_maps = None
        
        logger.info(f"Distiller initialized: mode={mode}, temperature={temperature}, alpha={alpha}")
    
    def compile(
        self,
        optimizer: Union[str, tf.keras.optimizers.Optimizer],
        metrics: Optional[list] = None,
        student_loss_fn: Optional[Callable] = None,
        distillation_loss_fn: Optional[Callable] = None,
        temperature_schedule: Optional[Callable] = None,
        **kwargs
    ):
        """
        Compile the distiller.

        Args:
            optimizer:            Optimizer for the student.
            metrics:              Metrics to track (e.g. ['accuracy']).
            student_loss_fn:      Loss against hard labels
                                  (default: SparseCategoricalCrossentropy, from_logits=False
                                   — matches softmax output models like v4/v16).
            distillation_loss_fn: Loss for distillation (default: KLDivergence).
            temperature_schedule: Callable(epoch) → float, for dynamic temperature.
        """
        super().compile(optimizer=optimizer, metrics=metrics or [], **kwargs)

        # from_logits=False because models output softmax probabilities
        self.student_loss_fn = student_loss_fn or tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, name="student_loss"
        )
        self.distillation_loss_fn = distillation_loss_fn or tf.keras.losses.KLDivergence(
            name="distill_loss"
        )
        self.temperature_schedule = temperature_schedule

        logger.info(f"Distiller compiled with optimizer={optimizer}")
    
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Single training step."""
        x, y = data
        
        # Get teacher predictions (frozen) — returns softmax probabilities
        teacher_probs = self.teacher(x, training=False)

        # Temperature scheduling
        temp = self.temperature
        if self.temperature_schedule:
            temp = self.temperature_schedule(self.current_epoch)

        with tf.GradientTape() as tape:
            student_probs = self.student(x, training=True)

            # Hard-label loss (against ground-truth integer labels)
            student_loss = self.student_loss_fn(y, student_probs)

            # Distillation loss (teacher → student knowledge transfer)
            distill_loss = self._compute_distillation_loss(
                teacher_probs, student_probs, temp
            )

            # Combined loss
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

            # Optional attention transfer loss
            if self.use_attention_transfer:
                att_loss = self._compute_attention_loss(x)
                loss += 0.1 * att_loss

        # Apply gradients
        trainable_vars = self.student.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Return results including our custom losses
        # self.compute_metrics returns a dict of results
        results = self.compute_metrics(x, y, student_probs)
        results.update({
            "loss": loss,
            "student_loss": student_loss,
            "distill_loss": distill_loss,
        })
        return results
    
    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Single test/validation step."""
        x, y = data
        student_probs = self.student(x, training=False)

        loss = self.student_loss_fn(y, student_probs)
        
        # self.compute_metrics returns a dict of results
        results = self.compute_metrics(x, y, student_probs)
        results["loss"] = loss
        return results

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through student (returns softmax probabilities)."""
        return self.student(inputs, training=training)
    
    def _compute_distillation_loss(
        self,
        teacher_probs: tf.Tensor,
        student_probs: tf.Tensor,
        temperature: float,
    ) -> tf.Tensor:
        """
        Compute distillation loss when models output softmax probabilities.

        Temperature scaling without raw logits:
            prob_T = softmax( log(prob) / T )
        This is mathematically equivalent to softmax(logits / T) and reduces
        to the original distribution when T=1.

        A small EPS is added before log() to avoid log(0).

        Args:
            teacher_probs: Softmax output of teacher (batch, classes).
            student_probs: Softmax output of student (batch, classes).
            temperature:   Softening temperature (>1 = softer, =1 = unchanged).

        Returns:
            Scalar distillation loss.
        """
        EPS = 1e-7

        if self.mode == "soft":
            # Recover pseudo-logits via log, apply temperature, re-normalise
            teacher_T = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(teacher_probs, EPS, 1.0)) / temperature
            )
            student_T = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(student_probs, EPS, 1.0)) / temperature
            )
            return self.distillation_loss_fn(teacher_T, student_T)

        elif self.mode == "hard":
            # Pseudo-labels: argmax of teacher probabilities
            teacher_preds = tf.argmax(teacher_probs, axis=1)
            return self.student_loss_fn(teacher_preds, student_probs)

        else:  # hybrid
            teacher_T = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(teacher_probs, EPS, 1.0)) / temperature
            )
            student_T = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(student_probs, EPS, 1.0)) / temperature
            )
            soft_loss = self.distillation_loss_fn(teacher_T, student_T)

            teacher_preds = tf.argmax(teacher_probs, axis=1)
            hard_loss = self.student_loss_fn(teacher_preds, student_probs)

            return 0.7 * soft_loss + 0.3 * hard_loss
    
    def _compute_attention_loss(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute attention transfer loss between teacher and student.
        (Experimental feature)
        
        Args:
            x: Input tensor
        
        Returns:
            Attention loss tensor
        """
        # Simplified implementation - would need to extract intermediate activations
        # For now, returns zero loss
        return tf.constant(0.0, dtype=tf.float32)
    
    def get_student(self) -> tf.keras.Model:
        """Return the trained student model."""
        return self.student
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'alpha': self.alpha,
            'mode': self.mode,
            'use_attention_transfer': self.use_attention_transfer,
            'attention_layer_names': self.attention_layer_names,
        })
        return config


class ProgressiveDistiller(Distiller):
    """
    Progressive distillation with dynamic temperature and alpha.
    Starts with high temperature (softer targets) and low alpha (more teacher guidance),
    then gradually transitions to lower temperature and higher alpha (more hard labels).
    """
    
    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        initial_temperature: float = 8.0,
        final_temperature: float = 2.0,
        initial_alpha: float = 0.3,
        final_alpha: float = 0.8,
        total_epochs: int = 50,
        mode: str = 'soft',
        **kwargs
    ):
        """
        Initialize progressive distiller.
        
        Args:
            student: Student model to train
            teacher: Teacher model
            initial_temperature: Starting temperature
            final_temperature: Ending temperature
            initial_alpha: Starting alpha
            final_alpha: Ending alpha
            total_epochs: Total training epochs
            mode: Distillation mode
        """
        super().__init__(
            student=student,
            teacher=teacher,
            temperature=initial_temperature,
            alpha=initial_alpha,
            mode=mode,
            **kwargs
        )
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_epochs = total_epochs
        
        # Estimate total steps if we want to switch to a step-based schedule later
        self.total_steps = 0  
    
    def _update_schedule(self):
        """Update temperature and alpha based on progress."""
        progress = min(1.0, self.current_epoch / max(1, self.total_epochs))
        
        # Linearly decrease temperature
        self.temperature = (
            self.initial_temperature * (1 - progress) + 
            self.final_temperature * progress
        )
        
        # Linearly increase alpha (more weight on hard labels over time)
        self.alpha = (
            self.initial_alpha * (1 - progress) + 
            self.final_alpha * progress
        )
    
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Single training step with progressive scheduling."""
        self._update_schedule()
        return super().train_step(data)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'initial_alpha': self.initial_alpha,
            'final_alpha': self.final_alpha,
            'total_epochs': self.total_epochs,
        })
        return config


class EnsembleDistiller(Distiller):
    """
    Distillation from multiple teachers.
    Combines predictions from multiple teachers for richer soft targets.
    """
    
    def __init__(
        self,
        student: tf.keras.Model,
        teachers: list,
        teacher_weights: Optional[list] = None,
        temperature: float = 4.0,
        alpha: float = 0.7,
        **kwargs
    ):
        """
        Initialize ensemble distiller.
        
        Args:
            student: Student model to train
            teachers: List of teacher models
            teacher_weights: Weights for each teacher (default: equal)
            temperature: Softening temperature
            alpha: Balance between hard labels and teacher
        """
        # Use first teacher as primary for compatibility
        super().__init__(
            student=student,
            teacher=teachers[0],
            temperature=temperature,
            alpha=alpha,
            **kwargs
        )
        
        self.teachers = teachers
        self.num_teachers = len(teachers)
        
        # Set teacher weights
        if teacher_weights is None:
            self.teacher_weights = [1.0 / self.num_teachers] * self.num_teachers
        else:
            self.teacher_weights = teacher_weights
        
        # Freeze all teachers
        for teacher in self.teachers:
            teacher.trainable = False
    
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Single training step with ensemble of teachers."""
        x, y = data
        
        # Get predictions from all teachers
        teacher_logits_list = []
        for teacher in self.teachers:
            logits = teacher(x, training=False)
            teacher_logits_list.append(logits)
        
        # Weighted average of teacher predictions
        weighted_logits = tf.zeros_like(teacher_logits_list[0])
        for logits, weight in zip(teacher_logits_list, self.teacher_weights):
            weighted_logits += weight * logits
        
        # Temperature scheduling
        temp = self.temperature
        if self.temperature_schedule:
            temp = self.temperature_schedule(self.current_epoch)
        
        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            
            # Hard label loss
            student_loss = self.student_loss_fn(y, student_logits)
            
            # Distillation loss
            EPS = 1e-7
            # Recover pseudo-logits via log, apply temperature, re-normalise
            teacher_T = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(weighted_logits, EPS, 1.0)) / temp
            )
            student_T = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(student_logits, EPS, 1.0)) / temp
            )
            distill_loss = self.distillation_loss_fn(teacher_T, student_T)
            
            # Combined loss
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        
        # Apply gradients
        trainable_vars = self.student.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        # 9. Update metrics
        results = self.compute_metrics(x, y, student_logits)
        
        self.current_epoch += 1
        
        # Return results including our custom losses
        results.update({
            'loss': loss,
            'student_loss': student_loss,
            'distill_loss': distill_loss,
        })
        return results


class MixedInputDistiller(Distiller):
    """
    Distiller that handles different input shapes for teacher and student.
    Commonly used when distilling an RGB teacher into a Grayscale student.
    """

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        teacher_input_fn: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the mixed-input distiller.

        Args:
            student:          Student model (e.g. grayscale)
            teacher:          Teacher model (e.g. RGB)
            teacher_input_fn: Function to convert student input to teacher input
                             (e.g., lambda x: tf.image.grayscale_to_rgb(x))
        """
        super().__init__(student, teacher, **kwargs)
        self.teacher_input_fn = teacher_input_fn

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Single training step with input conversion."""
        x_student, y = data

        # 1. Convert student input to teacher input (e.g., grayscale -> RGB)
        if self.teacher_input_fn:
            x_teacher = self.teacher_input_fn(x_student)
        else:
            x_teacher = x_student

        # 2. Get teacher predictions (frozen)
        teacher_probs = self.teacher(x_teacher, training=False)

        # 3. Temperature scheduling
        temp = self.temperature
        if self.temperature_schedule:
            temp = self.temperature_schedule(self.current_epoch)

        with tf.GradientTape() as tape:
            # 4. Get student predictions
            student_probs = self.student(x_student, training=True)

            # 5. Hard-label loss
            student_loss = self.student_loss_fn(y, student_probs)

            # 6. Distillation loss
            distill_loss = self._compute_distillation_loss(
                teacher_probs, student_probs, temp
            )

            # 7. Combined loss
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        # 8. Apply gradients
        trainable_vars = self.student.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # 9. Update metrics
        results = self.compute_metrics(x_student, y, student_probs)

        # Return results including our custom losses
        results.update({
            "loss": loss,
            "student_loss": student_loss,
            "distill_loss": distill_loss,
        })
        return results

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Standard test step using student's native input."""
        x_student, y = data
        student_probs = self.student(x_student, training=False)

        loss = self.student_loss_fn(y, student_probs)
        
        # self.compute_metrics returns a dict of results
        results = self.compute_metrics(x_student, y, student_probs)
        results["loss"] = loss
        return results

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        # Note: teacher_input_fn is generally not serializable if it's a lambda
        return config