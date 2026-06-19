"""
Knowledge distillation core implementation.

Models can output either raw logits (activation=None) or softmax probabilities, based on:
  - Loss     : SparseCategoricalCrossentropy(from_logits=params.USE_LOGITS)
  - Distill  : KL( teacher_T || student_T ) where targets are temperature-scaled.

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
import config as params
import config.distillation as dist_cfg

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


# ── Helper: detect output format ─────────────────────────────────────
def _detect_teacher_format(model: tf.keras.Model) -> str:
    """
    Detect whether *any* output of *model* is softmax or logits.
    Used by Distiller to normalise teacher outputs to softmax.

    Returns 'softmax', 'logits', or 'unknown'.
    """
    import config as params

    try:
        if "EnsembleTeacher" in str(type(model)):
            return 'softmax'

        outputs = getattr(model, 'outputs', None)
        if outputs is None or not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

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

        # Fallback: output name heuristics
        for out_tensor in outputs:
            name = getattr(out_tensor, 'name', '').lower()
            if 'softmax' in name or 'output' in name:
                return 'softmax'

    except Exception:
        pass
    return 'unknown'


def _extract_teacher_probs(model: tf.keras.Model,
                            raw_output) -> tf.Tensor:
    """
    Convert teacher raw output to a single softmax-probability tensor.

    - Multi-output -> pick the softmax head, or softmax the first.
    - Single logits -> apply softmax.
    - Single softmax -> return as-is.
    """
    if isinstance(raw_output, (list, tuple)) and len(raw_output) > 1:
        # Try to find the softmax head
        for t in raw_output:
            name = getattr(t, 'name', '').lower()
            t_t = tf.convert_to_tensor(t)
            if 'softmax' in name or 'output' in name:
                return t_t
        # Fallback: softmax first output
        return tf.nn.softmax(tf.convert_to_tensor(raw_output[0]))
    else:
        t = tf.convert_to_tensor(raw_output if not isinstance(raw_output, (list, tuple)) else raw_output[0])
        fmt = _detect_teacher_format(model)
        if fmt == 'softmax':
            return t
        else:
            return tf.nn.softmax(t)


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
        distiller = Distiller(student, teacher, temperature=DISTILLATION_TEMPERATURE, alpha=DISTILLATION_ALPHA)
        distiller.compile(optimizer='adam', metrics=['accuracy'])
        distiller.fit(train_data, validation_data=val_data)
    """
    
    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        temperature: float = dist_cfg.DISTILLATION_TEMPERATURE,
        alpha: float = dist_cfg.DISTILLATION_ALPHA,
        mode: str = dist_cfg.DISTILLATION_MODE,
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
        
        # Detect output format for teacher and student
        self.teacher_is_logit = self._is_logit_output(self.teacher)
        self.student_is_logit = self._is_logit_output(self.student)

        logger.info(f"Distiller initialized: mode={mode}, temperature={temperature}, alpha={alpha}")
        logger.info(f"Format detection: teacher_is_logit={self.teacher_is_logit}, student_is_logit={self.student_is_logit}")

        # ── Validate teacher output format ─────────────────────────────
        try:
            dummy_shape = self.teacher.input_shape
            if not isinstance(dummy_shape, list) and None not in dummy_shape[1:]:
                dummy_input = tf.zeros((1,) + dummy_shape[1:], dtype=tf.float32)
                dummy_raw = self.teacher(dummy_input, training=False)
                dummy_probs = _extract_teacher_probs(self.teacher, dummy_raw)
                prob_sum = tf.reduce_sum(dummy_probs, axis=-1)
                mean_sum = tf.reduce_mean(prob_sum).numpy()
                if abs(mean_sum - 1.0) > 0.5:
                    logger.warning(
                        f"⚠️ Teacher output probabilities sum to {mean_sum:.4f} (expected ~1.0). "
                        f"Format detection (teacher_is_logit={self.teacher_is_logit}) may be wrong."
                    )
                else:
                    logger.debug(f"Teacher format validated: prob_sum={mean_sum:.4f}")
        except Exception as e:
            logger.debug(f"Teacher format validation skipped (non-deterministic shape?): {e}")
    
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
                                  (default: SparseCategoricalCrossentropy, from_logits=params.USE_LOGITS
                                   — matches current global configuration).
            distillation_loss_fn: Loss for distillation (default: KLDivergence).
            temperature_schedule: Callable(epoch) → float, for dynamic temperature.
        """
        # Ensure loss_tracker is initialized so Keras properly tracks val_loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_metric = self.loss_tracker
        
        super().compile(
            optimizer=optimizer,
            metrics=metrics or [],
            loss=None,  # We handle loss manually in train_step/test_step
            **kwargs
        )

        # Configure loss based on global params.USE_LOGITS
        self.student_loss_fn = student_loss_fn or tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=params.USE_LOGITS, name="student_loss"
        )
        self.distillation_loss_fn = distillation_loss_fn or tf.keras.losses.KLDivergence(
            name="distill_loss"
        )
        self.temperature_schedule = temperature_schedule

        logger.info(f"Distiller compiled with optimizer={optimizer}")
    
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Single training step."""
        x, y = data
        
        # Get teacher predictions (frozen) — may be logits, softmax, or multi-output
        teacher_raw = self.teacher(x, training=False)
        # CRITICAL FIX: normalise to softmax probabilities
        teacher_probs = _extract_teacher_probs(self.teacher, teacher_raw)

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

        # 8. Apply gradients
        trainable_vars = self.student.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # 9. Update metrics
        self.compiled_metrics.update_state(y, student_probs)
        self.loss_tracker.update_state(loss)
        
        # Return results including our custom losses
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "loss": self.loss_tracker.result(),
            "student_loss": student_loss,
            "distill_loss": distill_loss,
        })
        return results
    
    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Single test/validation step."""
        x, y = data
        
        # Get teacher predictions for distillation loss calculation in validation
        teacher_raw = self.teacher(x, training=False)
        teacher_probs = _extract_teacher_probs(self.teacher, teacher_raw)
        student_probs = self.student(x, training=False)

        # Temperature
        temp = self.temperature
        if self.temperature_schedule:
            temp = self.temperature_schedule(self.current_epoch)

        # Losses
        student_loss = self.student_loss_fn(y, student_probs)
        distill_loss = self._compute_distillation_loss(teacher_probs, student_probs, temp)
        
        # Combined loss (matches train_step logic for consistency)
        loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        # Track combined loss for logging.
        self.loss_tracker.update_state(loss)
        
        # Compute accuracy manually as a scalar for reliable logging.
        # Keras 3 auto-prefixes every key in test_step with "val_".
        # We return ONLY bare names — Keras produces the val_ variants.
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(student_probs, axis=-1), y), tf.float32)
        )
        results = {
            "loss": self.loss_tracker.result(),
            "accuracy": acc,
            "student_loss": student_loss,
            "distill_loss": distill_loss,
        }
        
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
            # 1. Handle Teacher
            if self.teacher_is_logit:
                t_logits = teacher_probs
            else:
                # Recover pseudo-logits from probabilities
                t_logits = tf.math.log(tf.clip_by_value(teacher_probs, EPS, 1.0))
            
            # 2. Handle Student
            if self.student_is_logit:
                s_logits = student_probs
            else:
                # Recover pseudo-logits from probabilities
                s_logits = tf.math.log(tf.clip_by_value(student_probs, EPS, 1.0))

            teacher_T = tf.nn.softmax(t_logits / temperature)
            student_T = tf.nn.softmax(s_logits / temperature)
            
            return self.distillation_loss_fn(teacher_T, student_T)

        elif self.mode == "hard":
            # Pseudo-labels: argmax of teacher probabilities
            teacher_preds = tf.argmax(teacher_probs, axis=1)
            return self.student_loss_fn(teacher_preds, student_probs)

        else:  # hybrid
            if self.teacher_is_logit:
                t_logits = teacher_probs
            else:
                t_logits = tf.math.log(tf.clip_by_value(teacher_probs, EPS, 1.0))
            
            if self.student_is_logit:
                s_logits = student_probs
            else:
                s_logits = tf.math.log(tf.clip_by_value(student_probs, EPS, 1.0))

            teacher_T = tf.nn.softmax(t_logits / temperature)
            student_T = tf.nn.softmax(s_logits / temperature)
            
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
    
    def _is_logit_output(self, model: tf.keras.Model) -> bool:
        """Heuristic check if a model outputs raw logits or softmax probabilities."""
        try:
            # 1. Check if it's an EnsembleTeacher (which current implementation always outputs softmax)
            # Use name check to avoid circular imports
            if "EnsembleTeacher" in str(type(model)):
                return False 
            
            # 2. Check output layer activation
            last_layer = None
            if hasattr(model, 'outputs') and len(model.outputs) > 0:
                output_tensor = model.outputs[0]
                # Keras 3 robust way
                if hasattr(output_tensor, 'node') and hasattr(output_tensor.node, 'layer'):
                    last_layer = output_tensor.node.layer
                # Keras 2 robust way
                elif hasattr(output_tensor, '_keras_history'):
                    last_layer = output_tensor._keras_history[0]
            
            # Fallback for Sequential or other models where outputs might not have history
            if last_layer is None and hasattr(model, 'layers') and len(model.layers) > 0:
                last_layer = model.layers[-1]

            if last_layer is not None:
                # Explicit check for Softmax layer (no activation attribute)
                if "Softmax" in last_layer.__class__.__name__:
                    return False
                
                # Check for activation attribute (Common for Dense/Conv layers)
                if hasattr(last_layer, 'activation'):
                    act = last_layer.activation
                    if act is None: return True
                    
                    # Handle string-based activations (e.g., 'softmax', 'linear')
                    name = act if isinstance(act, str) else getattr(act, '__name__', '').lower()
                    if name == 'linear': return True
                    if name == 'softmax': return False
                    
                    # Check for linear activation objects
                    if act == tf.keras.activations.linear: return True
        except:
            pass
        # Default to whatever the current global configuration suggests if we can't be sure
        return params.USE_LOGITS

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
        temperature: Optional[float] = None,  # Backward compat: overrides initial_temperature
        alpha: Optional[float] = None,        # Backward compat: overrides initial_alpha
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
            temperature: Backward compat alias for initial_temperature
            alpha: Backward compat alias for initial_alpha
        """
        # Handle backward-compatible parameter names
        if temperature is not None:
            initial_temperature = temperature
        if alpha is not None:
            initial_alpha = alpha
        
        # Remove any conflicting kwargs that would be passed to Distiller
        kwargs.pop('temperature', None)
        kwargs.pop('alpha', None)
        
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
            student_probs = self.student(x, training=True)
            
            # Hard label loss
            student_loss = self.student_loss_fn(y, student_probs)
            
            # Distillation loss
            EPS = 1e-7
            # Recover pseudo-logits via log, apply temperature, re-normalise
            teacher_T = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(weighted_logits, EPS, 1.0)) / temp
            )
            student_T = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(student_probs, EPS, 1.0)) / temp
            )
            distill_loss = self.distillation_loss_fn(teacher_T, student_T)
            
            # Combined loss
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        
        # Apply gradients
        trainable_vars = self.student.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, student_probs)
        
        # Return results including our custom losses
        results = {m.name: m.result() for m in self.metrics}
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
        self.compiled_metrics.update_state(y, student_probs)

        # Return results including our custom losses
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "loss": loss,
            "student_loss": student_loss,
            "distill_loss": distill_loss,
        })
        return results

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Test step with input conversion and full loss calculation."""
        x_student, y = data
        
        if self.teacher_input_fn:
            x_teacher = self.teacher_input_fn(x_student)
        else:
            x_teacher = x_student
            
        teacher_probs = self.teacher(x_teacher, training=False)
        student_probs = self.student(x_student, training=False)

        temp = self.temperature
        if self.temperature_schedule:
            temp = self.temperature_schedule(self.current_epoch)

        student_loss = self.student_loss_fn(y, student_probs)
        distill_loss = self._compute_distillation_loss(teacher_probs, student_probs, temp)
        
        loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        self.loss_tracker.update_state(loss)
        
        # Same policy as Distiller.test_step(): bare names only; Keras prefixes.
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(student_probs, axis=-1), y), tf.float32)
        )
        results = {
            "loss": self.loss_tracker.result(),
            "accuracy": acc,
            "student_loss": student_loss,
            "distill_loss": distill_loss,
        }
        
        return results

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        # Note: teacher_input_fn is generally not serializable if it's a lambda
        return config