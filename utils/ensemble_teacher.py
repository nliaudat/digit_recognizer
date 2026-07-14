import tensorflow as tf
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Union

import config.distillation as dist_cfg

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


class AdaptiveEnsembleTeacher(tf.keras.Model):
    """
    Multi-teacher ensemble with per-teacher temperature scaling and
    stop_gradient self-distillation support.

    Core insight (from the literature):
        Lower-accuracy teachers have *less confident* soft targets, which
        paradoxically makes them *more informative* for distillation because
        their probability distributions carry richer "dark knowledge" about
        class relationships.  This class amplifies that signal by boosting
        the temperature for low-accuracy teachers.

    Features
    --------
    - Per-teacher temperature: T_i = base_T + boost * accuracy_deficit
      where accuracy_deficit = (median_acc - acc_i) / (median_acc - min_acc + EPS).
      Lower accuracy → higher T → flatter distribution → more dark knowledge.
    - Self-teacher support: one index can be marked as the student's own
      frozen prediction.  Its gradient is stopped so the student can't
      "cheat" by matching its own output.
    - Self-teacher weight ramps linearly over training (start→end)
      via the `progress` parameter.
    - All config values live in ``config/distillation.py``.

    Parameters
    ----------
    teachers : list[tf.keras.Model]
        The teacher models.
    teacher_accuracies : list[float]
        Float accuracy for each teacher (used to compute per-teacher T).
    teacher_weights : list[float] or None
        Fixed weights for each teacher (default: accuracy-proportional).
    self_teacher_idx : int or None
        Index in `teachers` that is the student's own frozen checkpoint.
        If provided, stop_gradient is applied and its weight ramps.
    temperature : float
        Base temperature for the median-accuracy teacher.
    **kwargs
        Passed to tf.keras.Model.
    """

    def __init__(
        self,
        teachers: list[tf.keras.Model],
        teacher_accuracies: list[float],
        teacher_weights: Optional[list[float]] = None,
        self_teacher_idx: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teachers = teachers
        self.num_teachers = len(teachers)
        self.teacher_accuracies = np.array(teacher_accuracies, dtype=np.float32)
        self.self_teacher_idx = self_teacher_idx
        self.base_temperature = temperature if temperature is not None else dist_cfg.ENSEMBLE_BASE_TEMPERATURE
        self.low_acc_boost = dist_cfg.ENSEMBLE_LOW_ACCURACY_TEMP_BOOST

        # ── Weights ────────────────────────────────────────────────────────
        if teacher_weights is not None:
            self.teacher_weights = np.array(teacher_weights, dtype=np.float32)
        else:
            # Default: softmax over accuracies (or uniform if all zero)
            accs = self.teacher_accuracies.copy()
            accs = np.maximum(accs, 1e-7)
            self.teacher_weights = accs / accs.sum()

        # ── Freeze teachers ────────────────────────────────────────────────
        for teacher in self.teachers:
            teacher.trainable = False

        # ── Detect output format for each teacher ──────────────────────────
        self._teacher_formats = []
        for i, teacher in enumerate(self.teachers):
            fmt = _detect_output_format(teacher)
            self._teacher_formats.append(fmt)
            logger.info(f"  AdaptiveTeacher {i} ({teacher.name}): format={fmt}, acc={teacher_accuracies[i]:.4f}")

        # ── Pre-compute per-teacher temperatures ───────────────────────────
        self._teacher_temperatures = self._compute_per_teacher_temps()
        for i, t in enumerate(self._teacher_temperatures):
            tag = " [SELF]" if i == self_teacher_idx else ""
            logger.info(f"  → T[{i}] = {t:.2f}{tag}")

        # ── Self-distillation ramp ─────────────────────────────────────────
        self._self_weight_start = dist_cfg.ENSEMBLE_SELF_DISTILL_WEIGHT_START
        self._self_weight_end = dist_cfg.ENSEMBLE_SELF_DISTILL_WEIGHT_END
        self._self_temperature = dist_cfg.ENSEMBLE_SELF_DISTILL_TEMPERATURE
        self._self_enabled = (
            dist_cfg.ENSEMBLE_SELF_DISTILLATION_ENABLED
            and self_teacher_idx is not None
        )
        if self._self_enabled:
            logger.info(
                f"  Self-distillation enabled: weight {self._self_weight_start} → {self._self_weight_end}, "
                f"T={self._self_temperature}"
            )

        logger.info(
            f"AdaptiveEnsembleTeacher: {self.num_teachers} teachers, "
            f"base_T={temperature}, boost={self.low_acc_boost}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _compute_per_teacher_temps(self) -> np.ndarray:
        """
        Compute per-teacher temperatures based on accuracy rank.

        Teacher at median accuracy → base temperature.
        Teacher below median → boosted:  T_i = base_T + boost * deficit_factor
        where deficit_factor linearly maps [min_acc, median_acc] → [1.0, 0.0].
        """
        accs = self.teacher_accuracies
        n = len(accs)

        if n <= 1:
            return np.array([self.base_temperature], dtype=np.float32)

        # Exclude self-teacher from median calculation if it's marked
        if self.self_teacher_idx is not None and n > 1:
            _accs = np.delete(accs, self.self_teacher_idx)
        else:
            _accs = accs

        median_acc = float(np.median(_accs))
        min_acc = float(_accs.min())
        eps = 1e-7

        temps = np.full(n, fill_value=self.base_temperature, dtype=np.float32)

        for i in range(n):
            if i == self.self_teacher_idx:
                continue  # handled in call()
            deficit = (median_acc - accs[i]) / (median_acc - min_acc + eps)
            deficit = float(np.clip(deficit, 0.0, 1.0))
            temps[i] = self.base_temperature + self.low_acc_boost * deficit

        return temps

    def _get_self_weight(self, progress: float) -> float:
        """Linear ramp from start to end weight."""
        progress = float(np.clip(progress, 0.0, 1.0))
        return (
            self._self_weight_start * (1.0 - progress)
            + self._self_weight_end * progress
        )

    # ── Forward pass ────────────────────────────────────────────────────

    def call(self, inputs: tf.Tensor, training: bool = False, progress: float = 1.0) -> tf.Tensor:
        """
        Forward pass with per-teacher temperature scaling.

        Args:
            inputs: Batch of input images.
            training: Whether in training mode.
            progress: Training progress [0, 1].  Controls the self-teacher
                      weight ramp.  Default 1.0 (fully ramped).

        Returns:
            Weighted ensemble softmax probabilities.
        """
        # Step 1: collect and normalise each teacher's output to softmax
        normalized_outputs = []
        for i, teacher in enumerate(self.teachers):
            raw = teacher(inputs, training=training)
            probs = _normalize_to_softmax(teacher, raw)
            normalized_outputs.append(probs)
            logger.debug(f"  AdaptiveTeacher[{i}]: probs shape={probs.shape}")

        # Step 2: apply per-teacher temperature and re-softmax
        eps = 1e-7
        tempered = []
        for i, probs in enumerate(normalized_outputs):
            if i == self.self_teacher_idx and self._self_enabled:
                # Self-teacher: use self-distillation temperature
                t = self._self_temperature
            else:
                t = self._teacher_temperatures[i]

            # Recover pseudo-logits, scale by temperature, re-softmax
            pseudo_logits = tf.math.log(tf.clip_by_value(probs, eps, 1.0))
            softened = tf.nn.softmax(pseudo_logits / t)
            tempered.append(softened)

        # Step 3: weighted average
        weights = tf.constant(self.teacher_weights, dtype=tf.float32)

        # If self-teacher is active, adjust weights according to progress
        if self._self_enabled:
            w = self._get_self_weight(progress)
            other_count = self.num_teachers - 1
            if other_count > 0:
                # Redistribute: self gets `w`, the rest share (1 - w) proportionally
                self_w = tf.constant(w, dtype=tf.float32)
                # Use numpy array self.teacher_weights (not tf.Tensor weights) to avoid
                # "Scalar tensor has no len()" in graph mode.
                other_weights = tf.constant(
                    [self.teacher_weights[i] for i in range(self.num_teachers) if i != self.self_teacher_idx],
                    dtype=tf.float32,
                )
                other_weights = other_weights / tf.reduce_sum(other_weights)
                other_weights = other_weights * (1.0 - self_w)

                # Rebuild full weight vector
                full_weights = []
                oi = 0
                for i in range(self.num_teachers):
                    if i == self.self_teacher_idx:
                        full_weights.append(self_w)
                    else:
                        full_weights.append(other_weights[oi])
                        oi += 1
                weights = tf.stack(full_weights)
        else:
            # Normalize weights to sum to 1
            weights = weights / tf.reduce_sum(weights)

        # Apply weighted sum
        weighted = tf.zeros_like(tempered[0])
        for i, t_soft in enumerate(tempered):
            w_i = weights[i]
            if i == self.self_teacher_idx and self._self_enabled:
                # ⚠️ Critical: stop gradient on self-teacher branch.
                # The student should move TOWARD its own frozen predictions,
                # not be able to change its predictions to perfectly match itself.
                weighted += w_i * tf.stop_gradient(t_soft)
            else:
                weighted += w_i * t_soft

        logger.debug(f"AdaptiveEnsembleTeacher output shape: {weighted.shape}")
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
            'teacher_weights': self.teacher_weights.tolist(),
            'teacher_accuracies': self.teacher_accuracies.tolist(),
            'self_teacher_idx': self.self_teacher_idx,
            'base_temperature': float(self.base_temperature),
        })
        return config

    def verify(self, test_data, num_samples=100):
        """Verify teacher ensemble is working properly."""
        x_test, y_test = test_data
        predictions = self.predict(x_test[:num_samples])
        acc = np.mean(np.argmax(predictions, axis=1) == y_test[:num_samples])
        logger.info(f"AdaptiveEnsembleTeacher verification accuracy: {acc:.4f}")
        return acc
