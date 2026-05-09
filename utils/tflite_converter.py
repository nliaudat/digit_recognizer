"""
utils/tflite_converter.py
=========================
Strategy pattern for TFLite model conversion.

Provides a clean, extensible set of conversion strategies that replace the
monolithic conversion methods in train_modelmanager.py.

Strategies:
  - DynamicRangeStrategy — weight-only quantization (no representative data)
  - Float16Strategy — float16 quantization
  - FullIntegerStrategy — full int8 quantization with representative dataset
  - QATStrategy — quantization-aware training model conversion
  - AutoStrategy — tries multiple strategies and picks the best

Usage:
    strategy = FullIntegerStrategy(representative_data=dataset)
    tflite_model = strategy.convert(model)
"""

import abc
import logging
import os
import tempfile
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

import parameters as params
from utils import suppress_all_output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Base strategy
# ---------------------------------------------------------------------------

class ConversionStrategy(abc.ABC):
    """Abstract base for a TFLite conversion strategy."""

    def __init__(self, debug: bool = False):
        self.debug = debug

    @abc.abstractmethod
    def convert(self, model: tf.keras.Model) -> bytes:
        """Convert the Keras model to a TFLite blob. Returns raw bytes."""
        ...

    def name(self) -> str:
        """Human-readable strategy name."""
        return self.__class__.__name__.replace("Strategy", "")

    def _apply_xnnpack_fix(self, converter: tf.lite.TFLiteConverter,
                           quantize: bool = False) -> tf.lite.TFLiteConverter:
        """Apply XNNPACK delegate fix (TFLITE_BUILTINS_INT8 for full int8)."""
        if quantize and getattr(params, 'USE_TFLITE_BUILTINS_INT8_ONLY', False):
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            if not hasattr(converter, 'representative_dataset') or converter.representative_dataset is None:
                raise ValueError(
                    "USE_TFLITE_BUILTINS_INT8_ONLY requires a representative_dataset!"
                )
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif getattr(params, 'DISABLE_XNNPACK', True):
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        return converter

    def _validate_no_delegates(self, tflite_model: bytes, model_name: str = "model"):
        """Validate that the TFLite model has no XNNPACK delegates baked in."""
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(tflite_model)
            tmp_path = f.name
        try:
            interp = tf.lite.Interpreter(
                model_path=tmp_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            interp.allocate_tensors()
            logger.info(f"✅ [{model_name}] No delegates — TFLite Micro compatible")
        except Exception as e:
            logger.error(f"❌ [{model_name}] Delegate validation FAILED: {e}")
            raise
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
#  Concrete strategies
# ---------------------------------------------------------------------------

class DynamicRangeStrategy(ConversionStrategy):
    """Weight-only quantization — no representative dataset needed."""

    def convert(self, model: tf.keras.Model) -> bytes:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # No representative dataset — weights only
        converter = self._apply_xnnpack_fix(converter, quantize=False)
        with suppress_all_output(not self.debug):
            return converter.convert()


class Float16Strategy(ConversionStrategy):
    """Float16 quantization — reduces model size by ~50%."""

    def convert(self, model: tf.keras.Model) -> bytes:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter = self._apply_xnnpack_fix(converter, quantize=False)
        with suppress_all_output(not self.debug):
            return converter.convert()


class FullIntegerStrategy(ConversionStrategy):
    """Full int8 quantization with representative dataset."""

    def __init__(self, representative_dataset: Callable, debug: bool = False):
        super().__init__(debug=debug)
        self.representative_dataset = representative_dataset

    def convert(self, model: tf.keras.Model) -> bytes:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_dataset
        converter = self._apply_xnnpack_fix(converter, quantize=True)
        with suppress_all_output(not self.debug):
            tflite_model = converter.convert()
        self._validate_no_delegates(tflite_model, "full_integer")
        return tflite_model


class QATStrategy(ConversionStrategy):
    """Conversion for Quantization-Aware Training models."""

    def __init__(self, representative_dataset: Optional[Callable] = None,
                 debug: bool = False):
        super().__init__(debug=debug)
        self.representative_dataset = representative_dataset

    def convert(self, model: tf.keras.Model) -> bytes:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if self.representative_dataset is not None:
            converter.representative_dataset = self.representative_dataset

        # QAT-specific settings
        converter.experimental_new_quantizer = True
        converter._experimental_disable_per_channel = False

        converter = self._apply_xnnpack_fix(converter, quantize=True)
        with suppress_all_output(not self.debug):
            tflite_model = converter.convert()
        self._validate_no_delegates(tflite_model, "qat")
        return tflite_model


class AutoStrategy(ConversionStrategy):
    """
    Try multiple strategies and return the best one based on accuracy.

    Strategies tried (in order):
      1. QATStrategy (if model is QAT)
      2. FullIntegerStrategy
      3. DynamicRangeStrategy (fallback)
    """

    def __init__(self, representative_dataset: Callable,
                 x_test: np.ndarray, y_test: np.ndarray,
                 debug: bool = False):
        super().__init__(debug=debug)
        self.representative_dataset = representative_dataset
        self.x_test = x_test
        self.y_test = y_test

    def convert(self, model: tf.keras.Model) -> bytes:
        from utils.train_qat_helper import _is_qat_model

        strategies = []
        is_qat = _is_qat_model(model)

        if is_qat:
            strategies.append(("QAT", QATStrategy(self.representative_dataset, self.debug)))

        strategies.append(("Full Integer", FullIntegerStrategy(self.representative_dataset, self.debug)))
        strategies.append(("Dynamic Range", DynamicRangeStrategy(self.debug)))

        best_blob = None
        best_acc = -1.0
        best_name = "None"

        for name, strategy in strategies:
            logger.info(f"  Trying {name}...")
            try:
                blob = strategy.convert(model)
                acc = self._quick_evaluate(blob)
                logger.info(f"    {name}: accuracy={acc:.4f}, size={len(blob)/1024:.1f}KB")
                if acc > best_acc:
                    best_acc = acc
                    best_blob = blob
                    best_name = name
            except Exception as e:
                logger.warning(f"    {name} failed: {e}")

        if best_blob is None:
            raise RuntimeError("All conversion strategies failed!")

        logger.info(f"🏆 Best strategy: {best_name} (accuracy={best_acc:.4f})")
        return best_blob

    def _quick_evaluate(self, tflite_model: bytes, num_samples: int = 100) -> float:
        """Quick accuracy evaluation of a TFLite model blob."""
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(tflite_model)
            tmp_path = f.name
        try:
            interpreter = tf.lite.Interpreter(model_path=tmp_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]

            correct = 0
            total = min(num_samples, len(self.x_test))
            for i in range(total):
                input_data = self.x_test[i:i+1].astype(np.float32)
                if input_data.max() <= 1.0:
                    if input_details['dtype'] == np.int8:
                        input_data = (input_data * 255.0 - 128.0).astype(np.int8)
                    elif input_details['dtype'] == np.uint8:
                        input_data = (input_data * 255.0).astype(np.uint8)

                interpreter.set_tensor(input_details['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details['index'])
                pred = np.argmax(output)
                true = int(np.argmax(self.y_test[i])) if self.y_test.ndim > 1 else int(self.y_test[i])
                if pred == true:
                    correct += 1
            return correct / total
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def create_converter(strategy_name: str = "auto", **kwargs) -> ConversionStrategy:
    """
    Factory function to create a conversion strategy by name.

    Args:
        strategy_name: One of "auto", "dynamic_range", "float16",
                       "full_integer", "qat"
        **kwargs: Passed to the strategy constructor.

    Returns:
        A ConversionStrategy instance.
    """
    strategy_map = {
        "dynamic_range": DynamicRangeStrategy,
        "float16": Float16Strategy,
        "full_integer": FullIntegerStrategy,
        "qat": QATStrategy,
        "auto": AutoStrategy,
    }
    cls = strategy_map.get(strategy_name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                         f"Available: {list(strategy_map.keys())}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
#  Convenience: build representative dataset from cached data
# ---------------------------------------------------------------------------

def build_representative_dataset(x_train_raw: np.ndarray,
                                 num_samples: Optional[int] = None,
                                 preprocess_fn: Optional[Callable] = None):
    """
    Build a representative dataset generator for TFLite calibration.

    Args:
        x_train_raw: Raw uint8 training images.
        num_samples: Number of calibration samples (default: QUANTIZE_NUM_SAMPLES).
        preprocess_fn: Preprocessing function (default: preprocess_for_inference).

    Returns:
        A generator function suitable for converter.representative_dataset.
    """
    if num_samples is None:
        num_samples = getattr(params, 'QUANTIZE_NUM_SAMPLES', 200)
    if preprocess_fn is None:
        from utils.preprocess import preprocess_for_inference
        preprocess_fn = preprocess_for_inference

    calibration_data = preprocess_fn(x_train_raw[:num_samples])

    # Ensure float32 [0, 1]
    if calibration_data.dtype != np.float32:
        calibration_data = calibration_data.astype(np.float32)
    if calibration_data.max() > 1.0:
        calibration_data = calibration_data / 255.0

    def _generator():
        for i in range(len(calibration_data)):
            yield [calibration_data[i:i+1]]

    return _generator
