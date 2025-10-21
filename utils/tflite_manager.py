import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from contextlib import contextmanager
import parameters as params
from utils.logging import log_print
from utils import get_data_splits, preprocess_images

class TFLiteModelManager:
    def __init__(self, output_dir, debug=False):
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        self.debug = debug

    def _completely_suppress_output(self):
        """Completely suppress all output during TFLite conversion and model export"""
        @contextmanager
        def suppress_output():
            # Set TensorFlow C++ backend log level
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            # Set TensorFlow Python logger level
            try:
                tf.get_logger().setLevel('ERROR')
            except Exception:
                pass
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
        return suppress_output()

    def verify_model_for_conversion(self, model):
        """Verify model is compatible with TFLite conversion"""
        try:
            test_input = tf.random.normal([1] + list(params.INPUT_SHAPE))
            test_output = model(test_input)
            expected_output_shape = (1, params.NB_CLASSES)
            if test_output.shape != expected_output_shape:
                log_print(f"⚠️  Output shape mismatch: {test_output.shape} vs {expected_output_shape}", level=1)
            if tf.reduce_any(tf.math.is_nan(test_output)):
                log_print("❌ Model output contains NaN values", level=1)
                return False
            return True
        except Exception as e:
            log_print(f"❌ Model verification failed: {e}", level=1)
            return False

    def save_trainable_checkpoint(self, model, accuracy, epoch):
        """Save model in trainable format"""
        timestamp = datetime.now().strftime("%H%M%S")
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch:03d}_acc_{accuracy:.4f}_{timestamp}.keras")
        model.save(checkpoint_path)
        if self.debug:
            log_print(f"💾 Saved trainable checkpoint: {checkpoint_path}", level=2)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_checkpoint_path = os.path.join(self.output_dir, "best_model.keras")
            model.save(best_checkpoint_path)
            if self.debug:
                log_print(f"🏆 New best model saved: {best_checkpoint_path}", level=2)
        return checkpoint_path

    def _is_qat_model(self, model):
        """Check if model is a QAT model"""
        for layer in model.layers:
            if hasattr(layer, 'quantize_config') or 'quant' in layer.name.lower():
                return True
        return False

    def _convert_qat_model(self, model, filename, representative_data=None):
        """Convert QAT model to TFLite with proper representative dataset"""
        try:
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("🎯 Converting QAT model to TFLite...", level=2)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_data is None:
                def qat_representative_dataset():
                    (x_train_raw, y_train_raw), _, _ = get_data_splits()
                    calibration_data = x_train_raw[:params.QUANTIZE_NUM_SAMPLES]
                    calibration_processed = preprocess_images(calibration_data, for_training=False)
                    if calibration_processed.dtype != np.float32:
                        if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                            log_print(f"🔄 Converting calibration data from {calibration_processed.dtype} to float32", level=2)
                        calibration_processed = calibration_processed.astype(np.float32)
                        if calibration_processed.max() > 1.0:
                            calibration_processed = calibration_processed / 255.0
                    if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                        log_print(f"🔧 QAT Calibration: {len(calibration_processed)} samples, "
                                  f"dtype: {calibration_processed.dtype}, "
                                  f"range: [{calibration_processed.min():.3f}, {calibration_processed.max():.3f}]", level=2)
                    if calibration_processed.dtype != np.float32:
                        raise ValueError(f"QAT calibration data must be float32, got {calibration_processed.dtype}")
                    for i in range(len(calibration_processed)):
                        yield [calibration_processed[i:i+1]]
                converter.representative_dataset = qat_representative_dataset
            else:
                converter.representative_dataset = representative_data
            if params.ESP_DL_QUANTIZE:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                    log_print("🔧 QAT → ESP-DL INT8 quantization", level=2)
            else:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                    log_print("🔧 QAT → Standard UINT8 quantization", level=2)
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                tflite_model = converter.convert()
                log_print("🔧 TFLite conversion completed with debug output", level=2)
            else:
                with self._completely_suppress_output():
                    tflite_model = converter.convert()
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print(f"✅ QAT conversion successful", level=2)
            return self._save_tflite_file(tflite_model, filename, True)
        except Exception as e:
            log_print(f"❌ QAT conversion failed: {e}", level=1)
            log_print("🔄 Attempting QAT fallback conversion...", level=1)
            return self._convert_qat_model_fallback_enhanced(model, filename)

    def _convert_qat_model_fallback_enhanced(self, model, filename):
        """Enhanced fallback conversion for QAT model with better debugging"""
        try:
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("🔄 Trying enhanced QAT fallback conversion...", level=2)
                log_print("🔍 Diagnosing QAT conversion issue...", level=2)
            test_input = tf.random.normal([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
            try:
                test_output = model(test_input)
                if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                    log_print(f"✅ Model accepts float32 inputs: output shape {test_output.shape}", level=2)
            except Exception as e:
                if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                    log_print(f"❌ Model input test failed: {e}", level=1)
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("🔧 Strategy 1: Dynamic range quantization...", level=2)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                tflite_model = converter.convert()
            else:
                with self._completely_suppress_output():
                    tflite_model = converter.convert()
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("✅ Dynamic range quantization successful", level=2)
            return self._save_tflite_file(tflite_model, filename, True)
        except Exception as e:
            log_print(f"❌ Enhanced QAT fallback failed: {e}", level=1)
            log_print("🔄 Final fallback: Saving without quantization...", level=1)
            return self.save_as_tflite(model, filename, quantize=False)

    def _convert_qat_model_fallback(self, model, filename):
        """Legacy fallback - redirect to enhanced version"""
        return self._convert_qat_model_fallback_enhanced(model, filename)

    def save_as_tflite(self, model, filename, quantize=False, representative_data=None):
        """Save model as TFLite with proper QAT handling"""
        try:
            if not model.built:
                dummy_input = tf.zeros([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
                _ = model(dummy_input)
            if quantize and self._is_qat_model(model):
                return self._convert_qat_model(model, filename, representative_data)
            return self.save_as_tflite_savedmodel(model, filename, quantize, representative_data)
        except Exception as e:
            log_print(f"❌ TFLite conversion failed: {e}", level=1)
            return self.save_as_tflite_savedmodel(model, filename, quantize, representative_data)

    def save_as_tflite_savedmodel(self, model, filename, quantize=False, representative_data=None):
        """Use SavedModel approach for conversion"""
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = os.path.join(temp_dir, "saved_model")
                model.export(model_dir)
                converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
                if quantize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    if representative_data is not None:
                        converter.representative_dataset = representative_data
                    else:
                        def default_representative_dataset():
                            for _ in range(params.QUANTIZE_NUM_SAMPLES):
                                data = np.random.rand(1, *params.INPUT_SHAPE).astype(np.float32)
                                yield [data]
                        converter.representative_dataset = default_representative_dataset
                    if params.ESP_DL_QUANTIZE:
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                        converter.inference_input_type = tf.int8
                        converter.inference_output_type = tf.int8
                    else:
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                        converter.inference_input_type = tf.uint8
                        converter.inference_output_type = tf.uint8
                with self._completely_suppress_output():
                    tflite_model = converter.convert()
                return self._save_tflite_file(tflite_model, filename, quantize)
        except Exception as e:
            log_print(f"❌ SavedModel conversion failed: {e}", level=1)
            raise

    def _save_tflite_file(self, tflite_model, filename, quantize):
        """Save TFLite model to file"""
        model_path = os.path.join(self.output_dir, filename)
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        model_size_kb = len(tflite_model) / 1024
        if self.debug:
            quant_type = "INT8" if (quantize and params.ESP_DL_QUANTIZE) else "UINT8" if quantize else "Float32"
            log_print(f"💾 Saved {filename} ({quant_type}): {model_size_kb:.1f} KB", level=2)
        return tflite_model, model_size_kb

    def save_best_model(self, model, accuracy, representative_data=None):
        """Save model if it's the best so far"""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            if not self.verify_model_for_conversion(model):
                log_print("⚠️  Skipping TFLite conversion due to model issues", level=1)
                return None
            if self.debug:
                log_print(f"🎯 New best accuracy: {accuracy:.4f}, saving TFLite model...", level=2)
            try:
                tflite_model, size_kb = self.save_as_tflite(
                    model, 
                    params.TFLITE_FILENAME, 
                    quantize=True, 
                    representative_data=representative_data
                )
                self.save_as_tflite(
                    model, 
                    params.FLOAT_TFLITE_FILENAME, 
                    quantize=False, 
                    representative_data=representative_data
                )
                log_print(f"🏆 Best TFLite model saved: {params.TFLITE_FILENAME} ({size_kb:.1f} KB)", level=2)
                return tflite_model
            except Exception as e:
                log_print(f"❌ Failed to save best TFLite model: {e}", level=1)
                return None
        return None
