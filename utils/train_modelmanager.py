# utils/train_modelmanager.py
import os
from datetime import datetime

import tensorflow as tf
import numpy as np

# --------------------------------------------------------------------------- #
#  Absolute imports (no leading dots) – these work when train.py is run directly
# --------------------------------------------------------------------------- #
import parameters as params
from utils import (
    get_data_splits,
    get_calibration_data,
    suppress_all_output,
)
# Remove the problematic import and define the function locally
# from utils.preprocess import _is_qat_model  # REMOVE THIS LINE

def _is_qat_model(model: tf.keras.Model) -> bool:
    """
    Heuristic detection of a QAT‑wrapped model.
    Local implementation to avoid import issues.
    """
    # Check for quantization layers
    for layer in model.layers:
        layer_name = layer.name.lower()
        layer_class = layer.__class__.__name__.lower()
        
        # Check for quantization indicators
        if (hasattr(layer, 'quantize_config') or 
            'quant' in layer_name or 
            'qat' in layer_name or
            'quantize' in layer_class):
            return True
    
    # Check model name and attributes
    model_name = model.name.lower() if hasattr(model, 'name') else ''
    if 'qat' in model_name or 'quant' in model_name:
        return True
    
    # Check if model was created within quantize_scope
    if hasattr(model, '_quantize_scope'):
        return True
        
    return False


class TFLiteModelManager:
    """Handles conversion, checkpointing and best‑model bookkeeping."""

    def __init__(self, output_dir: str, debug: bool = False):
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        self.debug = debug
        self._already_quantised = False

    # -----------------------------------------------------------------
    #  Sanity‑check before conversion
    # -----------------------------------------------------------------
    def verify_model_for_conversion(self, model: tf.keras.Model) -> bool:
        """Run a quick forward‑pass sanity check."""
        try:
            test_input = tf.random.normal([1] + list(params.INPUT_SHAPE))
            out = model(test_input)
            expected = (1, params.NB_CLASSES)
            if out.shape != expected:
                print(f"⚠️  Unexpected output shape {out.shape} (expected {expected})")
            if tf.reduce_any(tf.math.is_nan(out)):
                print("❌ Model produced NaNs")
                return False
            return True
        except Exception as exc:  # pragma: no cover
            print(f"❌ Model verification failed: {exc}")
            return False

    # -----------------------------------------------------------------
    #  Save a trainable checkpoint (Keras 3 .keras format)
    # -----------------------------------------------------------------
    def save_trainable_checkpoint(self, model: tf.keras.Model, accuracy: float, epoch: int):
        ts = datetime.now().strftime("%H%M%S")
        ckpt_path = os.path.join(
            self.output_dir,
            f"checkpoint_epoch_{epoch:03d}_acc_{accuracy:.4f}_{ts}.keras",
        )
        model.save(ckpt_path)  # Keras 3 native saving
        if self.debug:
            print(f"💾 Checkpoint saved: {ckpt_path}")

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_path = os.path.join(self.output_dir, "best_model.keras")
            model.save(best_path)
            if self.debug:
                print(f"🏆 New best model saved: {best_path}")

        return ckpt_path

    # -----------------------------------------------------------------
    #  Build a TFLiteConverter with the correct options
    # -----------------------------------------------------------------
    def _make_converter(
        self,
        model: tf.keras.Model,
        quantize: bool,
        representative_data=None,
    ) -> tf.lite.TFLiteConverter:
        """Create a TFLiteConverter configured for PTQ or QAT."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if not quantize:
            return converter

        # ----- Quantisation path -------------------------------------------------
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # QAT models already embed scales → no representative dataset needed
        if _is_qat_model(model):
            if self.debug:
                print("🔧 QAT model – skipping representative dataset")
        else:
            # PTQ – we need a calibration set
            if representative_data is None:
                # Default calibration using real training data
                def default_calib():
                    (x_train_raw, _), _, _ = get_data_splits()
                    cal = get_calibration_data(
                        x_train_raw[: params.QUANTIZE_NUM_SAMPLES]
                    )
                    for i in range(len(cal)):
                        yield [cal[i : i + 1]]

                converter.representative_dataset = default_calib
            else:
                converter.representative_dataset = representative_data

        # ESP‑DL vs standard UINT8
        if params.ESP_DL_QUANTIZE:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            if self.debug:
                print("🎯 ESP‑DL INT8 quantisation selected")
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            if self.debug:
                print("🎯 Standard UINT8 quantisation selected")

        # Enable the newer quantiser (helps QAT models)
        converter.experimental_new_quantizer = True
        return converter

    # -----------------------------------------------------------------
    #  Quick test of a TFLite model (debug‑only output)
    # -----------------------------------------------------------------
    def test_tflite_model(self, tflite_path: str) -> bool:
        """Load a TFLite model and (optionally) print a short summary."""
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            if self.debug:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print("✅ TFLite model loaded successfully:")
                print(
                    f"   Input : dtype={input_details[0]['dtype']}, shape={input_details[0]['shape']}"
                )
                print(
                    f"   Output: dtype={output_details[0]['dtype']}, shape={output_details[0]['shape']}"
                )
            return True
        except Exception as e:
            print(f"❌ TFLite model test failed: {e}")
            return False

    # -----------------------------------------------------------------
    #  Low‑level file writer (used by both PTQ and QAT paths)
    # -----------------------------------------------------------------
    def _save_tflite_file(
        self, tflite_blob: bytes, filename: str, quantize: bool = False
    ):
        """
        Write a TFLite byte‑blob to disk and (optionally) verify it.

        Returns
        -------
        tuple (tflite_blob, size_kb) on success, (None, 0) on failure.
        """
        try:
            out_path = os.path.join(self.output_dir, filename)
            with open(out_path, "wb") as f:
                f.write(tflite_blob)

            size_kb = len(tflite_blob) / 1024
            tag = "Quantized" if quantize else "Float32"

            if self.debug:
                print(f"💾 Saved {filename} ({tag}): {size_kb:.1f} KB")
                self.test_tflite_model(out_path)
            else:
                # Silent verification – raise only if the file cannot be opened
                try:
                    tf.lite.Interpreter(model_path=out_path).allocate_tensors()
                except Exception as ver_err:
                    print(f"❌ Saved TFLite file verification failed: {ver_err}")
                    return None, 0

            return tflite_blob, size_kb

        except Exception as write_err:  # pragma: no cover
            print(f"❌ Failed to save TFLite file {filename}: {write_err}")
            return None, 0

    # -----------------------------------------------------------------
    #  Public API – save as TFLite (handles PTQ, QAT, and debug mode)
    # -----------------------------------------------------------------
    def save_as_tflite(
        self,
        model: tf.keras.Model,
        filename: str,
        quantize: bool = False,
        representative_data=None,
    ):
        """
        Convert ``model`` to TFLite and write ``filename`` into ``self.output_dir``.
        """
        
        if quantize and self.debug:
            print("🔧 Quantization Debug Info:")
            print(f"   - Model built: {model.built}")
            print(f"   - Input shape: {model.input_shape}")
            print(f"   - Output shape: {model.output_shape}")
            print(f"   - Is QAT model: {_is_qat_model(model)}")
            print(f"   - ESP_DL_QUANTIZE: {params.ESP_DL_QUANTIZE}")
        
        # Only prevent double quantization, not all conversions
        if quantize:
            if self._already_quantised:
                if self.debug:
                    print("Warning: attempted second quantisation – operation ignored")
                return None, 0
            self._already_quantised = True
            
        # 1 Ensure the model graph exists (required for Keras 3)
        if not model.built:
            dummy = tf.zeros([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
            _ = model(dummy)

        # 2 Build the appropriate converter
        converter = self._make_converter(
            model, quantize=quantize, representative_data=representative_data
        )

        # 3 Perform the conversion (silently unless debug=True)
        try:
            with suppress_all_output(self.debug):
                tflite_blob = converter.convert()
        except Exception as exc:  # pragma: no cover
            print(f"❌ TFLite conversion failed: {exc}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None, 0

        # 4 Write the file (and optionally verify it)
        return self._save_tflite_file(tflite_blob, filename, quantize)

    # -----------------------------------------------------------------
    #  Save the *best* model seen so far (both TFLite + float)
    # -----------------------------------------------------------------
    def save_best_model(
        self,
        model: tf.keras.Model,
        accuracy: float,
        representative_data=None,
    ):
        """Persist the model if it beats the current best validation accuracy."""
        if accuracy <= self.best_accuracy:
            return None

        if not self.verify_model_for_conversion(model):
            print("⚠️  Skipping TFLite conversion – model failed verification")
            return None

        if self.debug:
            print(f"🎯 New best accuracy: {accuracy:.4f} – converting to TFLite…")

        try:
            # Quantised version
            q_blob, q_size = self.save_as_tflite(
                model,
                params.TFLITE_FILENAME,
                quantize=True,
                representative_data=representative_data,
            )
            # Float version
            self.save_as_tflite(
                model,
                params.FLOAT_TFLITE_FILENAME,
                quantize=False,
            )
            if self.debug:
                q_path = os.path.join(self.output_dir, params.TFLITE_FILENAME)
                f_path = os.path.join(self.output_dir, params.FLOAT_TFLITE_FILENAME)
                if os.path.exists(q_path):
                    print(
                        f"💾 Models saved – Quantised: {os.path.getsize(q_path)/1024:.1f} KB, "
                        f"Float: {os.path.getsize(f_path)/1024:.1f} KB"
                    )
            return q_size
        except Exception as exc:  # pragma: no cover
            print(f"❌ Failed to save best TFLite model: {exc}")
            return None