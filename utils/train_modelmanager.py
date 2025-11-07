# utils/train_modelmanager.py
import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from utils.train_qat_helper import _is_qat_model

# --------------------------------------------------------------------------- #
#  Absolute imports (no leading dots) – these work when train.py is run directly
# --------------------------------------------------------------------------- #
import parameters as params
from utils import (
    get_data_splits,
    get_calibration_data,
    suppress_all_output,
)
from utils.logging import log_print


class TFLiteModelManager:
    """Handles conversion, checkpointing and best model bookkeeping."""

    def __init__(self, output_dir: str, debug: bool = False):
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        self.debug = debug
        self._already_quantised = False

    # -----------------------------------------------------------------
    #  Sanity check before conversion
    # -----------------------------------------------------------------
    def verify_model_for_conversion(self, model: tf.keras.Model) -> bool:
        """Run a quick forwardpass sanity check."""
        try:
            test_input = tf.random.normal([1] + list(params.INPUT_SHAPE))
            out = model(test_input)
            expected = (1, params.NB_CLASSES)
            if out.shape != expected:
                print(f"Unexpected output shape {out.shape} (expected {expected})")
            if tf.reduce_any(tf.math.is_nan(out)):
                print("Model produced NaNs")
                return False
            return True
        except Exception as exc:  # pragma: no cover
            print(f"Model verification failed: {exc}")
            return False


    def _is_qat_model(self, model: tf.keras.Model) -> bool:
        """More reliable QAT model detection"""
        # This local method SHADOWS the imported one!
    
    
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
            print(f"Checkpoint saved: {ckpt_path}")

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_path = os.path.join(self.output_dir, "best_model.keras")
            model.save(best_path)
            if self.debug:
                print(f"New best model saved: {best_path}")

        return ckpt_path

    # -----------------------------------------------------------------
    #  Enhanced Conversion Methods
    # -----------------------------------------------------------------
    def _convert_qat_model(self, model, filename, representative_data=None):
        """Convert QAT model to TFLite with proper representative dataset"""
        try:
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("🎯 Converting QAT model to TFLite...", level=2)

            # Use direct conversion for QAT models
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # QAT models need representative dataset with EXACTLY the same preprocessing as training
            if representative_data is None:
                def qat_representative_dataset():
                    from utils import get_data_splits
                    from utils.preprocess import preprocess_for_training  # Use training preprocessing!
                    
                    # Get raw training data
                    (x_train_raw, y_train_raw), _, _ = get_data_splits()
                    
                    # Use a subset for calibration - ensure we have enough samples
                    # num_samples = min(100, len(x_train_raw), params.QUANTIZE_NUM_SAMPLES)
                    num_samples = min(len(x_train_raw), params.QUANTIZE_NUM_SAMPLES)
                    calibration_data = x_train_raw[:num_samples]
                    
                    # CRITICAL FIX: Use the SAME preprocessing as during QAT training
                    calibration_processed = preprocess_for_training(calibration_data)  # CHANGED: for_training=True
                    
                    # Ensure proper data type and range for QAT
                    if calibration_processed.dtype != np.float32:
                        calibration_processed = calibration_processed.astype(np.float32)
                    
                    # For QAT, data should be in the same range as during training
                    # QAT training uses [0, 1] range, so calibration should too
                    if calibration_processed.max() > 1.0:
                        calibration_processed = calibration_processed / 255.0
                    
                    if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                        print(f"🔧 QAT Calibration: {len(calibration_processed)} samples, "
                              f"dtype: {calibration_processed.dtype}, "
                              f"range: [{calibration_processed.min():.3f}, {calibration_processed.max():.3f}]")
                    
                    # Verify data matches QAT training expectations
                    if calibration_processed.dtype != np.float32:
                        raise ValueError(f"QAT calibration data must be float32, got {calibration_processed.dtype}")
                    
                    # Yield data in the correct format
                    for i in range(len(calibration_processed)):
                        yield [calibration_processed[i:i+1]]
                
                converter.representative_dataset = qat_representative_dataset
            else:
                converter.representative_dataset = representative_data

            # QAT-specific conversion settings
            if params.ESP_DL_QUANTIZE:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                if self.debug:
                    print("🎯 ESP-DL INT8 quantization for QAT")
            else:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                if self.debug:
                    print("🎯 Standard UINT8 quantization for QAT")

            # Additional QAT-specific settings
            converter.experimental_new_quantizer = True  # Use new quantizer for better QAT support
            converter._experimental_disable_per_channel = False  # Enable per-channel quantization
            
            # Convert with output suppression
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
                
            return self._save_tflite_file(tflite_model, filename, True)
            
        except Exception as e:
            log_print(f"❌ QAT conversion failed: {e}", level=1)
            # Enhanced fallback with better debugging
            return self._convert_qat_model_fallback_enhanced(model, filename)

    def _convert_qat_model_fallback_enhanced(self, model, filename):
        """Enhanced fallback conversion for QAT model with comprehensive debugging"""
        try:
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("🔄 Trying enhanced QAT fallback conversion...", level=2)
                log_print("🔍 Diagnosing QAT conversion issue...", level=2)
            
            # Test model with sample input
            test_input = tf.random.normal([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
            try:
                test_output = model(test_input)
                if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                    log_print(f"✅ Model accepts float32 inputs: output shape {test_output.shape}", level=2)
            except Exception as e:
                if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                    log_print(f"❌ Model input test failed: {e}", level=1)
            
            # Strategy 1: Dynamic range quantization only (most reliable)
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("🔧 Strategy 1: Dynamic range quantization only...", level=2)
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Don't use representative dataset in fallback
            # This often causes issues with QAT models
            
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
            
            # Final fallback: Just save the model without quantization
            log_print("🔄 Final fallback: Saving without quantization...", level=1)
            return self.save_as_tflite_simple_keras3(model, filename, quantize=False)

    def _convert_qat_model_fallback(self, model, filename):
        """Legacy fallback - redirect to enhanced version"""
        return self._convert_qat_model_fallback_enhanced(model, filename)

    def _convert_standard_quantized(self, model, filename, representative_data=None):
        """Convert standard (non-QAT) model to quantized TFLite"""
        try:
            if self.debug:
                print("🔧 Converting standard model with quantization...")
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Use representative dataset for proper quantization
            if representative_data is None:
                def default_representative_dataset():
                    from utils import get_data_splits
                    from utils.preprocess import preprocess_for_inference
                    
                    (x_train_raw, _), _, _ = get_data_splits()
                    calibration_data = x_train_raw[:params.QUANTIZE_NUM_SAMPLES]
                    calibration_processed = preprocess_for_inference(calibration_data)
                    
                    # Ensure proper format for quantization
                    if calibration_processed.dtype != np.float32:
                        calibration_processed = calibration_processed.astype(np.float32)
                    if calibration_processed.max() > 1.0:
                        calibration_processed = calibration_processed / 255.0
                    
                    if self.debug:
                        print(f"🔧 Calibration data: {calibration_processed.dtype}, "
                              f"range: [{calibration_processed.min():.3f}, {calibration_processed.max():.3f}]")
                    
                    for i in range(len(calibration_processed)):
                        yield [calibration_processed[i:i+1]]
                
                converter.representative_dataset = default_representative_dataset
            else:
                converter.representative_dataset = representative_data
            
            # Set quantization based on ESP-DL setting
            if params.ESP_DL_QUANTIZE:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                if self.debug:
                    print("🎯 ESP-DL INT8 quantization")
            else:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                if self.debug:
                    print("🎯 Standard UINT8 quantization")
            
            # Convert with output suppression
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            return self._save_tflite_file(tflite_model, filename, True)
            
        except Exception as e:
            if self.debug:
                print(f"❌ Standard quantization failed: {e}")
                print("🔄 Falling back to dynamic range quantization...")
            
            # Fallback to dynamic range only
            return self.save_as_tflite_simple_keras3(model, filename, quantize=True)

    # -----------------------------------------------------------------
    #  Enhanced Conversion Strategy Testing
    # -----------------------------------------------------------------
    def test_conversion_strategies(self, model, x_test, y_test, x_train_raw):
        """Test multiple conversion strategies to find the best one"""
        print("\n🧪 TESTING CONVERSION STRATEGIES")
        print("=" * 50)
        
        strategies = []
        
        # Strategy 1: Enhanced QAT approach
        print("1. Testing enhanced QAT conversion...")
        try:
            tflite_blob, size = self._convert_qat_model(model, "test_enhanced_qat.tflite")
            if tflite_blob is not None:
                accuracy = self._evaluate_tflite_model("test_enhanced_qat.tflite", x_test, y_test)
                strategies.append(("Enhanced QAT", accuracy, size))
                print(f"   Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Strategy 2: Dynamic range only (no representative data)
        print("2. Testing dynamic range only...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # No representative dataset
            tflite_model = converter.convert()
            with open("test_dynamic_range.tflite", "wb") as f:
                f.write(tflite_model)
            accuracy = self._evaluate_tflite_model("test_dynamic_range.tflite", x_test, y_test)
            strategies.append(("Dynamic Range Only", accuracy, len(tflite_model)/1024))
            print(f"   Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Strategy 3: Float16 fallback
        if not params.ESP_DL_QUANTIZE:  # Float16 not supported for ESP-DL
            print("3. Testing Float16 fallback...")
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                tflite_model = converter.convert()
                with open("test_float16.tflite", "wb") as f:
                    f.write(tflite_model)
                accuracy = self._evaluate_tflite_model("test_float16.tflite", x_test, y_test)
                strategies.append(("Float16", accuracy, len(tflite_model)/1024))
                print(f"   Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"   Failed: {e}")
        
        # Cleanup test files
        for fname in ["test_enhanced_qat.tflite", "test_dynamic_range.tflite", "test_float16.tflite"]:
            if os.path.exists(fname):
                os.remove(fname)
        
        # Print results
        print(f"\n📊 CONVERSION STRATEGY RESULTS:")
        for name, acc, size in sorted(strategies, key=lambda x: x[1], reverse=True):
            print(f"   {name}: {acc:.4f} accuracy, {size:.1f} KB")
        
        # Return the best strategy
        if strategies:
            best_strategy = max(strategies, key=lambda x: x[1])
            print(f"🏆 BEST STRATEGY: {best_strategy[0]} ({best_strategy[1]:.4f})")
            return best_strategy[0]
        return None

    def _evaluate_tflite_model(self, tflite_path, x_test, y_test):
        """Quick evaluation of TFLite model accuracy"""
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            correct = 0
            total = min(100, len(x_test))  # Quick test with 100 samples
            
            for i in range(total):
                # Prepare input based on model requirements
                input_data = x_test[i:i+1]
                if input_details[0]['dtype'] == np.int8:
                    input_data = (input_data * 255 - 128).astype(np.int8)
                elif input_details[0]['dtype'] == np.uint8:
                    input_data = (input_data * 255).astype(np.uint8)
                else:
                    input_data = input_data.astype(np.float32)
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                
                pred = np.argmax(output)
                true = np.argmax(y_test[i]) if len(y_test.shape) > 1 else y_test[i]
                
                if pred == true:
                    correct += 1
            
            return correct / total
        except Exception as e:
            print(f"❌ TFLite evaluation failed: {e}")
            return 0.0

    def _get_test_data(self):
        """Get test data for strategy testing"""
        from utils import get_data_splits
        from utils.preprocess import preprocess_for_training, preprocess_for_inference
        
        (x_train_raw, y_train_raw), (x_val_raw, y_val_raw), (x_test_raw, y_test_raw) = get_data_splits()
        
        # Use small subsets for quick testing
        x_test = preprocess_for_inference(x_test_raw[:100])
        y_test = y_test_raw[:100]
        
        # Convert labels if needed
        if params.MODEL_ARCHITECTURE == "original_haverland":
            y_test = tf.keras.utils.to_categorical(y_test, params.NB_CLASSES)
        
        return x_test, y_test, x_train_raw

    # -----------------------------------------------------------------
    #  Enhanced Main Conversion Methods
    # -----------------------------------------------------------------
    def save_as_tflite_enhanced(self, model, filename, quantize=False, representative_data=None):
        """Enhanced version with strategy testing and validation"""
        try:
            if self.debug:
                print(f"🔧 Starting enhanced TFLite conversion for {filename}...")
            
            # Step 1: Validate model
            if not self.validate_model_before_conversion(model):
                print("🚨 Model validation failed - cannot convert")
                return None, 0
            
            # Step 2: For QAT models, test multiple strategies
            if quantize and self._is_qat_model(model):
                print("🎯 QAT Model: Testing conversion strategies...")
                
                # Test different approaches
                x_test, y_test, x_train_raw = self._get_test_data()
                best_strategy = self.test_conversion_strategies(model, x_test, y_test, x_train_raw)
                
                if best_strategy == "Enhanced QAT":
                    return self._convert_qat_model(model, filename, representative_data)
                elif best_strategy == "Dynamic Range Only":
                    return self._convert_dynamic_range_only(model, filename)
                elif best_strategy == "Float16":
                    return self._convert_float16(model, filename)
            
            # Step 3: Use standard logic for other cases
            return self.save_as_tflite(model, filename, quantize, representative_data)
            
        except Exception as e:
            print(f"❌ Enhanced conversion failed: {e}")
            # Final fallback
            return self.save_as_tflite_simple_keras3(model, filename, quantize=False)

    def validate_model_before_conversion(self, model):
        """Validate model is ready for conversion"""
        if self.debug:
            print("\n🔍 VALIDATING MODEL BEFORE CONVERSION")
            print("=" * 50)
        
        # Test 1: Model can handle inference
        try:
            test_input = tf.random.uniform([1] + list(params.INPUT_SHAPE), 0, 1, dtype=tf.float32)
            output = model(test_input)
            if self.debug:
                print(f"✅ Model inference test: output shape {output.shape}")
        except Exception as e:
            print(f"❌ Model inference failed: {e}")
            return False
        
        # Test 2: Check for QAT layers
        if self._is_qat_model(model):
            qat_layers = sum(1 for layer in model.layers if hasattr(layer, 'quantize_config'))
            print(f"✅ QAT model detected: {qat_layers} quantization layers")
        else:
            if self.debug:
                print("ℹ️  Standard model (non-QAT)")
        
        # Test 3: Check output range
        test_outputs = []
        for _ in range(5):
            test_input = tf.random.uniform([1] + list(params.INPUT_SHAPE), 0, 1, dtype=tf.float32)
            output = model(test_input)
            test_outputs.append(output.numpy())
        
        all_outputs = np.concatenate(test_outputs)
        if self.debug:
            print(f"✅ Output range: [{all_outputs.min():.6f}, {all_outputs.max():.6f}]")
        
        # Check for problematic outputs
        if np.any(np.isnan(all_outputs)):
            print("🚨 CRITICAL: Model producing NaN outputs!")
            return False
        if np.any(np.isinf(all_outputs)):
            print("🚨 CRITICAL: Model producing Inf outputs!")
            return False
        
        return True

    def _convert_dynamic_range_only(self, model, filename):
        """Convert with dynamic range quantization only"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # No representative dataset
        tflite_model = converter.convert()
        return self._save_tflite_file(tflite_model, filename, True)

    def _convert_float16(self, model, filename):
        """Convert with float16 quantization"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        return self._save_tflite_file(tflite_model, filename, True)

    # -----------------------------------------------------------------
    #  Original Conversion Methods (Updated)
    # -----------------------------------------------------------------
    def save_as_tflite(self, model, filename, quantize=False, representative_data=None):
        """Save model as TFLite with proper QAT handling and debug control"""
        try:
            if self.debug:
                print(f"🔧 Converting {filename} to TFLite...")
                print(f"   Quantize: {quantize}, QAT Model: {self._is_qat_model(model)}")
            
            # Ensure model is built
            if not model.built:
                dummy_input = tf.zeros([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
                _ = model(dummy_input)
            
            # Handle QAT models specifically
            if quantize and self._is_qat_model(model):
                if self.debug:
                    print("🎯 Converting QAT model to quantized TFLite...")
                return self._convert_qat_model(model, filename, representative_data)
            
            # For non-QAT models, use the simple Keras 3 approach but ensure quantization works
            if quantize:
                if self.debug:
                    print("🎯 Converting standard model to quantized TFLite...")
                return self._convert_standard_quantized(model, filename, representative_data)
            else:
                if self.debug:
                    print("🔧 Converting to float TFLite...")
                return self.save_as_tflite_simple_keras3(model, filename, quantize=False)
                
        except Exception as e:
            if self.debug:
                print(f"❌ Primary TFLite conversion failed: {e}")
                print("🔄 Falling back to saved model approach...")
            
            # Fallback to saved model approach
            return self.save_as_tflite_savedmodel(model, filename, quantize, representative_data)

    def save_as_tflite_simple_keras3(self, model, filename, quantize=False):
        """Simple Keras 3 compatible conversion"""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                # No representative dataset for simple conversion
            
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            return self._save_tflite_file(tflite_model, filename, quantize)
                
        except Exception as e:
            print(f"❌ Simple Keras 3 conversion failed: {e}")
            raise

    def save_as_tflite_savedmodel(self, model, filename, quantize=False, representative_data=None):
        """Fallback conversion using SavedModel format"""
        try:
            # Save model temporarily
            temp_dir = os.path.join(self.output_dir, "temp_savedmodel")
            os.makedirs(temp_dir, exist_ok=True)
            model.save(temp_dir)
            
            # Convert from SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                if representative_data is not None:
                    converter.representative_dataset = representative_data
            
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
            
            return self._save_tflite_file(tflite_model, filename, quantize)
                
        except Exception as e:
            print(f"❌ SavedModel conversion failed: {e}")
            raise

    def _completely_suppress_output(self):
        """Completely suppress all output during conversion"""
        import os
        import sys
        from contextlib import contextmanager
        
        @contextmanager
        def suppress():
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
        
        return suppress()

    # -----------------------------------------------------------------
    #  Build a TFLiteConverter with the correct options
    # -----------------------------------------------------------------
    def _make_converter(self, model: tf.keras.Model,
                        quantize: bool,
                        representative_data=None) -> tf.lite.TFLiteConverter:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if not quantize:
            return converter

        # -------------------- Quantisation path --------------------
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # QAT models already embed scales → no representative dataset needed
        if self._is_qat_model(model):
            # Nothing to do – the fake quant layers provide the scales
            pass
        else:
            # PTQ – we need a calibration set
            if representative_data is None:
                # Default calibration – use the **new** helper that returns float32 data
                def default_calib():
                    (x_train_raw, _), _, _ = get_data_splits()
                    cal = get_calibration_data(
                        x_train_raw[: params.QUANTIZE_NUM_SAMPLES]
                    )
                    for batch in cal():
                        yield batch
                converter.representative_dataset = default_calib
            else:
                converter.representative_dataset = representative_data

            # Choose the correct integer type for the target platform
            if params.ESP_DL_QUANTIZE:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            else:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8

            # Enable the newer quantiser (helps QAT models)
            converter.experimental_new_quantizer = True
        return converter

    # -----------------------------------------------------------------
    #  Quick test of a TFLite model (debug only output)
    # -----------------------------------------------------------------
    def test_tflite_model(self, tflite_path: str) -> bool:
        """Load a TFLite model and (optionally) print a short summary."""
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            if self.debug:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print("TFLite model loaded successfully:")
                print(
                    f"   Input : dtype={input_details[0]['dtype']}, shape={input_details[0]['shape']}"
                )
                print(
                    f"   Output: dtype={output_details[0]['dtype']}, shape={output_details[0]['shape']}"
                )
            return True
        except Exception as e:
            print(f"TFLite model test failed: {e}")
            return False

    # -----------------------------------------------------------------
    #  Low level file writer (used by both PTQ and QAT paths)
    # -----------------------------------------------------------------
    def _save_tflite_file(
        self, tflite_blob: bytes, filename: str, quantize: bool = False
    ):
        """
        Write a TFLite byte blob to disk and (optionally) verify it.

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
                print(f"Saved {filename} ({tag}): {size_kb:.1f} KB")
                self.test_tflite_model(out_path)
            else:
                # Silent verification – raise only if the file cannot be opened
                try:
                    tf.lite.Interpreter(model_path=out_path).allocate_tensors()
                except Exception as ver_err:
                    print(f"Saved TFLite file verification failed: {ver_err}")
                    return None, 0

            return tflite_blob, size_kb

        except Exception as write_err:  # pragma: no cover
            print(f"Failed to save TFLite file {filename}: {write_err}")
            return None, 0

    # -----------------------------------------------------------------
    #  Public API – save as TFLite (handles PTQ, QAT, and debug mode)
    # -----------------------------------------------------------------
    def save_as_tflite_direct(self, model, filename, quantize=False, representative_data=None):
        """Direct Keras model conversion - Keras 3 compatible with REAL data"""
        try:
            # Ensure model is built
            if not model.built:
                dummy_input = tf.zeros([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
                _ = model(dummy_input)
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Use provided representative data or create REAL default
                if representative_data is not None:
                    converter.representative_dataset = representative_data
                else:
                    def real_representative_dataset():
                        from utils import get_data_splits
                        from utils.preprocess import preprocess_for_inference
                        
                        # Use real data instead of random
                        (x_train_raw, _), _, _ = get_data_splits()
                        # num_samples = min(100, len(x_train_raw), params.QUANTIZE_NUM_SAMPLES)
                        num_samples = min(len(x_train_raw), params.QUANTIZE_NUM_SAMPLES)
                        calibration_data = x_train_raw[:num_samples]
                        
                        # Preprocess properly
                        calibration_processed = preprocess_for_inference(calibration_data)
                        
                        # Ensure proper format
                        if calibration_processed.dtype != np.float32:
                            calibration_processed = calibration_processed.astype(np.float32)
                        if calibration_processed.max() > 1.0:
                            calibration_processed = calibration_processed / 255.0
                        
                        if self.debug:
                            print(f"🔧 Direct conversion - Real calibration: {calibration_processed.dtype}")
                        
                        for i in range(len(calibration_processed)):
                            yield [calibration_processed[i:i+1]]
                    
                    converter.representative_dataset = real_representative_dataset
                
                print("🎯 Using dynamic range quantization with REAL data (Keras 3 safe)")
            
            # Convert model
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            return self._save_tflite_file(tflite_model, filename, quantize)
                
        except Exception as e:
            print(f"❌ Direct TFLite conversion failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise

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
            print("Skipping TFLite conversion – model failed verification")
            return None

        if self.debug:
            print(f"New best accuracy: {accuracy:.4f} – converting to TFLite…")

        try:
            # Use enhanced conversion for better results
            q_blob, q_size = self.save_as_tflite_enhanced(
                model,
                params.TFLITE_FILENAME,
                quantize=True,
                representative_data=representative_data,
            )
            
            # If QAT model and no representative data, that's fine - just log it
            if representative_data is None and params.USE_QAT and params.QUANTIZE_MODEL:
                print("🎯 QAT model: Using model's quantization parameters")
            
            # Float version
            self.save_as_tflite_enhanced(
                model,
                params.FLOAT_TFLITE_FILENAME,
                quantize=False,
            )
            return q_size
        except Exception as exc:
            print(f"Failed to save best TFLite model: {exc}")
            # For QAT models, representative data might not be critical
            if params.USE_QAT and params.QUANTIZE_MODEL:
                print("⚠️  QAT model conversion failed, but this might not be critical")
            return None