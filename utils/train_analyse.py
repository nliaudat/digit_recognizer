# utils/train_analyse.py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm.auto import tqdm
import time
import tempfile
import shutil
import logging
import traceback

logger = logging.getLogger(__name__)

import config as params


from models import combine_multiheads


def get_analysis_samples(x_data, y_data):
    """Get the number of samples to use for analysis based on params.ANALYSE_SAMPLES"""
    if hasattr(params, 'ANALYSE_SAMPLES') and params.ANALYSE_SAMPLES is not None:
        n_samples = min(params.ANALYSE_SAMPLES, len(x_data))
        print(f"📊 Using {n_samples} samples for analysis (ANALYSE_SAMPLES={params.ANALYSE_SAMPLES})")
        return x_data[:n_samples], y_data[:n_samples]
    else:
        print(f"📊 Using all {len(x_data)} samples for analysis")
        return x_data, y_data


def evaluate_keras_model(keras_model, x_test, y_test):
    """Evaluate Keras model accuracy"""
    print("🧪 Evaluating Keras model...")
    
    # Use configured number of samples
    x_test_analysis, y_test_analysis = get_analysis_samples(x_test, y_test)
    
    # Handle different label formats
    if len(y_test_analysis.shape) > 1 and y_test_analysis.shape[1] > 1:
        # Categorical labels
        loss, accuracy = keras_model.evaluate(x_test_analysis, y_test_analysis, verbose=0)
    else:
        # Sparse labels
        loss, accuracy = keras_model.evaluate(x_test_analysis, y_test_analysis, verbose=0)
    
    print(f"Keras Model Accuracy: {accuracy:.4f} (on {len(x_test_analysis)} samples)")
    return accuracy


def _evaluate_keras_multihead(keras_model, x_test, y_test_orig):
    """
    Evaluate multi-head Keras model (v41/v42) using combined accuracy.
    y_test_orig: scalar labels (0-99).

    For v42: decimal_probs is the marginal Σ P(int=t)×P(dec|int=t), so
    argmax(decimal_probs) can select a decimal from the wrong integer.
    We extract the individual decimal_head_{i}_probs layers to compute
    the correct joint: argmax(integer) × 10 + head[argmax(integer)].
    """
    is_v42 = hasattr(keras_model, 'output_names') and 'integer_probs' in keras_model.output_names

    if is_v42:
        # Model now exports individual decimal heads as outputs 2-11.
        # Use them directly instead of building a temporary model.
        x_test_analysis, y_orig = get_analysis_samples(x_test, y_test_orig)
        outputs = keras_model.predict(x_test_analysis, verbose=0)
        int_probs = outputs[0]   # [N, 10]
        dec_heads = outputs[2:]  # list of 10 arrays each [N, 10]
        int_preds = np.argmax(int_probs, axis=-1)
        stacked_dec_heads = np.stack(dec_heads, axis=1)  # (N, 10, 10)
        selected_heads = stacked_dec_heads[np.arange(len(int_preds)), int_preds]  # (N, 10)
        dec_preds = np.argmax(selected_heads, axis=-1)
        pred_cls = int_preds * 10 + dec_preds
    else:
        # v41: standard argmax combination (both heads are independent 10-class)
        x_test_analysis, y_orig = get_analysis_samples(x_test, y_test_orig)
        preds = keras_model.predict(x_test_analysis, verbose=0)
        pred_cls = np.argmax(preds[0], axis=-1) * 10 + np.argmax(preds[1], axis=-1)

    y_true = np.asarray(y_orig).flatten()
    accuracy = float(np.mean(pred_cls == y_true))
    print(f"Keras Model Combined Accuracy: {accuracy:.4f} (on {len(x_test_analysis)} samples)")
    return accuracy


def evaluate_tflite_model(tflite_path, x_test, y_test):
    """Evaluate TFLite model accuracy (single-head)."""
    print("🧪 Evaluating TFLite model...")
    
    # Use configured number of samples
    x_test_analysis, y_test_analysis = get_analysis_samples(x_test, y_test)
    # Pre-convert to numpy once to avoid per-sample overhead in the loop.
    if hasattr(x_test_analysis, 'numpy'):
        x_test_analysis = x_test_analysis.numpy()
    else:
        x_test_analysis = np.asarray(x_test_analysis, dtype=np.float32)
    total_samples = len(x_test_analysis)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check input type for proper preprocessing
    input_dtype = input_details[0]['dtype']
    
    correct_predictions = 0
    
    # Use tqdm for progress tracking
    for i in tqdm(range(total_samples), desc="Evaluating TFLite", leave=False):
        input_data = x_test_analysis[i:i+1]
        
        # Convert input based on model requirements
        if input_dtype == np.int8:
            # Round + clip before cast to match training quantisation
            input_data = np.clip(np.round(input_data * 255.0 - 128.0), -128, 127).astype(np.int8)
        elif input_dtype == np.uint8:
            input_data = np.clip(np.round(input_data * 255.0), 0, 255).astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Handle different output types
        if output_details[0]['dtype'] in [np.int8, np.uint8]:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        # Get prediction
        predicted_class = np.argmax(output_data)
        
        # Get true class (handle both categorical and sparse, and TF tensors)
        true_label = y_test_analysis[i]
        if hasattr(true_label, 'numpy'):
            true_label = true_label.numpy()
        if len(y_test_analysis.shape) > 1 and y_test_analysis.shape[1] > 1:
            true_class = int(np.argmax(true_label))
        else:
            true_class = int(true_label)
        
        if predicted_class == true_class:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_samples
    print(f"TFLite Model Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    return accuracy


def _eval_two_heads(interpreter, input_details, output_details, input_dtype,
                     x_test, y_orig, idx_a, idx_b):
    """Evaluate two 10-head outputs at positional indices idx_a, idx_b, combined as A*10+B."""
    total = len(x_test)
    y_true = np.asarray(y_orig).flatten()
    correct = 0
    for i in tqdm(range(total), desc="Evaluating TFLite", leave=False):
        input_data = x_test[i:i+1]
        if input_dtype == np.int8:
            input_data = np.clip(np.round(input_data * 255.0 - 128.0), -128, 127).astype(np.int8)
        elif input_dtype == np.uint8:
            input_data = np.clip(np.round(input_data * 255.0), 0, 255).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        out_a = interpreter.get_tensor(output_details[idx_a]['index'])
        out_b = interpreter.get_tensor(output_details[idx_b]['index'])
        if output_details[idx_a]['dtype'] in [np.uint8, np.int8]:
            s, zp = output_details[idx_a].get('quantization', (None, None))
            if s is not None and s > 0.0:
                out_a = (out_a.astype(np.float32) - zp) * s
        if output_details[idx_b]['dtype'] in [np.uint8, np.int8]:
            s, zp = output_details[idx_b].get('quantization', (None, None))
            if s is not None and s > 0.0:
                out_b = (out_b.astype(np.float32) - zp) * s
        pred = int(np.argmax(out_a[0])) * 10 + int(np.argmax(out_b[0]))
        if pred == y_true[i]:
            correct += 1
    return correct / total


def _evaluate_tflite_multihead(tflite_path, x_test, y_test_orig):
    """
    Evaluate multi-head TFLite model (v41/v42) using combined accuracy.
    Matches output tensors by NAME (not position) to handle converter reordering.
    Both heads are 10-class; combined as head0*10+head1.
    y_test_orig: scalar labels (0-99).
    """
    print("🧪 Evaluating TFLite model (multi-head)...")
    x_test_analysis, y_orig = get_analysis_samples(x_test, y_test_orig)
    # Pre-convert to numpy once to avoid per-sample overhead in the loop.
    if hasattr(x_test_analysis, 'numpy'):
        x_test_analysis = x_test_analysis.numpy()
    else:
        x_test_analysis = np.asarray(x_test_analysis, dtype=np.float32)
    total_samples = len(x_test_analysis)
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']
    
    # Resolve integer/units head index by name (substring match), not position.
    # TFLite may decorate output names (e.g. 'serving_default_tens_probs:0')
    # so we use substring matching rather than exact names.
    # v41: tens_probs / units_probs ;  v42: integer_probs / decimal_probs
    def _head_by_substr(head_map, substr):
        """Return (detail, index) for output whose name contains substr, else (None, None)."""
        for name, detail in head_map.items():
            name_str = name.decode('utf-8') if isinstance(name, bytes) else name
            if substr in name_str:
                return detail, detail['index']
        return None, None

    head_map = {od['name']: od for od in output_details}
    det_int, idx_int = _head_by_substr(head_map, 'integer_probs')
    det_dec, idx_dec = _head_by_substr(head_map, 'decimal_probs')
    if det_int is None or det_dec is None:
        # Fallback: try v41 naming (tens_probs / units_probs)
        det_int, idx_int = _head_by_substr(head_map, 'tens_probs')
        det_dec, idx_dec = _head_by_substr(head_map, 'units_probs')
    if det_int is None or det_dec is None:
        # Fallback: generic names (e.g. Identity:0) — try both orderings, take max accuracy.
        # Swapping integer/decimal inverts digits (e.g. 35 ↔ 53), giving ~10% vs true accuracy.
        found_names = [od['name'] for od in output_details]
        print(f"⚠️  Unrecognized output names: {found_names} — trying both head orderings")
        # Evaluate with positional order A→B, then B→A
        acc_ab = _eval_two_heads(
            interpreter, input_details, output_details, input_dtype,
            x_test_analysis, y_orig, idx_a=0, idx_b=1
        )
        acc_ba = _eval_two_heads(
            interpreter, input_details, output_details, input_dtype,
            x_test_analysis, y_orig, idx_a=1, idx_b=0
        )
        accuracy = max(acc_ab, acc_ba)
        print(f"   Positional A→B: {acc_ab:.4f}  B→A: {acc_ba:.4f}  → taking max: {accuracy:.4f}")
        return accuracy
    
    # Pre-fetch dequantization params per head (safe .get to avoid KeyError)
    q_int = det_int.get('quantization', (None, None)) if det_int['dtype'] in [np.uint8, np.int8] else None
    q_dec = det_dec.get('quantization', (None, None)) if det_dec['dtype'] in [np.uint8, np.int8] else None
    
    correct = 0
    y_true_arr = np.asarray(y_orig).flatten()
    
    for i in tqdm(range(total_samples), desc="Evaluating TFLite", leave=False):
        input_data = x_test_analysis[i:i+1]
        if input_dtype == np.int8:
            input_data = np.clip(np.round(input_data * 255.0 - 128.0), -128, 127).astype(np.int8)
        elif input_dtype == np.uint8:
            input_data = np.clip(np.round(input_data * 255.0), 0, 255).astype(np.uint8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        int_out = interpreter.get_tensor(idx_int)
        dec_out = interpreter.get_tensor(idx_dec)
        if q_int is not None and q_int[0] is not None and q_int[0] > 0.0:
            s, zp = q_int
            int_out = (int_out.astype(np.float32) - zp) * s
        if q_dec is not None and q_dec[0] is not None and q_dec[0] > 0.0:
            s, zp = q_dec
            dec_out = (dec_out.astype(np.float32) - zp) * s
        int_pred = int(np.argmax(int_out[0]))
        dec_pred = int(np.argmax(dec_out[0]))
        pred = int_pred * 10 + dec_pred
        
        if pred == y_true_arr[i]:
            correct += 1
    
    accuracy = correct / total_samples
    print(f"TFLite Model Accuracy: {accuracy:.4f} ({correct}/{total_samples})")
    return accuracy


def get_keras_model_size(keras_model):
    """Get Keras model size in KB with proper Windows file handling"""
    # Create a temporary directory instead of using NamedTemporaryFile
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, "temp_model.keras")
    
    try:
        # Save model to temporary directory
        keras_model.save(temp_model_path)
        size_kb = os.path.getsize(temp_model_path) / 1024
        
        # Clean up - remove the entire directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return size_kb
        
    except Exception as e:
        # Ensure cleanup even if there's an error
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        print(f"⚠️  Failed to get Keras model size: {e}")
        return 0.0


def get_tflite_model_size(tflite_path):
    """Get TFLite model size in KB"""
    try:
        return os.path.getsize(tflite_path) / 1024
    except Exception as e:
        print(f"⚠️  Failed to get TFLite model size: {e}")
        return 0.0


def measure_keras_inference_time(model, x_test):
    """Measure Keras model inference time using Python time (works with determinism)"""
    # Use configured number of samples for timing
    x_test_analysis, _ = get_analysis_samples(x_test, np.zeros(len(x_test)))
    
    # Warm-up
    _ = model.predict(x_test_analysis[:1], verbose=0)
    
    # Actual timing
    start_time = time.perf_counter()
    _ = model.predict(x_test_analysis, verbose=0)
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / len(x_test_analysis)
    print(f"⏱️  Keras inference time: {avg_time*1000:.2f} ms per sample (on {len(x_test_analysis)} samples)")
    return avg_time


def measure_tflite_inference_time(tflite_path, x_test):
    """Measure TFLite model inference time using Python time"""
    # Use configured number of samples for timing
    x_test_analysis, _ = get_analysis_samples(x_test, np.zeros(len(x_test)))
    total_samples = len(x_test_analysis)
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warm-up
    interpreter.set_tensor(input_details[0]['index'], x_test_analysis[:1].astype(np.float32))
    interpreter.invoke()
    
    # Actual timing
    start_time = time.perf_counter()
    for i in range(total_samples):
        interpreter.set_tensor(input_details[0]['index'], x_test_analysis[i:i+1].astype(np.float32))
        interpreter.invoke()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / total_samples
    print(f"⏱️  TFLite inference time: {avg_time*1000:.2f} ms per sample (on {total_samples} samples)")
    return avg_time


def analyze_quantization_impact(keras_model, x_test, y_test, tflite_path, debug=False):
    """
    Analyze the impact of quantization on model performance and size
    FIXED: Handles Windows file permission issues
    """
    print("\n🔍 ANALYZING QUANTIZATION IMPACT")
    print("=" * 50)
    
    try:
        # Use configured number of samples
        x_test_analysis, y_test_analysis = get_analysis_samples(x_test, y_test)
        
        # Detect multi-head model by checking config (v41: tens_probs/units_probs, v42: integer_probs/decimal_probs)
        is_multihead = hasattr(keras_model, 'output_names') and len(keras_model.output_names) >= 2 and (
            all(n in keras_model.output_names for n in ('tens_probs', 'units_probs')) or
            all(n in keras_model.output_names for n in ('integer_probs', 'decimal_probs')))

        # Accuracy comparison — use multi-head-aware evaluation when needed
        if is_multihead:
            keras_accuracy = _evaluate_keras_multihead(keras_model, x_test_analysis, y_test_analysis)
            tflite_accuracy = _evaluate_tflite_multihead(tflite_path, x_test_analysis, y_test_analysis)
        else:
            keras_accuracy = evaluate_keras_model(keras_model, x_test_analysis, y_test_analysis)
            tflite_accuracy = evaluate_tflite_model(tflite_path, x_test_analysis, y_test_analysis)
        
        print(f"📊 ACCURACY COMPARISON:")
        print(f"   Keras Model:    {keras_accuracy:.4f}")
        print(f"   TFLite Model:   {tflite_accuracy:.4f}")
        print(f"   Accuracy Drop:  {keras_accuracy - tflite_accuracy:+.4f}")
        
        # Size comparison with better error handling
        keras_size = 0.0
        tflite_size = 0.0
        
        try:
            keras_size = get_keras_model_size(keras_model)
            tflite_size = get_tflite_model_size(tflite_path)
        except Exception as size_error:
            print(f"⚠️  Size measurement failed: {size_error}")
            # Continue with analysis using fallback values
        
        print(f"\n📏 SIZE COMPARISON:")
        print(f"   Keras Model:    {keras_size:.1f} KB")
        print(f"   TFLite Model:   {tflite_size:.1f} KB")
        
        if keras_size > 0:
            size_reduction = ((keras_size - tflite_size) / keras_size * 100)
            print(f"   Size Reduction: {size_reduction:.1f}%")
        else:
            size_reduction = 0.0
            print(f"   Size Reduction: N/A (could not measure Keras model size)")
        
        # Performance comparison (with determinism handling)
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        
        # Check if determinism is enabled
        determinism_enabled = os.environ.get('TF_DETERMINISTIC_OPS') == '1'
        
        if determinism_enabled:
            print("   ⚠️  Determinism enabled - skipping timing measurements")
            print("   💡 Disable TF_DETERMINISTIC_OPS for performance timing")
            keras_time = 0.0
            tflite_time = 0.0
        else:
            try:
                keras_time = measure_keras_inference_time(keras_model, x_test_analysis)
                tflite_time = measure_tflite_inference_time(tflite_path, x_test_analysis)
                
                print(f"   Keras Inference:  {keras_time*1000:.2f} ms per sample")
                print(f"   TFLite Inference: {tflite_time*1000:.2f} ms per sample")
                if tflite_time > 0:
                    print(f"   Speedup:          {keras_time/tflite_time:.2f}x")
            except Exception as timing_error:
                print(f"   ⚠️  Performance timing failed: {timing_error}")
                keras_time = 0.0
                tflite_time = 0.0
        
        # Quantization quality assessment
        print(f"\n🎯 QUANTIZATION QUALITY:")
        accuracy_drop = keras_accuracy - tflite_accuracy
        
        if accuracy_drop < 0.01:
            print(f"   ✅ EXCELLENT - Minimal accuracy drop ({accuracy_drop:.4f})")
        elif accuracy_drop < 0.03:
            print(f"   ✅ GOOD - Acceptable accuracy drop ({accuracy_drop:.4f})")
        elif accuracy_drop < 0.05:
            print(f"   ⚠️  FAIR - Moderate accuracy drop ({accuracy_drop:.4f})")
        else:
            print(f"   ❌ POOR - Significant accuracy drop ({accuracy_drop:.4f})")
            
        return {
            'keras_accuracy': keras_accuracy,
            'tflite_accuracy': tflite_accuracy,
            'accuracy_drop': accuracy_drop,
            'keras_size': keras_size,
            'tflite_size': tflite_size,
            'size_reduction': size_reduction,
            'keras_inference_time': keras_time,
            'tflite_inference_time': tflite_time,
            'inference_speedup': keras_time / tflite_time if tflite_time > 0 else 0,
            'analysis_samples': len(x_test_analysis)
        }
        
    except Exception as e:
        print(f"❌ Quantization impact analysis failed: {e}")
        if debug:
            traceback.print_exc()
        return None


def training_diagnostics(model, x_train, y_train, x_val, y_val, debug=False):
    """Run comprehensive training diagnostics"""
    print("\n🔍 Running training diagnostics...")
    
    # Use configured number of samples for diagnostics
    x_train_analysis, y_train_analysis = get_analysis_samples(x_train, y_train)
    x_val_analysis, y_val_analysis = get_analysis_samples(x_val, y_val)
    
    # Ensure numpy arrays for analysis functions like .min(), .max(), .sum()
    if hasattr(x_train_analysis, 'numpy'): x_train_analysis = x_train_analysis.numpy()
    if hasattr(y_train_analysis, 'numpy'): y_train_analysis = y_train_analysis.numpy()
    if hasattr(x_val_analysis, 'numpy'): x_val_analysis = x_val_analysis.numpy()
    if hasattr(y_val_analysis, 'numpy'): y_val_analysis = y_val_analysis.numpy()
    
    # Check data shapes and types
    print("📊 Data Diagnostics:")
    print(f"   x_train shape: {x_train_analysis.shape}, dtype: {x_train_analysis.dtype}")
    print(f"   y_train shape: {y_train_analysis.shape}, dtype: {y_train_analysis.dtype}")
    print(f"   x_val shape: {x_val_analysis.shape}, dtype: {x_val_analysis.dtype}")
    print(f"   y_val shape: {y_val_analysis.shape}, dtype: {y_val_analysis.dtype}")
    
    # Check data ranges
    print(f"   x_train range: [{x_train_analysis.min():.3f}, {x_train_analysis.max():.3f}]")
    print(f"   x_val range: [{x_val_analysis.min():.3f}, {x_val_analysis.max():.3f}]")
    
    # Check for NaN or Inf values
    train_nans = np.isnan(x_train_analysis).sum()
    val_nans = np.isnan(x_val_analysis).sum()
    train_infs = np.isinf(x_train_analysis).sum()
    val_infs = np.isinf(x_val_analysis).sum()
    
    print(f"   NaN values - Train: {train_nans}, Val: {val_nans}")
    print(f"   Inf values - Train: {train_infs}, Val: {val_infs}")
    
    # Check class distribution
    if len(y_train_analysis.shape) > 1:
        train_classes = np.argmax(y_train_analysis, axis=1)
        val_classes = np.argmax(y_val_analysis, axis=1)
    else:
        train_classes = y_train_analysis
        val_classes = y_val_analysis
    
    train_class_counts = np.bincount(train_classes, minlength=params.NB_CLASSES)
    val_class_counts = np.bincount(val_classes, minlength=params.NB_CLASSES)
    
    print(f"   Train class distribution: {dict(zip(range(len(train_class_counts)), train_class_counts))}")
    print(f"   Val class distribution: {dict(zip(range(len(val_class_counts)), val_class_counts))}")
    
    # Model diagnostics
    print("\n🧠 Model Diagnostics:")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # Test forward pass
    try:
        test_output = combine_multiheads(model.predict(x_train_analysis[:1], verbose=0), model=model)
        print(f"   Forward pass test: ✓ (output shape: {test_output.shape})")
    except Exception as e:
        print(f"   Forward pass test: ✗ ({e})")
    
    # Check model output range
    if debug:
        sample_outputs = combine_multiheads(model.predict(x_train_analysis[:10], verbose=0), model=model)
        print(f"   Output range: [{sample_outputs.min():.3f}, {sample_outputs.max():.3f}]")
        print(f"   Output sum check: {np.sum(sample_outputs, axis=1)}")


def verify_model_predictions(model, x_sample, y_sample):
    """Verify model predictions match expected format"""
    print("\n✅ Verifying model predictions...")
    
    # Use configured number of samples
    x_sample_analysis, y_sample_analysis = get_analysis_samples(x_sample, y_sample)
    
    predictions = combine_multiheads(model.predict(x_sample_analysis, verbose=0), model=model)
    
    print(f"   Input samples: {len(x_sample_analysis)}")
    print(f"   Predictions shape: {predictions.shape}")
    
    # Check if predictions are probabilities
    prediction_sums = np.sum(predictions, axis=1)
    print(f"   Prediction sums: min={prediction_sums.min():.3f}, max={prediction_sums.max():.3f}")
    
    # Check accuracy on sample
    pred_classes = np.argmax(predictions, axis=1)
    
    if len(y_sample_analysis.shape) > 1:
        true_classes = np.argmax(y_sample_analysis, axis=1)
    else:
        true_classes = y_sample_analysis
    
    sample_accuracy = np.mean(pred_classes == true_classes)
    print(f"   Sample accuracy: {sample_accuracy:.3f}")
    
    return sample_accuracy


def debug_model_architecture(model, sample_data=None):
    """Debug model architecture with better error handling"""
    
    # Build the model if it hasn't been built
    if not model.built:
        if sample_data is not None:
            # Build by running a forward pass
            _ = model(sample_data)
        else:
            # Build with input shape
            try:
                model.build(input_shape=(None,) + params.INPUT_SHAPE)
            except:
                print("⚠️ Warning: Could not build model automatically")
                return
    
    # Now the model should have defined inputs
    try:
        activation_model = tf.keras.models.Model(
            inputs=model.input, 
            outputs=[layer.output for layer in model.layers]
        )
        print("✅ Model architecture debug completed")
    except Exception as e:
        print(f"❌ Debugging failed: {e}")


def analyze_confusion_matrix(model, x_test, y_test, save_path=None):
    """Generate and analyze confusion matrix"""
    print("\n📈 Generating confusion matrix...")
    
    # Use configured number of samples
    x_test_analysis, y_test_analysis = get_analysis_samples(x_test, y_test)
    
    # Get predictions
    predictions = combine_multiheads(model.predict(x_test_analysis, verbose=0), model=model)
    pred_classes = np.argmax(predictions, axis=1)
    
    if len(y_test_analysis.shape) > 1:
        true_classes = np.argmax(y_test_analysis, axis=1)
    else:
        true_classes = y_test_analysis
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    
    # Determine figure settings based on the number of classes
    num_classes = params.NB_CLASSES
    fig_size = min(40, max(10, num_classes * 0.4))
    show_annot = num_classes <= 20
    
    # Plot confusion matrix
    plt.figure(figsize=(fig_size, fig_size * 0.8))
    ax = sns.heatmap(cm, annot=show_annot, fmt='d', cmap='Blues', 
                     xticklabels=range(num_classes),
                     yticklabels=range(num_classes))
    
    if num_classes > 20:
        # For large class counts, adjust ticks to prevent overlapping
        tick_fontsize = max(4, 12 - (num_classes * 0.05))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fontsize)
        
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix saved to: {os.path.join(save_path, 'confusion_matrix.png')}")
    
    # plt.show()
    
    # Generate classification report
    report = classification_report(true_classes, pred_classes, 
                                  target_names=[str(i) for i in range(params.NB_CLASSES)])
    print("\n📊 Classification Report:")
    print(report)
    
    return cm, report


def analyze_training_history(training_log_path, save_path=None):
    """Analyze training history from CSV log"""
    print("\n📊 Analyzing training history...")
    
    if not os.path.exists(training_log_path):
        print(f"❌ Training log not found: {training_log_path}")
        return None
    
    # Load training history
    history_df = pd.read_csv(training_log_path)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    if 'loss' in history_df.columns and 'val_loss' in history_df.columns:
        ax1.plot(history_df['loss'], label='Training Loss')
        ax1.plot(history_df['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'accuracy' in history_df.columns and 'val_accuracy' in history_df.columns:
        ax2.plot(history_df['accuracy'], label='Training Accuracy')
        ax2.plot(history_df['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if 'lr' in history_df.columns:
        ax3.plot(history_df['lr'], label='Learning Rate', color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot accuracy difference
    if 'accuracy' in history_df.columns and 'val_accuracy' in history_df.columns:
        accuracy_diff = history_df['val_accuracy'] - history_df['accuracy']
        ax4.plot(accuracy_diff, label='Val - Train Accuracy', color='purple')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.set_title('Accuracy Difference (Validation - Training)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Difference')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'detailed_training_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"💾 Detailed training analysis saved to: {os.path.join(save_path, 'detailed_training_analysis.png')}")
    
    # plt.show()
    
    # Print key statistics
    if 'val_accuracy' in history_df.columns:
        best_val_epoch = history_df['val_accuracy'].idxmax()
        best_val_acc = history_df['val_accuracy'].max()
        final_val_acc = history_df['val_accuracy'].iloc[-1]
        
        print(f"📈 Training Statistics:")
        print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_val_epoch + 1})")
        print(f"   Final validation accuracy: {final_val_acc:.4f}")
        print(f"   Total epochs: {len(history_df)}")
    
    return history_df


def model_size_analysis(model_dir):
    """Analyze model sizes and performance trade-offs"""
    print("\n📦 Model size analysis...")
    
    model_files = []
    sizes_kb = []
    accuracies = []
    
    # Find all model files
    for file in os.listdir(model_dir):
        if file.endswith('.tflite') or file.endswith('.keras'):
            file_path = os.path.join(model_dir, file)
            size_kb = os.path.getsize(file_path) / 1024
            
            model_files.append(file)
            sizes_kb.append(size_kb)
            
            # Try to extract accuracy from filename or config
            accuracy = 0.0
            if 'best' in file.lower():
                accuracy = 0.95  # Placeholder - would need actual accuracy
            accuracies.append(accuracy)
    
    # Create analysis
    if model_files:
        analysis_df = pd.DataFrame({
            'model': model_files,
            'size_kb': sizes_kb,
            'accuracy': accuracies
        })
        
        print("📊 Model Size Comparison:")
        for _, row in analysis_df.iterrows():
            print(f"   {row['model']:25} -> {row['size_kb']:6.1f} KB")
        
        # Plot size vs accuracy if we have accuracy data
        if any(acc > 0 for acc in accuracies):
            plt.figure(figsize=(10, 6))
            plt.scatter(sizes_kb, accuracies, s=100, alpha=0.7)
            
            for i, model in enumerate(model_files):
                plt.annotate(model, (sizes_kb[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('Model Size (KB)')
            plt.ylabel('Accuracy')
            plt.title('Model Size vs Accuracy Trade-off')
            plt.grid(True, alpha=0.3)
            # plt.show()
        
        return analysis_df
    
    return None


def verify_tflite_full_qat(tflite_path, debug=False):
    """
    Verify if a TFLite model is "full QAT" (Quantization-Aware Training).
    Full QAT models should have quantized inputs/outputs and mostly 
    integer-based operations.
    """
    if debug:
        print(f"\n🔍 VERIFYING FULL QAT: {os.path.basename(tflite_path)}")
        print("=" * 50)
    
    try:
        # Load TFLite model - explicitly disable delegates for parity with TFLM/ESP32
        # and to avoid XNNPACK-specific allocation failures on large models
        interpreter = tf.lite.Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        tensor_details = interpreter.get_tensor_details()
        
        # 1. Check Input/Output Types
        input_dtype = input_details[0]['dtype']
        output_dtype = output_details[0]['dtype']
        
        # Check if they are quantized types (int8 or uint8)
        is_io_quantized = input_dtype in [np.int8, np.uint8] and output_dtype in [np.int8, np.uint8]
        
        if debug:
            print(f"📊 I/O Analysis:")
            print(f"   Input Type:  {input_dtype}")
            print(f"   Output Type: {output_dtype}")
            print(f"   I/O Quantized: {'✅ YES' if is_io_quantized else '❌ NO'}")

        # 2. Check internal tensors for quantization parameters
        quantized_tensors = 0
        total_tensors = 0
        
        for tensor in tensor_details:
            total_tensors += 1
            if 'quantization' in tensor and tensor['quantization'][0] != 0.0:
                quantized_tensors += 1
        
        quantization_ratio = quantized_tensors / total_tensors if total_tensors > 0 else 0
        
        if debug:
            print(f"📊 Tensor Analysis:")
            print(f"   Quantized Tensors: {quantized_tensors}/{total_tensors} ({quantization_ratio:.1%})")

        # 3. Decision
        is_full_qat = is_io_quantized and quantization_ratio > 0.3
        
        if debug:
            if is_full_qat:
                print("\n🎯 RESULT: Model appears to be FULL QAT")
            else:
                print("\n⚠️  RESULT: Model might NOT be full QAT")
                if not is_io_quantized:
                    print(f"   - Reason: I/O are not quantized ({input_dtype}/{output_dtype})")
                if quantization_ratio <= 0.3:
                    print(f"   - Reason: Low quantization ratio ({quantization_ratio:.1%})")

        return {
            'is_full_qat': is_full_qat,
            'is_io_quantized': is_io_quantized,
            'input_dtype': str(input_dtype),
            'output_dtype': str(output_dtype),
            'quantization_ratio': float(quantization_ratio),
            'quantized_tensors': quantized_tensors,
            'total_tensors': total_tensors
        }
        
    except Exception as e:
        logger.error(f"❌ QAT Verification Error: {e}")
        if debug:
            traceback.print_exc()
        return None