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

import parameters as params


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


def evaluate_tflite_model(tflite_path, x_test, y_test):
    """Evaluate TFLite model accuracy"""
    print("🧪 Evaluating TFLite model...")
    
    # Use configured number of samples
    x_test_analysis, y_test_analysis = get_analysis_samples(x_test, y_test)
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
        # Prepare input
        input_data = x_test_analysis[i:i+1]
        
        # Convert input based on model requirements
        if input_dtype == np.int8:
            # Convert float [0,1] to int8 [-128, 127]
            input_data = (input_data * 255.0 - 128.0).astype(np.int8)
        elif input_dtype == np.uint8:
            # Convert float [0,1] to uint8 [0, 255]
            input_data = (input_data * 255.0).astype(np.uint8)
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
        
        # Get true class (handle both categorical and sparse)
        if len(y_test_analysis.shape) > 1 and y_test_analysis.shape[1] > 1:
            true_class = np.argmax(y_test_analysis[i])
        else:
            true_class = y_test_analysis[i]
        
        if predicted_class == true_class:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_samples
    print(f"TFLite Model Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
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
        
        # Accuracy comparison
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
            import traceback
            traceback.print_exc()
        return None


def training_diagnostics(model, x_train, y_train, x_val, y_val, debug=False):
    """Run comprehensive training diagnostics"""
    print("\n🔍 Running training diagnostics...")
    
    # Use configured number of samples for diagnostics
    x_train_analysis, y_train_analysis = get_analysis_samples(x_train, y_train)
    x_val_analysis, y_val_analysis = get_analysis_samples(x_val, y_val)
    
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
        test_output = model.predict(x_train_analysis[:1], verbose=0)
        print(f"   Forward pass test: ✓ (output shape: {test_output.shape})")
    except Exception as e:
        print(f"   Forward pass test: ✗ ({e})")
    
    # Check model output range
    if debug:
        sample_outputs = model.predict(x_train_analysis[:10], verbose=0)
        print(f"   Output range: [{sample_outputs.min():.3f}, {sample_outputs.max():.3f}]")
        print(f"   Output sum check: {np.sum(sample_outputs, axis=1)}")


def verify_model_predictions(model, x_sample, y_sample):
    """Verify model predictions match expected format"""
    print("\n✅ Verifying model predictions...")
    
    # Use configured number of samples
    x_sample_analysis, y_sample_analysis = get_analysis_samples(x_sample, y_sample)
    
    predictions = model.predict(x_sample_analysis, verbose=0)
    
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
    predictions = model.predict(x_test_analysis, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    if len(y_test_analysis.shape) > 1:
        true_classes = np.argmax(y_test_analysis, axis=1)
    else:
        true_classes = y_test_analysis
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(params.NB_CLASSES),
                yticklabels=range(params.NB_CLASSES))
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