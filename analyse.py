import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
from tqdm.auto import tqdm
import logging
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from contextlib import contextmanager

# Import shared modules
from models import create_model, compile_model, model_summary
from utils import get_data_splits, preprocess_images
import parameters as params


def evaluate_tflite_model(tflite_path, x_test, y_test):
    """Evaluate TFLite model accuracy"""
    print("🧪 Evaluating TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check input type for proper preprocessing
    input_dtype = input_details[0]['dtype']
    
    correct_predictions = 0
    total_samples = len(x_test)
    
    # Use tqdm for progress tracking
    for i in tqdm(range(total_samples), desc="Evaluating TFLite", leave=False):
        # Prepare input
        input_data = x_test[i:i+1]
        
        # Convert input based on model requirements
        if input_dtype == np.int8:
            # Scale to int8 range
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.int8)
        elif input_dtype == np.uint8:
            # Scale to uint8 range
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.uint8)
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
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            true_class = np.argmax(y_test[i])
        else:
            true_class = y_test[i]
        
        if predicted_class == true_class:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_samples
    print(f"📊 TFLite Model Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    return accuracy

def analyze_quantization_impact(keras_model, x_test, y_test, tflite_path):
    """Analyze the impact of quantization on model performance"""
    print("\n🔍 Analyzing quantization impact...")
    
    # Get Keras model predictions
    keras_predictions = keras_model.predict(x_test, verbose=0)
    keras_pred_classes = np.argmax(keras_predictions, axis=1)
    
    # Get true classes
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        true_classes = np.argmax(y_test, axis=1)
    else:
        true_classes = y_test
    
    # Get TFLite model predictions
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    tflite_predictions = []
    
    for i in range(len(x_test)):
        input_data = x_test[i:i+1]
        
        # Handle quantization
        input_dtype = input_details[0]['dtype']
        if input_dtype in [np.int8, np.uint8]:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(input_dtype)
        else:
            input_data = input_data.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize if needed
        if output_details[0]['dtype'] in [np.int8, np.uint8]:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        tflite_predictions.append(output_data[0])
    
    tflite_predictions = np.array(tflite_predictions)
    tflite_pred_classes = np.argmax(tflite_predictions, axis=1)
    
    # Calculate metrics
    keras_accuracy = np.mean(keras_pred_classes == true_classes)
    tflite_accuracy = np.mean(tflite_pred_classes == true_classes)
    
    # Calculate prediction agreement
    agreement = np.mean(keras_pred_classes == tflite_pred_classes)
    
    # Calculate confidence differences
    keras_confidences = np.max(keras_predictions, axis=1)
    tflite_confidences = np.max(tflite_predictions, axis=1)
    confidence_diff = np.mean(np.abs(keras_confidences - tflite_confidences))
    
    print(f"📊 Quantization Analysis:")
    print(f"   Keras Model Accuracy: {keras_accuracy:.4f}")
    print(f"   TFLite Model Accuracy: {tflite_accuracy:.4f}")
    print(f"   Accuracy Drop: {keras_accuracy - tflite_accuracy:.4f}")
    print(f"   Prediction Agreement: {agreement:.4f}")
    print(f"   Avg Confidence Difference: {confidence_diff:.4f}")
    
    # Find disagreements
    disagreements = keras_pred_classes != tflite_pred_classes
    if np.any(disagreements):
        disagreement_indices = np.where(disagreements)[0]
        print(f"   Disagreements: {len(disagreement_indices)} samples")
        
        # Analyze first few disagreements
        print(f"   Sample disagreements:")
        for i in disagreement_indices[:5]:
            keras_conf = keras_confidences[i]
            tflite_conf = tflite_confidences[i]
            true_class = true_classes[i]
            keras_class = keras_pred_classes[i]
            tflite_class = tflite_pred_classes[i]
            
            print(f"     Sample {i}: True={true_class}, Keras={keras_class}({keras_conf:.3f}), "
                  f"TFLite={tflite_class}({tflite_conf:.3f})")
    
    return {
        'keras_accuracy': keras_accuracy,
        'tflite_accuracy': tflite_accuracy,
        'accuracy_drop': keras_accuracy - tflite_accuracy,
        'agreement': agreement,
        'confidence_diff': confidence_diff
    }

def debug_tflite_model(tflite_path, sample_input=None):
    """Debug TFLite model details and test with sample input"""
    print("\n🔧 Debugging TFLite model...")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("📋 TFLite Model Details:")
        print(f"   Input details: {input_details}")
        print(f"   Output details: {output_details}")
        
        # Test with sample input if provided
        if sample_input is not None:
            print("\n🧪 Testing with sample input...")
            
            input_data = sample_input
            input_dtype = input_details[0]['dtype']
            
            # Handle quantization
            if input_dtype in [np.int8, np.uint8]:
                input_scale, input_zero_point = input_details[0]['quantization']
                input_data = input_data / input_scale + input_zero_point
                input_data = input_data.astype(input_dtype)
                print(f"   Input quantization: scale={input_scale}, zero_point={input_zero_point}")
            else:
                input_data = input_data.astype(np.float32)
            
            print(f"   Input shape: {input_data.shape}, dtype: {input_data.dtype}")
            print(f"   Input range: [{input_data.min():.3f}, {input_data.max():.3f}]")
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Handle output quantization
            if output_details[0]['dtype'] in [np.int8, np.uint8]:
                output_scale, output_zero_point = output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
                print(f"   Output quantization: scale={output_scale}, zero_point={output_zero_point}")
            
            print(f"   Output shape: {output_data.shape}")
            print(f"   Output: {output_data.flatten()}")
            print(f"   Predicted class: {np.argmax(output_data)}")
            
            return output_data
        
    except Exception as e:
        print(f"❌ TFLite debug failed: {e}")
    
    return None

def training_diagnostics(model, x_train, y_train, x_val, y_val, debug=False):
    """Run comprehensive training diagnostics"""
    print("\n🔍 Running training diagnostics...")
    
    # Check data shapes and types
    print("📊 Data Diagnostics:")
    print(f"   x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
    print(f"   y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"   x_val shape: {x_val.shape}, dtype: {x_val.dtype}")
    print(f"   y_val shape: {y_val.shape}, dtype: {y_val.dtype}")
    
    # Check data ranges
    print(f"   x_train range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"   x_val range: [{x_val.min():.3f}, {x_val.max():.3f}]")
    
    # Check for NaN or Inf values
    train_nans = np.isnan(x_train).sum()
    val_nans = np.isnan(x_val).sum()
    train_infs = np.isinf(x_train).sum()
    val_infs = np.isinf(x_val).sum()
    
    print(f"   NaN values - Train: {train_nans}, Val: {val_nans}")
    print(f"   Inf values - Train: {train_infs}, Val: {val_infs}")
    
    # Check class distribution
    if len(y_train.shape) > 1:
        train_classes = np.argmax(y_train, axis=1)
        val_classes = np.argmax(y_val, axis=1)
    else:
        train_classes = y_train
        val_classes = y_val
    
    train_class_counts = np.bincount(train_classes)
    val_class_counts = np.bincount(val_classes)
    
    print(f"   Train class distribution: {dict(zip(range(len(train_class_counts)), train_class_counts))}")
    print(f"   Val class distribution: {dict(zip(range(len(val_class_counts)), val_class_counts))}")
    
    # Model diagnostics
    print("\n🧠 Model Diagnostics:")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # Test forward pass
    try:
        test_output = model.predict(x_train[:1], verbose=0)
        print(f"   Forward pass test: ✓ (output shape: {test_output.shape})")
    except Exception as e:
        print(f"   Forward pass test: ✗ ({e})")
    
    # Check model output range
    if debug:
        sample_outputs = model.predict(x_train[:10], verbose=0)
        print(f"   Output range: [{sample_outputs.min():.3f}, {sample_outputs.max():.3f}]")
        print(f"   Output sum check: {np.sum(sample_outputs, axis=1)}")

def verify_model_predictions(model, x_sample, y_sample):
    """Verify model predictions match expected format"""
    print("\n✅ Verifying model predictions...")
    
    predictions = model.predict(x_sample, verbose=0)
    
    print(f"   Input samples: {len(x_sample)}")
    print(f"   Predictions shape: {predictions.shape}")
    
    # Check if predictions are probabilities
    prediction_sums = np.sum(predictions, axis=1)
    print(f"   Prediction sums: min={prediction_sums.min():.3f}, max={prediction_sums.max():.3f}")
    
    # Check accuracy on sample
    pred_classes = np.argmax(predictions, axis=1)
    
    if len(y_sample.shape) > 1:
        true_classes = np.argmax(y_sample, axis=1)
    else:
        true_classes = y_sample
    
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
        # Rest of your debugging code...
    except Exception as e:
        print(f"❌ Debugging failed: {e}")

def analyze_confusion_matrix(model, x_test, y_test, save_path=None):
    """Generate and analyze confusion matrix"""
    print("\n📈 Generating confusion matrix...")
    
    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    if len(y_test.shape) > 1:
        true_classes = np.argmax(y_test, axis=1)
    else:
        true_classes = y_test
    
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
    
    plt.show()
    
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
    
    plt.show()
    
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
        if file.endswith('.tflite') or file.endswith('.h5'):
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
            plt.show()
        
        return analysis_df
    
    return None

def comprehensive_model_analysis(model_path, x_test, y_test, output_dir):
    """Run comprehensive analysis on a trained model"""
    print("🔍 Running comprehensive model analysis...")
    
    # Load model
    if model_path.endswith('.tflite'):
        model = tf.keras.models.load_model(model_path)
    else:
        # For TFLite models, we need to use the interpreter
        model = None
    
    # Run various analyses
    if model:
        # Confusion matrix
        analyze_confusion_matrix(model, x_test, y_test, output_dir)
        
        # Training history analysis if available
        training_log_path = os.path.join(output_dir, 'training_log.csv')
        if os.path.exists(training_log_path):
            analyze_training_history(training_log_path, output_dir)
    
    # Model size analysis
    model_size_analysis(output_dir)
    
    print("✅ Comprehensive analysis completed!")

def main():
    """Main analysis function - can be used for post-training analysis"""
    parser = argparse.ArgumentParser(description='Model Analysis')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models and logs')
    parser.add_argument('--analyze_all', action='store_true',
                       help='Run comprehensive analysis')
    parser.add_argument('--confusion_matrix', action='store_true',
                       help='Generate confusion matrix')
    parser.add_argument('--training_analysis', action='store_true',
                       help='Analyze training history')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        return
    
    # Load test data
    print("📊 Loading test data...")
    _, _, (x_test, y_test) = get_data_splits()
    x_test = preprocess_images(x_test)
    
    # Convert labels if needed
    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test, params.NB_CLASSES)
    
    # Run requested analyses
    if args.analyze_all or args.confusion_matrix:
        # Find the best model
        model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.tflite')]
        if model_files:
            best_model_path = os.path.join(args.model_dir, model_files[0])
            model = tf.keras.models.load_model(best_model_path)
            analyze_confusion_matrix(model, x_test, y_test, args.model_dir)
    
    if args.analyze_all or args.training_analysis:
        training_log_path = os.path.join(args.model_dir, 'training_log.csv')
        analyze_training_history(training_log_path, args.model_dir)
    
    if args.analyze_all:
        comprehensive_model_analysis(args.model_dir, x_test, y_test, args.model_dir)

if __name__ == "__main__":
    main()
    
    
# py analyse.py --model_dir exported_models/digit_recognizer_v4_10cls_QAT_QUANT_GRAY