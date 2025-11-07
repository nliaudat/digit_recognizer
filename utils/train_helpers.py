# utils/train_helpers.py
import os
import json
import tensorflow as tf
from datetime import datetime
import parameters as params

def print_training_summary(model, x_train, x_val, x_test, debug=False):
    """Print comprehensive training summary"""
    print("\n" + "="*60)
    print("🎯 TRAINING SUMMARY")
    print("="*60)
    
    print(f"📊 Data Shapes:")
    print(f"   Training data:   {x_train.shape}")
    print(f"   Validation data: {x_val.shape}")
    print(f"   Test data:       {x_test.shape}")
    
    print(f"\n🔧 Model Architecture: {params.MODEL_ARCHITECTURE}")
    print(f"   Input shape: {params.INPUT_SHAPE}")
    print(f"   Number of classes: {params.NB_CLASSES}")
    print(f"   Total parameters: {model.count_params():,}")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {params.EPOCHS}")
    print(f"   Batch size: {params.BATCH_SIZE}")
    print(f"   Learning rate: {params.LEARNING_RATE}")
    print(f"   Quantization: {params.QUANTIZE_MODEL}")
    print(f"   QAT: {params.USE_QAT}")
    print(f"   ESP-DL: {params.ESP_DL_QUANTIZE}")
    
    print(f"\n🎯 Quantization Configuration:")
    print(f"   Quantization: {params.QUANTIZE_MODEL}")
    print(f"   QAT: {params.USE_QAT}")
    print(f"   ESP-DL: {params.ESP_DL_QUANTIZE}")
    
    if params.USE_QAT and params.QUANTIZE_MODEL:
        print(f"   QAT Data Format: UINT8 [0, 255]")
        print(f"   Training matches inference format: ✅")
    elif params.QUANTIZE_MODEL:
        print(f"   PTQ Data Format: Train=Float32 [0,1], Infer=UINT8 [0,255]")
    else:
        print(f"   Data Format: Float32 [0, 1]")
    
    if debug:
        print(f"\n🔍 Debug Info:")
        print(f"   Model layers: {len(model.layers)}")
        print(f"   Model built: {model.built}")

def save_model_summary_to_file(model, training_dir):
    """Save model summary to file with proper encoding"""
    summary_path = os.path.join(training_dir, "model_summary.txt")
    
    # Method 1: Try with UTF-8 encoding first
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Save string representation
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"💾 Model summary saved to: {summary_path}")
        return
    except Exception as e:
        print(f"⚠️  UTF-8 encoding failed: {e}")
    
    # Method 2: Try to capture summary as string and clean it
    try:
        # Capture summary as string
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        
        # Clean any problematic characters
        cleaned_lines = []
        for line in summary_lines:
            # Remove or replace problematic Unicode characters
            cleaned_line = line.replace('─', '-').replace('┌', '+').replace('┐', '+').replace('└', '+').replace('┘', '+').replace('├', '+').replace('┤', '+').replace('│', '|')
            cleaned_lines.append(cleaned_line)
        
        # Write cleaned summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        print(f"💾 Model summary (cleaned) saved to: {summary_path}")
        return
    except Exception as e:
        print(f"⚠️  Cleaned summary failed: {e}")
    
    # Method 3: Fallback - just save basic model info
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {params.MODEL_ARCHITECTURE}\n")
            f.write(f"Input Shape: {params.INPUT_SHAPE}\n")
            f.write(f"Number of Classes: {params.NB_CLASSES}\n")
            f.write(f"Total Parameters: {model.count_params():,}\n")
            f.write(f"Number of Layers: {len(model.layers)}\n")
            f.write("\nLayer Information:\n")
            for i, layer in enumerate(model.layers):
                f.write(f"Layer {i}: {layer.name} - {type(layer).__name__}\n")
                f.write(f"  Output Shape: {layer.output_shape}\n")
                f.write(f"  Parameters: {layer.count_params()}\n")
        print(f"💾 Basic model info saved to: {summary_path}")
    except Exception as e:
        print(f"❌ Failed to save model summary: {e}")

def save_training_config(training_dir, tflite_size, keras_size, tflite_manager, 
                        test_accuracy, tflite_accuracy, training_time, debug, model=None):
    """Save training configuration and results to file"""
    config_path = os.path.join(training_dir, "training_config.txt")
    
    config_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_duration': str(training_time),
        'model_architecture': params.MODEL_ARCHITECTURE,
        'input_shape': params.INPUT_SHAPE,
        'num_classes': params.NB_CLASSES,
        'epochs': params.EPOCHS,
        'batch_size': params.BATCH_SIZE,
        'learning_rate': params.LEARNING_RATE,
        'quantization_enabled': params.QUANTIZE_MODEL,
        'qat_enabled': params.USE_QAT,
        'esp_dl_enabled': params.ESP_DL_QUANTIZE,
        'data_augmentation': params.USE_DATA_AUGMENTATION,
        'grayscale': params.USE_GRAYSCALE,
        'best_validation_accuracy': float(tflite_manager.best_accuracy),
        'test_accuracy_keras': float(test_accuracy),
        'test_accuracy_tflite': float(tflite_accuracy),
        'model_size_keras_kb': float(keras_size),
        'model_size_tflite_kb': float(tflite_size),
        'total_parameters': model.count_params() if model else 0
    }
    
    # Calculate size reduction if quantization is enabled
    if params.QUANTIZE_MODEL and keras_size > 0:
        size_reduction = ((keras_size - tflite_size) / keras_size) * 100
        config_data['size_reduction_percent'] = float(size_reduction)
    
    # Write to file with UTF-8 encoding
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("TRAINING CONFIGURATION AND RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📅 Training Information:\n")
            f.write(f"   Timestamp: {config_data['timestamp']}\n")
            f.write(f"   Duration: {config_data['training_duration']}\n\n")
            
            f.write("🔧 Model Configuration:\n")
            f.write(f"   Architecture: {config_data['model_architecture']}\n")
            f.write(f"   Input shape: {config_data['input_shape']}\n")
            f.write(f"   Number of classes: {config_data['num_classes']}\n")
            f.write(f"   Total parameters: {config_data['total_parameters']:,}\n\n")
            
            f.write("⚙️  Training Parameters:\n")
            f.write(f"   Epochs: {config_data['epochs']}\n")
            f.write(f"   Batch size: {config_data['batch_size']}\n")
            f.write(f"   Learning rate: {config_data['learning_rate']}\n")
            f.write(f"   Data augmentation: {config_data['data_augmentation']}\n")
            f.write(f"   Grayscale: {config_data['grayscale']}\n\n")
            
            f.write("🎯 Quantization Settings:\n")
            f.write(f"   Quantization enabled: {config_data['quantization_enabled']}\n")
            f.write(f"   QAT enabled: {config_data['qat_enabled']}\n")
            f.write(f"   ESP-DL enabled: {config_data['esp_dl_enabled']}\n\n")
            
            f.write("📊 Results:\n")
            f.write(f"   Best validation accuracy: {config_data['best_validation_accuracy']:.4f}\n")
            f.write(f"   Test accuracy (Keras): {config_data['test_accuracy_keras']:.4f}\n")
            f.write(f"   Test accuracy (TFLite): {config_data['test_accuracy_tflite']:.4f}\n")
            f.write(f"   Keras model size: {config_data['model_size_keras_kb']:.1f} KB\n")
            f.write(f"   TFLite model size: {config_data['model_size_tflite_kb']:.1f} KB\n")
            
            if 'size_reduction_percent' in config_data:
                f.write(f"   Size reduction: {config_data['size_reduction_percent']:.1f}%\n")
        
        print(f"💾 Training configuration saved to: {config_path}")
        
        # Also save as JSON for programmatic access
        json_path = os.path.join(training_dir, "training_config.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
    except UnicodeEncodeError:
        # Fallback: write without emojis
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("TRAINING CONFIGURATION AND RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Training Information:\n")
            f.write(f"   Timestamp: {config_data['timestamp']}\n")
            f.write(f"   Duration: {config_data['training_duration']}\n\n")
            
            f.write("Model Configuration:\n")
            f.write(f"   Architecture: {config_data['model_architecture']}\n")
            f.write(f"   Input shape: {config_data['input_shape']}\n")
            f.write(f"   Number of classes: {config_data['num_classes']}\n")
            f.write(f"   Total parameters: {config_data['total_parameters']:,}\n\n")
            
            f.write("Training Parameters:\n")
            f.write(f"   Epochs: {config_data['epochs']}\n")
            f.write(f"   Batch size: {config_data['batch_size']}\n")
            f.write(f"   Learning rate: {config_data['learning_rate']}\n")
            f.write(f"   Data augmentation: {config_data['data_augmentation']}\n")
            f.write(f"   Grayscale: {config_data['grayscale']}\n\n")
            
            f.write("Quantization Settings:\n")
            f.write(f"   Quantization enabled: {config_data['quantization_enabled']}\n")
            f.write(f"   QAT enabled: {config_data['qat_enabled']}\n")
            f.write(f"   ESP-DL enabled: {config_data['esp_dl_enabled']}\n\n")
            
            f.write("Results:\n")
            f.write(f"   Best validation accuracy: {config_data['best_validation_accuracy']:.4f}\n")
            f.write(f"   Test accuracy (Keras): {config_data['test_accuracy_keras']:.4f}\n")
            f.write(f"   Test accuracy (TFLite): {config_data['test_accuracy_tflite']:.4f}\n")
            f.write(f"   Keras model size: {config_data['model_size_keras_kb']:.1f} KB\n")
            f.write(f"   TFLite model size: {config_data['model_size_tflite_kb']:.1f} KB\n")
            
            if 'size_reduction_percent' in config_data:
                f.write(f"   Size reduction: {config_data['size_reduction_percent']:.1f}%\n")
        
        print(f"💾 Training configuration (no emojis) saved to: {config_path}")
    
    return config_data

def save_training_csv(training_dir, history):
    """Save training history to CSV file"""
    import pandas as pd
    
    if history and hasattr(history, 'history'):
        csv_path = os.path.join(training_dir, "training_log.csv")
        
        # Create DataFrame from history
        history_df = pd.DataFrame(history.history)
        history_df['epoch'] = range(1, len(history_df) + 1)
        
        # Reorder columns to have epoch first
        cols = ['epoch'] + [col for col in history_df.columns if col != 'epoch']
        history_df = history_df[cols]
        
        history_df.to_csv(csv_path, index=False)
        print(f"💾 Training log saved to: {csv_path}")
        
        return history_df
    return None