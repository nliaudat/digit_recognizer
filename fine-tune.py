# fine-tune.py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
from tqdm.auto import tqdm
import logging
import shutil
from pathlib import Path
import cv2
import glob
import re

# Import your existing modules
from train import (
    set_all_seeds, setup_tensorflow_logging, 
    suppress_all_output, TFLiteModelManager, TrainingMonitor,
    create_callbacks, create_representative_dataset, setup_gpu,
    print_training_summary
)
from utils import get_data_splits, preprocess_images
import parameters as params

class FineTuneManager:
    """Manager for fine-tuning pre-trained models"""
    
    def __init__(self, model_dir=None, debug=False):
        self.debug = debug
        self.model_dir = model_dir or params.OUTPUT_DIR
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for fine-tuning"""
        setup_tensorflow_logging(self.debug)
        if self.debug:
            print("🔧 Fine-tuning debug mode enabled")

    def reconstruct_model_from_savedmodel(self, savedmodel_path):
        """Reconstruct a trainable model from SavedModel directory"""
        try:
            # Method 1: Try to find a .keras file in the same directory
            directory = os.path.dirname(savedmodel_path) if os.path.isdir(savedmodel_path) else savedmodel_path
            keras_files = [f for f in os.listdir(directory) if f.endswith('.keras')]
            
            if keras_files:
                # Load the .keras file instead
                keras_path = os.path.join(directory, keras_files[0])
                print(f"🔍 Found .keras file: {keras_files[0]}, loading that instead")
                return tf.keras.models.load_model(keras_path)
            
            # Method 2: Create a new model with same architecture and load weights
            print("🔧 Reconstructing model architecture and loading weights...")
            
            # Create a new instance of the same model architecture
            from models import create_model
            model = create_model()
            
            # Try to load weights if they exist
            weights_path = os.path.join(savedmodel_path, "variables", "variables")
            if os.path.exists(weights_path + ".index"):
                model.load_weights(weights_path)
                print("✅ Loaded weights from SavedModel")
            else:
                print("⚠️  No weights found in SavedModel, using initialized weights")
            
            return model
            
        except Exception as e:
            print(f"❌ Failed to reconstruct from SavedModel: {e}")
            raise ValueError("Could not load SavedModel. Please use .keras format for fine-tuning.")

    def find_best_trainable_checkpoint(self):
        """Find the best trainable checkpoint - prioritizing .keras format"""
        if not os.path.exists(self.model_dir):
            return None
        
        # Look for .keras files first (Keras 3 format)
        keras_files = [f for f in os.listdir(self.model_dir) if f.endswith('.keras')]
        
        if keras_files:
            # Prioritize specific files in order
            preferred_files = [
                "best_model.keras",
                "final_model.keras"
            ]
            
            for preferred_file in preferred_files:
                if preferred_file in keras_files:
                    return os.path.join(self.model_dir, preferred_file)
            
            # Look for checkpoint files with highest accuracy
            checkpoint_files = [f for f in keras_files if f.startswith('checkpoint_epoch_')]
            if checkpoint_files:
                # Sort by accuracy (extract accuracy from filename)
                def extract_accuracy(filename):
                    try:
                        # Extract accuracy from filename like: checkpoint_epoch_050_acc_0.9500_143022.keras
                        acc_part = filename.split('_acc_')[1].split('_')[0]
                        return float(acc_part)
                    except:
                        return 0.0
                
                best_checkpoint = max(checkpoint_files, key=extract_accuracy)
                return os.path.join(self.model_dir, best_checkpoint)
            
            # Return any .keras file as fallback
            return os.path.join(self.model_dir, keras_files[0])
        
        # Fallback to H5 files (legacy)
        h5_files = [f for f in os.listdir(self.model_dir) if f.endswith('.h5')]
        if h5_files:
            print("⚠️  Using legacy H5 format - consider retraining with .keras format")
            # Similar logic as above for H5 files
            preferred_files = ["best_model.h5", "final_model.h5"]
            for preferred_file in preferred_files:
                if preferred_file in h5_files:
                    return os.path.join(self.model_dir, preferred_file)
            
            checkpoint_files = [f for f in h5_files if f.startswith('checkpoint_epoch_')]
            if checkpoint_files:
                best_checkpoint = max(checkpoint_files, key=lambda x: float(x.split('_acc_')[1].split('_')[0]))
                return os.path.join(self.model_dir, best_checkpoint)
            
            return os.path.join(self.model_dir, h5_files[0])
        
        # Look for SavedModel directories as last resort
        savedmodel_dirs = [d for d in os.listdir(self.model_dir) 
                          if os.path.isdir(os.path.join(self.model_dir, d)) and 'savedmodel' in d]
        if savedmodel_dirs:
            print("⚠️  Found SavedModel directories - attempting to reconstruct model...")
            # Return the most recent SavedModel directory
            savedmodel_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_dir, x)), reverse=True)
            return os.path.join(self.model_dir, savedmodel_dirs[0])
        
        return None
    
    def load_pretrained_model(self, model_path=None):
        """Load a pre-trained model for fine-tuning - Keras 3 compatible"""
        if model_path is None:
            # Find the best trainable checkpoint
            model_path = self.find_best_trainable_checkpoint()
            if model_path is None:
                raise ValueError("No trainable checkpoints found. Please train a model first with checkpoint saving enabled.")
        
        if self.debug:
            print(f"📁 Loading model from: {model_path}")
        
        try:
            # ✅ UPDATED: Keras 3 compatible model loading
            if model_path.endswith('.keras'):
                # Load .keras format (recommended for Keras 3)
                model = tf.keras.models.load_model(model_path)
                print("✅ Loaded .keras model")
            elif model_path.endswith('.h5'):
                # Load H5 format (legacy support)
                model = tf.keras.models.load_model(model_path)
                print("✅ Loaded H5 model")
            elif os.path.isdir(model_path):
                # ✅ UPDATED: For SavedModel directories, use TFSMLayer for inference or load weights
                print("⚠️  SavedModel format detected - attempting to reconstruct model...")
                model = self.reconstruct_model_from_savedmodel(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            # Ensure model is built
            if not model.built:
                print("🔧 Building model...")
                model.build(input_shape=(None,) + params.INPUT_SHAPE)
            
            # Verify the model can do a forward pass
            test_input = tf.random.normal([1] + list(params.INPUT_SHAPE))
            test_output = model(test_input)
            print(f"✅ Model verified - input: {test_input.shape}, output: {test_output.shape}")
            
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def load_images_from_folder(self, folder_path, target_shape=None):
        """Load all images from a folder with their paths and labels"""
        images = []
        labels = []
        image_paths = []
        
        supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        
        for format in supported_formats:
            for img_path in glob.glob(os.path.join(folder_path, format)):
                try:
                    # Load image
                    if params.USE_GRAYSCALE:
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    if image is None:
                        continue
                    
                    # Resize if target shape provided
                    if target_shape is not None:
                        image = cv2.resize(image, (target_shape[1], target_shape[0]))
                    
                    # Normalize to [0, 1]
                    image = image.astype(np.float32) / 255.0
                    
                    # Get label from folder name
                    label = os.path.basename(folder_path)
                    
                    images.append(image)
                    labels.append(int(label))
                    image_paths.append(img_path)
                    
                except Exception as e:
                    print(f"⚠️  Error loading {img_path}: {e}")
                    continue
        
        return images, labels, image_paths
    
    def prepare_fine_tuning_data(self, original_data_ratio=0.1, augmented_data_ratio=0.9):
        """Prepare data for fine-tuning with custom ratios"""
        print("📊 Preparing fine-tuning dataset...")
        
        # Load original data
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data_splits()
        
        # Apply preprocessing
        # x_train = preprocess_images(x_train)
        # x_val = preprocess_images(x_val)
        # x_test = preprocess_images(x_test)
        
        # Fine-tuning never needs QAT -tyle uint8 data; we always train on
        # float32 [0,1] regardless of the global quantisation flags.
        x_train = preprocess_for_training(x_train)
        x_val   = preprocess_for_training(x_val)
        x_test  = preprocess_for_training(x_test)       
            
        # DEBUG: Check label shapes and types
        print(f"🔍 Label shapes - y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
        
        # Use sparse categorical crossentropy (integer labels)
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
        y_test = y_test.astype(np.int32)
        print("🔧 Using sparse categorical crossentropy")
        
        # Check for augmented data in DATA_SOURCES
        augmented_data_dir = None
        for source in params.DATA_SOURCES:
            if 'augmented' in source.get('name', '').lower():
                augmented_data_dir = source['path']
                break
        
        # Also check common augmented directory names
        if augmented_data_dir is None or not os.path.exists(augmented_data_dir):
            possible_dirs = [
                "dataset_augmented",
                "augmented_dataset", 
                "augmented_data",
                "data_augmented",
                "datasets/meterdigits_augmented"
            ]
            for dir_name in possible_dirs:
                if os.path.exists(dir_name):
                    augmented_data_dir = dir_name
                    break
        
        x_augmented_list = []
        y_augmented_list = []
        
        if augmented_data_dir and os.path.exists(augmented_data_dir):
            print(f"🔍 Found augmented data: {augmented_data_dir}")
            
            # Load augmented data using our local function
            for class_label in range(params.NB_CLASSES):
                class_dir = os.path.join(augmented_data_dir, str(class_label))
                if os.path.exists(class_dir):
                    images, labels, _ = self.load_images_from_folder(class_dir, target_shape=(params.INPUT_HEIGHT, params.INPUT_WIDTH))
                    if images:
                        x_augmented_list.extend(images)
                        y_augmented_list.extend(labels)
            
            if x_augmented_list:
                x_augmented = np.array(x_augmented_list)
                y_augmented = np.array(y_augmented_list)
                
                # Apply preprocessing to augmented data
                x_augmented = preprocess_for_training(x_augmented)
                
                # Use sparse labels for augmented data too
                y_augmented = y_augmented.astype(np.int32)
                
                print(f"✅ Loaded {len(x_augmented)} augmented samples")
                
                # Combine datasets according to ratios
                original_samples = int(len(x_train) * original_data_ratio)
                augmented_samples = int(len(x_augmented) * augmented_data_ratio)
                
                # Take subset of original data
                indices = np.random.choice(len(x_train), original_samples, replace=False)
                x_train_ft = x_train[indices]
                y_train_ft = y_train[indices]
                
                # Take subset of augmented data
                indices_aug = np.random.choice(len(x_augmented), augmented_samples, replace=False)
                x_train_ft = np.concatenate([x_train_ft, x_augmented[indices_aug]], axis=0)
                y_train_ft = np.concatenate([y_train_ft, y_augmented[indices_aug]], axis=0)
                
                # Shuffle the combined dataset
                shuffle_indices = np.random.permutation(len(x_train_ft))
                x_train_ft = x_train_ft[shuffle_indices]
                y_train_ft = y_train_ft[shuffle_indices]
                
                print(f"🎯 Fine-tuning dataset:")
                print(f"   Original samples: {original_samples}")
                print(f"   Augmented samples: {augmented_samples}")
                print(f"   Total: {len(x_train_ft)}")
                print(f"   Label shape: {y_train_ft.shape}")
                
                return (x_train_ft, y_train_ft), (x_val, y_val), (x_test, y_test)
        
        # Fallback: use original data with reduced samples
        print("⚠️  No augmented data found, using reduced original dataset")
        samples = int(len(x_train) * original_data_ratio)
        indices = np.random.choice(len(x_train), samples, replace=False)
        
        x_train_ft = x_train[indices]
        y_train_ft = y_train[indices]
        
        # Shuffle
        shuffle_indices = np.random.permutation(len(x_train_ft))
        x_train_ft = x_train_ft[shuffle_indices]
        y_train_ft = y_train_ft[shuffle_indices]
        
        print(f"🎯 Fine-tuning dataset (fallback):")
        print(f"   Total: {len(x_train_ft)}")
        print(f"   Label shape: {y_train_ft.shape}")
        
        return (x_train_ft, y_train_ft), (x_val, y_val), (x_test, y_test)
    
    def create_fine_tuning_model(self, base_model, fine_tune_strategy='full', learning_rate_multiplier=0.1):
        """Create fine-tuning model with different strategies"""
        
        # Use sparse categorical crossentropy for integer labels
        loss_fn = 'sparse_categorical_crossentropy'
        print("🔧 Using sparse_categorical_crossentropy loss")
        
        if fine_tune_strategy == 'full':
            # Fine-tune all layers
            for layer in base_model.layers:
                layer.trainable = True
            
            # Use lower learning rate for fine-tuning
            fine_tune_lr = params.LEARNING_RATE * learning_rate_multiplier
            base_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                loss=loss_fn,
                metrics=['accuracy']
            )
            print(f"🔧 Full fine-tuning: all layers trainable, LR: {fine_tune_lr}")
            
        elif fine_tune_strategy == 'last_layer':
            # Freeze all layers except the last one
            for layer in base_model.layers[:-1]:
                layer.trainable = False
            base_model.layers[-1].trainable = True
            
            fine_tune_lr = params.LEARNING_RATE * learning_rate_multiplier
            base_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                loss=loss_fn,
                metrics=['accuracy']
            )
            print(f"🔧 Last layer fine-tuning: only last layer trainable, LR: {fine_tune_lr}")
            
        elif fine_tune_strategy == 'feature_extractor':
            # Freeze base layers, add new classifier
            for layer in base_model.layers:
                layer.trainable = False
            
            # Remove the last layer and add new classifier
            base_output = base_model.layers[-2].output
            x = tf.keras.layers.Dense(128, activation='relu')(base_output)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=params.LEARNING_RATE),
                loss=loss_fn,
                metrics=['accuracy']
            )
            print("🔧 Feature extractor: frozen base + new classifier")
            return model
            
        else:
            raise ValueError(f"Unknown fine-tuning strategy: {fine_tune_strategy}")
        
        return base_model

def fine_tune_model(
    fine_tune_strategy='full',
    data_ratio=0.1,
    augmented_ratio=0.9,
    model_path=None,
    learning_rate_multiplier=0.1,
    debug=False
):
    """Main fine-tuning function"""
    
    # Set up
    set_all_seeds(params.SHUFFLE_SEED)
    setup_tensorflow_logging(debug)
    
    # Create output directory
    color_mode = "GRAY" if params.USE_GRAYSCALE else "RGB"
    fine_tune_dir = os.path.join(
        params.OUTPUT_DIR, 
        f"fine_tune_{params.MODEL_ARCHITECTURE}_{params.NB_CLASSES}cls_{color_mode}"
    )
    if os.path.exists(fine_tune_dir):
        import shutil
        shutil.rmtree(fine_tune_dir)
    os.makedirs(fine_tune_dir, exist_ok=True)
    
    print("🚀 Starting Model Fine-Tuning")
    print("=" * 60)
    
    # Initialize managers
    ft_manager = FineTuneManager(model_path, debug)
    
    # Load pre-trained model
    print("📥 Loading pre-trained model...")
    model = ft_manager.load_pretrained_model(model_path)
    
    # DEBUG: Check model output shape
    print(f"🔍 Model output shape: {model.output_shape}")
    
    # Prepare fine-tuning data
    print("📊 Preparing fine-tuning dataset...")
    data = ft_manager.prepare_fine_tuning_data(data_ratio, augmented_ratio)
    (x_train_ft, y_train_ft), (x_val, y_val), (x_test, y_test) = data
    
    # Apply fine-tuning strategy
    print(f"🎯 Fine-tuning strategy: {fine_tune_strategy}")
    model = ft_manager.create_fine_tuning_model(
        model, fine_tune_strategy, learning_rate_multiplier
    )
    
    # Create callbacks for fine-tuning
    representative_data = create_representative_dataset(x_train_ft)
    tflite_manager = TFLiteModelManager(fine_tune_dir, debug)
    monitor = TrainingMonitor(fine_tune_dir, debug)
    monitor.set_model(model)
    
    # Use fewer epochs for fine-tuning
    # fine_tune_epochs = max(20, params.EPOCHS // 2)
    fine_tune_epochs = max(20, params.EPOCHS *2)
    
    callbacks = create_callbacks(
        fine_tune_dir, tflite_manager, representative_data, 
        fine_tune_epochs, monitor, debug
    )
    
    # Fine-tune the model
    print("\n🎯 Starting fine-tuning...")
    print("-" * 60)
    
    start_time = datetime.now()
    
    # Test a single batch first to catch any shape mismatches
    print("🧪 Testing with single batch...")
    try:
        test_batch_size = min(32, len(x_train_ft))
        test_loss, test_acc = model.evaluate(
            x_train_ft[:test_batch_size], 
            y_train_ft[:test_batch_size], 
            verbose=0
        )
        print(f"✅ Single batch test passed - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    except Exception as e:
        print(f"❌ Single batch test failed: {e}")
        print(f"   Input shape: {x_train_ft[:test_batch_size].shape}")
        print(f"   Label shape: {y_train_ft[:test_batch_size].shape}")
        print(f"   Model output shape: {model.output_shape}")
        raise
    
    history = model.fit(
        x_train_ft, y_train_ft,
        batch_size=params.BATCH_SIZE,
        epochs=fine_tune_epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=0
    )
    
    training_time = datetime.now() - start_time
    
    # Evaluate fine-tuned model
    print("\n📈 Evaluating fine-tuned model...")
    
    # Keras model evaluation
    train_accuracy = model.evaluate(x_train_ft, y_train_ft, verbose=0)[1]
    val_accuracy = model.evaluate(x_val, y_val, verbose=0)[1]
    test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    
    # Save TFLite models
    final_quantized, quantized_size = tflite_manager.save_as_tflite(
        model, "fine_tuned_quantized.tflite", quantize=True, 
        representative_data=representative_data
    )
    final_float, float_size = tflite_manager.save_as_tflite(
        model, "fine_tuned_float.tflite", quantize=False
    )
    
    # Save trainable checkpoint
    checkpoint_path = os.path.join(fine_tune_dir, f"fine_tuned_model_acc_{test_accuracy:.4f}.keras")
    model.save(checkpoint_path)
    print(f"💾 Saved trainable model: {checkpoint_path}")
    
    # Save results
    monitor.save_training_plots()
    
    # Print results
    print("\n" + "=" * 60)
    print("🏁 FINE-TUNING COMPLETED")
    print("=" * 60)
    print(f"⏱️  Fine-tuning time: {training_time}")
    print(f"📊 Results:")
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Validation Accuracy: {val_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Best Validation Accuracy: {tflite_manager.best_accuracy:.4f}")
    print(f"   Quantized Model Size: {quantized_size:.1f} KB")
    
    return model, history, fine_tune_dir

def main():
    """Main entry point for fine-tuning"""
    parser = argparse.ArgumentParser(description='Model Fine-Tuning')
    parser.add_argument('--strategy', type=str, default='full',
                       choices=['full', 'last_layer', 'feature_extractor'],
                       help='Fine-tuning strategy')
    parser.add_argument('--data_ratio', type=float, default=0.1,
                       help='Ratio of original data to use (default: 0.1)')
    parser.add_argument('--augmented_ratio', type=float, default=0.9,
                       help='Ratio of augmented data to use (default: 0.9)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model directory')
    parser.add_argument('--learning_rate_multiplier', type=float, default=0.5,
                       help='Learning rate multiplier for fine-tuning (default: 0.1)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        model, history, output_dir = fine_tune_model(
            fine_tune_strategy=args.strategy,
            data_ratio=args.data_ratio,
            augmented_ratio=args.augmented_ratio,
            model_path=args.model_path,
            learning_rate_multiplier=args.learning_rate_multiplier,
            debug=args.debug
        )
        print(f"\n✅ Fine-tuning completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Fine-tuning interrupted by user")
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()