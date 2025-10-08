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
    
    def __init__(self, debug=False):
        self.debug = debug
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for fine-tuning"""
        setup_tensorflow_logging(self.debug)
        if self.debug:
            print("🔧 Fine-tuning debug mode enabled")
    
    def find_best_trainable_checkpoint(self, models_dir="exported_models"):
        """Find the best trainable checkpoint (H5 or SavedModel)"""
        if not os.path.exists(models_dir):
            return None
            
        # Look for all training directories
        training_dirs = []
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path) and any(x in item for x in ['training', 'fine_tune']):
                training_dirs.append(item_path)
        
        if not training_dirs:
            print("❌ No training directories found")
            return None
        
        # Sort by modification time (newest first)
        training_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_training_dir = training_dirs[0]
        
        print(f"🔍 Searching in: {latest_training_dir}")
        
        # Look for H5 checkpoints first
        h5_files = [f for f in os.listdir(latest_training_dir) if f.endswith('.h5') and 'best_model' in f]
        
        if h5_files:
            # Sort by accuracy in filename
            def extract_accuracy(filename):
                match = re.search(r'acc_([\d.]+)', filename)
                return float(match.group(1)) if match else 0.0
            
            h5_files.sort(key=extract_accuracy, reverse=True)
            best_h5 = os.path.join(latest_training_dir, h5_files[0])
            print(f"✅ Found H5 checkpoint: {best_h5}")
            return best_h5
        
        # Look for SavedModel
        savedmodel_dir = os.path.join(latest_training_dir, "best_model_savedmodel")
        if os.path.exists(savedmodel_dir):
            print(f"✅ Found SavedModel: {savedmodel_dir}")
            return savedmodel_dir
        
        # Fallback: look for any H5 file
        any_h5 = [f for f in os.listdir(latest_training_dir) if f.endswith('.h5')]
        if any_h5:
            any_h5.sort(key=lambda x: os.path.getmtime(os.path.join(latest_training_dir, x)), reverse=True)
            fallback_h5 = os.path.join(latest_training_dir, any_h5[0])
            print(f"⚠️ Using fallback H5: {fallback_h5}")
            return fallback_h5
        
        print("❌ No trainable checkpoints found")
        return None
    
    def load_pretrained_model(self, model_path=None):
        """Load a pre-trained model for fine-tuning"""
        if model_path is None:
            # Find the best trainable checkpoint
            model_path = self.find_best_trainable_checkpoint()
            if model_path is None:
                raise ValueError("No trainable checkpoints found. Please train a model first with checkpoint saving enabled.")
        
        if self.debug:
            print(f"📁 Loading model from: {model_path}")
        
        try:
            # Load the model
            if os.path.isdir(model_path):  # SavedModel format
                model = tf.keras.models.load_model(model_path)
                print("✅ Loaded SavedModel")
            else:  # H5 format
                model = tf.keras.models.load_model(model_path)
                print("✅ Loaded H5 model")
            
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
        x_train = preprocess_images(x_train)
        x_val = preprocess_images(x_val)
        x_test = preprocess_images(x_test)
        
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
                x_augmented = preprocess_images(x_augmented)
                
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fine_tune_dir = os.path.join(
        params.OUTPUT_DIR, 
        f"fine_tune_{params.MODEL_ARCHITECTURE}_{timestamp}"
    )
    os.makedirs(fine_tune_dir, exist_ok=True)
    
    print("🚀 Starting Model Fine-Tuning")
    print("=" * 60)
    
    # Initialize managers
    ft_manager = FineTuneManager(debug)
    
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
    fine_tune_epochs = max(20, params.EPOCHS // 2)
    
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
    checkpoint_path = os.path.join(fine_tune_dir, f"fine_tuned_model_acc_{test_accuracy:.4f}.h5")
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
    parser.add_argument('--learning_rate_multiplier', type=float, default=0.1,
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