# utils/augmentation.py
import tensorflow as tf
import numpy as np
import parameters as params
from utils.data_pipeline import create_tf_dataset_from_arrays

# # -------------------------------------------------------------
# #  NEW helper that augments a *batched* tensor image by image
# # -------------------------------------------------------------
# def _apply_augmentation_to_batch(batch_images, batch_labels, pipeline):
    # """
    # `batch_images`  : Tensor of shape (B, H, W, C)
    # `batch_labels`  : Tensor of shape (B, …)

    # Returns a new batch where every image has been passed through
    # `pipeline` (which expects a single image, i.e. rank 3).
    # """
    # # tf.map_fn runs the lambda on each element of the first dimension (the batch)
    # augmented_images = tf.map_fn(
        # lambda img: pipeline(img, training=True),   # keep training=True for randomness
        # batch_images,
        # fn_output_signature=batch_images.dtype
    # )
    # return augmented_images, batch_labels
    
# -------------------------------------------------------------
#  NEW: simple per image augmentation wrapper
# -------------------------------------------------------------
def _augment_one_image(image, label, pipeline):
    """
    Apply the augmentation pipeline to a *single* image.
    The pipeline expects a rank 3 tensor (H, W, C) and returns the
    same rank 3 tensor.  The label is passed through unchanged.
    """
    # `pipeline` is a `tf.keras.Sequential` that expects a single image.
    # We keep `training=True` so that random transforms are active.
    aug_img = pipeline(image, training=True)
    return aug_img, label

def create_augmentation_pipeline():
    """
    Create a comprehensive data augmentation pipeline with data type preservation and value clamping
    """
    augmentation_layers = []
    
    # Ensure float32 for augmentation operations
    augmentation_layers.append(
        tf.keras.layers.Lambda(
            lambda x: tf.cast(x, tf.float32),
            name='ensure_float32'
        )
    )
    
    # Verify and clamp data range for augmentation
    augmentation_layers.append(
        tf.keras.layers.Lambda(
            lambda x: tf.clip_by_value(x, 0.0, 1.0),  # Ensure [0,1] range
            name='verify_range_before_augmentation'
        )
    )
    
    # Rotation
    if params.AUGMENTATION_ROTATION_RANGE > 0:
        rotation_factor = params.AUGMENTATION_ROTATION_RANGE / 360.0
        augmentation_layers.append(
            tf.keras.layers.RandomRotation(
                factor=rotation_factor,
                fill_mode='constant',
                fill_value=0.0,
                name='random_rotation'
            )
        )

    # Translation
    if params.AUGMENTATION_WIDTH_SHIFT_RANGE > 0 or params.AUGMENTATION_HEIGHT_SHIFT_RANGE > 0:
        augmentation_layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=params.AUGMENTATION_HEIGHT_SHIFT_RANGE,
                width_factor=params.AUGMENTATION_WIDTH_SHIFT_RANGE,
                fill_mode='constant',
                fill_value=0.0,
                name='random_translation'
            )
        )

    # Zoom
    if params.AUGMENTATION_ZOOM_RANGE > 0:
        augmentation_layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=params.AUGMENTATION_ZOOM_RANGE,
                width_factor=params.AUGMENTATION_ZOOM_RANGE,
                fill_mode='constant',
                fill_value=0.0,
                name='random_zoom'
            )
        )

    # Brightness - WITH SAFE RANGE
    if params.AUGMENTATION_BRIGHTNESS_RANGE != [1.0, 1.0]:
        min_delta = params.AUGMENTATION_BRIGHTNESS_RANGE[0] - 1.0
        max_delta = params.AUGMENTATION_BRIGHTNESS_RANGE[1] - 1.0
        augmentation_layers.append(
            tf.keras.layers.RandomBrightness(
                factor=(min_delta, max_delta),
                value_range=(0, 1),  # Explicitly define expected range
                name='random_brightness'
            )
        )

    # Contrast - WITH VALUE PROTECTION
    if params.AUGMENTATION_CONTRAST_RANGE > 0:
        # Add protection before contrast to prevent extreme values
        augmentation_layers.append(
            tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, 0.1, 0.9),  # Clip before contrast
                name='pre_contrast_clip'
            )
        )
        augmentation_layers.append(
            tf.keras.layers.RandomContrast(
                factor=params.AUGMENTATION_CONTRAST_RANGE,
                name='random_contrast'
            )
        )

    # Flips
    if params.AUGMENTATION_HORIZONTAL_FLIP:
        augmentation_layers.append(
            tf.keras.layers.RandomFlip(
                mode='horizontal',
                name='random_horizontal_flip'
            )
        )

    if params.AUGMENTATION_VERTICAL_FLIP:
        augmentation_layers.append(
            tf.keras.layers.RandomFlip(
                mode='vertical',
                name='random_vertical_flip'
            )
        )
        
    # Add small random noise to prevent dead neurons (helps stability)
    augmentation_layers.append(
        tf.keras.layers.GaussianNoise(
            stddev=0.001,  # Very small noise
            name='stability_noise'
        )
    )
    
    # FINAL VALUE CLAMPING 
    augmentation_layers.append(
        tf.keras.layers.Lambda(
            lambda x: tf.clip_by_value(x, 0.0, 1.0),  # Ensure valid range
            name='final_value_clamp'
        )
    )
    
    # Ensure final float32 output
    augmentation_layers.append(
        tf.keras.layers.Lambda(
            lambda x: tf.cast(x, tf.float32),
            name='ensure_float32_output'
        )
    )
    
    # Create augmentation pipeline
    augmentation_pipeline = tf.keras.Sequential(augmentation_layers, name='augmentation_pipeline')
    
    return augmentation_pipeline, len(augmentation_layers)

def apply_augmentation_to_dataset(dataset, augmentation_pipeline):
    """
    Apply augmentation pipeline to tf.data.Dataset
    """
    return dataset.map(
        lambda x, y: (augmentation_pipeline(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

def test_augmentation_pipeline(augmentation_pipeline, sample_data, debug=False):
    """
    Test the augmentation pipeline with sample data
    """
    if debug:
        print("🧪 Testing augmentation pipeline...")
        
        # Test with sample data
        sample_output = augmentation_pipeline(sample_data, training=True)
        
        print(f"   Input range: [{sample_data.numpy().min():.3f}, {sample_data.numpy().max():.3f}]")
        print(f"   Output range: [{sample_output.numpy().min():.3f}, {sample_output.numpy().max():.3f}]")
        print(f"   Input shape: {sample_data.shape}")
        print(f"   Output shape: {sample_output.shape}")
        
        return sample_output
    return None

def get_augmentation_summary():
    """
    Get a summary of the augmentation configuration
    """
    summary = {
        'rotation_range': params.AUGMENTATION_ROTATION_RANGE,
        'width_shift_range': params.AUGMENTATION_WIDTH_SHIFT_RANGE,
        'height_shift_range': params.AUGMENTATION_HEIGHT_SHIFT_RANGE,
        'zoom_range': params.AUGMENTATION_ZOOM_RANGE,
        'brightness_range': params.AUGMENTATION_BRIGHTNESS_RANGE,
        'contrast_range': params.AUGMENTATION_CONTRAST_RANGE,
        'horizontal_flip': params.AUGMENTATION_HORIZONTAL_FLIP,
        'vertical_flip': params.AUGMENTATION_VERTICAL_FLIP,
        'enabled': params.USE_DATA_AUGMENTATION
    }
    
    return summary

def print_augmentation_summary():
    """
    Print a formatted summary of augmentation settings
    """
    if not params.USE_DATA_AUGMENTATION:
        print("ℹ️  Data augmentation: Disabled")
        return
    
    summary = get_augmentation_summary()
    
    print("🔄 Data Augmentation Configuration:")
    print("   " + "-" * 40)
    
    if summary['rotation_range'] > 0:
        print(f"   ✓ Rotation: ±{summary['rotation_range']}°")
    
    if summary['width_shift_range'] > 0 or summary['height_shift_range'] > 0:
        print(f"   ✓ Translation: W{summary['width_shift_range']}, H{summary['height_shift_range']}")
    
    if summary['zoom_range'] > 0:
        print(f"   ✓ Zoom: {summary['zoom_range']}")
    
    if summary['brightness_range'] != [1.0, 1.0]:
        print(f"   ✓ Brightness: {summary['brightness_range']}")
    
    if summary['contrast_range'] > 0:
        print(f"   ✓ Contrast: {summary['contrast_range']}")
    
    if summary['horizontal_flip']:
        print(f"   ✓ Horizontal Flip: Enabled")
    
    if summary['vertical_flip']:
        print(f"   ✓ Vertical Flip: Enabled")
    
    print("   " + "-" * 40)

class AugmentationSafetyMonitor(tf.keras.callbacks.Callback):
    """
    Monitor training to detect if augmentation is causing issues
    """
    def __init__(self, validation_data, debug=False, safety_threshold=10.0, learning_threshold=0.15, patience_epochs=5):
        """
        Args:
            validation_data: Tuple of (x_val, y_val) for emergency validation
            debug: Whether to print debug information
            safety_threshold: If val_loss > this, augmentation is considered broken
            learning_threshold: Minimum accuracy expected after patience_epochs
            patience_epochs: Number of epochs to wait before checking learning progress
        """
        super().__init__()
        self.validation_data = validation_data
        self.debug = debug
        self.safety_threshold = safety_threshold
        self.learning_threshold = learning_threshold
        self.patience_epochs = patience_epochs
        self.emergency_triggered = False
        
        if debug:
            print(f"🔒 AugmentationSafetyMonitor initialized:")
            print(f"   Safety threshold (val_loss): {safety_threshold}")
            print(f"   Learning threshold (val_acc): {learning_threshold}")
            print(f"   Patience epochs: {patience_epochs}")
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Check for augmentation-related issues at the end of each epoch
        """
        if logs is None:
            return
            
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        
        # Check for catastrophic validation loss (augmentation producing invalid data)
        if val_loss > self.safety_threshold and not self.emergency_triggered:
            print(f"🚨 EMERGENCY: Validation loss {val_loss:.1f} exceeds safety threshold!")
            print("   Possible causes:")
            print("   - Augmentation producing NaN/inf values")
            print("   - Data range corrupted by augmentation")
            print("   - Labels corrupted by augmentation pipeline")
            print("   - Model instability due to extreme augmentations")
            
            # Run emergency validation to confirm
            self._emergency_validation(epoch)
            self.emergency_triggered = True
        
        # Check if model is not learning (augmentation destroying signal)
        if (epoch > self.patience_epochs and 
            val_acc < self.learning_threshold and 
            not self.emergency_triggered):
            
            print(f"🚨 EMERGENCY: Model not learning after {epoch} epochs (val_acc: {val_acc:.3f})")
            print("   Possible causes:")
            print("   - Augmentation too aggressive, destroying class information")
            print("   - Data normalization issues in augmentation pipeline")
            print("   - Label corruption in augmentation")
            print("   - Learning rate too high/low for augmented data")
            
            self._emergency_validation(epoch)
            self.emergency_triggered = True
        
        # Additional sanity checks
        if self.debug and epoch % 10 == 0:
            self._sanity_check(epoch, logs)
    
    def _emergency_validation(self, epoch):
        """
        Run emergency validation to diagnose augmentation issues
        """
        print(f"🔍 Running emergency diagnostics at epoch {epoch}...")
        
        try:
            # Test with a small batch of validation data
            x_val, y_val = self.validation_data
            sample_size = min(32, len(x_val))
            
            # Check data ranges
            x_sample = x_val[:sample_size]
            print(f"   Validation data range: [{x_sample.min():.3f}, {x_sample.max():.3f}]")
            print(f"   Validation data mean: {x_sample.mean():.3f}")
            print(f"   Validation data dtype: {x_sample.dtype}")
            
            # Check for NaN/inf
            if np.any(np.isnan(x_sample)):
                print("   ❌ NaN values detected in validation data!")
            if np.any(np.isinf(x_sample)):
                print("   ❌ Inf values detected in validation data!")
                
            # Check label distribution
            unique_labels, counts = np.unique(y_val[:sample_size], return_counts=True)
            print(f"   Unique labels in sample: {len(unique_labels)}")
            print(f"   Label distribution: {dict(zip(unique_labels, counts))}")
            
            # CRITICAL: Check if data is normalized
            if x_sample.max() > 5.0:  # Data should be normalized to ~0-1 range
                print("   🚨 CRITICAL: Data appears to be unnormalized!")
                print("   Expected range: [0, 1] or [-1, 1]")
                print("   Actual range: [{:.3f}, {:.3f}]".format(x_sample.min(), x_sample.max()))
                print("   This will cause massive loss values!")
                
        except Exception as e:
            print(f"   ⚠️ Emergency validation failed: {e}")
        
    def _sanity_check(self, epoch, logs):
        """
        Regular sanity checks during training
        """
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        
        # Check for overfitting to augmentation
        if train_loss < 0.1 and val_loss > 1.0 and epoch > 10:
            print(f"⚠️  Possible overfitting to augmented data (epoch {epoch})")
            print(f"   Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        
        # Check for divergence between train and val
        loss_ratio = val_loss / train_loss if train_loss > 0 else 1.0
        if loss_ratio > 5.0 and epoch > 5:
            print(f"⚠️  Large train/val loss divergence (ratio: {loss_ratio:.1f})")
            print("   Augmentation may be making training too easy")
    
    def on_train_end(self, logs=None):
        """
        Final report at the end of training
        """
        if not self.emergency_triggered:
            print("✅ AugmentationSafetyMonitor: No critical issues detected during training")
        else:
            print("🚨 AugmentationSafetyMonitor: Critical issues were detected!")
            print("   Consider reviewing your augmentation pipeline configuration")

def create_augmentation_safety_monitor(validation_data, debug=False):
    """
    Helper function to create an AugmentationSafetyMonitor
    """
    return AugmentationSafetyMonitor(
        validation_data=validation_data,
        debug=debug,
        safety_threshold=10.0,
        learning_threshold=0.15,
        patience_epochs=5
    )

def setup_augmentation_for_training(x_train, y_train_final,
                                    x_val, y_val_final, debug=False):
    """
    Build a tf.data pipeline that **applies augmentation before batching**.
    This avoids the rank mismatch error that occurs when treating a
    single image as a batch.
    """
    print("🔄 Setting up data augmentation pipeline...")

    # 1️⃣  Create the augmentation pipeline (unchanged)
    augmentation_pipeline, num_layers = create_augmentation_pipeline()
    print(f"✅ Augmentation pipeline created with {num_layers} layers")
    print_augmentation_summary()

    # 2️⃣  Build the base (unbatched) datasets
    if params.USE_TF_DATA_PIPELINE:
        print("🔧 Using tf.data pipeline with augmentation...")
        train_dataset = create_tf_dataset_from_arrays(
            x_train, y_train_final, training=True)
        val_dataset   = create_tf_dataset_from_arrays(
            x_val,   y_val_final,   training=False)
    else:
        print("🔧 Using standard arrays with augmentation...")
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train_final))
        val_dataset   = tf.data.Dataset.from_tensor_slices(
            (x_val,   y_val_final))

    # 3️⃣  **Apply augmentation per image** (before batching)
    #    The lambda receives a single image + its label.
    train_dataset = train_dataset.map(
        lambda img, lbl: _augment_one_image(img, lbl, augmentation_pipeline),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 4️⃣  Now batch / shuffle / prefetch
    train_dataset = train_dataset.shuffle(1000) \
                                 .batch(params.BATCH_SIZE) \
                                 .prefetch(tf.data.AUTOTUNE)

    # Validation data – **no augmentation**
    val_dataset = val_dataset.batch(params.BATCH_SIZE) \
                             .prefetch(tf.data.AUTOTUNE)

    # 5️⃣  Quick sanity check
    print("🧪 Testing data pipeline...")
    sample_batch = next(iter(train_dataset))
    sample_x, sample_y = sample_batch
    print(f"   Sample batch – X range: [{sample_x.numpy().min():.3f}, "
          f"{sample_x.numpy().max():.3f}]")
    print(f"   Sample batch – Y shape: {sample_y.numpy().shape}")

    return train_dataset, val_dataset, augmentation_pipeline