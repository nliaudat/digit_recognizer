# data_pipeline.py
import tensorflow as tf
import numpy as np
import os
from utils import preprocess_images
import parameters as params

def create_tf_dataset_from_arrays(x_data, y_data, training=True, batch_size=None):
    """Create tf.data.Dataset from pre-loaded arrays"""
    if batch_size is None:
        batch_size = params.BATCH_SIZE
    
    # Convert to tensors
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    
    # Apply consistent preprocessing (NO AUGMENTATION)
    def preprocess_fn(image, label):
        # Ensure proper data type
        image = tf.cast(image, tf.float32)
        
        # Use the SAME normalization as preprocess_images
        image = image / 255.0  # Consistent with preprocess_images
        
        # Ensure correct shape
        if len(image.shape) == 2:  # Grayscale without channel
            image = tf.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] == 3 and params.USE_GRAYSCALE:
            # Convert RGB to grayscale if needed
            image = tf.image.rgb_to_grayscale(image)
        
        return image, label
    
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if training:
        dataset = dataset.shuffle(params.TF_DATA_SHUFFLE_BUFFER)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_tf_data_splits_from_arrays(x_train, y_train, x_val, y_val, x_test, y_test):
    """Get data splits as tf.data.Dataset objects from pre-loaded arrays"""
    # Convert to tf.data.Dataset
    train_dataset = create_tf_dataset_from_arrays(x_train, y_train, training=True)
    val_dataset = create_tf_dataset_from_arrays(x_val, y_val, training=False)
    test_dataset = create_tf_dataset_from_arrays(x_test, y_test, training=False)
    
    print(f"📊 TF.Data Pipeline Created from arrays:")
    print(f"   Training samples: {len(x_train)}")
    print(f"   Validation samples: {len(x_val)}")
    print(f"   Test samples: {len(x_test)}")
    
    return train_dataset, val_dataset, test_dataset

# Keep the original function for backward compatibility
def get_tf_data_splits():
    """Legacy function - loads data internally (not recommended)"""
    print("⚠️  Using legacy data loading in data_pipeline.py")
    from utils import get_data_splits
    
    # Get original splits
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data_splits()
    
    # Apply preprocessing
    x_train = preprocess_images(x_train, for_training=True)
    x_val = preprocess_images(x_val, for_training=True)
    x_test = preprocess_images(x_test, for_training=True)
    
    return get_tf_data_splits_from_arrays(x_train, y_train, x_val, y_val, x_test, y_test)

class DataPipeline:
    """Advanced data pipeline with caching and optimization"""
    
    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def build_pipeline_from_arrays(self, x_train, y_train, x_val, y_val, x_test, y_test, cache_dir=None):
        """Build optimized data pipeline from pre-loaded arrays"""
        train_ds, val_ds, test_ds = get_tf_data_splits_from_arrays(
            x_train, y_train, x_val, y_val, x_test, y_test
        )
        
        # Cache datasets for performance
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            train_ds = train_ds.cache(os.path.join(cache_dir, 'train_cache'))
            val_ds = val_ds.cache(os.path.join(cache_dir, 'val_cache'))
            test_ds = test_ds.cache(os.path.join(cache_dir, 'test_cache'))
        
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds
        
        return self
    
    def get_datasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset