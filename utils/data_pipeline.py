# data_pipeline.py
import tensorflow as tf
import numpy as np
import os
import parameters as params

def create_tf_dataset_from_arrays(x_data, y_data, training=True, batch_size=None):
    """Create tf.data.Dataset from PREPROCESSED arrays (NO additional preprocessing)"""
    if batch_size is None:
        batch_size = params.BATCH_SIZE
    
    # Convert to tensors - data is already preprocessed in train.py
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    
    # NO ADDITIONAL PREPROCESSING - data is already normalized to correct range
    # Just ensure correct data types
    def ensure_correct_format(image, label):
        # Ensure proper data type
        image = tf.cast(image, tf.float32)
        return image, label
    
    dataset = dataset.map(ensure_correct_format, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if training:
        dataset = dataset.shuffle(params.TF_DATA_SHUFFLE_BUFFER)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_tf_data_splits_from_arrays(x_train, y_train, x_val, y_val, x_test, y_test):
    """Get data splits as tf.data.Dataset objects from PREPROCESSED arrays"""
    print(f"📊 Creating TF.Data pipeline from preprocessed arrays:")
    print(f"   Data range - Train: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"   Data shapes - Train: {x_train.shape}, Val: {x_val.shape}")
    
    # Convert to tf.data.Dataset
    train_dataset = create_tf_dataset_from_arrays(x_train, y_train, training=True)
    val_dataset = create_tf_dataset_from_arrays(x_val, y_val, training=False)
    test_dataset = create_tf_dataset_from_arrays(x_test, y_test, training=False)
    
    return train_dataset, val_dataset, test_dataset

class DataPipeline:
    """Advanced data pipeline with caching and optimization"""
    
    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def build_pipeline_from_arrays(self, x_train, y_train, x_val, y_val, x_test, y_test, cache_dir=None):
        """Build optimized data pipeline from pre-loaded PREPROCESSED arrays"""
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