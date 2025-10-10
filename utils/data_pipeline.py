# data_pipeline.py
import tensorflow as tf
import numpy as np
from utils import preprocess_images
import parameters as params

def create_tf_dataset(images, labels, training=True, batch_size=None):
    """Create optimized tf.data.Dataset pipeline"""
    if batch_size is None:
        batch_size = params.BATCH_SIZE
    
    # Convert to tensors
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Apply preprocessing
    def preprocess_fn(image, label):
        # Ensure proper preprocessing
        image = tf.cast(image, tf.float32)
        
        # Apply model-specific preprocessing
        if params.ESP_DL_QUANTIZE:
            # Normalize to [-1, 1] for ESP-DL
            image = (image / 127.5) - 1.0
        else:
            # Normalize to [0, 1]
            image = image / 255.0
        
        # Ensure correct shape
        if len(image.shape) == 2:  # Grayscale without channel
            image = tf.expand_dims(image, axis=-1)
        
        return image, label
    
    dataset = dataset.map(preprocess_fn, num_parallel_calls=params.TF_DATA_PARALLEL_CALLS)
    
    # Training-specific augmentations
    if training and params.USE_DATA_AUGMENTATION:
        dataset = dataset.map(augment_image, num_parallel_calls=params.TF_DATA_PARALLEL_CALLS)
    
    # Shuffle and batch
    if training:
        dataset = dataset.shuffle(params.TF_DATA_SHUFFLE_BUFFER)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(params.TF_DATA_PREFETCH_SIZE)
    
    return dataset

def augment_image(image, label):
    """Apply data augmentation using tf.image operations"""
    # Random rotations
    angle = tf.random.uniform([], -0.1, 0.1)  # ±5.7 degrees
    image = tfa.image.rotate(image, angles=angle)
    
    # Random zoom
    scale = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.shape(image)[:2] * scale, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, tf.shape(image)[0], tf.shape(image)[1])
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Ensure values are still in valid range
    image = tf.clip_by_value(image, -1.0 if params.ESP_DL_QUANTIZE else 0.0, 1.0)
    
    return image, label

def get_tf_data_splits():
    """Get data splits as tf.data.Dataset objects"""
    from utils import get_data_splits
    
    # Get original splits
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data_splits()
    
    # Convert to tf.data.Dataset
    train_dataset = create_tf_dataset(x_train, y_train, training=True)
    val_dataset = create_tf_dataset(x_val, y_val, training=False)
    test_dataset = create_tf_dataset(x_test, y_test, training=False)
    
    print(f"📊 TF.Data Pipeline Created:")
    print(f"   Training batches: {len(list(train_dataset))}")
    print(f"   Validation batches: {len(list(val_dataset))}")
    print(f"   Test batches: {len(list(test_dataset))}")
    
    return train_dataset, val_dataset, test_dataset

class DataPipeline:
    """Advanced data pipeline with caching and optimization"""
    
    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def build_pipeline(self, cache_dir=None):
        """Build optimized data pipeline"""
        train_ds, val_ds, test_ds = get_tf_data_splits()
        
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