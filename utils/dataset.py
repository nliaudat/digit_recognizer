import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import parameters as params
from .dataset_loader import get_data_splits, load_custom_dataset
from .multi_source_loader import get_data_splits, load_combined_dataset

def load_digit_dataset():
    """
    Legacy function for compatibility with existing code
    """
    images, labels = load_combined_dataset()
    
    # Use specified percentage
    total_samples = len(images)
    training_samples = int(total_samples * params.TRAINING_PERCENTAGE)
    
    indices = np.random.permutation(total_samples)[:training_samples]
    x_train = images[indices]
    y_train = labels[indices]
    
    # For test, use different samples
    test_size = min(10000, total_samples // 5)
    test_indices = np.random.permutation(total_samples)[:test_size]
    x_test = images[test_indices]
    y_test = labels[test_indices]
    
    return (x_train, y_train), (x_test, y_test)

def get_data_splits():
    """
    Get train/validation/test splits according to parameters
    """
    (x_train, y_train), (x_test, y_test) = load_digit_dataset()
    
    # Further split training into train/validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, 
        test_size=params.VALIDATION_SPLIT, 
        random_state=42
    )
    
    print(f"Data splits:")
    print(f"  Training: {len(x_train)} samples")
    print(f"  Validation: {len(x_val)} samples") 
    print(f"  Test: {len(x_test)} samples")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)