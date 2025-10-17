import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import parameters as params

def load_custom_dataset():
    """
    Load custom dataset with proper shuffling
    """
    if not os.path.exists(params.DATASET_PATH):
        print(f"Dataset path not found: {params.DATASET_PATH}")
        # print("Using MNIST as fallback...")
        return #load_mnist_fallback()
    
    if params.DATASET_TYPE == "folder_structure":
        images, labels = load_folder_structure_dataset()
    elif params.DATASET_TYPE == "label_file":
        images, labels = load_label_file_dataset()
    else:
        raise ValueError(f"Unknown dataset type: {params.DATASET_TYPE}")
    
    # SHUFFLE THE ENTIRE DATASET
    return shuffle_dataset(images, labels)

def load_folder_structure_dataset():
    """
    Load dataset organized in folders by class
    """
    images = []
    labels = []
    
    for class_label in range(params.NB_CLASSES):
        class_dir = os.path.join(params.DATASET_PATH, str(class_label))
        
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
            
        for filename in os.listdir(class_dir):
            if any(filename.lower().endswith(ext) for ext in params.IMAGE_EXTENSIONS):
                image_path = os.path.join(class_dir, filename)
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image: {image_path}")
                    continue
                
                images.append(image)
                labels.append(class_label)
    
    if len(images) == 0:
        print("No images found in dataset directory")
        return #load_mnist_fallback()
    
    print(f"Loaded {len(images)} images from folder structure")
    return np.array(images), np.array(labels)

def load_label_file_dataset():
    """
    Load dataset with images in one folder and labels in a file
    """
    images = []
    labels = []
    
    label_file_path = os.path.join(params.DATASET_PATH, 'labels.txt')
    images_dir = os.path.join(params.DATASET_PATH, 'images')
    
    if not os.path.exists(label_file_path) or not os.path.exists(images_dir):
        print("Label file or images directory not found.")
        return #load_mnist_fallback()
    
    # Read labels file
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            filename = parts[0]
            label = int(parts[1])
            
            image_path = os.path.join(images_dir, filename)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(label)
    
    print(f"Loaded {len(images)} images from label file")
    return np.array(images), np.array(labels)

def shuffle_dataset(images, labels):
    """
    Shuffle images and labels together maintaining correspondence
    """
    # Create shuffled indices
    indices = np.random.permutation(len(images))
    
    # Apply same shuffle to both images and labels
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    
    print(f"Shuffled {len(images)} samples")
    return shuffled_images, shuffled_labels

def load_mnist_fallback():
    """
    Fallback to MNIST dataset if custom dataset is not available
    """
    print("Using MNIST dataset as fallback...")
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Combine train and test for custom splitting
    images = np.concatenate([x_train, x_test])
    labels = np.concatenate([y_train, y_test])
    
    # Convert to proper format
    if not params.USE_GRAYSCALE:
        images = np.stack([images] * 3, axis=-1)
    else:
        images = np.expand_dims(images, axis=-1)
    
    # MNIST is already shuffled, but we'll shuffle again to be safe
    return shuffle_dataset(images, labels)

def get_data_splits():
    """
    Get train/validation/test splits with proper shuffling
    """
    # Load and shuffle the entire dataset
    images, labels = load_custom_dataset()
    
    print(f"Dataset distribution before splitting:")
    for i in range(params.NB_CLASSES):
        count = np.sum(labels == i)
        print(f"  Class {i}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Use only the specified percentage of data
    total_samples = len(images)
    training_samples = int(total_samples * params.TRAINING_PERCENTAGE)
    
    # Take the first N samples (already shuffled)
    images = images[:training_samples]
    labels = labels[:training_samples]
    
    # Split into train+val and test WITH SHUFFLING
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images, labels, 
        test_size=0.2, 
        random_state=42,
        shuffle=True,  # Ensure shuffling during split
        stratify=labels  # Maintain class distribution
    )
    
    # Further split train+val into train and val WITH SHUFFLING
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, 
        test_size=params.VALIDATION_SPLIT, 
        random_state=42,
        shuffle=True,  # Ensure shuffling during split
        stratify=y_train_val  # Maintain class distribution
    )
    
    print(f"\nFinal Data splits:")
    print(f"  Total samples: {total_samples}")
    print(f"  Used samples: {len(images)} ({params.TRAINING_PERCENTAGE*100}%)")
    print(f"  Training: {len(x_train)}")
    print(f"  Validation: {len(x_val)}")
    print(f"  Test: {len(x_test)}")
    
    # Verify class distribution in splits
    print(f"\nClass distribution in splits:")
    for split_name, x, y in [("Training", x_train, y_train), 
                            ("Validation", x_val, y_val), 
                            ("Test", x_test, y_test)]:
        print(f"  {split_name}: ", end="")
        for i in range(params.NB_CLASSES):
            count = np.sum(y == i)
            print(f"{i}:{count} ", end="")
        print()
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)