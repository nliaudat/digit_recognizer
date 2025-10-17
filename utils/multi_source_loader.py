import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import parameters as params

class MultiSourceDataLoader:
    def __init__(self):
        self.all_images = []
        self.all_labels = []
        self.source_stats = {}
    
    def load_all_sources(self):
        """
        Load and combine all data sources
        """
        print("Loading multiple data sources...")
        print("=" * 50)
        
        for source_config in params.DATA_SOURCES:
            source_name = source_config['name']
            source_type = source_config['type']
            source_path = source_config['path']
            source_weight = source_config.get('weight', 1.0)
            
            print(f"Loading source: {source_name} (type: {source_type})")
            
            if source_type == 'builtin':
                images, labels = self.load_builtin_dataset(source_name)
            elif source_type == 'folder_structure':
                images, labels = self.load_folder_structure(source_path)
            elif source_type == 'label_file':
                images, labels = self.load_label_file_dataset(source_path)
            else:
                print(f"Unknown source type: {source_type}, skipping...")
                continue
            
            if len(images) == 0:
                print(f"  No data loaded from {source_name}, skipping...")
                continue
            
            # Apply sampling weight (undersample if weight < 1.0)
            if source_weight < 1.0 and len(images) > 0:
                sample_size = int(len(images) * source_weight)
                indices = np.random.choice(len(images), sample_size, replace=False)
                images = images[indices]
                labels = labels[indices]
                print(f"  Sampled {sample_size} images (weight: {source_weight})")
            
            # Store source statistics
            self.source_stats[source_name] = {
                'count': len(images),
                'class_distribution': self.get_class_distribution(labels)
            }
            
            # Add to combined dataset
            self.all_images.append(images)
            self.all_labels.append(labels)
            
            print(f"  Loaded {len(images)} images")
            print(f"  Class distribution: {self.source_stats[source_name]['class_distribution']}")
            print("-" * 30)
        
        if len(self.all_images) == 0:
            print("No data sources could be loaded.")
            return # self.load_mnist_fallback()
        
        # Combine all sources
        combined_images = np.concatenate(self.all_images, axis=0)
        combined_labels = np.concatenate(self.all_labels, axis=0)
        
        print(f"\nCombined dataset:")
        print(f"  Total images: {len(combined_images)}")
        print(f"  Sources: {list(self.source_stats.keys())}")
        
        return combined_images, combined_labels
    
    def load_builtin_dataset(self, dataset_name):
        """
        Load built-in datasets
        """
        if dataset_name.lower() == 'mnist':
            import tensorflow as tf
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            
            # Combine train and test
            images = np.concatenate([x_train, x_test])
            labels = np.concatenate([y_train, y_test])
            
            # Convert to proper format
            if not params.USE_GRAYSCALE:
                images = np.stack([images] * 3, axis=-1)
            else:
                images = np.expand_dims(images, axis=-1)
            
            return images, labels
        else:
            print(f"Unknown builtin dataset: {dataset_name}")
            return np.array([]), np.array([])
    
    def load_folder_structure(self, dataset_path):
        """
        Load dataset from folder structure
        """
        if not os.path.exists(dataset_path):
            print(f"  Dataset path not found: {dataset_path}")
            return np.array([]), np.array([])
        
        images = []
        labels = []
        
        for class_label in range(params.NB_CLASSES):
            class_dir = os.path.join(dataset_path, str(class_label))
            
            if not os.path.exists(class_dir):
                print(f"  Class directory not found: {class_dir}")
                continue
                
            for filename in os.listdir(class_dir):
                if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                    image_path = os.path.join(class_dir, filename)
                    
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    images.append(image)
                    labels.append(class_label)
        
        return np.array(images), np.array(labels)
    
    def load_label_file_dataset(self, dataset_path):
        """
        Load dataset with label file - FIXED VERSION for variable image sizes
        """
        try:
            label_file_path = os.path.join(dataset_path, 'labels.txt')
            images_dir = os.path.join(dataset_path, 'images')
            
            # Validate inputs
            if not os.path.exists(dataset_path):
                print(f"❌ Dataset path does not exist: {dataset_path}")
                return np.array([]), np.array([])
            
            if not os.path.exists(label_file_path):
                print(f"❌ Label file not found: {label_file_path}")
                return np.array([]), np.array([])
                
            if not os.path.exists(images_dir):
                print(f"❌ Images directory not found: {images_dir}")
                return np.array([]), np.array([])
            
            images = []
            labels = []
            skipped_files = 0
            valid_files = 0
            
            print(f"📁 Loading dataset from: {dataset_path}")
            
            with open(label_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                print("⚠️  Label file is empty")
                return np.array([]), np.array([])
            
            print(f"📄 Found {len(lines)} entries in label file")
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                # Split by tab instead of space
                parts = line.split('\t')
                if len(parts) < 2:
                    print(f"⚠️  Line {line_num}: Invalid format (expected: filename label), got: '{line}'")
                    skipped_files += 1
                    continue
                
                filename = parts[0]
                label_str = parts[1]
                
                # Enhanced label validation
                try:
                    label = int(label_str)
                    if label < 0 or label >= params.NB_CLASSES:
                        print(f"⚠️  Line {line_num}: Label {label} out of range [0, {params.NB_CLASSES-1}] for file '{filename}'")
                        skipped_files += 1
                        continue
                except ValueError as e:
                    print(f"⚠️  Line {line_num}: Invalid label '{label_str}' for file '{filename}'. Error: {e}")
                    skipped_files += 1
                    continue
                
                image_path = os.path.join(images_dir, filename)
                
                if not os.path.exists(image_path):
                    print(f"⚠️  Line {line_num}: Image file not found: {image_path}")
                    skipped_files += 1
                    continue
                
                # Load and validate image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if params.USE_GRAYSCALE else cv2.IMREAD_COLOR)
                if image is None:
                    print(f"⚠️  Line {line_num}: Failed to load image: {image_path}")
                    skipped_files += 1
                    continue
                
                if image.size == 0:
                    print(f"⚠️  Line {line_num}: Empty image: {image_path}")
                    skipped_files += 1
                    continue
                
                # RESIZE ALL IMAGES TO CONSISTENT SIZE
                try:
                    image = cv2.resize(image, (params.INPUT_WIDTH, params.INPUT_HEIGHT))
                    
                    # Add channel dimension if needed
                    if params.USE_GRAYSCALE and len(image.shape) == 2:
                        image = np.expand_dims(image, axis=-1)
                    elif not params.USE_GRAYSCALE and len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    
                    images.append(image)
                    labels.append(label)
                    valid_files += 1
                    
                    if valid_files % 1000 == 0:
                        print(f"  Loaded {valid_files} images...")
                        
                except Exception as resize_error:
                    print(f"⚠️  Line {line_num}: Failed to resize image {image_path}: {resize_error}")
                    skipped_files += 1
                    continue
            
            if valid_files == 0:
                print("❌ No valid images were loaded")
                return np.array([]), np.array([])
            
            # Convert to numpy arrays with explicit dtype and shape validation
            try:
                images_array = np.array(images, dtype=np.uint8)
                labels_array = np.array(labels, dtype=np.int32)
                
                print(f"✅ Successfully loaded {valid_files} images, {skipped_files} files skipped")
                print(f"📊 Dataset shape: Images {images_array.shape}, Labels {labels_array.shape}")
                
                return images_array, labels_array
                
            except Exception as array_error:
                print(f"❌ Failed to create numpy arrays: {array_error}")
                print(f"   Image shapes: {[img.shape for img in images[:10]]}")  # Debug first 10 images
                return np.array([]), np.array([])
                
        except Exception as e:
            print(f"💥 Unexpected error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([])
    
    
    def load_mnist_fallback(self):
        """
        Fallback to MNIST if no sources work
        """
        return self.load_builtin_dataset('mnist')
    
    def get_class_distribution(self, labels):
        """
        Get distribution of classes
        """
        distribution = {}
        for i in range(params.NB_CLASSES):
            count = np.sum(labels == i)
            if count > 0:
                distribution[i] = count
        return distribution
    
    def print_detailed_stats(self):
        """
        Print detailed statistics about loaded data
        """
        print("\n" + "=" * 50)
        print("DATA SOURCE STATISTICS")
        print("=" * 50)
        
        for source_name, stats in self.source_stats.items():
            print(f"\nSource: {source_name}")
            print(f"  Total images: {stats['count']}")
            print(f"  Class distribution:")
            for class_id, count in stats['class_distribution'].items():
                print(f"    Class {class_id}: {count} images")
        
        # Combined statistics
        if len(self.all_images) > 0:
            combined_images = np.concatenate(self.all_images, axis=0)
            combined_labels = np.concatenate(self.all_labels, axis=0)
            
            print(f"\nCOMBINED DATASET:")
            print(f"  Total images: {len(combined_images)}")
            print(f"  Class distribution:")
            for i in range(params.NB_CLASSES):
                count = np.sum(combined_labels == i)
                percentage = (count / len(combined_labels)) * 100
                print(f"    Class {i}: {count} images ({percentage:.1f}%)")

def shuffle_dataset(images, labels, seed=params.SHUFFLE_SEED):
    """
    Shuffle images and labels together
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(images))
    return images[indices], labels[indices]

def load_combined_dataset():
    """
    Main function to load all data sources
    """
    loader = MultiSourceDataLoader()
    images, labels = loader.load_all_sources()
    loader.print_detailed_stats()
    
    # Shuffle the combined dataset
    images, labels = shuffle_dataset(images, labels)
    
    return images, labels

def get_data_splits():
    """
    Get train/validation/test splits from combined data sources
    """
    # Load and combine all data sources
    images, labels = load_combined_dataset()
    
    # Use specified percentage of data
    total_samples = len(images)
    training_samples = int(total_samples * params.TRAINING_PERCENTAGE)
    
    # Take the first N samples (already shuffled)
    images = images[:training_samples]
    labels = labels[:training_samples]
    
    print(f"\nUsing {len(images)} samples ({params.TRAINING_PERCENTAGE*100}% of available)")
    
    # Split into train+val and test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images, labels, 
        test_size=0.2, 
        random_state=params.SHUFFLE_SEED,
        shuffle=True,
        stratify=labels
    )
    
    # Further split train+val into train and val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, 
        test_size=params.VALIDATION_SPLIT, 
        random_state=params.SHUFFLE_SEED,
        shuffle=True,
        stratify=y_train_val
    )
    
    print(f"\nFinal Data Splits:")
    print(f"  Training: {len(x_train)} samples")
    print(f"  Validation: {len(x_val)} samples")
    print(f"  Test: {len(x_test)} samples")
    
    # Print final class distribution
    print(f"\nFinal Class Distribution:")
    for split_name, x, y in [("Training", x_train, y_train), 
                            ("Validation", x_val, y_val), 
                            ("Test", x_test, y_test)]:
        print(f"  {split_name}:")
        for i in range(params.NB_CLASSES):
            count = np.sum(y == i)
            if count > 0:
                percentage = (count / len(y)) * 100
                print(f"    Class {i}: {count} ({percentage:.1f}%)")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)