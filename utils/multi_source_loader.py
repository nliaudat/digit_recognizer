# multi_source_loader.py
import os
import cv2
import hashlib
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import parameters as params
from utils.logging import log_print

# Global variable to cache loaded data (in-memory, single process)
_loaded_data = None

# ---------------------------------------------------------------------------
# Disk-level NPZ cache helpers
# ---------------------------------------------------------------------------

def _cache_key(source_configs: list) -> str:
    """Deterministic cache key from source paths + label files + nb_classes."""
    fingerprint = json.dumps(
        [{k: v for k, v in s.items()} for s in source_configs],
        sort_keys=True,
    ) + f"|nb_classes={params.NB_CLASSES}|grayscale={params.USE_GRAYSCALE}"
    return hashlib.md5(fingerprint.encode()).hexdigest()[:12]


def _cache_path(source_configs: list) -> str:
    """Return the .npz file path for this configuration."""
    cache_dir = getattr(params, "DATASET_CACHE_DIR", ".dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"dataset_{_cache_key(source_configs)}.npz")


def _load_cache(source_configs: list):
    """Return (images, labels) from disk cache if valid, else None."""
    path = _cache_path(source_configs)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        images, labels = data["images"], data["labels"]
        print(f"⚡ Loaded dataset from disk cache: {path} "
              f"({len(images)} images, {os.path.getsize(path) / 1e6:.1f} MB)")
        return images, labels
    except Exception as e:
        print(f"⚠️  Disk cache corrupted ({e}), rebuilding…")
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _save_cache(source_configs: list, images: np.ndarray, labels: np.ndarray):
    """Persist images + labels to disk as a compressed NPZ file."""
    path = _cache_path(source_configs)
    try:
        np.savez_compressed(path, images=images, labels=labels)
        print(f"💾 Dataset cached to disk: {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"⚠️  Could not save disk cache: {e}")

class MultiSourceDataLoader:
    def __init__(self):
        self.all_images = []
        self.all_labels = []
        self.source_stats = {}
    
    def load_all_sources(self):
        """
        Load and combine all data sources
        """
        global _loaded_data
        # Return cached data if available
        if _loaded_data is not None:
            print("📊 Using cached dataset...")
            return _loaded_data
        
        print("Loading multiple data sources...")
        print("=" * 50)
        
        for source_config in params.DATA_SOURCES:
            source_name = source_config['name']
            source_type = source_config['type']
            source_path = source_config['path']
            source_weight = source_config.get('weight', 1.0)
            
            log_print(f"Loading source: {source_name} (type: {source_type})", level=2)
            
            if source_type == 'builtin':
                images, labels = self.load_builtin_dataset(source_name)
            elif source_type == 'folder_structure':
                images, labels = self.load_folder_structure(source_path)
            elif source_type == 'label_file':
                # Get optional labels file name, default to 'labels.txt'
                labels_file = source_config.get('labels', 'labels.txt')
                images, labels = self.load_label_file_dataset(source_path, labels_file)
            else:
                log_print(f"Unknown source type: {source_type}, skipping...", level=1)
                continue
            
            if len(images) == 0:
                log_print(f"  No data loaded from {source_name}, skipping...", level=1)
                continue
            
            # Apply sampling weight (undersample if weight < 1.0)
            if source_weight < 1.0 and len(images) > 0:
                sample_size = int(len(images) * source_weight)
                indices = np.random.choice(len(images), sample_size, replace=False)
                images = images[indices]
                labels = labels[indices]
                log_print(f"  Sampled {sample_size} images (weight: {source_weight})", level=2)
            
            # Store source statistics
            self.source_stats[source_name] = {
                'count': len(images),
                'class_distribution': self.get_class_distribution(labels)
            }
            
            # Add to combined dataset
            self.all_images.append(images)
            self.all_labels.append(labels)
            
            log_print(f"  Loaded {len(images)} images", level=2)
            log_print(f"  Class distribution: {self.source_stats[source_name]['class_distribution']}", level=2)
            log_print("-" * 30, level=2)
        
        if len(self.all_images) == 0:
            log_print("No data sources could be loaded.", level=1)
            images, labels = self.load_mnist_fallback()
        else:
            # Combine all sources
            images = np.concatenate(self.all_images, axis=0)
            labels = np.concatenate(self.all_labels, axis=0)
        
        # Cache the loaded data
        _loaded_data = (images, labels)
        log_print(f"\nCombined dataset:", level=2)
        log_print(f"  Total images: {len(images)}", level=2)
        log_print(f"  Sources: {list(self.source_stats.keys())}", level=2)
        return images, labels
    

    def load_builtin_dataset(self, dataset_name):
        """Load built-in datasets"""
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
    
    def _decode_image(self, image_path: str, label: int, target_h: int, target_w: int,
                      grayscale: bool):
        """Decode and resize a single image. Returns (image, label) or None on error."""
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(image_path, flag)
        if img is None or img.size == 0:
            return None
        img = cv2.resize(img, (target_w, target_h))
        if grayscale and img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        elif not grayscale and img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img, label

    def _parallel_load(self, tasks: list, desc: str = "") -> tuple:
        """Run _decode_image in parallel. tasks = list of (path, label)."""
        h, w = params.INPUT_HEIGHT, params.INPUT_WIDTH
        grayscale = params.USE_GRAYSCALE
        n_workers = min(8, os.cpu_count() or 4)

        images, labels = [], []
        errors = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(self._decode_image, p, lbl, h, w, grayscale): (p, lbl)
                for p, lbl in tasks
            }
            done = 0
            total = len(futures)
            report_every = max(1, total // 10)
            for f in as_completed(futures):
                result = f.result()
                done += 1
                if result is None:
                    errors += 1
                else:
                    images.append(result[0])
                    labels.append(result[1])
                if done % report_every == 0:
                    print(f"  {desc}: {done}/{total} loaded…", flush=True)
        if errors:
            print(f"  ⚠️  {errors} images could not be loaded and were skipped.")
        return images, labels

    def load_folder_structure(self, dataset_path):
        """Load dataset from folder structure using parallel decoding."""
        if not os.path.exists(dataset_path):
            print(f"  Dataset path not found: {dataset_path}")
            return np.array([]), np.array([])

        tasks = []
        for class_label in range(params.NB_CLASSES):
            class_dir = os.path.join(dataset_path, str(class_label))
            if not os.path.exists(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if any(filename.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp')):
                    tasks.append((os.path.join(class_dir, filename), class_label))

        if not tasks:
            return np.array([]), np.array([])

        print(f"  📂 Loading {len(tasks)} images in parallel from {dataset_path}")
        images, labels = self._parallel_load(tasks, desc=os.path.basename(dataset_path))
        if not images:
            return np.array([]), np.array([])
        return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)
    
    def load_label_file_dataset(self, dataset_path, labels_file='labels.txt'):
        """Load dataset from a label file using parallel image decoding."""
        label_file_path = os.path.join(dataset_path, labels_file)
        images_dir = os.path.join(dataset_path, 'images')

        for path, desc in [
            (dataset_path, "Dataset path"),
            (label_file_path, "Label file"),
            (images_dir, "Images directory"),
        ]:
            if not os.path.exists(path):
                print(f"❌ {desc} does not exist: {path}")
                return np.array([]), np.array([])

        print(f"📁 Loading dataset from: {dataset_path}")
        print(f"📄 Using label file: {labels_file}")

        with open(label_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print("⚠️  Label file is empty")
            return np.array([]), np.array([])

        # Parse the label file first (fast, no I/O)
        tasks = []
        skipped = 0
        for line_num, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 2:
                skipped += 1
                continue
            filename, label_str = parts[0], parts[1]
            try:
                label = int(label_str)
            except ValueError:
                skipped += 1
                continue
            if label < 0 or label >= params.NB_CLASSES:
                skipped += 1
                continue
            image_path = os.path.join(images_dir, filename)
            if not os.path.exists(image_path):
                skipped += 1
                continue
            tasks.append((image_path, label))

        print(f"📄 {len(tasks)} valid entries found ({skipped} skipped in label file)")
        if not tasks:
            return np.array([]), np.array([])

        # Parallel decode
        t0 = time.time()
        images, labels = self._parallel_load(tasks, desc=os.path.basename(dataset_path))
        elapsed = time.time() - t0

        if not images:
            print("❌ No valid images were loaded")
            return np.array([]), np.array([])

        images_array = np.array(images, dtype=np.uint8)
        labels_array = np.array(labels, dtype=np.int32)
        print(f"✅ Loaded {len(images_array)} images in {elapsed:.1f}s "
              f"(shape: {images_array.shape})")
        return images_array, labels_array
    
    def load_mnist_fallback(self):
        """Fallback to MNIST if no sources work"""
        return self.load_builtin_dataset('mnist')
    
    def get_class_distribution(self, labels):
        """Get distribution of classes"""
        distribution = {}
        for i in range(params.NB_CLASSES):
            count = np.sum(labels == i)
            if count > 0:
                distribution[i] = count
        return distribution
    
    def print_detailed_stats(self):
        """Print detailed statistics about loaded data"""
        print("\n" + "=" * 50)
        print("DATA SOURCE STATISTICS")
        print("=" * 50)
        
        for source_name, stats in self.source_stats.items():
            print(f"\nSource: {source_name}")
            print(f"  Total images: {stats['count']}")
            print(f"  Class distribution:")
            for class_id, count in stats['class_distribution'].items():
                print(f"    Class {class_id}: {count} images")
        
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
    """Shuffle images and labels together"""
    np.random.seed(seed)
    indices = np.random.permutation(len(images))
    return images[indices], labels[indices]

def load_combined_dataset():
    """Main function to load all data sources.

    Load order:
      1. In-memory cache  (same process, e.g. train_all.py loops)
      2. Disk NPZ cache   (cross-run, survives Docker restarts)
      3. Full parallel decode from source files
    """
    global _loaded_data
    if _loaded_data is not None:
        print("📊 Using cached dataset...")
        return _loaded_data

    # Try disk cache
    cached = _load_cache(params.DATA_SOURCES)
    if cached is not None:
        images, labels = cached
        images, labels = shuffle_dataset(images, labels)
        _loaded_data = (images, labels)
        return images, labels

    # Full load
    t0 = time.time()
    loader = MultiSourceDataLoader()
    images, labels = loader.load_all_sources()
    loader.print_detailed_stats()
    print(f"⏱  Total source-load time: {time.time() - t0:.1f}s")

    images, labels = shuffle_dataset(images, labels)
    _save_cache(params.DATA_SOURCES, images, labels)
    _loaded_data = (images, labels)
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

def clear_cache():
    """Clear the cached dataset"""
    global _loaded_data
    _loaded_data = None
    print("🧹 Cleared dataset cache")
