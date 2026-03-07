import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.datasets import mnist
import json
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm

# Configuration
INPUT_WIDTH = 32
INPUT_HEIGHT = 20
INPUT_CHANNELS = 1
MNIST_NPY_DIR = "mnist_dataset"
MNIST_FOLDER_DIR = "mnist_dataset_folders"

def download_and_prepare_mnist():
    """Download MNIST dataset and prepare it for DATA_SOURCES"""
    print("📥 Downloading MNIST dataset...")
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"✅ MNIST dataset downloaded:")
    print(f"   Training: {x_train.shape} {y_train.shape}")
    print(f"   Test: {x_test.shape} {y_test.shape}")
    
    # Create MNIST directory for numpy files
    os.makedirs(MNIST_NPY_DIR, exist_ok=True)
    
    def preprocess_images(images, labels, split_name):
        """Preprocess and save images as numpy files"""
        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        # Resize to target shape (20x32)
        resized_images = []
        for img in tqdm(images, desc=f"Resizing {split_name}"):
            resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
            resized_images.append(resized)
        
        resized_images = np.array(resized_images)
        
        # Add channel dimension
        if len(resized_images.shape) == 3:
            resized_images = np.expand_dims(resized_images, axis=-1)
        
        # Save as numpy files
        np.save(os.path.join(MNIST_NPY_DIR, f"x_{split_name}.npy"), resized_images)
        np.save(os.path.join(MNIST_NPY_DIR, f"y_{split_name}.npy"), labels)
        
        print(f"💾 Saved {split_name}: {resized_images.shape} {labels.shape}")
        return resized_images, labels
    
    # Process both splits
    x_train_proc, y_train_proc = preprocess_images(x_train, y_train, "train")
    x_test_proc, y_test_proc = preprocess_images(x_test, y_test, "test")
    
    # Create dataset info
    dataset_info = {
        "name": "MNIST",
        "type": "classification",
        "description": "Modified National Institute of Standards and Technology database",
        "num_classes": 10,
        "input_shape": [INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
        "samples": {
            "train": len(x_train_proc),
            "test": len(x_test_proc)
        },
        "files": {
            "x_train": f"{MNIST_NPY_DIR}/x_train.npy",
            "y_train": f"{MNIST_NPY_DIR}/y_train.npy", 
            "x_test": f"{MNIST_NPY_DIR}/x_test.npy",
            "y_test": f"{MNIST_NPY_DIR}/y_test.npy"
        }
    }
    
    # Save dataset info
    with open(os.path.join(MNIST_NPY_DIR, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"📊 Dataset info saved to: {MNIST_NPY_DIR}/dataset_info.json")
    
    return dataset_info

def create_mnist_folder_structure():
    """Convert MNIST numpy files to flat folder structure for DATA_SOURCES"""
    print("🔄 Creating MNIST flat folder structure...")
    print(f"   Folder structure: Flat (no train/test subfolders)")
    
    # Load MNIST data from .npy files
    try:
        x_train = np.load(f'{MNIST_NPY_DIR}/x_train.npy')
        y_train = np.load(f'{MNIST_NPY_DIR}/y_train.npy')
        x_test = np.load(f'{MNIST_NPY_DIR}/x_test.npy')
        y_test = np.load(f'{MNIST_NPY_DIR}/y_test.npy')
        
        print(f"✅ Loaded MNIST data:")
        print(f"   Training: {x_train.shape} -> {y_train.shape}")
        print(f"   Test: {x_test.shape} -> {y_test.shape}")
        print(f"   Data range: [{x_train.min():.3f}, {x_train.max():.3f}]")
        
    except FileNotFoundError as e:
        print(f"❌ MNIST numpy files not found. Run download first.")
        print(f"📁 Looking in: {MNIST_NPY_DIR}/")
        if os.path.exists(MNIST_NPY_DIR):
            print("Files found:")
            for f in os.listdir(MNIST_NPY_DIR):
                print(f"   - {f}")
        return False
    
    # Remove existing folder structure
    if os.path.exists(MNIST_FOLDER_DIR):
        shutil.rmtree(MNIST_FOLDER_DIR)
    
    # Create flat folder structure - no train/test subfolders
    # Create digit folders (0-9) directly in the main directory
    for digit in range(10):
        digit_dir = os.path.join(MNIST_FOLDER_DIR, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
    
    # Combine train and test data for flat structure
    all_images = np.concatenate([x_train, x_test], axis=0)
    all_labels = np.concatenate([y_train, y_test], axis=0)
    
    print(f"📊 Combined data for flat structure:")
    print(f"   Total images: {len(all_images)}")
    print(f"   Total labels: {len(all_labels)}")
    
    # Save images to respective folders
    for i in tqdm(range(len(all_images)), desc="Saving images to flat structure"):
        image = all_images[i]
        label = int(all_labels[i])
        
        # Convert to 0-255 range for saving
        if image.max() <= 1.0:
            save_image = (image * 255).astype(np.uint8)
        else:
            save_image = image.astype(np.uint8)
        
        # Remove channel dimension if single channel
        if len(save_image.shape) == 3 and save_image.shape[2] == 1:
            save_image = save_image[:, :, 0]
        
        # Create filename (include source: train or test)
        source = "train" if i < len(x_train) else "test"
        original_idx = i if i < len(x_train) else i - len(x_train)
        filename = f"mnist_{source}_{original_idx:05d}.png"
        filepath = os.path.join(MNIST_FOLDER_DIR, str(label), filename)
        cv2.imwrite(filepath, save_image)
    
    print(f"💾 Saved {len(all_images)} images to {MNIST_FOLDER_DIR}")
    print(f"✅ MNIST flat folder structure created: {MNIST_FOLDER_DIR}")
    
    # Verify the flat structure
    print(f"📁 Flat folder structure created:")
    total_images = 0
    for digit in range(10):
        digit_dir = os.path.join(MNIST_FOLDER_DIR, str(digit))
        if os.path.exists(digit_dir):
            num_files = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
            total_images += num_files
            print(f"   {digit}/: {num_files} images")
    print(f"   Total: {total_images} images")
    
    return True

def verify_mnist_data():
    """Verify MNIST data in both formats"""
    print("🔍 Verifying MNIST data...")
    
    # Check numpy files
    npy_files_exist = all([
        os.path.exists(f'{MNIST_NPY_DIR}/x_train.npy'),
        os.path.exists(f'{MNIST_NPY_DIR}/y_train.npy'),
        os.path.exists(f'{MNIST_NPY_DIR}/x_test.npy'),
        os.path.exists(f'{MNIST_NPY_DIR}/y_test.npy')
    ])
    
    # Check folder structure
    folder_structure_exists = os.path.exists(MNIST_FOLDER_DIR)
    
    print(f"📊 MNIST Data Status:")
    print(f"   NumPy files: {'✅' if npy_files_exist else '❌'}")
    print(f"   Folder structure: {'✅' if folder_structure_exists else '❌'}")
    print(f"   Structure: Flat folders")
    
    if npy_files_exist:
        try:
            x_train = np.load(f'{MNIST_NPY_DIR}/x_train.npy')
            y_train = np.load(f'{MNIST_NPY_DIR}/y_train.npy')
            print(f"   NumPy - Training: {x_train.shape} {y_train.shape}")
            print(f"   NumPy - Data range: [{x_train.min():.3f}, {x_train.max():.3f}]")
        except:
            pass
    
    if folder_structure_exists:
        total_images = 0
        for digit in range(10):
            digit_dir = os.path.join(MNIST_FOLDER_DIR, str(digit))
            if os.path.exists(digit_dir):
                num_files = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
                total_images += num_files
        print(f"   Folder - Total images: {total_images}")
        
        # Show distribution
        print(f"   Folder - Class distribution:")
        for digit in range(10):
            digit_dir = os.path.join(MNIST_FOLDER_DIR, str(digit))
            if os.path.exists(digit_dir):
                num_files = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
                print(f"     {digit}: {num_files} images")

def create_data_sources_snippet():
    """Create DATA_SOURCES snippet for parameters.py"""
    snippet = f"""
# MNIST Dataset (added by mnist.py)
{{
    'name': 'MNIST',
    'type': 'folder_structure',
    'path': '{MNIST_FOLDER_DIR}',
    'weight': 0.2,
}},
"""
    
    print("📋 Add this to your DATA_SOURCES in parameters.py:")
    print("=" * 60)
    print(snippet)
    print("=" * 60)
    
    # Also save to file
    with open("mnist_data_source_snippet.txt", "w") as f:
        f.write(snippet)
    
    print(f"💾 Snippet also saved to: mnist_data_source_snippet.txt")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MNIST Dataset Setup')
    parser.add_argument('--download', action='store_true', help='Download and prepare MNIST dataset as numpy files')
    parser.add_argument('--create-folders', action='store_true', help='Create flat folder structure from numpy files')
    parser.add_argument('--verify', action='store_true', help='Verify MNIST data')
    parser.add_argument('--create-snippet', action='store_true', help='Create DATA_SOURCES snippet')
    parser.add_argument('--all', action='store_true', help='Run complete setup (download + create folders + snippet)')
    
    args = parser.parse_args()
    
    print("🚀 MNIST Dataset Setup")
    print("=" * 50)
    
    if args.download:
        download_and_prepare_mnist()
    
    if args.create_folders:
        create_mnist_folder_structure()
    
    if args.verify:
        verify_mnist_data()
    
    if args.create_snippet:
        create_data_sources_snippet()
    
    if args.all:
        print("🔄 Running complete MNIST setup...")
        download_and_prepare_mnist()
        if create_mnist_folder_structure():
            verify_mnist_data()
            create_data_sources_snippet()
        print("✅ MNIST setup complete!")
    
    # If no arguments provided, run complete setup
    if not any([args.download, args.create_folders, args.verify, args.create_snippet, args.all]):
        print("No specific command provided. Running complete setup...")
        print("🔄 Running complete MNIST setup...")
        download_and_prepare_mnist()
        if create_mnist_folder_structure():
            verify_mnist_data()
            create_data_sources_snippet()
        print("✅ MNIST setup complete!")