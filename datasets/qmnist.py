### https://github.com/facebookresearch/qmnist


import numpy as np
import os
import cv2
import json
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm
import requests
import gzip
import pickle
import torch
from torchvision.datasets import QMNIST

# Configuration
INPUT_WIDTH = 32
INPUT_HEIGHT = 20
INPUT_CHANNELS = 1
QMNIST_FOLDER_DIR = "qmnist_dataset_folders"

def download_qmnist():
    """Download QMNIST dataset using PyTorch"""
    print("📥 Downloading QMNIST dataset...")
    
    try:
        # Download QMNIST train set
        print("   Downloading training set...")
        qmnist_train = QMNIST(root='./qmnist_data', what='train', download=True)
        
        # Download QMNIST test set  
        print("   Downloading test set...")
        qmnist_test = QMNIST(root='./qmnist_data', what='test', download=True)
        
        print(f"✅ QMNIST dataset downloaded:")
        print(f"   Training: {len(qmnist_train)} samples")
        print(f"   Test: {len(qmnist_test)} samples")
        
        return qmnist_train, qmnist_test
        
    except Exception as e:
        print(f"❌ Error downloading QMNIST: {e}")
        print("💡 Make sure you have torch and torchvision installed:")
        print("   pip install torch torchvision")
        return None, None

def process_qmnist_data(qmnist_train, qmnist_test):
    """Process QMNIST data and convert to numpy arrays"""
    print("🔄 Processing QMNIST data...")
    
    def extract_data(qmnist_dataset):
        """Extract images and labels from QMNIST dataset"""
        images = []
        labels = []
        
        for i in tqdm(range(len(qmnist_dataset)), desc="Extracting data"):
            image, label = qmnist_dataset[i]
            
            # Convert PIL image to numpy array
            image_np = np.array(image)
            
            # QMNIST returns a tuple (label, writer_id) - we just want the label
            if isinstance(label, (list, tuple)):
                label = label[0]  # Take the first element (digit label)
            
            images.append(image_np)
            labels.append(label)
        
        return np.array(images), np.array(labels)
    
    # Extract training data
    x_train, y_train = extract_data(qmnist_train)
    x_test, y_test = extract_data(qmnist_test)
    
    print(f"✅ QMNIST data processed:")
    print(f"   Training: {x_train.shape} -> {y_train.shape}")
    print(f"   Test: {x_test.shape} -> {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def preprocess_images(images, labels, description=""):
    """Preprocess images (resize and normalize)"""
    print(f"🔄 Preprocessing {description} images...")
    
    processed_images = []
    
    for img in tqdm(images, desc=f"Processing {description}"):
        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
        
        # Resize to target shape (20x32)
        img_resized = cv2.resize(img_normalized, (INPUT_WIDTH, INPUT_HEIGHT))
        
        # Add channel dimension
        if len(img_resized.shape) == 2:
            img_resized = np.expand_dims(img_resized, axis=-1)
        
        processed_images.append(img_resized)
    
    processed_images = np.array(processed_images)
    
    print(f"   Processed shape: {processed_images.shape}")
    print(f"   Data range: [{processed_images.min():.3f}, {processed_images.max():.3f}]")
    
    return processed_images, labels

def create_qmnist_folder_structure(x_train, y_train, x_test, y_test):
    """Create flat folder structure for QMNIST"""
    print("🔄 Creating QMNIST flat folder structure...")
    print(f"   Folder structure: Flat (no train/test subfolders)")
    
    # Remove existing folder structure
    if os.path.exists(QMNIST_FOLDER_DIR):
        shutil.rmtree(QMNIST_FOLDER_DIR)
    
    # Create digit folders (0-9) directly in the main directory
    for digit in range(10):
        digit_dir = os.path.join(QMNIST_FOLDER_DIR, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
    
    # Combine train and test data for flat structure
    all_images = np.concatenate([x_train, x_test], axis=0)
    all_labels = np.concatenate([y_train, y_test], axis=0)
    
    print(f"📊 Combined data for flat structure:")
    print(f"   Total images: {len(all_images)}")
    print(f"   Total labels: {len(all_labels)}")
    
    # Save images to respective folders
    for i in tqdm(range(len(all_images)), desc="Saving QMNIST images"):
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
        filename = f"qmnist_{source}_{original_idx:05d}.png"
        filepath = os.path.join(QMNIST_FOLDER_DIR, str(label), filename)
        cv2.imwrite(filepath, save_image)
    
    print(f"💾 Saved {len(all_images)} images to {QMNIST_FOLDER_DIR}")
    print(f"✅ QMNIST flat folder structure created: {QMNIST_FOLDER_DIR}")
    
    # Verify the flat structure
    print(f"📁 QMNIST folder structure created:")
    total_images = 0
    for digit in range(10):
        digit_dir = os.path.join(QMNIST_FOLDER_DIR, str(digit))
        if os.path.exists(digit_dir):
            num_files = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
            total_images += num_files
            print(f"   {digit}/: {num_files} images")
    print(f"   Total: {total_images} images")
    
    return True

def verify_qmnist_data():
    """Verify QMNIST data"""
    print("🔍 Verifying QMNIST data...")
    
    # Check folder structure
    folder_structure_exists = os.path.exists(QMNIST_FOLDER_DIR)
    
    print(f"📊 QMNIST Data Status:")
    print(f"   Folder structure: {'✅' if folder_structure_exists else '❌'}")
    print(f"   Structure: Flat folders")
    
    if folder_structure_exists:
        total_images = 0
        for digit in range(10):
            digit_dir = os.path.join(QMNIST_FOLDER_DIR, str(digit))
            if os.path.exists(digit_dir):
                num_files = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
                total_images += num_files
        print(f"   Total images: {total_images}")
        
        # Show distribution
        print(f"   Class distribution:")
        for digit in range(10):
            digit_dir = os.path.join(QMNIST_FOLDER_DIR, str(digit))
            if os.path.exists(digit_dir):
                num_files = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
                print(f"     {digit}: {num_files} images")
        
        # Show sample images info
        print(f"🎨 Sample images info:")
        for digit in range(3):  # Show first 3 digits
            digit_dir = os.path.join(QMNIST_FOLDER_DIR, str(digit))
            if os.path.exists(digit_dir):
                sample_files = sorted(os.listdir(digit_dir))[:2]
                for sample_file in sample_files:
                    sample_path = os.path.join(digit_dir, sample_file)
                    sample_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
                    print(f"   {digit}/{sample_file}: shape {sample_img.shape}, range [{sample_img.min()}, {sample_img.max()}]")

def create_data_sources_snippet():
    """Create DATA_SOURCES snippet for parameters.py"""
    snippet = f"""
# QMNIST Dataset (added by qmnist.py)
{{
    'name': 'QMNIST',
    'type': 'folder_structure',
    'path': '{QMNIST_FOLDER_DIR}',
    'weight': 0.3,
}},
"""
    
    print("📋 Add this to your DATA_SOURCES in parameters.py:")
    print("=" * 60)
    print(snippet)
    print("=" * 60)
    
    # Also save to file
    with open("qmnist_data_source_snippet.txt", "w") as f:
        f.write(snippet)
    
    print(f"💾 Snippet also saved to: qmnist_data_source_snippet.txt")

def main():
    """Main function to run QMNIST setup"""
    parser = argparse.ArgumentParser(description='QMNIST Dataset Setup')
    parser.add_argument('--download', action='store_true', help='Download and prepare QMNIST dataset')
    parser.add_argument('--create-folders', action='store_true', help='Create flat folder structure')
    parser.add_argument('--verify', action='store_true', help='Verify QMNIST data')
    parser.add_argument('--create-snippet', action='store_true', help='Create DATA_SOURCES snippet')
    parser.add_argument('--all', action='store_true', help='Run complete setup')
    
    args = parser.parse_args()
    
    print("🚀 QMNIST Dataset Setup")
    print("=" * 50)
    
    if args.download or args.all:
        # Download QMNIST
        qmnist_train, qmnist_test = download_qmnist()
        if qmnist_train is None:
            return
        
        # Process data
        (x_train, y_train), (x_test, y_test) = process_qmnist_data(qmnist_train, qmnist_test)
        
        # Preprocess images
        x_train_proc, y_train_proc = preprocess_images(x_train, y_train, "training")
        x_test_proc, y_test_proc = preprocess_images(x_test, y_test, "test")
    
    if args.create_folders or args.all:
        if 'x_train_proc' not in locals():
            print("❌ No processed data found. Run download first.")
            return
        create_qmnist_folder_structure(x_train_proc, y_train_proc, x_test_proc, y_test_proc)
    
    if args.verify or args.all:
        verify_qmnist_data()
    
    if args.create_snippet or args.all:
        create_data_sources_snippet()
    
    # If no arguments provided, run complete setup
    if not any([args.download, args.create_folders, args.verify, args.create_snippet, args.all]):
        print("No specific command provided. Running complete setup...")
        print("🔄 Running complete QMNIST setup...")
        
        # Download QMNIST
        qmnist_train, qmnist_test = download_qmnist()
        if qmnist_train is None:
            return
        
        # Process data
        (x_train, y_train), (x_test, y_test) = process_qmnist_data(qmnist_train, qmnist_test)
        
        # Preprocess images
        x_train_proc, y_train_proc = preprocess_images(x_train, y_train, "training")
        x_test_proc, y_test_proc = preprocess_images(x_test, y_test, "test")
        
        # Create folder structure
        if create_qmnist_folder_structure(x_train_proc, y_train_proc, x_test_proc, y_test_proc):
            verify_qmnist_data()
            create_data_sources_snippet()
        
        print("✅ QMNIST setup complete!")

if __name__ == "__main__":
    main()