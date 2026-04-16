#!/usr/bin/env python3
"""
static_mixup_augmentation.py - Pure Mixup for Continuous Regression Labels (0.0-9.9)
Labels extracted from filename pattern: X.Y_*.jpg where X.Y is the label
Example: 9.1_1756145124.jpg → label = 9.1
Handles circular boundary (9.9 ↔ 0.0)
"""

import numpy as np
import os
import cv2
import random
import json
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import re
import glob

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CONFIG = {
    "image_params": {
        "width": 20,
        "height": 32,
        "channels": 3,
        "grayscale": False
    },
    "dataset": {
        "input_dir": "../real_integra/images",
        "output_dir": "../real_integra/images_mixup_only",
        "label_pattern": r"^(\d+\.\d+)_",
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp"]
    },
    "mixup": {
        "enabled": True,
        "alpha": 0.2,
        "max_label_distance": 0.2,
        "circular": True,
        "variants_per_image": 2,
        "max_label": 9.9,
        "min_label": 0.0
    },
    "processing": {
        "threads": 4,
        "save_images": True,
        "random_seed": 42
    }
}

# ==============================================================================
# CIRCULAR LABEL UTILITIES
# ==============================================================================

class CircularLabelUtils:
    @staticmethod
    def distance(l1, l2, max_label=9.9):
        """Circular distance with wrap-around (9.9 ↔ 0.0)"""
        step = 0.1
        max_val = max_label + step
        diff = abs(l1 - l2)
        return min(diff, max_val - diff)
    
    @staticmethod
    def are_close(l1, l2, max_distance=0.2, circular=True, max_label=9.9):
        if circular:
            return CircularLabelUtils.distance(l1, l2, max_label) <= max_distance
        return abs(l1 - l2) <= max_distance
    
    @staticmethod
    def mix(l1, l2, lam, circular=True, max_label=9.9, min_label=0.0):
        """Mix two continuous labels with circular wrap handling"""
        if not circular or abs(l1 - l2) <= 5.0:
            mixed = lam * l1 + (1 - lam) * l2
        else:
            if l1 < l2:
                l1_norm = l1 + 10.0
                l2_norm = l2
            else:
                l1_norm = l1
                l2_norm = l2 + 10.0
            mixed = lam * l1_norm + (1 - lam) * l2_norm
            mixed = mixed % 10.0
        
        mixed = round(mixed * 10) / 10.0
        return max(min_label, min(mixed, max_label))

# ==============================================================================
# PURE MIXUP AUGMENTOR
# ==============================================================================

class PureMixupAugmentor:
    def __init__(self, config):
        self.config = config
        self.mix_config = config["mixup"]
        self.alpha = self.mix_config["alpha"]
        self.max_distance = self.mix_config["max_label_distance"]
        self.circular = self.mix_config["circular"]
        self.max_label = self.mix_config["max_label"]
        self.min_label = self.mix_config["min_label"]
        
        self.width = config["image_params"]["width"]
        self.height = config["image_params"]["height"]
        self.channels = config["image_params"]["channels"]
        self.grayscale = config["image_params"]["grayscale"]
    
    def ensure_shape(self, img):
        if len(img.shape) == 2 and not self.grayscale:
            img = np.stack([img] * 3, axis=-1)
        elif len(img.shape) == 3 and self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
        elif len(img.shape) == 2 and self.grayscale:
            img = np.expand_dims(img, axis=-1)
        
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        
        return img
    
    def mix_pair(self, img1, label1, img2, label2):
        if not CircularLabelUtils.are_close(label1, label2, self.max_distance, 
                                             self.circular, self.max_label):
            return None, None
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_img = lam * img1 + (1 - lam) * img2
        mixed_img = np.clip(mixed_img, 0, 1)
        
        mixed_label = CircularLabelUtils.mix(label1, label2, lam, 
                                              self.circular, self.max_label, self.min_label)
        
        return mixed_img, mixed_label

# ==============================================================================
# MAIN AUGMENTATION CLASS
# ==============================================================================

class StaticMixupGenerator:
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Loaded config: {config_path}")
        else:
            self.config = DEFAULT_CONFIG
            print(f"⚠️  Using default config")
        
        self.mixup = PureMixupAugmentor(self.config)
        
        self.input_dir = self.config["dataset"]["input_dir"]
        self.output_dir = self.config["dataset"]["output_dir"]
        self.label_pattern = self.config["dataset"].get("label_pattern", r"^(\d+\.\d+)_")
        self.image_extensions = self.config["dataset"].get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp"])
        
        self.threads = self.config["processing"]["threads"]
        self.save_images = self.config["processing"]["save_images"]
        self.variants_per_image = self.config["mixup"]["variants_per_image"]
        
        seed = self.config["processing"].get("random_seed")
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            print(f"🔧 Random seed: {seed}")
        
        print(f"\n🎯 Pure Static Mixup Generator")
        print(f"   Input: {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Label pattern: {self.label_pattern}")
        print(f"   Mixup alpha: {self.mixup.alpha}")
        print(f"   Max label distance: {self.mixup.max_distance}")
        print(f"   Circular labels: {self.mixup.circular}")
        print(f"   Variants per image: {self.variants_per_image}")
    
    def extract_label_from_filename(self, filename):
        """Extract label from filename (e.g., '9.1_1756145124.jpg' -> 9.1)"""
        match = re.match(self.label_pattern, filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def load_dataset(self):
        print(f"\n📥 Loading dataset from flat directory...")
        
        images = []
        labels = []
        filepaths = []
        label_stats = defaultdict(int)
        
        # Find all image files
        for ext in self.image_extensions:
            pattern = os.path.join(self.input_dir, f"*{ext}")
            filepaths.extend(glob.glob(pattern))
            pattern_upper = os.path.join(self.input_dir, f"*{ext.upper()}")
            filepaths.extend(glob.glob(pattern_upper))
        
        filepaths = list(set(filepaths))
        print(f"   Found {len(filepaths)} image files")
        
        for filepath in tqdm(filepaths, desc="Loading images"):
            filename = os.path.basename(filepath)
            label = self.extract_label_from_filename(filename)
            
            if label is None:
                print(f"   ⚠️ Could not extract label from: {filename}")
                continue
            
            if label < 0.0 or label > 9.9:
                print(f"   ⚠️ Label out of range (0.0-9.9): {label} in {filename}")
                continue
            
            if self.config["image_params"]["grayscale"]:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"   ⚠️ Could not load image: {filepath}")
                continue
            
            img = img.astype(np.float32) / 255.0
            img = self.mixup.ensure_shape(img)
            
            images.append(img)
            labels.append(label)
            label_stats[label] += 1
        
        print(f"\n✅ Loaded {len(images)} images")
        if images:
            print(f"   Label range: {min(labels):.1f} - {max(labels):.1f}")
        print(f"   Unique labels: {len(label_stats)} (out of 100 possible)")
        
        if label_stats:
            sample_labels = sorted(label_stats.keys())[:10]
            print(f"\n   Label distribution sample:")
            for lbl in sample_labels:
                print(f"     {lbl:.1f}: {label_stats[lbl]} images")
        
        return images, labels, filepaths
    
    def find_compatible_indices(self, labels, target_idx):
        """Find indices of images with labels close to target label"""
        target_label = labels[target_idx]
        compatible = []
        
        for i, label in enumerate(labels):
            if i == target_idx:
                continue
            distance = CircularLabelUtils.distance(target_label, label, self.mixup.max_label)
            if distance <= self.mixup.max_distance:
                compatible.append(i)
        
        return compatible
    
    def generate_mixup_variants(self, images, labels, paths):
        print(f"\n🔄 Generating Mixup variants...")
        
        if len(images) < 2:
            print("❌ Need at least 2 images for Mixup!")
            return [], [], []
        
        mixup_images = []
        mixup_labels = []
        mixup_paths = []
        
        print(f"   Building compatibility matrix for {len(images)} images...")
        compatible_cache = {}
        for i in range(len(images)):
            compatible_cache[i] = self.find_compatible_indices(labels, i)
        
        images_with_partners = sum(1 for idx in compatible_cache if compatible_cache[idx])
        print(f"   {images_with_partners}/{len(images)} images have compatible partners")
        
        if images_with_partners == 0:
            print(f"❌ No compatible pairs found! Try increasing max_label_distance (current: {self.mixup.max_distance})")
            return [], [], []
        
        for idx in tqdm(range(len(images)), desc="Generating Mixup"):
            compatible = compatible_cache[idx]
            if not compatible:
                continue
            
            for var_id in range(self.variants_per_image):
                partner_idx = random.choice(compatible)
                
                mixed_img, mixed_lbl = self.mixup.mix_pair(
                    images[idx], labels[idx],
                    images[partner_idx], labels[partner_idx]
                )
                
                if mixed_img is not None:
                    mixup_images.append(mixed_img)
                    mixup_labels.append(mixed_lbl)
                    mixup_paths.append(paths[idx])
        
        print(f"\n✅ Generated {len(mixup_images)} Mixup variants")
        return mixup_images, mixup_labels, mixup_paths
    
    def save_image(self, img, label, original_path, aug_id):
        """Save mixup image with label in filename"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        orig_filename = os.path.basename(original_path)
        orig_name, ext = os.path.splitext(orig_filename)
        
        # Remove old label prefix if present
        if re.match(r"^\d+\.\d+_", orig_name):
            orig_name = re.sub(r"^\d+\.\d+_", "", orig_name)
        
        new_filename = f"{label:.1f}_{orig_name}_mixup_{aug_id:04d}{ext}"
        filepath = os.path.join(self.output_dir, new_filename)
        
        save_img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if len(save_img.shape) == 3 and save_img.shape[2] == 1:
            save_img = save_img[:, :, 0]
        
        cv2.imwrite(filepath, save_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath
    
    def run(self):
        images, labels, paths = self.load_dataset()
        
        if len(images) < 2:
            print("❌ Need at least 2 images for Mixup!")
            return
        
        mix_imgs, mix_lbls, mix_paths = self.generate_mixup_variants(images, labels, paths)
        
        if not mix_imgs:
            print("❌ No Mixup variants generated!")
            return
        
        if self.save_images:
            print(f"\n💾 Saving {len(mix_imgs)} images...")
            
            def save_wrapper(args):
                i, img, lbl, path = args
                return self.save_image(img, lbl, path, i)
            
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                saved = list(tqdm(
                    executor.map(save_wrapper, 
                        [(i, mix_imgs[i], mix_lbls[i], mix_paths[i]) 
                         for i in range(len(mix_imgs))]),
                    total=len(mix_imgs), 
                    desc="Saving"
                ))
        
        print(f"\n{'='*50}")
        print(f"✅ Static Mixup Complete!")
        print(f"{'='*50}")
        print(f"   Original images:     {len(images)}")
        print(f"   Mixup variants:      {len(mix_imgs)}")
        print(f"   Total images:        {len(images) + len(mix_imgs)}")
        print(f"   Output directory:    {self.output_dir}")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "mixup_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'num_original': len(images),
                'num_mixup': len(mix_imgs),
                'mixup_config': {
                    'alpha': self.mixup.alpha,
                    'max_label_distance': self.mixup.max_distance,
                    'circular': self.mixup.circular,
                    'variants_per_image': self.variants_per_image
                }
            }, f, indent=2)
        print(f"   Metadata:            {metadata_path}")
        print(f"{'='*50}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pure Static Mixup for Regression')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--input', type=str, default=None, help='Input directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--distance', type=float, default=None, help='Max label distance')
    parser.add_argument('--variants', type=int, default=None, help='Variants per image')
    parser.add_argument('--create_config', action='store_true', help='Create default config file')  # Changed from create-config
    parser.add_argument('--debug', action='store_true', help='Debug dataset')
    
    args = parser.parse_args()
    
    if args.create_config:  # Changed from create-config
        with open('mixup_config.json', 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print("✅ Created mixup_config.json")
        return
    
    if args.debug:
        input_dir = args.input or DEFAULT_CONFIG["dataset"]["input_dir"]
        print(f"Debugging: {input_dir}")
        if os.path.exists(input_dir):
            files = glob.glob(os.path.join(input_dir, "*.jpg"))[:10]
            for f in files:
                name = os.path.basename(f)
                match = re.match(r"^(\d+\.\d+)_", name)
                if match:
                    print(f"  {name} → label = {match.group(1)}")
                else:
                    print(f"  {name} → NO MATCH")
        else:
            print(f"  Directory not found: {input_dir}")
        return
    
    generator = StaticMixupGenerator(args.config)
    
    if args.input:
        generator.input_dir = args.input
        generator.config["dataset"]["input_dir"] = args.input
    if args.output:
        generator.output_dir = args.output
        generator.config["dataset"]["output_dir"] = args.output
    if args.distance:
        generator.mixup.max_distance = args.distance
        generator.config["mixup"]["max_label_distance"] = args.distance
    if args.variants:
        generator.variants_per_image = args.variants
        generator.config["mixup"]["variants_per_image"] = args.variants
    
    generator.run()

if __name__ == "__main__":
    main()