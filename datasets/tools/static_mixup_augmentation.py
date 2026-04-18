#!/usr/bin/env python3
"""
static_mixup.py - Adaptive Static Mixup for Regression Labels (0.0-9.9)
Dynamically loads per-class accuracy from CSV file to determine augmentation factors

Tiered factors based on accuracy:
- <90%: 3x variants
- 90-95%: 2x variants  
- >95%: 1x variants (base)

Labels: 0.0-9.9 (100 classes) | Circular boundary: 9.9 ↔ 0.0
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
import csv
import sys

# Force UTF-8 output on Windows to support emojis
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass # Fallback for environments where reconfigure might fail


# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================

DEFAULT_CONFIG = {
    "image_params": {
        "width": 20,
        "height": 32,
        "channels": 3,
        "grayscale": False
    },
    "dataset": {
        "input_dir": "../Tenth-of-step-of-a-meter-digit/images",
        "output_dir": "../Tenth-of-step-of-a-meter-digit/static_augmentation_mixup",
        "accuracy_csv": "per_class_accuracy.csv"  # Path to CSV with class accuracies
    },
    "mixup": {
        "enabled": True,
        "alpha": 0.2,
        "max_label_distance": 0.1,
        "circular": True,
        "max_label": 9.9,
        "min_label": 0.0
    },
    "adaptive_sampling": {
        "enabled": True,
        "base_variants_per_image": 1,
        "tiered_factors": {
            "low": {"max_accuracy": 0.90, "factor": 15},
            "medium": {"max_accuracy": 0.95, "factor": 5},
            "high": {"factor": 0}
        }
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
        step = 0.1
        max_val = max_label + step
        diff = abs(l1 - l2)
        return min(diff, max_val - diff)
    
    @staticmethod
    def are_close(l1, l2, max_distance=0.1, circular=True, max_label=9.9):
        if circular:
            return CircularLabelUtils.distance(l1, l2, max_label) <= max_distance
        return abs(l1 - l2) <= max_distance
    
    @staticmethod
    def mix(l1, l2, lam, circular=True, max_label=9.9, min_label=0.0):
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
# ACCURACY LOADER
# ==============================================================================

def load_accuracies_from_csv(csv_path):
    """
    Load class accuracies from CSV file.
    Expected format: Class,Total,Correct,Accuracy
    Or: Class,Accuracy
    """
    accuracies = {}
    
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV file not found: {csv_path}")
        print(f"   Using default: all classes at 100% accuracy (no adaptive sampling)")
        return accuracies
    
    with open(csv_path, 'r') as f:
        # Detect delimiter
        first_line = f.readline()
        f.seek(0)
        
        if ',' in first_line:
            delimiter = ','
        elif ';' in first_line:
            delimiter = ';'
        elif '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','
        
        reader = csv.reader(f, delimiter=delimiter)
        
        # Get header
        header = next(reader)
        header_lower = [h.lower().strip() for h in header]
        
        # Find column indices
        class_idx = None
        accuracy_idx = None
        
        for i, col in enumerate(header_lower):
            if col in ['class', 'label', 'digit']:
                class_idx = i
            elif col in ['accuracy', 'acc']:
                accuracy_idx = i
        
        # If not found by name, assume first column is class, fourth is accuracy
        if class_idx is None:
            class_idx = 0
        if accuracy_idx is None:
            # Check if we have Total/Correct columns to compute accuracy
            total_idx = None
            correct_idx = None
            for i, col in enumerate(header_lower):
                if col in ['total', 'count', 'samples']:
                    total_idx = i
                elif col in ['correct', 'correct_count', 'hits']:
                    correct_idx = i
            
            if total_idx is not None and correct_idx is not None:
                # Compute accuracy from Total and Correct
                for row in reader:
                    if len(row) > max(total_idx, correct_idx):
                        try:
                            class_val = int(row[class_idx].strip())
                            total = float(row[total_idx])
                            correct = float(row[correct_idx])
                            accuracy = correct / total if total > 0 else 0
                            accuracies[class_val] = accuracy
                        except (ValueError, IndexError):
                            continue
                return accuracies
            else:
                # Assume accuracy is in the 4th column (index 3)
                accuracy_idx = 3
        
        # Parse rows
        for row in reader:
            if len(row) > max(class_idx, accuracy_idx):
                try:
                    class_val = int(row[class_idx].strip())
                    accuracy = float(row[accuracy_idx].strip())
                    accuracies[class_val] = accuracy
                except (ValueError, IndexError):
                    continue
    
    print(f"✅ Loaded {len(accuracies)} class accuracies from {csv_path}")
    return accuracies

# ==============================================================================
# MIXUP AUGMENTOR
# ==============================================================================

class PureMixupAugmentor:
    def __init__(self, config, accuracies):
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
        
        # Adaptive sampling config
        self.adaptive = config.get("adaptive_sampling", {})
        self.adaptive_enabled = self.adaptive.get("enabled", False)
        self.base_variants = self.adaptive.get("base_variants_per_image", 1)
        self.tiers = self.adaptive.get("tiered_factors", {})
        self.accuracies = accuracies  # Now loaded from CSV
    
    def get_variants_per_class(self, label):
        """Determine how many variants based on accuracy tier"""
        if not self.adaptive_enabled or not self.accuracies:
            return self.base_variants
        
        int_class = int(round(label * 10))
        accuracy = self.accuracies.get(int_class, 1.0)
        
        # Determine factor based on accuracy tier
        if accuracy < self.tiers.get("low", {}).get("max_accuracy", 0.90):
            factor = self.tiers.get("low", {}).get("factor", 3)
        elif accuracy < self.tiers.get("medium", {}).get("max_accuracy", 0.95):
            factor = self.tiers.get("medium", {}).get("factor", 2)
        else:
            factor = self.tiers.get("high", {}).get("factor", 1)
        
        return factor
    
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
    def __init__(self, config=None, csv_path=None):
        self.config = config or DEFAULT_CONFIG
        
        # Load accuracies from CSV
        csv_file = csv_path or self.config["dataset"].get("accuracy_csv", "per_class_accuracy.csv")
        self.accuracies = load_accuracies_from_csv(csv_file)
        
        self.mixup = PureMixupAugmentor(self.config, self.accuracies)
        
        self.input_dir = self.config["dataset"]["input_dir"]
        self.output_dir = self.config["dataset"]["output_dir"]
        
        self.threads = self.config["processing"]["threads"]
        self.save_images = self.config["processing"]["save_images"]
        
        seed = self.config["processing"].get("random_seed")
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            print(f"🔧 Random seed: {seed}")
        
        print(f"\n{'='*65}")
        print(f"🎯 Adaptive Static Mixup Generator")
        print(f"{'='*65}")
        print(f"   Input:  {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Alpha:  {self.mixup.alpha}")
        print(f"   Max label distance: {self.mixup.max_distance}")
        print(f"   Circular labels: {self.mixup.circular}")
        
        if self.mixup.adaptive_enabled and self.accuracies:
            print(f"\n   📊 TIERED ADAPTIVE SAMPLING:")
            print(f"      < 90% accuracy:  {self.mixup.tiers.get('low', {}).get('factor', 3)}x variants")
            print(f"      90-95% accuracy: {self.mixup.tiers.get('medium', {}).get('factor', 2)}x variants")
            print(f"      > 95% accuracy:  {self.mixup.tiers.get('high', {}).get('factor', 1)}x variants")
            
            # Show classes by tier
            low_classes = []
            medium_classes = []
            high_classes = []
            
            for class_key, acc in self.accuracies.items():
                label_val = class_key / 10.0
                if acc < 0.90:
                    low_classes.append((label_val, acc))
                elif acc < 0.95:
                    medium_classes.append((label_val, acc))
                else:
                    high_classes.append((label_val, acc))
            
            if low_classes:
                print(f"\n   🔴 LOWER ACCURACY (<90%) → {self.mixup.tiers.get('low', {}).get('factor', 3)}x:")
                for label_val, acc in sorted(low_classes, key=lambda x: x[1])[:10]:
                    print(f"      Class {label_val:.1f}: {acc:.1%}")
                if len(low_classes) > 10:
                    print(f"      ... and {len(low_classes) - 10} more")
            
            if medium_classes:
                print(f"\n   🟡 MEDIUM ACCURACY (90-95%) → {self.mixup.tiers.get('medium', {}).get('factor', 2)}x:")
                for label_val, acc in sorted(medium_classes, key=lambda x: x[1])[:10]:
                    print(f"      Class {label_val:.1f}: {acc:.1%}")
                if len(medium_classes) > 10:
                    print(f"      ... and {len(medium_classes) - 10} more")
            
            if high_classes:
                print(f"\n   🟢 HIGH ACCURACY (>95%) → {self.mixup.tiers.get('high', {}).get('factor', 1)}x:")
                print(f"      {len(high_classes)} classes at high accuracy")
        else:
            print(f"\n   ⚠️  Adaptive sampling disabled or no accuracy data")
            print(f"   Using base variants: {self.mixup.base_variants}x per image")
        
        print(f"{'='*65}\n")
    
    def extract_label_from_filename(self, filename):
        match = re.match(r"^(\d+\.\d+)_", filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def load_dataset(self):
        print(f"📥 Loading dataset...")
        
        images = []
        labels = []
        filepaths = []
        label_stats = defaultdict(int)
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        filepaths = []
        for ext in extensions:
            filepaths.extend(glob.glob(os.path.join(self.input_dir, f"*{ext}")))
            filepaths.extend(glob.glob(os.path.join(self.input_dir, f"*{ext.upper()}")))
        
        filepaths = list(set(filepaths))
        print(f"   Found {len(filepaths)} image files")
        
        for filepath in tqdm(filepaths, desc="Loading images"):
            filename = os.path.basename(filepath)
            label = self.extract_label_from_filename(filename)
            
            if label is None:
                continue
            
            if label < 0.0 or label > 9.9:
                continue
            
            if self.config["image_params"]["grayscale"]:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
            
            img = img.astype(np.float32) / 255.0
            img = self.mixup.ensure_shape(img)
            
            images.append(img)
            labels.append(label)
            label_stats[label] += 1
        
        print(f"\n✅ Loaded {len(images)} images")
        if images:
            print(f"   Label range: {min(labels):.1f} - {max(labels):.1f}")
        print(f"   Unique labels: {len(label_stats)}/100")
        
        return images, labels, filepaths
    
    def find_compatible_indices(self, labels, target_idx):
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
        print(f"\n🔄 Generating adaptive Mixup variants...")
        
        if len(images) < 2:
            print("❌ Need at least 2 images for Mixup!")
            return [], [], []
        
        # Pre-compute compatibility
        print(f"   Building compatibility matrix...")
        compatible_cache = {}
        for i in range(len(images)):
            compatible_cache[i] = self.find_compatible_indices(labels, i)
        
        images_with_partners = sum(1 for idx in compatible_cache if compatible_cache[idx])
        print(f"   {images_with_partners}/{len(images)} images have compatible partners")
        
        if images_with_partners == 0:
            print(f"❌ No compatible pairs! Try increasing max_label_distance")
            return [], [], []
        
        # Generate variants with adaptive sampling per class
        mixup_images = []
        mixup_labels = []
        mixup_paths = []
        
        # Track statistics
        variants_per_class = defaultdict(int)
        
        for idx in tqdm(range(len(images)), desc="Generating Mixup"):
            compatible = compatible_cache[idx]
            if not compatible:
                continue
            
            label = labels[idx]
            factor = self.mixup.get_variants_per_class(label)
            
            for var_id in range(factor):
                partner_idx = random.choice(compatible)
                
                mixed_img, mixed_lbl = self.mixup.mix_pair(
                    images[idx], labels[idx],
                    images[partner_idx], labels[partner_idx]
                )
                
                if mixed_img is not None:
                    mixup_images.append(mixed_img)
                    mixup_labels.append(mixed_lbl)
                    mixup_paths.append(paths[idx])
                    variants_per_class[label] += 1
        
        print(f"\n✅ Generated {len(mixup_images)} Mixup variants")
        
        # Show distribution by tier
        if self.accuracies:
            print(f"\n   📊 Generation summary by tier:")
            tier_stats = defaultdict(lambda: {"count": 0, "total_variants": 0})
            
            for label, count in variants_per_class.items():
                int_class = int(round(label * 10))
                acc = self.accuracies.get(int_class, 1.0)
                
                if acc < 0.90:
                    tier = "<90%"
                elif acc < 0.95:
                    tier = "90-95%"
                else:
                    tier = ">95%"
                
                tier_stats[tier]["count"] += 1
                tier_stats[tier]["total_variants"] += count
            
            for tier, stats in sorted(tier_stats.items()):
                print(f"      {tier}: {stats['count']} classes, {stats['total_variants']} variants")
        
        return mixup_images, mixup_labels, mixup_paths
    
    def save_image(self, img, label, original_path, aug_id):
        os.makedirs(self.output_dir, exist_ok=True)
        
        orig_filename = os.path.basename(original_path)
        orig_name, ext = os.path.splitext(orig_filename)
        
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
            print(f"\n💾 Saving {len(mix_imgs)} images to: {self.output_dir}")
            
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
        
        print(f"\n{'='*65}")
        print(f"✅ Static Mixup Complete!")
        print(f"{'='*65}")
        print(f"   Original images:     {len(images)}")
        print(f"   Mixup variants:      {len(mix_imgs)}")
        print(f"   Total images:        {len(images) + len(mix_imgs)}")
        print(f"   Output directory:    {self.output_dir}")
        print(f"{'='*65}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Adaptive Static Mixup for Regression')
    parser.add_argument('--input', type=str, default=None, help='Input directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON (for compatibility)')
    parser.add_argument('--csv', type=str, default=None, help='Path to per_class_accuracy.csv')
    parser.add_argument('--distance', type=float, default=None, help='Max label distance')
    parser.add_argument('--debug', action='store_true', help='Debug dataset')
    parser.add_argument('--create-config', action='store_true', help='Create default config file')
    parser.add_argument('--yes', action='store_true', help='Auto-confirm (for compatibility)')

    
    args = parser.parse_args()
    
    if args.create_config:
        with open('mixup_config.json', 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print("✅ Created mixup_config.json")
        print("\nEdit this file to customize:")
        print("  - dataset.accuracy_csv: path to your per_class_accuracy.csv")
        print("  - adaptive_sampling.tiered_factors: adjust multipliers")
        return
    
    config = DEFAULT_CONFIG.copy()
    
    if args.input:
        config["dataset"]["input_dir"] = args.input
    if args.output:
        config["dataset"]["output_dir"] = args.output
    if args.csv:
        config["dataset"]["accuracy_csv"] = args.csv
    if args.distance:
        config["mixup"]["max_label_distance"] = args.distance
    
    if args.debug:
        input_dir = config["dataset"]["input_dir"]
        csv_path = args.csv or config["dataset"].get("accuracy_csv", "per_class_accuracy.csv")
        
        print(f"🔍 Debugging:")
        print(f"   Input directory: {input_dir}")
        print(f"   CSV file: {csv_path}")
        
        if os.path.exists(input_dir):
            files = glob.glob(os.path.join(input_dir, "*.jpg"))[:10]
            print(f"\n   Sample filenames:")
            for f in files:
                name = os.path.basename(f)
                match = re.match(r"^(\d+\.\d+)_", name)
                if match:
                    print(f"     ✓ {name} → label = {match.group(1)}")
                else:
                    print(f"     ✗ {name} → NO MATCH")
        else:
            print(f"   Directory not found: {input_dir}")
        
        # Load and display accuracies
        accuracies = load_accuracies_from_csv(csv_path)
        if accuracies:
            print(f"\n   Loaded {len(accuracies)} class accuracies:")
            low_acc = [(k, v) for k, v in accuracies.items() if v < 0.90]
            if low_acc:
                print(f"      Classes with <90% accuracy:")
                for k, v in sorted(low_acc, key=lambda x: x[1])[:15]:
                    label_val = k / 10.0
                    print(f"        Class {label_val:.1f}: {v:.1%}")
        return
    
    generator = StaticMixupGenerator(config, args.csv)
    generator.run()

if __name__ == "__main__":

    main()