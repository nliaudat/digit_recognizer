import tensorflow as tf
import numpy as np
import os
import cv2
from scipy import ndimage
import random
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import shutil
import argparse
import glob
from pathlib import Path
import sys

# ==============================================================================
# DATA AUGMENTATION PARAMETERS (INCLUDED DIRECTLY)
# ==============================================================================

# Image Parameters - CORRECTED: width first, then height
INPUT_WIDTH = 20
INPUT_HEIGHT = 32
INPUT_CHANNELS = 1  # 1 for grayscale, 3 for RGB
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)  # (height, width, channels)
USE_GRAYSCALE = (INPUT_CHANNELS == 1)

# Dataset Paths
INPUT_DATASET_DIR = "dataset"  # Default input folder
AUGMENTED_DATA_DIR = None      # Auto-generated: [input]_augmented

# Enable/disable augmentation
USE_DATA_AUGMENTATION = True
AUGMENTATION_MULTIPLIER = 3  # How many augmented samples per original sample

# Basic Image Transformations
AUG_ROTATION_RANGE = 15  # degrees (±15)
AUG_ZOOM_RANGE = 0.2     # ±20% zoom
AUG_WIDTH_SHIFT_RANGE = 0.1  # ±10% horizontal shift
AUG_HEIGHT_SHIFT_RANGE = 0.1 # ±10% vertical shift
AUG_SHEAR_RANGE = 0.2    # ±20% shear

# Color/Style Augmentations
AUG_BRIGHTNESS_RANGE = [0.8, 1.2]  # 80% to 120% brightness
AUG_CONTRAST_RANGE = [0.8, 1.2]    # 80% to 120% contrast
AUG_COLOR_JITTER = 0.1             # 10% color variation
AUG_GAUSSIAN_NOISE_STD = 0.05      # Standard deviation for Gaussian noise

# Random Erasing/Cutout
AUG_USE_RANDOM_ERASING = True
AUG_ERASING_MAX_AREA = 0.1         # Max 10% of image area
AUG_ERASING_ASPECT_RATIO = (0.3, 3.3)  # Aspect ratio range

# Spatial Transformations
AUG_HORIZONTAL_FLIP = False        # Usually False for digits
AUG_VERTICAL_FLIP = False          # Usually False for digits
AUG_RANDOM_CROP = True
AUG_CROP_PERCENT = 0.9             # Crop to 90% of original size
AUG_PERSPECTIVE_TRANSFORM = True
AUG_PERSPECTIVE_SCALE = 0.1        # 10% perspective distortion

# Flashlight Disturbance Augmentation
AUG_FLASHLIGHT_DISTURBANCE = True
AUG_FLASHLIGHT_INTENSITY = 0.8           # Maximum brightness intensity (0.0 to 1.0)
AUG_FLASHLIGHT_RADIUS_RANGE = [0.1, 0.3] # Radius as fraction of image size (10% to 30%)
AUG_FLASHLIGHT_PROGRESSIVE = True        # Whether the effect is progressive (fades out)
AUG_FLASHLIGHT_AFFECTED_AREA = 0.25      # Target affected area (1/4 of image)
AUG_FLASHLIGHT_PROBABILITY = 0.3         # Probability of applying this augmentation (30%)

# Advanced Settings
AUGMENTATION_THREADS = 4           # Parallel processing threads
AUGMENTATION_BATCH_SIZE = 32       # Batch size for augmentation
SAVE_AUGMENTED_IMAGES = True       # Save augmented images to disk
AUGMENT_ON_THE_FLY = False         # Single-shot augmentation (not during training)

# ==============================================================================
# AUGMENTATION CLASS
# ==============================================================================

class SingleShotAugmentor:
    """Single-shot data augmentation that processes entire dataset folders"""
    
    def __init__(self, input_dir=None, output_dir=None):
        self.input_dir = input_dir or INPUT_DATASET_DIR
        self.output_dir = output_dir or self._get_default_output_dir()
        self.input_shape = INPUT_SHAPE
        self.use_grayscale = USE_GRAYSCALE
        
        # Validate input directory
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        self.setup_directories()
        print(f"🎯 Single-shot augmentation configured:")
        print(f"   Input: {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Target shape: {self.input_shape} (height={INPUT_HEIGHT}, width={INPUT_WIDTH}, channels={INPUT_CHANNELS})")
        print(f"   Grayscale: {self.use_grayscale}")
    
    def _get_default_output_dir(self):
        """Generate default output directory name"""
        input_name = os.path.basename(os.path.normpath(INPUT_DATASET_DIR))
        return f"{input_name}_augmented"
    
    def setup_directories(self):
        """Setup directory structure matching the original dataset"""
        if os.path.exists(self.output_dir):
            print(f"⚠️  Output directory already exists: {self.output_dir}")
            response = input("Continue and overwrite? (y/n): ").lower().strip()
            if response != 'y':
                print("Augmentation cancelled.")
                exit(0)
            else:
                shutil.rmtree(self.output_dir)
        
        os.makedirs(self.output_dir)
        print(f"📁 Created output directory: {self.output_dir}")
        
        # Copy directory structure from original dataset
        for root, dirs, files in os.walk(self.input_dir):
            # Create corresponding directory in augmented dataset
            relative_path = os.path.relpath(root, self.input_dir)
            aug_path = os.path.join(self.output_dir, relative_path)
            
            if not os.path.exists(aug_path):
                os.makedirs(aug_path)
                print(f"📁 Created directory: {aug_path}")
    
    def load_images_from_folder(self, folder_path):
        """Load all images from a folder with their paths and labels"""
        images = []
        labels = []
        image_paths = []
        
        supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        
        for format in supported_formats:
            for img_path in glob.glob(os.path.join(folder_path, format)):
                try:
                    # Load image
                    if self.use_grayscale:
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    if image is None:
                        continue
                    
                    # Normalize to [0, 1]
                    image = image.astype(np.float32) / 255.0
                    
                    # Get label from folder name
                    label = os.path.basename(folder_path)
                    
                    images.append(image)
                    labels.append(label)
                    image_paths.append(img_path)
                    
                except Exception as e:
                    print(f"⚠️  Error loading {img_path}: {e}")
                    continue
        
        return images, labels, image_paths
    
    def load_entire_dataset(self):
        """Load entire dataset from folder structure"""
        print(f"📥 Loading dataset from: {self.input_dir}")
        
        all_images = []
        all_labels = []
        all_paths = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.input_dir):
            if files:  # Only process directories that contain files
                images, labels, paths = self.load_images_from_folder(root)
                all_images.extend(images)
                all_labels.extend(labels)
                all_paths.extend(paths)
                
                print(f"   Loaded {len(images)} images from {os.path.basename(root)}")
        
        # Convert to numpy arrays
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        
        print(f"✅ Dataset loaded:")
        print(f"   Total images: {len(all_images)}")
        print(f"   Classes: {np.unique(all_labels)}")
        print(f"   Image shape: {all_images[0].shape}")
        
        return all_images, all_labels, all_paths
    
    def ensure_correct_shape(self, image):
        """Ensure image has the correct shape for the model"""
        if len(image.shape) == 2 and not self.use_grayscale:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and self.use_grayscale and image.shape[2] == 3:
            # Convert RGB to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and self.use_grayscale and image.shape[2] == 1:
            # Already grayscale with channel dimension
            pass
        elif len(image.shape) == 2 and self.use_grayscale:
            # Add channel dimension to grayscale
            image = np.expand_dims(image, axis=-1)
        
        # Resize to target shape if needed - CORRECTED: width first, then height
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))  # width, height
        
        return image
    
    def apply_rotation(self, image, angle=None):
        """Apply random rotation"""
        if angle is None:
            angle = random.uniform(-AUG_ROTATION_RANGE, AUG_ROTATION_RANGE)
        
        # Ensure correct shape
        image = self.ensure_correct_shape(image)
        
        if len(image.shape) == 2:
            return ndimage.rotate(image, angle, reshape=False, mode='nearest')
        else:
            rotated = np.zeros_like(image)
            for channel in range(image.shape[2]):
                rotated[:, :, channel] = ndimage.rotate(
                    image[:, :, channel], angle, reshape=False, mode='nearest'
                )
            return rotated
    
    def apply_zoom(self, image, zoom_factor=None):
        """Apply random zoom"""
        if zoom_factor is None:
            zoom_factor = random.uniform(1 - AUG_ZOOM_RANGE, 1 + AUG_ZOOM_RANGE)
        
        image = self.ensure_correct_shape(image)
        h, w = image.shape[:2]  # h = height, w = width
        
        # Calculate zoom boundaries
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        if zoom_factor > 1:
            # Zoom in - crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            zoomed = image[start_h:start_h + h, start_w:start_w + w]
        else:
            # Zoom out - pad with zeros
            zoomed = np.zeros_like(image)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            zoomed[start_h:start_h + new_h, start_w:start_w + new_w] = cv2.resize(
                image, (new_w, new_h)  # width, height
            )
        
        return self.ensure_correct_shape(zoomed)
    
    def apply_shift(self, image, shift_x=None, shift_y=None):
        """Apply random translation"""
        if shift_x is None:
            shift_x = random.uniform(-AUG_WIDTH_SHIFT_RANGE, AUG_WIDTH_SHIFT_RANGE)
        if shift_y is None:
            shift_y = random.uniform(-AUG_HEIGHT_SHIFT_RANGE, AUG_HEIGHT_SHIFT_RANGE)
        
        image = self.ensure_correct_shape(image)
        h, w = image.shape[:2]  # h = height, w = width
        shift_x_pixels = int(shift_x * w)  # width shift
        shift_y_pixels = int(shift_y * h)  # height shift
        
        M = np.float32([[1, 0, shift_x_pixels], [0, 1, shift_y_pixels]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)  # width, height
        
        return self.ensure_correct_shape(shifted)
    
    def apply_shear(self, image, shear=None):
        """Apply random shearing"""
        if shear is None:
            shear = random.uniform(-AUG_SHEAR_RANGE, AUG_SHEAR_RANGE)
        
        image = self.ensure_correct_shape(image)
        h, w = image.shape[:2]  # h = height, w = width
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)  # width, height
        return self.ensure_correct_shape(sheared)
    
    def adjust_brightness(self, image, factor=None):
        """Adjust image brightness"""
        if factor is None:
            factor = random.uniform(AUG_BRIGHTNESS_RANGE[0], AUG_BRIGHTNESS_RANGE[1])
        
        image = self.ensure_correct_shape(image)
        brightened = np.clip(image * factor, 0, 1)
        return self.ensure_correct_shape(brightened)
    
    def adjust_contrast(self, image, factor=None):
        """Adjust image contrast"""
        if factor is None:
            factor = random.uniform(AUG_CONTRAST_RANGE[0], AUG_CONTRAST_RANGE[1])
        
        image = self.ensure_correct_shape(image)
        if len(image.shape) == 2:
            mean_val = np.mean(image)
            contrasted = np.clip((image - mean_val) * factor + mean_val, 0, 1)
        else:
            contrasted = np.zeros_like(image)
            for channel in range(image.shape[2]):
                channel_mean = np.mean(image[:, :, channel])
                contrasted[:, :, channel] = np.clip(
                    (image[:, :, channel] - channel_mean) * factor + channel_mean, 0, 1
                )
        
        return self.ensure_correct_shape(contrasted)
    
    def apply_color_jitter(self, image):
        """Apply random color jittering"""
        image = self.ensure_correct_shape(image)
        
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = self.adjust_brightness(image)
            image = self.adjust_contrast(image)
        else:
            jittered = np.zeros_like(image)
            for channel in range(image.shape[2]):
                channel_data = image[:, :, channel]
                brightness_factor = random.uniform(1 - AUG_COLOR_JITTER, 1 + AUG_COLOR_JITTER)
                contrast_factor = random.uniform(1 - AUG_COLOR_JITTER, 1 + AUG_COLOR_JITTER)
                brightened = channel_data * brightness_factor
                channel_mean = np.mean(brightened)
                contrasted = (brightened - channel_mean) * contrast_factor + channel_mean
                jittered[:, :, channel] = np.clip(contrasted, 0, 1)
            image = jittered
        
        return self.ensure_correct_shape(image)
    
    def add_gaussian_noise(self, image):
        """Add Gaussian noise to image"""
        image = self.ensure_correct_shape(image)
        noise = np.random.normal(0, AUG_GAUSSIAN_NOISE_STD, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
        return self.ensure_correct_shape(noisy_image)
    
    def apply_random_erasing(self, image):
        """Apply random erasing/cutout"""
        if not AUG_USE_RANDOM_ERASING:
            return image
        
        image = self.ensure_correct_shape(image)
        h, w = image.shape[:2]  # h = height, w = width
        erase_area = random.uniform(0.02, AUG_ERASING_MAX_AREA) * (h * w)
        aspect_ratio = random.uniform(AUG_ERASING_ASPECT_RATIO[0], AUG_ERASING_ASPECT_RATIO[1])
        
        erase_h = int(np.sqrt(erase_area * aspect_ratio))  # height
        erase_w = int(np.sqrt(erase_area / aspect_ratio))  # width
        erase_h = min(erase_h, h - 1)
        erase_w = min(erase_w, w - 1)
        
        if erase_h > 0 and erase_w > 0:
            top = random.randint(0, h - erase_h)    # height position
            left = random.randint(0, w - erase_w)   # width position
            erased = image.copy()
            if len(image.shape) == 2:
                erased[top:top + erase_h, left:left + erase_w] = 0
            else:
                erased[top:top + erase_h, left:left + erase_w, :] = 0
        
        return self.ensure_correct_shape(erased)
    
    def apply_random_crop(self, image):
        """Apply random cropping"""
        if not AUG_RANDOM_CROP:
            return image
        
        image = self.ensure_correct_shape(image)
        h, w = image.shape[:2]  # h = height, w = width
        crop_size = int(min(h, w) * AUG_CROP_PERCENT)
        
        top = random.randint(0, h - crop_size)    # height position
        left = random.randint(0, w - crop_size)   # width position
        cropped = image[top:top + crop_size, left:left + crop_size]
        resized = cv2.resize(cropped, (w, h))  # width, height
        
        return self.ensure_correct_shape(resized)
    
    def apply_perspective_transform(self, image):
        """Apply perspective transformation"""
        if not AUG_PERSPECTIVE_TRANSFORM:
            return image
        
        image = self.ensure_correct_shape(image)
        h, w = image.shape[:2]  # h = height, w = width
        src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        max_distortion = AUG_PERSPECTIVE_SCALE * min(h, w)
        dst_points = src_points + np.random.uniform(-max_distortion, max_distortion, src_points.shape)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        perspective = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)  # width, height
        return self.ensure_correct_shape(perspective)
    
    def apply_flashlight_disturbance(self, image):
        """Apply flashlight disturbance - bright white spot affecting 1/4 of image"""
        if not AUG_FLASHLIGHT_DISTURBANCE:
            return image
        
        # Random chance to apply this augmentation
        if random.random() > AUG_FLASHLIGHT_PROBABILITY:
            return image
        
        # Ensure correct shape
        image = self.ensure_correct_shape(image)
        h, w = image.shape[:2]  # h = height, w = width
        
        # Calculate target affected area (1/4 of image)
        target_area = AUG_FLASHLIGHT_AFFECTED_AREA * (h * w)
        
        # Determine flashlight radius to achieve approximately 1/4 area coverage
        # Area of circle = πr², so r = sqrt(area/π)
        target_radius = np.sqrt(target_area / np.pi)
        max_possible_radius = min(h, w) / 2
        
        # Clamp radius to reasonable range and apply random variation
        min_radius = int(max_possible_radius * AUG_FLASHLIGHT_RADIUS_RANGE[0])
        max_radius = int(max_possible_radius * AUG_FLASHLIGHT_RADIUS_RANGE[1])
        flashlight_radius = random.randint(min_radius, max_radius)
        
        # Random position for flashlight center (avoid edges)
        margin = flashlight_radius + 5
        center_x = random.randint(margin, w - margin)  # width position
        center_y = random.randint(margin, h - margin)  # height position
        
        # Random intensity variation
        intensity = random.uniform(AUG_FLASHLIGHT_INTENSITY * 0.7, AUG_FLASHLIGHT_INTENSITY)
        
        # Create flashlight effect
        disturbed = image.copy()
        
        if AUG_FLASHLIGHT_PROGRESSIVE:
            # Create coordinate grids for vectorized operations
            y_coords, x_coords = np.ogrid[:h, :w]  # y = height, x = width
            
            # Calculate distance from flashlight center for each pixel
            distance_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # Create smooth intensity falloff using Gaussian function
            sigma = flashlight_radius / 2.5  # Controls falloff steepness
            intensity_map = intensity * np.exp(-(distance_map**2) / (2 * sigma**2))
            
            # Apply flashlight effect
            if len(disturbed.shape) == 2:
                # Grayscale
                disturbed = np.clip(disturbed + intensity_map, 0, 1)
            else:
                # Color - apply to all channels
                for channel in range(disturbed.shape[2]):
                    disturbed[:, :, channel] = np.clip(
                        disturbed[:, :, channel] + intensity_map, 0, 1
                    )
            
            # Optional: Add slight blur at the edges for more realistic light transition
            if random.random() < 0.3:
                kernel_size = random.choice([3, 5])
                if len(disturbed.shape) == 2:
                    disturbed = cv2.GaussianBlur(disturbed, (kernel_size, kernel_size), 0)
                else:
                    for channel in range(disturbed.shape[2]):
                        disturbed[:, :, channel] = cv2.GaussianBlur(
                            disturbed[:, :, channel], (kernel_size, kernel_size), 0
                        )
        else:
            # Simple circular flashlight (uniform brightness within radius)
            y_coords, x_coords = np.ogrid[:h, :w]  # y = height, x = width
            mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= flashlight_radius**2
            
            if len(disturbed.shape) == 2:
                # Grayscale
                disturbed[mask] = np.clip(disturbed[mask] + intensity, 0, 1)
            else:
                # Color
                for channel in range(disturbed.shape[2]):
                    disturbed[mask, channel] = np.clip(
                        disturbed[mask, channel] + intensity, 0, 1
                    )
        
        return self.ensure_correct_shape(disturbed)
    
    def augment_single_image(self, image, label, original_path, augmentation_id):
        """Apply comprehensive augmentation to a single image"""
        image = self.ensure_correct_shape(image)
        augmented = image.copy()
        
        # Select augmentations to apply
        augmentations_to_apply = [
            self.apply_rotation,
            self.apply_zoom,
            self.apply_shift,
            self.apply_shear
        ]
        
        # Add color augmentations
        color_augmentations = [
            self.adjust_brightness,
            self.adjust_contrast,
            self.apply_color_jitter,
            self.add_gaussian_noise
        ]
        augmentations_to_apply.extend(random.sample(color_augmentations, random.randint(1, 2)))
        
        # Add spatial augmentations (including flashlight disturbance)
        spatial_augmentations = [
            self.apply_random_erasing,
            self.apply_random_crop,
            self.apply_perspective_transform,
            self.apply_flashlight_disturbance
        ]
        if random.random() < 0.5:
            augmentations_to_apply.append(random.choice(spatial_augmentations))
        
        # Apply in random order
        random.shuffle(augmentations_to_apply)
        
        for augmentation in augmentations_to_apply:
            try:
                if augmentation.__name__ in ['apply_color_jitter', 'add_gaussian_noise', 
                                           'apply_random_erasing', 'apply_random_crop',
                                           'apply_perspective_transform', 'apply_flashlight_disturbance']:
                    augmented = augmentation(augmented)
                else:
                    augmented = augmentation(augmented)
            except Exception as e:
                continue
        
        augmented = np.clip(augmented, 0, 1)
        augmented = self.ensure_correct_shape(augmented)
        
        # Save augmented image
        self.save_augmented_image(augmented, label, original_path, augmentation_id)
        
        return augmented
    
    def save_augmented_image(self, image, label, original_path, augmentation_id):
        """Save augmented image to disk preserving folder structure"""
        # Get relative path from input directory
        relative_path = os.path.relpath(os.path.dirname(original_path), self.input_dir)
        aug_dir = os.path.join(self.output_dir, relative_path)
        os.makedirs(aug_dir, exist_ok=True)
        
        # Create filename
        original_filename = os.path.splitext(os.path.basename(original_path))[0]
        filename = f"aug_{original_filename}_{augmentation_id:03d}.png"
        filepath = os.path.join(aug_dir, filename)
        
        # Convert and save
        if image.max() <= 1.0:
            save_image = (image * 255).astype(np.uint8)
        else:
            save_image = image.astype(np.uint8)
        
        # Remove channel dimension for grayscale if needed
        if len(save_image.shape) == 3 and save_image.shape[2] == 1:
            save_image = save_image[:, :, 0]
        
        cv2.imwrite(filepath, save_image)
    
    def run_augmentation(self):
        """Run single-shot augmentation on entire dataset"""
        print("🚀 Starting single-shot augmentation...")
        
        # Load dataset
        images, labels, paths = self.load_entire_dataset()
        
        # Run augmentation
        total_original = len(images)
        total_augmented = total_original * AUGMENTATION_MULTIPLIER
        
        print(f"🔧 Augmentation settings:")
        print(f"   Multiplier: {AUGMENTATION_MULTIPLIER}x")
        print(f"   Expected output: {total_augmented} images")
        print(f"   Threads: {AUGMENTATION_THREADS}")
        print(f"   Target shape: {INPUT_HEIGHT}x{INPUT_WIDTH}x{INPUT_CHANNELS}")
        print(f"   Flashlight disturbance: {'Enabled' if AUG_FLASHLIGHT_DISTURBANCE else 'Disabled'}")
        
        # Use parallel processing
        with ThreadPoolExecutor(max_workers=AUGMENTATION_THREADS) as executor:
            futures = []
            
            for i in range(len(images)):
                for aug_id in range(AUGMENTATION_MULTIPLIER):
                    futures.append(
                        executor.submit(
                            self.augment_single_image, 
                            images[i], labels[i], paths[i], aug_id
                        )
                    )
            
            # Process with progress bar
            completed = 0
            for future in tqdm(futures, desc="Augmenting images", total=len(futures)):
                try:
                    future.result()
                    completed += 1
                except Exception as e:
                    print(f"⚠️  Augmentation failed: {e}")
                    continue
        
        print(f"✅ Single-shot augmentation completed!")
        print(f"   Original images: {total_original}")
        print(f"   Augmented images: {completed}")
        print(f"   Total images: {total_original + completed}")
        print(f"   Output directory: {self.output_dir}")


def main():
    """Command line interface for single-shot augmentation"""
    parser = argparse.ArgumentParser(description='Single-shot dataset augmentation')
    parser.add_argument('--input', type=str, default=INPUT_DATASET_DIR,
                       help='Input dataset directory (default: dataset)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: [input]_augmented)')
    parser.add_argument('--multiplier', type=int, default=AUGMENTATION_MULTIPLIER,
                       help=f'Augmentation multiplier (default: {AUGMENTATION_MULTIPLIER})')
    
    args = parser.parse_args()
    
    # Update parameters from command line
    if args.multiplier != AUGMENTATION_MULTIPLIER:
        globals()['AUGMENTATION_MULTIPLIER'] = args.multiplier
    
    try:
        # Run augmentation
        augmentor = SingleShotAugmentor(input_dir=args.input, output_dir=args.output)
        augmentor.run_augmentation()
        
    except Exception as e:
        print(f"❌ Augmentation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()