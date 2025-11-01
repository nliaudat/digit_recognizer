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
import json
from pathlib import Path
import sys

# ==============================================================================
# CONFIGURATION MANAGEMENT
# ==============================================================================

class AugmentationConfig:
    """Manage augmentation configuration from JSON file"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or "augmentation_params.json"
        self.config = self.load_config()
        self.apply_config()
    
    def load_config(self):
        """Load configuration from JSON file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                print(f"✅ Loaded configuration from: {self.config_path}")
                return config
            except Exception as e:
                print(f"❌ Error loading config file: {e}")
                print("🔄 Using default configuration")
                return self.get_default_config()
        else:
            print(f"⚠️  Config file not found: {self.config_path}")
            print("🔄 Using default configuration")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            "image_parameters": {
                "input_width": 20,
                "input_height": 32,
                "input_channels": 3,
                "use_grayscale": False
            },
            "dataset_paths": {
                "input_dataset_dir": "dataset",
                "augmented_data_dir": None
            },
            "augmentation_settings": {
                "use_data_augmentation": True,
                "augmentation_multiplier": 1,
                "augmentation_threads": 4,
                "augmentation_batch_size": 32,
                "save_augmented_images": True,
                "augment_on_the_fly": False
            },
            "augmentations": {
                "rotation": {"enabled": True, "probability": 0.3, "rotation_range": 5},
                "zoom": {"enabled": True, "probability": 0.3, "zoom_range": 0.1},
                "shift": {"enabled": True, "probability": 0.4, "width_shift_range": 0.05, "height_shift_range": 0.05},
                "shear": {"enabled": True, "probability": 0.2, "shear_range": 0.1},
                "brightness": {"enabled": True, "probability": 0.3, "brightness_range": [0.9, 1.1]},
                "contrast": {"enabled": True, "probability": 0.3, "contrast_range": [0.9, 1.1]},
                "color_jitter": {"enabled": True, "probability": 0.2, "color_jitter_range": 0.05},
                "gaussian_noise": {"enabled": True, "probability": 0.2, "gaussian_noise_std": 0.05},
                "random_erasing": {"enabled": False, "probability": 0.1, "erasing_max_area": 0.1, "erasing_aspect_ratio": [0.3, 3.3]},
                "random_crop": {"enabled": True, "probability": 0.1, "crop_percent": 0.9},
                "perspective": {"enabled": True, "probability": 0.1, "perspective_scale": 0.1},
                "flashlight": {"enabled": True, "probability": 0.3, "flashlight_intensity": 0.8, "flashlight_radius_range": [0.1, 0.3], "flashlight_progressive": True, "flashlight_affected_area": 0.25, "flashlight_probability": 0.3}
            }
        }
    
    def apply_config(self):
        """Apply configuration to global variables"""
        # Image parameters
        img_params = self.config["image_parameters"]
        globals()["INPUT_WIDTH"] = img_params["input_width"]
        globals()["INPUT_HEIGHT"] = img_params["input_height"]
        globals()["INPUT_CHANNELS"] = img_params["input_channels"]
        globals()["USE_GRAYSCALE"] = img_params["use_grayscale"]
        globals()["INPUT_SHAPE"] = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
        
        # Dataset paths
        ds_paths = self.config["dataset_paths"]
        globals()["INPUT_DATASET_DIR"] = ds_paths["input_dataset_dir"]
        globals()["AUGMENTED_DATA_DIR"] = ds_paths["augmented_data_dir"]
        
        # Augmentation settings
        aug_settings = self.config["augmentation_settings"]
        globals()["USE_DATA_AUGMENTATION"] = aug_settings["use_data_augmentation"]
        globals()["AUGMENTATION_MULTIPLIER"] = aug_settings["augmentation_multiplier"]
        globals()["AUGMENTATION_THREADS"] = aug_settings["augmentation_threads"]
        globals()["AUGMENTATION_BATCH_SIZE"] = aug_settings["augmentation_batch_size"]
        globals()["SAVE_AUGMENTED_IMAGES"] = aug_settings["save_augmented_images"]
        globals()["AUGMENT_ON_THE_FLY"] = aug_settings["augment_on_the_fly"]
        
        # Individual augmentation parameters
        augmentations = self.config["augmentations"]
        
        # Rotation
        if "rotation" in augmentations:
            rotation = augmentations["rotation"]
            globals()["AUG_ROTATION_RANGE"] = rotation["rotation_range"]
        
        # Zoom
        if "zoom" in augmentations:
            zoom = augmentations["zoom"]
            globals()["AUG_ZOOM_RANGE"] = zoom["zoom_range"]
        
        # Shift
        if "shift" in augmentations:
            shift = augmentations["shift"]
            globals()["AUG_WIDTH_SHIFT_RANGE"] = shift["width_shift_range"]
            globals()["AUG_HEIGHT_SHIFT_RANGE"] = shift["height_shift_range"]
        
        # Shear
        if "shear" in augmentations:
            shear = augmentations["shear"]
            globals()["AUG_SHEAR_RANGE"] = shear["shear_range"]
        
        # Brightness
        if "brightness" in augmentations:
            brightness = augmentations["brightness"]
            globals()["AUG_BRIGHTNESS_RANGE"] = brightness["brightness_range"]
        
        # Contrast
        if "contrast" in augmentations:
            contrast = augmentations["contrast"]
            globals()["AUG_CONTRAST_RANGE"] = contrast["contrast_range"]
        
        # Color jitter
        if "color_jitter" in augmentations:
            color_jitter = augmentations["color_jitter"]
            globals()["AUG_COLOR_JITTER"] = color_jitter["color_jitter_range"]
        
        # Gaussian noise
        if "gaussian_noise" in augmentations:
            gaussian_noise = augmentations["gaussian_noise"]
            globals()["AUG_GAUSSIAN_NOISE_STD"] = gaussian_noise["gaussian_noise_std"]
        
        # Random erasing
        if "random_erasing" in augmentations:
            random_erasing = augmentations["random_erasing"]
            globals()["AUG_USE_RANDOM_ERASING"] = random_erasing["enabled"]
            globals()["AUG_ERASING_MAX_AREA"] = random_erasing["erasing_max_area"]
            globals()["AUG_ERASING_ASPECT_RATIO"] = tuple(random_erasing["erasing_aspect_ratio"])
        
        # Random crop
        if "random_crop" in augmentations:
            random_crop = augmentations["random_crop"]
            globals()["AUG_RANDOM_CROP"] = random_crop["enabled"]
            globals()["AUG_CROP_PERCENT"] = random_crop["crop_percent"]
        
        # Perspective
        if "perspective" in augmentations:
            perspective = augmentations["perspective"]
            globals()["AUG_PERSPECTIVE_TRANSFORM"] = perspective["enabled"]
            globals()["AUG_PERSPECTIVE_SCALE"] = perspective["perspective_scale"]
        
        # Flashlight
        if "flashlight" in augmentations:
            flashlight = augmentations["flashlight"]
            globals()["AUG_FLASHLIGHT_DISTURBANCE"] = flashlight["enabled"]
            globals()["AUG_FLASHLIGHT_INTENSITY"] = flashlight["flashlight_intensity"]
            globals()["AUG_FLASHLIGHT_RADIUS_RANGE"] = flashlight["flashlight_radius_range"]
            globals()["AUG_FLASHLIGHT_PROGRESSIVE"] = flashlight["flashlight_progressive"]
            globals()["AUG_FLASHLIGHT_AFFECTED_AREA"] = flashlight["flashlight_affected_area"]
            globals()["AUG_FLASHLIGHT_PROBABILITY"] = flashlight["flashlight_probability"]
    
    def get_augmentation_probability(self, aug_name):
        """Get probability for a specific augmentation"""
        if aug_name in self.config["augmentations"]:
            return self.config["augmentations"][aug_name]["probability"]
        return 0.0
    
    def is_augmentation_enabled(self, aug_name):
        """Check if a specific augmentation is enabled"""
        if aug_name in self.config["augmentations"]:
            return self.config["augmentations"][aug_name]["enabled"]
        return False
    
    def get_enabled_augmentations(self):
        """Get list of enabled augmentations"""
        enabled = []
        for aug_name, aug_config in self.config["augmentations"].items():
            if aug_config["enabled"]:
                enabled.append(aug_name)
        return enabled
    
    def save_config(self, filepath=None):
        """Save current configuration to JSON file"""
        save_path = filepath or self.config_path
        try:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"💾 Configuration saved to: {save_path}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")

# Initialize global configuration
config_manager = AugmentationConfig()

# Available augmentations mapping
AUGMENTATION_MAP = {
    'rotation': 'apply_rotation',
    'zoom': 'apply_zoom',
    'shift': 'apply_shift',
    'shear': 'apply_shear',
    'brightness': 'adjust_brightness',
    'contrast': 'adjust_contrast',
    'color_jitter': 'apply_color_jitter',
    'gaussian_noise': 'add_gaussian_noise',
    'random_erasing': 'apply_random_erasing',
    'random_crop': 'apply_random_crop',
    'perspective': 'apply_perspective_transform',
    'flashlight': 'apply_flashlight_disturbance'
}

# Global variable to store selected augmentations
SELECTED_AUGMENTATIONS = None

# ==============================================================================
# AUGMENTATION CLASS
# ==============================================================================

class SingleShotAugmentor:
    """Single-shot data augmentation that processes entire dataset folders"""
    
    def __init__(self, input_dir=None, output_dir=None, selected_augmentations=None, config_path=None):
        # Load configuration FIRST
        self.config = AugmentationConfig(config_path)
        
        # Now use configured values, with command-line overrides
        self.input_dir = input_dir or self.config.config["dataset_paths"]["input_dataset_dir"] or INPUT_DATASET_DIR
        self.output_dir = output_dir or self._get_default_output_dir()
        self.input_shape = INPUT_SHAPE
        self.use_grayscale = USE_GRAYSCALE
        self.selected_augmentations = selected_augmentations or SELECTED_AUGMENTATIONS
        
        # Validate input directory
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        self.setup_directories()
        print(f"🎯 Single-shot augmentation configured:")
        print(f"   Input: {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Target shape: {self.input_shape} (height={INPUT_HEIGHT}, width={INPUT_WIDTH}, channels={INPUT_CHANNELS})")
        print(f"   Grayscale: {self.use_grayscale}")
        print(f"   Configuration: {self.config.config_path}")
        
        if self.selected_augmentations:
            print(f"   Selected augmentations: {', '.join(self.selected_augmentations)}")
        else:
            enabled_augs = self.config.get_enabled_augmentations()
            print(f"   Enabled augmentations: {', '.join(enabled_augs)}")
    
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
    
    def load_images_from_folder(self, folder_path):
        """Load all images from a folder with their paths and labels"""
        images = []
        labels = []
        image_paths = []
        
        supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        
        for format in supported_formats:
            for img_path in glob.glob(os.path.join(folder_path, format)):
                try:
                    # Load image - OpenCV loads as BGR by default
                    if self.use_grayscale:
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        # Keep as BGR - don't convert to RGB
                    
                    if image is None:
                        print(f"⚠️  Could not load image: {img_path}")
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
            # Convert grayscale to BGR (3 channels)
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and self.use_grayscale and image.shape[2] == 3:
            # Convert BGR to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
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
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
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
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
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
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
        if shear is None:
            shear = random.uniform(-AUG_SHEAR_RANGE, AUG_SHEAR_RANGE)
        
        image = self.ensure_correct_shape(image)
        h, w = image.shape[:2]  # h = height, w = width
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)  # width, height
        return self.ensure_correct_shape(sheared)
    
    def adjust_brightness(self, image, factor=None):
        """Adjust image brightness"""
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
        if factor is None:
            factor = random.uniform(AUG_BRIGHTNESS_RANGE[0], AUG_BRIGHTNESS_RANGE[1])
        
        image = self.ensure_correct_shape(image)
        brightened = np.clip(image * factor, 0, 1)
        return self.ensure_correct_shape(brightened)
    
    def adjust_contrast(self, image, factor=None):
        """Adjust image contrast"""
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
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
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
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
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
        image = self.ensure_correct_shape(image)
        noise = np.random.normal(0, AUG_GAUSSIAN_NOISE_STD, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
        return self.ensure_correct_shape(noisy_image)
    
    def apply_random_erasing(self, image):
        """Apply random erasing/cutout"""
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
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
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
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
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
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
        """Apply flashlight disturbance - bright white spot overexposing the image"""
        # Reset random seed for this function
        random.seed()
        np.random.seed()
        
        if not AUG_FLASHLIGHT_DISTURBANCE:
            return image
        
        # Random chance to apply this augmentation
        if random.random() > AUG_FLASHLIGHT_PROBABILITY:
            return image
        
        # Ensure correct shape and make a copy
        image = self.ensure_correct_shape(image)
        disturbed = image.copy()
        h, w = disturbed.shape[:2]  # h = height, w = width
        
        # Calculate target affected area (1/4 of image)
        target_area = AUG_FLASHLIGHT_AFFECTED_AREA * (h * w)
        
        # Determine flashlight radius to achieve approximately 1/4 area coverage
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
        
        # Random intensity variation - use higher intensity for stronger overexposure
        intensity = random.uniform(AUG_FLASHLIGHT_INTENSITY * 0.8, AUG_FLASHLIGHT_INTENSITY)
        
        # Create coordinate grids for vectorized operations
        y_coords, x_coords = np.ogrid[:h, :w]  # y = height, x = width
        
        # Calculate distance from flashlight center for each pixel
        distance_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        if AUG_FLASHLIGHT_PROGRESSIVE:
            # Create smooth intensity falloff using Gaussian function
            sigma = flashlight_radius / 2.0  # Controls falloff steepness
            intensity_map = intensity * np.exp(-(distance_map**2) / (2 * sigma**2))
        else:
            # Sharp circular falloff
            intensity_map = np.where(distance_map <= flashlight_radius, intensity, 0)
        
        # Apply flashlight effect - ADD brightness to create overexposure
        if len(disturbed.shape) == 2:
            # Grayscale - add intensity to create bright white spot
            disturbed = np.clip(disturbed + intensity_map, 0, 1)
            
            # Ensure center is fully white for strong overexposure effect
            center_mask = distance_map <= (flashlight_radius * 0.3)
            disturbed[center_mask] = 1.0  # Pure white at center
            
        else:
            # Color - apply to all channels to create white light
            for channel in range(disturbed.shape[2]):
                channel_data = disturbed[:, :, channel]
                augmented_channel = np.clip(channel_data + intensity_map, 0, 1)
                
                # Ensure center is fully white for strong overexposure effect
                center_mask = distance_map <= (flashlight_radius * 0.3)
                augmented_channel[center_mask] = 1.0  # Pure white at center
                
                disturbed[:, :, channel] = augmented_channel
        
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
        
        return self.ensure_correct_shape(disturbed)
    
    def get_available_augmentations(self):
        """Get list of available augmentation functions based on selection and configuration"""
        if self.selected_augmentations:
            # Use selected augmentations
            aug_list = self.selected_augmentations
        else:
            # Use enabled augmentations from config
            aug_list = self.config.get_enabled_augmentations()
        
        available_augs = []
        for aug_name in aug_list:
            if aug_name in AUGMENTATION_MAP:
                method_name = AUGMENTATION_MAP[aug_name]
                method = getattr(self, method_name)
                
                # Get probability from configuration
                probability = self.config.get_augmentation_probability(aug_name)
                
                # Set augmentation type for grouping
                if aug_name in ['rotation', 'zoom', 'shift', 'shear', 'perspective']:
                    aug_type = 'spatial'
                elif aug_name in ['brightness', 'contrast', 'color_jitter', 'gaussian_noise']:
                    aug_type = 'color'
                else:
                    aug_type = 'other'
                
                available_augs.append((method, probability, aug_type, aug_name))
        
        return available_augs
    
    def augment_single_image(self, image, label, original_path, augmentation_id):
        """Apply conservative augmentation to a single image"""
        # Initialize random seed for this specific augmentation
        random.seed()  # Use system time for true randomness
        np.random.seed()  # Reset numpy random seed
        
        image = self.ensure_correct_shape(image)
        augmented = image.copy()
        
        # Get available augmentations based on selection and configuration
        all_augmentations = self.get_available_augmentations()
        
        if not all_augmentations:
            print("⚠️  No augmentations selected or available!")
            return augmented
        
        # Separate augmentations by type
        spatial_augmentations = [(func, prob, name) for func, prob, aug_type, name in all_augmentations if aug_type == 'spatial']
        color_augmentations = [(func, prob, name) for func, prob, aug_type, name in all_augmentations if aug_type == 'color']
        other_augmentations = [(func, prob, name) for func, prob, aug_type, name in all_augmentations if aug_type == 'other']
        
        # Apply spatial augmentations (at most 2)
        applied_spatial = 0
        random.shuffle(spatial_augmentations)
        for aug_func, probability, aug_name in spatial_augmentations:
            if applied_spatial < 2 and random.random() < probability:
                try:
                    augmented = aug_func(augmented)
                    applied_spatial += 1
                    # print(f"Applied spatial augmentation: {aug_name}")  # Debug
                except Exception as e:
                    # print(f"Failed to apply {aug_name}: {e}")  # Debug
                    continue
        
        # Apply color augmentations (at most 1)
        if color_augmentations and random.random() < 0.5:  # 50% chance of any color augmentation
            random.shuffle(color_augmentations)
            for aug_func, probability, aug_name in color_augmentations:
                if random.random() < probability:
                    try:
                        augmented = aug_func(augmented)
                        # print(f"Applied color augmentation: {aug_name}")  # Debug
                        break  # Only apply one color augmentation
                    except Exception as e:
                        # print(f"Failed to apply {aug_name}: {e}")  # Debug
                        continue
        
        # Apply other augmentations (10% chance)
        if other_augmentations and random.random() < 0.1:
            random.shuffle(other_augmentations)
            aug_func, probability, aug_name = random.choice(other_augmentations)
            if random.random() < probability:
                try:
                    augmented = aug_func(augmented)
                    # print(f"Applied other augmentation: {aug_name}")  # Debug
                except Exception as e:
                    # print(f"Failed to apply {aug_name}: {e}")  # Debug
                    pass
        
        augmented = np.clip(augmented, 0, 1)
        augmented = self.ensure_correct_shape(augmented)
        
        # Save augmented image
        if SAVE_AUGMENTED_IMAGES:
            self.save_augmented_image(augmented, label, original_path, augmentation_id)
        
        return augmented
    
    def save_augmented_image(self, image, label, original_path, augmentation_id):
        """Save augmented image to disk preserving folder structure"""
        # Get relative path from input directory
        relative_path = os.path.relpath(os.path.dirname(original_path), self.input_dir)
        aug_dir = os.path.join(self.output_dir, relative_path)
        os.makedirs(aug_dir, exist_ok=True)
        
        # Create filename - use JPG
        original_filename = os.path.splitext(os.path.basename(original_path))[0]
        filename = f"{original_filename}_{augmentation_id:03d}_aug.jpg"
        filepath = os.path.join(aug_dir, filename)
        
        # Convert from [0,1] float to [0,255] uint8
        if image.max() <= 1.0:
            save_image = (image * 255).astype(np.uint8)
        else:
            save_image = image.astype(np.uint8)
        
        # Remove channel dimension for grayscale if needed
        if len(save_image.shape) == 3 and save_image.shape[2] == 1:
            save_image = save_image[:, :, 0]
        
        # Save as JPEG with high quality
        cv2.imwrite(filepath, save_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    def run_augmentation(self):
        """Run single-shot augmentation on entire dataset"""
        print("🚀 Starting single-shot augmentation...")
        
        # Load dataset
        images, labels, paths = self.load_entire_dataset()
        
        # Run augmentation
        total_original = len(images)
        total_augmented = total_original * AUGMENTATION_MULTIPLIER
        
        # Get enabled augmentations for display
        enabled_augs = self.selected_augmentations or self.config.get_enabled_augmentations()
        
        print(f"🔧 Augmentation settings:")
        print(f"   Multiplier: {AUGMENTATION_MULTIPLIER}x")
        print(f"   Expected output: {total_augmented} images")
        print(f"   Threads: {AUGMENTATION_THREADS}")
        print(f"   Target shape: {INPUT_HEIGHT}x{INPUT_WIDTH}x{INPUT_CHANNELS}")
        print(f"   Configuration file: {self.config.config_path}")
        print(f"   Enabled augmentations: {', '.join(enabled_augs)}")
        
        # Display augmentation probabilities
        print(f"   Augmentation probabilities:")
        for aug_name in enabled_augs:
            prob = self.config.get_augmentation_probability(aug_name)
            print(f"     - {aug_name}: {prob:.1%}")
        
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


def configure_augmentations(selected_augs):
    """Configure global augmentation settings based on selected augmentations"""
    global SELECTED_AUGMENTATIONS
    SELECTED_AUGMENTATIONS = selected_augs
    
    # Print available augmentations if help is requested
    if 'help' in selected_augs or 'list' in selected_augs:
        print("📋 Available augmentations:")
        for aug_name, method_name in AUGMENTATION_MAP.items():
            print(f"   - {aug_name}")
        print("\n💡 Usage examples:")
        print("   python augmentation.py --augmentations rotation zoom shift")
        print("   python augmentation.py --augmentations brightness contrast gaussian_noise")
        print("   python augmentation.py --augmentations flashlight perspective")
        print("   python augmentation.py --config custom_params.json")
        return True
    return False


def main():
    """Command line interface for single-shot augmentation"""
    parser = argparse.ArgumentParser(description='Single-shot dataset augmentation')
    parser.add_argument('--input', type=str, default=None,
                       help='Input dataset directory (default: from config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: [input]_augmented)')
    parser.add_argument('--multiplier', type=int, default=None,
                       help='Augmentation multiplier (default: from config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results (default: random)')
    parser.add_argument('--augmentations', type=str, nargs='+', default=None,
                       help='List of augmentations to apply (e.g., rotation zoom shift). Available: rotation, zoom, shift, shear, brightness, contrast, color_jitter, gaussian_noise, random_erasing, random_crop, perspective, flashlight. Use "help" or "list" to see all options.')
    parser.add_argument('--config', type=str, default='augmentation_params.json',
                       help='Path to JSON configuration file (default: augmentation_params.json)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a default configuration file and exit')
    
    args = parser.parse_args()
    
    # Create config file if requested
    if args.create_config:
        config = AugmentationConfig()
        config.save_config(args.config)
        return
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🔧 Using random seed: {args.seed}")
    else:
        # Ensure true randomness
        random.seed()
        np.random.seed()
        print("🔧 Using system time for random seed")
    
    # Handle augmentation selection
    selected_augs = None
    if args.augmentations:
        # Convert to lowercase and remove duplicates
        selected_augs = list(set([aug.lower() for aug in args.augmentations]))
        
        # Check if help is requested
        if configure_augmentations(selected_augs):
            return
    
    try:
        # Run augmentation with configuration
        augmentor = SingleShotAugmentor(
            input_dir=args.input, 
            output_dir=args.output, 
            selected_augmentations=selected_augs,
            config_path=args.config
        )
        
        # Override multiplier from command line if provided
        if args.multiplier is not None:
            globals()['AUGMENTATION_MULTIPLIER'] = args.multiplier
        
        augmentor.run_augmentation()
        
    except Exception as e:
        print(f"❌ Augmentation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()