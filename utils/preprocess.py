import cv2
import numpy as np
import tensorflow as tf
import parameters as params

def preprocess_images(images, target_size=None, grayscale=None):
    """
    Preprocess images for ESP-DL compatibility
    """
    if target_size is None:
        target_size = (params.INPUT_WIDTH, params.INPUT_HEIGHT)
    if grayscale is None:
        grayscale = params.USE_GRAYSCALE
    
    processed_images = []
    
    for image in images:
        # Resize to target size
        image = cv2.resize(image, target_size)
        
        # Convert to grayscale if required
        if grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif not grayscale and len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Add channel dimension if missing
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        processed_images.append(image)
    
    # Convert to numpy array and normalize
    processed_images = np.array(processed_images, dtype=np.float32) / 255.0
    
    # Ensure correct shape
    if grayscale and processed_images.shape[-1] != 1:
        processed_images = np.expand_dims(processed_images, axis=-1)
        processed_images = processed_images.mean(axis=-1, keepdims=True)
    
    return processed_images

def preprocess_single_image(image):
    """
    Preprocess a single image for inference
    """
    return preprocess_images([image])[0]