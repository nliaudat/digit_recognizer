# preprocess.py
import cv2
import numpy as np
import tensorflow as tf
import parameters as params

def preprocess_images(images, target_size=None, grayscale=None, for_training=True):
    """
    Preprocess images following QAT requirements
    
    Args:
        for_training: If True, normalize to [0,1] for training
                     If False, use the same for conversion (TFLite handles quantization)
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
    
    # Convert to numpy array as float32
    processed_images = np.array(processed_images, dtype=np.float32)
    
    # For QAT: Always normalize to [0, 1] range during training AND conversion
    # TFLite converter will handle the quantization scaling
    processed_images = processed_images / 255.0
    
    # Add data validation
    if processed_images.std() < 0.01:
        print(f"⚠️  WARNING: Low data variance - std={processed_images.std():.6f}")
        print(f"   Sample values: {processed_images[0].flatten()[:10]}")
    
    # DEBUG: Print data range
    print(f"🔍 Preprocessing - Range: [{processed_images.min():.3f}, {processed_images.max():.3f}], Shape: {processed_images.shape}")
    
    return processed_images
    
def predict_single_image(image):
    """
    Preprocess a single image for inference
    """
    return preprocess_images([image])[0]