# preprocess.py
import cv2
import numpy as np
import tensorflow as tf
import parameters as params

def preprocess_images(images, target_size=None, grayscale=None, for_training=True):
    """
    Preprocess images with QAT and quantization awareness
    
    Args:
        for_training: If True, handle QAT appropriately
                     If False, use inference preprocessing
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
    
    # Convert to numpy array
    processed_images = np.array(processed_images, dtype=np.float32)
    
    # CRITICAL: QAT-COMPATIBLE PREPROCESSING
    if params.USE_QAT:
        # For Quantization Aware Training: Use the same range as deployment
        if params.ESP_DL_QUANTIZE:
            # ESP-DL INT8: normalize to [-1, 1] for QAT
            processed_images = (processed_images / 127.5) - 1.0
            quantization_type = "QAT + ESP-DL INT8 [-1, 1]"
        else:
            # Standard UINT8: normalize to [0, 1] for QAT
            processed_images = processed_images / 255.0
            quantization_type = "QAT + Standard UINT8 [0, 1]"
    else:
        # For standard training (no QAT)
        if params.ESP_DL_QUANTIZE:
            # ESP-DL INT8: normalize to [-1, 1]
            processed_images = (processed_images / 127.5) - 1.0
            quantization_type = "Standard + ESP-DL INT8 [-1, 1]"
        else:
            # Standard UINT8: normalize to [0, 1]
            processed_images = processed_images / 255.0
            quantization_type = "Standard + UINT8 [0, 1]"
    
    # Add data validation
    if processed_images.std() < 0.01:
        print(f"⚠️  WARNING: Low data variance - std={processed_images.std():.6f}")
        print(f"   Sample values: {processed_images[0].flatten()[:10]}")
    
    # DEBUG: Print data range and QAT status
    print(f"🔍 Preprocessing - {quantization_type}")
    print(f"   QAT Enabled: {params.USE_QAT}")
    print(f"   Range: [{processed_images.min():.3f}, {processed_images.max():.3f}]")
    print(f"   Shape: {processed_images.shape}, Mean: {processed_images.mean():.3f}, Std: {processed_images.std():.3f}")
    
    return processed_images

def preprocess_images_esp_dl(images):
    """
    ESP-DL specific preprocessing for INT8 quantization
    Explicit function for conversion pipeline
    """
    return preprocess_images(images, for_training=False)

def preprocess_images_for_qat_calibration(images):
    """
    Special preprocessing for QAT model calibration
    Uses the same preprocessing as training but ensures consistency
    """
    print("🎯 QAT Calibration Preprocessing")
    return preprocess_images(images, for_training=True)

def validate_preprocessing_consistency():
    """
    Validate that preprocessing is consistent with QAT and quantization settings
    """
    print("\n🔍 VALIDATING QAT & PREPROCESSING CONSISTENCY")
    print("=" * 50)
    
    # Create test data
    test_images = np.random.randint(0, 255, (10, 32, 20, 1), dtype=np.uint8)
    
    # Process test data
    processed = preprocess_images(test_images)
    
    # Determine expected ranges based on QAT and quantization settings
    if params.USE_QAT:
        if params.ESP_DL_QUANTIZE:
            expected_min, expected_max = -1.0, 1.0
            config_info = "QAT + ESP-DL INT8 (range: [-1, 1])"
        else:
            expected_min, expected_max = 0.0, 1.0
            config_info = "QAT + Standard UINT8 (range: [0, 1])"
    else:
        if params.ESP_DL_QUANTIZE:
            expected_min, expected_max = -1.0, 1.0
            config_info = "Standard + ESP-DL INT8 (range: [-1, 1])"
        else:
            expected_min, expected_max = 0.0, 1.0
            config_info = "Standard + UINT8 (range: [0, 1])"
    
    actual_min, actual_max = processed.min(), processed.max()
    
    print(f"📊 Configuration: {config_info}")
    print(f"📊 QAT Enabled: {params.USE_QAT}")
    print(f"📊 ESP-DL Quantization: {params.ESP_DL_QUANTIZE}")
    print(f"📊 Expected Range: [{expected_min}, {expected_max}]")
    print(f"📊 Actual Range: [{actual_min:.3f}, {actual_max:.3f}]")
    print(f"📊 Data Shape: {processed.shape}")
    print(f"📊 Data Type: {processed.dtype}")
    
    # Check if range is correct
    range_ok = (abs(actual_min - expected_min) <= 0.1 and 
                abs(actual_max - expected_max) <= 0.1)
    
    if range_ok:
        print("✅ Preprocessing consistency: VALID")
        
        # Additional QAT-specific validation
        if params.USE_QAT:
            print("💡 QAT Configuration Validated:")
            print("   - Using deployment-compatible preprocessing")
            print("   - Fake quantization will be applied during training")
            print("   - Model should quantize well to TFLite")
    else:
        print("❌ Preprocessing consistency: INVALID")
        print("💡 Check your QAT and quantization settings in parameters.py")
    
    print("=" * 50)
    return range_ok

def get_preprocessing_info():
    """
    Get current preprocessing configuration including QAT status
    """
    if params.USE_QAT:
        if params.ESP_DL_QUANTIZE:
            normalization_range = '[-1, 1] (QAT + INT8)'
        else:
            normalization_range = '[0, 1] (QAT + UINT8)'
    else:
        if params.ESP_DL_QUANTIZE:
            normalization_range = '[-1, 1] (Standard + INT8)'
        else:
            normalization_range = '[0, 1] (Standard + UINT8)'
    
    return {
        'qat_enabled': params.USE_QAT,
        'esp_dl_quantize': params.ESP_DL_QUANTIZE,
        'quantize_model': params.QUANTIZE_MODEL,
        'input_shape': params.INPUT_SHAPE,
        'normalization_range': normalization_range,
        'recommendation': 'QAT compatible' if params.USE_QAT else 'Standard training'
    }

def predict_single_image(image):
    """
    Preprocess a single image for inference
    Uses the same preprocessing as training for consistency
    """
    return preprocess_images([image], for_training=False)[0]

# QAT-specific helper functions
def check_qat_compatibility():
    """
    Check if current settings are compatible with QAT
    """
    issues = []
    
    if params.USE_QAT and not params.QUANTIZE_MODEL:
        issues.append("QAT requires QUANTIZE_MODEL = True")
    
    if params.USE_QAT and not hasattr(tf, 'mot'):
        issues.append("QAT requires tensorflow-model-optimization package")
    
    if params.USE_QAT and params.USE_DATA_AUGMENTATION:
        print("⚠️  QAT with data augmentation: Ensure augmentation doesn't change data range significantly")
    
    return len(issues) == 0, issues