# preprocess.py
import cv2
import numpy as np
import tensorflow as tf
import parameters as params

def validate_quantization_combination():
    """Validate quantization parameters with CORRECTED ESP-DL understanding"""
    valid = True
    message = ""
    
    if not params.QUANTIZE_MODEL:
        if params.ESP_DL_QUANTIZE:
            valid = False
            message = "❌ INVALID: ESP_DL_QUANTIZE=True requires QUANTIZE_MODEL=True"
        elif params.USE_QAT:
            message = "⚠️  QAT training but no quantization applied (QUANTIZE_MODEL=False)"
        else:
            message = "✅ Float32 training & inference [0, 1]"
    else:
        if params.USE_QAT:
            if params.ESP_DL_QUANTIZE:
                message = "✅ QAT + INT8 quantization for ESP-DL [0, 255]"
            else:
                message = "✅ QAT + UINT8 quantization [0, 1]"
        else:
            if params.ESP_DL_QUANTIZE:
                message = "✅ Standard training + INT8 post-quantization (ESP-DL) [0, 255]"
            else:
                message = "✅ Standard training + UINT8 post-quantization [0, 1]"
    
    return valid, message


def preprocess_images(images, target_size=None, grayscale=None, for_training=True):
    """
    Preprocess images for digit recognition training and inference.
    
    This function handles all quantization scenarios by separating training 
    and inference preprocessing logic. Training always uses a consistent 
    format, while inference format depends on the target quantization.
    
    Args:
        images (numpy.ndarray): Input images in any format (uint8/float32)
        target_size (tuple, optional): Target size (width, height). 
            Defaults to parameters.INPUT_WIDTH, parameters.INPUT_HEIGHT
        grayscale (bool, optional): Convert to grayscale. 
            Defaults to parameters.USE_GRAYSCALE
        for_training (bool): Determines preprocessing mode:
            - True: Training mode - consistent format for stable training
            - False: Inference mode - format optimized for target deployment
    
    Returns:
        numpy.ndarray: Preprocessed images in the appropriate format
        
    Preprocessing Behavior:
    
    for_training=True (Training Mode):
        - Always returns float32 [0, 1] regardless of quantization settings
        - Ensures stable and consistent training across all configurations
        - Used during model training and validation
        - Format: float32, range [0.0, 1.0]
    
    for_training=False (Inference Mode):
        Returns format based on quantization target:
        
        - ESP-DL Quantization (QUANTIZE_MODEL=True, ESP_DL_QUANTIZE=True):
          Format: uint8 [0, 255]
          Used for: ESP-DL microcontroller deployment with INT8 quantization
        
        - Standard Quantization (QUANTIZE_MODEL=True, ESP_DL_QUANTIZE=False):
          Format: float32 [0, 1] 
          Used for: Standard TFLite UINT8 quantization
        
        - No Quantization (QUANTIZE_MODEL=False):
          Format: float32 [0, 1]
          Used for: Float32 TFLite models or direct inference
    
    Examples:
        >>> # Training data preprocessing
        >>> x_train = preprocess_images(x_train_raw, for_training=True)
        >>> print(f"Training range: [{x_train.min():.3f}, {x_train.max():.3f}]")
        Training range: [0.000, 1.000]
        
        >>> # Inference data preprocessing for ESP-DL
        >>> x_infer = preprocess_images(x_test_raw, for_training=False)
        >>> print(f"ESP-DL range: [{x_infer.min()}, {x_infer.max()}]")
        ESP-DL range: [0, 255]
        
        >>> # Representative dataset for quantization
        >>> rep_data = preprocess_images(calib_data, for_training=False)
    
    Notes:
        - Prevents double preprocessing by using consistent logic
        - Maintains compatibility with QAT (Quantization Aware Training)
        - Handles both grayscale and RGB input formats
        - Automatically resizes images to target dimensions
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
    
    # Convert to numpy array first
    processed_images = np.array(processed_images)
    
    # CRITICAL: Use for_training parameter to determine preprocessing
    if for_training:
        # TRAINING DATA - Always use float32 [0, 1] regardless of quantization settings
        if processed_images.dtype != np.float32:
            processed_images = processed_images.astype(np.float32)
        if processed_images.max() > 1.0:
            processed_images = processed_images / 255.0
        
        quantization_info = "Float32 [0, 1] (Training)"
        
    else:
        # INFERENCE DATA - Depends on quantization target
        if params.QUANTIZE_MODEL:
            if params.ESP_DL_QUANTIZE:
                # ESP-DL INT8 quantization: Use UINT8 [0, 255]
                if processed_images.dtype != np.uint8:
                    if processed_images.max() <= 1.0:
                        processed_images = (processed_images * 255).astype(np.uint8)
                    else:
                        processed_images = processed_images.astype(np.uint8)
                quantization_info = "UINT8 [0, 255] to ESP-DL INT8"
            else:
                # Standard TFLite UINT8 quantization: Use float32 [0, 1]
                if processed_images.dtype != np.float32:
                    processed_images = processed_images.astype(np.float32)
                if processed_images.max() > 1.0:
                    processed_images = processed_images / 255.0
                quantization_info = "Float32 [0, 1] to TFLite UINT8"
        else:
            # No quantization: Use float32 [0, 1]
            if processed_images.dtype != np.float32:
                processed_images = processed_images.astype(np.float32)
            if processed_images.max() > 1.0:
                processed_images = processed_images / 255.0
            quantization_info = "Float32 [0, 1] (No quantization)"
    
    print(f"DEBUG: Preprocessing - {quantization_info}")
    print(f"   Range: [{processed_images.min():.3f}, {processed_images.max():.3f}]")
    print(f"   dtype: {processed_images.dtype}")
    
    return processed_images




def preprocess_images_esp_dl(images, target_size=None):
    """
    ESP-DL specific preprocessing - CORRECTED
    Returns UINT8 [0, 255] images as ESP-DL expects
    """
    if target_size is None:
        target_size = (params.INPUT_WIDTH, params.INPUT_HEIGHT)
    
    processed_images = []
    
    for image in images:
        # Resize to target size
        image = cv2.resize(image, target_size)
        
        # Convert to grayscale if required
        if params.USE_GRAYSCALE and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif not params.USE_GRAYSCALE and len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Add channel dimension if missing
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        processed_images.append(image)
    
    # ESP-DL expects UINT8 [0, 255]
    return np.array(processed_images, dtype=np.uint8)

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
    print("\n🔍 VALIDATING PREPROCESSING CONSISTENCY")
    print("=" * 60)
    
    # First validate the quantization combination
    is_valid, msg = validate_quantization_combination()
    if not is_valid:
        print(f"❌ {msg}")
        return False
        
    print(f"✅ {msg}")
    
    # Create test data
    test_images = np.random.randint(0, 255, (5, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS), dtype=np.uint8)
    
    # Test training preprocessing
    train_processed = preprocess_images(test_images, for_training=True)
    
    # Test inference preprocessing  
    infer_processed = preprocess_images(test_images, for_training=False)
    
    # Determine expected ranges
    if params.USE_QAT and params.ESP_DL_QUANTIZE:
        expected_train_range = (-1.0, 1.0)
        expected_infer_range = (-1.0, 1.0) if params.QUANTIZE_MODEL else (0.0, 1.0)
    elif params.USE_QAT and not params.ESP_DL_QUANTIZE:
        expected_train_range = (0.0, 1.0)
        expected_infer_range = (0.0, 1.0)
    else:
        expected_train_range = (0.0, 1.0)
        if params.QUANTIZE_MODEL and params.ESP_DL_QUANTIZE:
            expected_infer_range = (-1.0, 1.0)
        else:
            expected_infer_range = (0.0, 1.0)
    
    train_min, train_max = train_processed.min(), train_processed.max()
    infer_min, infer_max = infer_processed.min(), infer_processed.max()
    
    print(f"\n📊 Training Preprocessing:")
    print(f"   Expected: [{expected_train_range[0]}, {expected_train_range[1]}]")
    print(f"   Actual:   [{train_min:.3f}, {train_max:.3f}]")
    
    print(f"\n📊 Inference Preprocessing:")
    print(f"   Expected: [{expected_infer_range[0]}, {expected_infer_range[1]}]")
    print(f"   Actual:   [{infer_min:.3f}, {infer_max:.3f}]")
    
    # Check if ranges are correct (with tolerance)
    train_ok = (train_min >= expected_train_range[0] - 1e-6 and 
                train_max <= expected_train_range[1] + 1e-6)
    infer_ok = (infer_min >= expected_infer_range[0] - 1e-6 and 
                infer_max <= expected_infer_range[1] + 1e-6)
    
    if train_ok and infer_ok:
        print("\n✅ Preprocessing consistency: VALID")
        
        # Additional QAT-specific validation
        if params.USE_QAT:
            print("💡 QAT Configuration Validated:")
            if params.QUANTIZE_MODEL:
                print("   - Using deployment-compatible preprocessing")
                print("   - Fake quantization will be applied during training")
                print("   - Model should quantize well to TFLite")
            else:
                print("   ⚠️  QAT enabled but QUANTIZE_MODEL=False")
                print("   💡 Quantization won't be applied to final model")
    else:
        print("\n❌ Preprocessing consistency: INVALID")
        if not train_ok:
            print("   - Training preprocessing range incorrect")
        if not infer_ok:
            print("   - Inference preprocessing range incorrect")
    
    print("=" * 50)
    return train_ok and infer_ok

def get_preprocessing_info():
    """
    Get current preprocessing configuration including QAT status
    """
    # Determine current mode
    if not params.QUANTIZE_MODEL:
        if params.USE_QAT:
            mode = "QAT Training (No quantization applied)"
            if params.ESP_DL_QUANTIZE:
                normalization = "[-1,1] (simulating ESP-DL)"
            else:
                normalization = "[0,1] (simulating UINT8)"
        else:
            mode = "Float32 Training"
            normalization = "[0,1]"
    else:
        if params.USE_QAT:
            if params.ESP_DL_QUANTIZE:
                mode = "QAT + INT8 ESP-DL Quantization"
                normalization = "Training: [-1,1], Inference: [-1,1]"
            else:
                mode = "QAT + UINT8 Quantization" 
                normalization = "Training: [0,1], Inference: [0,1]"
        else:
            if params.ESP_DL_QUANTIZE:
                mode = "Standard Training + INT8 ESP-DL Quantization"
                normalization = "Training: [0,1], Inference: [-1,1]"
            else:
                mode = "Standard Training + UINT8 Quantization"
                normalization = "Training: [0,1], Inference: [0,1]"
    
    return {
        'quantize_model': params.QUANTIZE_MODEL,
        'use_qat': params.USE_QAT,
        'esp_dl_quantize': params.ESP_DL_QUANTIZE,
        'mode': mode,
        'normalization': normalization,
        'input_shape': params.INPUT_SHAPE,
        'recommendation': 'QAT compatible' if params.USE_QAT else 'Standard training'
    }

def predict_single_image(image):
    """
    Preprocess a single image for inference
    Uses the same preprocessing as training for consistency
    """
    return preprocess_images([image], for_training=False)[0]

def test_all_preprocessing_combinations():
    """
    Test all 9 preprocessing combinations for verification
    """
    print("\n🧪 TESTING ALL 9 PREPROCESSING COMBINATIONS")
    print("=" * 70)
    
    # Save original values
    original_quantize = params.QUANTIZE_MODEL
    original_qat = params.USE_QAT
    original_esp_dl = params.ESP_DL_QUANTIZE
    
    test_combinations = [
        (False, False, False, "Float32 training & inference"),
        (False, False, True,  "INVALID: ESP-DL without quantization"),
        (False, True,  False, "QAT training, float32 inference"), 
        (False, True,  True,  "INVALID: ESP-DL without quantization"),
        (True,  False, False, "Standard training, UINT8 quantization"),
        (True,  False, True,  "Standard training, INT8 quantization (ESP-DL)"),
        (True,  True,  False, "QAT training, UINT8 quantization"),
        (True,  True,  True,  "QAT training, INT8 quantization (ESP-DL)"),
    ]
    
    results = {}
    
    for quantize, qat, esp_dl, description in test_combinations:
        print(f"\n🔍 Testing: {description}")
        print(f"   QUANTIZE_MODEL={quantize}, USE_QAT={qat}, ESP_DL_QUANTIZE={esp_dl}")
        
        # Set test combination
        params.QUANTIZE_MODEL = quantize
        params.USE_QAT = qat
        params.ESP_DL_QUANTIZE = esp_dl
        
        # Skip invalid combinations
        if not quantize and esp_dl:
            print("   ❌ SKIPPED: Invalid combination")
            results[description] = "INVALID"
            continue
            
        try:
            # Create test data
            test_images = np.random.randint(0, 255, (3, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS), dtype=np.uint8)
            
            # Test training preprocessing
            train_processed = preprocess_images(test_images, for_training=True)
            train_range = f"[{train_processed.min():.3f}, {train_processed.max():.3f}]"
            
            # Test inference preprocessing  
            infer_processed = preprocess_images(test_images, for_training=False)
            infer_range = f"[{infer_processed.min():.3f}, {infer_processed.max():.3f}]"
            
            print(f"   ✅ Training: {train_range}, Inference: {infer_range}")
            results[description] = f"SUCCESS - Train: {train_range}, Infer: {infer_range}"
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results[description] = f"FAILED - {e}"
    
    # Restore original values
    params.QUANTIZE_MODEL = original_quantize
    params.USE_QAT = original_qat
    params.ESP_DL_QUANTIZE = original_esp_dl
    
    print("\n" + "=" * 70)
    print("📊 PREPROCESSING TEST RESULTS:")
    print("=" * 70)
    for desc, result in results.items():
        print(f"  {desc:45} : {result}")
    
    return results

# QAT-specific helper functions
def check_qat_compatibility(qat_available):
    """
    Check if current settings are compatible with QAT
    Returns: (is_compatible, warnings, errors, info)
    """
    warnings = []
    errors = []
    info = []
    
    # Critical errors that prevent QAT from working
    if params.USE_QAT and not qat_available:
        errors.append("QAT requires tensorflow-model-optimization package. Install with: pip install tensorflow-model-optimization")
    
    # Warnings (things that might affect performance but won't break QAT)
    if params.USE_QAT and not params.QUANTIZE_MODEL:
        warnings.append("QAT is enabled but QUANTIZE_MODEL is False - quantization won't be applied to final model")
    
    if params.USE_QAT and params.USE_DATA_AUGMENTATION:
        warnings.append("Data augmentation with QAT: Ensure augmentations don't significantly change data distribution")
    
    # Info messages for best practices
    if params.USE_QAT and qat_available:
        info.append("✅ QAT compatible - model will be trained with quantization awareness")
        if params.QUANTIZE_MODEL:
            info.append("✅ Post-training quantization will be applied after QAT")
        else:
            info.append("⚠️  Post-training quantization disabled - QAT benefits may not be realized")
    
    return len(errors) == 0, warnings, errors, info

if __name__ == "__main__":
    # Test the current configuration
    validate_preprocessing_consistency()
    
    # Show current preprocessing info
    info = get_preprocessing_info()
    print(f"\n📋 CURRENT PREPROCESSING CONFIGURATION:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Optionally test all combinations
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-all', action='store_true', help='Test all 9 preprocessing combinations')
    args = parser.parse_args()
    
    if args.test_all:
        test_all_preprocessing_combinations()
        
def diagnose_quantization_settings():
    """Diagnose current quantization settings and suggest fixes"""
    print("\n🔍 QUANTIZATION SETTINGS DIAGNOSIS")
    print("=" * 50)
    
    issues = []
    suggestions = []
    
    # Check individual parameters
    print(f"QUANTIZE_MODEL: {params.QUANTIZE_MODEL}")
    print(f"USE_QAT: {params.USE_QAT}")
    print(f"ESP_DL_QUANTIZE: {params.ESP_DL_QUANTIZE}")
    
    # Check combinations
    if params.ESP_DL_QUANTIZE and not params.QUANTIZE_MODEL:
        issues.append("ESP_DL_QUANTIZE requires QUANTIZE_MODEL")
        suggestions.append("Set QUANTIZE_MODEL = True")
    
    if params.USE_QAT and not params.QUANTIZE_MODEL:
        issues.append("USE_QAT requires QUANTIZE_MODEL")
        suggestions.append("Set QUANTIZE_MODEL = True")
    
    if params.USE_QAT and params.ESP_DL_QUANTIZE:
        print("✅ QAT + ESP-DL: Training for INT8 quantization")
    
    elif params.USE_QAT and not params.ESP_DL_QUANTIZE:
        print("✅ QAT only: Training for UINT8 quantization")
    
    elif not params.USE_QAT and params.ESP_DL_QUANTIZE:
        print("✅ ESP-DL only: Standard training + INT8 post-quantization")
    
    elif not params.USE_QAT and params.QUANTIZE_MODEL and not params.ESP_DL_QUANTIZE:
        print("✅ Standard quantization: Training + UINT8 post-quantization")
    
    else:
        print("✅ Float32: No quantization")
    
    # Print issues and suggestions
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 SUGGESTIONS:")
        for suggestion in suggestions:
            print(f"   - {suggestion}")
    else:
        print("\n✅ No parameter conflicts detected")
    
    return len(issues) == 0
    

def debug_preprocessing_flow():
    """Debug function to trace preprocessing flow and detect double processing"""
    print("\n🔍 DEBUG: Tracing Preprocessing Flow")
    print("=" * 50)
    
    # Create test data
    test_images_raw = np.random.randint(0, 255, (2, 28, 28, 1), dtype=np.uint8)
    print(f"Raw data range: [{test_images_raw.min()}, {test_images_raw.max()}]")
    print(f"Raw data dtype: {test_images_raw.dtype}")
    
    # Simulate preprocessing
    test_processed = preprocess_images(test_images_raw, for_training=True)
    print(f"After preprocessing: [{test_processed.min():.3f}, {test_processed.max():.3f}]")
    print(f"After preprocessing dtype: {test_processed.dtype}")
    
    # Check if this matches expected range
    expected_range = ""
    if params.QUANTIZE_MODEL and params.ESP_DL_QUANTIZE:
        expected_range = "UINT8 [0, 255]"
        if test_processed.dtype == np.uint8 and test_processed.max() > 1.0:
            print("✅ Preprocessing correct for ESP-DL quantization")
        else:
            print("❌ Preprocessing incorrect for ESP-DL quantization")
    elif params.QUANTIZE_MODEL:
        expected_range = "Float32 [0, 1]"
        if test_processed.dtype == np.float32 and test_processed.max() <= 1.0:
            print("✅ Preprocessing correct for standard quantization")
        else:
            print("❌ Preprocessing incorrect for standard quantization")
    else:
        expected_range = "Float32 [0, 1]"
        if test_processed.dtype == np.float32 and test_processed.max() <= 1.0:
            print("✅ Preprocessing correct for float32 training")
        else:
            print("❌ Preprocessing incorrect for float32 training")
    
    print(f"Expected range: {expected_range}")
    
    # Additional check for double preprocessing
    if test_processed.max() < 0.1 and test_processed.dtype == np.float32:
        print("🚨 WARNING: Possible double preprocessing detected!")
        print("   Data range suggests over-normalization (should be [0, 1], not [0, ~0.0039])")
    
    return test_processed