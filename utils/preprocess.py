# preprocess.py
import cv2
import numpy as np
import tensorflow as tf
import parameters as params

def validate_quantization_combination():
    """Validate all 9 possible combinations of quantization parameters"""
    valid = True
    message = ""
    
    if not params.QUANTIZE_MODEL:
        if params.ESP_DL_QUANTIZE:
            valid = False
            message = "❌ INVALID: ESP_DL_QUANTIZE=True requires QUANTIZE_MODEL=True"
        elif params.USE_QAT:
            message = "⚠️  QAT training but no quantization applied (QUANTIZE_MODEL=False)"
        else:
            message = "✅ Float32 training & inference"
    else:
        if params.USE_QAT:
            if params.ESP_DL_QUANTIZE:
                message = "✅ QAT + INT8 quantization for ESP-DL"
            else:
                message = "✅ QAT + UINT8 quantization"
        else:
            if params.ESP_DL_QUANTIZE:
                message = "✅ Standard training + INT8 post-quantization (ESP-DL)"
            else:
                message = "✅ Standard training + UINT8 post-quantization"
    
    return valid, message

def preprocess_images(images, target_size=None, grayscale=None, for_training=True):
    """
    Universal preprocessing that handles all 9 quantization combinations
    
    Args:
        images: Input images (numpy array)
        target_size: Target image size (width, height)
        grayscale: Convert to grayscale
        for_training: True for training, False for inference/conversion
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
    
    # UNIVERSAL PREPROCESSING LOGIC FOR ALL 9 CASES
    if for_training:
        # TRAINING: Handle QAT vs standard training
        if params.USE_QAT and params.ESP_DL_QUANTIZE:
            # QAT + ESP-DL: Use [-1, 1] for consistent quantization simulation
            processed_images = (processed_images / 127.5) - 1.0
            quantization_info = "QAT + ESP-DL Training [-1, 1]"
        elif params.USE_QAT and not params.ESP_DL_QUANTIZE:
            # QAT + Standard: Use [0, 1] for consistent quantization simulation
            processed_images = processed_images / 255.0
            quantization_info = "QAT + Standard Training [0, 1]"
        else:
            # Standard training (no QAT): Always use [0, 1]
            processed_images = processed_images / 255.0
            quantization_info = "Standard Training [0, 1]"
    else:
        # INFERENCE/CONVERSION: Adjust based on target quantization
        if params.QUANTIZE_MODEL and params.ESP_DL_QUANTIZE:
            # ESP-DL INT8: [-1, 1] range
            processed_images = (processed_images / 127.5) - 1.0
            quantization_info = "ESP-DL INT8 Inference [-1, 1]"
        else:
            # Standard UINT8 or Float32: [0, 1] range  
            processed_images = processed_images / 255.0
            quantization_info = "Standard Inference [0, 1]"
    
    # Add data validation
    if processed_images.std() < 0.01:
        print(f"⚠️  WARNING: Low data variance - std={processed_images.std():.6f}")
    
    print(f"🔍 Preprocessing - {quantization_info}")
    print(f"   QAT: {params.USE_QAT}, ESP-DL: {params.ESP_DL_QUANTIZE}, Quantize: {params.QUANTIZE_MODEL}")
    print(f"   Range: [{processed_images.min():.3f}, {processed_images.max():.3f}]")
    print(f"   Shape: {processed_images.shape}, Mean: {processed_images.mean():.3f}")
    
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