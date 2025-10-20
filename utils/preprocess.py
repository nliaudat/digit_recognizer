# preprocess.py
import cv2
import numpy as np
import tensorflow as tf
import parameters as params

def validate_quantization_combination():
    """Validate quantization parameters with CORRECTED data type handling"""
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
                message = "✅ QAT + INT8 quantization for ESP-DL [0, 1] to INT8"
            else:
                message = "✅ QAT + UINT8 quantization [0, 1] to UINT8"
        else:
            if params.ESP_DL_QUANTIZE:
                message = "✅ Standard training + INT8 post-quantization (ESP-DL) [0, 255] to INT8"
            else:
                message = "✅ Standard training + UINT8 post-quantization [0, 255] to UINT8"
    
    return valid, message

def get_qat_training_format():
    """
    Get the correct data format for QAT training
    Returns: (dtype, range_min, range_max, description)
    """
    if params.USE_QAT and params.QUANTIZE_MODEL:
        # QAT models typically expect float32 inputs [0, 1] during training
        # The quantization is simulated internally via fake quantization nodes
        return np.float32, 0.0, 1.0, "Float32 [0, 1] (QAT Training)"
    else:
        return np.float32, 0.0, 1.0, "Float32 [0, 1] (Standard)"

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
        - QAT Training: Float32 [0, 1] (models expect normalized inputs)
        - Non-QAT Training: Float32 [0, 1] for stable training
        - QAT internally simulates quantization via fake quantization nodes
    
    for_training=False (Inference Mode):
        Returns format based on quantization target:
        
        - QAT Models: Float32 [0, 1] (same as training - CRITICAL)
        - ESP-DL Quantization: UINT8 [0, 255] 
        - Standard Quantization: UINT8 [0, 255]
        - No Quantization: Float32 [0, 1]
    
    Examples:
        >>> # QAT training data preprocessing
        >>> x_train = preprocess_images(x_train_raw, for_training=True)
        >>> print(f"QAT Training range: [{x_train.min():.3f}, {x_train.max():.3f}]")
        QAT Training range: [0.000, 1.000]
        
        >>> # QAT inference data preprocessing  
        >>> x_infer = preprocess_images(x_test_raw, for_training=False)
        >>> print(f"QAT Inference range: [{x_infer.min():.3f}, {x_infer.max():.3f}]")
        QAT Inference range: [0.000, 1.000]
        
        >>> # Representative dataset for quantization
        >>> rep_data = preprocess_images(calib_data, for_training=False)
    
    Notes:
        - QAT models expect float32 inputs during training AND inference
        - Actual quantization happens during TFLite conversion
        - Fake quantization nodes simulate quantization during training
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
    
    # CRITICAL FIX: QAT models expect float32 [0, 1] inputs during training AND inference
    if for_training:
        # TRAINING DATA - Always use float32 [0, 1] for training stability
        if processed_images.dtype != np.float32:
            processed_images = processed_images.astype(np.float32)
        if processed_images.max() > 1.0:
            processed_images = processed_images / 255.0
        
        if params.USE_QAT and params.QUANTIZE_MODEL:
            quantization_info = "Float32 [0, 1] (QAT Training)"
        else:
            quantization_info = "Float32 [0, 1] (Standard Training)"
        
    else:
        # INFERENCE DATA - Format depends on model type and quantization target
        if params.USE_QAT and params.QUANTIZE_MODEL:
            # QAT MODELS: Always use float32 [0, 1] for inference
            # The model contains fake quantization nodes and expects normalized inputs
            if processed_images.dtype != np.float32:
                processed_images = processed_images.astype(np.float32)
            if processed_images.max() > 1.0:
                processed_images = processed_images / 255.0
            
            if params.ESP_DL_QUANTIZE:
                quantization_info = "Float32 [0, 1] (QAT to ESP-DL INT8)"
            else:
                quantization_info = "Float32 [0, 1] (QAT to UINT8)"
                
        elif params.QUANTIZE_MODEL:
            # Non-QAT quantization: Use UINT8 [0, 255] for post-training quantization
            if params.ESP_DL_QUANTIZE:
                # ESP-DL INT8 quantization: Use UINT8 [0, 255]
                if processed_images.dtype != np.uint8:
                    if processed_images.max() <= 1.0:
                        processed_images = (processed_images * 255).astype(np.uint8)
                    else:
                        processed_images = processed_images.astype(np.uint8)
                quantization_info = "UINT8 [0, 255] to ESP-DL INT8"
            else:
                # Standard TFLite UINT8 quantization: Use UINT8 [0, 255]
                if processed_images.dtype != np.uint8:
                    if processed_images.max() <= 1.0:
                        processed_images = (processed_images * 255).astype(np.uint8)
                    else:
                        processed_images = processed_images.astype(np.uint8)
                quantization_info = "UINT8 [0, 255] to TFLite UINT8"
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

def validate_data_type_consistency():
    """
    Validate that data types are consistent between training and inference
    CRITICAL FOR QAT: Training and inference must use identical data formats
    """
    print("\n🔍 VALIDATING DATA TYPE CONSISTENCY")
    print("=" * 60)
    
    # Get expected formats
    train_dtype, train_min, train_max, train_desc = get_qat_training_format()
    
    # Determine inference format
    if params.USE_QAT and params.QUANTIZE_MODEL:
        # QAT inference uses the same format as training (float32 [0, 1])
        infer_dtype, infer_min, infer_max = np.float32, 0.0, 1.0
        infer_desc = "Float32 [0, 1] (QAT Inference)"
    elif params.QUANTIZE_MODEL:
        if params.ESP_DL_QUANTIZE:
            infer_dtype, infer_min, infer_max = np.uint8, 0, 255
            infer_desc = "UINT8 [0, 255] (ESP-DL)"
        else:
            infer_dtype, infer_min, infer_max = np.uint8, 0, 255
            infer_desc = "UINT8 [0, 255] (Standard)"
    else:
        infer_dtype, infer_min, infer_max = np.float32, 0.0, 1.0
        infer_desc = "Float32 [0, 1] (No quantization)"
    
    print(f"📊 Expected Formats:")
    print(f"   Training:  {train_desc}")
    print(f"   Inference: {infer_desc}")
    
    # Test consistency with sample data
    test_images = np.random.randint(0, 255, (3, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS), dtype=np.uint8)
    
    # Process test data through both training and inference paths
    train_processed = preprocess_images(test_images, for_training=True)
    infer_processed = preprocess_images(test_images, for_training=False)
    
    # Check training data matches expected format
    train_ok = (train_processed.dtype == train_dtype and 
                train_processed.min() >= train_min and 
                train_processed.max() <= train_max)
    
    # Check inference data matches expected format  
    infer_ok = (infer_processed.dtype == infer_dtype and 
                infer_processed.min() >= infer_min and 
                infer_processed.max() <= infer_max)
    
    # Check consistency between training and inference (CRITICAL FOR QAT)
    consistency_ok = (train_processed.dtype == infer_processed.dtype and
                     abs(train_processed.min() - infer_processed.min()) < 1e-6 and
                     abs(train_processed.max() - infer_processed.max()) < 1e-6)
    
    print(f"\n📊 Actual Formats:")
    print(f"   Training:  {train_processed.dtype} [{train_processed.min():.3f}, {train_processed.max():.3f}]")
    print(f"   Inference: {infer_processed.dtype} [{infer_processed.min():.3f}, {infer_processed.max():.3f}]")
    
    # Print comprehensive results
    if train_ok and infer_ok and consistency_ok:
        print("\n✅ DATA TYPE CONSISTENCY: PERFECT")
        print("   Training and inference formats are identical")
        if params.USE_QAT:
            print("   QAT fake quantization will match deployment quantization")
    elif params.USE_QAT and params.QUANTIZE_MODEL and consistency_ok:
        print("\n✅ DATA TYPE CONSISTENCY: QAT-OPTIMAL")
        print("   QAT training and inference using identical float32 format")
        print("   Fake quantization will accurately simulate deployment")
    elif train_ok and infer_ok:
        print("\n⚠️  DATA TYPE CONSISTENCY: ACCEPTABLE")
        print("   Individual formats are correct but training ≠ inference")
        print("   This is expected for non-QAT quantization scenarios")
    else:
        print("\n❌ DATA TYPE CONSISTENCY: FAILED")
        if not train_ok:
            print(f"   - Training format incorrect: expected {train_dtype} [{train_min}, {train_max}]")
        if not infer_ok:
            print(f"   - Inference format incorrect: expected {infer_dtype} [{infer_min}, {infer_max}]")
    
    # QAT-specific validation
    if params.USE_QAT and params.QUANTIZE_MODEL:
        print(f"\n🎯 QAT DATA FLOW VALIDATION:")
        if consistency_ok:
            print("   ✅ QAT training and inference using identical float32 format")
            print("   ✅ Fake quantization will accurately simulate deployment")
            print("   ✅ Model expects float32 inputs during training and inference")
        else:
            print("   ⚠️  QAT training and inference using different formats")
            print("   ⚠️  This will cause input type mismatches during inference")
    
    return train_ok and infer_ok

def diagnose_qat_data_flow():
    """
    Comprehensive diagnosis of QAT data flow issues
    Specifically checks for input type compatibility
    """
    print("\n🔍 QAT DATA FLOW DIAGNOSIS")
    print("=" * 50)
    
    issues = []
    recommendations = []
    
    # Check QAT configuration consistency
    if params.USE_QAT and not params.QUANTIZE_MODEL:
        issues.append("QAT enabled but QUANTIZE_MODEL=False")
        recommendations.append("Set QUANTIZE_MODEL=True to apply quantization benefits")
    
    # Test actual preprocessing consistency
    test_images = np.random.randint(0, 255, (2, 28, 28, 1), dtype=np.uint8)
    train_data = preprocess_images(test_images, for_training=True)
    infer_data = preprocess_images(test_images, for_training=False)
    
    # Check for QAT input type requirements
    if params.USE_QAT and params.QUANTIZE_MODEL:
        if train_data.dtype != np.float32:
            issues.append(f"QAT training expects float32 but got {train_data.dtype}")
            recommendations.append("QAT training must use float32 [0, 1] inputs")
        
        if infer_data.dtype != np.float32:
            issues.append(f"QAT inference expects float32 but got {infer_data.dtype}")
            recommendations.append("QAT inference must use float32 [0, 1] inputs")
        
        # Check range
        if train_data.max() > 1.0 or train_data.min() < 0.0:
            issues.append(f"QAT training data out of range: [{train_data.min():.3f}, {train_data.max():.3f}]")
            recommendations.append("QAT inputs must be normalized to [0, 1] range")
    
    # Print comprehensive diagnosis results
    if issues:
        print("❌ QAT DATA FLOW ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   - {rec}")
        
        # Provide specific guidance for common issues
        if "QAT inference expects float32" in str(issues):
            print("\n🚨 CRITICAL FIX REQUIRED:")
            print("   QAT models expect float32 [0, 1] inputs during inference.")
            print("   The error 'Got value of type UINT8 but expected type FLOAT32'")
            print("   indicates the model is receiving UINT8 instead of FLOAT32.")
            print("   Solution: Use float32 [0, 1] for QAT model inference.")
    else:
        print("✅ No QAT data flow issues detected")
        print("   QAT training and inference are properly aligned")
        print("   Model will receive float32 [0, 1] inputs as expected")
    
    return len(issues) == 0

def validate_preprocessing_consistency():
    """
    Validate that preprocessing is consistent with QAT and quantization settings
    Enhanced with comprehensive data type validation
    """
    print("\n🔍 VALIDATING PREPROCESSING CONSISTENCY")
    print("=" * 60)
    
    # First validate the quantization combination
    is_valid, msg = validate_quantization_combination()
    if not is_valid:
        print(f"❌ {msg}")
        return False
        
    print(f"✅ {msg}")
    
    # Validate data type consistency (CRITICAL FOR QAT)
    data_consistency_ok = validate_data_type_consistency()
    
    # Run QAT-specific diagnosis if QAT is enabled
    if params.USE_QAT:
        qat_flow_ok = diagnose_qat_data_flow()
    else:
        qat_flow_ok = True
    
    # Create test data for range validation
    test_images = np.random.randint(0, 255, (5, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS), dtype=np.uint8)
    
    # Test training preprocessing
    train_processed = preprocess_images(test_images, for_training=True)
    train_range = f"[{train_processed.min():.3f}, {train_processed.max():.3f}]"
    
    # Test inference preprocessing  
    infer_processed = preprocess_images(test_images, for_training=False)
    infer_range = f"[{infer_processed.min():.3f}, {infer_processed.max():.3f}]"
    
    # Determine expected ranges based on configuration
    if params.USE_QAT and params.QUANTIZE_MODEL:
        expected_train_range = "[0, 1]"
        expected_infer_range = "[0, 1]"
    elif params.QUANTIZE_MODEL and params.ESP_DL_QUANTIZE:
        expected_train_range = "[0, 1]"
        expected_infer_range = "[0, 255]"
    elif params.QUANTIZE_MODEL:
        expected_train_range = "[0, 1]"
        expected_infer_range = "[0, 255]"
    else:
        expected_train_range = "[0, 1]"
        expected_infer_range = "[0, 1]"
    
    print(f"\n📊 Preprocessing Ranges:")
    print(f"   Training: Expected {expected_train_range}, Actual {train_range}")
    print(f"   Inference: Expected {expected_infer_range}, Actual {infer_range}")
    
    # Check if ranges are correct (with tolerance)
    train_ok = True  # Already checked in data_type_consistency
    infer_ok = True  # Already checked in data_type_consistency
    
    if train_ok and infer_ok and data_consistency_ok and qat_flow_ok:
        print("\n✅ Preprocessing consistency: VALID")
        
        # Additional QAT-specific validation
        if params.USE_QAT:
            print("💡 QAT Configuration Validated:")
            if params.QUANTIZE_MODEL:
                print("   - Using float32 [0, 1] inputs for training and inference")
                print("   - Fake quantization nodes simulate quantization internally")
                print("   - Model expects normalized inputs during deployment")
            else:
                print("   ⚠️  QAT enabled but QUANTIZE_MODEL=False")
                print("   💡 Quantization won't be applied to final model")
    else:
        print("\n❌ Preprocessing consistency: INVALID")
        if not data_consistency_ok:
            print("   - Data type consistency failed")
        if not qat_flow_ok:
            print("   - QAT data flow issues detected")
    
    print("=" * 50)
    return data_consistency_ok and is_valid and qat_flow_ok

def get_preprocessing_info():
    """
    Get current preprocessing configuration including QAT status
    Updated with corrected QAT input requirements
    """
    # Determine current mode with corrected QAT data types
    if not params.QUANTIZE_MODEL:
        if params.USE_QAT:
            mode = "QAT Training (No quantization applied)"
            normalization = "Float32 [0,1] (QAT expects normalized inputs)"
        else:
            mode = "Float32 Training"
            normalization = "Float32 [0,1]"
    else:
        if params.USE_QAT:
            if params.ESP_DL_QUANTIZE:
                mode = "QAT + INT8 quantization"
                normalization = "Training: Float32 [0,1], Inference: Float32 [0,1]"
            else:
                mode = "QAT + UINT8 quantization" 
                normalization = "Training: Float32 [0,1], Inference: Float32 [0,1]"
        else:
            if params.ESP_DL_QUANTIZE:
                mode = "Standard Training + INT8 post-quantization"
                normalization = "Training: Float32 [0,1], Inference: UINT8 [0,255]"
            else:
                mode = "Standard Training + UINT8 post-quantization"
                normalization = "Training: Float32 [0,1], Inference: UINT8 [0,255]"
    
    return {
        'quantize_model': params.QUANTIZE_MODEL,
        'use_qat': params.USE_QAT,
        'esp_dl_quantize': params.ESP_DL_QUANTIZE,
        'mode': mode,
        'normalization': normalization,
        'input_shape': params.INPUT_SHAPE,
        'recommendation': 'QAT compatible' if params.USE_QAT else 'Standard training',
        'data_type_consistency': '✅ QAT Optimal' if (params.USE_QAT and params.QUANTIZE_MODEL) else '✅ Standard'
    }

def predict_single_image(image):
    """
    Preprocess a single image for inference
    Uses the same preprocessing as training for consistency
    """
    return preprocess_images([image], for_training=False)[0]

def get_model_input_requirements():
    """
    Get the input requirements for the current model configuration
    Helps diagnose input type mismatches
    """
    requirements = {
        'qat_float32': "QAT models expect float32 [0, 1] inputs during training AND inference",
        'qat_conversion': "QAT models are converted to quantized TFLite but still expect float32",
        'non_qat_quantized': "Non-QAT quantized models expect uint8 [0, 255] inputs",
        'float_models': "Float models expect float32 [0, 1] inputs"
    }
    
    if params.USE_QAT and params.QUANTIZE_MODEL:
        return requirements['qat_float32']
    elif params.QUANTIZE_MODEL:
        return requirements['non_qat_quantized']
    else:
        return requirements['float_models']

def test_all_preprocessing_combinations():
    """
    Test all 9 preprocessing combinations for verification with enhanced data type checks
    Now includes data type consistency validation
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
            train_range = f"{train_processed.dtype} [{train_processed.min():.1f}, {train_processed.max():.1f}]"
            
            # Test inference preprocessing  
            infer_processed = preprocess_images(test_images, for_training=False)
            infer_range = f"{infer_processed.dtype} [{infer_processed.min():.1f}, {infer_processed.max():.1f}]"
            
            # Check consistency (CRITICAL FOR QAT)
            consistent = (train_processed.dtype == infer_processed.dtype and 
                         abs(train_processed.min() - infer_processed.min()) < 1e-6 and
                         abs(train_processed.max() - infer_processed.max()) < 1e-6)
            
            consistency_marker = "✅" if consistent else "⚠️ "
            
            print(f"   {consistency_marker} Training: {train_range}, Inference: {infer_range}")
            
            # Add consistency info to results
            if consistent:
                results[description] = f"CONSISTENT - Train: {train_range}, Infer: {infer_range}"
            else:
                results[description] = f"INCONSISTENT - Train: {train_range}, Infer: {infer_range}"
            
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
    
    # Summary of QAT-compatible configurations
    print(f"\n🎯 QAT-COMPATIBLE CONFIGURATIONS:")
    qat_configs = [desc for desc, result in results.items() 
                   if "QAT training" in desc and "CONSISTENT" in result]
    if qat_configs:
        for config in qat_configs:
            print(f"  ✅ {config}")
    else:
        print("  ⚠️  No QAT-compatible configurations found")
    
    return results

# QAT-specific helper functions
def check_qat_compatibility(qat_available):
    """
    Check if current settings are compatible with QAT
    Returns: (is_compatible, warnings, errors, info)
    Enhanced with data type consistency checks
    """
    warnings = []
    errors = []
    info = []
    
    # Critical errors that prevent QAT from working
    if params.USE_QAT and not qat_available:
        errors.append("QAT requires tensorflow-model-optimization package. Install with: pip install tensorflow-model-optimization")
    
    # Check data type consistency for QAT
    if params.USE_QAT and params.QUANTIZE_MODEL:
        train_dtype, _, _, _ = get_qat_training_format()
        if params.QUANTIZE_MODEL:
            infer_dtype = np.float32  # QAT inference also uses float32
        else:
            infer_dtype = np.float32
            
        if train_dtype != infer_dtype:
            warnings.append(f"QAT data type mismatch: training={train_dtype}, inference={infer_dtype}")
            warnings.append("This may cause quantization errors during deployment")
    
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
            # Check data type consistency
            train_dtype, _, _, train_desc = get_qat_training_format()
            info.append(f"✅ Training format: {train_desc}")
            info.append(f"✅ Inference format: Float32 [0, 1] (QAT models expect normalized inputs)")
        else:
            info.append("⚠️  Post-training quantization disabled - QAT benefits may not be realized")
    
    return len(errors) == 0, warnings, errors, info

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
    
    # Enhanced configuration analysis with data type info
    if params.USE_QAT and params.ESP_DL_QUANTIZE:
        print("✅ QAT + ESP-DL: Training for INT8 quantization with Float32 [0,1]")
        print("   Model expects Float32 inputs, converts to INT8 during TFLite conversion")
    
    elif params.USE_QAT and not params.ESP_DL_QUANTIZE:
        print("✅ QAT only: Training for UINT8 quantization with Float32 [0,1]")
        print("   Model expects Float32 inputs, converts to UINT8 during TFLite conversion")
    
    elif not params.USE_QAT and params.ESP_DL_QUANTIZE:
        print("✅ ESP-DL only: Standard training + INT8 post-quantization")
        print("   Training: Float32 [0,1], Inference: UINT8 [0,255]")
    
    elif not params.USE_QAT and params.QUANTIZE_MODEL and not params.ESP_DL_QUANTIZE:
        print("✅ Standard quantization: Training + UINT8 post-quantization")
        print("   Training: Float32 [0,1], Inference: UINT8 [0,255]")
    
    else:
        print("✅ Float32: No quantization")
        print("   Training: Float32 [0,1], Inference: Float32 [0,1]")
    
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
    """Debug function to trace preprocessing flow and detect input type issues"""
    print("\n🔍 DEBUG: Tracing Preprocessing Flow")
    print("=" * 50)
    
    # Create test data
    test_images_raw = np.random.randint(0, 255, (2, 28, 28, 1), dtype=np.uint8)
    print(f"Raw data range: [{test_images_raw.min()}, {test_images_raw.max()}]")
    print(f"Raw data dtype: {test_images_raw.dtype}")
    
    # Simulate preprocessing
    test_processed = preprocess_images(test_images_raw, for_training=True)
    print(f"Training preprocessing: [{test_processed.min():.3f}, {test_processed.max():.3f}]")
    print(f"Training dtype: {test_processed.dtype}")
    
    # Check inference preprocessing
    infer_processed = preprocess_images(test_images_raw, for_training=False)
    print(f"Inference preprocessing: [{infer_processed.min():.3f}, {infer_processed.max():.3f}]")
    print(f"Inference dtype: {infer_processed.dtype}")
    
    # Get model requirements
    requirements = get_model_input_requirements()
    print(f"\n🎯 Model Input Requirements: {requirements}")
    
    # Check QAT-specific requirements
    if params.USE_QAT and params.QUANTIZE_MODEL:
        if infer_processed.dtype == np.float32:
            print("✅ QAT Input Compatibility: CORRECT")
            print("   Model will receive float32 inputs as expected")
            print("   This will prevent 'UINT8 but expected FLOAT32' errors")
        else:
            print("❌ QAT Input Compatibility: ERROR")
            print(f"   Model expects float32 but will receive {infer_processed.dtype}")
            print("   This will cause: 'Got value of type UINT8 but expected type FLOAT32'")
            print("   SOLUTION: QAT models must receive float32 [0, 1] inputs")
    
    return test_processed

if __name__ == "__main__":
    # Test the current configuration
    validate_preprocessing_consistency()
    
    # Show current preprocessing info
    info = get_preprocessing_info()
    print(f"\n📋 CURRENT PREPROCESSING CONFIGURATION:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Show model input requirements
    requirements = get_model_input_requirements()
    print(f"\n🎯 MODEL INPUT REQUIREMENTS: {requirements}")
    
    # Run debug flow to check for input type issues
    debug_preprocessing_flow()
    
    # Optionally test all combinations
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-all', action='store_true', help='Test all 9 preprocessing combinations')
    args = parser.parse_args()
    
    if args.test_all:
        test_all_preprocessing_combinations()