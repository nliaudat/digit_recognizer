# utils/preprocess.py
import cv2
import numpy as np
import tensorflow as tf
import parameters as params

  
    
def _preprocess_common(images, target_size, grayscale):
    """Resize / colour convert images, preserving the original dtype."""
    if target_size is None:
        target_size = (params.INPUT_WIDTH, params.INPUT_HEIGHT)
    if grayscale is None:
        grayscale = params.USE_GRAYSCALE

    processed = []
    for image in images:
        img = cv2.resize(image, target_size)

        if grayscale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif not grayscale and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        processed.append(img)

    return np.array(processed)


def preprocess_for_training(images, target_size=None, grayscale=None):
    """
    Return data format based on quantization settings.
    For QAT: Use float32 [0,1] internally but the model will handle quantization
    """
    arr = _preprocess_common(images, target_size, grayscale)
    
    if params.USE_QAT and params.QUANTIZE_MODEL:
        # QAT training: convert to float32 [0,1] for training
        # The QAT model will handle fake quantization internally
        print("QAT Training: Using Float32 [0,1] with fake quantization")
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr
    else:
        # Standard training: float32 [0,1]
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr


# def preprocess_for_inference(images, target_size=None, grayscale=None):
    # """
    # Return data in the exact format the exported TFLite model expects.
    # FIXED VERSION - handles single images and batches properly
    # """
    # # Handle single image input
    # single_image = False
    # if isinstance(images, np.ndarray) and len(images.shape) == 3:
        # single_image = True
        # images = [images]
    
    # arr = _preprocess_common(images, target_size, grayscale)
    
    # # Handle single image output
    # if single_image:
        # arr = arr[0]  # Return single image instead of batch

    # if not params.QUANTIZE_MODEL:
        # # No quantization: float32 [0,1]
        # arr = arr.astype(np.float32)
        # if arr.max() > 1.0:
            # arr = arr / 255.0
        # return arr

    # # Quantised inference path - FIXED LOGIC
    # if params.ESP_DL_QUANTIZE:
        # # ESP-DL expects int8 [-128, 127]
        # arr = arr.astype(np.int8)
        # # if params.USE_QAT:
            # # print("Inference: INT8 [0,255] for ESP-DL QAT")
        # # else:
            # # print("Inference: INT8 [0,255] for ESP-DL PTQ")
    # else:
        # # Standard quantization expects uint8 [0, 255]
        # arr = arr.astype(np.uint8)
        # # if params.USE_QAT:
            # # print("Inference: UINT8 [0,255] (QAT deployment)")
        # # else:
            # # print("Inference: UINT8 [0,255] (PTQ)")

    # return arr
    
def preprocess_for_inference(images, target_size=None, grayscale=None):
    """
    Return data in the exact format the exported TFLite model expects.
    FIXED VERSION with correct ESP-DL handling
    """
    # Handle single image input
    single_image = False
    if isinstance(images, np.ndarray) and len(images.shape) == 3:
        single_image = True
        images = [images]
    
    arr = _preprocess_common(images, target_size, grayscale)
    
    # Handle single image output
    if single_image:
        arr = arr[0]  # Return single image instead of batch

    print(f"🔧 Preprocessing - Input range: [{arr.min()}, {arr.max()}], dtype: {arr.dtype}")

    # FOR STANDARD QUANTIZED UINT8 MODELS: Keep as uint8 [0, 255]
    if params.QUANTIZE_MODEL and not params.ESP_DL_QUANTIZE:
        # Ensure uint8 [0, 255] format
        if arr.dtype != np.uint8:
            # If float32 [0,1], convert back to uint8 [0,255]
            if arr.dtype == np.float32 and arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        
        # Ensure range is [0, 255]
        arr = np.clip(arr, 0, 255)
        print(f"   Output: uint8 [{arr.min()}, {arr.max()}] (Standard quantization)")
        return arr

    # FOR ESP-DL QUANTIZED MODELS: Convert to int8 [-128, 127]
    if params.QUANTIZE_MODEL and params.ESP_DL_QUANTIZE:
        # First ensure we have uint8 [0, 255]
        if arr.dtype != np.uint8:
            if arr.dtype == np.float32 and arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        
        # CORRECT ESP-DL conversion: uint8 [0,255] -> int8 [-128,127]
        arr = (arr.astype(np.int32) - 128).astype(np.int8)
        print(f"   Output: int8 [{arr.min()}, {arr.max()}] (ESP-DL quantization)")
        return arr

    # For float models: float32 [0,1]
    if not params.QUANTIZE_MODEL:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        print(f"   Output: float32 [{arr.min():.3f}, {arr.max():.3f}] (No quantization)")
        return arr

    return arr

def preprocess_images(images, target_size=None, grayscale=None, for_training=True):
    """
    Backwards compatible wrapper kept for legacy callers.
    Delegates to the new explicit helpers.
    """
    if for_training:
        return preprocess_for_training(images, target_size, grayscale)
    else:
        return preprocess_for_inference(images, target_size, grayscale)


def preprocess_images_esp_dl(images, target_size=None):
    """ESP DL specific preprocessing – returns UINT8 [0, 255]."""
    if target_size is None:
        target_size = (params.INPUT_WIDTH, params.INPUT_HEIGHT)

    processed_images = []

    for image in images:
        image = cv2.resize(image, target_size)

        if params.USE_GRAYSCALE and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif not params.USE_GRAYSCALE and len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        processed_images.append(image)

    return np.array(processed_images, dtype=np.uint8)


def preprocess_images_for_qat_calibration(images):
    """Special preprocessing for QAT model calibration."""
    print("QAT Calibration Preprocessing")
    return preprocess_for_training(images)


# def get_qat_training_format():
    # """Return the data format used during QAT training."""
    # if params.USE_QAT and params.QUANTIZE_MODEL:
        # if params.ESP_DL_QUANTIZE:
            # return np.uint8, 0, 255, "UINT8 [0, 255] (ESP-DL QAT)"
        # else:
            # return np.uint8, 0, 255, "UINT8 [0, 255] (Standard QAT)"
    # else:
        # return np.float32, 0.0, 1.0, "Float32 [0, 1] (Standard)"
def get_qat_training_format():
    """Return the data format used during QAT training."""
    # CORRECTED: QAT training uses float32 [0,1] regardless of ESP-DL setting
    if params.USE_QAT and params.QUANTIZE_MODEL:
        return np.float32, 0.0, 1.0, "Float32 [0, 1] (QAT Training)"
    else:
        return np.float32, 0.0, 1.0, "Float32 [0, 1] (Standard)"

def validate_preprocessing_consistency():
    """
    Validate that data types are consistent between training and inference.
    CRITICAL FOR QAT: Training and inference must use identical data formats.
    """
    print("\n🔍 VALIDATING DATA TYPE CONSISTENCY")
    print("=" * 60)

    # Expected formats
    train_dtype, train_min, train_max, train_desc = get_qat_training_format()

    if params.QUANTIZE_MODEL:
        if params.ESP_DL_QUANTIZE:
            infer_dtype, infer_min, infer_max = np.uint8, 0, 255
            infer_desc = "UINT8 [0, 255] (ESP-DL)"
        else:
            infer_dtype, infer_min, infer_max = np.uint8, 0, 255
            infer_desc = "UINT8 [0, 255] (Standard)"
    else:
        infer_dtype, infer_min, infer_max = np.float32, 0.0, 1.0
        infer_desc = "Float32 [0, 1] (No quantization)"

    # Sample data
    test_images = np.random.randint(
        0,
        255,
        (3, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS),
        dtype=np.uint8,
    )

    train_processed = preprocess_for_training(test_images)
    infer_processed = preprocess_for_inference(test_images)

    train_ok = (
        train_processed.dtype == train_dtype
        and train_processed.min() >= train_min
        and train_processed.max() <= train_max
    )
    infer_ok = (
        infer_processed.dtype == infer_dtype
        and infer_processed.min() >= infer_min
        and infer_processed.max() <= infer_max
    )
    consistency_ok = (
        train_processed.dtype == infer_processed.dtype
        and abs(train_processed.min() - infer_processed.min()) < 1e-6
        and abs(train_processed.max() - infer_processed.max()) < 1e-6
    )

    print(f"\n📊 Actual Formats:")
    print(
        f"   Training:  {train_processed.dtype} [{train_processed.min():.3f}, {train_processed.max():.3f}]"
    )
    print(
        f"   Inference: {infer_processed.dtype} [{infer_processed.min():.3f}, {infer_processed.max():.3f}]"
    )

    if train_ok and infer_ok and consistency_ok:
        print("\n✅ DATA TYPE CONSISTENCY: PERFECT")
        print("   Training and inference formats are identical")
        print("   QAT fake quantization will match deployment quantization")
    elif train_ok and infer_ok:
        print("\n⚠️  DATA TYPE CONSISTENCY: ACCEPTABLE")
        print("   Individual formats are correct but training ≠ inference")
        print("   This may cause QAT quantization mismatches")
    else:
        print("\n❌ DATA TYPE CONSISTENCY: FAILED")
        if not train_ok:
            print(
                f"   - Training format incorrect: expected {train_dtype} [{train_min}, {train_max}]"
            )
        if not infer_ok:
            print(
                f"   - Inference format incorrect: expected {infer_dtype} [{infer_min}, {infer_max}]"
            )

    # QAT specific validation
    if params.USE_QAT and params.QUANTIZE_MODEL:
        print(f"\n🎯 QAT DATA FLOW VALIDATION:")
        if consistency_ok:
            print("   ✅ QAT training and inference using identical data format")
            print("   ✅ Fake quantization will accurately simulate deployment")
        else:
            print("   ⚠️  QAT training and inference using different formats")
            print("   ⚠️  This can cause accuracy degradation after conversion")

    return train_ok and infer_ok


def get_preprocessing_info():
    """Return a summary of the current preprocessing configuration."""
    # CORRECTED: QAT training uses float32 [0,1], inference uses integer types
    if not params.QUANTIZE_MODEL:
        if params.USE_QAT:
            mode = "QAT Training (No quantization applied)"
            normalization = "Float32 [0,1] (simulating quantization)"
        else:
            mode = "Float32 Training"
            normalization = "Float32 [0,1]"
    else:
        if params.USE_QAT:
            if params.ESP_DL_QUANTIZE:
                mode = "QAT + INT8 ESP-DL Quantization"
                normalization = "Training: Float32 [0,1], Inference: INT8 [0,255]"
            else:
                mode = "QAT + UINT8 Quantization"
                normalization = "Training: Float32 [0,1], Inference: UINT8 [0,255]"
        else:
            if params.ESP_DL_QUANTIZE:
                mode = "Standard Training + INT8 ESP-DL Quantization"
                normalization = "Training: Float32 [0,1], Inference: INT8 [0,255]"
            else:
                mode = "Standard Training + UINT8 Quantization"
                normalization = "Training: Float32 [0,1], Inference: UINT8 [0,255]"

    return {
        "quantize_model": params.QUANTIZE_MODEL,
        "use_qat": params.USE_QAT,
        "esp_dl_quantize": params.ESP_DL_QUANTIZE,
        "mode": mode,
        "normalization": normalization,
        "input_shape": params.INPUT_SHAPE,
        "recommendation": "QAT compatible" if params.USE_QAT else "Standard training",
        "data_type_consistency": "✅ Perfect" if not params.USE_QAT else "⚠️ Different (expected)"
    }


def predict_single_image(image):
    """Preprocess a single image for inference."""
    return preprocess_for_inference([image])[0]


def test_all_preprocessing_combinations():
    """
    Run all nine possible preprocessing configurations and report whether
    training  and inference preprocessing produce compatible data types and
    value ranges.

    The nine configurations correspond to the three boolean flags:

        * QUANTIZE_MODEL
        * USE_QAT
        * ESP_DL_QUANTIZE

    The function prints a human readable table and returns a dictionary
    mapping the textual description of each configuration to the result
    string (e.g. “CONSISTENT …”, “INCONSISTENT …”, “INVALID”, or “FAILED”).
    """
    print("\n🧪 TESTING ALL 9 PREPROCESSING COMBINATIONS")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Preserve the original flag values so we can restore them later.
    # -----------------------------------------------------------------
    orig_quant = params.QUANTIZE_MODEL
    orig_qat   = params.USE_QAT
    orig_esp   = params.ESP_DL_QUANTIZE

    # -----------------------------------------------------------------
    # Define the nine flag permutations together with a short description.
    # -----------------------------------------------------------------
    combos = [
        (False, False, False, "Float32 training & inference"),
        (False, False, True , "INVALID: ESP DL without quantization"),
        (False, True , False, "QAT training, float32 inference"),
        (False, True , True , "INVALID: ESP DL without quantization"),
        (True , False, False, "Standard training, UINT8 quantization"),
        (True , False, True , "Standard training, INT8 quantization (ESP DL)"),
        (True , True , False, "QAT training, UINT8 quantization"),
        (True , True , True , "QAT training, INT8 quantization (ESP DL)"),
    ]

    results = {}

    # -----------------------------------------------------------------
    # Iterate over every combination, temporarily set the global flags,
    # run the preprocessing helpers and record the outcome.
    # -----------------------------------------------------------------
    for quant, qat, esp, description in combos:
        print(f"\n🔍 Testing: {description}")
        print(f"   QUANTIZE_MODEL={quant}, USE_QAT={qat}, ESP_DL_QUANTIZE={esp}")

        # Apply the temporary flag values
        params.QUANTIZE_MODEL = quant
        params.USE_QAT        = qat
        params.ESP_DL_QUANTIZE = esp

        # Skip configurations that are mathematically impossible
        if not quant and esp:
            print("   ❌ SKIPPED: Invalid combination (ESP DL requires quantisation)")
            results[description] = "INVALID"
            continue

        try:
            # Create a small random batch that matches the expected input shape
            test_imgs = np.random.randint(
                0,
                255,
                (3, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS),
                dtype=np.uint8,
            )

            # Run the two preprocessing pipelines
            train_proc = preprocess_for_training(test_imgs)
            infer_proc = preprocess_for_inference(test_imgs)

            # Are the two outputs identical in dtype and numeric range?
            consistent = (
                train_proc.dtype == infer_proc.dtype
                and abs(train_proc.min() - infer_proc.min()) < 1e-6
                and abs(train_proc.max() - infer_proc.max()) < 1e-6
            )

            marker = "✅" if consistent else "⚠️"
            print(
                f"   {marker} Training: {train_proc.dtype} "
                f"[{train_proc.min():.1f}, {train_proc.max():.1f}], "
                f"Inference: {infer_proc.dtype} "
                f"[{infer_proc.min():.1f}, {infer_proc.max():.1f}]"
            )

            if consistent:
                results[description] = (
                    f"CONSISTENT - Train: {train_proc.dtype} "
                    f"[{train_proc.min():.1f}, {train_proc.max():.1f}], "
                    f"Infer: {infer_proc.dtype} "
                    f"[{infer_proc.min():.1f}, {infer_proc.max():.1f}]"
                )
            else:
                results[description] = (
                    f"INCONSISTENT - Train: {train_proc.dtype} "
                    f"[{train_proc.min():.1f}, {train_proc.max():.1f}], "
                    f"Infer: {infer_proc.dtype} "
                    f"[{infer_proc.min():.1f}, {infer_proc.max():.1f}]"
                )

        except Exception as exc:   # pragma: no cover
            print(f"   ❌ FAILED: {exc}")
            results[description] = f"FAILED - {exc}"

    # -----------------------------------------------------------------
    # Restore the original flag values so the rest of the program behaves
    # exactly as it did before the test.
    # -----------------------------------------------------------------
    params.QUANTIZE_MODEL = orig_quant
    params.USE_QAT        = orig_qat
    params.ESP_DL_QUANTIZE = orig_esp

    # -----------------------------------------------------------------
    # Print a nicely formatted summary table.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 PREPROCESSING TEST RESULTS:")
    print("=" * 70)
    for desc, outcome in results.items():
        print(f"{desc:45} : {outcome}")

    # -----------------------------------------------------------------
    # Highlight any configurations that are both QAT enabled *and*
    # internally consistent – those are the ones you can safely use for
    # Quantisation Aware Training.
    # -----------------------------------------------------------------
    qat_compatible = [
        d for d, o in results.items() if "QAT" in d and "CONSISTENT" in o
    ]

    print("\n🎯 QAT COMPATIBLE CONFIGURATIONS:")
    if qat_compatible:
        for cfg in qat_compatible:
            print(f"  ✅ {cfg}")
    else:
        print("  ⚠️  No QAT compatible configurations found")

    return results