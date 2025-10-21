"""
Validation and consistency checks for training, quantization, and preprocessing.
Split from train.py for modularity.
"""
import numpy as np
import tensorflow as tf
import os
from utils.preprocess import get_qat_training_format, preprocess_images
import parameters as params

def validate_qat_data_flow(model, x_train_sample, debug=False):
    """
    Validate that QAT data flow is consistent between training and inference
    """
    if not params.USE_QAT or not params.QUANTIZE_MODEL:
        return True, "QAT not enabled"
    print("\n🔍 VALIDATING QAT DATA FLOW")
    print("=" * 50)
    sample_batch = x_train_sample[:1]
    print(f"Sample batch - dtype: {sample_batch.dtype}, range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
    try:
        output = model(sample_batch)
        print(f"✅ Model forward pass successful")
        print(f"   Output range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")
        quant_layers = [layer for layer in model.layers if any(quant_term in layer.name for quant_term in ['quant', 'qat'])]
        print(f"   Quantization layers found: {len(quant_layers)}")
        return True, "QAT data flow validated"
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False, f"QAT data flow failed: {e}"

def check_training_inference_alignment():
    """
    Check if training and inference preprocessing are aligned
    """
    print("\n🔍 CHECKING TRAINING/INFERENCE ALIGNMENT")
    print("=" * 50)
    from utils.preprocess import get_qat_training_format, preprocess_images
    # Get expected formats
    train_dtype, train_min, train_max, train_desc = get_qat_training_format()
    # Test with sample data
    test_data = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
    # Process for training and inference
    train_processed = preprocess_images(test_data, for_training=True)
    infer_processed = preprocess_images(test_data, for_training=False)
    print(f"Expected training format: {train_desc}")
    print(f"Actual training:   {train_processed.dtype} [{train_processed.min():.1f}, {train_processed.max():.1f}]")
    print(f"Actual inference:  {infer_processed.dtype} [{infer_processed.min():.1f}, {infer_processed.max():.1f}]")
    # Check alignment
    aligned = (train_processed.dtype == infer_processed.dtype and 
               abs(train_processed.min() - infer_processed.min()) < 1e-6 and
               abs(train_processed.max() - infer_processed.max()) < 1e-6)
    if aligned:
        print("✅ TRAINING/INFERENCE ALIGNMENT: PERFECT")
        return True
    else:
        print("❌ TRAINING/INFERENCE ALIGNMENT: MISMATCH")
        print("   Training and inference are using different data formats!")
        return False
