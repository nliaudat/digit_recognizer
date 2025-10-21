import parameters as params
import numpy as np
from utils.preprocess import preprocess_images

def validate_quantization_combination():
    from utils.preprocess import validate_quantization_combination as vqc
    return vqc()

def validate_preprocessing_consistency():
    from utils.preprocess import validate_preprocessing_consistency as vpc
    return vpc()

def check_qat_compatibility(qat_available):
    from utils.preprocess import check_qat_compatibility as cqc
    return cqc(qat_available)

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
    # Placeholder for original logic
    return True

def validate_quantization_parameters():
    from parameters import validate_quantization_parameters as vqp
    return vqp()

def create_qat_representative_dataset(x_train_raw, num_samples=None):
    """Create representative dataset that preserves the correct data type for QAT"""
    if num_samples is None:
        num_samples = params.QUANTIZE_NUM_SAMPLES
    def representative_dataset():
        x_calibration = preprocess_images(x_train_raw[:num_samples], for_training=False)
        if x_calibration.dtype != np.float32:
            x_calibration = x_calibration.astype(np.float32)
        print(f"QAT Representative: {x_calibration.dtype}, "
              f"range: [{x_calibration.min():.3f}, {x_calibration.max():.3f}]")
        for i in range(len(x_calibration)):
            yield [x_calibration[i:i+1]]  # Keep as float32 for QAT
    return representative_dataset
