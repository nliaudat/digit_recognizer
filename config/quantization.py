"""
config/quantization.py — Quantization configuration.

All quantization-related parameters: master mode control, individual flags,
TQT (Trainable Quantization Thresholds) settings, and XNNPACK control.
"""

import os
import tensorflow as tf

# ==============================================================================
# QUANTIZATION MODE - MASTER CONTROL
# ==============================================================================
# Options:
#   - "none":        Float32 training & inference (Reference baseline)
#   - "ptq":         Post-Training Quantization (standard TFLite UINT8/INT8)
#   - "qat":         Quantization Aware Training (QAT with fake quantization)
#   - "tqt":         Trainable Quantization Thresholds (ESP-DL pipeline, RECOMMENDED)
#   - "auto":        Automatically select best based on model and hardware
#
# When QUANTIZATION_MODE is set, individual flags below are automatically configured.
# To manually override individual flags, set QUANTIZATION_MODE = None.
# ==============================================================================

QUANTIZATION_MODE = "tqt"  # Options: "none", "ptq", "qat", "tqt", "auto"

# Legacy individual flags (automatically overridden when QUANTIZATION_MODE is not None)
QUANTIZE_MODEL = True
ESP_DL_QUANTIZE = False
USE_QAT = True
QAT_QUANTIZE_ALL = True
QAT_SCHEME = '8bit'

# TQT (ESP-DL) Trainable Quantization Thresholds Pipeline
USE_TQT_PIPELINE = False
USE_TQT_FOR_TFLITE = False

# Apply quantization mode configuration
def _configure_quantization_mode():
    """Configure quantization flags based on QUANTIZATION_MODE"""
    global QUANTIZE_MODEL, ESP_DL_QUANTIZE, USE_QAT, USE_TQT_PIPELINE, USE_TQT_FOR_TFLITE
    
    if QUANTIZATION_MODE is None:
        return
    
    mode = QUANTIZATION_MODE.lower()
    
    if mode == "none":
        QUANTIZE_MODEL = False
        ESP_DL_QUANTIZE = False
        USE_QAT = False
        USE_TQT_PIPELINE = False
        USE_TQT_FOR_TFLITE = False   
        
    elif mode == "ptq":
        QUANTIZE_MODEL = True
        ESP_DL_QUANTIZE = False
        USE_QAT = False
        USE_TQT_PIPELINE = False
        USE_TQT_FOR_TFLITE = False   
        
    elif mode == "qat":
        QUANTIZE_MODEL = True
        ESP_DL_QUANTIZE = False
        USE_QAT = True
        USE_TQT_PIPELINE = False
        USE_TQT_FOR_TFLITE = False   
        
    elif mode == "tqt":
        QUANTIZE_MODEL = True
        ESP_DL_QUANTIZE = True
        USE_QAT = False
        USE_TQT_PIPELINE = True
        USE_TQT_FOR_TFLITE = True   
        
    elif mode == "auto":
        _auto_configure_quantization()
        
    else:
        raise ValueError(f"❌ Invalid QUANTIZATION_MODE: {QUANTIZATION_MODE}. "
                        f"Must be one of: 'none', 'ptq', 'qat', 'tqt', 'auto'")

def _auto_configure_quantization():
    """Automatically select quantization mode based on model and target hardware"""
    global QUANTIZE_MODEL, ESP_DL_QUANTIZE, USE_QAT, USE_TQT_PIPELINE, USE_TQT_FOR_TFLITE
    from config import AVAILABLE_MODELS, MODEL_ARCHITECTURE
    
    esp_compatible_models = [m for m in AVAILABLE_MODELS if not m.endswith("_teacher")]
    
    if MODEL_ARCHITECTURE in esp_compatible_models:
        QUANTIZE_MODEL = True
        ESP_DL_QUANTIZE = True
        USE_QAT = False
        USE_TQT_PIPELINE = True
        USE_TQT_FOR_TFLITE = True
    else:
        QUANTIZE_MODEL = False
        ESP_DL_QUANTIZE = False
        USE_QAT = False
        USE_TQT_PIPELINE = False
        USE_TQT_FOR_TFLITE = False

# Apply configuration
_configure_quantization_mode()

# Disable XNNPACK delegate for TFLite Micro compatibility
DISABLE_XNNPACK = True

# Force TFLITE_BUILTINS_INT8 as the only supported ops for all TFLite conversions
USE_TFLITE_BUILTINS_INT8_ONLY = True

# TQT Parameters (only relevant when USE_TQT_PIPELINE=True)
TQT_NUM_BITS = 8
TQT_TARGET = 'esp32'
TQT_EXPORT_ALL_TARGETS = True
TQT_ALL_TARGETS = ['esp32', 'esp32s3', 'esp32p4']

def _detect_tqt_device():
    return "cpu"

TQT_DEVICE = _detect_tqt_device()

_TQT_DEFAULTS = {
    "esp32": {
        "TQT_STEPS":               200,
        "TQT_LR":                  1e-6,
        "TQT_BLOCK_SIZE":          2,
        "TQT_INT_LAMBDA":          0.10,
        "TQT_IS_SCALE_TRAINABLE":  True,
        "TQT_IS_WEIGHT_TRAINABLE": False,
        "TQT_COLLECTING_DEVICE":   TQT_DEVICE,
        "TQT_CALIB_STEPS":         300,
        "TQT_CALIB_BATCH_SIZE":    1,
    },
    "esp32s3": {
        "TQT_STEPS":               200,
        "TQT_LR":                  1e-6,
        "TQT_BLOCK_SIZE":          2,
        "TQT_INT_LAMBDA":          0.05,
        "TQT_IS_SCALE_TRAINABLE":  True,
        "TQT_IS_WEIGHT_TRAINABLE": False,
        "TQT_COLLECTING_DEVICE":   TQT_DEVICE,
        "TQT_CALIB_STEPS":         300,
        "TQT_CALIB_BATCH_SIZE":    1,
    },
    "esp32p4": {
        "TQT_STEPS":               200,
        "TQT_LR":                  1e-6,
        "TQT_BLOCK_SIZE":          2,
        "TQT_INT_LAMBDA":          0.0,
        "TQT_IS_SCALE_TRAINABLE":  True,
        "TQT_IS_WEIGHT_TRAINABLE": False,
        "TQT_COLLECTING_DEVICE":   TQT_DEVICE,
        "TQT_CALIB_STEPS":         300,
        "TQT_CALIB_BATCH_SIZE":    1,
    },
}

_tqt_cfg = _TQT_DEFAULTS.get(TQT_TARGET, _TQT_DEFAULTS["esp32"])
TQT_STEPS               = _tqt_cfg["TQT_STEPS"]
TQT_LR                  = _tqt_cfg["TQT_LR"]
TQT_BLOCK_SIZE          = _tqt_cfg["TQT_BLOCK_SIZE"]
TQT_INT_LAMBDA          = _tqt_cfg["TQT_INT_LAMBDA"]
TQT_IS_SCALE_TRAINABLE  = _tqt_cfg["TQT_IS_SCALE_TRAINABLE"]
TQT_IS_WEIGHT_TRAINABLE = _tqt_cfg["TQT_IS_WEIGHT_TRAINABLE"]
TQT_COLLECTING_DEVICE   = _tqt_cfg["TQT_COLLECTING_DEVICE"]
TQT_CALIB_STEPS         = _tqt_cfg["TQT_CALIB_STEPS"]
TQT_CALIB_BATCH_SIZE    = _tqt_cfg["TQT_CALIB_BATCH_SIZE"]

# Dataset disk cache directory
DATASET_CACHE_DIR = os.environ.get("DATASET_CACHE_DIR", ".dataset_cache")

# Data pipeline configuration
USE_TF_DATA_PIPELINE = True

try:
    TF_DATA_PARALLEL_CALLS = tf.data.AUTOTUNE
    TF_DATA_PREFETCH_SIZE = tf.data.AUTOTUNE
except (ImportError, NameError, AttributeError):
    TF_DATA_PARALLEL_CALLS = -1
    TF_DATA_PREFETCH_SIZE = -1

TF_DATA_SHUFFLE_BUFFER = 1000
QUANTIZE_NUM_SAMPLES = 22000
