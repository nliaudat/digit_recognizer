"""
qat_debug.py — Debug/diagnostic utilities for Quantization Aware Training.

These helpers are separated from ``train_qat_helper.py`` to keep the main
production code focused on essential QAT logic. They are typically called
with ``--debug`` or during interactive troubleshooting sessions.
"""

import logging
import os
import sys
import numpy as np
import tensorflow as tf

import parameters as params
from utils.preprocess import (
    get_qat_training_format, preprocess_for_inference, preprocess_for_training
)
from utils.train_qat_helper import validate_quantization_combination

logger = logging.getLogger(__name__)


def debug_preprocessing_flow():
    """Debug function to trace preprocessing flow and detect double processing"""
    print("\n🔍 DEBUG: Tracing Preprocessing Flow")
    print("=" * 50)

    # Create test data
    test_images_raw = np.random.randint(0, 255, (2, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS), dtype=np.uint8)
    print(f"Raw data range: [{test_images_raw.min()}, {test_images_raw.max()}]")
    print(f"Raw data dtype: {test_images_raw.dtype}")

    print(f"\n📊 Current Configuration:")
    print(f"   QUANTIZE_MODEL: {params.QUANTIZE_MODEL}")
    print(f"   USE_QAT: {params.USE_QAT}")
    print(f"   ESP_DL_QUANTIZE: {params.ESP_DL_QUANTIZE}")

    # Test BOTH training and inference modes
    print(f"\n🧪 Testing Training Mode (for_training=True):")
    train_processed = preprocess_for_training(test_images_raw)
    print(f"   Result: {train_processed.dtype} [{train_processed.min():.3f}, {train_processed.max():.3f}]")

    print(f"\n🧪 Testing Inference Mode (for_training=False):")
    infer_processed = preprocess_for_inference(test_images_raw)
    print(f"   Result: {infer_processed.dtype} [{infer_processed.min():.3f}, {infer_processed.max():.3f}]")

    # Determine expected behavior
    print(f"\n✅ Expected Behavior:")
    if params.QUANTIZE_MODEL:
        if params.USE_QAT:
            # QAT: Both training and inference should use UINT8
            expected_train = "UINT8 [0, 255]"
            expected_infer = "UINT8 [0, 255]"
            print("   QAT Mode: Training and inference both use UINT8 [0, 255]")
        else:
            # Standard quantization: Training uses float32, inference uses UINT8
            expected_train = "Float32 [0, 1]"
            expected_infer = "UINT8 [0, 255]"
            print("   Standard Quant: Training=Float32 [0,1], Inference=UINT8 [0,255]")
    else:
        # No quantization: Both use float32
        expected_train = "Float32 [0, 1]"
        expected_infer = "Float32 [0, 1]"
        print("   No Quantization: Training and inference both use Float32 [0, 1]")

    # Check consistency
    print(f"\n🔍 Consistency Check:")
    if params.USE_QAT and params.QUANTIZE_MODEL:
        # QAT requires training and inference to be identical
        if train_processed.dtype == infer_processed.dtype:
            print("✅ QAT Consistency: Perfect - training matches inference")
        else:
            print("❌ QAT Consistency: FAILED - training ≠ inference")
    else:
        print("ℹ️  Non-QAT mode: Training/inference differences are expected")

    # Check for double preprocessing
    if train_processed.max() < 0.1 and train_processed.dtype == np.float32:
        print("🚨 WARNING: Possible double preprocessing in training!")

    if infer_processed.max() < 0.1 and infer_processed.dtype == np.float32:
        print("🚨 WARNING: Possible double preprocessing in inference!")

    return infer_processed  # Return inference result as it's typically what matters for deployment


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
        print("✅ QAT + ESP-DL: Training for INT8 quantization with UINT8 [0,255]")

    elif params.USE_QAT and not params.ESP_DL_QUANTIZE:
        print("✅ QAT only: Training for UINT8 quantization with UINT8 [0,255]")

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


def debug_qat_layers(model):
    """Debug function to see what's actually in the model"""
    print("\n🔍 DETAILED MODEL LAYER ANALYSIS:")
    print("=" * 50)

    for i, layer in enumerate(model.layers):
        layer_info = f"Layer {i}: {type(layer).__name__:20} - {layer.name:20}"

        # Check for quantization attributes
        quant_attrs = []
        if hasattr(layer, 'quantize_config'):
            quant_attrs.append('quantize_config')
        if hasattr(layer, '_quantize_wrapper'):
            quant_attrs.append('_quantize_wrapper')
        if hasattr(layer, '_quantizeable'):
            quant_attrs.append('_quantizeable')

        if quant_attrs:
            layer_info += f" → Quantization: {quant_attrs}"

        print(f"   {layer_info}")

        # Check layer weights for quantization
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            if weights:
                print(f"      Weights: {[w.shape for w in weights]}")


def two_phase_qat_training(x_train, y_train, x_val, y_val):
    """
    Two-phase training:
    1. Train standard model to convergence
    2. Convert to QAT and fine-tune

    NOTE: This function is experimental and not used in the standard pipeline.
    """
    print("\n🎯 TWO-PHASE QAT TRAINING")
    print("=" * 50)

    # Phase 1: Train standard model
    print("📚 Phase 1: Training standard model...")
    old_qat = params.USE_QAT
    params.USE_QAT = False

    from models import create_model, compile_model

    standard_model = create_model()
    standard_model = compile_model(standard_model)

    # Train standard model properly
    standard_history = standard_model.fit(
        x_train, y_train,
        epochs=min(50, params.EPOCHS // 2),
        batch_size=params.BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_accuracy'
            )
        ]
    )

    standard_acc = standard_history.history['val_accuracy'][-1]
    print(f"✅ Standard model trained: {standard_acc:.4f} val accuracy")

    if standard_acc < 0.8:
        print("🚨 Standard model not learning well - cannot proceed with QAT")
        params.USE_QAT = old_qat
        return standard_model

    # Phase 2: Convert to QAT and fine-tune
    print("🔄 Phase 2: Converting to QAT and fine-tuning...")
    params.USE_QAT = True

    try:
        try:
            import tensorflow_model_optimization as tfmot
        except ImportError:
            raise ImportError("tensorflow_model_optimization required for Phase 2")

        # Create QAT model
        with tfmot.quantization.keras.quantize_scope():
            qat_model = create_model()

        # Build the model
        qat_model.build(input_shape=(None,) + params.INPUT_SHAPE)

        # Copy weights layer by layer
        print("📥 Transferring weights to QAT model...")
        weights_transferred = 0
        for qat_layer, std_layer in zip(qat_model.layers, standard_model.layers):
            try:
                if (hasattr(qat_layer, 'get_weights') and hasattr(std_layer, 'get_weights') and
                    len(qat_layer.get_weights()) == len(std_layer.get_weights())):
                    qat_layer.set_weights(std_layer.get_weights())
                    weights_transferred += 1
            except Exception as e:
                print(f"⚠️  Could not transfer weights for {qat_layer.name}: {e}")

        print(f"✅ Transferred weights for {weights_transferred} layers")

        # Compile with slightly higher learning rate for fine-tuning
        qat_optimizer = tf.keras.optimizers.Adam(learning_rate=params.LEARNING_RATE * 2.0)
        loss = (
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=params.USE_LOGITS)
            if params.MODEL_ARCHITECTURE != "original_haverland"
            else tf.keras.losses.CategoricalCrossentropy()
        )
        qat_model.compile(optimizer=qat_optimizer, loss=loss, metrics=['accuracy'])

        # Fine-tune for a few epochs
        print("🎯 Fine-tuning QAT model...")
        qat_history = qat_model.fit(
            x_train, y_train,
            epochs=min(20, params.EPOCHS // 4),
            batch_size=params.BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=1
        )

        qat_acc = qat_history.history['val_accuracy'][-1]
        accuracy_drop = standard_acc - qat_acc

        print(f"✅ QAT model fine-tuned:")
        print(f"   Standard model: {standard_acc:.4f}")
        print(f"   QAT model: {qat_acc:.4f}")
        print(f"   Accuracy drop: {accuracy_drop:.4f}")

        if accuracy_drop < 0.05:
            print("🎉 QAT successful! Minimal accuracy drop.")
        else:
            print("⚠️  QAT caused significant accuracy drop")

        params.USE_QAT = old_qat
        return qat_model

    except Exception as e:
        print(f"❌ QAT conversion failed: {e}")
        print("🔄 Using standard model")
        params.USE_QAT = old_qat
        return standard_model