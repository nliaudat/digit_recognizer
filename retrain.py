#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
retrain.py – Fine-tune an existing digit recognizer model specifically on bad predictions.

Usage:
  python retrain.py --model_path exported_models/my_model/my_model.keras --bad_images_dir datasets/real_integra_bad_predictions
"""

import argparse
import os
import sys
import tensorflow as tf
from pathlib import Path

# Project imports
import parameters as params
from utils.multi_source_loader import MultiSourceDataLoader, shuffle_dataset
from utils.preprocess import preprocess_for_training
from utils.augmentation import setup_augmentation_for_training
from utils.train_callbacks import create_callbacks
from utils.train_qat_helper import validate_qat_data_flow, create_qat_representative_dataset

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False
    tfmot = None

def main():
    parser = argparse.ArgumentParser(description="Fine-tune an existing model on bad predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .keras model to fine-tune")
    parser.add_argument("--bad_images_dir", type=str, default="datasets/real_integra_bad_predictions", help="Folder containing bad images")
    parser.add_argument("--epochs", type=int, default=30, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Very low learning rate to prevent catastrophic forgetting")
    parser.add_argument("--weight", type=float, default=20.0, help="Weight multiplier for the bad images dataset")
    parser.add_argument("--tqt", action="store_true", help="Enable TQT/ESP-DL quantization pipeline after fine-tuning")
    parser.add_argument("--device", type=str, default=params.TQT_COLLECTING_DEVICE, choices=["cpu", "cuda"], 
                        help=f"Device for TQT (default: {params.TQT_COLLECTING_DEVICE})")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("🚀 STARTING FINE-TUNING PROCESS")
    print("="*50)
    
    if args.tqt:
        params.USE_TQT_PIPELINE = True
    
    if args.device:
        params.TQT_COLLECTING_DEVICE = args.device
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        sys.exit(1)

    print(f"📦 Loading pre-trained model: {args.model_path}")
    
    # Needs QAT scope if model was trained with QAT
    from utils.losses import DynamicSparseFocalLoss, DynamicFocalLoss, sparse_focal_loss, focal_loss
    custom_objects = {
        "DynamicSparseFocalLoss": DynamicSparseFocalLoss,
        "DynamicFocalLoss": DynamicFocalLoss,
        "sparse_focal_loss": sparse_focal_loss,
        "focal_loss": focal_loss
    }
    
    if QAT_AVAILABLE:
        custom_objects["quantize_scope"] = tfmot.quantization.keras.quantize_scope
    
    try:
        if QAT_AVAILABLE:
            with tfmot.quantization.keras.quantize_scope():
                model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects, safe_mode=False)
        else:
            model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects, safe_mode=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # 1. Dynamically Adjust Data Sources
    # We want to keep the original data (to prevent forgetting) but heavily weight the bad images
    bad_source_found = False
    for source in params.DATA_SOURCES:
        if source['path'] == args.bad_images_dir:
            source['weight'] = args.weight
            print(f"⚖️  Updated weight for {args.bad_images_dir} to {args.weight}x")
            bad_source_found = True
            break
            
    if not bad_source_found:
        print(f"➕ Adding {args.bad_images_dir} to Data Sources with weight {args.weight}x")
        params.DATA_SOURCES.append({
            'name': 'fine_tuning_bad_predictions',
            'type': 'label_file',
            'labels': f'labels_{params.NB_CLASSES}_shuffle.txt',
            'path': args.bad_images_dir,
            'weight': args.weight
        })

    # 2. Load the weighted dataset
    print("\n📊 Loading datasets (Original Data + Heavy Weighted Bad Predictions)...")
    from utils.multi_source_loader import load_combined_dataset
    x_data, y_data = load_combined_dataset()
    # x_data, y_data already shuffled by load_combined_dataset
    
    # Split directly 80/20 train/val for fine-tuning
    split_idx = int(len(x_data) * 0.8)
    x_train_raw, x_val_raw = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]
    
    print(f"✅ Loaded {len(x_train_raw)} training images and {len(x_val_raw)} validation images.")

    # 3. Create Dataset Pipelines
    print("\n🔄 Preprocessing and applying Augmentation...")
    batch_size = params.BATCH_SIZE
    
    x_train = preprocess_for_training(x_train_raw)
    x_val = preprocess_for_training(x_val_raw)
    
    train_dataset, val_dataset, _ = setup_augmentation_for_training(
        x_train, y_train, x_val, y_val, debug=False
    )

    # 4. Compile with fine-tuning learning rate
    print(f"\n⚙️ Compiling model with Low LR: {args.lr}")
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr) # Standard for this project
    
    # Use Dynamic loss if IntelligentFocalLossController is active to prevent mid-train recompile crashes
    if params.LOSS_TYPE in ["IntelligentFocalLossController", "focal_loss"]:
        from utils.losses import DynamicSparseFocalLoss
        loss_fn = DynamicSparseFocalLoss(from_logits=params.USE_LOGITS)
        print(f"🎯 Using DynamicSparseFocalLoss (Alpha: {params.FOCAL_ALPHA}, Gamma: {params.FOCAL_GAMMA})")
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=params.USE_LOGITS)
        print(f"🎯 Using Standard SparseCategoricalCrossentropy")
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # 5. Fine-Tune
    print("\n🔥 Starting Fine-Tuning Training loop...")
    
    # Set up export directory following project standard
    color_suffix = "GRAY" if params.INPUT_CHANNELS == 1 else "RGB"
    base_dir = f"exported_models/{params.NB_CLASSES}cls_{color_suffix}"
    export_dir = os.path.join(base_dir, f"retrained_{params.MODEL_ARCHITECTURE}")
    os.makedirs(export_dir, exist_ok=True)
    
    # ── Project Infrastructure for Callbacks ──
    from utils.train_modelmanager import TFLiteModelManager
    from utils.train_trainingmonitor import TrainingMonitor
    
    tflite_manager = TFLiteModelManager(export_dir)
    monitor = TrainingMonitor(export_dir)
    rep_x_processed = preprocess_for_training(x_train_raw[:100])
    
    callbacks = create_callbacks(
        output_dir=export_dir,
        tflite_manager=tflite_manager,
        representative_data=rep_x_processed,
        total_epochs=args.epochs,
        monitor=monitor,
        debug=True,
        validation_data=val_dataset,
        x_train_raw=x_train_raw[:100]
    )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    print("\n✅ Fine-tuning complete!")
    
    final_keras_name = f"retrained_{params.MODEL_ARCHITECTURE}.keras"
    final_keras_path = os.path.join(export_dir, final_keras_name)
    model.save(final_keras_path)
    print(f"📦 Saved fine-tuned Keras model to: {final_keras_path}")
    
    print("\n🎯 Converting to TFLite (INT8 Quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Provide representative dataset mapping to preprocessing
    rep_x = x_train_raw[:500]
    def representative_dataset_gen():
        for i in range(len(rep_x)):
            img = preprocess_for_training(tf.expand_dims(rep_x[i], 0))
            yield [img]
            
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if params.ESP_DL_QUANTIZE:
        print("INFO: Using INT8 quantization scheme for ESP-DL.")
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        print("INFO: Using standard UINT8 quantization scheme.")
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    try:
        tflite_model = converter.convert()
        tflite_path = os.path.join(export_dir, f"retrained_{params.MODEL_ARCHITECTURE}_ptq.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"✅ PTQ TFLite model saved to: {tflite_path}")
    except Exception as e:
        print(f"❌ TFLite Conversion failed: {e}")
        
    # ── TQT / ESP-DL Quantitative Pipeline ──
    if args.tqt:
        print("\n🚀 TRIGGERING ESP-DL TQT PIPELINE FOR FINETUNED MODEL...")
        from utils.export_onnx import export_keras_to_onnx
        # We use the name from params
        student_variant = params.MODEL_ARCHITECTURE
        onnx_path = os.path.join(export_dir, f"retrained_{student_variant}.onnx")
        
        if export_keras_to_onnx(model, onnx_path):
            import subprocess
            color_mode = "gray" if params.INPUT_CHANNELS == 1 else "rgb"
            cmd = [
                sys.executable, "quantize_espdl.py",
                "--model", f"retrained_{student_variant}",
                "--onnx", onnx_path,
                "--output", export_dir,
                "--bits", str(params.TQT_NUM_BITS),
                "--target", params.TQT_TARGET,
                "--classes", str(params.NB_CLASSES),
                "--color", color_mode,
                "--steps", str(params.TQT_STEPS),
                "--lr", str(params.TQT_LR),
                "--device", params.TQT_COLLECTING_DEVICE,
                "--no_simplify",
                "--skip_onnx_export",
                "--tflite" 
            ]
            
            print(f"   Executing TQT command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                print("✅ TQT Pipeline finished successfully for finetuned model")
            except subprocess.CalledProcessError as e:
                print(f"❌ TQT Pipeline failed with exit code {e.returncode}")
            except Exception as e:
                print(f"❌ TQT Pipeline error: {e}")
        else:
            print("❌ TQT Pipeline aborted: Finetuned ONNX export failed")

    print("\n🎉 FINE-TUNING WORKFLOW CONCLUDED SUCCESSFULLY!")
    print(f"Results located in: {export_dir}")

if __name__ == "__main__":
    main()
