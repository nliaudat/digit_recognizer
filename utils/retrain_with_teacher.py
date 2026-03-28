#!/usr/bin/env python3
"""
Retrain existing edge models (v4, v16, etc.) using a teacher model.
This does not modify models/__init__.py - imports directly from model files.
"""

import os
import sys
import argparse
import logging
import json
import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# Add project root to path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import parameters as params
from utils.distiller import Distiller, ProgressiveDistiller
from utils.model_distiller_utils import (
    export_student_for_edge,
    evaluate_distilled_model,
    get_model_size_kb,
    freeze_teacher_model,
)
from utils.train_distill_helper import load_distillation_data

# Import teacher models directly
from models.digit_recognizer_v30_teacher import create_v30_teacher
from models.digit_recognizer_v31_teacher import create_v31_teacher

# Import existing edge models directly (no __init__.py modification)
try:
    from models.digit_recognizer_v3 import create_digit_recognizer_v3
except ImportError:
    create_digit_recognizer_v3 = None

try:
    from models.digit_recognizer_v4 import create_digit_recognizer_v4
except ImportError:
    create_digit_recognizer_v4 = None

try:
    from models.digit_recognizer_v7 import create_digit_recognizer_v7
except ImportError:
    create_digit_recognizer_v7 = None

try:
    from models.digit_recognizer_v15 import create_digit_recognizer_v15
except ImportError:
    create_digit_recognizer_v15 = None

try:
    from models.digit_recognizer_v16 import create_digit_recognizer_v16
except ImportError:
    create_digit_recognizer_v16 = None

try:
    from models.digit_recognizer_v17 import create_digit_recognizer_v17
except ImportError:
    create_digit_recognizer_v17 = None

try:
    from models.digit_recognizer_v18 import create_digit_recognizer_v18
except ImportError:
    create_digit_recognizer_v18 = None

try:
    from models.digit_recognizer_v19 import create_digit_recognizer_v19
except ImportError:
    create_digit_recognizer_v19 = None

# Registry for existing edge models
EXISTING_EDGE_MODELS = {
    "v3": {
        "model_fn": create_digit_recognizer_v3,
        "description": "Fast overall inference speed",
        "size_kb": 38.4,
        "default_input_shape": (32, 20, 3),
        "available": create_digit_recognizer_v3 is not None,
    },
    "v4": {
        "model_fn": create_digit_recognizer_v4,
        "description": "Excellent accuracy under 100KB",
        "size_kb": 78.3,
        "default_input_shape": (32, 20, 3),
        "available": create_digit_recognizer_v4 is not None,
    },
    "v7": {
        "model_fn": create_digit_recognizer_v7,
        "description": "Fastest inference speed",
        "size_kb": 47.2,
        "default_input_shape": (32, 20, 3),
        "available": create_digit_recognizer_v7 is not None,
    },
    "v15": {
        "model_fn": create_digit_recognizer_v15,
        "description": "Best accuracy under 100KB",
        "size_kb": 100.0,
        "default_input_shape": (32, 20, 3),
        "available": create_digit_recognizer_v15 is not None,
    },
    "v16": {
        "model_fn": create_digit_recognizer_v16,
        "description": "High accuracy MobileNetV2-based",
        "size_kb": 128.8,
        "default_input_shape": (32, 20, 3),
        "available": create_digit_recognizer_v16 is not None,
    },
    "v17": {
        "model_fn": create_digit_recognizer_v17,
        "description": "GhostNet-inspired alternative",
        "size_kb": 71.0,
        "default_input_shape": (32, 20, 3),
        "available": create_digit_recognizer_v17 is not None,
    },
    "v18": {
        "model_fn": create_digit_recognizer_v18,
        "description": "New variant with strong performance",
        "size_kb": 97.4,
        "default_input_shape": (32, 20, 3),
        "available": create_digit_recognizer_v18 is not None,
    },
    "v19": {
        "model_fn": create_digit_recognizer_v19,
        "description": "High-capacity variant",
        "size_kb": 132.2,
        "default_input_shape": (32, 20, 3),
        "available": create_digit_recognizer_v19 is not None,
    },
}

TEACHERS = {
    "v30": create_v30_teacher,
    "v31": create_v31_teacher,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain existing edge models with teacher supervision"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Existing edge model to retrain. Available: {list(EXISTING_EDGE_MODELS.keys())}"
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="v30",
        choices=list(TEACHERS.keys()),
        help="Teacher model for distillation"
    )
    
    # Dataset
    parser.add_argument(
        "--classes",
        type=int,
        default=10,
        choices=[10, 100],
        help="Number of classes"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="gray",
        choices=["gray", "rgb"],
        help="Input color mode"
    )
    
    # Distillation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Balance between hard and soft labels"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="soft",
        choices=["soft", "hard", "hybrid"],
        help="Distillation mode"
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Use progressive distillation"
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Retraining epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (lower for fine-tuning)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to existing model weights (optional)"
    )
    
    # Teacher
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="Path to teacher weights (optional)"
    )
    parser.add_argument(
        "--teacher-pretrained",
        action="store_true",
        default=True,
        help="Use pretrained teacher"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_models/",
        help="Output directory"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Export quantized TFLite"
    )
    
    return parser.parse_args()


def load_or_create_model(
    model_name: str,
    num_classes: int,
    color_mode: str,
    load_path: Optional[str] = None
) -> tf.keras.Model:
    """
    Load existing model or create new one.
    """
    # Validate model exists
    if model_name not in EXISTING_EDGE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(EXISTING_EDGE_MODELS.keys())}")
    
    model_info = EXISTING_EDGE_MODELS[model_name]
    
    if not model_info["available"]:
        raise ImportError(f"Model {model_name} cannot be imported. Check if the model file exists.")
    
    channels = 1 if color_mode == "gray" else 3
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
    
    model_fn = model_info["model_fn"]
    
    # Create model
    try:
        model = model_fn(num_classes=num_classes, input_shape=input_shape)
    except TypeError:
        # Fallback for models that don't accept input_shape parameter
        logger.warning(f"{model_name} may not accept input_shape parameter, trying without...")
        model = model_fn(num_classes=num_classes)
    
    # Load weights if provided
    if load_path and os.path.exists(load_path):
        logger.info(f"Loading weights from {load_path}")
        model.load_weights(load_path)
    else:
        logger.info(f"Creating new {model_name} model with random weights")
    
    logger.info(f"Model: {model_name}, params: {model.count_params():,}")
    
    return model


def retrain_with_teacher(
    model: tf.keras.Model,
    teacher: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace
) -> Tuple[Distiller, str]:
    """
    Retrain model using teacher distillation.
    """
    logger.info("=" * 60)
    logger.info(f"Retraining {args.model} with teacher {args.teacher}")
    logger.info(f"Mode: {args.mode}, T={args.temperature}, α={args.alpha}")
    logger.info("=" * 60)
    
    # Freeze teacher
    teacher = freeze_teacher_model(teacher)
    
    # Create distiller
    if args.progressive:
        distiller = ProgressiveDistiller(
            student=model,
            teacher=teacher,
            initial_temperature=8.0,
            final_temperature=2.0,
            initial_alpha=0.3,
            final_alpha=0.8,
            total_epochs=args.epochs,
            mode=args.mode,
        )
    else:
        distiller = Distiller(
            student=model,
            teacher=teacher,
            temperature=args.temperature,
            alpha=args.alpha,
            mode=args.mode,
        )
    
    # Compile with lower learning rate for fine-tuning
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        metrics=["accuracy"],
    )
    
    # Callbacks
    checkpoint_dir = "checkpoints/retrain"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    ckpt_path = os.path.join(
        checkpoint_dir,
        f"{args.model}_retrained_{args.classes}cls_{args.color}_{timestamp}.keras"
    )
    
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]
    
    # Train
    history = distiller.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Checkpoint saved: {ckpt_path}")
    
    return distiller, ckpt_path


def main():
    args = parse_args()
    
    # Validate model exists and is available
    if args.model not in EXISTING_EDGE_MODELS:
        logger.error(f"Model '{args.model}' not found.")
        logger.info(f"Available models: {list(EXISTING_EDGE_MODELS.keys())}")
        sys.exit(1)
    
    model_info = EXISTING_EDGE_MODELS[args.model]
    if not model_info["available"]:
        logger.error(f"Model '{args.model}' cannot be imported.")
        logger.info("Check that the model file exists in models/ directory.")
        sys.exit(1)
    
    # Determine input shape
    channels = 1 if args.color == "gray" else 3
    input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
    
    logger.info("=" * 60)
    logger.info("🚀 Retraining Existing Model with Teacher")
    logger.info("=" * 60)
    logger.info(f"  Model:      {args.model}")
    logger.info(f"  Teacher:    {args.teacher}")
    logger.info(f"  Classes:    {args.classes}")
    logger.info(f"  Color:      {args.color.upper()}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Alpha:      {args.alpha}")
    logger.info(f"  Mode:       {args.mode}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  LR:         {args.lr}")
    logger.info("=" * 60)
    
    # Load dataset
    x_train, y_train, x_val, y_val, x_test, y_test = load_distillation_data(
        num_classes=args.classes,
        color_mode=args.color,
    )
    
    # Load or create teacher
    if args.teacher_checkpoint and os.path.exists(args.teacher_checkpoint):
        logger.info(f"Loading teacher from {args.teacher_checkpoint}")
        teacher = tf.keras.models.load_model(args.teacher_checkpoint)
    else:
        teacher_builder = TEACHERS[args.teacher]
        teacher_input_shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
        teacher = teacher_builder(
            num_classes=args.classes,
            input_shape=teacher_input_shape,
            pretrained=args.teacher_pretrained,
        )
    
    # Load or create model to retrain
    model = load_or_create_model(
        args.model,
        args.classes,
        args.color,
        args.load_checkpoint
    )
    
    # Evaluate baseline accuracy
    logger.info("\n📊 Baseline evaluation (before retraining):")
    baseline_metrics = evaluate_distilled_model(model, (x_test, y_test))
    logger.info(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # Retrain with teacher
    distiller, checkpoint_path = retrain_with_teacher(
        model, teacher,
        x_train, y_train,
        x_val, y_val,
        args
    )
    
    # Get retrained model
    retrained_model = distiller.get_student()
    
    # Evaluate after retraining
    logger.info("\n📊 After retraining evaluation:")
    after_metrics = evaluate_distilled_model(retrained_model, (x_test, y_test))
    logger.info(f"  Accuracy: {after_metrics['accuracy']:.4f}")
    improvement = (after_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    logger.info(f"  Improvement: {improvement:+.2f}%")
    
    # Export
    if args.quantize:
        color_label = args.color.upper()
        timestamp = datetime.now().strftime("%m%d_%H%M")
        export_dir = os.path.join(
            args.output_dir,
            f"{args.classes}cls_{color_label}",
            f"{args.model}_retrained_{timestamp}"
        )
        os.makedirs(export_dir, exist_ok=True)
        
        export_path = os.path.join(export_dir, "model", args.model)
        tflite_path = export_student_for_edge(
            retrained_model,
            export_path,
            quantize=True,
            target_hardware="esp32"
        )
        logger.info(f"Exported to: {tflite_path}")
    
    # Save results
    results = {
        "model": args.model,
        "teacher": args.teacher,
        "classes": args.classes,
        "color_mode": args.color,
        "distillation": {
            "mode": args.mode,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "progressive": args.progressive,
        },
        "baseline_accuracy": baseline_metrics["accuracy"],
        "retrained_accuracy": after_metrics["accuracy"],
        "improvement": after_metrics["accuracy"] - baseline_metrics["accuracy"],
        "checkpoint": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
    }
    
    results_path = os.path.join("checkpoints", f"retrain_{args.model}_{args.classes}cls.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {results_path}")
    
    return retrained_model, results


if __name__ == "__main__":
    main()