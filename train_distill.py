#!/usr/bin/env python3
"""
train_distill.py  —  Top-level entry point for knowledge distillation.

Two-phase workflow
──────────────────
Phase 1 (--phase teacher):
    Train the teacher model normally. The teacher uses a large backbone
    (EfficientNetB0 or ResNet50) with optional ImageNet pre-training.
    The trained weights are saved to the checkpoint directory and reused
    in Phase 2.

    BETTER to train the teacher the normal way using train.py. It has augmentation and learning rate scheduler.

Phase 2 (--phase student):
    Load the pre-trained teacher (auto-detected from checkpoint dir) and
    train a lightweight student via knowledge distillation. The student
    is exported as a quantized TFLite model for edge deployment.

# Distill from your high-performance teacher
#python train_distill.py --phase student --teacher v30 --student v30_medium --classes 100 --color rgb --load-teacher "exported_models/100cls_RGB/digit_recognizer_v30_teacher_100cls_TIMESTAMP/model/best_model.keras" --epochs 80

Full pipeline (--phase all):
    Runs Phase 1 then Phase 2 sequentially.

Retrain existing models (--retrain-existing):
    Retrain existing edge models (v4, v16, etc.) using teacher guidance.
    Preserves model architecture while improving accuracy.

Examples
────────
# Train EfficientNet teacher — 10 classes, grayscale
python train_distill.py --phase teacher --teacher v30 --classes 10 --color gray

# Train EfficientNet teacher — 100 classes, RGB
python train_distill.py --phase teacher --teacher v30 --classes 100 --color rgb \\
    --teacher-epochs 100 --lr 0.001 --pretrained

# Distill a medium student from an existing teacher checkpoint
python train_distill.py --phase student --teacher v30 --student v30_medium \\
    --classes 100 --color rgb \\
    --load-teacher checkpoints/teacher_v30_100cls_rgb.keras \\
    --temperature 4 --alpha 0.7 --mode soft --epochs 80

# Full pipeline in one shot (train teacher then distill)
python train_distill.py --phase all --teacher v30 --student v30_medium \\
    --classes 10 --color gray --teacher-epochs 60 --epochs 60

# Retrain existing model with teacher
python train_distill.py --retrain-existing --existing-model v4 --teacher v30 \\
    --classes 10 --color gray --epochs 30

# Retrain v16 from checkpoint
python train_distill.py --retrain-existing --existing-model v16 \\
    --load-model-checkpoint checkpoints/v16_best.keras --teacher v30 --progressive

Available students
──────────────────
v30_micro, v30_small, v30_medium, v30_large  (depthwise separable)
v31_micro, v31_small, v31_medium, v31_large  (inverted residual + SE)

Available existing models for retraining
────────────────────────────────────────
v3, v4, v6, v7, v15, v16, v17, v18, v19
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# ── ensure project root is on sys.path ────────────────────────────────────
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Set class/channel env-vars BEFORE importing parameters.py
# (will be overridden again inside load_distillation_data, but avoids the
#  interactive prompt that parameters.py triggers when both env-vars are absent)
if "DIGIT_NB_CLASSES" not in os.environ:
    os.environ["DIGIT_NB_CLASSES"] = "10"
if "DIGIT_INPUT_CHANNELS" not in os.environ:
    os.environ["DIGIT_INPUT_CHANNELS"] = "1"

import parameters as params
from utils.train_distill_helper import (
    TEACHERS,
    STUDENTS,
    run_distillation_pipeline,
    train_teacher,
    load_distillation_data,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge distillation for digit recognizer",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Phase ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["teacher", "student", "all"],
        help=(
            "Pipeline phase:\n"
            "  teacher – train teacher only\n"
            "  student – distill student from saved teacher\n"
            "  all     – teacher then student (default)"
        ),
    )

    # ── Model selection ────────────────────────────────────────────────────
    parser.add_argument(
        "--teacher",
        type=str,
        default="v30",
        choices=list(TEACHERS.keys()),
        help="Teacher backbone (default: v30 = EfficientNetB0)"
    )
    parser.add_argument(
        "--student",
        type=str,
        default="v30_medium",
        choices=list(STUDENTS.keys()),
        help="Student variant (default: v30_medium)"
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--classes",
        type=int,
        default=10,
        choices=[10, 100],
        help="Number of output classes (default: 10)"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="gray",
        choices=["gray", "rgb"],
        help="Input color mode (default: gray)"
    )

    # ── Teacher training ───────────────────────────────────────────────────
    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=60,
        metavar="N",
        help="Teacher training epochs (default: 60)"
    )
    parser.add_argument(
        "--teacher-lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="Teacher learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained backbone for teacher (default: True)"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train teacher backbone from scratch (disables --pretrained)"
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze teacher backbone during training"
    )
    parser.add_argument(
        "--load-teacher",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to an existing teacher .keras file (skip teacher training)"
    )

    # ── Distillation ───────────────────────────────────────────────────────
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        metavar="N",
        help="Student distillation epochs (default: 60)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="Student learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=params.DISTILLATION_TEMPERATURE,
        help=f"Distillation temperature (default: {params.DISTILLATION_TEMPERATURE})"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=params.DISTILLATION_ALPHA,
        help=(
            f"Hard-label weight 0→1 (0=all teacher, 1=all hard) "
            f"(default: {params.DISTILLATION_ALPHA})"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=params.DISTILLATION_MODE,
        choices=["soft", "hard", "hybrid"],
        help=f"Distillation mode (default: {params.DISTILLATION_MODE})"
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        default=params.USE_PROGRESSIVE_DISTILLATION,
        help="Use ProgressiveDistiller (dynamic temperature & alpha)"
    )

    # ── Shared / infrastructure ────────────────────────────────────────────
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        metavar="B",
        help="Batch size for both teacher and student training (default: 32)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        metavar="DIR",
        help="Directory for saving checkpoints (default: checkpoints/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for TFLite export (auto-generated if not set)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip TFLite quantization export"
    )
    parser.add_argument(
        "--target-hardware",
        type=str,
        default="esp32",
        choices=["esp32", "raspberry_pi", "generic"],
        help="Target hardware for TFLite quantization (default: esp32)"
    )

    # ── Retrain existing model ────────────────────────────────────────────
    parser.add_argument(
        "--retrain-existing",
        action="store_true",
        help="Retrain an existing edge model (v4, v16) with teacher"
    )

    parser.add_argument(
        "--existing-model",
        type=str,
        default=None,
        choices=["v3", "v4", "v6", "v7", "v15", "v16", "v17", "v18", "v19"],
        help="Existing edge model to retrain (requires --retrain-existing)"
    )
    
    parser.add_argument(
        "--load-model-checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to existing model weights for retraining"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    
    # Validate retraining doesn't conflict with phase
    if args.retrain_existing and args.existing_model:
        if args.phase != "all":
            logger.info("--retrain-existing overrides --phase. Running retraining pipeline.")
        
        from utils.retrain_with_teacher import main as retrain_main
        
        # Redirect arguments to retrain_with_teacher
        sys.argv = [
            sys.argv[0],
            "--model", args.existing_model,
            "--teacher", args.teacher,
            "--classes", str(args.classes),
            "--color", args.color,
            "--temperature", str(args.temperature),
            "--alpha", str(args.alpha),
            "--mode", args.mode,
            "--progressive" if args.progressive else "",
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--batch-size", str(args.batch),
            "--load-checkpoint", args.load_model_checkpoint if args.load_model_checkpoint else "",
            "--teacher-checkpoint", args.load_teacher if args.load_teacher else "",
            "--quantize" if not args.no_quantize else "",
            "--output-dir", args.output_dir if args.output_dir else "",
        ]
        # Remove empty strings
        sys.argv = [arg for arg in sys.argv if arg]
        
        retrain_main()
        return

    pretrained    = args.pretrained and not args.no_pretrained
    color_mode    = args.color
    num_classes   = args.classes
    export_quant  = not args.no_quantize

    logger.info("=" * 60)
    logger.info("🚀  Distillation Configuration")
    logger.info("=" * 60)
    logger.info(f"  Phase:           {args.phase}")
    logger.info(f"  Teacher:         {args.teacher}")
    logger.info(f"  Student:         {args.student}")
    logger.info(f"  Classes:         {num_classes}")
    logger.info(f"  Color mode:      {color_mode.upper()}")
    logger.info(f"  Pretrained:      {pretrained}")
    logger.info(f"  Temperature:     {args.temperature}")
    logger.info(f"  Alpha:           {args.alpha}")
    logger.info(f"  Mode:            {args.mode}")
    logger.info(f"  Progressive:     {args.progressive}")
    logger.info("=" * 60)

    if args.phase == "teacher":
        # ── Train teacher only ────────────────────────────────────────────
        x_train, y_train, x_val, y_val, x_test, y_test = load_distillation_data(
            num_classes=num_classes,
            color_mode=color_mode,
        )
        # ── Output directory logic (similar to train.py) ───────────────────
        color_label = color_mode.upper()
        timestamp = datetime.now().strftime("%m%d_%H%M")
        run_folder = f"teacher_{args.teacher}_{num_classes}cls_{color_label}_{timestamp}"
        
        # Base directory consistent with train.py
        output_dir = os.path.join(
            "exported_models",
            f"{num_classes}cls_{color_label}",
            run_folder
        )
        # Main model assets go into 'model' subdirectory
        model_dir = os.path.join(output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        train_teacher(
            teacher_type=args.teacher,
            num_classes=num_classes,
            color_mode=color_mode,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=args.teacher_epochs,
            batch_size=args.batch,
            learning_rate=args.teacher_lr,
            checkpoint_dir=model_dir, # Save in model/ folder
            pretrained=pretrained,
            freeze_backbone=args.freeze_backbone,
        )
        logger.info(f"Teacher training session saved to: {output_dir}")

    elif args.phase == "student":
        # ── Distill student from existing teacher ─────────────────────────
        run_distillation_pipeline(
            teacher_type=args.teacher,
            student_variant=args.student,
            num_classes=num_classes,
            color_mode=color_mode,
            teacher_checkpoint=args.load_teacher,
            # Skip teacher training
            teacher_epochs=0,
            teacher_lr=args.teacher_lr,
            teacher_pretrained=pretrained,
            teacher_freeze_backbone=args.freeze_backbone,
            # Student
            student_epochs=args.epochs,
            student_lr=args.lr,
            temperature=args.temperature,
            alpha=args.alpha,
            mode=args.mode,
            use_progressive=args.progressive,
            batch_size=args.batch,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            export_quantized=export_quant,
            target_hardware=args.target_hardware,
        )

    else:  # "all"
        # ── Full pipeline ─────────────────────────────────────────────────
        results = run_distillation_pipeline(
            teacher_type=args.teacher,
            student_variant=args.student,
            num_classes=num_classes,
            color_mode=color_mode,
            teacher_checkpoint=args.load_teacher,
            teacher_epochs=args.teacher_epochs,
            teacher_lr=args.teacher_lr,
            teacher_pretrained=pretrained,
            teacher_freeze_backbone=args.freeze_backbone,
            student_epochs=args.epochs,
            student_lr=args.lr,
            temperature=args.temperature,
            alpha=args.alpha,
            mode=args.mode,
            use_progressive=args.progressive,
            batch_size=args.batch,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            export_quantized=export_quant,
            target_hardware=args.target_hardware,
        )

        logger.info("\n" + "=" * 60)
        logger.info("✅  DISTILLATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Teacher accuracy:  {results['teacher']['accuracy']:.4f}")
        logger.info(f"  Student accuracy:  {results['student']['accuracy']:.4f}")
        logger.info(
            f"  Accuracy retention: "
            f"{results['comparison']['accuracy_retention']:.2%}"
        )
        logger.info(
            f"  Compression ratio:  "
            f"{results['comparison']['compression_ratio']:.1f}×"
        )
        if results["student"].get("tflite_path"):
            logger.info(f"  TFLite export:     {results['student']['tflite_path']}")


if __name__ == "__main__":
    main()