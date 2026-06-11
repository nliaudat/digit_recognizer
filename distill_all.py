#!/usr/bin/env python3
"""
distill_all.py  —  Multi-teacher ensemble distillation command builder.

Reads the ``model_comparison.csv`` for a given ``(classes, color)``
combination, collects *all* teacher directories that contain a valid ``.keras``
file, and builds the corresponding ``train_distill.py --phase student``
command string.

The generated command distills every available teacher as an ensemble into
a lightweight student model.

Modes
─────
    --script    (default) Print the command line only — copy-paste friendly.
    --execute   Print the command and then run it immediately.

Examples
────────
    # Print the command (copy-paste safe)
    python distill_all.py --classes 10 --color rgb

    # Print + run
    python distill_all.py --classes 10 --color rgb --execute

    # Custom student + hyper-parameters
    python distill_all.py --cls 10 --color rgb --student v15 \\
        --temperature 8.0 --alpha 0.7 --epochs 200

    # Only grayscale teachers
    python distill_all.py --classes 100 --color gray --student v23
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Path helpers  (mirrors distill_best.py conventions)
# ═══════════════════════════════════════════════════════════════════════════

def csv_path(num_classes: int, color_mode: str) -> str:
    color_label = color_mode.upper()
    return f"exported_models/{num_classes}cls_{color_label}/test_results/model_comparison.csv"


def export_base_dir(num_classes: int, color_mode: str) -> str:
    color_label = color_mode.upper()
    return f"exported_models/{num_classes}cls_{color_label}"


# ═══════════════════════════════════════════════════════════════════════════
#  CSV parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_csv(filepath: str) -> list[dict]:
    """Parse model_comparison.csv into a list of row dicts."""
    records = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["Accuracy"] = float(row["Accuracy"])
            records.append(row)
    return records


def collect_teacher_dirs(records: list[dict], export_base: str) -> list[str]:
    """
    Return a deduplicated, ordered list of directory names (relative to
    *export_base*) that contain at least one ``.keras`` file.
    """
    seen: set[str] = set()
    directories: list[str] = []

    for rec in records:
        directory = rec.get("Directory", "").strip()
        if not directory:
            continue
        if directory in seen:
            continue

        # Verify the directory actually exists and contains a .keras file
        full_path = os.path.join(export_base, directory)
        keras_files = list(Path(full_path).rglob("*.keras"))
        if not keras_files:
            logger.warning(f"  ⚠️  Skipping '{directory}' — no .keras file found")
            continue

        seen.add(directory)
        directories.append(directory)
        logger.info(
            f"  ✓  {directory}  "
            f"(acc={rec['Accuracy']:.2%})"
        )

    return directories


# ═══════════════════════════════════════════════════════════════════════════
#  Command builder
# ═══════════════════════════════════════════════════════════════════════════

def build_command(
    teacher_dirs: list[str],
    student: str,
    classes: int,
    color: str,
    temperature: float,
    alpha: float,
    epochs: int,
    mode: str,
    progressive: bool,
    tqt: bool,
) -> list[str]:
    """
    Build the ``train_distill.py --phase student`` command as a list of
    arguments suitable for ``subprocess.run``.
    """
    cmd = [
        sys.executable or "python",
        "train_distill.py",
        "--phase", "student",
    ]

    # Teachers (directories relative to exported_models/{cls}cls_{COLOR}/)
    cmd.append("--teachers")
    cmd.extend(teacher_dirs)

    # Student
    cmd.extend(["--student", student])

    # Hyper-parameters
    cmd.extend(["--temperature", str(temperature)])
    cmd.extend(["--alpha", str(alpha)])
    cmd.extend(["--color", color])
    cmd.extend(["--classes", str(classes)])
    cmd.extend(["--epochs", str(epochs)])
    cmd.extend(["--mode", mode])

    if progressive:
        cmd.append("--progressive")
    if tqt:
        cmd.append("--tqt")

    return cmd


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-teacher ensemble distillation command builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--classes", "--cls",
        dest="classes",
        type=int,
        required=True,
        choices=[10, 100],
        help="Number of output classes (10 or 100) [required]",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="rgb",
        choices=["gray", "rgb"],
        help="Color mode (default: rgb)",
    )
    parser.add_argument(
        "--student",
        type=str,
        default="v23",
        help="Student model architecture (default: v23)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=6.0,
        help="Distillation temperature (default: 6.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hard-label weight 0→1 (default: 0.5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Max distillation epochs (default: 250)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="soft",
        choices=["soft", "hard", "hybrid"],
        help="Distillation mode (default: soft)",
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        default=True,
        help="Use ProgressiveDistiller (default: enabled)",
    )
    parser.add_argument(
        "--no-progressive",
        action="store_false",
        dest="progressive",
        help="Disable ProgressiveDistiller",
    )
    parser.add_argument(
        "--tqt",
        action="store_true",
        default=True,
        help="Enable TQT/ESP-DL quantization pipeline (default: enabled)",
    )
    parser.add_argument(
        "--no-tqt",
        action="store_false",
        dest="tqt",
        help="Disable TQT/ESP-DL quantization pipeline",
    )

    # ── Mode selection ──────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--script",
        action="store_true",
        default=False,
        help="Print the command line only (default behaviour without --execute)",
    )
    group.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Print the command AND run it immediately",
    )

    return parser.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────
    csv_fpath = csv_path(args.classes, args.color)
    export_base = export_base_dir(args.classes, args.color)

    if not os.path.isfile(csv_fpath):
        logger.error(
            f"❌  CSV not found: {csv_fpath}\n"
            f"    Have you run the benchmark first?"
        )
        sys.exit(1)

    # ── Parse CSV and collect teacher directories ─────────────────────────
    logger.info(f"📄  Reading {csv_fpath}")
    records = parse_csv(csv_fpath)

    logger.info(f"\n🔍  Scanning for teachers with .keras files under {export_base}")
    teacher_dirs = collect_teacher_dirs(records, export_base)

    if not teacher_dirs:
        logger.error(
            "❌  No teachers with .keras files found.\n"
            "    Make sure at least one model has been trained and exported."
        )
        sys.exit(1)

    # ── Build command ─────────────────────────────────────────────────────
    cmd = build_command(
        teacher_dirs=teacher_dirs,
        student=args.student,
        classes=args.classes,
        color=args.color,
        temperature=args.temperature,
        alpha=args.alpha,
        epochs=args.epochs,
        mode=args.mode,
        progressive=args.progressive,
        tqt=args.tqt,
    )

    # ── Print summary ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("📋  DISTILL ALL — Command summary")
    print("=" * 70)
    print(f"   Teachers ({len(teacher_dirs)}):")
    for d in teacher_dirs:
        print(f"     • {d}")
    print(f"   Student:      {args.student}")
    print(f"   Classes:      {args.classes}")
    print(f"   Color:        {args.color.upper()}")
    print(f"   Temperature:  {args.temperature}")
    print(f"   Alpha:        {args.alpha}")
    print(f"   Epochs:       {args.epochs}")
    print(f"   Mode:         {args.mode}")
    print(f"   Progressive:  {args.progressive}")
    print(f"   TQT:          {args.tqt}")
    print()

    # ── Print the command ────────────────────────────────────────────────
    # Use shlex.quote for safe copy-paste on the shell
    import shlex
    command_str = " ".join(shlex.quote(c) for c in cmd)
    print("─" * 70)
    print(f"$ {command_str}")
    print("─" * 70)
    print()

    if args.execute:
        logger.info("🚀  Executing command...\n")
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or ".")
        if result.returncode != 0:
            logger.error(f"❌  Command failed with exit code {result.returncode}")
            sys.exit(result.returncode)
        logger.info("✅  Distillation completed successfully.")
    else:
        logger.info(
            "💡  Use --execute to run, or copy-paste the command above."
        )


if __name__ == "__main__":
    main()