#!/usr/bin/env python3
"""
distill_all.py  —  Multi-teacher ensemble distillation command generator.

Reads the ``model_comparison.csv`` for a given ``(classes, color)``
combination, collects all model directories that contain a valid ``.keras``
file, and generates one ``train_distill.py --phase student`` command per
student variant.

For each student model, all *other* models serve as the teacher ensemble.
This way every model benefits from the collective knowledge of the rest.

Modes
─────
    --script    (default) Print the command line only — copy-paste friendly.
    --execute   Print the command and then run it immediately (one at a time).

Examples
────────
    # Generate commands for ALL models as students (8 → 8 commands)
    python distill_all.py --classes 10 --color rgb

    # Generate + execute all (runs sequentially)
    python distill_all.py --classes 10 --color rgb --execute

    # Only a specific student (one command)
    python distill_all.py --classes 10 --color rgb --student v4

    # Custom hyper-parameters for all students
    python distill_all.py --cls 10 --color rgb --temperature 8.0 --alpha 0.7 --epochs 200

    # Only grayscale
    python distill_all.py --classes 100 --color gray --mode hard
"""

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Path helpers
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


# ═══════════════════════════════════════════════════════════════════════════
#  Student version extraction
# ═══════════════════════════════════════════════════════════════════════════

_STUDENT_RE = re.compile(
    r"digit_recognizer_(v\d+)_"
)


def extract_student_version(directory: str) -> str | None:
    """
    Extract the student version (e.g. ``v16``) from a directory name.

    Pattern: ``digit_recognizer_v16_10cls_RGB_TQT_SOFTMAX_0610_2030``
    Returns: ``v16``
    """
    m = _STUDENT_RE.search(directory)
    return m.group(1) if m else None


# ═══════════════════════════════════════════════════════════════════════════
#  Collect teachers
# ═══════════════════════════════════════════════════════════════════════════

class ModelEntry:
    """One model found in the CSV with a valid .keras file."""

    def __init__(self, directory: str, accuracy: float, version: str):
        self.directory = directory
        self.accuracy = accuracy
        self.version = version

    def __repr__(self) -> str:
        return f"{self.version} ({self.accuracy:.2%})"


def collect_entries(records: list[dict], export_base: str) -> list[ModelEntry]:
    """
    Return a deduplicated, ordered list of ``ModelEntry`` items whose
    directories exist and contain at least one ``.keras`` file.
    """
    seen_dir: set[str] = set()
    seen_ver: set[str] = set()
    entries: list[ModelEntry] = []

    for rec in records:
        directory = rec.get("Directory", "").strip()
        if not directory:
            continue
        if directory in seen_dir:
            continue

        version = extract_student_version(directory)
        if version is None:
            logger.warning(f"  ⚠️  Could not extract version from '{directory}' — skipping")
            continue

        # Verify the directory actually contains a .keras file
        full_path = os.path.join(export_base, directory)
        keras_files = list(Path(full_path).rglob("*.keras"))
        if not keras_files:
            logger.warning(f"  ⚠️  Skipping '{directory}' — no .keras file found")
            continue

        # Keep the entry with the highest accuracy for this version
        if version in seen_ver:
            # Replace with better one
            idx = next(i for i, e in enumerate(entries) if e.version == version)
            if rec["Accuracy"] > entries[idx].accuracy:
                entries[idx] = ModelEntry(directory, rec["Accuracy"], version)
            continue

        seen_dir.add(directory)
        seen_ver.add(version)
        entries.append(ModelEntry(directory, rec["Accuracy"], version))
        logger.info(f"  ✓  {version:>4}  →  {directory}  (acc={rec['Accuracy']:.2%})")

    return entries


# ═══════════════════════════════════════════════════════════════════════════
#  Command builder
# ═══════════════════════════════════════════════════════════════════════════

def build_command(
    teachers: list[ModelEntry],
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

    # Teachers — all other model directories
    cmd.append("--teachers")
    for t in teachers:
        cmd.append(t.directory)

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
        description="Multi-teacher ensemble distillation command generator",
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
        default=None,
        help="Single student model version (e.g. v4). "
             "If omitted, generates commands for ALL available versions.",
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
        help="Print the command(s) only (default behaviour without --execute).",
    )
    group.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Print the command AND run it immediately.",
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

    # ── Parse CSV and collect model entries ───────────────────────────────
    logger.info(f"📄  Reading {csv_fpath}")
    records = parse_csv(csv_fpath)

    logger.info(f"\n🔍  Scanning for models with .keras files under {export_base}")
    entries = collect_entries(records, export_base)

    if not entries:
        logger.error(
            "❌  No models with .keras files found.\n"
            "    Make sure at least one model has been trained and exported."
        )
        sys.exit(1)

    # ── Determine which students to process ───────────────────────────────
    if args.student:
        # Single student — validate it exists
        matching = [e for e in entries if e.version == args.student]
        if not matching:
            logger.error(
                f"❌  Student version '{args.student}' not found in CSV.\n"
                f"    Available versions: {', '.join(e.version for e in entries)}"
            )
            sys.exit(1)
        student_versions = [args.student]
    else:
        # ALL versions — each takes a turn as student
        student_versions = [e.version for e in entries]

    # ── Generate commands ────────────────────────────────────────────────
    import shlex  # for safe quoting

    for sv in student_versions:
        # All entries except the student
        teachers = [e for e in entries if e.version != sv]

        if not teachers:
            logger.warning(f"  ⚠️  Skipping {sv} — no other models to use as teachers")
            continue

        cmd = build_command(
            teachers=teachers,
            student=sv,
            classes=args.classes,
            color=args.color,
            temperature=args.temperature,
            alpha=args.alpha,
            epochs=args.epochs,
            mode=args.mode,
            progressive=args.progressive,
            tqt=args.tqt,
        )

        command_str = " ".join(shlex.quote(c) for c in cmd)

        # ── Print ────────────────────────────────────────────────────────
        print()
        print("=" * 70)
        print(f"🎯  Distill all INTO student:  {sv}")
        print(f"    Teachers ({len(teachers)}):")
        for t in teachers:
            print(f"      • {t.version:>4}  {t.directory}")
        print("─" * 70)
        print(f"$ {command_str}")
        print("=" * 70)

        # ── Execute ──────────────────────────────────────────────────────
        if args.execute:
            logger.info(f"🚀  Executing {sv} distillation...\n")
            result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or ".")
            if result.returncode != 0:
                logger.error(
                    f"❌  {sv} distillation failed with exit code {result.returncode}"
                )
                sys.exit(result.returncode)
            logger.info(f"✅  {sv} distillation completed successfully.\n")

    if not args.execute:
        print()
        logger.info(
            "💡  Use --execute to run all, or copy-paste individual commands above."
        )


if __name__ == "__main__":
    main()