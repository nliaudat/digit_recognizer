#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_quant_analysis.py – Execute the comprehensive quantisation analysis
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------
# Make sure the repository root is on the import path
# ----------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in os.sys.path:
    os.sys.path.insert(0, str(repo_root))

# ----------------------------------------------------------------------
# Project‑specific imports (these live in the top‑level `train.py`)
# ----------------------------------------------------------------------
from train import get_data_splits, create_model, compile_model
from utils.quantization_analysis import analyze_quantization_impact, QuantizationAnalyzer
import parameters as params


def main():
    parser = argparse.ArgumentParser(
        description="Run QuantizationAnalyzer on the latest trained model."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="exported_models",
        help="Root folder that contains the training output directories.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debug information.",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Locate the most recent training folder (the one you just trained)
    # --------------------------------------------------------------
    export_root = Path(args.model_dir)
    if not export_root.is_dir():
        raise FileNotFoundError(f"Folder not found: {export_root}")

    # Find the newest sub‑folder (by modification time)
    latest_folder = max(
        (d for d in export_root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    print(f"🔍 Using latest training folder: {latest_folder}")

    # --------------------------------------------------------------
    # Find the quantised TFLite model inside that folder
    # --------------------------------------------------------------
    tflite_path = None
    for f in latest_folder.rglob("*.tflite"):
        if not f.name.lower().endswith("_float.tflite"):
            tflite_path = f
            break
    if tflite_path is None:
        raise FileNotFoundError("No quantised .tflite model found in the folder.")
    print(f"📦 Quantised model: {tflite_path}")

    # --------------------------------------------------------------
    # Load the test data (the same split you used for training)
    # --------------------------------------------------------------
    (_, _), (_, _), (x_test, y_test) = get_data_splits()

    # --------------------------------------------------------------
    # Re‑create the exact Keras model architecture that was trained
    # --------------------------------------------------------------
    model = create_model()
    # Use the same loss type that the trainer used
    loss_type = (
        "categorical_crossentropy"
        if params.MODEL_ARCHITECTURE == "original_haverland"
        else "sparse_categorical_crossentropy"
    )
    model = compile_model(model, loss_type=loss_type)

    # --------------------------------------------------------------
    # Load the best weights saved by the trainer
    # --------------------------------------------------------------
    best_keras_path = latest_folder / "best_model.keras"
    if not best_keras_path.is_file():
        raise FileNotFoundError(f"Best Keras model not found: {best_keras_path}")
    model.load_weights(best_keras_path)
    print(f"✅ Loaded Keras weights from: {best_keras_path}")

    # --------------------------------------------------------------
    # Run the quantitative analysis
    # --------------------------------------------------------------
    results = analyze_quantization_impact(
        keras_model=model,
        tflite_model_path=str(tflite_path),
        x_test=x_test,
        y_test=y_test,
        debug=args.debug,
    )

    # --------------------------------------------------------------
    # Save a human‑readable report next to the TFLite file
    # --------------------------------------------------------------
    analyzer = QuantizationAnalyzer(debug=args.debug)
    analyzer.analysis_results = results
    analyzer.generate_quantization_report(str(latest_folder))


if __name__ == "__main__":
    main()