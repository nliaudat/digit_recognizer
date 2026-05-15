#!/usr/bin/env python3
"""
utils/convert_teacher_checkpoints.py
────────────────────────────────────
Extract the weights H5 file from a teacher .keras checkpoint **without**
loading the model into TensorFlow.

A .keras file is a zip archive containing ``model.weights.h5`` (the weights
in standard HDF5 format).  By extracting with pure Python ``zipfile`` we
avoid two problems:

1. ``free(): invalid pointer`` — C++ memory corruption when loading multiple
   models with complex custom layers (v29's AdaptiveHybridBinarization, v28's
   AdaptiveMeanBinarization) in the same CUDA context.
2. ``ModuleNotFoundError: tf_keras.src.models.functional`` — TF version
   mismatch between Keras v3 (save format) and tf_keras (load format).

The architecture is reconstructed from Python source code in the main process
via ``create_model_by_name()``, then weights are loaded with ``load_weights()``.

Usage (called by train_distill_helper.py / retrain_with_teacher.py):
    python utils/convert_teacher_checkpoints.py <model_name> <checkpoint_path> <output_dir>

Output:
    <output_dir>/<model_name>.weights.h5
"""

import os
import shutil
import sys
import zipfile


def main():
    if len(sys.argv) < 4:
        print("Usage: convert_teacher_checkpoints.py <model_name> <checkpoint_path> <output_dir>",
              file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_dir = sys.argv[3]

    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # ── Extract weights from the .keras zip archive ──────────────────────
    # A .keras file is a zip containing:
    #   - model.weights.h5   (the weights in HDF5 format)
    #   - model.keras.json   (the architecture config)
    #   - metadata.json      (training metadata)
    # We only need model.weights.h5 — no TF loading required.
    weights_path = os.path.join(output_dir, f"{model_name}.weights.h5")

    try:
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            # Keras v3 stores weights at the root of the zip
            if "model.weights.h5" in zf.namelist():
                with zf.open("model.weights.h5") as src, open(weights_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            # Keras v2 stores weights in a subdirectory
            elif "model/model.weights.h5" in zf.namelist():
                with zf.open("model/model.weights.h5") as src, open(weights_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                print(
                    f"ERROR: No model.weights.h5 found in {checkpoint_path}. "
                    f"Contents: {zf.namelist()}",
                    file=sys.stderr,
                )
                sys.exit(1)
    except zipfile.BadZipFile:
        print(f"ERROR: {checkpoint_path} is not a valid zip file.", file=sys.stderr)
        sys.exit(1)

    print(weights_path)  # ← stdout: the caller captures this


if __name__ == "__main__":
    main()
