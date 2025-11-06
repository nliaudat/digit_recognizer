#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
debug.py – Multi‑sample sanity‑check for a quantised TFLite digit recogniser.

Features
--------
* Auto‑detect the latest quantised *.tflite* model (or specify one with --model).
* Choose how many random training images to test (default 1) via --num-samples.
* Optional random seed for reproducibility (--seed).
* Option to skip the inference‑time preprocessing step (--no-preprocess).
* Correct de‑quantisation of uint8 outputs (no extra soft‑max).
* Prints the exact model filename that is used for inference.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------
# Make sure the repository root is on the import path
# ----------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# ----------------------------------------------------------------------
# Project‑specific imports (the ones you confirmed)
# ----------------------------------------------------------------------
from predict import load_random_image_from_dataset
from utils.preprocess import preprocess_for_inference
from utils import get_data_splits
import parameters as params


# ----------------------------------------------------------------------
# Helper: find the newest quantised .tflite file (ignore *_float.tflite*)
# ----------------------------------------------------------------------
def find_latest_quantised_model(export_dir: Path) -> Path | None:
    if not export_dir.is_dir():
        return None

    candidates = sorted(
        export_dir.rglob("*.tflite"),
        key=os.path.getmtime,
        reverse=True,
    )
    for cand in candidates:
        if not cand.name.lower().endswith("_float.tflite"):
            return cand
    return None


# ----------------------------------------------------------------------
# Load a TFLite interpreter and allocate tensors
# ----------------------------------------------------------------------
def load_interpreter(model_path: Path) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


# ----------------------------------------------------------------------
# Pull input / output tensor descriptors from the interpreter
# ----------------------------------------------------------------------
def get_io_details(interpreter: tf.lite.Interpreter):
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return input_detail, output_detail


# ----------------------------------------------------------------------
# Run a single inference, **correctly** de‑quantising the output
# ----------------------------------------------------------------------
def run_inference(interpreter: tf.lite.Interpreter,
                  input_detail,
                  output_detail,
                  image: np.ndarray) -> np.ndarray:
    """
    Feed `image` (already pre‑processed) to the interpreter and return a
    **float32 probability vector**.

    * Handles both quantised (uint8/int8) and float models.
    * Does **not** apply a second soft‑max to quantised outputs – they are
      already soft‑max‑scaled before quantisation.
    """
    # --------------------------------------------------------------
    # 1️⃣  Add batch dimension if missing
    # --------------------------------------------------------------
    if image.ndim == len(input_detail["shape"]) - 1:
        image = np.expand_dims(image, axis=0)

    # --------------------------------------------------------------
    # 2️⃣  Cast to the dtype expected by the model
    # --------------------------------------------------------------
    if image.dtype != input_detail["dtype"]:
        image = image.astype(input_detail["dtype"])

    # --------------------------------------------------------------
    # 3️⃣  Run the interpreter
    # --------------------------------------------------------------
    interpreter.set_tensor(input_detail["index"], image)
    interpreter.invoke()

    # --------------------------------------------------------------
    # 4️⃣  Retrieve the raw output (still uint8/int8 if model is quantised)
    # --------------------------------------------------------------
    raw_out = interpreter.get_tensor(output_detail["index"]).squeeze()

    # --------------------------------------------------------------
    # 5️⃣  De‑quantise if necessary
    # --------------------------------------------------------------
    if output_detail["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_detail["quantization"]
        # Convert to float32 in the original range (0‑1 for a soft‑max)
        probs = (raw_out.astype(np.float32) - zero_point) * scale

        # The model already emitted a soft‑max‑scaled distribution,
        # only quantised to 0‑255.  Normalise so the sum is exactly 1.
        # (Dividing by the sum is equivalent to dividing by 255 because
        # the quantiser chose a scale that maps 0‑1 → 0‑255.)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
    else:
        # Float model – already probabilities
        probs = raw_out.astype(np.float32)

    return probs


# ----------------------------------------------------------------------
# Main sanity‑check driver
# ----------------------------------------------------------------------
def sanity_check(model_path: Path,
                 num_samples: int = 1,
                 seed: int | None = None,
                 skip_preprocess: bool = False) -> None:
    """
    Run the sanity‑check on `num_samples` random training images.

    Prints per‑sample stats and a final accuracy summary.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # --------------------------------------------------------------
    # Load the training split once
    # --------------------------------------------------------------
    (x_train_raw, y_train_raw), _, _ = get_data_splits()
    n_total = len(x_train_raw)

    if num_samples > n_total:
        raise ValueError(
            f"Requested {num_samples} samples but only {n_total} training images exist."
        )

    # --------------------------------------------------------------
    # Load the TFLite model
    # --------------------------------------------------------------
    interpreter = load_interpreter(model_path)
    inp_detail, out_detail = get_io_details(interpreter)

    print("\n=== MODEL INFORMATION ===")
    print(f"Model file : {model_path}")
    print(f"Input  : dtype={inp_detail['dtype']}, shape={inp_detail['shape']}")
    print(f"Output : dtype={out_detail['dtype']}, shape={out_detail['shape']}")

    # --------------------------------------------------------------
    # Run inference on the requested number of random samples
    # --------------------------------------------------------------
    correct = 0
    for i in range(num_samples):
        idx = random.randrange(n_total)
        raw_img = x_train_raw[idx]
        true_label = int(y_train_raw[idx])

        print(f"\n--- Sample {i + 1}/{num_samples} (index {idx}) ---")
        print("Raw image:")
        print(f" dtype : {raw_img.dtype}")
        print(f" shape : {raw_img.shape}")
        print(f" range : [{raw_img.min():.3f}, {raw_img.max():.3f}]")

        # ----------------------------------------------------------
        # Preprocess for inference (unless user asked to skip)
        # ----------------------------------------------------------
        if skip_preprocess:
            proc_img = raw_img
            print("\n⚠️  Skipping preprocessing as requested.")
        else:
            proc_img = preprocess_for_inference([raw_img])[0]
            print("\nAfter preprocess_for_inference:")
            print(f" dtype : {proc_img.dtype}")
            print(f" shape : {proc_img.shape}")
            print(f" range : [{proc_img.min():.3f}, {proc_img.max():.3f}]")

        # ----------------------------------------------------------
        # Run the model
        # ----------------------------------------------------------
        probs = run_inference(interpreter, inp_detail, out_detail, proc_img)
        pred_class = int(np.argmax(probs))

        # ----------------------------------------------------------
        # Show results
        # ----------------------------------------------------------
        print("\nInference result:")
        print(f" Probabilities (len={len(probs)}):")
        print("  " + " ".join(f"{p:0.4f}" for p in probs))
        print(f" Predicted class : {pred_class}")
        print(f" Ground‑truth    : {true_label}")

        if pred_class == true_label:
            correct += 1
            print(" ✅ Match")
        else:
            print(" ❌ Mismatch")

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f" Tested {num_samples} sample(s) from the training set")
    print(f" Correct predictions : {correct}/{num_samples} "
          f"({correct / num_samples * 100:.2f}%)")
    print(f" Model file used      : {model_path}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi‑sample sanity‑check for a quantised TFLite digit recogniser."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to a .tflite model. If omitted, the script picks the newest "
             "quantised model under exported_models/.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="How many random training images to test (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible image selection.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip the preprocess_for_inference step (useful for debugging).",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Resolve the model path
    # --------------------------------------------------------------
    if args.model:
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
    else:
        export_root = Path("exported_models")
        model_path = find_latest_quantised_model(export_root)
        if model_path is None:
            raise FileNotFoundError(
                "No quantised .tflite model found under ./exported_models/. "
                "Run training first or specify --model."
            )
        print(f"🔍 Auto‑selected latest model: {model_path}")

    # --------------------------------------------------------------
    # Run the sanity check
    # --------------------------------------------------------------
    sanity_check(
        model_path=model_path,
        num_samples=args.num_samples,
        seed=args.seed,
        skip_preprocess=args.no_preprocess,
    )


if __name__ == "__main__":
    main()