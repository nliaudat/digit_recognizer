#!/usr/bin/env python3
"""
repair_int8.py
==============
Re-generates TFLite files from ONNX models in exported_models/TQT folders,
with INT8-only ops (TFLITE_BUILTINS_INT8) for TFLite Micro compatibility.

Usage:
    python repair_int8.py                          # Process all TQT folders
    python repair_int8.py --dry-run                # Show what would be done
    python repair_int8.py --no-backup              # Overwrite without backup
    python repair_int8.py --folder <path>          # Process a single folder
"""

import argparse
import io
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

# Force UTF-8 output on Windows to support emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Disable XNNPACK BEFORE any TF imports
os.environ["TF_ENABLE_XNNPACK"] = "0"

import numpy as np
import onnx2tf

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Input shape constants
INPUT_HEIGHT = 32
INPUT_WIDTH = 20

# TFLite variants to generate (same as tflite_suite_export in quantize_espdl.py)
# NOTE: output_float16_quantized_tflite is NOT supported by this onnx2tf version
# NOTE: We no longer rely on onnx2tf to produce TFLite files directly.
#       Instead, onnx2tf generates a SavedModel, and we convert manually
#       with TFLiteConverter.from_saved_model() to control supported_ops.
VARIANTS = [
    ("_float32",            {"quantize": False}),
    ("_dynamic_range_quant", {"quantize": True, "full_int": False}),
    ("_full_integer_quant",  {"quantize": True, "full_int": True}),
]


def parse_folder_name(folder_name: str) -> dict:
    """
    Parse a TQT folder name to extract metadata.
    
    Examples:
        digit_recognizer_v15_10cls_RGB_TQT_SOFTMAX_0417_1526
        digit_recognizer_v16_10cls_TQT_GRAY_0410_1840
        retrained_v16_TQT
        digit_recognizer_v23_100cls_RGB_TQT_SOFTMAX_0418_0709
    
    Returns dict with: model_base, nb_classes, channels, is_retrained
    """
    info = {
        "model_base": None,
        "nb_classes": 10,
        "channels": 3,  # default RGB
        "is_retrained": False,
    }
    
    # Detect retrained_vXX_TQT pattern
    retrained_match = re.match(r'retrained_(v\d+)_TQT', folder_name)
    if retrained_match:
        info["model_base"] = f"best_model_finetuned"
        info["is_retrained"] = True
        return info
    
    # Extract class count
    cls_match = re.search(r'(\d+)cls', folder_name)
    if cls_match:
        info["nb_classes"] = int(cls_match.group(1))
    
    # Extract color mode
    if "_GRAY" in folder_name or "_GRAY_" in folder_name:
        info["channels"] = 1
    elif "_RGB" in folder_name or "_RGB_" in folder_name:
        info["channels"] = 3
    
    # Extract model architecture name (e.g., digit_recognizer_v15)
    arch_match = re.match(r'(digit_recognizer_v\d+)', folder_name)
    if arch_match:
        info["model_base"] = arch_match.group(1)
    
    return info


def find_base_onnx(folder_path: str, info: dict) -> str | None:
    """
    Find the base (non-quantized) ONNX file in the folder.
    Returns the path or None.
    """
    folder = Path(folder_path)
    
    if info["is_retrained"]:
        # retrained_vXX_TQT uses best_model_finetuned.onnx
        candidates = [
            folder / "best_model_finetuned.onnx",
            folder / "best_model.onnx",
        ]
    else:
        model_base = info["model_base"]
        if model_base:
            # Prefer the base ONNX (without _quantized suffix)
            candidates = [
                folder / f"{model_base}.onnx",
            ]
            # Also check for any .onnx that doesn't contain "_quantized"
            for f in sorted(folder.glob("*.onnx")):
                if "_quantized" not in f.name:
                    candidates.append(f)
        else:
            # Fallback: any non-quantized .onnx
            candidates = sorted(folder.glob("*.onnx"))
            candidates = [c for c in candidates if "_quantized" not in c.name]
    
    for c in candidates:
        if c.exists():
            return str(c)
    
    return None


def find_existing_tflite_files(folder_path: str, info: dict) -> list[tuple[str, str]]:
    """
    Find existing TFLite files that match the variant pattern.
    Returns list of (variant_suffix, file_path).
    """
    folder = Path(folder_path)
    existing = []
    
    # Determine the base name pattern
    if info["is_retrained"]:
        base_pattern = "best_model_finetuned_quantized"
    else:
        base_pattern = f"{info['model_base']}_quantized" if info["model_base"] else ""
    
    if base_pattern:
        for suffix, _ in VARIANTS:
            pattern = f"{base_pattern}{suffix}.tflite"
            fpath = folder / pattern
            if fpath.exists():
                existing.append((suffix, str(fpath)))
    
    return existing


def generate_calibration_data(channels: int, num_samples: int = 100) -> np.ndarray:
    """
    Generate dummy calibration data for onnx2tf.
    Shape: (num_samples, INPUT_HEIGHT, INPUT_WIDTH, channels) in NHWC format.
    """
    data = np.random.uniform(0.0, 1.0, size=(num_samples, INPUT_HEIGHT, INPUT_WIDTH, channels))
    return data.astype(np.float32)


def _find_saved_model(output_dir: str) -> str | None:
    """Find the SavedModel folder generated by onnx2tf."""
    saved_model_dir = os.path.join(output_dir, "saved_model")
    if os.path.isdir(saved_model_dir):
        return saved_model_dir
    # Fallback: look for any saved_model.pb
    for root, dirs, files in os.walk(output_dir):
        if "saved_model.pb" in files:
            return root
    return None


def _validate_no_delegates(tflite_model, model_name="model"):
    """
    Validate that the TFLite model has no XNNPACK delegates baked in.
    
    Uses binary search for 'DELEGATE' in the flatbuffer, which is the
    most reliable method. The Interpreter-based check with
    BUILTIN_WITHOUT_DEFAULT_DELEGATES can give false positives.
    """
    if b'DELEGATE' in tflite_model:
        raise RuntimeError(
            f"❌ [{model_name}] DELEGATE op found in flatbuffer! "
            f"XNNPACK delegate was baked into the model."
        )
    print(f"      ✅ [{model_name}] No delegates — TFLite Micro compatible")


def _convert_savedmodel_to_tflite(
    saved_model_path: str,
    target_path: str,
    quantize: bool,
    full_int: bool,
    channels: int,
    variant_name: str,
) -> bool:
    """
    Convert a SavedModel to TFLite with full control over supported_ops.
    
    This is the KEY fix: onnx2tf bakes XNNPACK delegates into its TFLite output,
    but we bypass that by using TFLiteConverter.from_saved_model() directly,
    where we can set supported_ops ourselves.
    
    For quantized variants (dynamic_range_quant, full_integer_quant):
        Use TFLITE_BUILTINS_INT8 — forces INT8-only ops, prevents XNNPACK delegates.
    For float32 variant:
        Use TFLITE_BUILTINS — prevents XNNPACK delegates while keeping float ops.
    """
    import tensorflow as tf
    
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Provide calibration data for full integer quantization
            calib_data = generate_calibration_data(channels, num_samples=100)
            
            def representative_dataset():
                for i in range(len(calib_data)):
                    yield [calib_data[i:i+1]]
            
            converter.representative_dataset = representative_dataset
            
            if full_int:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        
        # CORRECT FLOW: Set supported_ops LAST, right before convert()
        # This is the critical fix — prevents XNNPACK delegates from being baked in.
        # For quantized models, use TFLITE_BUILTINS_INT8 (forces INT8-only).
        # For float32, use TFLITE_BUILTINS (keeps float ops but still prevents delegates).
        if quantize:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # Assert it's set correctly
        expected = tf.lite.OpsSet.TFLITE_BUILTINS_INT8 if quantize else tf.lite.OpsSet.TFLITE_BUILTINS
        assert converter.target_spec.supported_ops == [expected], \
            f"Expected {expected}, got {converter.target_spec.supported_ops}"
        
        tflite_model = converter.convert()
        
        # Validate no delegates baked in
        _validate_no_delegates(tflite_model, os.path.basename(target_path))
        
        # Save
        with open(target_path, "wb") as f:
            f.write(tflite_model)
        
        size_kb = os.path.getsize(target_path) / 1024
        print(f"      ✅ Saved: {os.path.basename(target_path)} ({size_kb:.1f} KB)")
        return True
        
    except Exception as e:
        print(f"      ❌ Failed to convert {variant_name}: {e}")
        return False


def convert_with_onnx2tf(onnx_path: str, output_dir: str, channels: int) -> str | None:
    """
    Run onnx2tf.convert to produce a SavedModel (NOT a TFLite file).
    
    Returns the path to the generated SavedModel, or None on failure.
    
    NOTE: We explicitly do NOT ask onnx2tf to produce TFLite output.
    Instead, we use it only for ONNX→SavedModel conversion, then
    manually convert with TFLiteConverter.from_saved_model() where
    we control supported_ops.
    """
    calib_data = generate_calibration_data(channels)
    calib_npy_path = os.path.join(output_dir, "_repair_calib_temp.npy")
    
    try:
        np.save(calib_npy_path, calib_data)
        
        # Run onnx2tf WITHOUT any TFLite output flags — just produce SavedModel
        # NOTE: We explicitly avoid passing TFLite-related flags because different
        # onnx2tf versions have different parameter names. Since we only need the
        # SavedModel (which onnx2tf always produces), we skip all TFLite flags.
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            custom_input_op_name_np_data_path=[["input", calib_npy_path, [0, 0, 0], [1, 1, 1]]],
            not_use_onnxsim=True,
            non_verbose=True,
            overwrite_input_shape=["input", 1, channels, INPUT_HEIGHT, INPUT_WIDTH],
        )
        
        # Find the SavedModel
        saved_model_path = _find_saved_model(output_dir)
        if saved_model_path is None:
            print(f"      ❌ onnx2tf did not produce a SavedModel")
            return None
        
        return saved_model_path
        
    except Exception as e:
        print(f"      ❌ onnx2tf failed: {e}")
        return None
    finally:
        if os.path.exists(calib_npy_path):
            os.remove(calib_npy_path)


def repair_tqt_folder(folder_path: str, dry_run: bool = False, no_backup: bool = False) -> bool:
    """
    Repair all TFLite files in a single TQT folder.
    Returns True if at least one TFLite was regenerated.
    
    Strategy:
        1. Run onnx2tf ONCE to produce a SavedModel
        2. Convert the SavedModel to 3 TFLite variants using
           TFLiteConverter.from_saved_model() with proper supported_ops
    """
    folder_name = os.path.basename(folder_path)
    info = parse_folder_name(folder_name)
    
    print(f"\n{'='*70}")
    print(f"📁 Folder: {folder_name}")
    print(f"   Model: {info['model_base'] or 'unknown'}, "
          f"Classes: {info['nb_classes']}, "
          f"Channels: {info['channels']} ({'GRAY' if info['channels'] == 1 else 'RGB'})")
    
    # Find base ONNX
    onnx_path = find_base_onnx(folder_path, info)
    if not onnx_path:
        print(f"   ⚠️  No base ONNX file found, skipping")
        return False
    
    print(f"   📄 ONNX: {os.path.basename(onnx_path)}")
    
    # Determine target filenames
    if info["is_retrained"]:
        base_name = "best_model_finetuned_quantized"
    else:
        base_name = f"{info['model_base']}_quantized"
    
    if dry_run:
        for suffix, _ in VARIANTS:
            target_name = f"{base_name}{suffix}.tflite"
            print(f"   Would convert -> {target_name}")
        return True
    
    # Step 1: Run onnx2tf ONCE to produce SavedModel
    print(f"\n   🚀 Running onnx2tf (ONNX → SavedModel)...")
    saved_model_path = convert_with_onnx2tf(onnx_path, folder_path, info["channels"])
    if saved_model_path is None:
        print(f"   ❌ onnx2tf failed, skipping folder")
        return False
    
    print(f"      ✅ SavedModel at: {saved_model_path}")
    
    # Step 2: Convert SavedModel to 3 TFLite variants
    success_count = 0
    
    for suffix, flags in VARIANTS:
        target_name = f"{base_name}{suffix}.tflite"
        target_path = os.path.join(folder_path, target_name)
        
        print(f"\n   🔄 Variant: {suffix}")
        
        # Backup existing file
        if os.path.exists(target_path) and not no_backup:
            backup_path = target_path + ".bak"
            if not os.path.exists(backup_path):
                shutil.move(target_path, backup_path)
                print(f"      💾 Backed up existing -> {os.path.basename(backup_path)}")
            else:
                os.remove(target_path)
                print(f"      🗑️  Removed existing (backup already present)")
        elif os.path.exists(target_path) and no_backup:
            os.remove(target_path)
            print(f"      🗑️  Removed existing (no backup)")
        
        # Convert SavedModel → TFLite with proper supported_ops
        if _convert_savedmodel_to_tflite(
            saved_model_path=saved_model_path,
            target_path=target_path,
            quantize=flags.get("quantize", False),
            full_int=flags.get("full_int", False),
            channels=info["channels"],
            variant_name=suffix,
        ):
            success_count += 1
    
    # Clean up onnx2tf artifacts (SavedModel, tf_layers, etc.)
    for folder_name in ["saved_model", "tf_layers", "variables"]:
        p = os.path.join(folder_path, folder_name)
        if os.path.exists(p):
            shutil.rmtree(p)
    
    # Clean up saved_model.pb (onnx2tf creates this instead of saved_model/ folder)
    saved_model_pb = os.path.join(folder_path, "saved_model.pb")
    if os.path.exists(saved_model_pb):
        os.remove(saved_model_pb)
    
    # Clean up fingerprint.pb (onnx2tf artifact)
    fingerprint_pb = os.path.join(folder_path, "fingerprint.pb")
    if os.path.exists(fingerprint_pb):
        os.remove(fingerprint_pb)
    
    # Clean up any stray onnx2tf temp files
    for f in os.listdir(folder_path):
        if f.endswith(".tflite") and f not in [f"{base_name}{s}.tflite" for s, _ in VARIANTS]:
            fpath = os.path.join(folder_path, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
    
    return success_count > 0


def find_tqt_folders(base_dir: str) -> list[str]:
    """Find all subdirectories containing 'TQT' in their name, searching recursively."""
    tqt_folders = []
    base = Path(base_dir)
    if not base.exists():
        print(f"❌ Directory not found: {base_dir}")
        return tqt_folders
    
    # Search recursively through all subdirectories
    for entry in base.rglob("*"):
        if entry.is_dir() and "TQT" in entry.name:
            tqt_folders.append(str(entry))
    
    return sorted(tqt_folders)


def main():
    parser = argparse.ArgumentParser(
        description="Re-generate TFLite files from TQT ONNX models with XNNPACK disabled"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("--no-backup", action="store_true",
                        help="Overwrite existing TFLite files without creating .bak backups")
    parser.add_argument("--folder", type=str, default=None,
                        help="Process a single TQT folder instead of all")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔧 TFLite XNNPACK Repair Tool")
    print("=" * 70)
    print(f"   XNNPACK: DISABLED (TF_ENABLE_XNNPACK=0)")
    print(f"   Dry run: {'YES' if args.dry_run else 'NO'}")
    print(f"   Backup:  {'NO' if args.no_backup else 'YES (.bak)'}")
    print()
    
    if args.folder:
        # Process a single folder
        if not os.path.isdir(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            sys.exit(1)
        folders = [args.folder]
    else:
        # Find all TQT folders in exported_models
        exported_dir = os.path.join(PROJECT_ROOT, "exported_models")
        folders = find_tqt_folders(exported_dir)
        
        if not folders:
            print(f"❌ No TQT folders found in {exported_dir}")
            sys.exit(1)
        
        print(f"📋 Found {len(folders)} TQT folders to process:")
        for f in folders:
            print(f"   - {os.path.basename(f)}")
    
    total_success = 0
    total_failed = 0
    
    for folder_path in folders:
        try:
            if repair_tqt_folder(folder_path, dry_run=args.dry_run, no_backup=args.no_backup):
                total_success += 1
            else:
                total_failed += 1
        except Exception as e:
            print(f"\n   ❌ Error processing {os.path.basename(folder_path)}: {e}")
            traceback.print_exc()
            total_failed += 1
    
    print(f"\n{'='*70}")
    if args.dry_run:
        print(f"✅ Dry run complete. {total_success} folders would be processed.")
    else:
        print(f"✅ Done. Processed {total_success} folders successfully, {total_failed} failed.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
