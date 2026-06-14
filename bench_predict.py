"""
bench_predict.py
================
CLI entry point for benchmarking TFLite digit recognition models.

All logic lives in the `benchmark/` package. This script provides
the command-line interface and environment setup only.

Usage examples:
    py bench_predict.py --list
    py bench_predict.py --model digit_recognizer_v15.tflite
    py bench_predict.py --test_all
    py bench_predict.py --espdl some_model.espdl
"""

import argparse
import logging
import os
import sys

# ── Environment Setup ──────────────────────────────────────────────────────
# Pre-parse classes and color mode to set env-vars BEFORE config is imported.
# (This avoids the interactive prompt in config.parameters.py when run from CLI)
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--classes', type=int)
_pre_parser.add_argument('--color', type=str)
_pre_args, _ = _pre_parser.parse_known_args()

if _pre_args.classes:
    os.environ['DIGIT_NB_CLASSES'] = str(_pre_args.classes)
if _pre_args.color:
    if _pre_args.color.lower() == 'gray':
        os.environ['DIGIT_INPUT_CHANNELS'] = '1'
    elif _pre_args.color.lower() == 'rgb':
        os.environ['DIGIT_INPUT_CHANNELS'] = '3'

# ── Project & Benchmark Imports ──────────────────────────────────────────
import config as params
from config.validation import validate_full_config

from benchmark import (
    find_model_path,
    get_all_models,
    test_single_model,
    test_all_models,
)

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(table_data, headers=None, tablefmt=None, stralign=None):
        if not table_data: return ""
        if not headers: headers = [f"Col{i}" for i in range(len(table_data[0]))]
        cols = list(zip(*([headers] + table_data)))
        col_widths = [max(len(str(x)) for x in col) for col in cols]
        lines = []
        header_line = " | ".join(f"{str(h):{w}}" for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        for row in table_data:
            lines.append(" | ".join(f"{str(val):{w}}" for val, w in zip(row, col_widths)))
        return "\n".join(lines)


# ── Local helpers (thin wrappers) ────────────────────────────────────────

def list_available_models(quantized_only=False, subfolder=None, input_dir=None,
                          exclude_model=None, model_list=None, iot_compat=True):
    """List all available models in a table format and exit."""
    models = get_all_models(quantized_only=quantized_only, subfolder=subfolder,
                            input_dir=input_dir, exclude_model=exclude_model,
                            model_list=model_list, iot_compat=iot_compat)
    if not models:
        print("No models found.")
        return

    headers = ['Directory', 'Model', 'Type', 'Params', 'Size (KB)']
    table_data = []
    for m in models:
        params_count = m['parameters']
        if params_count >= 1_000_000:
            params_str = f"{params_count/1_000_000:.1f}M"
        elif params_count >= 1_000:
            params_str = f"{params_count/1_000:.1f}K"
        else:
            params_str = str(params_count)
        table_data.append([m['directory'], m['name'], m['type'], params_str, f"{m['size_kb']:.1f}"])

    print("\nAvailable models found:")
    print(tabulate(table_data, headers=headers, tablefmt='simple_grid', stralign='right'))


def inspect_espdl(espdl_path: str) -> dict:
    """Inspect an .espdl file (size, header, quantization metadata)."""
    if not os.path.exists(espdl_path):
        print(f"espdl file not found: {espdl_path}")
        return {}

    size_bytes = os.path.getsize(espdl_path)
    size_kb = size_bytes / 1024

    print("\n" + "=" * 60)
    print("  ESPDL MODEL METADATA")
    print("=" * 60)
    print(f"  File  : {espdl_path}")
    print(f"  Size  : {size_kb:.1f} KB  ({size_bytes:,} bytes)")
    print()
    print("  Quantization: TQT-optimized Power-of-2 INT8 (Per-Tensor, Symmetric)")
    print("  NOTE: This .espdl uses TQT-learned scales -- NOT standard PTQ.")
    print("        Do not compare accuracy with the onnx2tf .tflite directly.")
    print()

    try:
        with open(espdl_path, "rb") as f:
            header = f.read(16).hex()
        print(f"  Header (hex): {header}")
    except (IOError, OSError) as e:
        print(f"  [!] Could not read header: {e}")

    return {"path": espdl_path, "size_bytes": size_bytes, "size_kb": size_kb}


# ── Main CLI ─────────────────────────────────────────────────────────────

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Digit Recognition Benchmarking System')

    # Optional overrides for dataset configuration (auto-detected if omitted)
    parser.add_argument('--classes', type=int, choices=[10, 100],
                        help='Force the number of classes (10 or 100). Auto-detected from folder name if omitted.')
    parser.add_argument('--color', type=str, choices=['rgb', 'gray'],
                        help='Force a specific color mode. Auto-detected from folder name if omitted.')

    # Mode selection
    parser.add_argument('--test_all', action='store_true',
                        help='Perform a full benchmark of all available models in the input directory.')
    parser.add_argument('--model', type=str,
                        help='Test a single specific model by its filename (e.g., digit_recognizer_v15.tflite).')
    parser.add_argument('--model_list', type=str, nargs='+',
                        help='Compare a specific subset of models. Provide one or more model names OR directory names.')
    parser.add_argument('--list', action='store_true',
                        help='List all compatible models found and exit without benchmarking.')

    # Filtering and Path Configuration
    parser.add_argument('--input_dir', type=str, default='exported_models',
                        help='Base directory to search for models (default: exported_models)')
    parser.add_argument('--subfolder', type=str,
                        help='Restrict search to a specific subfolder within the input directory.')
    parser.add_argument('--exclude_model', '--exclude_models',
                        dest='exclude_model', type=str, nargs='+', default=None,
                        help='Exclude models containing these strings from the benchmark.')
    parser.add_argument('--quantized', action='store_true', default=True,
                        help='Only include quantized models (True by default).')
    parser.add_argument('--no-quantized', action='store_false', dest='quantized',
                        help='Include all models, including floating-point versions.')
    parser.add_argument('--iot-compat', action='store_true', default=True,
                        help='Filter models for IoT compatibility: exclude float32 and dynamic_range models.')
    parser.add_argument('--no-iot-compat', action='store_false', dest='iot_compat',
                        help='Include all models regardless of IoT compatibility.')

    # Dataset and Testing Configuration
    parser.add_argument('--test_images', type=int, default=0,
                        help='Number of images to test per model. 0 means use the entire dataset.')
    parser.add_argument('--all_datasets', action='store_true', default=True,
                        help='Use images from all available data sources (True by default).')
    parser.add_argument('--no-all_datasets', action='store_false', dest='all_datasets',
                        help='Restrict testing to the standard test set only.')

    # Output and Debugging
    parser.add_argument('--list-failed', action='store_true',
                        help='Generate a detailed CSV file with information on all misclassifications.')
    parser.add_argument('--save-failed', action='store_true',
                        help='Save misclassified images into a "failed-predictions" folder.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable verbose output for debugging model predictions and data loading.')
    parser.add_argument('--tolerance', type=float, default=0.1,
                        help='Acceptable error tolerance in decimal scale (default: 0.1).')
    parser.add_argument('--espdl', type=str, default=None,
                        help='Path to a .espdl file to inspect (size, header, quantization metadata).')
    parser.add_argument('--new', type=str,
                        help='Test a new model and update the existing CSV results.')
    parser.add_argument('--simulate-esp32', action='store_true', default=False,
                        help='Simulate ESP32 inference by adding quantization noise (default: False).')
    parser.add_argument('--no-simulate-esp32', action='store_false', dest='simulate_esp32',
                        help='Disable ESP32 simulation (faster benchmark, PC-only accuracy).')

    args, unknown = parser.parse_known_args()

    # Handle --espdl inspection mode
    if args.espdl:
        inspect_espdl(args.espdl)
        return

    # Resolve input directory
    if args.input_dir == 'exported_models':
        args.input_dir = params.OUTPUT_DIR

    # Handle --list
    if args.list:
        list_available_models(quantized_only=args.quantized, subfolder=args.subfolder,
                              input_dir=args.input_dir, exclude_model=args.exclude_model,
                              model_list=args.model_list, iot_compat=args.iot_compat)
        return

    # Handle --new model
    if args.new:
        print(f"🚀 Benchmarking new model and updating CSV: {args.new}")
        test_all_models(
            num_test_images=args.test_images, quantized_only=args.quantized,
            debug=args.debug, use_all_datasets=args.all_datasets,
            list_failed=args.list_failed, save_failed=args.save_failed,
            subfolder=args.subfolder, input_dir=args.input_dir,
            exclude_model=args.exclude_model,
            override_classes=args.classes, override_color=args.color,
            model_list=[args.new], tolerance=args.tolerance,
            update_csv=True, iot_compat=args.iot_compat,
            simulate_esp32=args.simulate_esp32,
        )
        return

    # Handle single model prediction
    if args.model:
        if os.path.isfile(args.model) and args.model.endswith('.tflite'):
            model_path = args.model
        else:
            model_path = find_model_path(args.model, input_dir=args.input_dir)

        if model_path is None:
            print(f"❌ Model '{args.model}' not found in {args.input_dir} (or as an exact path)!")
            print("Available models:")
            list_available_models(quantized_only=args.quantized, subfolder=args.subfolder,
                                  input_dir=args.input_dir, exclude_model=args.exclude_model)
            return

        test_single_model(
            model_path=model_path, num_test_images=args.test_images,
            debug=args.debug, use_all_datasets=args.all_datasets,
            list_failed=args.list_failed, save_failed=args.save_failed,
            output_dir=args.input_dir,
            override_classes=args.classes, override_color=args.color,
            tolerance=args.tolerance,
        )
        return

    # Handle --test_all or --model_list
    elif args.test_all or args.model_list:
        test_all_models(
            num_test_images=args.test_images, quantized_only=args.quantized,
            debug=args.debug, use_all_datasets=args.all_datasets,
            list_failed=args.list_failed, save_failed=args.save_failed,
            subfolder=args.subfolder, input_dir=args.input_dir,
            exclude_model=args.exclude_model,
            override_classes=args.classes, override_color=args.color,
            model_list=args.model_list, tolerance=args.tolerance,
            iot_compat=args.iot_compat, simulate_esp32=args.simulate_esp32,
        )
        return

    # Default behavior
    if args.debug:
        print("Debug mode requires either --model or --test_all")
        return

    print("No specific mode selected. Running default benchmark...")
    print("Use --list to see available models, --model to test specific model, or --test_all for full benchmark.")
    print("-" * 60)

    test_all_models(
        quantized_only=args.quantized, num_test_images=args.test_images,
        debug=args.debug, use_all_datasets=args.all_datasets,
        list_failed=args.list_failed, save_failed=args.save_failed,
        subfolder=args.subfolder, input_dir=args.input_dir,
        override_classes=args.classes, override_color=args.color,
        model_list=args.model_list, tolerance=args.tolerance,
        iot_compat=args.iot_compat, simulate_esp32=args.simulate_esp32,
    )


if __name__ == "__main__":
    validate_full_config()
    main()