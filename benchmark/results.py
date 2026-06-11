"""
benchmark/results.py
====================
Result analysis, confusion matrices, CSV generation, and summary tables
for TFLite model benchmarking.

Extracted from bench_predict.py.
"""

import csv
import logging
import os
import random
import shutil
import time
from collections import Counter
from datetime import datetime

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config as params
from .predictor import TFLiteDigitPredictor, get_model_metadata, get_model_parameters_count
from .data import load_test_dataset_with_labels

logger = logging.getLogger(__name__)

# Try tabulate with fallback
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(table_data, headers=None, tablefmt=None, stralign=None):
        if not table_data:
            return ""
        if not headers:
            headers = [f"Col{i}" for i in range(len(table_data[0]))]
        cols = list(zip(*([headers] + table_data)))
        col_widths = [max(len(str(x)) for x in col) for col in cols]
        lines = []
        header_line = " | ".join(f"{str(h):{w}}" for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        for row in table_data:
            lines.append(" | ".join(f"{str(val):{w}}" for val, w in zip(row, col_widths)))
        return "\n".join(lines)


# ── Model Evaluation ──────────────────────────────────────────────────────

def test_model_on_dataset(model_path, num_test_images=0, debug=False,
                          use_all_datasets=True, collect_failed=False,
                          model_name=None, tolerance=0.1,
                          simulate_esp32=True):
    """Test a model on random images from dataset and return accuracy and performance metrics.

    Args:
        simulate_esp32: If True, also run ESP32-simulated inference and return its accuracy.
    Returns:
        (accuracy, total_tested, avg_inference_time, inferences_per_second,
         failed_predictions, all_predictions_lite, esp32_accuracy, accuracy_real_only)
    """
    try:
        predictor = TFLiteDigitPredictor(model_path)
    except Exception as e:
        logger.error(f"Skipping model {model_path}: {e}")
        return 0.0, 0, 0.0, 0.0, [], [], 0.0

    correct_predictions = 0
    total_tested = 0
    total_inference_time = 0.0
    failed_predictions = []
    all_predictions_lite = []

    # Real-only vs full-set counters
    correct_real = 0
    total_real = 0

    # ESP32 simulation counters
    esp32_correct = 0
    esp32_predictions_lite = []

    # Load test data
    test_data = load_test_dataset_with_labels(num_test_images, use_all_datasets)

    if not test_data:
        logger.error("No test data available")
        return 0.0, 0, 0.0, 0.0, [], [], 0.0, 0.0, 0.0

    # Warm-up run
    if len(test_data) > 0:
        warmup_image, _, _, _ = test_data[0]
        if warmup_image is not None:
            try:
                predictor.predict(warmup_image, debug=False)
            except:
                pass

    # Test with progress bar
    if debug:
        test_iterator = test_data
        print(f"Testing {len(test_data)} images in debug mode...")
    else:
        test_iterator = tqdm(test_data, desc=f"Testing {os.path.basename(model_path)}", leave=False)

    for item in test_iterator:
        image, true_label, original_fname, is_augmented = item
        if image is None:
            continue

        try:
            start_time = time.perf_counter()
            prediction, confidence, _ = predictor.predict(image, debug=debug)
            end_time = time.perf_counter()

            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time

            model_scale = predictor.num_classes / 10.0
            pred_digit = float(prediction) / model_scale
            true_digit = float(true_label) / model_scale

            diff = abs(true_digit - pred_digit) % 10.0
            circular_diff = min(diff, 10.0 - diff)

            if circular_diff <= tolerance:
                correct_predictions += 1
                if not is_augmented:
                    correct_real += 1
                if debug:
                    print(f"✓ Correct: {pred_digit:.1f} (true: {true_digit:.1f}, confidence: {confidence:.3f})")
            else:
                if debug:
                    print(f"✗ Wrong: {prediction} (true: {true_label}, confidence: {confidence:.3f})")
                failed_predictions.append({
                    'image': image,
                    'true_label': true_label,
                    'predicted_label': prediction,
                    'confidence': confidence,
                    'model': model_name or os.path.basename(model_path),
                    'image_source': 'dataset',
                    'original_fname': original_fname,
                    'num_classes': predictor.num_classes,
                })

            if not is_augmented:
                total_real += 1

            all_predictions_lite.append({
                'true_label': true_label,
                'predicted_label': prediction,
                'model': model_name or os.path.basename(model_path),
                'num_classes': predictor.num_classes,
                'tolerance': tolerance,
            })

            # ESP32 simulation pass
            if simulate_esp32:
                esp32_prediction, esp32_confidence, _ = predictor.predict_esp32(image, debug=debug)
                esp32_pred_digit = float(esp32_prediction) / model_scale
                esp32_diff = abs(true_digit - esp32_pred_digit) % 10.0
                esp32_circular_diff = min(esp32_diff, 10.0 - esp32_diff)

                if esp32_circular_diff <= tolerance:
                    esp32_correct += 1

                esp32_predictions_lite.append({
                    'true_label': true_label,
                    'predicted_label': esp32_prediction,
                    'model': (model_name or os.path.basename(model_path)) + "_esp32",
                    'num_classes': predictor.num_classes,
                    'tolerance': tolerance,
                })

            total_tested += 1

        except Exception as e:
            if debug:
                print(f"Prediction error: {e}")
            continue

    # Calculate metrics
    accuracy = correct_predictions / total_tested if total_tested > 0 else 0.0
    avg_inference_time = total_inference_time / total_tested if total_tested > 0 else 0.0
    inferences_per_second = 1000 / avg_inference_time if avg_inference_time > 0 else 0.0
    accuracy_real_only = correct_real / total_real if total_real > 0 else 0.0
    esp32_accuracy = esp32_correct / total_tested if total_tested > 0 and simulate_esp32 else 0.0

    # Verify failed count
    expected_failed = total_tested - correct_predictions
    if len(failed_predictions) != expected_failed:
        print(f"❌ CRITICAL ERROR: Failed count mismatch for {model_name}!")
        print(f"   Expected failed: {expected_failed} (total_tested: {total_tested} - correct: {correct_predictions})")
        print(f"   Actual failed collected: {len(failed_predictions)}")

    if debug:
        print(f"Final accuracy: {accuracy:.3f} ({correct_predictions}/{total_tested})")
        print(f"Real-only accuracy: {accuracy_real_only:.3f} ({correct_real}/{total_real})")
        if simulate_esp32:
            print(f"ESP32-sim accuracy: {esp32_accuracy:.3f} ({esp32_correct}/{total_tested})")
        print(f"Failed predictions: {len(failed_predictions)}")
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Inferences per second: {inferences_per_second:.0f}")

    return (accuracy, total_tested, avg_inference_time, inferences_per_second,
            failed_predictions, all_predictions_lite, esp32_accuracy, accuracy_real_only)


# ── Results Presentation ──────────────────────────────────────────────────

def generate_confusion_matrix(all_results, output_dir=None):
    """Generate a confusion matrix heatmap and per-class accuracy CSV from all predictions.

    Args:
        all_results: list of dicts with keys 'true_label', 'predicted_label', 'model'
        output_dir: directory where test_results/ will be written
    """
    if output_dir is None:
        output_dir = params.OUTPUT_DIR

    if not all_results:
        return {}

    graphs_dir = os.path.join(output_dir, "test_results", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "test_results")

    # Group by model
    model_results = {}
    for r in all_results:
        m = r.get('model', 'unknown')
        if m not in model_results:
            model_results[m] = []
        model_results[m].append(r)

    generated_files = {}

    for model_name, m_results in model_results.items():
        y_true = [r['true_label'] for r in m_results]
        y_pred = [r['predicted_label'] for r in m_results]
        num_classes = m_results[0].get('num_classes', params.NB_CLASSES)
        tolerance = m_results[0].get('tolerance', 0.1)

        classes = sorted(set(y_true) | set(y_pred))
        n = len(classes)
        idx = {c: i for i, c in enumerate(classes)}

        # Build confusion matrix
        cm = np.zeros((n, n), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            cm[idx[yt], idx[yp]] += 1

        # Plot heatmap
        fig_size = max(10, n // 2)
        plt.figure(figsize=(fig_size, fig_size))
        row_sums = cm.sum(axis=1, keepdims=True).clip(1)
        cm_norm = cm.astype(float) / row_sums

        plt.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(label='Fraction of true class')
        tick_labels = [str(c) for c in classes]
        plt.xticks(range(n), tick_labels, rotation=90, fontsize=max(6, 10 - n // 15))
        plt.yticks(range(n), tick_labels, fontsize=max(6, 10 - n // 15))
        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in ('_', '-')).rstrip().replace('.tflite', '')
        cm_filename = f"confusion_matrix_{safe_model_name}.png"
        cm_path = os.path.join(graphs_dir, cm_filename)
        plt.savefig(cm_path, dpi=200, bbox_inches='tight')
        plt.close()

        # Per-class accuracy CSV
        per_class_dir = os.path.join(results_dir, "per_class_accuracy")
        os.makedirs(per_class_dir, exist_ok=True)
        per_class_rows = []
        for c in classes:
            ci = idx[c]
            total = cm[ci].sum()
            correct = 0
            scale = num_classes / 10.0
            c_digit = float(c) / scale
            for yp_class in classes:
                yp_digit = float(yp_class) / scale
                diff = abs(c_digit - yp_digit) % 10.0
                circular_diff = min(diff, 10.0 - diff)
                if circular_diff <= tolerance:
                    correct += cm[ci, idx[yp_class]]

            per_class_rows.append({
                'Class': c,
                'Total': int(total),
                'Correct': int(correct),
                'Accuracy': f"{correct / total:.4f}" if total > 0 else 'N/A',
            })

        csv_filename = f"per_class_accuracy_{safe_model_name}.csv"
        per_class_csv = os.path.join(per_class_dir, csv_filename)
        with open(per_class_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Class', 'Total', 'Correct', 'Accuracy'])
            writer.writeheader()
            writer.writerows(per_class_rows)

        generated_files[model_name] = (cm_path, per_class_csv)

    print(f"   🔢 Generated confusion matrices and CSVs for {len(generated_files)} models")
    return generated_files


def generate_comparison_graphs(results, quantized_only=True, use_all_datasets=True,
                                output_dir=None, simulate_esp32=False):
    """Generate separate comparison graphs for the benchmark results."""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR

    graphs_dir = os.path.join(output_dir, "test_results", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    quant_suffix = "quantized" if quantized_only else "all"
    dataset_suffix = "full" if use_all_datasets else "sampled"

    # Prepare data
    plot_data = []
    for result in results:
        accuracy = float(result['Accuracy'])
        inferences_per_second = float(result['Inf/s'])
        if accuracy > 0 and inferences_per_second > 0:
            dir_name = result['Directory']
            model_name = result['Model']
            label = f"{dir_name}\n{model_name.replace('.tflite', '')}"
            params_str = result['Params']
            if 'M' in params_str:
                params_val = float(params_str.replace('M', '')) * 1_000_000
            elif 'K' in params_str:
                params_val = float(params_str.replace('K', '')) * 1_000
            else:
                params_val = float(params_str)
            plot_data.append({
                'label': label, 'directory': dir_name, 'model_name': model_name,
                'accuracy': accuracy * 100,
                'inferences_per_second': inferences_per_second,
                'size_kb': float(result['Size (KB)']),
                'parameters_million': params_val / 1_000_000,
            })

    # ESP32 plot data
    esp32_plot_data = []
    if simulate_esp32:
        for result in results:
            esp32_acc = float(result.get('ESP32_Accuracy', 0.0))
            pc_acc = float(result['Accuracy'])
            inferences_per_second = float(result['Inf/s'])
            if esp32_acc > 0 and inferences_per_second > 0:
                dir_name = result['Directory']
                model_name = result['Model']
                label = f"{dir_name}\n{model_name.replace('.tflite', '')}"
                params_str = result['Params']
                if 'M' in params_str:
                    params_val = float(params_str.replace('M', '')) * 1_000_000
                elif 'K' in params_str:
                    params_val = float(params_str.replace('K', '')) * 1_000
                else:
                    params_val = float(params_str)
                esp32_plot_data.append({
                    'label': label, 'directory': dir_name, 'model_name': model_name,
                    'accuracy': esp32_acc * 100, 'pc_accuracy': pc_acc * 100,
                    'inferences_per_second': inferences_per_second,
                    'size_kb': float(result['Size (KB)']),
                    'parameters_million': params_val / 1_000_000,
                })

    if not plot_data:
        print("⚠️  No valid data points to generate comparison graphs.")
        return []

    labels = [d['label'] for d in plot_data]
    accuracies = [d['accuracy'] for d in plot_data]
    inferences_per_second = [d['inferences_per_second'] for d in plot_data]
    sizes_kb = [d['size_kb'] for d in plot_data]
    parameters = [d['parameters_million'] for d in plot_data]
    model_names = [d['model_name'] for d in plot_data]

    graph_paths = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']

    # Graph 1: Accuracy vs Speed
    plt.figure(figsize=(14, 10))
    for i, (label, x, y) in enumerate(zip(labels, inferences_per_second, accuracies)):
        plt.scatter(x, y, c=[colors[i]], s=120, marker=markers[i % len(markers)],
                    alpha=0.8, label=label, edgecolors='black', linewidth=0.5)
    plt.xlabel('Inferences per Second', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Speed\n({quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.tight_layout()
    graph1_path = os.path.join(graphs_dir, f"accuracy_vs_speed_{quant_suffix}_{dataset_suffix}.png")
    plt.savefig(graph1_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph1_path)

    # Graph 2: Model Size vs Accuracy
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(sizes_kb, accuracies, c=parameters, s=100, alpha=0.7, cmap='viridis')
    for i, (label, x, y) in enumerate(zip(labels, sizes_kb, accuracies)):
        plt.annotate(label.split('\n')[0], (x, y), xytext=(8, 8), textcoords='offset points',
                     fontsize=8, alpha=0.9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.xlabel('Model Size (KB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Model Size\n({quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Parameters (Millions)', fontsize=10)
    plt.tight_layout()
    graph2_path = os.path.join(graphs_dir, f"accuracy_vs_size_{quant_suffix}_{dataset_suffix}.png")
    plt.savefig(graph2_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph2_path)

    # Graph 3: Parameters vs Speed
    plt.figure(figsize=(14, 10))
    for i, (label, x, y) in enumerate(zip(labels, parameters, inferences_per_second)):
        plt.scatter(x, y, c=[colors[i]], s=120, marker=markers[i % len(markers)],
                    alpha=0.8, label=label, edgecolors='black', linewidth=0.5)
    plt.xlabel('Parameters (Millions)', fontsize=12)
    plt.ylabel('Inferences per Second', fontsize=12)
    plt.title(f'Speed vs Model Complexity\n({quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.tight_layout()
    graph3_path = os.path.join(graphs_dir, f"speed_vs_complexity_{quant_suffix}_{dataset_suffix}.png")
    plt.savefig(graph3_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph3_path)

    # Graph 4: Inference Speed Bar Chart
    plt.figure(figsize=(16, 10))
    y_pos = np.arange(len(labels))
    bars = plt.barh(y_pos, inferences_per_second, color=colors, alpha=0.7)
    plt.yticks(y_pos, [label.split('\n')[0] for label in labels], fontsize=10)
    plt.xlabel('Inferences per Second', fontsize=12)
    plt.title(f'Inference Speed Comparison\n({quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    for i, (bar, acc, speed) in enumerate(zip(bars, accuracies, inferences_per_second)):
        plt.text(bar.get_width() + max(inferences_per_second) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{int(speed)} inf/s\n{acc:.1f}% acc',
                 ha='left', va='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7,
                                     label=f"{labels[i].split(chr(10))[0]}\n{model_names[i]}")
                       for i in range(len(labels))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., fontsize=8)
    plt.tight_layout()
    graph4_path = os.path.join(graphs_dir, f"speed_comparison_{quant_suffix}_{dataset_suffix}.png")
    plt.savefig(graph4_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph4_path)

    # Graph 5: Accuracy Bar Chart
    plt.figure(figsize=(16, 10))
    y_pos = np.arange(len(labels))
    bar_colors = ['lightgreen' if acc == max(accuracies) else colors[i] for i, acc in enumerate(accuracies)]
    bars = plt.barh(y_pos, accuracies, color=bar_colors, alpha=0.7)
    plt.yticks(y_pos, [label.split('\n')[0] for label in labels], fontsize=10)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy Comparison\n({quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    for i, (bar, acc, speed) in enumerate(zip(bars, accuracies, inferences_per_second)):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{acc:.1f}% acc\n{int(speed)} inf/s',
                 ha='left', va='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7,
                                     label=f"{labels[i].split(chr(10))[0]}\n{model_names[i]}")
                       for i in range(len(labels))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., fontsize=8)
    plt.tight_layout()
    graph5_path = os.path.join(graphs_dir, f"accuracy_comparison_{quant_suffix}_{dataset_suffix}.png")
    plt.savefig(graph5_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph5_path)

    # ESP32-specific graphs
    if simulate_esp32 and esp32_plot_data:
        esp32_graphs_dir = os.path.join(graphs_dir, "esp32")
        os.makedirs(esp32_graphs_dir, exist_ok=True)

        esp32_labels = [d['label'] for d in esp32_plot_data]
        esp32_accuracies = [d['accuracy'] for d in esp32_plot_data]
        esp32_pc_accuracies = [d['pc_accuracy'] for d in esp32_plot_data]
        esp32_sizes_kb = [d['size_kb'] for d in esp32_plot_data]

        # ESP32 Graph 1: PC vs ESP32 Accuracy
        plt.figure(figsize=(16, 10))
        y_pos = np.arange(len(esp32_labels))
        bar_height = 0.35
        bars1 = plt.barh(y_pos - bar_height / 2, esp32_pc_accuracies, bar_height,
                         color='steelblue', alpha=0.7, label='PC Accuracy')
        bars2 = plt.barh(y_pos + bar_height / 2, esp32_accuracies, bar_height,
                         color='coral', alpha=0.7, label='ESP32 Sim.')
        plt.yticks(y_pos, [label.split('\n')[0] for label in esp32_labels], fontsize=10)
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.title(f'PC vs ESP32-Simulated Accuracy\n({quant_suffix}, {dataset_suffix})',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.legend(fontsize=11)
        for i, (bar1, bar2, pc_acc, esp32_acc) in enumerate(
                zip(bars1, bars2, esp32_pc_accuracies, esp32_accuracies)):
            gap = pc_acc - esp32_acc
            plt.text(max(bar1.get_width(), bar2.get_width()) + 0.5,
                     bar1.get_y() + bar1.get_height() / 2,
                     f'PC:{pc_acc:.1f}% ESP32:{esp32_acc:.1f}% Gap:{gap:+.1f}%',
                     ha='left', va='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
        plt.tight_layout()
        esp32_graph1 = os.path.join(esp32_graphs_dir,
                                    f"pc_vs_esp32_accuracy_{quant_suffix}_{dataset_suffix}.png")
        plt.savefig(esp32_graph1, dpi=300, bbox_inches='tight')
        plt.close()
        graph_paths.append(esp32_graph1)

        # ESP32 Graph 2: Accuracy Gap
        gaps = [pc - esp for pc, esp in zip(esp32_pc_accuracies, esp32_accuracies)]
        sorted_indices = np.argsort(gaps)[::-1]
        plt.figure(figsize=(14, 8))
        sorted_labels = [esp32_labels[i].split('\n')[0] for i in sorted_indices]
        sorted_gaps = [gaps[i] for i in sorted_indices]
        colors_gap = ['red' if g > 2 else 'orange' if g > 1 else 'green' for g in sorted_gaps]
        bars = plt.barh(range(len(sorted_gaps)), sorted_gaps, color=colors_gap, alpha=0.7)
        plt.yticks(range(len(sorted_gaps)), sorted_labels, fontsize=10)
        plt.xlabel('Accuracy Gap (PC - ESP32) %', fontsize=12)
        plt.title(f'ESP32 Accuracy Degradation\n({quant_suffix}, {dataset_suffix})',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        for i, (bar, gap) in enumerate(zip(bars, sorted_gaps)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{gap:+.1f}%', ha='left', va='center', fontsize=9)
        plt.tight_layout()
        esp32_graph2 = os.path.join(esp32_graphs_dir,
                                    f"esp32_accuracy_gap_{quant_suffix}_{dataset_suffix}.png")
        plt.savefig(esp32_graph2, dpi=300, bbox_inches='tight')
        plt.close()
        graph_paths.append(esp32_graph2)

        # ESP32 Graph 3: ESP32 Accuracy vs Size
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(esp32_sizes_kb, esp32_accuracies, c=esp32_accuracies,
                              s=100, alpha=0.7, cmap='RdYlGn')
        for i, (label, x, y) in enumerate(zip(esp32_labels, esp32_sizes_kb, esp32_accuracies)):
            plt.annotate(label.split('\n')[0], (x, y), xytext=(8, 8),
                         textcoords='offset points', fontsize=8, alpha=0.9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.xlabel('Model Size (KB)', fontsize=12)
        plt.ylabel('ESP32-Sim Accuracy (%)', fontsize=12)
        plt.title(f'ESP32-Sim Accuracy vs Model Size\n({quant_suffix}, {dataset_suffix})',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter)
        cbar.set_label('ESP32 Accuracy (%)', fontsize=10)
        plt.tight_layout()
        esp32_graph3 = os.path.join(esp32_graphs_dir,
                                    f"esp32_accuracy_vs_size_{quant_suffix}_{dataset_suffix}.png")
        plt.savefig(esp32_graph3, dpi=300, bbox_inches='tight')
        plt.close()
        graph_paths.append(esp32_graph3)

    print(f"📊 Generated {len(graph_paths)} comparison graphs:")
    for path in graph_paths:
        print(f"   📈 {os.path.basename(path)}")

    return graph_paths


# ── Batch Testing ─────────────────────────────────────────────────────────

def test_all_models(num_test_images=0, quantized_only=False, debug=False,
                    use_all_datasets=True, list_failed=False, save_failed=False,
                    subfolder=None, input_dir=None, exclude_model=None,
                    override_classes=None, override_color=None,
                    model_list=None, tolerance=0.1, update_csv=False,
                    iot_compat=True, simulate_esp32=True):
    """Test all valid models with optional subfolder filtering and model exclusion."""
    from .data import get_all_models, configure_parameters_for_model

    if input_dir is None:
        input_dir = params.OUTPUT_DIR

    models = get_all_models(quantized_only=quantized_only, subfolder=subfolder,
                            input_dir=input_dir, exclude_model=exclude_model,
                            debug=debug, model_list=model_list, iot_compat=iot_compat)

    if not models:
        if subfolder:
            print(f"No models found in subfolder '{subfolder}'")
        else:
            print("No models found to test.")
        return

    # Auto-configure dataset params
    if models:
        configure_parameters_for_model(models[0]['directory'],
                                       override_classes=override_classes,
                                       override_color=override_color)

    subfolder_info = f" in subfolder '{subfolder}'" if subfolder else ""
    if use_all_datasets or num_test_images == 0:
        print(f"\nTesting {len(models)} models{subfolder_info} on ALL available images from all datasets...")
    else:
        print(f"\nTesting {len(models)} models{subfolder_info} on {num_test_images} images...")
    print("-" * 80)

    results = []
    all_failed_predictions = []
    all_predictions = []

    if debug:
        model_iterator = models
        print("Debug mode - showing detailed results for each model")
    else:
        model_iterator = tqdm(models, desc="Testing all models")

    for model_info in model_iterator:
        if debug:
            print(f"\n🔍 Testing model: {model_info['directory']}/{model_info['name']}")

        results_data = test_model_on_dataset(
            model_info['path'],
            num_test_images=num_test_images,
            debug=debug,
            use_all_datasets=use_all_datasets,
            collect_failed=(list_failed or save_failed),
            model_name=model_info['name'],
            tolerance=tolerance,
            simulate_esp32=simulate_esp32,
        )

        if results_data is None or results_data[1] == 0:
            if debug:
                print(f"⚠️  Skipping results for {model_info['name']} due to failure")
            continue

        (accuracy, tested_count, avg_inference_time, inferences_per_second,
         failed_predictions, all_predictions_lite, esp32_accuracy, accuracy_real_only) = results_data

        all_predictions.extend(all_predictions_lite)
        for failure in failed_predictions:
            failure['model_directory'] = model_info['directory']
            all_failed_predictions.append(failure)

        params_count = model_info['parameters']
        if params_count >= 1_000_000:
            params_str = f"{params_count/1_000_000:.1f}M"
        elif params_count >= 1_000:
            params_str = f"{params_count/1_000:.1f}K"
        else:
            params_str = f"{params_count}"

        results.append({
            'Model': model_info['name'],
            'Directory': model_info['directory'],
            'Type': model_info['type'],
            'Params': params_str,
            'Params_Raw': params_count,
            'Size (KB)': f"{model_info['size_kb']:.1f}",
            'Size_Raw': model_info['size_kb'],
            'Accuracy': f"{accuracy:.4f}",
            'Accuracy_Raw': accuracy,
            'Accuracy_RealOnly': f"{accuracy_real_only:.4f}",
            'Accuracy_RealOnly_Raw': accuracy_real_only,
            'ESP32_Accuracy': f"{esp32_accuracy:.4f}",
            'ESP32_Accuracy_Raw': esp32_accuracy,
            'Inf Time (ms)': f"{avg_inference_time:.2f}",
            'Inf Time_Raw': avg_inference_time,
            'Inf/s': f"{inferences_per_second:.0f}",
            'Inf/s_Raw': inferences_per_second,
            'Tested': tested_count,
            'Failed_Count': len(failed_predictions),
        })

        if debug:
            print(f"✅ Completed: {model_info['directory']}/{model_info['name']} - Accuracy: {accuracy:.3f}")

    # Handle failed predictions
    if list_failed and all_failed_predictions:
        generate_failed_predictions_csv(all_failed_predictions, input_dir)
    if save_failed and all_failed_predictions:
        save_failed_images(all_failed_predictions, input_dir)

    # Sort by accuracy descending
    results.sort(key=lambda x: x['Accuracy_Raw'], reverse=True)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY RESULTS")
    if subfolder:
        print(f"SUBFOLDER: {subfolder}")
    if use_all_datasets or num_test_images == 0:
        print(f"DATASETS: ALL available images")
    else:
        print(f"DATASETS: {num_test_images} sampled images")
    print(f"MODELS: {'Quantized only' if quantized_only else 'All models'}")
    if exclude_model:
        excl_str = ", ".join(exclude_model) if not isinstance(exclude_model, str) else exclude_model
        print(f"EXCLUDED MODELS: {excl_str}")
    if list_failed or save_failed:
        print(f"FAILED PREDICTIONS: {len(all_failed_predictions)} total across all models")
    print(f"{'='*80}")

    if simulate_esp32:
        headers = ['Directory', 'Model', 'Type', 'Params', 'Size', 'Accuracy',
                    'RealOnly', 'ESP32', 'Gap', 'Inf/s', 'Images', 'Failed']
    else:
        headers = ['Directory', 'Model', 'Type', 'Params', 'Size', 'Accuracy',
                    'RealOnly', 'Inf/s', 'Images', 'Failed']
    table_data = []
    for result in results:
        row = [
            result['Directory'], result['Model'], result['Type'], result['Params'],
            result['Size (KB)'], f"{float(result['Accuracy']):.3f}",
            f"{float(result['Accuracy_RealOnly']):.3f}",
        ]
        if simulate_esp32:
            esp32_acc = float(result['ESP32_Accuracy'])
            gap = (float(result['Accuracy']) - esp32_acc) * 100
            row.append(f"{esp32_acc:.3f}")
            row.append(f"{gap:+.1f}%")
        row.extend([result['Inf/s'], result['Tested'], result['Failed_Count']])
        table_data.append(row)
    print(tabulate(table_data, headers=headers, tablefmt='simple_grid', stralign='right'))

    # Best models
    if results and results[0]['Accuracy_Raw'] > 0:
        best_accuracy = max(results, key=lambda x: x['Accuracy_Raw'])
        fastest_model = max(results, key=lambda x: x['Inf/s_Raw'])
        print(f"\n🏆 BEST BY ACCURACY: {best_accuracy['Directory']}/{best_accuracy['Model']}")
        print(f"   Accuracy: {float(best_accuracy['Accuracy']):.3f}, Speed: {best_accuracy['Inf/s']} inf/s")
        if simulate_esp32:
            best_esp32 = max(results, key=lambda x: x['ESP32_Accuracy_Raw'])
            print(f"\n🔌 BEST ESP32-SIM: {best_esp32['Directory']}/{best_esp32['Model']}")
        print(f"⚡ FASTEST MODEL: {fastest_model['Directory']}/{fastest_model['Model']}")

    # Generate graphs
    graph_paths = generate_comparison_graphs(results, quantized_only=quantized_only,
                                              use_all_datasets=use_all_datasets,
                                              output_dir=input_dir,
                                              simulate_esp32=simulate_esp32)

    # Save CSV
    csv_path = save_results_to_csv(results, quantized_only=quantized_only,
                                    use_all_images=use_all_datasets,
                                    test_images_count=num_test_images,
                                    output_dir=input_dir, update_csv=update_csv)

    # Generate markdown report
    if csv_path and graph_paths:
        markdown_path = generate_markdown_report(csv_path, graph_paths, results,
                                                  quantized_only=quantized_only,
                                                  use_all_datasets=use_all_datasets,
                                                  test_images_count=num_test_images,
                                                  output_dir=input_dir,
                                                  simulate_esp32=simulate_esp32)
        print(f"📄 Comprehensive markdown report generated: {markdown_path}")

    # Confusion matrix
    if all_predictions:
        print(f"\n🔢 Generating confusion matrix from {len(all_predictions)} predictions...")
        generate_confusion_matrix(all_predictions, output_dir=input_dir)

    return results, all_failed_predictions


# ── CSV and Reports ───────────────────────────────────────────────────────

def save_results_to_csv(results, quantized_only=True, use_all_images=True,
                        test_images_count=0, output_dir=None, update_csv=False):
    """Save FULL results to CSV file with all information."""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR

    results_dir = os.path.join(output_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "model_comparison.csv")

    csv_data = []
    for result in results:
        params_str = result['Params']
        if 'M' in params_str:
            params_count = float(params_str.replace('M', '')) * 1_000_000
        elif 'K' in params_str:
            params_count = float(params_str.replace('K', '')) * 1_000
        else:
            params_count = float(params_str)

        csv_data.append({
            'Model': result['Model'],
            'Directory': result['Directory'],
            'Type': result['Type'],
            'Parameters': int(params_count),
            'Size_KB': float(result['Size (KB)']),
            'Accuracy': float(result['Accuracy']),
            'Accuracy_RealOnly': float(result.get('Accuracy_RealOnly', 0.0)),
            'ESP32_Accuracy': float(result.get('ESP32_Accuracy', 0.0)),
            'Inference_Time_ms': float(result['Inf Time (ms)']),
            'Inferences_per_second': float(result['Inf/s']),
            'Tested_Images': result['Tested'],
        })

    # Update logic - merge with existing CSV
    if update_csv and os.path.exists(csv_path):
        try:
            existing_data = []
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
            data_map = {(row['Directory'], row['Model']): row for row in existing_data}
            for new_row in csv_data:
                key = (new_row['Directory'], new_row['Model'])
                data_map[key] = {k: str(v) for k, v in new_row.items()}
            csv_data = list(data_map.values())
            print(f"🔄 Appended/Updated {len(results)} models into existing CSV (total: {len(csv_data)})")
        except Exception as e:
            print(f"⚠️  Could not update existing CSV: {e}. Falling back to overwrite.")

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model', 'Directory', 'Type', 'Parameters', 'Size_KB',
                      'Accuracy', 'Accuracy_RealOnly', 'ESP32_Accuracy',
                      'Inference_Time_ms', 'Inferences_per_second', 'Tested_Images']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    print(f"💾 Full results saved to: {csv_path}")
    return csv_path


# ── IoT Model Analysis ────────────────────────────────────────────────────

def calculate_best_iot_model(df, accuracy_weight=0.5, size_weight=0.3, speed_weight=0.2):
    """Dynamically calculate the best IoT model based on weighted criteria."""
    df = df.copy()
    df = df[(df['Accuracy'] > 0) & (df['Inferences_per_second'] > 0) & (df['Size_KB'] > 0)].copy()

    if df.empty:
        return {
            'best_overall': None, 'best_accuracy_small': None,
            'best_speed_small': None, 'smallest_adequate': None,
            'all_scores': pd.DataFrame(columns=['Model', 'Accuracy', 'Size_KB',
                                                  'Inferences_per_second', 'iot_score',
                                                  'accuracy_per_kb']),
            'weights_used': {'accuracy': accuracy_weight, 'size': size_weight, 'speed': speed_weight},
        }

    df['accuracy_norm'] = df['Accuracy'] / df['Accuracy'].max()
    df['size_norm'] = (1 / df['Size_KB']) / (1 / df['Size_KB']).max()
    df['speed_norm'] = df['Inferences_per_second'] / df['Inferences_per_second'].max()
    df['iot_score'] = (df['accuracy_norm'] * accuracy_weight +
                       df['size_norm'] * size_weight +
                       df['speed_norm'] * speed_weight)
    best_iot_model = df.loc[df['iot_score'].idxmax()]
    df['accuracy_per_kb'] = df['Accuracy'] / df['Size_KB']
    if 'Parameters' in df.columns:
        df['accuracy_per_param'] = df['Accuracy'] / df['Parameters']
    else:
        df['accuracy_per_param'] = 0

    small_models = df[df['Size_KB'] <= 100]
    accurate_models = df[df['Accuracy'] >= 0.90]
    best_accuracy_small = small_models.loc[small_models['Accuracy'].idxmax()] if not small_models.empty else df.loc[df['Size_KB'].idxmin()]
    best_speed_small = small_models.loc[small_models['Inferences_per_second'].idxmax()] if not small_models.empty else df.loc[df['Inferences_per_second'].idxmax()]
    smallest_adequate = accurate_models.loc[accurate_models['Size_KB'].idxmin()] if not accurate_models.empty else df.loc[df['Accuracy'].idxmax()]

    result_columns = ['Model', 'Accuracy', 'Size_KB', 'Inferences_per_second', 'iot_score', 'accuracy_per_kb']
    if 'Parameters' in df.columns:
        result_columns.append('Parameters')
    if 'accuracy_per_param' in df.columns:
        result_columns.append('accuracy_per_param')

    return {
        'best_overall': best_iot_model, 'best_accuracy_small': best_accuracy_small,
        'best_speed_small': best_speed_small, 'smallest_adequate': smallest_adequate,
        'all_scores': df[result_columns].sort_values('iot_score', ascending=False),
        'weights_used': {'accuracy': accuracy_weight, 'size': size_weight, 'speed': speed_weight},
    }


def generate_iot_recommendation_section(f, df):
    """Generate dynamic IoT recommendations section."""
    if len(df) <= 1:
        f.write("## 💡 IoT-Specific Recommendations\n\n")
        f.write("*Not enough models for comparative IoT analysis*\n\n")
        return

    analysis = calculate_best_iot_model(df)
    df_with_scores = analysis['all_scores']

    if df_with_scores.empty:
        f.write("## 💡 IoT-Specific Recommendations\n\n")
        f.write("*No models with valid metrics for comparative IoT analysis after filtering.*\n\n")
        return

    best_model = analysis['best_overall']
    best_accuracy_small = analysis['best_accuracy_small']
    best_speed_small = analysis['best_speed_small']
    smallest_adequate = analysis['smallest_adequate']

    f.write("## 💡 IoT-Specific Recommendations\n\n")
    f.write("### 🏆 Dynamic IoT Model Selection\n\n")
    f.write("#### 🎯 Best Overall for ESP32\n")
    f.write(f"- **Model**: **{best_model['Model']}**\n")
    f.write(f"- **IoT Score**: {best_model['iot_score']:.3f}\n")
    f.write(f"- **Accuracy**: {best_model['Accuracy']:.3f}\n")
    f.write(f"- **Size**: {best_model['Size_KB']:.1f} KB\n")
    f.write(f"- **Speed**: {best_model['Inferences_per_second']:.0f} inf/s\n")
    accuracy_per_kb = best_model.get('accuracy_per_kb', best_model['Accuracy'] / best_model['Size_KB'])
    f.write(f"- **Efficiency**: {accuracy_per_kb:.4f} accuracy per KB\n\n")

    f.write("#### 📊 IoT Model Comparison (Under 100KB)\n")
    f.write("| Model | Accuracy | Size | Speed | IoT Score | Use Case |\n")
    f.write("|-------|----------|------|-------|-----------|----------|\n")
    small_models = df_with_scores[df_with_scores['Size_KB'] <= 100].nlargest(5, 'iot_score')
    if small_models.empty:
        f.write("| *No models under 100KB* | - | - | - | - | - |\n")
    else:
        for _, model in small_models.iterrows():
            use_case = "Alternative"
            if best_model is not None and model['Model'] == best_model['Model']:
                use_case = "🏆 **BEST BALANCED**"
            elif best_accuracy_small is not None and model['Model'] == best_accuracy_small['Model']:
                use_case = "🎯 Best Accuracy"
            elif best_speed_small is not None and model['Model'] == best_speed_small['Model']:
                use_case = "⚡ Fastest"
            elif smallest_adequate is not None and model['Model'] == smallest_adequate['Model']:
                use_case = "💾 Smallest Adequate"
            f.write(f"| {model['Model']} | {model['Accuracy']:.3f} | {model['Size_KB']:.1f}KB | "
                    f"{model['Inferences_per_second']:.0f}/s | {model['iot_score']:.3f} | {use_case} |\n")

    f.write("\n")
    f.write("#### 🔧 Alternative IoT Scenarios\n\n")
    if best_accuracy_small is not None:
        f.write(f"**For Accuracy-Critical IoT:** Choice: {best_accuracy_small['Model']}, "
                f"Accuracy: {best_accuracy_small['Accuracy']:.3f}, "
                f"Trade-off: {best_accuracy_small['Size_KB']:.1f}KB\n\n")
    if best_speed_small is not None:
        f.write(f"**For Speed-Critical IoT:** Choice: {best_speed_small['Model']}, "
                f"Speed: {best_speed_small['Inferences_per_second']:.0f} inf/s, "
                f"Trade-off: {best_speed_small['Accuracy']:.3f} accuracy\n\n")
    if smallest_adequate is not None:
        f.write(f"**For Memory-Constrained IoT:** Choice: {smallest_adequate['Model']}, "
                f"Size: {smallest_adequate['Size_KB']:.1f}KB, "
                f"Trade-off: {smallest_adequate['Accuracy']:.3f} accuracy\n\n")


def generate_markdown_report(csv_path, graph_paths, results, quantized_only=True,
                             use_all_datasets=True, test_images_count=0,
                             output_dir=None, simulate_esp32=False):
    """Generate a comprehensive Markdown report from CSV results and graphs."""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR

    df = pd.read_csv(csv_path)
    reports_dir = os.path.join(output_dir, "test_results")
    os.makedirs(reports_dir, exist_ok=True)

    report_path = os.path.join(reports_dir, "readme.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Digit Recognition Benchmark Report\n\n")

        # ESP32 Section
        if simulate_esp32 and 'ESP32_Accuracy' in df.columns:
            f.write("## 🔌 ESP32 Hardware Simulation\n\n")
            f.write("> **What this is**: Each model was also tested through an ESP32-simulated inference "
                    "pipeline that adds quantization noise to simulate the integer-only arithmetic of "
                    "TFLite Micro on ESP32. Models with a smaller gap between PC and ESP32 accuracy "
                    "are more robust for real hardware deployment.\n\n")
            f.write("| Model | PC Accuracy | ESP32 Sim. | Gap | Verdict |\n")
            f.write("|-------|-------------|------------|-----|---------|\n")
            esp32_df = df[df['ESP32_Accuracy'] > 0].sort_values('ESP32_Accuracy', ascending=False)
            for _, row in esp32_df.iterrows():
                gap = (row['Accuracy'] - row['ESP32_Accuracy']) * 100
                if gap < 0.5: verdict = "✅ Excellent (robust)"
                elif gap < 1.5: verdict = "⚠️ Good (minor loss)"
                elif gap < 3.0: verdict = "⚠️ Moderate loss"
                else: verdict = "❌ Significant loss"
                f.write(f"| {row['Model']} | {row['Accuracy']:.3f} | {row['ESP32_Accuracy']:.3f} | "
                        f"{gap:+.1f}% | {verdict} |\n")
            f.write("\n")
            quant_suffix = "quantized" if quantized_only else "all"
            dataset_suffix = "full" if use_all_datasets else "sampled"
            f.write("### 📊 ESP32 Simulation Graphs\n\n")
            f.write(f"![PC vs ESP32 Accuracy](graphs/esp32/pc_vs_esp32_accuracy_{quant_suffix}_{dataset_suffix}.png)\n\n")
            f.write(f"![ESP32 Accuracy Gap](graphs/esp32/esp32_accuracy_gap_{quant_suffix}_{dataset_suffix}.png)\n\n")
            f.write(f"![ESP32 Accuracy vs Size](graphs/esp32/esp32_accuracy_vs_size_{quant_suffix}_{dataset_suffix}.png)\n\n")
            f.write("---\n\n")

        # Executive Summary
        f.write("## 📊 Executive Summary\n\n")
        iot_analysis = calculate_best_iot_model(df)
        if iot_analysis['best_overall'] is not None:
            best_iot = iot_analysis['best_overall']
            best_accuracy = df.loc[df['Accuracy'].idxmax()]
            fastest = df.loc[df['Inferences_per_second'].idxmax()]
            smallest = df.loc[df['Size_KB'].idxmin()]
            f.write(f"- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Models Tested**: {len(df)} {'quantized' if quantized_only else 'all'} models\n")
            f.write(f"- **Best IoT Model**: **{best_iot['Model']}** ({best_iot['Size_KB']:.1f}KB, "
                    f"{best_iot['Accuracy']:.3f} acc, {best_iot['Inferences_per_second']:.0f} inf/s)\n")
            f.write(f"- **Best Accuracy**: **{best_accuracy['Model']}** ({best_accuracy['Accuracy']:.3f})\n")
            f.write(f"- **Fastest Model**: **{fastest['Model']}** ({fastest['Inferences_per_second']:.0f} inf/s)\n")
            f.write(f"- **Smallest Model**: **{smallest['Model']}** ({smallest['Size_KB']:.1f} KB)\n\n")
        else:
            f.write(f"- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Models Tested**: {len(df)} models\n")
            f.write("- *No models with valid metrics for comparative analysis.*\n\n")

        f.write("## 📈 Performance vs Size\n\n")
        quant_suffix = "quantized" if quantized_only else "all"
        dataset_suffix = "full" if use_all_datasets else "sampled"
        f.write(f"![Accuracy vs Size](graphs/accuracy_vs_size_{quant_suffix}_{dataset_suffix}.png)\n\n")

        # Detailed Results
        f.write("## 📋 Detailed Results\n\n")
        f.write("| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |\n")
        f.write("|-------|-----------|----------|-------|------------|-----------|\n")
        scored_df = iot_analysis['all_scores']
        for _, row in scored_df.iterrows():
            parameters = row.get('Parameters', 'N/A')
            f.write(f"| {row['Model']} | {row['Size_KB']:.1f} | {row['Accuracy']:.3f} | "
                    f"{row['Inferences_per_second']:.0f} | {parameters} | {row['iot_score']:.3f} |\n")
        f.write("\n")

        generate_iot_recommendation_section(f, df)
        f.write("---\n")
        f.write("*Report generated automatically by Digit Recognition Benchmarking Tool*\n")

    print(f"📄 Markdown report generated: {report_path}")
    return report_path


# ── Image / CSV Export Helpers ────────────────────────────────────────────

def save_failed_images(failed_predictions, output_dir):
    """Save failed prediction images to directory for manual review."""
    failed_dir = os.path.join(output_dir, "failed-predictions")
    os.makedirs(failed_dir, exist_ok=True)

    saved_count = 0
    for failure in failed_predictions:
        try:
            image_data = failure['image']
            true_label = failure['true_label']
            predicted_label = failure['predicted_label']
            confidence = failure['confidence']
            original_fname = failure.get('original_fname', 'unknown')

            filename = f"{original_fname}_{predicted_label:.1f}_conf_{confidence:.3f}.jpg"
            filepath = os.path.join(failed_dir, filename)
            if os.path.exists(filepath):
                filename = f"{original_fname}_{predicted_label:.1f}_conf_{confidence:.3f}_{random.randint(100, 999)}.jpg"
                filepath = os.path.join(failed_dir, filename)

            if isinstance(image_data, np.ndarray):
                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                    img = Image.fromarray(image_data.astype(np.uint8))
                elif len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[2] == 1):
                    if len(image_data.shape) == 3:
                        image_data = image_data.squeeze()
                    img = Image.fromarray(image_data.astype(np.uint8))
                img.save(filepath)
                saved_count += 1
            elif isinstance(image_data, str) and os.path.exists(image_data):
                shutil.copy2(image_data, filepath)
                saved_count += 1
        except Exception as e:
            logger.warning(f"Could not save failed image: {e}")
            continue

    print(f"💾 Saved {saved_count} failed images to: {failed_dir}")
    return failed_dir


def generate_failed_predictions_csv(failed_predictions, output_dir):
    """Generate CSV file with details of failed predictions."""
    if not failed_predictions:
        print("No failed predictions to save.")
        return None

    results_dir = os.path.join(output_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "failed_predictions.csv")

    csv_data = []
    for i, failure in enumerate(failed_predictions):
        row = {
            'index': i,
            'true_label': failure['true_label'],
            'predicted_label': failure['predicted_label'],
            'confidence': f"{failure['confidence']:.4f}",
            'model': failure.get('model', 'unknown'),
            'model_directory': failure.get('model_directory', 'unknown'),
            'image_source': failure.get('image_source', 'unknown'),
        }
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)

    print(f"📊 Failed predictions CSV saved to: {csv_path}")
    print(f"   Total failed predictions: {len(failed_predictions)}")
    if len(failed_predictions) > 0:
        print("\nFailed predictions by true label:")
        failed_by_label = df.groupby('true_label').size()
        for label, count in failed_by_label.items():
            print(f"  Label {label}: {count} failures")
    return csv_path


def test_single_model(model_path, num_test_images=0, debug=False,
                      use_all_datasets=True, list_failed=False,
                      save_failed=False, output_dir=None,
                      override_classes=None, override_color=None,
                      tolerance=0.1):
    """Test a single model and optionally collect failed predictions."""
    from .data import configure_parameters_for_model

    predictor = TFLiteDigitPredictor(model_path)
    configure_parameters_for_model(os.path.basename(model_path),
                                   override_classes=override_classes,
                                   override_color=override_color)

    test_data = load_test_dataset_with_labels(num_test_images, use_all_datasets)
    if not test_data:
        print("❌ No test data available")
        return

    print(f"Testing model: {os.path.basename(model_path)}")
    print(f"Test images: {len(test_data)}")
    print("-" * 50)

    correct_predictions = 0
    total_tested = 0
    total_inference_time = 0.0
    failed_predictions = []
    all_predictions_lite = []

    # Warm-up
    if len(test_data) > 0:
        warmup_image, _, _, _ = test_data[0]
        if warmup_image is not None:
            try:
                predictor.predict(warmup_image, debug=False)
            except:
                pass

    for i, (image, true_label, original_fname, _) in enumerate(test_data):
        if image is None:
            continue
        try:
            start_time = time.perf_counter()
            prediction, confidence, _ = predictor.predict(image, debug=debug)
            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time

            dataset_scale = params.NB_CLASSES / 10.0
            true_digit = float(true_label) / dataset_scale
            model_scale = predictor.num_classes / 10.0
            pred_digit = float(prediction) / model_scale
            diff = abs(true_digit - pred_digit) % 10.0
            circular_diff = min(diff, 10.0 - diff)

            if circular_diff <= tolerance:
                correct_predictions += 1
                if debug:
                    print(f"✓ {i:4d}: Correct - Pred: {pred_digit:.1f}, True: {true_digit:.1f}, Conf: {confidence:.3f}")
            else:
                if debug:
                    print(f"✗ {i:4d}: Wrong - Pred: {pred_digit:.1f}, True: {true_digit:.1f}, Conf: {confidence:.3f}")
                failed_predictions.append({
                    'image': image, 'true_label': round(true_digit, 1),
                    'predicted_label': round(pred_digit, 1), 'confidence': confidence,
                    'model': os.path.basename(model_path), 'image_source': 'dataset',
                    'index': i, 'original_fname': original_fname,
                })

            all_predictions_lite.append({
                'true_label': round(true_digit, 1),
                'predicted_label': round(pred_digit, 1),
                'model': os.path.basename(model_path),
                'num_classes': predictor.num_classes,
                'tolerance': tolerance,
            })
            total_tested += 1

        except Exception as e:
            if debug:
                print(f"Prediction error on image {i}: {e}")
            continue

    accuracy = correct_predictions / total_tested if total_tested > 0 else 0.0
    avg_inference_time = total_inference_time / total_tested if total_tested > 0 else 0.0
    inferences_per_second = 1000 / avg_inference_time if avg_inference_time > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"RESULTS for {os.path.basename(model_path)}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_tested})")
    print(f"Failed predictions: {len(failed_predictions)}")
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    print(f"Inferences per second: {inferences_per_second:.0f}")

    expected_failed = total_tested - correct_predictions
    if len(failed_predictions) != expected_failed:
        print(f"❌ CRITICAL ERROR: Failed count mismatch!")
        print(f"   Expected failed: {expected_failed}, Actual: {len(failed_predictions)}")
        if list_failed or save_failed:
            list_failed = False
            save_failed = False

    if list_failed and failed_predictions:
        generate_failed_predictions_csv(failed_predictions, output_dir)
    if save_failed and failed_predictions:
        save_failed_images(failed_predictions, output_dir)

    if all_predictions_lite:
        generate_confusion_matrix(all_predictions_lite, output_dir=params.OUTPUT_DIR)

    return accuracy, total_tested, avg_inference_time, inferences_per_second, failed_predictions, all_predictions_lite