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
from collections import Counter
from datetime import datetime
import time

import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import parameters as params
from .predictor import TFLiteDigitPredictor, get_model_metadata
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


def test_model_on_dataset(model_path, num_test_images=0, debug=False,
                          use_all_datasets=True, collect_failed=False,
                          model_name=None):
    """
    Test a single TFLite model on the test dataset.

    Returns dict with accuracy, per-class accuracy, predictions, timing, etc.
    """
    predictor = TFLiteDigitPredictor(model_path)
    test_data = load_test_dataset_with_labels(
        num_samples=num_test_images,
        use_all_datasets=use_all_datasets,
    )

    if num_test_images > 0:
        test_data = test_data[:num_test_images]

    total = len(test_data)
    correct = 0
    class_correct = Counter()
    class_total = Counter()
    predictions = []
    failed_predictions = []
    inference_times = []

    for image, true_label, fname, is_augmented in tqdm(test_data, desc=f"Testing {model_name or os.path.basename(model_path)}"):
        t0 = time.time()
        output = predictor.predict(image, debug=debug)
        elapsed = time.time() - t0
        inference_times.append(elapsed)

        predicted_label = np.argmax(output[0])
        confidence = float(np.max(output[0]))

        class_total[true_label] += 1
        if predicted_label == true_label:
            correct += 1
            class_correct[true_label] += 1
        else:
            if collect_failed:
                failed_predictions.append({
                    'image': image,
                    'true_label': int(true_label),
                    'predicted_label': int(predicted_label),
                    'confidence': confidence,
                    'fname': fname,
                })

        predictions.append({
            'true_label': int(true_label),
            'predicted_label': int(predicted_label),
            'confidence': confidence,
            'fname': fname,
            'is_augmented': is_augmented,
        })

    accuracy = correct / total if total > 0 else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0

    per_class_accuracy = {}
    for cls in sorted(class_total.keys()):
        per_class_accuracy[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0

    return {
        'model_path': model_path,
        'model_name': model_name or os.path.basename(model_path),
        'total_images': total,
        'correct': correct,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'predictions': predictions,
        'failed_predictions': failed_predictions,
        'avg_inference_time': avg_inference_time,
        'inference_times': inference_times,
    }


def test_all_models(num_test_images=0, quantized_only=False, debug=False,
                    use_all_datasets=True, list_failed=False, save_failed=False,
                    output_dir=None, model_list=None, simulate_esp32=False,
                    iot_compat=True):
    """Test all available models and return results list."""
    from .data import get_all_models

    models = get_all_models(
        quantized_only=quantized_only,
        model_list=model_list,
        iot_compat=iot_compat,
    )

    if not models:
        logger.error("No models found to test!")
        return []

    results = []
    for model_info in models:
        logger.info(f"\nTesting model: {model_info['name']}")
        result = test_model_on_dataset(
            model_path=model_info['path'],
            num_test_images=num_test_images,
            debug=debug,
            use_all_datasets=use_all_datasets,
            collect_failed=(list_failed or save_failed),
            model_name=model_info['name'],
        )
        result['model_info'] = model_info
        results.append(result)

        acc = result['accuracy']
        logger.info(f"  Accuracy: {acc:.4f} ({result['correct']}/{result['total_images']})")

    return results


def generate_confusion_matrix(all_results, output_dir=None):
    """Generate a confusion matrix heatmap and per-class accuracy CSV."""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for result in all_results:
        model_name = result['model_name'].replace('.tflite', '')
        predictions = result['predictions']

        # Build confusion matrix
        n_classes = params.NB_CLASSES
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for p in predictions:
            cm[p['true_label'], p['predicted_label']] += 1

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
               xlabel='Predicted label', ylabel='True label',
               title=f'Confusion Matrix - {model_name}')

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Confusion matrix saved: {cm_path}")

        # Per-class accuracy CSV
        csv_path = os.path.join(output_dir, f'per_class_accuracy_{model_name}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Total', 'Correct', 'Accuracy'])
            for cls in range(n_classes):
                total = np.sum(cm[cls, :])
                correct = cm[cls, cls]
                acc = correct / total if total > 0 else 0
                writer.writerow([cls, total, correct, f'{acc:.4f}'])
        logger.info(f"Per-class accuracy saved: {csv_path}")


def generate_comparison_graphs(results, quantized_only=True, use_all_datasets=True,
                                output_dir=None, simulate_esp32=False):
    """Generate comparison graphs for benchmark results."""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if not results:
        logger.warning("No results to graph")
        return []

    df = pd.DataFrame(results)
    graph_paths = []

    # 1. Accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r['model_name'][:30] for r in results]
    accs = [r['accuracy'] * 100 for r in results]
    bars = ax.bar(range(len(names)), accs)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% target')
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    graph_paths.append(path)

    # 2. Inference time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    times = [r['avg_inference_time'] * 1000 for r in results]  # ms
    bars = ax.bar(range(len(names)), times)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Avg Inference Time (ms)')
    ax.set_title('Inference Speed Comparison')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{t:.1f}ms', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, 'inference_time_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    graph_paths.append(path)

    # 3. Accuracy vs Size scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = [r.get('model_info', {}).get('file_size_bytes', 0) / 1024 for r in results]
    scatter = ax.scatter(sizes, accs, c=times, s=100, cmap='viridis')
    for i, name in enumerate(names):
        ax.annotate(name[:20], (sizes[i], accs[i]), fontsize=7, alpha=0.7)
    ax.set_xlabel('Model Size (KB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Model Size (color = inference time)')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Inference Time (ms)')
    fig.tight_layout()
    path = os.path.join(output_dir, 'accuracy_vs_size.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    graph_paths.append(path)

    return graph_paths


def save_results_to_csv(results, quantized_only=True, use_all_images=True,
                        test_images_count=0, output_dir=None, update_csv=False):
    """Save results to CSV file."""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f'benchmark_results_{timestamp}.csv')

    rows = []
    for r in results:
        model_info = r.get('model_info', {})
        row = {
            'Model Name': r['model_name'],
            'Folder': model_info.get('folder', ''),
            'Accuracy': f"{r['accuracy']:.4f}",
            'Correct': r['correct'],
            'Total': r['total_images'],
            'Avg Inference Time (ms)': f"{r['avg_inference_time'] * 1000:.3f}",
            'File Size (KB)': f"{model_info.get('file_size_bytes', 0) / 1024:.1f}",
            'Input Shape': str(model_info.get('input_shape', '')),
            'Output Shape': str(model_info.get('output_shape', '')),
            'Input Dtype': str(model_info.get('input_dtype', '')),
            'Output Dtype': str(model_info.get('output_dtype', '')),
            'Is Quantized': model_info.get('is_quantized', False),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

    # Print summary table
    summary_data = []
    for r in results:
        summary_data.append([
            r['model_name'][:40],
            f"{r['accuracy']:.4f}",
            f"{r['avg_inference_time'] * 1000:.1f}ms",
            f"{r.get('model_info', {}).get('file_size_bytes', 0) / 1024:.0f}KB",
        ])
    headers = ['Model', 'Accuracy', 'Avg Time', 'Size']
    print("\n" + tabulate(summary_data, headers=headers, tablefmt='grid'))
    print(f"\nFull results: {csv_path}")

    return csv_path


def generate_markdown_report(csv_path, graph_paths, results, quantized_only=True,
                              use_all_datasets=True, test_images_count=0,
                              output_dir=None, simulate_esp32=False):
    """Generate a comprehensive Markdown report from CSV results and graphs."""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_path = os.path.join(output_dir, 'benchmark_report.md')

    with open(report_path, 'w') as f:
        f.write(f"# Benchmark Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write(f"**Configuration:** {params.NB_CLASSES} classes, "
                f"{'Grayscale' if params.USE_GRAYSCALE else 'RGB'}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Model | Accuracy | Avg Time | Size |\n")
        f.write("|-------|----------|----------|------|\n")
        for r in results:
            f.write(f"| {r['model_name']} | {r['accuracy']:.4f} | "
                    f"{r['avg_inference_time'] * 1000:.1f}ms | "
                    f"{r.get('model_info', {}).get('file_size_bytes', 0) / 1024:.0f}KB |\n")

        # Graphs
        if graph_paths:
            f.write("\n## Graphs\n\n")
            for gp in graph_paths:
                rel_path = os.path.relpath(gp, output_dir)
                f.write(f"![{os.path.basename(gp)}]({rel_path})\n\n")

        # IoT recommendation
        if results:
            df = pd.DataFrame(results)
            f.write("\n## IoT Recommendation\n\n")
            f.write(generate_iot_recommendation_section(df))

    logger.info(f"Report saved to: {report_path}")
    return report_path


def calculate_best_iot_model(df, accuracy_weight=0.5, size_weight=0.3,
                              speed_weight=0.2):
    """Calculate the best IoT model based on weighted scoring."""
    if df.empty:
        return None

    # Normalize metrics
    max_acc = df['accuracy'].max()
    min_size = df['model_info'].apply(lambda x: x.get('file_size_bytes', 0)).min()
    min_time = df['avg_inference_time'].min()

    scores = []
    for _, row in df.iterrows():
        acc_score = row['accuracy'] / max_acc if max_acc > 0 else 0
        size = row['model_info'].get('file_size_bytes', 0)
        size_score = min_size / size if size > 0 else 0
        time_score = min_time / row['avg_inference_time'] if row['avg_inference_time'] > 0 else 0

        total = (accuracy_weight * acc_score +
                 size_weight * size_score +
                 speed_weight * time_score)
        scores.append(total)

    best_idx = np.argmax(scores)
    return df.iloc[best_idx] if not df.empty else None


def generate_iot_recommendation_section(df):
    """Generate IoT recommendation section text."""
    best = calculate_best_iot_model(df)
    if best is None:
        return "No models available for IoT recommendation.\n"

    text = []
    text.append(f"### Recommended Model for IoT Deployment\n\n")
    text.append(f"**{best['model_name']}**\n\n")
    text.append(f"- Accuracy: {best['accuracy']:.4f}\n")
    text.append(f"- Inference Time: {best['avg_inference_time'] * 1000:.1f}ms\n")
    text.append(f"- Model Size: {best.get('model_info', {}).get('file_size_bytes', 0) / 1024:.0f}KB\n\n")

    # List all models ranked
    text.append("### All Models Ranked\n\n")
    text.append("| Rank | Model | Score | Accuracy | Time (ms) | Size (KB) |\n")
    text.append("|------|-------|-------|----------|-----------|-----------|\n")

    scores = []
    for _, row in df.iterrows():
        acc_score = row['accuracy'] / df['accuracy'].max()
        min_size = df['model_info'].apply(lambda x: x.get('file_size_bytes', 0)).min()
        size = row['model_info'].get('file_size_bytes', 0)
        size_score = min_size / size if size > 0 else 0
        min_time = df['avg_inference_time'].min()
        time_score = min_time / row['avg_inference_time'] if row['avg_inference_time'] > 0 else 0
        total = 0.5 * acc_score + 0.3 * size_score + 0.2 * time_score
        scores.append(total)

    ranked_indices = np.argsort(scores)[::-1]
    for rank, idx in enumerate(ranked_indices, 1):
        row = df.iloc[idx]
        text.append(f"| {rank} | {row['model_name'][:40]} | {scores[idx]:.3f} | "
                f"{row['accuracy']:.4f} | "
                f"{row['avg_inference_time'] * 1000:.1f} | "
                f"{row.get('model_info', {}).get('file_size_bytes', 0) / 1024:.0f} |\n")

    return "".join(text)


def save_failed_images(failed_predictions, output_dir):
    """Save failed prediction images to directory for manual review."""
    if not failed_predictions:
        logger.info("No failed predictions to save")
        return

    failed_dir = os.path.join(output_dir, 'failed_predictions')
    os.makedirs(failed_dir, exist_ok=True)

    for fp in failed_predictions:
        img = fp['image']
        fname = f"true_{fp['true_label']}_pred_{fp['predicted_label']}_{fp['fname']}"
        save_path = os.path.join(failed_dir, fname)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze()
        cv2_img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        cv2.imwrite(save_path, cv2_img)

    logger.info(f"Saved {len(failed_predictions)} failed predictions to {failed_dir}")


def generate_failed_predictions_csv(failed_predictions, output_dir):
    """Generate CSV file with details of failed predictions."""
    if not failed_predictions:
        return

    csv_path = os.path.join(output_dir, 'failed_predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'True Label', 'Predicted Label', 'Confidence'])
        for fp in failed_predictions:
            writer.writerow([
                fp['fname'],
                fp['true_label'],
                fp['predicted_label'],
                f"{fp['confidence']:.4f}",
            ])
    logger.info(f"Failed predictions CSV saved: {csv_path}")


def test_single_model(model_path, num_test_images=0, debug=False,
                      use_all_datasets=True, list_failed=False,
                      save_failed=False, output_dir=None,
                      model_name=None, simulate_esp32=False):
    """Test a single model and print results."""
    result = test_model_on_dataset(
        model_path=model_path,
        num_test_images=num_test_images,
        debug=debug,
        use_all_datasets=use_all_datasets,
        collect_failed=(list_failed or save_failed),
        model_name=model_name,
    )

    print(f"\nResults for {result['model_name']}:")
    print(f"  Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total_images']})")
    print(f"  Avg inference time: {result['avg_inference_time'] * 1000:.3f}ms")

    if result['per_class_accuracy']:
        print("\nPer-class accuracy:")
        for cls, acc in sorted(result['per_class_accuracy'].items()):
            print(f"  Class {cls}: {acc:.4f}")

    if list_failed and result['failed_predictions']:
        print(f"\nFailed predictions ({len(result['failed_predictions'])}):")
        for fp in result['failed_predictions'][:20]:
            print(f"  {fp['fname']}: true={fp['true_label']}, pred={fp['predicted_label']}, conf={fp['confidence']:.4f}")

    if save_failed and result['failed_predictions']:
        save_dir = output_dir or params.OUTPUT_DIR
        save_failed_images(result['failed_predictions'], save_dir)
        generate_failed_predictions_csv(result['failed_predictions'], save_dir)

    return result
