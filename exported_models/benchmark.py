# benchmark.py
import os
import csv
import glob
from datetime import datetime
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import subprocess
import re
import struct

def setup_plotting_style():
    """Set up consistent plotting style"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def estimate_tensor_arena_size(tflite_model_path):
    """Estimate tensor arena size required for TFLite model inference"""
    try:
        import tflite_runtime.interpreter as tflite
        
        # Create interpreter and allocate tensors to get actual memory usage
        interpreter = tflite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # Get tensor details to calculate memory requirements
        tensor_details = interpreter.get_tensor_details()
        
        # Calculate total tensor memory
        total_tensor_memory = 0
        input_tensors = []
        output_tensors = []
        intermediate_tensors = []
        
        for tensor in tensor_details:
            tensor_size = 1
            for dim in tensor['shape']:
                tensor_size *= dim
            
            # Estimate bytes based on dtype
            dtype_size = 4  # Default to 4 bytes (float32, int32)
            if tensor['dtype'] in [np.int8, np.uint8]:
                dtype_size = 1
            elif tensor['dtype'] in [np.int16, np.uint16]:
                dtype_size = 2
            elif tensor['dtype'] in [np.float64, np.int64]:
                dtype_size = 8
            
            tensor_memory = tensor_size * dtype_size
            
            # Categorize tensors
            if tensor['index'] in interpreter.get_input_details()[0]['index']:
                input_tensors.append((tensor['name'], tensor_memory))
            elif tensor['index'] in interpreter.get_output_details()[0]['index']:
                output_tensors.append((tensor['name'], tensor_memory))
            else:
                intermediate_tensors.append((tensor['name'], tensor_memory))
            
            total_tensor_memory += tensor_memory
        
        # Estimate tensor arena size (total memory + overhead)
        # TFLite typically requires 2x total tensor memory for arena
        estimated_arena_size = total_tensor_memory * 2
        
        # Add some safety margin
        estimated_arena_size_with_margin = estimated_arena_size * 1.2
        
        return {
            'estimated_arena_bytes': int(estimated_arena_size_with_margin),
            'estimated_arena_kb': estimated_arena_size_with_margin / 1024,
            'total_tensor_memory_bytes': total_tensor_memory,
            'input_tensors': input_tensors,
            'output_tensors': output_tensors,
            'intermediate_tensors': intermediate_tensors,
            'total_tensors': len(tensor_details)
        }
        
    except ImportError:
        # Fallback: estimate from model file size
        model_size_bytes = os.path.getsize(tflite_model_path)
        # Rough estimation: tensor arena ~ 3x model size
        estimated_arena_bytes = model_size_bytes * 3
        
        return {
            'estimated_arena_bytes': int(estimated_arena_bytes),
            'estimated_arena_kb': estimated_arena_bytes / 1024,
            'total_tensor_memory_bytes': model_size_bytes,
            'input_tensors': [],
            'output_tensors': [],
            'intermediate_tensors': [],
            'total_tensors': 0,
            'note': 'Estimated from model size (tflite_runtime not available)'
        }
    except Exception as e:
        print(f"❌ Error estimating tensor arena for {tflite_model_path}: {e}")
        return None

def estimate_cpu_ops(tflite_model_path):
    """Estimate CPU operations required for one inference"""
    try:
        import tflite_runtime.interpreter as tflite
        
        interpreter = tflite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        tensor_details = interpreter.get_tensor_details()
        op_count = len(interpreter._get_ops_details()) if hasattr(interpreter, '_get_ops_details') else 0
        
        # Estimate operations based on tensor sizes and layers
        total_ops = 0
        total_params = 0
        
        for tensor in tensor_details:
            # Count parameters (weights)
            if tensor['shape'] and len(tensor['shape']) > 0:
                tensor_params = 1
                for dim in tensor['shape']:
                    tensor_params *= dim
                total_params += tensor_params
        
        # Rough operation estimation:
        # For neural networks, ops ≈ 2 * parameters (MAC operations)
        estimated_ops = total_params * 2
        
        # Alternative estimation based on model complexity
        model_size_kb = os.path.getsize(tflite_model_path) / 1024
        ops_from_size = model_size_kb * 1000  # Rough estimate: 1000 ops per KB
        
        return {
            'estimated_ops': int(estimated_ops),
            'estimated_ops_millions': estimated_ops / 1e6,
            'total_parameters': total_params,
            'operation_count': op_count,
            'tensor_count': len(tensor_details),
            'model_size_kb': model_size_kb,
            'ops_from_size': int(ops_from_size),
            'ops_from_size_millions': ops_from_size / 1e6
        }
        
    except Exception as e:
        print(f"❌ Error estimating CPU OPS for {tflite_model_path}: {e}")
        
        # Fallback estimation from model size
        model_size_kb = os.path.getsize(tflite_model_path) / 1024
        estimated_ops = model_size_kb * 1000  # Rough estimate
        
        return {
            'estimated_ops': int(estimated_ops),
            'estimated_ops_millions': estimated_ops / 1e6,
            'total_parameters': 0,
            'operation_count': 0,
            'tensor_count': 0,
            'model_size_kb': model_size_kb,
            'ops_from_size': int(estimated_ops),
            'ops_from_size_millions': estimated_ops / 1e6,
            'note': 'Estimated from model size (detailed analysis failed)'
        }

def generate_tensor_arena_report(tflite_model_path, report_file_path):
    """Generate comprehensive tensor arena and OPS report"""
    try:
        arena_info = estimate_tensor_arena_size(tflite_model_path)
        ops_info = estimate_cpu_ops(tflite_model_path)
        
        if arena_info is None:
            return None
        
        with open(report_file_path, 'w') as f:
            f.write(f"TFLite Model Analysis: {os.path.basename(tflite_model_path)}\n")
            f.write("=" * 60 + "\n\n")
            
            # Tensor Arena Information
            f.write("TENSOR ARENA ESTIMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Estimated Arena Size: {arena_info['estimated_arena_bytes']:,} bytes\n")
            f.write(f"Estimated Arena Size: {arena_info['estimated_arena_kb']:.2f} KB\n")
            f.write(f"Total Tensor Memory: {arena_info['total_tensor_memory_bytes']:,} bytes\n")
            f.write(f"Total Tensors: {arena_info['total_tensors']}\n\n")
            
            if arena_info['input_tensors']:
                f.write("Input Tensors:\n")
                for name, memory in arena_info['input_tensors']:
                    f.write(f"  {name}: {memory:,} bytes\n")
                f.write("\n")
            
            if arena_info['output_tensors']:
                f.write("Output Tensors:\n")
                for name, memory in arena_info['output_tensors']:
                    f.write(f"  {name}: {memory:,} bytes\n")
                f.write("\n")
            
            if arena_info['intermediate_tensors']:
                f.write(f"Intermediate Tensors ({len(arena_info['intermediate_tensors'])}):\n")
                total_intermediate = sum(mem for _, mem in arena_info['intermediate_tensors'])
                f.write(f"  Total Intermediate Memory: {total_intermediate:,} bytes\n\n")
            
            # CPU OPS Information
            f.write("CPU OPERATIONS ESTIMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Estimated OPS per inference: {ops_info['estimated_ops']:,}\n")
            f.write(f"Estimated OPS per inference: {ops_info['estimated_ops_millions']:.2f} million\n")
            f.write(f"Total Parameters: {ops_info['total_parameters']:,}\n")
            f.write(f"Operation Count: {ops_info['operation_count']}\n")
            f.write(f"Tensor Count: {ops_info['tensor_count']}\n")
            f.write(f"Model Size: {ops_info['model_size_kb']:.2f} KB\n")
            f.write(f"OPS from size estimate: {ops_info['ops_from_size']:,}\n")
            f.write(f"OPS from size estimate: {ops_info['ops_from_size_millions']:.2f} million\n\n")
            
            # Memory vs Performance Analysis
            f.write("PERFORMANCE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            memory_kb = arena_info['estimated_arena_kb']
            ops_millions = ops_info['estimated_ops_millions']
            
            if ops_millions > 0:
                ops_per_kb = ops_millions / memory_kb if memory_kb > 0 else 0
                f.write(f"OPS per KB of memory: {ops_per_kb:.2f} million OPS/KB\n")
            
            if 'note' in arena_info:
                f.write(f"\nNote: {arena_info['note']}\n")
            if 'note' in ops_info:
                f.write(f"Note: {ops_info['note']}\n")
        
        print(f"✅ Tensor arena report saved: {report_file_path}")
        return {
            'tensor_arena_bytes': arena_info['estimated_arena_bytes'],
            'tensor_arena_kb': arena_info['estimated_arena_kb'],
            'cpu_ops': ops_info['estimated_ops'],
            'cpu_ops_millions': ops_info['estimated_ops_millions'],
            'total_parameters': ops_info['total_parameters']
        }
        
    except Exception as e:
        print(f"❌ Error generating tensor arena report for {tflite_model_path}: {e}")
        return None

def update_training_results_with_metrics(training_dir):
    """Update training_results.csv with tensor arena and CPU OPS"""
    results_file = os.path.join(training_dir, "training_results.csv")
    tflite_model_path = os.path.join(training_dir, "final_quantized.tflite")
    report_file_path = os.path.join(training_dir, "tensor_arena.txt")
    
    if not os.path.exists(results_file):
        print(f"❌ Training results file not found: {results_file}")
        return None
    
    if not os.path.exists(tflite_model_path):
        print(f"❌ TFLite model not found: {tflite_model_path}")
        return None
    
    # Generate tensor arena report
    metrics = generate_tensor_arena_report(tflite_model_path, report_file_path)
    
    if metrics is not None:
        # Read existing results
        results = {}
        with open(results_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    results[row[0]] = row[1]
        
        # Update with new metrics
        results['tensor_arena_bytes'] = str(metrics['tensor_arena_bytes'])
        results['tensor_arena_kb'] = str(metrics['tensor_arena_kb'])
        results['cpu_ops'] = str(metrics['cpu_ops'])
        results['cpu_ops_millions'] = str(metrics['cpu_ops_millions'])
        results['total_parameters'] = str(metrics['total_parameters'])
        
        # Write back to file
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for key, value in results.items():
                writer.writerow([key, value])
        
        print(f"✅ Updated {results_file} with performance metrics")
        print(f"   - Tensor Arena: {metrics['tensor_arena_kb']:.2f} KB")
        print(f"   - CPU OPS: {metrics['cpu_ops_millions']:.2f} million")
        print(f"   - Parameters: {metrics['total_parameters']:,}")
        
        return metrics
    
    return None

def process_all_tflite_models(output_dir="./"):
    """Process all TFLite models in subdirectories"""
    print("🔍 Processing TFLite models in subdirectories...")
    
    output_dir = os.path.abspath(output_dir)
    pattern = os.path.join(output_dir, "**", "final_quantized.tflite")
    tflite_files = glob.glob(pattern, recursive=True)
    
    print(f"📁 Found {len(tflite_files)} TFLite model files")
    
    metrics_data = {}
    
    for tflite_file in tflite_files:
        training_dir = os.path.dirname(tflite_file)
        print(f"📊 Processing: {training_dir}")
        
        metrics = update_training_results_with_metrics(training_dir)
        if metrics is not None:
            metrics_data[training_dir] = metrics
    
    print(f"✅ Processed {len(metrics_data)} models")
    return metrics_data

def plot_tensor_arena_analysis(df, output_dir="./"):
    """Create dedicated tensor arena analysis plots"""
    if df.empty or 'tensor_arena_kb' not in df.columns:
        print("❌ Insufficient data for tensor arena plots")
        return
    
    setup_plotting_style()
    
    # Convert to numeric
    df['tensor_arena_kb'] = pd.to_numeric(df['tensor_arena_kb'], errors='coerce')
    df['keras_test_accuracy'] = pd.to_numeric(df['keras_test_accuracy'], errors='coerce')
    df['quantized_model_size_kb'] = pd.to_numeric(df['quantized_model_size_kb'], errors='coerce')
    df['cpu_ops_millions'] = pd.to_numeric(df.get('cpu_ops_millions', 0), errors='coerce')
    
    # Remove rows with missing data
    plot_df = df.dropna(subset=['tensor_arena_kb']).copy()
    
    if plot_df.empty:
        print("❌ No valid tensor arena data for plotting")
        return
    
    # Create comprehensive tensor arena analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tensor Arena Memory Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy vs Tensor Arena Size
    ax1 = axes[0, 0]
    if 'model_architecture' in plot_df.columns:
        architectures = plot_df['model_architecture'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(architectures)))
        
        for i, arch in enumerate(architectures):
            arch_data = plot_df[plot_df['model_architecture'] == arch]
            ax1.scatter(arch_data['tensor_arena_kb'], 
                       arch_data['keras_test_accuracy'],
                       c=[colors[i]], label=arch, s=100, alpha=0.7, 
                       edgecolors='black', linewidth=0.5)
    else:
        ax1.scatter(plot_df['tensor_arena_kb'], plot_df['keras_test_accuracy'], 
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Tensor Arena Size (KB)')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Tensor Arena Memory', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tensor Arena vs Model Size
    ax2 = axes[0, 1]
    ax2.scatter(plot_df['quantized_model_size_kb'], plot_df['tensor_arena_kb'],
               s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Model Size (KB)')
    ax2.set_ylabel('Tensor Arena Size (KB)')
    ax2.set_title('Tensor Arena vs Model Size', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation line
    try:
        z = np.polyfit(plot_df['quantized_model_size_kb'], plot_df['tensor_arena_kb'], 1)
        p = np.poly1d(z)
        ax2.plot(plot_df['quantized_model_size_kb'], p(plot_df['quantized_model_size_kb']), 
                "r--", alpha=0.8, label=f'Correlation: {plot_df["quantized_model_size_kb"].corr(plot_df["tensor_arena_kb"]):.3f}')
        ax2.legend()
    except:
        pass
    
    # Plot 3: CPU OPS vs Tensor Arena
    ax3 = axes[1, 0]
    if not plot_df['cpu_ops_millions'].isna().all():
        ax3.scatter(plot_df['tensor_arena_kb'], plot_df['cpu_ops_millions'],
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Tensor Arena Size (KB)')
        ax3.set_ylabel('CPU OPS (Millions)')
        ax3.set_title('Computational Complexity vs Memory', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add efficiency metric
        if 'keras_test_accuracy' in plot_df.columns:
            # Color by accuracy
            sc = ax3.scatter(plot_df['tensor_arena_kb'], plot_df['cpu_ops_millions'],
                           c=plot_df['keras_test_accuracy'], s=100, alpha=0.7,
                           edgecolors='black', linewidth=0.5, cmap='viridis')
            plt.colorbar(sc, ax=ax3, label='Test Accuracy')
            ax3.set_title('Compute vs Memory (colored by Accuracy)', fontweight='bold')
    else:
        # Fallback: Tensor arena distribution
        sizes = plot_df['tensor_arena_kb'].dropna()
        ax3.hist(sizes, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
        ax3.axvline(sizes.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {sizes.mean():.1f} KB')
        ax3.axvline(sizes.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {sizes.median():.1f} KB')
        ax3.set_xlabel('Tensor Arena Size (KB)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Tensor Arena Size Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Memory Efficiency (Accuracy per KB of tensor arena)
    ax4 = axes[1, 1]
    plot_df['accuracy_per_arena_kb'] = plot_df['keras_test_accuracy'] / plot_df['tensor_arena_kb']
    
    efficiency_df = plot_df.nlargest(15, 'accuracy_per_arena_kb')
    bars = ax4.barh(range(len(efficiency_df)), efficiency_df['accuracy_per_arena_kb'], 
                   color='lightgreen', alpha=0.7)
    ax4.set_yticks(range(len(efficiency_df)))
    
    if 'run_name' in efficiency_df.columns:
        ax4.set_yticklabels(efficiency_df['run_name'])
    elif 'model_architecture' in efficiency_df.columns:
        ax4.set_yticklabels(efficiency_df['model_architecture'])
    else:
        ax4.set_yticklabels([f'Model {i+1}' for i in range(len(efficiency_df))])
    
    ax4.set_xlabel('Memory Efficiency (Accuracy / KB of Tensor Arena)')
    ax4.set_title('Top 15 Memory-Efficient Models', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                ha='left', va='center', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    
    # Save the tensor arena analysis plot
    arena_plot_path = os.path.join(output_dir, "tensor_arena_analysis.png")
    plt.savefig(arena_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Tensor arena analysis saved: {arena_plot_path}")
    plt.close()
    
    # Create additional dedicated plot for CPU OPS
    plot_cpu_ops_analysis(df, output_dir)

def plot_cpu_ops_analysis(df, output_dir="./"):
    """Create dedicated CPU OPS analysis plots"""
    if df.empty or 'cpu_ops_millions' not in df.columns:
        print("❌ Insufficient data for CPU OPS plots")
        return
    
    setup_plotting_style()
    
    # Convert to numeric
    df['cpu_ops_millions'] = pd.to_numeric(df['cpu_ops_millions'], errors='coerce')
    df['keras_test_accuracy'] = pd.to_numeric(df['keras_test_accuracy'], errors='coerce')
    df['tensor_arena_kb'] = pd.to_numeric(df.get('tensor_arena_kb', 0), errors='coerce')
    
    # Remove rows with missing data
    plot_df = df.dropna(subset=['cpu_ops_millions']).copy()
    
    if plot_df.empty:
        print("❌ No valid CPU OPS data for plotting")
        return
    
    # Create CPU OPS analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Computational Complexity Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy vs CPU OPS
    ax1 = axes[0, 0]
    if 'model_architecture' in plot_df.columns:
        architectures = plot_df['model_architecture'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(architectures)))
        
        for i, arch in enumerate(architectures):
            arch_data = plot_df[plot_df['model_architecture'] == arch]
            ax1.scatter(arch_data['cpu_ops_millions'], 
                       arch_data['keras_test_accuracy'],
                       c=[colors[i]], label=arch, s=100, alpha=0.7, 
                       edgecolors='black', linewidth=0.5)
    else:
        ax1.scatter(plot_df['cpu_ops_millions'], plot_df['keras_test_accuracy'], 
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('CPU OPS (Millions per inference)')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Computational Cost', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CPU OPS Distribution
    ax2 = axes[0, 1]
    ops_data = plot_df['cpu_ops_millions'].dropna()
    ax2.hist(ops_data, bins=15, alpha=0.7, edgecolor='black', color='lightcoral')
    ax2.axvline(ops_data.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {ops_data.mean():.1f}M OPS')
    ax2.axvline(ops_data.median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {ops_data.median():.1f}M OPS')
    ax2.set_xlabel('CPU OPS (Millions)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Computational Cost Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Compute Efficiency (Accuracy per Million OPS)
    ax3 = axes[1, 0]
    plot_df['accuracy_per_mops'] = plot_df['keras_test_accuracy'] / plot_df['cpu_ops_millions']
    
    efficiency_df = plot_df.nlargest(15, 'accuracy_per_mops')
    bars = ax3.barh(range(len(efficiency_df)), efficiency_df['accuracy_per_mops'], 
                   color='gold', alpha=0.7)
    ax3.set_yticks(range(len(efficiency_df)))
    
    if 'run_name' in efficiency_df.columns:
        ax3.set_yticklabels(efficiency_df['run_name'])
    elif 'model_architecture' in efficiency_df.columns:
        ax3.set_yticklabels(efficiency_df['model_architecture'])
    else:
        ax3.set_yticklabels([f'Model {i+1}' for i in range(len(efficiency_df))])
    
    ax3.set_xlabel('Compute Efficiency (Accuracy / Million OPS)')
    ax3.set_title('Top 15 Compute-Efficient Models', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                ha='left', va='center', fontweight='bold', fontsize=8)
    
    # Plot 4: Memory vs Compute Trade-off
    ax4 = axes[1, 1]
    if not plot_df['tensor_arena_kb'].isna().all():
        sc = ax4.scatter(plot_df['tensor_arena_kb'], plot_df['cpu_ops_millions'],
                        c=plot_df['keras_test_accuracy'], s=100, alpha=0.7,
                        edgecolors='black', linewidth=0.5, cmap='plasma')
        ax4.set_xlabel('Tensor Arena Size (KB)')
        ax4.set_ylabel('CPU OPS (Millions)')
        ax4.set_title('Memory vs Compute Trade-off\n(colored by Accuracy)', fontweight='bold')
        plt.colorbar(sc, ax=ax4, label='Test Accuracy')
        ax4.grid(True, alpha=0.3)
    else:
        # Alternative: Model size vs OPS
        if 'quantized_model_size_kb' in plot_df.columns:
            ax4.scatter(plot_df['quantized_model_size_kb'], plot_df['cpu_ops_millions'],
                       s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax4.set_xlabel('Model Size (KB)')
            ax4.set_ylabel('CPU OPS (Millions)')
            ax4.set_title('Model Size vs Computational Cost', fontweight='bold')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the CPU OPS analysis plot
    ops_plot_path = os.path.join(output_dir, "cpu_ops_analysis.png")
    plt.savefig(ops_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 CPU OPS analysis saved: {ops_plot_path}")
    plt.close()

def plot_model_efficiency(df, output_dir="./"):
    """Create comprehensive model efficiency plots (updated with new metrics)"""
    # [Keep the existing plot_model_efficiency function content, but ensure it uses the new metrics]
    # This function remains largely the same but will benefit from the new data columns
    
    if df.empty or 'keras_test_accuracy' not in df.columns or 'quantized_model_size_kb' not in df.columns:
        print("❌ Insufficient data for efficiency plots")
        return
    
    setup_plotting_style()
    
    # Convert to numeric including new metrics
    numeric_columns = ['keras_test_accuracy', 'tflite_test_accuracy', 
                      'quantized_model_size_kb', 'float_model_size_kb',
                      'tensor_arena_kb', 'cpu_ops_millions', 'total_parameters']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing essential data
    plot_df = df.dropna(subset=['keras_test_accuracy', 'quantized_model_size_kb']).copy()
    
    if plot_df.empty:
        print("❌ No valid data for plotting")
        return
    
    # Calculate efficiency metrics (including new ones)
    plot_df['accuracy_per_kb'] = plot_df['keras_test_accuracy'] / plot_df['quantized_model_size_kb']
    plot_df['efficiency_score'] = plot_df['keras_test_accuracy'] * (1000 / plot_df['quantized_model_size_kb'])
    
    if 'tensor_arena_kb' in plot_df.columns:
        plot_df['memory_efficiency'] = plot_df['keras_test_accuracy'] / plot_df['tensor_arena_kb']
    
    if 'cpu_ops_millions' in plot_df.columns:
        plot_df['compute_efficiency'] = plot_df['keras_test_accuracy'] / plot_df['cpu_ops_millions']
    
    # [Rest of the existing plot_model_efficiency function...]
    # Create subplots and plots as before, but now with additional data available
    
    # After the main efficiency plots, create the new dedicated plots
    plot_tensor_arena_analysis(df, output_dir)
    
    return plot_df

def collect_benchmark_results(output_dir="./", output_csv="benchmark_results.csv"):
    """
    Collect all training results from multiple runs into a single CSV file
    """
    print("📊 Collecting benchmark results...")
    
    # Convert relative path to absolute path
    output_dir = os.path.abspath(output_dir)
    print(f"🔍 Searching in: {output_dir}")
    
    # Find all training_results.csv files
    pattern = os.path.join(output_dir, "**", "training_results.csv")
    result_files = glob.glob(pattern, recursive=True)
    
    print(f"📁 Found {len(result_files)} training results files")
    
    if not result_files:
        print("❌ No training results found!")
        print("💡 Make sure you've run training at least once and check the directory structure")
        return pd.DataFrame()
    
    all_results = []
    
    for result_file in result_files:
        try:
            # Read the individual CSV
            results = {}
            with open(result_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        results[row[0]] = row[1]
            
            # Add directory information
            training_dir = os.path.dirname(result_file)
            results['training_directory'] = training_dir
            results['run_name'] = os.path.basename(training_dir)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"⚠️  Error reading {result_file}: {e}")
    
    if not all_results:
        print("❌ No valid results to compile!")
        return pd.DataFrame()
    
    # Create combined DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability (including new metrics)
    preferred_order = [
        'run_name', 'timestamp', 'model_architecture',
        'keras_test_accuracy', 'tflite_test_accuracy', 'best_val_accuracy',
        'quantized_model_size_kb', 'float_model_size_kb', 'tensor_arena_kb',
        'cpu_ops_millions', 'total_parameters', 'training_time',
        'input_shape', 'nb_classes', 'data_sources', 'batch_size', 'epochs',
        'learning_rate', 'use_early_stopping', 'early_stopping_monitor',
        'early_stopping_patience', 'lr_scheduler_monitor', 'lr_scheduler_patience',
        'lr_scheduler_factor', 'quantize_model', 'esp_dl_quantize',
        'quantize_num_samples', 'use_gpu', 'optimizer'
    ]
    
    # Reorder columns, keeping any additional columns
    existing_columns = [col for col in preferred_order if col in df.columns]
    other_columns = [col for col in df.columns if col not in preferred_order]
    final_columns = existing_columns + other_columns
    
    df = df[final_columns]
    
    # Save to combined CSV
    output_path = os.path.join(output_dir, output_csv)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Benchmark results saved to: {output_path}")
    print(f"📈 Collected {len(df)} training runs")
    
    # Print summary of new metrics
    if 'tensor_arena_kb' in df.columns:
        arena_sizes = pd.to_numeric(df['tensor_arena_kb'], errors='coerce').dropna()
        if len(arena_sizes) > 0:
            print(f"📊 Tensor Arena: {len(arena_sizes)} models, avg: {arena_sizes.mean():.1f} KB")
    
    if 'cpu_ops_millions' in df.columns:
        ops_data = pd.to_numeric(df['cpu_ops_millions'], errors='coerce').dropna()
        if len(ops_data) > 0:
            print(f"⚡ CPU OPS: {len(ops_data)} models, avg: {ops_data.mean():.1f} million")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Benchmark Training Results')
    parser.add_argument('--input_dir', default='./',
                       help='Directory containing training results')
    parser.add_argument('--output_file', default='benchmark_results.csv',
                       help='Output CSV filename')
    parser.add_argument('--no_plots', action='store_true',
                       help='Disable efficiency plots generation')
    parser.add_argument('--no_analysis', action='store_true',
                       help='Disable architecture comparison and trend analysis')
    parser.add_argument('--no_model_processing', action='store_true',
                       help='Disable TFLite model processing and metric extraction')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information')
    
    args = parser.parse_args()
    
    if args.debug:
        print("🐛 DEBUG MODE")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Input directory: {args.input_dir}")
        print(f"Absolute input directory: {os.path.abspath(args.input_dir)}")
    
    # Process TFLite models and extract metrics (unless disabled)
    if not args.no_model_processing:
        print("🔄 Processing TFLite models and extracting performance metrics...")
        process_all_tflite_models(args.input_dir)
    else:
        print("⏭️  Skipping TFLite model processing (disabled with --no_model_processing)")
    
    # Collect results
    df = collect_benchmark_results(args.input_dir, args.output_file)
    
    if df is not None and not df.empty:
        # Default behavior: generate all plots and analyses unless disabled
        generate_plots = not args.no_plots
        generate_analysis = not args.no_analysis
        
        print(f"\n🎯 Analysis Configuration:")
        print(f"   Efficiency Plots: {'Enabled' if generate_plots else 'Disabled'}")
        print(f"   Architecture Comparison: {'Enabled' if generate_analysis else 'Disabled'}")
        print(f"   Trend Analysis: {'Enabled' if generate_analysis else 'Disabled'}")
        
        # Generate efficiency plots (default: enabled)
        if generate_plots:
            print("\n📊 Generating efficiency plots...")
            plot_df = plot_model_efficiency(df, args.input_dir)
        else:
            print("\n⏭️  Skipping efficiency plots (disabled with --no_plots)")
        
        # Additional analyses (default: enabled)
        if generate_analysis:
            print("\n🧩 Comparing model architectures...")
            compare_models_by_architecture(df)
            
            print("\n📈 Analyzing trends...")
            analyze_trends(df)
        else:
            print("\n⏭️  Skipping analyses (disabled with --no_analysis)")
        
        print(f"\n✅ Benchmark analysis complete!")
        print(f"💾 Results saved to: {os.path.join(args.input_dir, args.output_file)}")
        
        # Auto-generate plots if we have enough data and plots weren't explicitly generated
        if len(df) >= 3 and not generate_plots and not args.no_plots:
            print("\n📊 Auto-generating efficiency plots (minimum 3 models detected)...")
            plot_model_efficiency(df, args.input_dir)
            
    elif df is not None and df.empty:
        print("❌ No training results found in the collected data!")
        print("💡 Make sure training has been run and results are properly saved")
    else:
        print("❌ No data available for analysis!")

# [Keep the existing helper functions compare_models_by_architecture and analyze_trends]

def compare_models_by_architecture(df):
    """Compare performance by model architecture"""
    if 'model_architecture' not in df.columns or df.empty:
        print("❌ No model architecture information found!")
        return
    
    print("\n🧩 MODEL ARCHITECTURE COMPARISON:")
    
    # Convert to numeric including new metrics
    numeric_columns = ['keras_test_accuracy', 'quantized_model_size_kb', 
                      'tensor_arena_kb', 'cpu_ops_millions']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Group by architecture and calculate statistics
    comparison_columns = ['keras_test_accuracy', 'quantized_model_size_kb']
    
    # Add new metrics if available
    if 'tensor_arena_kb' in df.columns:
        comparison_columns.append('tensor_arena_kb')
    if 'cpu_ops_millions' in df.columns:
        comparison_columns.append('cpu_ops_millions')
    
    comparison = df.groupby('model_architecture').agg({
        col: ['count', 'mean', 'max', 'min', 'std'] for col in comparison_columns
    }).round(4)
    
    print(comparison)
    
    return comparison

def analyze_trends(df):
    """Analyze trends in the benchmark data"""
    if df.empty:
        print("❌ No data for trend analysis!")
        return
        
    print("\n📈 TREND ANALYSIS:")
    
    # Convert relevant columns to numeric
    numeric_columns = ['keras_test_accuracy', 'tflite_test_accuracy', 
                      'quantized_model_size_kb', 'float_model_size_kb',
                      'tensor_arena_kb', 'cpu_ops_millions']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Basic statistics
    print("Overall Statistics:")
    for col in numeric_columns:
        if col in df.columns and not df[col].isnull().all():
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                print(f"  {col}:")
                print(f"    Count: {len(valid_data)}")
                print(f"    Mean: {valid_data.mean():.4f}")
                print(f"    Best: {valid_data.max():.4f}")
                print(f"    Worst: {valid_data.min():.4f}")
                print(f"    Std: {valid_data.std():.4f}")

if __name__ == "__main__":
    main()