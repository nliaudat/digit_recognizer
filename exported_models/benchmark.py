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

def setup_plotting_style():
    """Set up consistent plotting style"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def plot_model_efficiency(df, output_dir="./"):
    """Create comprehensive model efficiency plots"""
    if df.empty or 'keras_test_accuracy' not in df.columns or 'quantized_model_size_kb' not in df.columns:
        print("❌ Insufficient data for efficiency plots")
        return
    
    setup_plotting_style()
    
    # Convert to numeric
    df['keras_test_accuracy'] = pd.to_numeric(df['keras_test_accuracy'], errors='coerce')
    df['tflite_test_accuracy'] = pd.to_numeric(df['tflite_test_accuracy'], errors='coerce')
    df['quantized_model_size_kb'] = pd.to_numeric(df['quantized_model_size_kb'], errors='coerce')
    df['params'] = pd.to_numeric(df.get('params', 0), errors='coerce')
    
    # Remove rows with missing data
    plot_df = df.dropna(subset=['keras_test_accuracy', 'quantized_model_size_kb']).copy()
    
    if plot_df.empty:
        print("❌ No valid data for plotting")
        return
    
    # Calculate efficiency metrics
    plot_df['accuracy_per_kb'] = plot_df['keras_test_accuracy'] / plot_df['quantized_model_size_kb']
    plot_df['efficiency_score'] = plot_df['keras_test_accuracy'] * (1000 / plot_df['quantized_model_size_kb'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Efficiency Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy vs Model Size (main efficiency plot)
    ax1 = axes[0, 0]
    if 'model_architecture' in plot_df.columns:
        architectures = plot_df['model_architecture'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(architectures)))
        
        for i, arch in enumerate(architectures):
            arch_data = plot_df[plot_df['model_architecture'] == arch]
            ax1.scatter(arch_data['quantized_model_size_kb'], 
                       arch_data['keras_test_accuracy'],
                       c=[colors[i]], label=arch, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    else:
        ax1.scatter(plot_df['quantized_model_size_kb'], plot_df['keras_test_accuracy'], 
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Model Size (KB)')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Model Size', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add efficiency lines (pareto frontier)
    try:
        # Find Pareto frontier
        sorted_df = plot_df.sort_values('quantized_model_size_kb')
        pareto_points = []
        max_acc = -1
        
        for _, row in sorted_df.iterrows():
            if row['keras_test_accuracy'] > max_acc:
                pareto_points.append((row['quantized_model_size_kb'], row['keras_test_accuracy']))
                max_acc = row['keras_test_accuracy']
        
        if len(pareto_points) > 1:
            pareto_x, pareto_y = zip(*pareto_points)
            ax1.plot(pareto_x, pareto_y, 'r--', alpha=0.7, linewidth=2, label='Pareto Frontier')
    except:
        pass
    
    # Plot 2: Efficiency Score Ranking
    ax2 = axes[0, 1]
    efficiency_df = plot_df.nlargest(15, 'efficiency_score')  # Top 15 most efficient
    bars = ax2.barh(range(len(efficiency_df)), efficiency_df['efficiency_score'])
    ax2.set_yticks(range(len(efficiency_df)))
    
    if 'run_name' in efficiency_df.columns:
        ax2.set_yticklabels(efficiency_df['run_name'])
    elif 'model_architecture' in efficiency_df.columns:
        ax2.set_yticklabels(efficiency_df['model_architecture'])
    else:
        ax2.set_yticklabels([f'Model {i+1}' for i in range(len(efficiency_df))])
    
    ax2.set_xlabel('Efficiency Score (Accuracy × 1000 / Size)')
    ax2.set_title('Top 15 Most Efficient Models', fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                ha='left', va='center', fontweight='bold')
    
    # Plot 3: Accuracy vs Parameters (if available)
    ax3 = axes[1, 0]
    if 'params' in plot_df.columns and not plot_df['params'].isna().all():
        ax3.scatter(plot_df['params'], plot_df['keras_test_accuracy'], 
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Number of Parameters')
        ax3.set_ylabel('Test Accuracy')
        ax3.set_title('Accuracy vs Model Complexity', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    else:
        # Alternative: Training time vs accuracy
        if 'training_time' in plot_df.columns:
            # Convert time strings to numeric (basic conversion)
            time_seconds = []
            for time_str in plot_df['training_time']:
                try:
                    if ':' in str(time_str):
                        parts = str(time_str).split(':')
                        if len(parts) == 3:  # HH:MM:SS
                            seconds = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                        else:  # MM:SS
                            seconds = int(parts[0])*60 + int(parts[1])
                        time_seconds.append(seconds)
                    else:
                        time_seconds.append(float(time_str))
                except:
                    time_seconds.append(np.nan)
            
            if len([x for x in time_seconds if not np.isnan(x)]) > 0:
                plot_df['training_seconds'] = time_seconds
                ax3.scatter(plot_df['training_seconds'], plot_df['keras_test_accuracy'], 
                           s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
                ax3.set_xlabel('Training Time (seconds)')
                ax3.set_ylabel('Test Accuracy')
                ax3.set_title('Accuracy vs Training Time', fontweight='bold')
                ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Size Distribution
    ax4 = axes[1, 1]
    sizes = plot_df['quantized_model_size_kb'].dropna()
    if not sizes.empty:
        ax4.hist(sizes, bins=15, alpha=0.7, edgecolor='black')
        ax4.axvline(sizes.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {sizes.mean():.1f} KB')
        ax4.axvline(sizes.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {sizes.median():.1f} KB')
        ax4.set_xlabel('Model Size (KB)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Model Size Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the main efficiency plot
    efficiency_plot_path = os.path.join(output_dir, "model_efficiency_analysis.png")
    plt.savefig(efficiency_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Efficiency plot saved: {efficiency_plot_path}")
    plt.close()
    
    # Create additional specialized plots
    create_specialized_plots(plot_df, output_dir)
    
    return plot_df

def create_specialized_plots(df, output_dir):
    """Create additional specialized efficiency plots"""
    setup_plotting_style()
    
    # Plot 1: Architecture comparison
    if 'model_architecture' in df.columns and len(df['model_architecture'].unique()) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Architecture vs Accuracy
        arch_data = df.groupby('model_architecture').agg({
            'keras_test_accuracy': ['mean', 'std', 'count'],
            'quantized_model_size_kb': 'mean',
            'efficiency_score': 'mean'
        }).round(4)
        
        arch_data.columns = ['_'.join(col).strip() for col in arch_data.columns.values]
        arch_data = arch_data.sort_values('efficiency_score_mean', ascending=False)
        
        # Plot accuracy by architecture
        architectures = arch_data.index
        y_pos = np.arange(len(architectures))
        
        ax1.barh(y_pos, arch_data['keras_test_accuracy_mean'], 
                xerr=arch_data['keras_test_accuracy_std'], 
                alpha=0.7, edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(architectures)
        ax1.set_xlabel('Test Accuracy')
        ax1.set_title('Accuracy by Model Architecture', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot efficiency by architecture
        ax2.barh(y_pos, arch_data['efficiency_score_mean'], alpha=0.7, edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(architectures)
        ax2.set_xlabel('Efficiency Score')
        ax2.set_title('Efficiency by Model Architecture', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        arch_plot_path = os.path.join(output_dir, "architecture_comparison.png")
        plt.savefig(arch_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Architecture comparison saved: {arch_plot_path}")
        plt.close()
    
    # Plot 2: Time series of model improvements (if timestamp available)
    if 'timestamp' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
            time_df = df.dropna(subset=['datetime']).sort_values('datetime')
            
            if len(time_df) > 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(time_df['datetime'], time_df['keras_test_accuracy'], 
                       'o-', linewidth=2, markersize=8, label='Accuracy')
                ax.set_xlabel('Time')
                ax.set_ylabel('Test Accuracy')
                ax.set_title('Model Performance Over Time', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Format x-axis
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                time_plot_path = os.path.join(output_dir, "performance_timeline.png")
                plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
                print(f"📊 Performance timeline saved: {time_plot_path}")
                plt.close()
        except:
            pass
    
    # Plot 3: Correlation heatmap
    numeric_columns = ['keras_test_accuracy', 'tflite_test_accuracy', 
                      'quantized_model_size_kb', 'float_model_size_kb']
    if 'params' in df.columns:
        numeric_columns.append('params')
    
    numeric_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(numeric_df) > 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax.set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values as text
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Feature Correlation Matrix', fontweight='bold')
        plt.colorbar(im)
        plt.tight_layout()
        
        corr_plot_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Correlation heatmap saved: {corr_plot_path}")
        plt.close()

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
    
    # Reorder columns for better readability
    preferred_order = [
        'run_name', 'timestamp', 'model_architecture',
        'keras_test_accuracy', 'tflite_test_accuracy', 'best_val_accuracy',
        'quantized_model_size_kb', 'float_model_size_kb', 'training_time',
        'input_shape', 'nb_classes', 'data_sources', 'batch_size', 'epochs',
        'learning_rate', 'use_early_stopping', 'early_stopping_monitor',
        'early_stopping_patience', 'lr_scheduler_monitor', 'lr_scheduler_patience',
        'lr_scheduler_factor', 'quantize_model', 'esp_dl_quantize',
        'quantize_num_samples', 'use_gpu'
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
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information')
    
    args = parser.parse_args()
    
    if args.debug:
        print("🐛 DEBUG MODE")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Input directory: {args.input_dir}")
        print(f"Absolute input directory: {os.path.abspath(args.input_dir)}")
    
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

# Add the missing helper functions
def compare_models_by_architecture(df):
    """Compare performance by model architecture"""
    if 'model_architecture' not in df.columns or df.empty:
        print("❌ No model architecture information found!")
        return
    
    print("\n🧩 MODEL ARCHITECTURE COMPARISON:")
    
    # Convert to numeric
    df['keras_test_accuracy'] = pd.to_numeric(df['keras_test_accuracy'], errors='coerce')
    df['quantized_model_size_kb'] = pd.to_numeric(df['quantized_model_size_kb'], errors='coerce')
    
    # Group by architecture and calculate statistics
    comparison = df.groupby('model_architecture').agg({
        'keras_test_accuracy': ['count', 'mean', 'max', 'min', 'std'],
        'quantized_model_size_kb': ['mean', 'min', 'max', 'std'],
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
                      'quantized_model_size_kb', 'float_model_size_kb']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Basic statistics
    print("Overall Statistics:")
    for col in numeric_columns:
        if col in df.columns and not df[col].isnull().all():
            valid_data = df[col].dropna()
            print(f"  {col}:")
            print(f"    Count: {len(valid_data)}")
            print(f"    Mean: {valid_data.mean():.4f}")
            print(f"    Best: {valid_data.max():.4f}")
            print(f"    Worst: {valid_data.min():.4f}")
            print(f"    Std: {valid_data.std():.4f}")

if __name__ == "__main__":
    main()