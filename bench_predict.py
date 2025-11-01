# bench_predict.py
import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from utils.preprocess import predict_single_image
import parameters as params
from tabulate import tabulate
import glob
from tqdm import tqdm
import csv
from datetime import datetime
import time
import matplotlib.pyplot as plt
from pathlib import Path
from utils.multi_source_loader import MultiSourceDataLoader, load_combined_dataset

class TFLiteDigitPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model()
    
    def load_model(self):
        """Load TFLite model"""
        print(f"Loading TFLite model: {self.model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input type: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def predict(self, image, debug=False):
        """Predict digit from image using TFLite"""
        # Preprocess image - use silent mode to avoid debug outputs
        processed_image = predict_single_image(image)
        
        # Handle channel mismatch - if model expects 3 channels but we have 1
        expected_channels = self.input_details[0]['shape'][3]
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 1 and expected_channels == 3:
            processed_image = np.repeat(processed_image, 3, axis=2)
        
        # Add batch dimension if not already present
        if len(processed_image.shape) == 3:
            input_data = np.expand_dims(processed_image, axis=0)
        else:
            input_data = processed_image
        
        # Handle quantization if needed
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = input_data.astype(np.uint8)
        elif self.input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            if input_data.dtype == np.float32:
                input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
            else:
                input_data = input_data.astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)
        
        # Verify shape matches expected input shape
        expected_shape = self.input_details[0]['shape']
        if input_data.shape != tuple(expected_shape):
            if input_data.size == np.prod(expected_shape):
                input_data = input_data.reshape(expected_shape)
            else:
                return -1, 0.0, np.zeros(self.output_details[0]['shape'][-1], dtype=np.float32)
        
        try:
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Handle output quantization if needed
            if self.output_details[0]['dtype'] in [np.uint8, np.int8]:
                output_scale, output_zero_point = self.output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            # Get prediction and confidence
            prediction = np.argmax(output_data[0])
            confidence = np.max(output_data[0])
            
            return prediction, confidence, output_data[0]
        
        except Exception as e:
            return -1, 0.0, np.zeros(self.output_details[0]['shape'][-1], dtype=np.float32)

# def predict_single_image_silent(image):
    # """
    # Silent version of predict_single_image that doesn't print debug outputs
    # """
    # # Copy the preprocessing logic from preprocess.py but without prints
    # target_size = (params.INPUT_WIDTH, params.INPUT_HEIGHT)
    # grayscale = params.USE_GRAYSCALE
    
    # # Resize to target size
    # image = cv2.resize(image, target_size)
    
    # # Convert to grayscale if required
    # if grayscale and len(image.shape) == 3:
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # elif not grayscale and len(image.shape) == 2:
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # # Add channel dimension if missing
    # if len(image.shape) == 2:
        # image = np.expand_dims(image, axis=-1)
    
    # # Apply the same preprocessing logic as preprocess_images but silently
    # if params.QUANTIZE_MODEL:
        # if params.ESP_DL_QUANTIZE:
            # # ESP-DL INT8 quantization: Use UINT8 [0, 255]
            # if image.dtype != np.uint8:
                # if image.max() <= 1.0:
                    # image = (image * 255).astype(np.uint8)
                # else:
                    # image = image.astype(np.uint8)
        # else:
            # # Standard TFLite UINT8 quantization: Use UINT8 [0, 255]
            # if image.dtype != np.uint8:
                # if image.max() <= 1.0:
                    # image = (image * 255).astype(np.uint8)
                # else:
                    # image = image.astype(np.uint8)
    # else:
        # # No quantization: Use float32 [0, 1]
        # if image.dtype != np.float32:
            # image = image.astype(np.float32)
        # if image.max() > 1.0:
            # image = image / 255.0
    
    # return image

def load_image_from_path(image_path, input_channels):
    """Load image from specified path based on model's input requirements"""
    if not os.path.exists(image_path):
        return None
    
    # Load image based on model's input requirements
    if input_channels == 1:
        # Model expects grayscale - load as 2D array
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Model expects color (RGB) - load as 3D array with 3 channels
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def find_model_path(model_name=None):
    """Find the model path based on model name"""
    # Look for training directories - exclude test_results and other non-training dirs
    all_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if os.path.isdir(os.path.join(params.OUTPUT_DIR, d))]
    
    # Filter out non-training directories
    training_dirs = []
    for dir_name in all_dirs:
        dir_path = os.path.join(params.OUTPUT_DIR, dir_name)
        # Check if this directory contains .tflite files and is likely a training directory
        tflite_files = [f for f in os.listdir(dir_path) if f.endswith('.tflite')]
        if tflite_files and not dir_name.startswith('test_results'):
            training_dirs.append(dir_name)
    
    if not training_dirs:
        print("No training directories with TFLite models found.")
        print(f"Please check if models exist in: {params.OUTPUT_DIR}")
        return None
    
    if model_name:
        # Remove .tflite extension if present for easier matching
        model_name_clean = model_name.replace('.tflite', '')
        
        # First, check if model_name matches a training directory
        matching_dirs = [d for d in training_dirs if model_name_clean in d]
        if matching_dirs:
            # Use the best matching directory (exact match first, then partial)
            best_match = None
            for dir_name in matching_dirs:
                if dir_name == model_name_clean:
                    best_match = dir_name
                    break
            if not best_match and matching_dirs:
                best_match = matching_dirs[0]  # Use first partial match
            
            training_path = os.path.join(params.OUTPUT_DIR, best_match)
            tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
            
            if tflite_files:
                # Prefer quantized models
                quantized_models = [f for f in tflite_files if 'quantized' in f.lower()]
                if quantized_models:
                    model_path = os.path.join(training_path, quantized_models[0])
                    return model_path
                else:
                    model_path = os.path.join(training_path, tflite_files[0])
                    return model_path
        
        # If no directory match, search for specific model files
        for training_dir in sorted(training_dirs, reverse=True):
            training_path = os.path.join(params.OUTPUT_DIR, training_dir)
            
            # Check for exact model file matches
            tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
            for model_file in tflite_files:
                model_file_clean = model_file.replace('.tflite', '')
                
                # Exact match or partial match
                if (model_name_clean == model_file_clean or 
                    model_name_clean in model_file_clean or
                    model_name == model_file):
                    
                    model_path = os.path.join(training_path, model_file)
                    return model_path
        
        print(f"Model or directory '{model_name}' not found in any training directory.")
        return None
    else:
        # Use default behavior - latest training directory
        latest_training = sorted(training_dirs)[-1]
        latest_dir_path = os.path.join(params.OUTPUT_DIR, latest_training)
        
        # Look for any .tflite file in the latest directory
        tflite_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.tflite')]
        
        if tflite_files:
            # Prefer quantized models if available
            quantized_models = [f for f in tflite_files if 'quantized' in f.lower()]
            if quantized_models:
                model_path = os.path.join(latest_dir_path, quantized_models[0])
                return model_path
            else:
                model_path = os.path.join(latest_dir_path, tflite_files[0])
                return model_path
        
        print(f"No TFLite model found in: {latest_dir_path}")
        return None

def get_model_parameters_count(model_path):
    """Get the number of parameters in a TFLite model"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        total_params = 0
        for tensor in interpreter.get_tensor_details():
            # Count only non-constant tensors (weights and biases)
            if 'buffer' not in tensor or tensor['buffer'] == 0:
                shape = tensor['shape']
                if shape is not None and len(shape) > 0:
                    params_in_tensor = np.prod(shape)
                    total_params += params_in_tensor
        
        return total_params
    except Exception as e:
        return 0

def is_quantized_model(model_path):
    """Check if a model is quantized by examining its input type"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        input_dtype = input_details[0]['dtype']
        
        # Quantized models typically use int8 or uint8
        return input_dtype in [np.int8, np.uint8]
    except:
        return False

def get_all_models(quantized_only=False):
    """Get all available models with parameters count - with error handling"""
    # Look for training directories - exclude test_results and other non-training dirs
    all_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if os.path.isdir(os.path.join(params.OUTPUT_DIR, d))]
    
    # Filter out non-training directories
    training_dirs = []
    for dir_name in all_dirs:
        dir_path = os.path.join(params.OUTPUT_DIR, dir_name)
        # Check if this directory contains .tflite files and is likely a training directory
        tflite_files = [f for f in os.listdir(dir_path) if f.endswith('.tflite')]
        if tflite_files and not dir_name.startswith('test_results'):
            training_dirs.append(dir_name)
    
    all_models = []
    
    for training_dir in training_dirs:
        training_path = os.path.join(params.OUTPUT_DIR, training_dir)
        
        # Look for any .tflite files in the directory
        tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
        
        for model_file in tflite_files:
            model_path = os.path.join(training_path, model_file)
            
            # Skip if file doesn't exist or is empty
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                continue
                
            # Verify the model can be loaded
            if not is_valid_tflite_model(model_path):
                continue
                
            if quantized_only and not is_quantized_model(model_path):
                continue
                
            try:
                model_size = os.path.getsize(model_path) / 1024
                model_type = "quantized" if is_quantized_model(model_path) else "float"
                parameters_count = get_model_parameters_count(model_path)
                
                all_models.append({
                    'path': model_path,
                    'name': model_file,
                    'directory': training_dir,
                    'size_kb': model_size,
                    'type': model_type,
                    'parameters': parameters_count
                })
                
            # except Exception:
                # continue
            except Exception as e:
                print(f"⚠️  Error processing model {training_dir}/{model_file}: {e}")  
                continue
    
    # Remove duplicates and sort by directory name
    unique_models = {}
    for model in all_models:
        key = f"{model['directory']}/{model['name']}"
        if key not in unique_models:
            unique_models[key] = model
    
    return list(unique_models.values())
    
def is_valid_tflite_model(model_path):
    """Check if a TFLite model file is valid and can be loaded"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return True
    except Exception as e:
        print(f"❌ Invalid TFLite model {os.path.basename(model_path)}: {e}")
        return False

def load_test_dataset_with_labels(num_samples=100, use_all_datasets=False):
    """
    Load test dataset with proper labels using multi_source_loader
    Returns: list of (image_array, true_label) tuples
    """
    print("📊 Loading test dataset with labels...")
    
    # Use the multi_source_loader to get properly labeled data
    loader = MultiSourceDataLoader()
    images, labels = loader.load_all_sources()
    
    if len(images) == 0:
        print("❌ No data loaded from any source")
        return []
    
    # Combine images and labels
    test_data = list(zip(images, labels))
    
    # Shuffle and limit samples
    np.random.shuffle(test_data)
    
    if not use_all_datasets and len(test_data) > num_samples:
        test_data = test_data[:num_samples]
    
    print(f"  Using {len(test_data)} test samples")
    return test_data

def test_model_on_dataset(model_path, num_test_images=100, debug=False, use_all_datasets=False):
    """Test a model on random images from dataset and return accuracy and performance metrics"""
    predictor = TFLiteDigitPredictor(model_path)
    correct_predictions = 0
    total_tested = 0
    total_inference_time = 0.0
    
    # Load test data with proper labels
    test_data = load_test_dataset_with_labels(num_test_images, use_all_datasets)
    
    if not test_data:
        print("❌ No test data available")
        return 0.0, 0, 0.0, 0.0
    
    # Warm-up run to avoid cold start timing issues
    if len(test_data) > 0:
        warmup_image, _ = test_data[0]
        if warmup_image is not None:
            try:
                predictor.predict(warmup_image, debug=False)
            except:
                pass
    
    # Test with progress bar (only if not in debug mode)
    if debug:
        # Debug mode - no progress bar, show individual results
        test_iterator = test_data
        print(f"Testing {len(test_data)} images in debug mode...")
    else:
        # Normal mode - use progress bar
        test_iterator = tqdm(test_data, desc=f"Testing {os.path.basename(model_path)}", leave=False)
    
    for image, true_label in test_iterator:
        if image is None:
            continue
        
        # Predict with timing
        try:
            start_time = time.perf_counter()
            prediction, confidence, _ = predictor.predict(image, debug=debug)
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            total_inference_time += inference_time
            
            # Check if prediction matches true label
            if prediction == true_label:
                correct_predictions += 1
                if debug:
                    print(f"✓ Correct: {prediction} (true: {true_label}, confidence: {confidence:.3f})")
            else:
                if debug:
                    print(f"✗ Wrong: {prediction} (true: {true_label}, confidence: {confidence:.3f})")
            
            total_tested += 1
            
        except Exception as e:
            if debug:
                print(f"Prediction error: {e}")
            continue
    
    # Calculate performance metrics
    accuracy = correct_predictions / total_tested if total_tested > 0 else 0.0
    
    avg_inference_time = total_inference_time / total_tested if total_tested > 0 else 0.0
    inferences_per_second = 1000 / avg_inference_time if avg_inference_time > 0 else 0.0
    
    if debug:
        print(f"Final accuracy: {accuracy:.3f} ({correct_predictions}/{total_tested})")
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Inferences per second: {inferences_per_second:.0f}")
    
    return accuracy, total_tested, avg_inference_time, inferences_per_second

def generate_comparison_graphs(results, quantized_only=False, use_all_datasets=False):
    """Generate separate comparison graphs for the benchmark results"""
    # Create graphs directory
    graphs_dir = os.path.join(params.OUTPUT_DIR, "test_results", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_suffix = "quantized" if quantized_only else "all"
    dataset_suffix = "full" if use_all_datasets else "sampled"
    
    # Prepare data for plotting
    labels = []  # Combined directory + model name for labels
    directories = []
    accuracies = []
    inferences_per_second = []
    sizes_kb = []
    parameters = []
    model_names = []
    
    for result in results:
        dir_name = result['Directory']
        model_name = result['Model']
        # Create unique label combining directory and model
        label = f"{dir_name}\n{model_name.replace('.tflite', '')}"
        labels.append(label)
        
        directories.append(dir_name)
        accuracies.append(float(result['Accuracy']) * 100)  # Convert to percentage
        inferences_per_second.append(float(result['Inf/s']))
        sizes_kb.append(float(result['Size (KB)']))
        model_names.append(model_name)
        
        # Convert parameters string to numeric
        params_str = result['Params']
        if 'M' in params_str:
            params_val = float(params_str.replace('M', '')) * 1_000_000
        elif 'K' in params_str:
            params_val = float(params_str.replace('K', '')) * 1_000
        else:
            params_val = float(params_str)
        parameters.append(params_val / 1_000_000)  # Convert to millions
    
    graph_paths = []
    
    # Generate unique colors and markers for each model
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    
    # Graph 1: Accuracy vs Inference Speed (main comparison)
    plt.figure(figsize=(14, 10))
    for i, (label, x, y) in enumerate(zip(labels, inferences_per_second, accuracies)):
        plt.scatter(x, y, c=[colors[i]], s=120, marker=markers[i % len(markers)], 
                   alpha=0.8, label=label, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Inferences per Second', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Speed\n({quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    
    plt.tight_layout()
    graph1_filename = f"accuracy_vs_speed_{timestamp}_{quant_suffix}_{dataset_suffix}.png"
    graph1_path = os.path.join(graphs_dir, graph1_filename)
    plt.savefig(graph1_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph1_path)
    
    # Graph 2: Model Size vs Accuracy
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(sizes_kb, accuracies, c=parameters, s=100, alpha=0.7, cmap='viridis')
    for i, (label, x, y) in enumerate(zip(labels, sizes_kb, accuracies)):
        plt.annotate(label.split('\n')[0],  # Just show directory name for clarity
                    (x, y), 
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=8, alpha=0.9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.xlabel('Model Size (KB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Model Size\n({quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Parameters (Millions)', fontsize=10)
    
    plt.tight_layout()
    graph2_filename = f"accuracy_vs_size_{timestamp}_{quant_suffix}_{dataset_suffix}.png"
    graph2_path = os.path.join(graphs_dir, graph2_filename)
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
    
    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    
    plt.tight_layout()
    graph3_filename = f"speed_vs_complexity_{timestamp}_{quant_suffix}_{dataset_suffix}.png"
    graph3_path = os.path.join(graphs_dir, graph3_filename)
    plt.savefig(graph3_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph3_path)
    
    # Graph 4: Inference Speed Bar Chart
    plt.figure(figsize=(16, 10))
    y_pos = np.arange(len(labels))
    bars = plt.barh(y_pos, inferences_per_second, color=colors, alpha=0.7)
    plt.yticks(y_pos, [label.split('\n')[0] for label in labels], fontsize=10)  # Just directory names
    plt.xlabel('Inferences per Second', fontsize=12)
    plt.title(f'Inference Speed Comparison\n({quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, acc, speed, model) in enumerate(zip(bars, accuracies, inferences_per_second, model_names)):
        width = bar.get_width()
        plt.text(width + max(inferences_per_second) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{int(speed)} inf/s\n{acc:.1f}% acc', 
                ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    # Create a separate legend for model names
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, 
                                   label=f"{labels[i].split('\\n')[0]}\n{model_names[i]}")
                      for i in range(len(labels))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
               borderaxespad=0., fontsize=8)
    
    plt.tight_layout()
    graph4_filename = f"speed_comparison_{timestamp}_{quant_suffix}_{dataset_suffix}.png"
    graph4_path = os.path.join(graphs_dir, graph4_filename)
    plt.savefig(graph4_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph4_path)
    
    # Graph 5: Accuracy Bar Chart
    plt.figure(figsize=(16, 10))
    y_pos = np.arange(len(labels))
    # Color by accuracy - highlight best in green
    bar_colors = ['lightgreen' if acc == max(accuracies) else colors[i] for i, acc in enumerate(accuracies)]
    bars = plt.barh(y_pos, accuracies, color=bar_colors, alpha=0.7)
    plt.yticks(y_pos, [label.split('\n')[0] for label in labels], fontsize=10)  # Just directory names
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy Comparison\n({quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, acc, speed, model) in enumerate(zip(bars, accuracies, inferences_per_second, model_names)):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}% acc\n{int(speed)} inf/s', 
                ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    # Create a separate legend for model names
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, 
                                   label=f"{labels[i].split('\\n')[0]}\n{model_names[i]}")
                      for i in range(len(labels))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
               borderaxespad=0., fontsize=8)
    
    plt.tight_layout()
    graph5_filename = f"accuracy_comparison_{timestamp}_{quant_suffix}_{dataset_suffix}.png"
    graph5_path = os.path.join(graphs_dir, graph5_filename)
    plt.savefig(graph5_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph5_path)
    
    print(f"📊 Generated {len(graph_paths)} comparison graphs:")
    for path in graph_paths:
        print(f"   📈 {os.path.basename(path)}")
    
    return graph_paths

def test_all_models(quantized_only=False, num_test_images=100, debug=False, use_all_datasets=False):
    """Test all available models and print summary table"""
    models = get_all_models(quantized_only=quantized_only)
    
    if not models:
        print("No models found to test.")
        return
    
    # Determine test configuration
    if use_all_datasets:
        print(f"\nTesting {len(models)} models on ALL available images from all datasets...")
    else:
        print(f"\nTesting {len(models)} models on {num_test_images} images...")
    
    print("-" * 80)
    
    results = []
    
    # Test all models (with or without progress bar based on debug mode)
    if debug:
        # Debug mode - no overall progress bar
        model_iterator = models
        print("Debug mode - showing detailed results for each model")
    else:
        # Normal mode - use progress bar for models
        model_iterator = tqdm(models, desc="Testing all models")
    
    for model_info in model_iterator:
        if debug:
            print(f"\n🔍 Testing model: {model_info['directory']}/{model_info['name']}")
        
        accuracy, tested_count, avg_inference_time, inferences_per_second = test_model_on_dataset(
            model_info['path'], 
            num_test_images=num_test_images,
            debug=debug,
            use_all_datasets=use_all_datasets
        )
        
        # Format parameters for display
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
            'Inf Time (ms)': f"{avg_inference_time:.2f}",
            'Inf Time_Raw': avg_inference_time,
            'Inf/s': f"{inferences_per_second:.0f}",
            'Inf/s_Raw': inferences_per_second,
            'Tested': tested_count
        })
        
        if debug:
            print(f"✅ Completed: {model_info['directory']}/{model_info['name']} - Accuracy: {accuracy:.3f}")
    
    # Sort by accuracy descending
    results.sort(key=lambda x: x['Accuracy_Raw'], reverse=True)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY RESULTS")
    if use_all_datasets:
        print(f"DATASETS: ALL available images")
    else:
        print(f"DATASETS: {num_test_images} sampled images")
    print(f"{'='*80}")
    
    # Simplified console output
    headers = ['Directory', 'Model', 'Type', 'Params', 'Size', 'Accuracy', 'Inf/s', 'Images']
    table_data = []
    for result in results:
        table_data.append([
            result['Directory'],
            result['Model'],
            result['Type'],
            result['Params'],
            result['Size (KB)'],
            f"{float(result['Accuracy']):.3f}",
            result['Inf/s'],
            result['Tested']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='simple_grid', stralign='right'))
    
    # Print best models by different criteria
    if results and results[0]['Accuracy_Raw'] > 0:
        best_accuracy = max(results, key=lambda x: x['Accuracy_Raw'])
        fastest_model = max(results, key=lambda x: x['Inf/s_Raw'])
        
        print(f"\n🏆 BEST BY ACCURACY: {best_accuracy['Directory']}/{best_accuracy['Model']}")
        print(f"   Accuracy: {float(best_accuracy['Accuracy']):.3f}, Speed: {best_accuracy['Inf/s']} inf/s")
        
        print(f"⚡ FASTEST MODEL: {fastest_model['Directory']}/{fastest_model['Model']}")
        print(f"   Speed: {fastest_model['Inf/s']} inf/s, Accuracy: {float(fastest_model['Accuracy']):.3f}")
    
    # Generate comparison graphs
    graph_paths = generate_comparison_graphs(
        results, 
        quantized_only=quantized_only,
        use_all_datasets=use_all_datasets
    )
    
    # Save full results to CSV
    csv_path = save_results_to_csv(
        results, 
        quantized_only=quantized_only,
        use_all_images=use_all_datasets,
        test_images_count=num_test_images
    )
    
    return results

def save_results_to_csv(results, quantized_only=False, use_all_images=False, test_images_count=100):
    """Save FULL results to CSV file with all information"""
    # Create results directory if it doesn't exist
    results_dir = os.path.join(params.OUTPUT_DIR, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_suffix = "quantized" if quantized_only else "all"
    dataset_suffix = "full" if use_all_images else f"{test_images_count}images"
    
    filename = f"model_comparison_{timestamp}_{quant_suffix}_{dataset_suffix}.csv"
    csv_path = os.path.join(results_dir, filename)
    
    # Prepare FULL data for CSV (all information)
    csv_data = []
    for result in results:
        # Convert params string back to number for CSV
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
            'Inference_Time_ms': float(result['Inf Time (ms)']),
            'Inferences_per_second': float(result['Inf/s']),
            'Tested_Images': result['Tested']
        })
    
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model', 'Directory', 'Type', 'Parameters', 'Size_KB', 'Accuracy', 'Inference_Time_ms', 'Inferences_per_second', 'Tested_Images']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"💾 Full results saved to: {csv_path}")
    return csv_path

def list_available_models(quantized_only=False):
    """List all available models in training directories"""
    models = get_all_models(quantized_only=quantized_only)
    
    if not models:
        print("No models found.")
        return
    
    print(f"Available {'quantized ' if quantized_only else ''}models:")
    print("-" * 50)
    
    for model in models:
        print(f"{model['directory']}/{model['name']} ({model['type']}, {model['size_kb']:.1f} KB, {model['parameters']} params)")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Digit Recognition Benchmarking')
    parser.add_argument('--model', type=str, help='Model name to use for prediction')
    parser.add_argument('--quantized', action='store_true', default=True, help='Use only quantized models')
    parser.add_argument('--test_all', action='store_true', default=True, help='Test all available models and print accuracy summary')
    parser.add_argument('--test_images', type=int, default=25000, help='Number of test images per model (default: 25000)')
    parser.add_argument('--all_datasets', action='store_true', default=True, help='Use all available images from dataset (overrides --test_images, only for --test_all)')
    parser.add_argument('--debug', action='store_true', help='Debug model output interpretation')
    parser.add_argument('--list', action='store_true', help='List all available models')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models(quantized_only=args.quantized)
        return
    
    if args.test_all:
        # Validate that --all_datasets is only used with --test_all
        if args.all_datasets and not args.test_all:
            print("Warning: --all_datasets can only be used with --test_all. Ignoring --all_datasets.")
            args.all_datasets = False
        
        test_all_models(
            quantized_only=args.quantized, 
            num_test_images=args.test_images,
            debug=args.debug,
            use_all_datasets=args.all_datasets
        )
        return
    
    if args.debug:
        print("Debug mode requires --test_all for benchmarking. Use --test_all --debug")
        return
    
    print("Use --test_all to run benchmarks or --list to see available models.")

if __name__ == "__main__":
    main()
    
    
# py bench_predict.py --test_all --quantized --test_images 25000