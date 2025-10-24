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
        print(f"Output type: {self.output_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        
        # Print quantization info
        if self.input_details[0]['quantization'][0] != 0:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
        else:
            print("Model type: Float32 (non-quantized)")
    
    def predict(self, image, rgb_mode=False, debug=False):
        """Predict digit from image using TFLite"""
        # Preprocess image
        processed_image = self.preprocess_image_for_prediction(image, rgb_mode=rgb_mode)
        
        # Add batch dimension
        input_data = np.expand_dims(processed_image, axis=0)
        
        # Handle quantization if needed
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = input_data.astype(np.uint8)
        elif self.input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)
        
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
        
        # DEBUG: Print raw output only in debug mode
        if debug:
            print(f"Raw output: {output_data[0]}")
            print(f"Output sum: {np.sum(output_data[0]):.6f}")
        
        # Get prediction and confidence
        prediction = np.argmax(output_data[0])
        confidence = np.max(output_data[0])
        
        return prediction, confidence, output_data[0]

    def preprocess_image_for_prediction(self, image, rgb_mode=False):
        """Preprocess image for prediction with RGB support"""
        if rgb_mode:
            # If image is grayscale but we need RGB, convert it
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # If image is RGB, ensure it's the right format
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Already RGB, do nothing
                pass
            else:
                raise ValueError(f"Unsupported image shape for RGB mode: {image.shape}")
        else:
            # If image is RGB but we need grayscale, convert it
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # If image is already grayscale, do nothing
            elif len(image.shape) == 2:
                pass
            else:
                raise ValueError(f"Unsupported image shape for grayscale mode: {image.shape}")
        
        # Use the existing preprocess function
        return predict_single_image(image)

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
        print(f"Error counting parameters for {model_path}: {e}")
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

def find_model_path(model_name=None, quantized_only=False):
    """Find the model path based on model name or use default behavior"""
    # Look for training directories
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if os.path.isdir(os.path.join(params.OUTPUT_DIR, d))]
    if not training_dirs:
        print("No training directories found. Please run train.py first.")
        return None
    
    if model_name:
        # Remove .tflite extension if present for easier matching
        model_name_clean = model_name.replace('.tflite', '')
        
        print(f"Searching for model: {model_name}")
        
        # Search through all training directories
        found_models = []
        
        for training_dir in sorted(training_dirs, reverse=True):
            training_path = os.path.join(params.OUTPUT_DIR, training_dir)
            
            # Check if model_name matches the training directory name
            if model_name_clean in training_dir:
                # Look for all models in this directory
                tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
                for model_file in tflite_files:
                    model_path = os.path.join(training_path, model_file)
                    if quantized_only and not is_quantized_model(model_path):
                        continue
                    found_models.append(model_path)
            
            # Check for exact model file matches
            tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
            for model_file in tflite_files:
                model_file_clean = model_file.replace('.tflite', '')
                
                # Exact match or partial match
                if (model_name_clean == model_file_clean or 
                    model_name_clean in model_file_clean or
                    model_name == model_file):
                    
                    model_path = os.path.join(training_path, model_file)
                    if quantized_only and not is_quantized_model(model_path):
                        continue
                    found_models.append(model_path)
        
        # Remove duplicates and sort
        found_models = list(set(found_models))
        found_models.sort()
        
        if found_models:
            if len(found_models) > 1:
                print(f"Multiple models found matching '{model_name}':")
                for i, model_path in enumerate(found_models, 1):
                    model_dir = os.path.basename(os.path.dirname(model_path))
                    model_file = os.path.basename(model_path)
                    model_type = "quantized" if is_quantized_model(model_path) else "float"
                    print(f"  {i}. {model_dir}/{model_file} ({model_type})")
                
                # Use the first one (usually most recent)
                selected_model = found_models[0]
                model_dir = os.path.basename(os.path.dirname(selected_model))
                model_file = os.path.basename(selected_model)
                print(f"Using: {model_dir}/{model_file}")
                return selected_model
            else:
                model_dir = os.path.basename(os.path.dirname(found_models[0]))
                model_file = os.path.basename(found_models[0])
                print(f"Found: {model_dir}/{model_file}")
                return found_models[0]
        
        print(f"Model '{model_name}' not found in any training directory")
        if quantized_only:
            print("No quantized model found with the specified name.")
        print("Available training directories and models:")
        for training_dir in training_dirs:
            print(f"  - {training_dir}")
            training_dir_path = os.path.join(params.OUTPUT_DIR, training_dir)
            if os.path.exists(training_dir_path):
                files = [f for f in os.listdir(training_dir_path) if f.endswith('.tflite')]
                for file in files:
                    file_path = os.path.join(training_dir_path, file)
                    model_type = "quantized" if is_quantized_model(file_path) else "float"
                    print(f"    └── {file} ({model_type})")
        return None
    else:
        # Use default behavior - latest training directory
        latest_training = sorted(training_dirs)[-1]
        
        # Define possible paths based on quantization preference
        possible_paths = []
        
        if quantized_only:
            # Only look for quantized models in the latest directory
            possible_paths = [
                os.path.join(params.OUTPUT_DIR, latest_training, "final_quantized.tflite"),
                # os.path.join(params.OUTPUT_DIR, latest_training, "model_quantized.tflite"),
                # os.path.join(params.OUTPUT_DIR, latest_training, "quantized.tflite"),
            ]
        else:
            # Look for all models in the latest directory
            possible_paths = [
                # os.path.join(params.OUTPUT_DIR, latest_training, params.TFLITE_FILENAME),
                os.path.join(params.OUTPUT_DIR, latest_training, "final_quantized.tflite"),
                os.path.join(params.OUTPUT_DIR, latest_training, "final_float.tflite"),
            ]
        
        # First, try the specific paths
        for model_path in possible_paths:
            if os.path.exists(model_path):
                if quantized_only and not is_quantized_model(model_path):
                    continue
                return model_path
        
        # If no specific model found, look for any .tflite file in the latest directory
        latest_dir_path = os.path.join(params.OUTPUT_DIR, latest_training)
        tflite_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.tflite')]
        
        if quantized_only:
            # Filter for quantized models only
            tflite_files = [f for f in tflite_files if is_quantized_model(os.path.join(latest_dir_path, f))]
        
        if tflite_files:
            model_path = os.path.join(latest_dir_path, tflite_files[0])
            return model_path
        
        print(f"No {'quantized ' if quantized_only else ''}TFLite model found in: {latest_dir_path}")
        print("Available files:")
        for file in os.listdir(latest_dir_path):
            if file.endswith('.tflite'):
                file_path = os.path.join(latest_dir_path, file)
                model_type = "quantized" if is_quantized_model(file_path) else "float"
                print(f"  - {file} ({model_type})")
        return None

def get_all_models(quantized_only=False):
    """Get all available models with parameters count - with error handling"""
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if os.path.isdir(os.path.join(params.OUTPUT_DIR, d))]
    all_models = []
    
    for training_dir in training_dirs:
        training_path = os.path.join(params.OUTPUT_DIR, training_dir)
        
        # Look for the standard model file names
        possible_model_files = [
            "final_quantized.tflite",
            "final_float.tflite", 
            # "model_quantized.tflite",
            # "model_float.tflite",
            # "quantized.tflite",
            # "float.tflite",
            # params.TFLITE_FILENAME  # Use the filename from parameters
        ]
        
        # Also look for any .tflite files in the directory
        all_tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
        
        # Combine both approaches
        model_files_to_check = []
        for model_file in possible_model_files:
            if model_file in all_tflite_files:
                model_files_to_check.append(model_file)
        
        # If no standard files found, use all available .tflite files
        if not model_files_to_check and all_tflite_files:
            model_files_to_check = all_tflite_files
        
        for model_file in model_files_to_check:
            model_path = os.path.join(training_path, model_file)
            
            # Skip if file doesn't exist or is empty
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                print(f"⚠️  Skipping invalid model file: {training_dir}/{model_file}")
                continue
                
            # Verify the model can be loaded
            if not is_valid_tflite_model(model_path):
                print(f"⚠️  Skipping corrupted model file: {training_dir}/{model_file}")
                continue
                
            if quantized_only and not is_quantized_model(model_path):
                continue
                
            try:
                model_size = os.path.getsize(model_path) / 1024
                model_type = "quantized" if is_quantized_model(model_path) else "float"
                parameters_count = get_model_parameters_count(model_path)
                
                all_models.append({
                    'path': model_path,
                    'name': model_file,  # Use the actual filename
                    'directory': training_dir,
                    'size_kb': model_size,
                    'type': model_type,
                    'parameters': parameters_count
                })
                print(f"✅ Found valid model: {training_dir}/{model_file}")
                
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

def count_total_images_in_datasets():
    """Count total number of images available in all datasets"""
    if not params.DATA_SOURCES:
        return 0
    
    total_images = 0
    for data_source in params.DATA_SOURCES:
        dataset_path = data_source['path']
        dataset_images = 0
        
        for digit in range(10):
            digit_folder = os.path.join(dataset_path, str(digit))
            if os.path.exists(digit_folder):
                image_files = [f for f in os.listdir(digit_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                dataset_images += len(image_files)
        
        print(f"  - {data_source['name']}: {dataset_images} images")
        total_images += dataset_images
    
    return total_images

def test_model_on_dataset(model_path, rgb_mode=False, num_test_images=100, debug=False, use_all_datasets=False):
    """Test a model on random images from dataset and return accuracy and performance metrics"""
    predictor = TFLiteDigitPredictor(model_path)
    correct_predictions = 0
    total_tested = 0
    total_inference_time = 0.0
    
    # Get dataset paths from all data sources
    if not params.DATA_SOURCES:
        print("No data sources found in parameters.py")
        return 0.0, 0, 0.0, 0.0
    
    # Collect all test images with their true labels from all datasets
    test_data = []
    
    for data_source in params.DATA_SOURCES:
        dataset_path = data_source['path']
        weight = data_source.get('weight', 1.0)  # Default weight is 1.0 if not specified
        
        for digit in range(10):
            digit_folder = os.path.join(dataset_path, str(digit))
            if not os.path.exists(digit_folder):
                continue
                
            # Get images for this digit
            image_files = [f for f in os.listdir(digit_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue
            
            if use_all_datasets:
                # Use all available images from this dataset
                for image_file in image_files:
                    test_data.append((os.path.join(digit_folder, image_file), digit, weight))
            else:
                # Calculate number of images to use from this dataset based on weight
                dataset_images_count = int((num_test_images // 10) * weight)
                dataset_images_count = max(1, dataset_images_count)  # At least 1 image per digit
                
                if len(image_files) > 0:
                    test_images = np.random.choice(
                        image_files, 
                        min(dataset_images_count, len(image_files)), 
                        replace=False
                    )
                    for image_file in test_images:
                        test_data.append((os.path.join(digit_folder, image_file), digit, weight))
    
    # Shuffle test data
    np.random.shuffle(test_data)
    
    # Warm-up run to avoid cold start timing issues
    if len(test_data) > 0:
        warmup_image_path, warmup_digit, _ = test_data[0]
        warmup_image = cv2.imread(warmup_image_path, cv2.IMREAD_GRAYSCALE if not rgb_mode else cv2.IMREAD_COLOR)
        if warmup_image is not None and rgb_mode:
            warmup_image = cv2.cvtColor(warmup_image, cv2.COLOR_BGR2RGB)
        if warmup_image is not None:
            try:
                predictor.predict(warmup_image, rgb_mode=rgb_mode, debug=False)
            except:
                pass
    
    # Test with progress bar
    for image_path, true_digit, weight in tqdm(test_data, desc=f"Testing {os.path.basename(model_path)}", leave=False):
        # Load image
        if rgb_mode:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            continue
        
        # Predict with timing
        try:
            start_time = time.perf_counter()
            prediction, confidence, _ = predictor.predict(image, rgb_mode=rgb_mode, debug=debug)
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            total_inference_time += inference_time
            
            if prediction == true_digit:
                correct_predictions += 1
            total_tested += 1
        except Exception as e:
            continue
    
    # Calculate performance metrics
    accuracy = correct_predictions / total_tested if total_tested > 0 else 0.0
    
    avg_inference_time = total_inference_time / total_tested if total_tested > 0 else 0.0
    inferences_per_second = 1000 / avg_inference_time if avg_inference_time > 0 else 0.0
    
    return accuracy, total_tested, avg_inference_time, inferences_per_second

def generate_comparison_graphs(results, rgb_mode=False, quantized_only=False, use_all_datasets=False):
    """Generate separate comparison graphs for the benchmark results"""
    # Create graphs directory
    graphs_dir = os.path.join(params.OUTPUT_DIR, "test_results", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "rgb" if rgb_mode else "grayscale"
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
    plt.title(f'Accuracy vs Speed\n({mode_suffix}, {quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    
    plt.tight_layout()
    graph1_filename = f"accuracy_vs_speed_{timestamp}_{mode_suffix}_{quant_suffix}_{dataset_suffix}.png"
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
    plt.title(f'Accuracy vs Model Size\n({mode_suffix}, {quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Parameters (Millions)', fontsize=10)
    
    plt.tight_layout()
    graph2_filename = f"accuracy_vs_size_{timestamp}_{mode_suffix}_{quant_suffix}_{dataset_suffix}.png"
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
    plt.title(f'Speed vs Model Complexity\n({mode_suffix}, {quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    
    plt.tight_layout()
    graph3_filename = f"speed_vs_complexity_{timestamp}_{mode_suffix}_{quant_suffix}_{dataset_suffix}.png"
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
    plt.title(f'Inference Speed Comparison\n({mode_suffix}, {quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
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
    graph4_filename = f"speed_comparison_{timestamp}_{mode_suffix}_{quant_suffix}_{dataset_suffix}.png"
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
    plt.title(f'Accuracy Comparison\n({mode_suffix}, {quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
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
    graph5_filename = f"accuracy_comparison_{timestamp}_{mode_suffix}_{quant_suffix}_{dataset_suffix}.png"
    graph5_path = os.path.join(graphs_dir, graph5_filename)
    plt.savefig(graph5_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph5_path)
    
    # Graph 6: Size vs Speed Trade-off
    plt.figure(figsize=(14, 10))
    for i, (label, x, y) in enumerate(zip(labels, sizes_kb, inferences_per_second)):
        plt.scatter(x, y, c=[colors[i]], s=120, marker=markers[i % len(markers)], 
                   alpha=0.8, label=label, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Model Size (KB)', fontsize=12)
    plt.ylabel('Inferences per Second', fontsize=12)
    plt.title(f'Size vs Speed Trade-off\n({mode_suffix}, {quant_suffix}, {dataset_suffix})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    
    plt.tight_layout()
    graph6_filename = f"size_vs_speed_{timestamp}_{mode_suffix}_{quant_suffix}_{dataset_suffix}.png"
    graph6_path = os.path.join(graphs_dir, graph6_filename)
    plt.savefig(graph6_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph6_path)
    
    print(f"📊 Generated {len(graph_paths)} separate comparison graphs:")
    for path in graph_paths:
        print(f"   📈 {os.path.basename(path)}")
    
    return graph_paths

def test_all_models(rgb_mode=False, quantized_only=False, num_test_images=100, debug=False, use_all_datasets=False):
    """Test all available models and print summary table"""
    models = get_all_models(quantized_only=quantized_only)
    
    if not models:
        print("No models found to test.")
        return
    
    # Determine test configuration
    if use_all_datasets:
        print(f"\nTesting {len(models)} models on ALL available images from all datasets:")
        total_images = count_total_images_in_datasets()
        actual_test_images = total_images
    else:
        print(f"\nTesting {len(models)} models on {num_test_images} images (distributed across datasets by weight)...")
        actual_test_images = num_test_images
    
    print("Mode:", "RGB" if rgb_mode else "Grayscale")
    print("-" * 80)
    
    results = []
    
    # Test all models with progress bar
    for model_info in tqdm(models, desc="Testing all models"):
        accuracy, tested_count, avg_inference_time, inferences_per_second = test_model_on_dataset(
            model_info['path'], 
            rgb_mode=rgb_mode, 
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
    
    # Sort by accuracy descending
    results.sort(key=lambda x: x['Accuracy_Raw'], reverse=True)
    
    # Print simplified console table
    print(f"\n{'='*80}")
    print(f"SUMMARY RESULTS ({'RGB' if rgb_mode else 'Grayscale'} mode)")
    if use_all_datasets:
        print(f"DATASETS: ALL datasets ({actual_test_images} total images)")
    else:
        print(f"DATASETS: Distributed sampling ({actual_test_images} target images)")
    print(f"{'='*80}")
    
    # Simplified console output - only essential columns
    headers = ['Directory', 'Params', 'Size', 'Accuracy', 'Inf/s', 'Images']
    table_data = []
    for result in results:
        # Shorten directory name for display
        short_dir = result['Directory'][:20] + '...' if len(result['Directory']) > 23 else result['Directory']
        
        table_data.append([
            short_dir,
            result['Params'],
            result['Size (KB)'],
            result['Accuracy'],
            result['Inf/s'],
            result['Tested']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='simple_grid', stralign='right'))
    
    # Print best models by different criteria
    if results and results[0]['Accuracy_Raw'] > 0:
        best_accuracy = max(results, key=lambda x: x['Accuracy_Raw'])
        fastest_model = min(results, key=lambda x: x['Inf Time_Raw'])
        
        # Find best balanced (accuracy * speed)
        balanced_scores = []
        for result in results:
            acc = result['Accuracy_Raw']
            speed = result['Inf/s_Raw']
            # Normalize and combine (you can adjust weights here)
            balanced_score = acc * (speed / 1000)  # Normalize speed
            balanced_scores.append((result, balanced_score))
        
        best_balanced = max(balanced_scores, key=lambda x: x[1])[0]
        
        print(f"\n🏆 BEST BY ACCURACY: {best_accuracy['Directory']}/{best_accuracy['Model']}")
        print(f"   Accuracy: {best_accuracy['Accuracy']}, Speed: {best_accuracy['Inf/s']} inf/s")
        
        print(f"⚡ FASTEST MODEL: {fastest_model['Directory']}/{fastest_model['Model']}")
        print(f"   Speed: {fastest_model['Inf/s']} inf/s, Accuracy: {fastest_model['Accuracy']}")
        
        print(f"⭐ BEST BALANCED: {best_balanced['Directory']}/{best_balanced['Model']}")
        print(f"   Accuracy: {best_balanced['Accuracy']}, Speed: {best_balanced['Inf/s']} inf/s")
    
    # Generate comparison graphs
    graph_path = generate_comparison_graphs(
        results, 
        rgb_mode=rgb_mode, 
        quantized_only=quantized_only,
        use_all_datasets=use_all_datasets
    )
    
    # Save full results to CSV
    csv_path = save_results_to_csv(
        results, 
        rgb_mode=rgb_mode, 
        quantized_only=quantized_only,
        use_all_images=use_all_datasets,
        test_images_count=actual_test_images
    )
    
    return results

def save_results_to_csv(results, rgb_mode=False, quantized_only=False, use_all_images=False, test_images_count=100):
    """Save FULL results to CSV file with all information"""
    # Create results directory if it doesn't exist
    results_dir = os.path.join(params.OUTPUT_DIR, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "rgb" if rgb_mode else "grayscale"
    quant_suffix = "quantized" if quantized_only else "all"
    dataset_suffix = "full" if use_all_images else f"{test_images_count}images"
    
    filename = f"model_comparison_{timestamp}_{mode_suffix}_{quant_suffix}_{dataset_suffix}.csv"
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
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if os.path.isdir(os.path.join(params.OUTPUT_DIR, d))]
    if not training_dirs:
        print("No training directories found.")
        return
    
    print(f"Available {'quantized ' if quantized_only else ''}models:")
    print("-" * 50)
    
    valid_models_found = False
    for training_dir in sorted(training_dirs, reverse=True):
        training_path = os.path.join(params.OUTPUT_DIR, training_dir)
        
        # Look for standard model files
        model_files = []
        for model_file in ["final_quantized.tflite", "final_float.tflite", "model_quantized.tflite", "model_float.tflite"]:
            if os.path.exists(os.path.join(training_path, model_file)):
                model_files.append(model_file)
        
        # If no standard files, look for any .tflite files
        if not model_files:
            model_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
        
        if quantized_only:
            model_files = [f for f in model_files if is_quantized_model(os.path.join(training_path, f))]
        
        if model_files:
            valid_models_found = True
            print(f"\n{training_dir}:")
            for model_file in model_files:
                model_path = os.path.join(training_path, model_file)
                if os.path.exists(model_path):
                    model_size = os.path.getsize(model_path) / 1024
                    model_type = "quantized" if is_quantized_model(model_path) else "float"
                    print(f"  └── {model_file} ({model_size:.1f} KB, {model_type})")
                else:
                    print(f"  └── {model_file} (FILE NOT FOUND)")
    
    if not valid_models_found:
        print("No valid model files found in any training directory.")
        print("Expected model files: final_quantized.tflite, final_float.tflite, etc.")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Digit Recognition Benchmarking')
    parser.add_argument('--model', type=str, help='Model name to use for prediction')
    parser.add_argument('--quantized', action='store_true', help='Use only quantized models')
    parser.add_argument('--RGB', action='store_true', help='Process image as RGB instead of grayscale')
    parser.add_argument('--test_all', action='store_true', help='Test all available models and print accuracy summary')
    parser.add_argument('--test_images', type=int, default=100, help='Number of test images per model (default: 100)')
    parser.add_argument('--all_datasets', action='store_true', help='Use all available images from dataset (overrides --test_images, only for --test_all)')
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
            rgb_mode=args.RGB, 
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
    
# py bench_predict.py --test_all --quantized --all_datasets
# py bench_predict.py --test_all --quantized --test_images 18000 (+0.x weight from augmented datasource)