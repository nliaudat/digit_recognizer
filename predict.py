# predict.py
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

def load_random_image_from_dataset(rgb_mode=False):
    """Load a random image from the first available data source"""
    if not params.DATA_SOURCES:
        print("No data sources found in parameters.py")
        return None
    
    # Use the first data source
    data_source = params.DATA_SOURCES[0]
    dataset_path = data_source['path']
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return None
    
    # Collect all images from the dataset
    image_paths = []
    for digit in range(10):
        digit_folder = os.path.join(dataset_path, str(digit))
        if os.path.exists(digit_folder):
            for file in os.listdir(digit_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(digit_folder, file))
    
    if not image_paths:
        print(f"No images found in {dataset_path}")
        return None
    
    # Select a random image
    random_image_path = np.random.choice(image_paths)
    print(f"Loading random image: {random_image_path}")
    
    # Load and return the image
    if rgb_mode:
        image = cv2.imread(random_image_path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(random_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to load image: {random_image_path}")
        return None
    
    return image

def load_image_from_path(image_path, rgb_mode=False):
    """Load image from specified path"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    if rgb_mode:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    print(f"Loaded image: {image_path}")
    return image

def find_model_path(model_name=None, quantized_only=False):
    """Find the model path based on model name or use default behavior"""
    # Look for training directories
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if os.path.isdir(os.path.join(params.OUTPUT_DIR, d))]
    if not training_dirs:
        print("No training directories found. Please run train.py first.")
        return None
    
    # If model_name is provided, it could be:
    # 1. Full model name with extension (e.g., "final_quantized.tflite")
    # 2. Model name without extension (e.g., "final_quantized")
    # 3. Training directory name (e.g., "digit_recognizer_v2_10_GR")
    # 4. Partial name match
    
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
                os.path.join(params.OUTPUT_DIR, latest_training, "model_quantized.tflite"),
                os.path.join(params.OUTPUT_DIR, latest_training, "quantized.tflite"),
            ]
        else:
            # Look for all models in the latest directory
            possible_paths = [
                os.path.join(params.OUTPUT_DIR, latest_training, params.TFLITE_FILENAME),
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

def list_available_models(quantized_only=False):
    """List all available models in training directories"""
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if os.path.isdir(os.path.join(params.OUTPUT_DIR, d))]
    if not training_dirs:
        print("No training directories found.")
        return
    
    print(f"Available {'quantized ' if quantized_only else ''}models:")
    print("-" * 50)
    
    for training_dir in sorted(training_dirs, reverse=True):
        training_path = os.path.join(params.OUTPUT_DIR, training_dir)
        tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
        
        if quantized_only:
            tflite_files = [f for f in tflite_files if is_quantized_model(os.path.join(training_path, f))]
        
        if tflite_files:
            print(f"\n{training_dir}:")
            for model_file in tflite_files:
                model_path = os.path.join(training_path, model_file)
                model_size = os.path.getsize(model_path) / 1024
                model_type = "quantized" if is_quantized_model(model_path) else "float"
                print(f"  └── {model_file} ({model_size:.1f} KB, {model_type})")

def get_all_models(quantized_only=False):
    """Get all available models with parameters count"""
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if os.path.isdir(os.path.join(params.OUTPUT_DIR, d))]
    all_models = []
    
    for training_dir in training_dirs:
        training_path = os.path.join(params.OUTPUT_DIR, training_dir)
        tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
        
        for model_file in tflite_files:
            model_path = os.path.join(training_path, model_file)
            if quantized_only and not is_quantized_model(model_path):
                continue
                
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
    
    return all_models

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
    
    print(f"\n💾 Full results saved to: {csv_path}")
    return csv_path

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
            'Params_Raw': params_count,  # Keep raw for sorting
            'Size (KB)': f"{model_info['size_kb']:.1f}",
            'Size_Raw': model_info['size_kb'],  # Keep raw for sorting
            'Accuracy': f"{accuracy:.4f}",
            'Accuracy_Raw': accuracy,  # Keep raw for sorting
            'Inf Time (ms)': f"{avg_inference_time:.2f}",
            'Inf Time_Raw': avg_inference_time,  # Keep raw for sorting
            'Inf/s': f"{inferences_per_second:.0f}",
            'Inf/s_Raw': inferences_per_second,  # Keep raw for sorting
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
    
    # Save full results to CSV
    csv_path = save_results_to_csv(
        results, 
        rgb_mode=rgb_mode, 
        quantized_only=quantized_only,
        use_all_images=use_all_datasets,
        test_images_count=actual_test_images
    )
    
    return results

def debug_model_output(quantized_only=False, rgb_mode=False):
    """Debug function to test model output interpretation"""
    model_path = find_model_path(quantized_only=quantized_only)
    if not model_path:
        return
    
    model_dir = os.path.basename(os.path.dirname(model_path))
    model_name = os.path.basename(model_path)
    print(f"\n=== MODEL USED: {model_dir}/{model_name} ===")
    
    predictor = TFLiteDigitPredictor(model_path)
    
    # Test with multiple random images
    print(f"\n=== DEBUGGING MODEL OUTPUT ({'quantized' if quantized_only else 'float'}, {'RGB' if rgb_mode else 'grayscale'}) ===")
    for i in range(3):
        print(f"\n--- Test {i+1} ---")
        if rgb_mode:
            test_image = np.random.randint(0, 255, (params.INPUT_HEIGHT, params.INPUT_WIDTH, 3), dtype=np.uint8)
        else:
            test_image = np.random.randint(0, 255, (params.INPUT_HEIGHT, params.INPUT_WIDTH), dtype=np.uint8)
        
        prediction, confidence, raw_output = predictor.predict(test_image, rgb_mode=rgb_mode, debug=True)
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.6f}")
        print(f"All confidences: {[f'{x:.6f}' for x in raw_output]}")
        
        # Check if softmax properties hold
        output_sum = np.sum(raw_output)
        print(f"Softmax sum: {output_sum:.6f} (should be ~1.0)")
        
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

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Digit Recognition Prediction')
    parser.add_argument('--img', type=str, help='Path to input image for prediction')
    parser.add_argument('--model', type=str, help='Model name to use for prediction (can be: model filename, training directory name, or partial match)')
    parser.add_argument('--quantized', action='store_true', help='Use only quantized models')
    parser.add_argument('--RGB', action='store_true', help='Process image as RGB instead of grayscale')
    parser.add_argument('--test_all', action='store_true', help='Test all available models and print accuracy summary')
    parser.add_argument('--test_images', type=int, default=1000, help='Number of test images per model (default: 1000)')
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
        debug_model_output(quantized_only=args.quantized, rgb_mode=args.RGB)
        return
    
    # Find model path
    model_path = find_model_path(args.model, quantized_only=args.quantized)
    if not model_path:
        return
    
    # Print model used before loading - include directory
    model_dir = os.path.basename(os.path.dirname(model_path))
    model_name = os.path.basename(model_path)
    print(f"\n=== MODEL USED: {model_dir}/{model_name} ===")
    
    # Load predictor
    predictor = TFLiteDigitPredictor(model_path)
    
    # Load image
    if args.img:
        image = load_image_from_path(args.img, rgb_mode=args.RGB)
    else:
        print("No image specified, loading random image from dataset...")
        image = load_random_image_from_dataset(rgb_mode=args.RGB)
    
    if image is None:
        print("Failed to load image")
        return
    
    # Perform prediction
    prediction, confidence, raw_output = predictor.predict(image, rgb_mode=args.RGB, debug=args.debug)
    print(f"\n=== PREDICTION RESULT ===")
    print(f"Predicted digit: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All probabilities: {[f'{x:.4f}' for x in raw_output]}")

if __name__ == "__main__":
    main()
    
# py predict.py --test_all --quantized --all_datasets
# py predict.py --test_all --quantized --test_images 18000 (+0.x weight from augmented datasource)