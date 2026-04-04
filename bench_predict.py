import tensorflow as tf
import numpy as np
import cv2
import os
import argparse

# Deferred imports to prioritize CLI arguments
params = None
preprocess_for_inference = None

try:
    from tabulate import tabulate
except ImportError:
    # Simple fallback for tabulate when it's not installed.
    def tabulate(table_data, headers=None, tablefmt=None, stralign=None):
        """Simple fallback for tabulate when it's not installed."""
        if not table_data: return ""
        if not headers: headers = [f"Col{i}" for i in range(len(table_data[0]))]
        # Basic alignment and spacing
        cols = list(zip(*([headers] + table_data)))
        col_widths = [max(len(str(x)) for x in col) for col in cols]
        
        lines = []
        # Header
        header_line = " | ".join(f"{str(h):{w}}" for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        # Data
        for row in table_data:
            lines.append(" | ".join(f"{str(val):{w}}" for val, w in zip(row, col_widths)))
        return "\n".join(lines)

import glob
from tqdm import tqdm
import csv
from datetime import datetime
import time
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from PIL import Image
import shutil


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
        
        from utils.model_distiller_utils import create_tflite_interpreter
        self.interpreter = create_tflite_interpreter(self.model_path)
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input type: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def predict(self, image, debug=False):
        """Predict digit from image using TFLite"""
        # Preprocess image - use silent mode to avoid debug outputs
        processed_image = preprocess_for_inference(image)
        
        # Handle channel mismatch - if model expects 3 channels but we have 1
        expected_channels = self.input_details[0]['shape'][3]
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 1 and expected_channels == 3:
            processed_image = np.repeat(processed_image, 3, axis=2)
        
        # Add batch dimension if not already present
        if len(processed_image.shape) == 3:
            input_data = np.expand_dims(processed_image, axis=0)
        else:
            input_data = processed_image
        
        # Robustly ensure input is scaled correctly based on what this specific model expects
        expected_dtype = self.input_details[0]['dtype']
        if expected_dtype == np.uint8:
            if input_data.dtype == np.float32 and input_data.max() <= 1.0:
                input_data = (input_data * 255.0).astype(np.uint8)
            else:
                input_data = input_data.astype(np.uint8)
        elif expected_dtype == np.int8:
            if input_data.dtype == np.float32 and input_data.max() <= 1.0:
                input_data = (input_data * 255.0 - 128).astype(np.int8)
            elif input_data.dtype == np.uint8:
                input_data = (input_data.astype(np.int32) - 128).astype(np.int8)
            else:
                input_data = input_data.astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)
            if input_data.max() > 1.0:
                input_data = input_data / 255.0
        
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

    @property
    def num_classes(self):
        """Get the number of classes this model was trained to predict"""
        return self.output_details[0]['shape'][-1]

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

def find_model_path(model_name=None, input_dir=None):
    """Find the model path based on model name"""
    if input_dir is None:
        input_dir = params.OUTPUT_DIR
        
    # Look for training directories - exclude test_results and other non-training dirs
    all_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Filter out non-training directories
    training_dirs = []
    for dir_name in all_dirs:
        dir_path = os.path.join(input_dir, dir_name)
        # Check if this directory contains .tflite files and is likely a training directory
        tflite_files = [f for f in os.listdir(dir_path) if f.endswith('.tflite') and not f.endswith('_float.tflite')]
        if tflite_files and not dir_name.startswith('test_results'):
            training_dirs.append(dir_name)
    
    if not training_dirs:
        print(f"No training directories with TFLite models found in {input_dir}.")
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
            
            training_path = os.path.join(input_dir, best_match)
            tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite') and not f.endswith('_float.tflite')]
            
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
            training_path = os.path.join(input_dir, training_dir)
            
            # Check for exact model file matches
            tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite') and not f.endswith('_float.tflite')]
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
        latest_dir_path = os.path.join(input_dir, latest_training)
        
        # Look for any .tflite file in the latest directory
        tflite_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.tflite') and not f.endswith('_float.tflite')]
        
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
        from utils.model_distiller_utils import create_tflite_interpreter
        interpreter = create_tflite_interpreter(model_path)
        
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
    """Check if a model is quantized by examining its input type or path"""
    try:
        # Trust directory/file naming conventions first (crucial for dynamic range QAT models 
        # which have int8 weights but float32 I/O to maintain XNNPACK compatibility)
        path_lower = model_path.lower()
        if 'qat' in path_lower or 'quant' in path_lower:
            return True
            
        from utils.model_distiller_utils import create_tflite_interpreter
        interpreter = create_tflite_interpreter(model_path)
        input_details = interpreter.get_input_details()
        input_dtype = input_details[0]['dtype']
        
        # Pure integer quantized models use int8 or uint8
        return input_dtype in [np.int8, np.uint8]
    except:
        return False

def get_all_models(quantized_only=False, subfolder=None, input_dir=None, exclude_model=None, debug=False, model_list=None):
    """Get all available models with parameters count - with error handling
    
    Args:
        quantized_only: If True, only return quantized models
        subfolder: If specified, only look in this specific subfolder
        exclude_model: List of model names or strings to exclude from testing
        debug: Enable debug output
        model_list: Optional list of specific models to include (names or directories)
    """
    # Look for training directories - exclude test_results and other non-training dirs
    all_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Filter out non-training directories
    training_dirs = []
    for dir_name in all_dirs:
        dir_path = os.path.join(input_dir, dir_name)
        # Check if this directory contains .tflite files and is likely a training directory
        tflite_files = [f for f in os.listdir(dir_path) if f.endswith('.tflite') and not f.endswith('_float.tflite')]
        if tflite_files and not dir_name.startswith('test_results'):
            training_dirs.append(dir_name)
    
    # Filter by subfolder if specified
    if subfolder:
        if subfolder in training_dirs:
            training_dirs = [subfolder]
        else:
            # Check if subfolder is a partial match
            matching_dirs = [d for d in training_dirs if subfolder in d]
            if matching_dirs:
                training_dirs = matching_dirs
            else:
                print(f"⚠️  Subfolder '{subfolder}' not found in training directories")
                print(f"   Available directories: {training_dirs}")
                return []
    
    all_models = []
    
    for training_dir in training_dirs:
        training_path = os.path.join(input_dir, training_dir)
        
        # Look for any .tflite files in the directory
        tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite') and not f.endswith('_float.tflite')]
        
        for model_file in tflite_files:
            # Skip excluded models if specified
            if exclude_model:
                # Handle both single string and list of strings for flexibility
                exclude_list = [exclude_model] if isinstance(exclude_model, str) else exclude_model
                if any(excl in model_file or excl in training_dir for excl in exclude_list):
                    if debug:
                        print(f"Skipping excluded model: {training_dir}/{model_file}")
                    continue
                
            # Filter by model_list if provided
            if model_list:
                full_model_path = f"{training_dir}/{model_file}"
                # Check if any item in model_list matches the directory, file, or full path string.
                match_found = any(
                    item == model_file or item == training_dir or item in full_model_path
                    for item in model_list
                )
                if not match_found:
                    continue

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
        from utils.model_distiller_utils import create_tflite_interpreter
        interpreter = create_tflite_interpreter(model_path)
        return True
    except Exception as e:
        if "Flex" in str(e) or "Select TensorFlow op(s)" in str(e):
            print(f"⚠️  Skipping GPU-only or Flex-dependent model {os.path.basename(model_path)}")
        else:
            print(f"❌ Invalid TFLite model {os.path.basename(model_path)}: {str(e).split(chr(10))[0][:150]}...")
        return False

def _decode_bench_image(image_path, label, fname, target_h, target_w, grayscale):
    import cv2
    import numpy as np
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(image_path, flag)
    if img is None or img.size == 0:
        return None
    img = cv2.resize(img, (target_w, target_h))
    if grayscale and img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    elif not grayscale and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return (img, label, fname)

_cached_test_data = None
_cached_test_data_params = None

def list_available_models(quantized_only=False, subfolder=None, input_dir=None, exclude_model=None, model_list=None):
    """List all available models in a table format and exit."""
    models = get_all_models(quantized_only=quantized_only, subfolder=subfolder, 
                            input_dir=input_dir, exclude_model=exclude_model, 
                            model_list=model_list)
    
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
            
        table_data.append([
            m['directory'],
            m['name'],
            m['type'],
            params_str,
            f"{m['size_kb']:.1f}"
        ])
    
    print("\nAvailable models found:")
    print(tabulate(table_data, headers=headers, tablefmt='simple_grid', stralign='right'))

def load_test_dataset_with_labels(num_samples=0, use_all_datasets=True):
    """
    Load test dataset with proper labels, tracking original filenames.
    Returns: list of (image_array, true_label, filename_no_ext) tuples
    """
    global _cached_test_data, _cached_test_data_params
    import json
    import hashlib
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    current_params = (use_all_datasets, params.NB_CLASSES, params.INPUT_CHANNELS, params.INPUT_WIDTH, params.INPUT_HEIGHT)
    
    if _cached_test_data is not None and _cached_test_data_params == current_params:
        test_data = list(_cached_test_data)
        
        # Apply sampling if requested
        if num_samples > 0 and len(test_data) > num_samples:
            np.random.shuffle(test_data)
            test_data = test_data[:num_samples]
            print(f"📊 Using cached test dataset ({len(test_data)} test samples, limited by --test_images)")
        else:
            print(f"📊 Using cached test dataset (ALL {len(test_data)} available test samples)")
        return test_data
        
    # Generate cache key based on sources and parameters
    sources_fingerprint = json.dumps([{k: v for k, v in s.items()} for s in params.DATA_SOURCES], sort_keys=True)
    cache_str = f"{sources_fingerprint}|classes={params.NB_CLASSES}|channels={params.INPUT_CHANNELS}|w={params.INPUT_WIDTH}|h={params.INPUT_HEIGHT}|all={use_all_datasets}"
    cache_key = hashlib.md5(cache_str.encode()).hexdigest()[:12]
    
    cache_dir = getattr(params, "DATASET_CACHE_DIR", ".dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"bench_cache_{cache_key}.npz")
    
    # Try loading from disk cache
    if os.path.exists(cache_path):
        try:
            print(f"📊 Loading dataset from fast disk cache ({cache_path})...")
            data = np.load(cache_path, allow_pickle=True)
            images, labels, fnames = data["images"], data["labels"], data["fnames"]
            test_data = list(zip(images, labels, fnames))
            
            _cached_test_data = list(test_data)
            _cached_test_data_params = current_params
            
            if num_samples > 0 and len(test_data) > num_samples:
                np.random.shuffle(test_data)
                test_data = test_data[:num_samples]
                print(f"  Using {len(test_data)} test samples (limited by --test_images)")
            else:
                print(f"  Using ALL {len(test_data)} available test samples")
            return test_data
        except Exception as e:
            print(f"⚠️  Could not load disk cache: {e}. Rebuilding...")
            
    print("📊 Loading test dataset with labels from disk...")
    
    tasks = []
    h = params.INPUT_HEIGHT
    w = params.INPUT_WIDTH
    grayscale = params.USE_GRAYSCALE

    for source_config in params.DATA_SOURCES:
        source_type = source_config.get('type', '')
        source_path = source_config.get('path', '')

        if source_type == 'label_file':
            labels_file = source_config.get('labels', 'labels.txt')
            label_file_path = os.path.join(source_path, labels_file)
            images_dir = os.path.join(source_path, 'images')
            if not os.path.exists(label_file_path) or not os.path.exists(images_dir):
                print(f"DEBUG: Source {source_path} missing labels ({label_file_path}) or images dir.")
                continue
            with open(label_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for raw in lines:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) < 2:
                    continue
                fname, label_str = parts[0], parts[1]
                try:
                    label = int(label_str)
                except ValueError:
                    continue
                if label < 0 or label >= params.NB_CLASSES:
                    continue
                img_path = os.path.join(images_dir, fname)
                if not os.path.exists(img_path):
                    continue
                tasks.append((img_path, label, os.path.splitext(fname)[0]))

        elif source_type == 'folder_structure':
            for class_label in range(params.NB_CLASSES):
                class_dir = os.path.join(source_path, str(class_label))
                if not os.path.exists(class_dir):
                    continue
                for fn in os.listdir(class_dir):
                    if not any(fn.lower().endswith(e) for e in ('.jpg', '.jpeg', '.png', '.bmp')):
                        continue
                    img_path = os.path.join(class_dir, fn)
                    tasks.append((img_path, class_label, os.path.splitext(fn)[0]))

    test_data = []
    if tasks:
        n_workers = min(8, os.cpu_count() or 4)
        print(f"  📂 Decoding {len(tasks)} images in parallel...")
        
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_decode_bench_image, path, lbl, fn, h, w, grayscale) for path, lbl, fn in tasks]
            
            done = 0
            for f in as_completed(futures):
                res = f.result()
                if res is not None:
                    test_data.append(res)
                done += 1
                if done % max(1, len(tasks) // 10) == 0:
                    print(f"    {done}/{len(tasks)} loaded...")

    if not test_data:
        print("❌ No data loaded from any source")
        return []

    # Apply weight-based sampling per source (already mixed; use total weight approx)
    np.random.shuffle(test_data)
    
    # Save to disk cache
    try:
        images = np.array([item[0] for item in test_data])
        labels = np.array([item[1] for item in test_data])
        fnames = np.array([item[2] for item in test_data])
        np.savez_compressed(cache_path, images=images, labels=labels, fnames=fnames)
        print(f"💾 Saved {len(test_data)} images to fast disk cache: {cache_path}")
    except Exception as e:
        print(f"⚠️  Could not save disk cache: {e}")

    # Cache the fully loaded and shuffled dataset
    _cached_test_data = list(test_data)
    _cached_test_data_params = current_params

    if num_samples > 0 and len(test_data) > num_samples:
        test_data = test_data[:num_samples]
        print(f"  Using {len(test_data)} test samples (limited by --test_images)")
    else:
        print(f"  Using ALL {len(test_data)} available test samples")

    return test_data

def configure_parameters_for_model(model_name_or_dir, override_classes=None, override_color=None):
    """
    Adjust globals in parameters.py for the specific model.
    Manual overrides from CLI take absolute precedence.
    """
    name_upper = str(model_name_or_dir).upper()
    changed = False
    
    # 1. Handle NB_CLASSES
    if override_classes is not None:
        if params.NB_CLASSES != override_classes:
            params.NB_CLASSES = override_classes
            changed = True
    elif '100CLS' in name_upper and params.NB_CLASSES != 100:
        params.NB_CLASSES = 100
        changed = True
    elif '10CLS' in name_upper and params.NB_CLASSES != 10:
        params.NB_CLASSES = 10
        changed = True
        
    # 2. Handle INPUT_CHANNELS / COLOR
    new_channels = params.INPUT_CHANNELS
    if override_color is not None:
        new_channels = 1 if override_color == 'gray' else 3
    elif 'GRAY' in name_upper:
        new_channels = 1
    elif 'RGB' in name_upper:
        new_channels = 3
        
    if params.INPUT_CHANNELS != new_channels:
        params.INPUT_CHANNELS = new_channels
        changed = True
        
    if changed:
        # Refresh derived parameters (INPUT_SHAPE, USE_GRAYSCALE, etc.)
        params.update_derived_parameters()
        
        # Update labels file in data sources based on new NB_CLASSES
        for source in params.DATA_SOURCES:
            if 'labels' in source or source.get('type') == 'label_file':
                source['labels'] = f'labels_{params.NB_CLASSES}_shuffle.txt'
        
        # Clear the dataset cache so it reloads with new parameters
        from utils.multi_source_loader import clear_cache
        clear_cache()
        print(f"🔄 Reconfigured test environment for {params.NB_CLASSES} classes in {'Grayscale' if params.USE_GRAYSCALE else 'RGB'}")

def test_model_on_dataset(model_path, num_test_images=0, debug=False, use_all_datasets=True, 
                         collect_failed=False, model_name=None, tolerance=0.1):
    """Test a model on random images from dataset and return accuracy and performance metrics"""
    
    try:
        predictor = TFLiteDigitPredictor(model_path)
    except Exception as e:
        print(f"❌ Skipping model {model_path}: {e}")
        return 0.0, 0, 0.0, 0.0, [], []
        
    correct_predictions = 0
    total_tested = 0
    total_inference_time = 0.0
    failed_predictions = []
    all_predictions_lite = []
    
    # Load test data with proper labels
    test_data = load_test_dataset_with_labels(num_test_images, use_all_datasets)
    
    if not test_data:
        print("❌ No test data available")
        return 0.0, 0, 0.0, 0.0, [], []
    
    # Warm-up run to avoid cold start timing issues
    if len(test_data) > 0:
        warmup_image, _, _ = test_data[0]
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
    
    for image, true_label, original_fname in test_iterator:
        if image is None:
            continue
        
        # Predict with timing
        try:
            start_time = time.perf_counter()
            prediction, confidence, _ = predictor.predict(image, debug=debug)
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            
            # Model scale depends on its actual output classes
            model_scale = predictor.num_classes / 10.0
            pred_digit = float(prediction) / model_scale
            true_digit = float(true_label) / model_scale
            
            diff = abs(true_digit - pred_digit) % 10.0
            circular_diff = min(diff, 10.0 - diff)  # dial wraps: 0.0 and 9.9 are 0.1 apart
            
            if circular_diff <= tolerance:
                correct_predictions += 1
                if debug:
                    print(f"✓ Correct: {pred_digit:.1f} (true: {true_digit:.1f}, confidence: {confidence:.3f})")
            else:
                if debug:
                    print(f"✗ Wrong: {prediction} (true: {true_label}, confidence: {confidence:.3f})")
                
                # ALWAYS collect failed prediction for accurate counting
                failed_predictions.append({
                    'image': image,
                    'true_label': true_label,
                    'predicted_label': prediction,
                    'confidence': confidence,
                    'model': model_name or os.path.basename(model_path),
                    'image_source': 'dataset',
                    'original_fname': original_fname,
                    'num_classes': predictor.num_classes
                })
            
            all_predictions_lite.append({
                'true_label': true_label,
                'predicted_label': prediction,
                'model': model_name or os.path.basename(model_path),
                'num_classes': predictor.num_classes,
                'tolerance': tolerance
            })
            
            total_tested += 1
            
        except Exception as e:
            if debug:
                print(f"Prediction error: {e}")
            continue
    
    # Calculate performance metrics
    accuracy = correct_predictions / total_tested if total_tested > 0 else 0.0
    avg_inference_time = total_inference_time / total_tested if total_tested > 0 else 0.0
    inferences_per_second = 1000 / avg_inference_time if avg_inference_time > 0 else 0.0
    
    # CRITICAL: Verify the failed count matches expected
    expected_failed = total_tested - correct_predictions
    if len(failed_predictions) != expected_failed:
        print(f"❌ CRITICAL ERROR: Failed count mismatch for {model_name}!")
        print(f"   Expected failed: {expected_failed} (total_tested: {total_tested} - correct: {correct_predictions})")
        print(f"   Actual failed collected: {len(failed_predictions)}")
        print(f"   This indicates a logic error in the prediction collection!")
    
    if debug:
        print(f"Final accuracy: {accuracy:.3f} ({correct_predictions}/{total_tested})")
        print(f"Failed predictions: {len(failed_predictions)}")
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Inferences per second: {inferences_per_second:.0f}")
    
    return accuracy, total_tested, avg_inference_time, inferences_per_second, failed_predictions, all_predictions_lite

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

        # ── Plot heatmap ────────────────────────────────────────────────────────────
        fig_size = max(10, n // 2)
        plt.figure(figsize=(fig_size, fig_size))
        # Normalize per row so each cell shows the fraction of that true class
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

        # ── Per-class accuracy CSV ───────────────────────────────────────────────────
        per_class_dir = os.path.join(results_dir, "per_class_accuracy")
        os.makedirs(per_class_dir, exist_ok=True)
        per_class_rows = []
        for c in classes:
            ci = idx[c]
            total = cm[ci].sum()
            
            # Count correct using circular_diff
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
                'Accuracy': f"{correct / total:.4f}" if total > 0 else 'N/A'
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

def generate_comparison_graphs(results, quantized_only=True, use_all_datasets=True, output_dir=None):
    """Generate separate comparison graphs for the benchmark results"""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR
        
    # Create graphs directory
    graphs_dir = os.path.join(output_dir, "test_results", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    quant_suffix = "quantized" if quantized_only else "all"
    dataset_suffix = "full" if use_all_datasets else "sampled"
    
    # Prepare data for plotting - filter out models with zero accuracy or speed for cleaner graphs
    plot_data = []
    for result in results:
        accuracy = float(result['Accuracy'])
        inferences_per_second = float(result['Inf/s'])
        
        # Filter out models with 0 accuracy or 0 inferences per second from graphs
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
                'label': label,
                'directory': dir_name,
                'model_name': model_name,
                'accuracy': accuracy * 100,  # Convert to percentage
                'inferences_per_second': inferences_per_second,
                'size_kb': float(result['Size (KB)']),
                'parameters_million': params_val / 1_000_000
            })
    
    if not plot_data:
        print("⚠️  No valid data points to generate comparison graphs (all models had 0 accuracy or 0 inf/s).")
        return []

    # Extract filtered data into lists
    labels = [d['label'] for d in plot_data]
    accuracies = [d['accuracy'] for d in plot_data]
    inferences_per_second = [d['inferences_per_second'] for d in plot_data]
    sizes_kb = [d['size_kb'] for d in plot_data]
    parameters = [d['parameters_million'] for d in plot_data]
    model_names = [d['model_name'] for d in plot_data]
    
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
    graph1_filename = f"accuracy_vs_speed_{quant_suffix}_{dataset_suffix}.png"
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
    graph2_filename = f"accuracy_vs_size_{quant_suffix}_{dataset_suffix}.png"
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
    graph3_filename = f"speed_vs_complexity_{quant_suffix}_{dataset_suffix}.png"
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
                                   label=f"{labels[i].split(chr(10))[0]}\n{model_names[i]}")
                      for i in range(len(labels))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
               borderaxespad=0., fontsize=8)
    
    plt.tight_layout()
    graph4_filename = f"speed_comparison_{quant_suffix}_{dataset_suffix}.png"
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
                                   label=f"{labels[i].split(chr(10))[0]}\n{model_names[i]}")
                      for i in range(len(labels))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
               borderaxespad=0., fontsize=8)
    
    plt.tight_layout()
    graph5_filename = f"accuracy_comparison_{quant_suffix}_{dataset_suffix}.png"
    graph5_path = os.path.join(graphs_dir, graph5_filename)
    plt.savefig(graph5_path, dpi=300, bbox_inches='tight')
    plt.close()
    graph_paths.append(graph5_path)
    
    print(f"📊 Generated {len(graph_paths)} comparison graphs:")
    for path in graph_paths:
        print(f"   📈 {os.path.basename(path)}")
    
    return graph_paths

def test_all_models(num_test_images=0, quantized_only=False, debug=False, 
                    use_all_datasets=True, list_failed=False, save_failed=False,
                    subfolder=None, input_dir=None, exclude_model=None,
                    override_classes=None, override_color=None, model_list=None, tolerance=0.1):
    """Test all valid models with optional subfolder filtering and model exclusion
    
    Args:
        num_test_images: Number of random images to test
        quantized_only: If True, only test quantized models
        debug: Print detailed output
        use_all_datasets: Use images from all datasets, not just the test set
        list_failed: Print a summary of failed predictions per class label
        save_failed: Save failed prediction images
        subfolder: Only test models from this specific subfolder
        model_list: Optional list of specific model names or directories to test
    """
    if input_dir is None:
        input_dir = params.OUTPUT_DIR
        
    models = get_all_models(quantized_only=quantized_only, subfolder=subfolder, 
                            input_dir=input_dir, exclude_model=exclude_model, 
                            debug=debug, model_list=model_list)
    
    if not models:
        if subfolder:
            print(f"No models found in subfolder '{subfolder}'")
        else:
            print("No models found to test.")
        return
    
    # Auto-configure dataset params based on the first model before loading test_data
    if models:
        configure_parameters_for_model(models[0]['directory'], override_classes=override_classes, override_color=override_color)
    
    # Determine test configuration
    subfolder_info = f" in subfolder '{subfolder}'" if subfolder else ""
    if use_all_datasets or num_test_images == 0:
        print(f"\nTesting {len(models)} models{subfolder_info} on ALL available images from all datasets...")
    else:
        print(f"\nTesting {len(models)} models{subfolder_info} on {num_test_images} images...")
    
    print("-" * 80)
    
    results = []
    all_failed_predictions = []  # Collect all failed predictions across models
    all_predictions = []
    
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
        
        # Pass collect_failed=True if either list_failed or save_failed is enabled
        collect_failed = list_failed or save_failed
        
        results_data = test_model_on_dataset(
            model_info['path'], 
            num_test_images=num_test_images,
            debug=debug,
            use_all_datasets=use_all_datasets,
            collect_failed=collect_failed,
            model_name=model_info['name'],
            tolerance=tolerance
        )
        
        if results_data is None or results_data[1] == 0:
            if debug:
                print(f"⚠️  Skipping results for {model_info['name']} due to failure")
            continue
            
        accuracy, tested_count, avg_inference_time, inferences_per_second, failed_predictions, all_predictions_lite = results_data
        
        all_predictions.extend(all_predictions_lite)
        
        # Add model info to failed predictions
        for failure in failed_predictions:
            failure['model_directory'] = model_info['directory']
            all_failed_predictions.append(failure)
        
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
            'Tested': tested_count,
            'Failed_Count': len(failed_predictions)
        })
        
        if debug:
            print(f"✅ Completed: {model_info['directory']}/{model_info['name']} - Accuracy: {accuracy:.3f}")
            if failed_predictions:
                print(f"   Failed predictions: {len(failed_predictions)}")
    
    # Handle failed predictions if requested
    if list_failed and all_failed_predictions:
        csv_path = generate_failed_predictions_csv(all_failed_predictions, input_dir)
    
    if save_failed and all_failed_predictions:
        failed_dir = save_failed_images(all_failed_predictions, input_dir)
    
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
    
    # Simplified console output (modified to show failed count)
    headers = ['Directory', 'Model', 'Type', 'Params', 'Size', 'Accuracy', 'Inf/s', 'Images', 'Failed']
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
            result['Tested'],
            result['Failed_Count']
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
        use_all_datasets=use_all_datasets,
        output_dir=input_dir
    )
    
    # Save full results to CSV
    csv_path = save_results_to_csv(
        results, 
        quantized_only=quantized_only,
        use_all_images=use_all_datasets,
        test_images_count=num_test_images,
        output_dir=input_dir
    )
    
    # Generate markdown report
    if csv_path and graph_paths:
        markdown_path = generate_markdown_report(
            csv_path,
            graph_paths,
            results,
            quantized_only=quantized_only,
            use_all_datasets=use_all_datasets,
            test_images_count=num_test_images,
            output_dir=input_dir
        )
        print(f"📄 Comprehensive markdown report generated: {markdown_path}")
    
    # Generate confusion matrix across all predictions
    if all_predictions:
        print(f"\n🔢 Generating confusion matrix from {len(all_predictions)} predictions...")
        generate_confusion_matrix(all_predictions, output_dir=input_dir)
    
    return results, all_failed_predictions

def save_results_to_csv(results, quantized_only=True, use_all_images=True, test_images_count=0, output_dir=None):
    """Save FULL results to CSV file with all information"""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR
        
    # Create results directory if it doesn't exist
    results_dir = os.path.join(output_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    filename = "model_comparison.csv"
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
    
def calculate_best_iot_model(df, accuracy_weight=0.5, size_weight=0.3, speed_weight=0.2):
    """
    Dynamically calculate the best IoT model based on weighted criteria
    
    Args:
        df: DataFrame with model results
        accuracy_weight: Weight for accuracy (0-1)
        size_weight: Weight for inverse size (0-1) 
        speed_weight: Weight for inference speed (0-1)
    
    Returns:
        Dictionary with best model and analysis
    """
    # Normalize the metrics
    df = df.copy()
    
    # Filter out models with 0 accuracy or 0 inferences per second to avoid division by zero or misleading scores
    df = df[(df['Accuracy'] > 0) & (df['Inferences_per_second'] > 0) & (df['Size_KB'] > 0)].copy()

    if df.empty:
        return {
            'best_overall': None,
            'best_accuracy_small': None,
            'best_speed_small': None,
            'smallest_adequate': None,
            'all_scores': pd.DataFrame(columns=['Model', 'Accuracy', 'Size_KB', 'Inferences_per_second', 'iot_score', 'accuracy_per_kb']),
            'weights_used': {
                'accuracy': accuracy_weight,
                'size': size_weight, 
                'speed': speed_weight
            }
        }

    # Higher accuracy is better
    df['accuracy_norm'] = df['Accuracy'] / df['Accuracy'].max()
    
    # Smaller size is better - use inverse
    df['size_norm'] = (1 / df['Size_KB']) / (1 / df['Size_KB']).max()
    
    # Higher speed is better
    df['speed_norm'] = df['Inferences_per_second'] / df['Inferences_per_second'].max()
    
    # Calculate IoT score with weights
    df['iot_score'] = (
        df['accuracy_norm'] * accuracy_weight +
        df['size_norm'] * size_weight + 
        df['speed_norm'] * speed_weight
    )
    
    # Find the best model
    best_iot_model = df.loc[df['iot_score'].idxmax()]
    
    # Calculate efficiency metrics (only if Parameters column exists)
    df['accuracy_per_kb'] = df['Accuracy'] / df['Size_KB']
    
    if 'Parameters' in df.columns:
        df['accuracy_per_param'] = df['Accuracy'] / df['Parameters']
    else:
        df['accuracy_per_param'] = 0  # Default value if Parameters not available
    
    # Find best in each category with error handling
    small_models = df[df['Size_KB'] <= 100]
    accurate_models = df[df['Accuracy'] >= 0.90]
    
    if not small_models.empty:
        best_accuracy_small = small_models.loc[small_models['Accuracy'].idxmax()]
        best_speed_small = small_models.loc[small_models['Inferences_per_second'].idxmax()]
    else:
        # Fallback to smallest models if none under 100KB
        best_accuracy_small = df.loc[df['Size_KB'].idxmin()]
        best_speed_small = df.loc[df['Inferences_per_second'].idxmax()]
    
    if not accurate_models.empty:
        smallest_adequate = accurate_models.loc[accurate_models['Size_KB'].idxmin()]
    else:
        # Fallback to most accurate if none meet accuracy threshold
        smallest_adequate = df.loc[df['Accuracy'].idxmax()]
    
    # Generate analysis - include all necessary columns
    result_columns = ['Model', 'Accuracy', 'Size_KB', 'Inferences_per_second', 'iot_score', 'accuracy_per_kb']
    if 'Parameters' in df.columns:
        result_columns.append('Parameters')
    if 'accuracy_per_param' in df.columns:
        result_columns.append('accuracy_per_param')
    
    analysis = {
        'best_overall': best_iot_model,
        'best_accuracy_small': best_accuracy_small,
        'best_speed_small': best_speed_small,
        'smallest_adequate': smallest_adequate,
        'all_scores': df[result_columns].sort_values('iot_score', ascending=False),
        'weights_used': {
            'accuracy': accuracy_weight,
            'size': size_weight, 
            'speed': speed_weight
        }
    }
    
    return analysis

def generate_iot_recommendation_section(f, df):
    """Generate dynamic IoT recommendations section"""
    
    # Only calculate if we have multiple models
    if len(df) <= 1:
        f.write("## 💡 IoT-Specific Recommendations\n\n")
        f.write("*Not enough models for comparative IoT analysis*\n\n")
        return
    
    # Use the provided DataFrame which should already have iot_score
    # If it doesn't have iot_score, calculate it
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
    
    # Calculate efficiency safely
    accuracy_per_kb = best_model.get('accuracy_per_kb', best_model['Accuracy'] / best_model['Size_KB'])
    f.write(f"- **Efficiency**: {accuracy_per_kb:.4f} accuracy per KB\n\n")
    
    f.write("#### 📊 IoT Model Comparison (Under 100KB)\n")
    f.write("| Model | Accuracy | Size | Speed | IoT Score | Use Case |\n")
    f.write("|-------|----------|------|-------|-----------|----------|\n")
    
    # Show top small models
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
                
            f.write(f"| {model['Model']} | {model['Accuracy']:.3f} | {model['Size_KB']:.1f}KB | {model['Inferences_per_second']:.0f}/s | {model['iot_score']:.3f} | {use_case} |\n")
    
    f.write("\n")
    
    f.write("#### 🔧 Alternative IoT Scenarios\n\n")
    
    if best_accuracy_small is not None:
        f.write("**For Accuracy-Critical IoT:**\n")
        f.write(f"- **Choice**: {best_accuracy_small['Model']}\n")
        f.write(f"- **Accuracy**: {best_accuracy_small['Accuracy']:.3f} (best under 100KB)\n")
        f.write(f"- **Trade-off**: {best_accuracy_small['Size_KB']:.1f}KB size\n\n")
    else:
        f.write("**For Accuracy-Critical IoT:** *No suitable models found.*\n\n")

    if best_speed_small is not None:
        f.write("**For Speed-Critical IoT:**\n")
        f.write(f"- **Choice**: {best_speed_small['Model']}\n")
        f.write(f"- **Speed**: {best_speed_small['Inferences_per_second']:.0f} inf/s (fastest under 100KB)\n")
        f.write(f"- **Trade-off**: {best_speed_small['Accuracy']:.3f} accuracy\n\n")
    else:
        f.write("**For Speed-Critical IoT:** *No suitable models found.*\n\n")

    if smallest_adequate is not None:
        f.write("**For Memory-Constrained IoT:**\n")
        f.write(f"- **Choice**: {smallest_adequate['Model']}\n")
        f.write(f"- **Size**: {smallest_adequate['Size_KB']:.1f}KB (smallest with ≥85% accuracy)\n")
        f.write(f"- **Trade-off**: {smallest_adequate['Accuracy']:.3f} accuracy\n\n")
    else:
        f.write("**For Memory-Constrained IoT:** *No suitable models found.*\n\n")
    
    f.write("#### 📈 Efficiency Analysis\n")
    f.write("| Model | Acc/KB | Acc/Param | Parameters | Verdict |\n")
    f.write("|-------|--------|-----------|------------|---------|\n")
    
    top_models = df_with_scores.nlargest(5, 'iot_score')
    for _, model in top_models.iterrows():
        acc_per_kb = model['Accuracy'] / model['Size_KB']
        
        # Handle Parameters column safely
        parameters = model.get('Parameters')
        if parameters is not None and parameters > 0:
            acc_per_param = model['Accuracy'] / parameters * 1000000
        else:
            parameters = 'N/A'
            acc_per_param = 'N/A'
        
        verdict = "⚖️ Good"
        if best_model is not None and model['Model'] == best_model['Model']:
            verdict = "🎯 **OPTIMAL**"
        elif model['Size_KB'] > 100:
            verdict = "❌ Too large"
            
        f.write(f"| {model['Model']} | {acc_per_kb:.4f} | {acc_per_param if acc_per_param != 'N/A' else 'N/A'} | {parameters} | {verdict} |\n")
    
    f.write("\n")

def generate_markdown_report(csv_path, graph_paths, results, quantized_only=True, use_all_datasets=True, test_images_count=0, output_dir=None):
    """Generate a comprehensive Markdown report from CSV results and graphs"""
    if output_dir is None:
        output_dir = params.OUTPUT_DIR
    
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Create report directory (now root test_results)
    reports_dir = os.path.join(output_dir, "test_results")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate report filename without timestamp
    report_filename = "readme.md"
    report_path = os.path.join(reports_dir, report_filename)
    
    # Generate markdown content
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Digit Recognition Benchmark Report\n\n")
        
        # Executive Summary with IoT focus
        f.write("## 📊 Executive Summary\n\n")
        
        # Calculate best models dynamically
        iot_analysis = calculate_best_iot_model(df)
        
        if iot_analysis['best_overall'] is not None:
            best_iot = iot_analysis['best_overall']
            best_accuracy = df.loc[df['Accuracy'].idxmax()]
            fastest = df.loc[df['Inferences_per_second'].idxmax()]
            smallest = df.loc[df['Size_KB'].idxmin()]
            
            f.write(f"- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Models Tested**: {len(df)} {'quantized' if quantized_only else 'all'} models\n")
            f.write(f"- **Best IoT Model**: **{best_iot['Model']}** ({best_iot['Size_KB']:.1f}KB, {best_iot['Accuracy']:.3f} acc, {best_iot['Inferences_per_second']:.0f} inf/s)\n")
            f.write(f"- **Best Accuracy**: **{best_accuracy['Model']}** ({best_accuracy['Accuracy']:.3f})\n")
            f.write(f"- **Fastest Model**: **{fastest['Model']}** ({fastest['Inferences_per_second']:.0f} inf/s)\n")
            f.write(f"- **Smallest Model**: **{smallest['Model']}** ({smallest['Size_KB']:.1f} KB)\n\n")
        else:
            f.write(f"- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Models Tested**: {len(df)} {'quantized' if quantized_only else 'all'} models\n")
            f.write("- *No models with valid metrics for comparative analysis.*\n\n")
            
        f.write("## 📈 Performance vs Size\n\n")
        quant_suffix = "quantized" if quantized_only else "all"
        dataset_suffix = "full" if use_all_datasets else "sampled"
        f.write(f"![Accuracy vs Size](graphs/accuracy_vs_size_{quant_suffix}_{dataset_suffix}.png)\n\n")
        
        # Results table
        f.write("## 📋 Detailed Results\n\n")
        f.write("| Model | Size (KB) | Accuracy | Inf/s | Parameters | IoT Score |\n")
        f.write("|-------|-----------|----------|-------|------------|-----------|\n")
        
        # Use the all_scores from analysis which includes iot_score
        scored_df = iot_analysis['all_scores']
        
        for _, row in scored_df.iterrows():
            # Get Parameters safely
            parameters = row.get('Parameters', 'N/A')
            f.write(f"| {row['Model']} | {row['Size_KB']:.1f} | {row['Accuracy']:.3f} | {row['Inferences_per_second']:.0f} | {parameters} | {row['iot_score']:.3f} |\n")
        f.write("\n")
        
        # Dynamic IoT Recommendations - pass the original df to maintain all columns
        generate_iot_recommendation_section(f, df)
        
        f.write("---\n")
        f.write("*Report generated automatically by Digit Recognition Benchmarking Tool*\n")
    
    print(f"📄 Markdown report generated: {report_path}")
    return report_path




def save_failed_images(failed_predictions, output_dir):
    """Save failed prediction images to directory for manual review"""
    failed_dir = os.path.join(output_dir, "failed-predictions")
    os.makedirs(failed_dir, exist_ok=True)
    
    saved_count = 0
    for i, failure in enumerate(failed_predictions):
        try:
            # Extract image data (could be array or path)
            image_data = failure['image']
            true_label = failure['true_label']
            predicted_label = failure['predicted_label']
            confidence = failure['confidence']
            
            # Generate filename: use original filename as stable unique prefix
            original_fname = failure.get('original_fname', 'unknown')
            filename = f"{original_fname}_{predicted_label:.1f}_conf_{confidence:.3f}.jpg"
            filepath = os.path.join(failed_dir, filename)
            
            # If same name (collision), add random 3 digits
            if os.path.exists(filepath):
                import random
                filename = f"{original_fname}_{predicted_label:.1f}_conf_{confidence:.3f}_{random.randint(100, 999)}.jpg"
                filepath = os.path.join(failed_dir, filename)
                
            # Handle different image data types
            if isinstance(image_data, np.ndarray):
                # If it's a numpy array (from dataset)
                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                    # RGB image
                    img = Image.fromarray(image_data.astype(np.uint8))
                elif len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[2] == 1):
                    # Grayscale image
                    if len(image_data.shape) == 3:
                        image_data = image_data.squeeze()
                    img = Image.fromarray(image_data.astype(np.uint8))
                img.save(filepath)
                saved_count += 1
                
            elif isinstance(image_data, str) and os.path.exists(image_data):
                # If it's a file path, copy the file
                shutil.copy2(image_data, filepath)
                saved_count += 1
                
        except Exception as e:
            print(f"Warning: Could not save failed image {i}: {e}")
            continue
    
    print(f"💾 Saved {saved_count} failed images to: {failed_dir}")
    return failed_dir

def generate_failed_predictions_csv(failed_predictions, output_dir):
    """Generate CSV file with details of failed predictions"""
    if not failed_predictions:
        print("No failed predictions to save.")
        return None
    
    # Create results directory
    results_dir = os.path.join(output_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename without timestamp
    csv_filename = "failed_predictions.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    # Prepare data for CSV
    csv_data = []
    for i, failure in enumerate(failed_predictions):
        row = {
            'index': i,
            'true_label': failure['true_label'],
            'predicted_label': failure['predicted_label'],
            'confidence': f"{failure['confidence']:.4f}",
            'model': failure.get('model', 'unknown'),
            'model_directory': failure.get('model_directory', 'unknown'),
            'image_source': failure.get('image_source', 'unknown')
        }
        csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"📊 Failed predictions CSV saved to: {csv_path}")
    print(f"   Total failed predictions: {len(failed_predictions)}")
    
    # Print summary by true label
    if len(failed_predictions) > 0:
        print("\nFailed predictions by true label:")
        failed_by_label = df.groupby('true_label').size()
        for label, count in failed_by_label.items():
            print(f"  Label {label}: {count} failures")
    
    return csv_path

def test_single_model(model_path, num_test_images=0, debug=False, use_all_datasets=True, 
                     list_failed=False, save_failed=False, output_dir=None,
                     override_classes=None, override_color=None, tolerance=0.1):
    """Test a single model and optionally collect failed predictions"""
    predictor = TFLiteDigitPredictor(model_path)
    
    # Auto-configure dataset params
    configure_parameters_for_model(os.path.basename(model_path), override_classes=override_classes, override_color=override_color)
    
    # Load test data with proper labels
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
    
    # Warm-up run
    if len(test_data) > 0:
        warmup_image, _, _ = test_data[0]
        if warmup_image is not None:
            try:
                predictor.predict(warmup_image, debug=False)
            except:
                pass
    
    # Test the model - ALWAYS collect failed predictions for accurate counting
    for i, (image, true_label, original_fname) in enumerate(test_data):
        if image is None:
            continue
        
        try:
            start_time = time.perf_counter()
            prediction, confidence, _ = predictor.predict(image, debug=debug)
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            
            # Compare in digit space (0.0 - 9.9)
            dataset_scale = params.NB_CLASSES / 10.0
            true_digit = float(true_label) / dataset_scale
            
            # Model scale depends on its actual output classes
            model_scale = predictor.num_classes / 10.0
            pred_digit = float(prediction) / model_scale
            
            diff = abs(true_digit - pred_digit) % 10.0
            circular_diff = min(diff, 10.0 - diff)  # dial wraps: 0.0 and 9.9 are 0.1 apart
            
            if circular_diff <= tolerance:
                correct_predictions += 1
                if debug:
                    print(f"✓ {i:4d}: Correct - Pred: {pred_digit:.1f}, True: {true_digit:.1f}, Conf: {confidence:.3f}")
            else:
                if debug:
                    print(f"✗ {i:4d}: Wrong - Pred: {pred_digit:.1f}, True: {true_digit:.1f}, Conf: {confidence:.3f}")
                
                # ALWAYS collect failed prediction for accurate counting
                failed_predictions.append({
                    'image': image,
                    'true_label': round(true_digit, 1),
                    'predicted_label': round(pred_digit, 1),
                    'confidence': confidence,
                    'model': os.path.basename(model_path),
                    'image_source': 'dataset',
                    'index': i,
                    'original_fname': original_fname
                })
            
            all_predictions_lite.append({
                'true_label': round(true_digit, 1),
                'predicted_label': round(pred_digit, 1),
                'model': os.path.basename(model_path), # model_name is not defined here, use os.path.basename(model_path)
                'num_classes': predictor.num_classes,
                'tolerance': tolerance
            })
            
            total_tested += 1
            
        except Exception as e:
            if debug:
                print(f"Prediction error on image {i}: {e}")
            continue
    
    # Calculate metrics
    accuracy = correct_predictions / total_tested if total_tested > 0 else 0.0
    avg_inference_time = total_inference_time / total_tested if total_tested > 0 else 0.0
    inferences_per_second = 1000 / avg_inference_time if avg_inference_time > 0 else 0.0
    
    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS for {os.path.basename(model_path)}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_tested})")
    print(f"Failed predictions: {len(failed_predictions)}")
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    print(f"Inferences per second: {inferences_per_second:.0f}")
    
    # CRITICAL: Verify the failed count matches expected
    expected_failed = total_tested - correct_predictions
    if len(failed_predictions) != expected_failed:
        print(f"❌ CRITICAL ERROR: Failed count mismatch!")
        print(f"   Expected failed: {expected_failed} (total_tested: {total_tested} - correct: {correct_predictions})")
        print(f"   Actual failed collected: {len(failed_predictions)}")
        print(f"   This indicates a logic error in the prediction collection!")
        print(f"   Please investigate the code - data may be incomplete!")
        
        # Don't proceed with failed prediction analysis if counts don't match
        if list_failed or save_failed:
            print(f"⚠️  Skipping failed prediction export due to count mismatch")
            list_failed = False
            save_failed = False
    
    # Handle failed predictions export if requested
    if list_failed and failed_predictions:
        generate_failed_predictions_csv(failed_predictions, output_dir)
    
    if save_failed and failed_predictions:
        save_failed_images(failed_predictions, output_dir)
    
    # Print failed predictions summary (only if counts match)
    if failed_predictions and len(failed_predictions) == expected_failed:
        print(f"\nFailed predictions breakdown:")
        failed_by_true_label = {}
        for failure in failed_predictions:
            true_label = failure['true_label']
            predicted_label = failure['predicted_label']
            if true_label not in failed_by_true_label:
                failed_by_true_label[true_label] = {}
            if predicted_label not in failed_by_true_label[true_label]:
                failed_by_true_label[true_label][predicted_label] = 0
            failed_by_true_label[true_label][predicted_label] += 1
        
        for true_label in sorted(failed_by_true_label.keys()):
            total_for_label = sum(failed_by_true_label[true_label].values())
            print(f"  True label {true_label}: {total_for_label} misclassifications")
            for pred_label, count in sorted(failed_by_true_label[true_label].items()):
                percentage = (count / total_for_label) * 100
                print(f"    → as {pred_label}: {count} times ({percentage:.1f}%)")
                
    # Generate confusion matrix for single model
    if all_predictions_lite:
        generate_confusion_matrix(all_predictions_lite, output_dir=params.OUTPUT_DIR)
    
    return accuracy, total_tested, avg_inference_time, inferences_per_second, failed_predictions, all_predictions_lite

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Digit Recognition Benchmarking System')
    
    # Optional overrides for dataset configuration (auto-detected if omitted)
    parser.add_argument('--classes', type=int, choices=[10, 100], 
                        help='Force the number of classes (10 or 100). Auto-detected from folder name if omitted (e.g., 10cls).')
    parser.add_argument('--color', type=str, choices=['rgb', 'gray'], 
                        help='Force a specific color mode. Auto-detected from folder name if omitted (e.g., GRAY).')
    
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
    parser.add_argument('--input_dir', type=str, default='exported_models', # Changed default to 'exported_models' string
                        help='Base directory to search for models (default: exported_models)')
    parser.add_argument('--subfolder', type=str, 
                        help='Restrict search to a specific subfolder within the input directory.')
    parser.add_argument('--exclude_model', '--exclude_models',
                        dest='exclude_model',
                        type=str, nargs='+', default=None,
                        help='Exclude models containing these strings from the benchmark.')
    parser.add_argument('--quantized', action='store_true', default=True, 
                        help='Only include quantized models (True by default).')
    parser.add_argument('--no-quantized', action='store_false', dest='quantized', 
                        help='Include all models, including floating-point versions.')
    
    # Dataset and Testing Configuration
    parser.add_argument('--test_images', type=int, default=0, 
                        help='Number of images to test per model. 0 means use the entire dataset (default: 0).')
    parser.add_argument('--all_datasets', action='store_true', default=True, 
                        help='Use images from all available data sources (True by default).')
    parser.add_argument('--no-all_datasets', action='store_false', dest='all_datasets', 
                        help='Restrict testing to the standard test set only.')
    
    # Output and Debugging
    parser.add_argument('--list-failed', action='store_true', 
                        help='Generate a detailed CSV file containing information on all misclassifications.')
    parser.add_argument('--save-failed', action='store_true', 
                        help='Save images that were misclassified into a "failed-predictions" folder.')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable verbose output for debugging model predictions and data loading.')
    parser.add_argument('--tolerance', type=float, default=0.1, 
                        help='Acceptable error tolerance in decimal scale (default: 0.1). E.g. +-0.1 allows +-1 class for 100 classes.')
    
    args, unknown = parser.parse_known_args()
    
    # We must explicitly define the environment before initializing `parameters.py`
    # Otherwise `import parameters` will block awaiting standard IO.
    if args.classes:
        os.environ['DIGIT_NB_CLASSES'] = str(args.classes)
    if args.color:
        if args.color.lower() == 'gray':
            os.environ['DIGIT_INPUT_CHANNELS'] = '1'
        elif args.color.lower() == 'rgb':
            os.environ['DIGIT_INPUT_CHANNELS'] = '3'
            
    # Now map the delayed imports globally
    global params
    global preprocess_for_inference
    import parameters as p
    from utils.preprocess import preprocess_for_inference as pfio
    params = p
    preprocess_for_inference = pfio
    
    # Use output dir from params if nothing was specified
    if args.input_dir == 'exported_models':
        args.input_dir = params.OUTPUT_DIR
    
    if args.list:
        list_available_models(quantized_only=args.quantized, subfolder=args.subfolder, 
                              input_dir=args.input_dir, exclude_model=args.exclude_model,
                              model_list=args.model_list)
        return
    
    # Handle single model prediction (highest priority)
    if args.model:
        if os.path.isfile(args.model) and args.model.endswith('.tflite'):
            model_path = args.model
        else:
            model_path = find_model_path(args.model, input_dir=args.input_dir)
            
        if model_path is None:
            print(f"❌ Model '{args.model}' not found in {args.input_dir} (or as an exact path)!")
            print("Available models:")
            list_available_models(quantized_only=args.quantized, subfolder=args.subfolder, input_dir=args.input_dir, exclude_model=args.exclude_model)
            return
        
        test_single_model(
            model_path=model_path,
            num_test_images=args.test_images,
            debug=args.debug,
            use_all_datasets=args.all_datasets,
            list_failed=args.list_failed,
            save_failed=args.save_failed,
            output_dir=args.input_dir,
            override_classes=args.classes,
            override_color=args.color,
            tolerance=args.tolerance
        )
        return
    
    # Handle test_all or model_list mode
    elif args.test_all or args.model_list:
        test_all_models(
            num_test_images=args.test_images, 
            quantized_only=args.quantized, 
            debug=args.debug,
            use_all_datasets=args.all_datasets,
            list_failed=args.list_failed,
            save_failed=args.save_failed,
            subfolder=args.subfolder,
            input_dir=args.input_dir,
            exclude_model=args.exclude_model,
            override_classes=args.classes,
            override_color=args.color,
            model_list=args.model_list,
            tolerance=args.tolerance
        )
        return
    
    # Default behavior (when no specific mode is specified)
    if args.debug:
        print("Debug mode requires either --model or --test_all")
        return
    
    # If no arguments provided, show help and run default benchmark
    print("No specific mode selected. Running default benchmark...")
    print("Use --list to see available models, --model to test specific model, or --test_all for full benchmark.")
    print("-" * 60)
    
    test_all_models(
        quantized_only=args.quantized, 
        num_test_images=args.test_images,
        debug=args.debug,
        use_all_datasets=args.all_datasets,
        list_failed=args.list_failed,
        save_failed=args.save_failed,
        subfolder=args.subfolder,
        input_dir=args.input_dir,
        override_classes=args.classes,
        override_color=args.color,
        model_list=args.model_list,
        tolerance=args.tolerance
    )

if __name__ == "__main__":
    main()
    
# py bench_predict.py --test_all

# py bench_predict.py --model digit_recognizer_v4.tflite --list-failed --save-failed
# py bench_predict.py --test_all --input_dir exported_models\100cls_RGB