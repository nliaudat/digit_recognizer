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
    """Get all available models"""
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
            all_models.append({
                'path': model_path,
                'name': model_file,
                'directory': training_dir,
                'size_kb': model_size,
                'type': model_type
            })
    
    return all_models

def test_model_on_dataset(model_path, rgb_mode=False, num_test_images=100, debug=False):
    """Test a model on random images from dataset and return accuracy"""
    predictor = TFLiteDigitPredictor(model_path)
    correct_predictions = 0
    total_tested = 0
    
    # Get dataset path
    if not params.DATA_SOURCES:
        print("No data sources found in parameters.py")
        return 0.0, 0
    
    dataset_path = params.DATA_SOURCES[0]['path']
    
    # Collect all test images with their true labels
    test_data = []
    for digit in range(10):
        digit_folder = os.path.join(dataset_path, str(digit))
        if not os.path.exists(digit_folder):
            continue
            
        # Get images for this digit
        image_files = [f for f in os.listdir(digit_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue
            
        # Test up to num_test_images//10 per digit
        test_per_digit = max(1, num_test_images // 10)
        test_images = np.random.choice(image_files, min(test_per_digit, len(image_files)), replace=False)
        
        for image_file in test_images:
            test_data.append((os.path.join(digit_folder, image_file), digit))
    
    # Shuffle test data
    np.random.shuffle(test_data)
    
    # Test with progress bar
    for image_path, true_digit in tqdm(test_data, desc=f"Testing {os.path.basename(model_path)}", leave=False):
        # Load image
        if rgb_mode:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            continue
        
        # Predict
        try:
            prediction, confidence, _ = predictor.predict(image, rgb_mode=rgb_mode, debug=debug)
            if prediction == true_digit:
                correct_predictions += 1
            total_tested += 1
        except Exception as e:
            continue
    
    accuracy = correct_predictions / total_tested if total_tested > 0 else 0.0
    return accuracy, total_tested

def test_all_models(rgb_mode=False, quantized_only=False, num_test_images=100, debug=False):
    """Test all available models and print summary table"""
    models = get_all_models(quantized_only=quantized_only)
    
    if not models:
        print("No models found to test.")
        return
    
    print(f"\nTesting {len(models)} models on {num_test_images} images...")
    print("Mode:", "RGB" if rgb_mode else "Grayscale")
    print("-" * 80)
    
    results = []
    
    # Test all models with progress bar
    for model_info in tqdm(models, desc="Testing all models"):
        accuracy, tested_count = test_model_on_dataset(
            model_info['path'], 
            rgb_mode=rgb_mode, 
            num_test_images=num_test_images,
            debug=debug
        )
        
        results.append({
            'Model': model_info['name'],
            'Directory': model_info['directory'],
            'Type': model_info['type'],
            'Size (KB)': f"{model_info['size_kb']:.1f}",
            'Accuracy': f"{accuracy:.4f}",
            'Tested Images': tested_count
        })
    
    # Sort by accuracy descending
    results.sort(key=lambda x: float(x['Accuracy']), reverse=True)
    
    # Print results table
    print(f"\n{'='*80}")
    print(f"SUMMARY RESULTS ({'RGB' if rgb_mode else 'Grayscale'} mode)")
    print(f"{'='*80}")
    print(tabulate(results, headers='keys', tablefmt='grid', stralign='right'))
    
    # Print best model
    if results and float(results[0]['Accuracy']) > 0:
        best = results[0]
        print(f"\n🎯 BEST MODEL: {best['Directory']}/{best['Model']} (Accuracy: {best['Accuracy']})")

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

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Digit Recognition Prediction')
    parser.add_argument('--img', type=str, help='Path to input image for prediction')
    parser.add_argument('--model', type=str, help='Model name to use for prediction (can be: model filename, training directory name, or partial match)')
    parser.add_argument('--quantized', action='store_true', help='Use only quantized models')
    parser.add_argument('--RGB', action='store_true', help='Process image as RGB instead of grayscale')
    parser.add_argument('--test_all', action='store_true', help='Test all available models and print accuracy summary')
    parser.add_argument('--test_images', type=int, default=1000, help='Number of test images per model (default: 100)')
    parser.add_argument('--debug', action='store_true', help='Debug model output interpretation')
    parser.add_argument('--list', action='store_true', help='List all available models')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models(quantized_only=args.quantized)
        return
    
    if args.test_all:
        test_all_models(
            rgb_mode=args.RGB, 
            quantized_only=args.quantized, 
            num_test_images=args.test_images,
            debug=args.debug  # Pass debug flag to testing
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