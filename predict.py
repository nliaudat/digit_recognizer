# predict.py
import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from utils.preprocess import preprocess_for_inference
from pathlib import Path
import parameters as params

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
    
    def debug_preprocessing(self, image, processed_image):
        """Debug preprocessing steps"""
        print(f"\n=== DEBUG PREPROCESSING ===")
        print(f"Original image - shape: {image.shape}, dtype: {image.dtype}")
        print(f"Original image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Processed image - shape: {processed_image.shape}, dtype: {processed_image.dtype}")
        print(f"Processed image range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        
        # Check model expectations
        expected_shape = self.input_details[0]['shape'][1:3]  # (height, width)
        expected_dtype = self.input_details[0]['dtype']
        print(f"Model expects - shape: {expected_shape}, dtype: {expected_dtype}")
        
        # # Verify preprocessing matches training
        # if processed_image.max() > 1.0:
            # print("⚠️  WARNING: Image not normalized to [0,1] range!")
        # if processed_image.dtype != np.float32:
            # print("⚠️  WARNING: Image not in float32 format!")
            
        # -----------------------------------------------------------------
        #  Sanity check – the image must already be in the dtype the model
        #  expects (uint8 for quantised models, float32 otherwise).
        # -----------------------------------------------------------------
        if params.QUANTIZE_MODEL:
            if processed_image.dtype != np.uint8:
                raise ValueError(
                    "Pre processed image dtype mismatch: "
                    f"expected uint8, got {processed_image.dtype}"
                )
        else:
            if processed_image.dtype != np.float32:
                raise ValueError(
                    "Pre processed image dtype mismatch: "
                    f"expected float32, got {processed_image.dtype}"
                )

    def predict(self, image, debug=False):
        """Predict digit from image using TFLite"""
        # Preprocess image
        processed_image = preprocess_for_inference(image)
        
        if debug:
            self.debug_preprocessing(image, processed_image)
        
        print(f"After preprocessing - shape: {processed_image.shape}, dtype: {processed_image.dtype}")
        
        # Handle channel mismatch - if model expects 3 channels but we have 1
        expected_channels = self.input_details[0]['shape'][3]
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 1 and expected_channels == 3:
            print("Converting grayscale to 3-channel by repeating channels")
            processed_image = np.repeat(processed_image, 3, axis=2)
        
        # Add batch dimension if not already present
        if len(processed_image.shape) == 3:
            input_data = np.expand_dims(processed_image, axis=0)
        else:
            input_data = processed_image
        
        print(f"After batch dimension - shape: {input_data.shape}, dtype: {input_data.dtype}")
        
        # Handle quantization if needed
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = input_data.astype(np.uint8)
        elif self.input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            print(f"Quantization params - scale: {input_scale}, zero_point: {input_zero_point}")
            # For int8 quantization, we need to quantize the float32 input
            if input_data.dtype == np.float32:
                input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
            else:
                input_data = input_data.astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)
        
        print(f"After quantization handling - shape: {input_data.shape}, dtype: {input_data.dtype}")
        print(f"Expected input shape: {self.input_details[0]['shape']}, dtype: {self.input_details[0]['dtype']}")
        
        # Verify shape matches expected input shape
        expected_shape = self.input_details[0]['shape']
        if input_data.shape != tuple(expected_shape):
            print(f"Shape mismatch! Got {input_data.shape}, expected {tuple(expected_shape)}")
            # Try to reshape if possible
            if input_data.size == np.prod(expected_shape):
                input_data = input_data.reshape(expected_shape)
                print(f"Reshaped to: {input_data.shape}")
            else:
                print("Cannot reshape - total elements don't match")
                # Return default values instead of None
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
            print(f"Error during inference: {e}")
            # Return default values instead of None
            return -1, 0.0, np.zeros(self.output_details[0]['shape'][-1], dtype=np.float32)


def load_random_image_from_dataset(input_channels):
    """Load a random image from the first available data source"""
    if not params.DATA_SOURCES:
        print("No data sources found in parameters.py")
        return None
    
    # Use the first data source
    data_source = params.DATA_SOURCES[0]
    dataset_path = data_source['path']
    
    # Convert to Path object for path operations
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        return None
    
    # Collect all images from the dataset and in subfolders
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # Find all image files first
    image_files = []
    for file_path in dataset_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    if not image_files:
        print(f"No images found in {dataset_path}")
        return None
    
    # Select a random image
    random_image_path = np.random.choice(image_files)
    print(f"Loading random image: {random_image_path}")
    
    # Load image based on model's input requirements
    if input_channels == 1:
        # Model expects grayscale - load as 2D array
        image = cv2.imread(str(random_image_path), cv2.IMREAD_GRAYSCALE)
    else:
        # Model expects color (RGB) - load as 3D array with 3 channels
        image = cv2.imread(str(random_image_path), cv2.IMREAD_COLOR)
        if image is not None:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        print(f"Failed to load image: {random_image_path}")
        return None
    
    print(f"Loaded image shape: {image.shape}")
    return image

def load_image_from_path(image_path, input_channels):
    """Load image from specified path based on model's input requirements"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
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
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    print(f"Loaded image: {image_path}, shape: {image.shape}")
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
        
        print(f"Searching for model: {model_name}")
        
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
                    print(f"Found in directory '{best_match}': {quantized_models[0]}")
                    return model_path
                else:
                    model_path = os.path.join(training_path, tflite_files[0])
                    print(f"Found in directory '{best_match}': {tflite_files[0]}")
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
                    print(f"Found: {training_dir}/{model_file}")
                    return model_path
        
        print(f"Model or directory '{model_name}' not found in any training directory.")
        print("Available models:")
        for training_dir in training_dirs:
            training_path = os.path.join(params.OUTPUT_DIR, training_dir)
            tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
            if tflite_files:
                print(f"  {training_dir}:")
                for model_file in tflite_files:
                    print(f"    └── {model_file}")
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
                print(f"Using latest quantized model: {latest_training}/{quantized_models[0]}")
                return model_path
            else:
                model_path = os.path.join(latest_dir_path, tflite_files[0])
                print(f"Using latest model: {latest_training}/{tflite_files[0]}")
                return model_path
        
        print(f"No TFLite model found in: {latest_dir_path}")
        return None

def test_model_with_known_image(predictor, input_channels):
    """Test the model with a simple known pattern to verify it's working"""
    print("\n🧪 TESTING WITH KNOWN PATTERN")
    
    # Create a simple test image (white digit on black background)
    test_image = np.zeros((32, 20), dtype=np.uint8)
    
    # Create a simple pattern that should be recognizable as some digit
    # For example, a vertical line in the middle (could be '1')
    test_image[5:27, 9:11] = 255
    
    print(f"Test image shape: {test_image.shape}, range: [{test_image.min()}, {test_image.max()}]")
    
    # Predict
    prediction, confidence, raw_output = predictor.predict(test_image, debug=True)
    
    print(f"Test prediction: {prediction}, confidence: {confidence:.4f}")
    return prediction, confidence
    
def debug_model_expectations(predictor):
    """Debug what the model actually expects - CORRECTED VERSION"""
    print("\n🔍 MODEL EXPECTATIONS:")
    print("=" * 50)
    
    input_details = predictor.input_details[0]
    print(f"Input dtype: {input_details['dtype']}")
    print(f"Input shape: {input_details['shape']}")
    print(f"Input name: {input_details['name']}")
    
    # Check quantization parameters
    if 'quantization' in input_details and input_details['quantization'] != (0, 0):
        scale, zero_point = input_details['quantization']
        print(f"Quantization - scale: {scale}, zero_point: {zero_point}")
        
        # CORRECTED: Show expected integer range and corresponding real value range
        input_dtype = input_details['dtype']
        
        if input_dtype == np.uint8:
            int_min, int_max = 0, 255
            real_min = scale * (int_min - zero_point)
            real_max = scale * (int_max - zero_point)
            print(f"Expected integer input: uint8 [{int_min}, {int_max}]")
            print(f"Corresponding real values: [{real_min:.6f}, {real_max:.6f}]")
            
        elif input_dtype == np.int8:
            int_min, int_max = -128, 127
            real_min = scale * (int_min - zero_point)
            real_max = scale * (int_max - zero_point)
            print(f"Expected integer input: int8 [{int_min}, {int_max}]")
            print(f"Corresponding real values: [{real_min:.6f}, {real_max:.6f}]")
            
        else:
            # For float models or unexpected types
            print(f"Expected input: {input_dtype} (no quantization)")
            
        # Show the dequantization formula
        print(f"Dequantization formula: real_value = {scale} * (quantized_value - {zero_point})")
        
    else:
        print("No quantization (float model)")
        print(f"Expected input: float32 [0.0, 1.0] (normalized)")
    
    # Check current parameters.py settings
    print(f"\n📊 PARAMETERS.PY SETTINGS:")
    print(f"QUANTIZE_MODEL: {params.QUANTIZE_MODEL}")
    print(f"USE_QAT: {params.USE_QAT}")
    print(f"ESP_DL_QUANTIZE: {params.ESP_DL_QUANTIZE}")

def main():
    """Simple prediction function"""
    parser = argparse.ArgumentParser(description='Digit Recognition Prediction')
    parser.add_argument('--img', type=str, help='Path to input image for prediction')
    parser.add_argument('--model', type=str, help='Model name to use for prediction')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--test', action='store_true', help='Test with known pattern first')
    
    args = parser.parse_args()
    
    # Find model path
    model_path = find_model_path(args.model)
    if not model_path:
        return
    
    # Load predictor
    predictor = TFLiteDigitPredictor(model_path)
    debug_model_expectations(predictor) 
    
    # Get model's input requirements
    input_shape = predictor.input_details[0]['shape']
    input_channels = input_shape[3]
    print(f"Model expects input with {input_channels} channel(s)")
    
    # Test with known pattern first if requested
    if args.test:
        test_model_with_known_image(predictor, input_channels)
    
    # Load image
    if args.img:
        image = load_image_from_path(args.img, input_channels)
    else:
        print("No image specified, loading random image from dataset...")
        image = load_random_image_from_dataset(input_channels)
    
    if image is None:
        print("Failed to load image")
        return
    
    # Perform prediction
    prediction, confidence, raw_output = predictor.predict(image, debug=args.debug)
    
    print(f"\n=== PREDICTION RESULT ===")
    print(f"Predicted digit: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    # Check if raw_output is valid before trying to iterate
    if raw_output is not None and hasattr(raw_output, '__iter__'):
        print(f"All probabilities: {[f'{x:.4f}' for x in raw_output]}")
    else:
        print("All probabilities: [Prediction failed]")

if __name__ == "__main__":
    main()