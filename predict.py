# predict.py
import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from utils.preprocess import predict_single_image
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
    
    def predict(self, image):
        """Predict digit from image using TFLite"""
        # Preprocess image
        processed_image = predict_single_image(image)
        
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
        
        # Get prediction and confidence
        prediction = np.argmax(output_data[0])
        confidence = np.max(output_data[0])
        
        return prediction, confidence, output_data[0]

def load_random_image_from_dataset():
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
    image = cv2.imread(random_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {random_image_path}")
        return None
    
    return image

def load_image_from_path(image_path):
    """Load image from specified path"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    print(f"Loaded image: {image_path}")
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

def main():
    """Simple prediction function"""
    parser = argparse.ArgumentParser(description='Digit Recognition Prediction')
    parser.add_argument('--img', type=str, help='Path to input image for prediction')
    parser.add_argument('--model', type=str, help='Model name to use for prediction')
    
    args = parser.parse_args()
    
    # Find model path
    model_path = find_model_path(args.model)
    if not model_path:
        return
    
    # Load predictor
    predictor = TFLiteDigitPredictor(model_path)
    
    # Load image
    if args.img:
        image = load_image_from_path(args.img)
    else:
        print("No image specified, loading random image from dataset...")
        image = load_random_image_from_dataset()
    
    if image is None:
        print("Failed to load image")
        return
    
    # Perform prediction
    prediction, confidence, raw_output = predictor.predict(image)
    
    print(f"\n=== PREDICTION RESULT ===")
    print(f"Predicted digit: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All probabilities: {[f'{x:.4f}' for x in raw_output]}")

if __name__ == "__main__":
    main()