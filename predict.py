# predict.py
import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from utils.preprocess import preprocess_single_image
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
        print(f"Output type: {self.output_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        
        # Print quantization info
        if self.input_details[0]['quantization'][0] != 0:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
    
    def predict(self, image):
        """Predict digit from image using TFLite"""
        # Preprocess image
        processed_image = preprocess_single_image(image)
        
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
        
        # DEBUG: Print raw output to understand the distribution
        print(f"Raw output: {output_data[0]}")
        print(f"Output sum: {np.sum(output_data[0]):.6f}")
        
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
    """Find the model path based on model name or use default behavior"""
    # Look for training directories
    # training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if d.startswith('training_')]
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR)]
    if not training_dirs:
        print("No training directories found. Please run train.py first.")
        return None
    
    if model_name:
        # Search for the specific model in training directories
        for training_dir in sorted(training_dirs, reverse=True):
            # Try multiple possible model file locations and names
            possible_paths = [
                # Model with specific name in training directory
                os.path.join(params.OUTPUT_DIR, training_dir, f"{model_name}.tflite"),
                # Default TFLITE_FILENAME in training directory
                os.path.join(params.OUTPUT_DIR, training_dir, params.TFLITE_FILENAME),
                # Model in a subdirectory matching the model name
                os.path.join(params.OUTPUT_DIR, training_dir, "models", f"{model_name}.tflite"),
            ]
            
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    print(f"Found model: {model_path}")
                    return model_path
        
        print(f"Model '{model_name}' not found in any training directory")
        print("Available training directories:")
        for training_dir in training_dirs:
            print(f"  - {training_dir}")
            training_dir_path = os.path.join(params.OUTPUT_DIR, training_dir)
            if os.path.exists(training_dir_path):
                files = [f for f in os.listdir(training_dir_path) if f.endswith('.tflite')]
                for file in files:
                    print(f"    └── {file}")
        return None
    else:
        # Use default behavior - latest training directory with default TFLITE_FILENAME
        latest_training = sorted(training_dirs)[-1]
        
        # Try multiple possible model file locations
        possible_paths = [
            os.path.join(params.OUTPUT_DIR, latest_training, params.TFLITE_FILENAME),
            os.path.join(params.OUTPUT_DIR, latest_training, "final_quantized.tflite"),
            os.path.join(params.OUTPUT_DIR, latest_training, "final_float.tflite"),
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                print(f"Using model: {model_path}")
                return model_path
        
        # If no specific model found, look for any .tflite file in the latest directory
        latest_dir_path = os.path.join(params.OUTPUT_DIR, latest_training)
        tflite_files = [f for f in os.listdir(latest_dir_path) if f.endswith('.tflite')]
        
        if tflite_files:
            model_path = os.path.join(latest_dir_path, tflite_files[0])
            print(f"Using first available model: {model_path}")
            return model_path
        
        print(f"No TFLite model found in: {latest_dir_path}")
        print("Available files:")
        for file in os.listdir(latest_dir_path):
            print(f"  - {file}")
        return None

def list_available_models():
    """List all available models in training directories"""
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if d.startswith('training_')]
    if not training_dirs:
        print("No training directories found.")
        return
    
    print("Available models:")
    print("-" * 50)
    
    for training_dir in sorted(training_dirs, reverse=True):
        training_path = os.path.join(params.OUTPUT_DIR, training_dir)
        tflite_files = [f for f in os.listdir(training_path) if f.endswith('.tflite')]
        
        if tflite_files:
            print(f"\n{training_dir}:")
            for model_file in tflite_files:
                model_path = os.path.join(training_path, model_file)
                model_size = os.path.getsize(model_path) / 1024
                print(f"  └── {model_file} ({model_size:.1f} KB)")

def debug_model_output():
    """Debug function to test model output interpretation"""
    model_path = find_model_path()
    if not model_path:
        return
    
    predictor = TFLiteDigitPredictor(model_path)
    
    # Test with multiple random images
    print("\n=== DEBUGGING MODEL OUTPUT ===")
    for i in range(3):
        print(f"\n--- Test {i+1} ---")
        test_image = np.random.randint(0, 255, (params.INPUT_HEIGHT, params.INPUT_WIDTH), dtype=np.uint8)
        prediction, confidence, raw_output = predictor.predict(test_image)
        
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
    parser.add_argument('--model', type=str, help='Model name to use for prediction')
    parser.add_argument('--debug', action='store_true', help='Debug model output interpretation')
    parser.add_argument('--list', action='store_true', help='List all available models')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if args.debug:
        debug_model_output()
        return
    
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