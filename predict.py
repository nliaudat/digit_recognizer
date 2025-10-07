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
        
        # Handle quantization if needed - FIXED VERSION
        if self.input_details[0]['dtype'] == np.uint8:
            # For uint8 models, we need to convert to uint8 without scaling
            # Assuming preprocess_single_image returns values in [0, 255] range
            input_data = input_data.astype(np.uint8)
        elif self.input_details[0]['dtype'] == np.int8:
            # For int8 models, apply quantization
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
        else:
            # For float32 models
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
        
        prediction = np.argmax(output_data[0])
        confidence = np.max(output_data[0])
        
        return prediction, confidence

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
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if d.startswith('training_')]
    if not training_dirs:
        print("No training directories found. Please run train.py first.")
        return None
    
    if model_name:
        # Search for the specific model
        for training_dir in sorted(training_dirs, reverse=True):
            model_path = os.path.join(params.OUTPUT_DIR, training_dir, f"{model_name}.tflite")
            if os.path.exists(model_path):
                return model_path
        
        # If not found, try with the default TFLITE_FILENAME
        for training_dir in sorted(training_dirs, reverse=True):
            model_path = os.path.join(params.OUTPUT_DIR, training_dir, params.TFLITE_FILENAME)
            if os.path.exists(model_path):
                print(f"Specific model '{model_name}' not found, using default: {params.TFLITE_FILENAME}")
                return model_path
        
        print(f"Model '{model_name}' not found in any training directory")
        return None
    else:
        # Use default behavior - latest training directory with default TFLITE_FILENAME
        latest_training = sorted(training_dirs)[-1]
        model_path = os.path.join(params.OUTPUT_DIR, latest_training, params.TFLITE_FILENAME)
        
        if not os.path.exists(model_path):
            print(f"TFLite model not found: {model_path}")
            return None
        
        return model_path

def main():
    """Main function with command line arguments"""
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
    prediction, confidence = predictor.predict(image)
    print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()