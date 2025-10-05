import tensorflow as tf
import numpy as np
import cv2
import os
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
        
        # Handle quantization if needed
        if self.input_details[0]['dtype'] == np.int8:
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
        prediction = np.argmax(output_data[0])
        confidence = np.max(output_data[0])
        
        return prediction, confidence

def test_prediction():
    """Test the TFLite model with a sample"""
    # Look for the latest training directory
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if d.startswith('training_')]
    if not training_dirs:
        print("No training directories found. Please run train.py first.")
        return
    
    latest_training = sorted(training_dirs)[-1]
    model_path = os.path.join(params.OUTPUT_DIR, latest_training, params.TFLITE_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"TFLite model not found: {model_path}")
        return
    
    predictor = TFLiteDigitPredictor(model_path)
    
    # Create a test image (replace with real image)
    test_image = np.random.randint(0, 255, (params.INPUT_HEIGHT, params.INPUT_WIDTH, 1), dtype=np.uint8)
    
    prediction, confidence = predictor.predict(test_image)
    print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")

if __name__ == "__main__":
    test_prediction()