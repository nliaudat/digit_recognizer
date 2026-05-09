"""
benchmark/predictor.py
======================
Pure TFLite inference logic extracted from bench_predict.py.

Provides:
  - TFLiteDigitPredictor — load and run inference on TFLite models
  - get_model_metadata — extract model metadata without loading
  - is_valid_tflite_model — validate TFLite file integrity
"""

import logging
import os
from pathlib import Path

import numpy as np

from utils.model_distiller_utils import create_tflite_interpreter
from utils.preprocess import preprocess_for_inference

logger = logging.getLogger(__name__)


class TFLiteDigitPredictor:
    """Load a TFLite model and run inference on digit images."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model()

    def load_model(self):
        """Load TFLite model"""
        logger.info(f"Loading TFLite model: {self.model_path}")

        self.interpreter = create_tflite_interpreter(self.model_path)

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        logger.info(f"Input shape: {self.input_details[0]['shape']}")
        logger.info(f"Input type: {self.input_details[0]['dtype']}")
        logger.info(f"Output shape: {self.output_details[0]['shape']}")

    def predict(self, image, debug=False):
        """Predict digit from image using TFLite"""
        # Preprocess image
        processed_image = preprocess_for_inference(image)

        # Handle channel mismatch
        expected_channels = self.input_details[0]['shape'][3]
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 1 and expected_channels == 3:
            processed_image = np.repeat(processed_image, 3, axis=2)

        # Add batch dimension if not already present
        if len(processed_image.shape) == 3:
            processed_image = np.expand_dims(processed_image, axis=0)

        # Handle quantization
        input_dtype = self.input_details[0]['dtype']
        if input_dtype in [np.uint8, np.int8]:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            if input_scale != 0:
                processed_image = (processed_image / input_scale) + input_zero_point
                processed_image = processed_image.astype(input_dtype)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Dequantize output if needed
        if self.output_details[0]['dtype'] in [np.uint8, np.int8]:
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        return output

    def predict_batch(self, images, debug=False):
        """Run inference on a batch of preprocessed images."""
        results = []
        for img in images:
            output = self.predict(img, debug=debug)
            results.append(output)
        return np.concatenate(results, axis=0)


def get_model_metadata(model_path):
    """
    Extract metadata from a TFLite model without loading the full predictor.

    Returns dict with: input_shape, input_dtype, output_shape, output_dtype,
                      quantization, num_params (estimated), is_quantized
    """
    try:
        interpreter = create_tflite_interpreter(model_path)
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        input_shape = input_details['shape']
        output_shape = output_details['shape']
        input_dtype = input_details['dtype']
        output_dtype = output_details['dtype']

        # Estimate parameter count from model file size
        file_size_bytes = os.path.getsize(model_path)
        # Rough estimate: most params are 1-byte (int8) or 4-byte (float32)
        is_quantized = output_dtype in [np.uint8, np.int8]
        bytes_per_param = 1 if is_quantized else 4
        estimated_params = file_size_bytes // bytes_per_param

        return {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'input_dtype': input_dtype,
            'output_dtype': output_dtype,
            'quantization': input_details.get('quantization'),
            'estimated_params': estimated_params,
            'is_quantized': is_quantized,
            'file_size_bytes': file_size_bytes,
        }
    except Exception as e:
        logger.warning(f"Could not read metadata from {model_path}: {e}")
        return None


def is_valid_tflite_model(model_path):
    """Check if a TFLite model file is valid and can be loaded."""
    try:
        interpreter = create_tflite_interpreter(model_path)
        interpreter.allocate_tensors()
        return True
    except Exception:
        return False
