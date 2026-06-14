"""
benchmark/predictor.py
======================
TFLite inference logic extracted from bench_predict.py.

Provides:
  - TFLiteDigitPredictor — load and run inference on TFLite models
  - get_model_metadata — extract model metadata
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
        """Predict digit from image using TFLite, returns (prediction, confidence, output_vector)."""
        # Preprocess image
        processed_image = preprocess_for_inference(image)

        # Handle channel mismatch
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

            # Autodetect if output is logits or softmax
            output_vector = output_data[0]
            output_sum = np.sum(output_vector)
            is_softmax = np.isclose(output_sum, 1.0, atol=0.02) and np.all(output_vector >= -0.05) and np.all(output_vector <= 1.05)

            if not is_softmax:
                # Use a numerically stable softmax implementation
                exp_data = np.exp(output_vector - np.max(output_vector))
                output_vector = exp_data / np.sum(exp_data)

            # Get prediction and confidence
            prediction = np.argmax(output_vector)
            confidence = np.max(output_vector)

            return prediction, confidence, output_vector

        except Exception as e:
            return -1, 0.0, np.zeros(self.output_details[0]['shape'][-1], dtype=np.float32)

    def predict_esp32(self, image, debug=False):
        """Simulate ESP32 inference with quantization noise injection.
        
        ESP32 uses TFLite Micro with integer-only kernels. This simulation:
        1. Quantizes input to INT8 and back (simulating camera + integer pipeline)
        2. Runs inference through the same TFLite model
        3. Adds quantization noise proportional to output scale on the logits
        4. Applies softmax and returns prediction
        """
        # Preprocess image
        processed_image = preprocess_for_inference(image)

        # Handle channel mismatch
        expected_channels = self.input_details[0]['shape'][3]
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 1 and expected_channels == 3:
            processed_image = np.repeat(processed_image, 3, axis=2)

        # Add batch dimension if not already present
        if len(processed_image.shape) == 3:
            input_data = np.expand_dims(processed_image, axis=0)
        else:
            input_data = processed_image

        # ── Step 1: Simulate ESP32 camera input quantization ──
        expected_dtype = self.input_details[0]['dtype']

        if expected_dtype in [np.uint8, np.int8]:
            if input_data.dtype == np.float32 and input_data.max() <= 1.0:
                input_uint8 = np.clip(np.round(input_data * 255.0), 0, 255).astype(np.uint8)
                noise = np.random.randint(-1, 2, size=input_uint8.shape).astype(np.int16)
                input_noisy = np.clip(input_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                input_data = input_noisy.astype(np.float32) / 255.0
            elif input_data.dtype == np.uint8:
                noise = np.random.randint(-1, 2, size=input_data.shape).astype(np.int16)
                input_data = np.clip(input_data.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            elif input_data.dtype == np.int8:
                noise = np.random.randint(-1, 2, size=input_data.shape).astype(np.int16)
                input_data = np.clip(input_data.astype(np.int16) + noise, -128, 127).astype(np.int8)
        else:
            noise_std = 1.0 / 255.0
            input_data = input_data.astype(np.float32) + np.random.normal(0, noise_std, size=input_data.shape)
            if input_data.max() > 1.0:
                input_data = input_data / 255.0

        # ── Step 2: Prepare input for the model ──
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
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # ── Step 3: Add output quantization noise ──
            if self.output_details[0]['dtype'] in [np.uint8, np.int8]:
                output_scale, output_zero_point = self.output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
                noise_scale = output_scale * 0.5
                output_data += np.random.normal(0, noise_scale, size=output_data.shape)
            else:
                noise_scale = 1e-4
                output_data += np.random.normal(0, noise_scale, size=output_data.shape)

            # Autodetect if output is logits or softmax
            output_vector = output_data[0]
            output_sum = np.sum(output_vector)
            is_softmax = np.isclose(output_sum, 1.0, atol=0.02) and np.all(output_vector >= -0.05) and np.all(output_vector <= 1.05)

            if not is_softmax:
                exp_data = np.exp(output_vector - np.max(output_vector))
                output_vector = exp_data / np.sum(exp_data)

            prediction = np.argmax(output_vector)
            confidence = np.max(output_vector)

            return prediction, confidence, output_vector

        except Exception as e:
            return -1, 0.0, np.zeros(self.output_details[0]['shape'][-1], dtype=np.float32)

    @property
    def num_classes(self):
        """Get the number of classes this model was trained to predict"""
        return self.output_details[0]['shape'][-1]


def get_model_metadata(model_path):
    """
    Extract multiple metadata items from a TFLite model in a single pass
    to avoid redundant interpreter allocations.
    Returns: (output_type, parameters_count)
    """
    try:
        interpreter = create_tflite_interpreter(model_path)
        interpreter.allocate_tensors()

        # 1. Detect output type
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare zero input
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        input_data = np.zeros(input_shape, dtype=input_dtype)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Handle output quantization if present
        if output_details[0]['dtype'] in [np.uint8, np.int8]:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        output_vector = output_data[0]
        output_sum = np.sum(output_vector)

        # Heuristic: Softmax sums to 1.0 and elements are within [0, 1]
        is_softmax = np.isclose(output_sum, 1.0, atol=0.02) and np.all(output_vector >= -0.05) and np.all(output_vector <= 1.05)

        # 2. Count parameters
        total_params = 0
        tensor_details = interpreter.get_tensor_details()
        for tensor in tensor_details:
            has_buffer = 'buffer' in tensor and tensor['buffer'] > 0
            if not has_buffer:
                try:
                    t_data = interpreter.get_tensor(tensor['index'])
                    if t_data is not None and np.any(t_data != 0):
                        has_buffer = True
                except:
                    pass
            if has_buffer:
                shape = tensor['shape']
                if shape is not None and len(shape) > 0:
                    total_params += int(np.prod(shape))

        # Determine the suffix (quant vs float)
        is_quant = input_dtype in [np.int8, np.uint8]
        if not is_quant:
            path_lower = model_path.lower()
            if 'qat' in path_lower or 'quant' in path_lower:
                is_quant = True

        q_suffix = " (quant)" if is_quant else " (float)"
        output_type = ("softmax" if is_softmax else "logits") + q_suffix

        return output_type, total_params
    except Exception as e:
        return "unknown", 0


def get_model_parameters_count(model_path):
    """Legacy helper - use get_model_metadata for efficiency"""
    _, count = get_model_metadata(model_path)
    return count


def get_model_output_type(model_path):
    """Legacy helper - use get_model_metadata for efficiency"""
    mtype, _ = get_model_metadata(model_path)
    return mtype


def is_valid_tflite_model(model_path):
    """Check if a TFLite model file is valid and can be loaded."""
    try:
        interpreter = create_tflite_interpreter(model_path)
        interpreter.allocate_tensors()
        return True
    except Exception as e:
        if "Flex" in str(e) or "Select TensorFlow op(s)" in str(e):
            logger.warning(f"Skipping GPU-only or Flex-dependent model {os.path.basename(model_path)}")
        else:
            logger.error(f"Invalid TFLite model {os.path.basename(model_path)}: {str(e).split(chr(10))[0][:150]}...")
        return False