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
    """Load a TFLite model and run inference on digit images.

    Supports multi-head models (e.g. v41) where the output is split across
    two 10-class tensors that must be combined: digit = tens * 10 + units.

    Multi-head detection is automatic when:
      - len(output_details) == 2
      - each output_detail shape[-1] == 10
      - the model name contains 'v41' or 'v42'

    Can also be set explicitly via the `multi_head` attribute.
    Head tensors are resolved by NAME (not position), so TFLite converter
    reordering does not silently swap integer and decimal outputs.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.multi_head = None  # None = auto-detect
        self.load_model()

    def load_model(self):
        """Load TFLite model"""
        logger.info(f"Loading TFLite model: {self.model_path}")

        self.interpreter = create_tflite_interpreter(self.model_path)

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        logger.info(f"Input shape: {self.input_details[0]['shape']}")
        logger.info(f"Input type: {self.input_details[0]['dtype']}")
        logger.info(f"Output count: {len(self.output_details)}")
        for i, od in enumerate(self.output_details):
            logger.info(f"Output {i}: shape={od['shape']}, dtype={od['dtype']}")

        # Auto-detect multi-head (v41 / v42)
        stem = Path(self.model_path).stem.lower()
        is_multihead_model = 'v41' in stem or 'v42' in stem
        has_two_10way = (
            len(self.output_details) == 2
            and self.output_details[0]['shape'][-1] == 10
            and self.output_details[1]['shape'][-1] == 10
        )
        if is_multihead_model and has_two_10way:
            self.multi_head = True
            self.idx_int = self.output_details[0]['index']   # actual tensor index (overwritten by name resolution below)
            self.idx_dec = self.output_details[1]['index']   # actual tensor index
            self.q_int = None  # dequant params for integer/tens head
            self.q_dec = None  # dequant params for decimal/units head
            # Resolve head indices by NAME to survive TFLite converter reordering.
            # v41: tens_probs / units_probs ;  v42: integer_probs / decimal_probs
            head_map = {od['name']: od for od in self.output_details}
            if 'tens_probs' in head_map and 'units_probs' in head_map:
                self.idx_int = head_map['tens_probs']['index']
                self.idx_dec = head_map['units_probs']['index']
                _t = head_map['tens_probs']
                _u = head_map['units_probs']
                self.q_int = _t.get('quantization', (None, None)) if _t['dtype'] in [np.uint8, np.int8] else None
                self.q_dec = _u.get('quantization', (None, None)) if _u['dtype'] in [np.uint8, np.int8] else None
            elif 'integer_probs' in head_map and 'decimal_probs' in head_map:
                self.idx_int = head_map['integer_probs']['index']
                self.idx_dec = head_map['decimal_probs']['index']
                _t = head_map['integer_probs']
                _u = head_map['decimal_probs']
                self.q_int = _t.get('quantization', (None, None)) if _t['dtype'] in [np.uint8, np.int8] else None
                self.q_dec = _u.get('quantization', (None, None)) if _u['dtype'] in [np.uint8, np.int8] else None
            else:
                found = [od['name'] for od in self.output_details]
                raise ValueError(
                    "Multi-head TFLite model detected but output names are not "
                    "recognized. Expected 'tens_probs'+'units_probs' (v41) or "
                    f"'integer_probs'+'decimal_probs' (v42). Found: {found}"
                )
            logger.info(f"🔀 Multi-head model detected: integer@{self.idx_int} decimal@{self.idx_dec} ({stem})")
        else:
            self.multi_head = False

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
            # Model expects float32 input. The preprocessor may have returned
            # uint8 [0,255] or int8 [-128,127] depending on global config flags
            # (QUANTIZE_MODEL, ESP_DL_QUANTIZE). Convert correctly to [0,1].
            if input_data.dtype == np.int8:
                # ESP-DL preprocessed: int8 [-128,127] → float32 [0,1]
                input_data = (input_data.astype(np.float32) + 128.0) / 255.0
            elif input_data.dtype == np.uint8:
                # Standard quant preprocessed: uint8 [0,255] → float32 [0,1]
                input_data = input_data.astype(np.float32) / 255.0
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

            # ── Multi-head (v41/v42) path: read both head outputs ──
            if self.multi_head and len(self.output_details) >= 2:
                # Read by resolved index (name-based, survives converter reordering)
                tens_data = self.interpreter.get_tensor(self.idx_int)
                units_data = self.interpreter.get_tensor(self.idx_dec)

                # Dequantize both if needed
                if self.q_int is not None and self.q_int[0] is not None:
                    s, zp = self.q_int
                    tens_data = (tens_data.astype(np.float32) - zp) * s
                if self.q_dec is not None and self.q_dec[0] is not None:
                    s, zp = self.q_dec
                    units_data = (units_data.astype(np.float32) - zp) * s

                tens_vec = tens_data[0]
                units_vec = units_data[0]

                # Softmax if logits — robust check matching single-head path
                tens_is_softmax = (np.isclose(np.sum(tens_vec), 1.0, atol=0.02)
                                  and np.all(tens_vec >= -0.05) and np.all(tens_vec <= 1.05))
                if not tens_is_softmax:
                    tens_vec = np.exp(tens_vec - np.max(tens_vec)) / np.sum(np.exp(tens_vec - np.max(tens_vec)))
                units_is_softmax = (np.isclose(np.sum(units_vec), 1.0, atol=0.02)
                                    and np.all(units_vec >= -0.05) and np.all(units_vec <= 1.05))
                if not units_is_softmax:
                    units_vec = np.exp(units_vec - np.max(units_vec)) / np.sum(np.exp(units_vec - np.max(units_vec)))

                tens_pred = int(np.argmax(tens_vec))
                units_pred = int(np.argmax(units_vec))
                prediction = tens_pred * 10 + units_pred
                # Combined confidence: geometric mean of both head confidences
                confidence = float(np.sqrt(np.max(tens_vec) * np.max(units_vec)))
                output_vector = np.zeros(100, dtype=np.float32)
                output_vector[prediction] = 1.0  # one-hot for the combined class

                return prediction, confidence, output_vector

            # ── Standard single-head path ──
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
        """Simulate ESP32 TFLite Micro inference with real uint8/int8 I/O.

        Unlike predict(), which feeds preprocessed float32 [0,1] and lets the
        Python TFLite interpreter handle quantization transparently, this
        method mimics what ESP32 firmware actually does:

          1. Converts input to the EXACT integer dtype the model expects
             (uint8 or int8) — just like the raw camera bytes + conversion on
             the ESP32.
          2. Optionally injects ±1 noise to simulate camera sensor noise
             before the fixed-point conversion.
          3. Feeds the raw integer tensor — no float32 round-trip.
          4. Reads the raw integer output tensor and dequantizes it using
             the model's scale/zero_point.
          5. Applies softmax if the dequantized output is logits.

        This gives a realistic benchmark of what the ESP32 would actually
        produce, rather than the float32 performance of the same weights.
        """
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

        expected_dtype = self.input_details[0]['dtype']

        # ── Stage 1: Convert to the exact integer dtype the model expects ──
        # ESP32 camera produces uint8 [0,255]. If the model expects uint8,
        # feed it directly. If it expects int8, subtract 128.
        if expected_dtype == np.uint8:
            # Model accepts raw camera bytes
            if input_data.dtype == np.float32 and input_data.max() <= 1.0:
                input_data = np.clip(np.round(input_data * 255.0), 0, 255).astype(np.uint8)
            elif input_data.dtype == np.int8:
                # int8 [-128,127] → uint8 [0,255]  (modulo cast is wrong!)
                input_data = (input_data.astype(np.int32) + 128).astype(np.uint8)
            else:
                input_data = input_data.astype(np.uint8)
            # Inject sensor noise (±1 on uint8)
            noise = np.random.randint(-1, 2, size=input_data.shape, dtype=np.int16)
            input_data = np.clip(input_data.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        elif expected_dtype == np.int8:
            # Model expects int8 [-128, 127] (ESP-DL path)
            if input_data.dtype == np.float32 and input_data.max() <= 1.0:
                input_uint8 = np.clip(np.round(input_data * 255.0), 0, 255).astype(np.uint8)
            elif input_data.dtype == np.int8:
                # int8 [-128,127] → uint8 [0,255]  (modulo cast is wrong!)
                input_uint8 = (input_data.astype(np.int32) + 128).astype(np.uint8)
            else:
                input_uint8 = input_data.astype(np.uint8)
            # Inject sensor noise (±1 on uint8)
            noise = np.random.randint(-1, 2, size=input_uint8.shape, dtype=np.int16)
            input_uint8 = np.clip(input_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            input_data = (input_uint8.astype(np.int32) - 128).astype(np.int8)

        else:
            # Float model — just add small noise
            noise_std = 1.0 / 255.0
            input_data = input_data.astype(np.float32) + np.random.normal(0, noise_std, size=input_data.shape)

        # Verify shape
        expected_shape = self.input_details[0]['shape']
        if input_data.shape != tuple(expected_shape):
            if input_data.size == np.prod(expected_shape):
                input_data = input_data.reshape(expected_shape)
            else:
                return -1, 0.0, np.zeros(self.output_details[0]['shape'][-1], dtype=np.float32)

        try:
            # ── Stage 2: Run inference ──
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # ── Multi-head (v41/v42) ESP32 path ──
            if self.multi_head and len(self.output_details) >= 2:
                tens_data = self.interpreter.get_tensor(self.idx_int)
                units_data = self.interpreter.get_tensor(self.idx_dec)
                if self.q_int is not None and self.q_int[0] is not None:
                    s, zp = self.q_int
                    tens_data = (tens_data.astype(np.float32) - zp) * s
                if self.q_dec is not None and self.q_dec[0] is not None:
                    s, zp = self.q_dec
                    units_data = (units_data.astype(np.float32) - zp) * s
                tens_vec = tens_data[0]; units_vec = units_data[0]
                tens_is_sm = np.isclose(np.sum(tens_vec), 1.0, atol=0.02) and np.all(tens_vec >= -0.05) and np.all(tens_vec <= 1.05)
                if not tens_is_sm:
                    tens_vec = np.exp(tens_vec - np.max(tens_vec)) / np.sum(np.exp(tens_vec - np.max(tens_vec)))
                units_is_sm = np.isclose(np.sum(units_vec), 1.0, atol=0.02) and np.all(units_vec >= -0.05) and np.all(units_vec <= 1.05)
                if not units_is_sm:
                    units_vec = np.exp(units_vec - np.max(units_vec)) / np.sum(np.exp(units_vec - np.max(units_vec)))
                tens_pred = int(np.argmax(tens_vec)); units_pred = int(np.argmax(units_vec))
                prediction = tens_pred * 10 + units_pred
                confidence = float(np.sqrt(np.max(tens_vec) * np.max(units_vec)))
                output_vector = np.zeros(100, dtype=np.float32)
                output_vector[prediction] = 1.0
                return prediction, confidence, output_vector

            # ── Standard single-head ESP32 path ──
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # ── Stage 3: Dequantize output ──
            if self.output_details[0]['dtype'] in [np.uint8, np.int8]:
                output_scale, output_zero_point = self.output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

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
        """Get the number of classes this model was trained to predict.

        For multi-head v41: each head is 10-class, but combined output is 100-class.
        """
        if self.multi_head:
            return 100
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