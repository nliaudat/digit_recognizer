"""
Utility functions for model distillation.
Includes helper functions for loading models, exporting, and evaluation.
"""

import tensorflow as tf
import numpy as np
import os
import json
import logging
from typing import Optional, Tuple, Dict, Any, Union, List, Callable
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def freeze_teacher_model(teacher: tf.keras.Model) -> tf.keras.Model:
    """
    Freeze teacher model for distillation.
    
    Args:
        teacher: Teacher model to freeze
    
    Returns:
        Frozen teacher model
    """
    teacher.trainable = False
    for layer in teacher.layers:
        layer.trainable = False
    
    logger.info(f"Teacher model frozen: {teacher.name}")
    return teacher


def get_model_size_kb(model: tf.keras.Model) -> float:
    """
    Get model size in KB after potential quantization.
    
    Args:
        model: Keras model
    
    Returns:
        Estimated model size in kilobytes
    """
    # Count total parameters
    total_params = model.count_params()
    
    # FP32 size
    fp32_size_bytes = total_params * 4
    
    # INT8 quantization reduces size by ~4x
    int8_size_bytes = fp32_size_bytes / 4
    
    return int8_size_bytes / 1024


def export_student_for_edge(
    student: tf.keras.Model,
    export_path: str,
    quantize: bool = True,
    representative_dataset: Optional[tf.data.Dataset] = None,
    target_hardware: str = "esp32"
) -> str:
    """
    Export student model for edge deployment.
    
    Args:
        student: Trained student model
        export_path: Path to save the model
        quantize: Apply post-training quantization
        representative_dataset: Dataset for quantization calibration
        target_hardware: Target hardware ('esp32', 'raspberry_pi', 'generic')
    
    Returns:
        Path to exported model
    """
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(student)
    
    if quantize:
        logger.info(f"Applying quantization for {target_hardware}")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure quantization based on target
        if target_hardware == "esp32":
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # Provide representative dataset if available
        if representative_dataset is not None:
            if isinstance(representative_dataset, np.ndarray):
                def rep_data_gen():
                    # Use a subset of provided data
                    num_samples = min(100, len(representative_dataset))
                    for i in range(num_samples):
                        # Ensure batch dimension of 1
                        sample = representative_dataset[i:i+1].astype(np.float32)
                        yield [sample]
                converter.representative_dataset = rep_data_gen
            else:
                converter.representative_dataset = lambda: representative_dataset
        else:
            # Create dummy representative dataset with correct shape
            # Get input shape from model (excluding batch dim)
            input_shape = student.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            
            # Ensure shape is a tuple of (height, width, channels)
            # student.input_shape can be (None, 32, 20, 3)
            shape_with_batch = list(input_shape)
            shape_with_batch[0] = 1 # Force batch size 1
            
            def dummy_dataset():
                for _ in range(100):
                    dummy_input = np.random.randn(*shape_with_batch).astype(np.float32)
                    yield [dummy_input]
            converter.representative_dataset = dummy_dataset
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    tflite_path = f"{export_path}.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Also save as H5 for compatibility
    keras_path = os.path.join(os.path.dirname(export_path), "best_model.keras")
    student.save(keras_path)
    
    size_kb = os.path.getsize(tflite_path) / 1024
    logger.info(f"Exported student to {tflite_path} ({size_kb:.1f} KB)")
    
    return tflite_path


def evaluate_distilled_model(
    model: tf.keras.Model,
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Evaluate distilled model on test data.
    
    Args:
        model: Model to evaluate
        test_data: Tuple of (x_test, y_test)
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    x_test, y_test = test_data
    
    # Get predictions
    logits = model.predict(x_test, batch_size=batch_size, verbose=0)
    predictions = np.argmax(logits, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(predictions == y_test)
    
    # Per-class accuracy
    num_classes = logits.shape[1]
    per_class_accuracy = {}
    for c in range(num_classes):
        mask = y_test == c
        if np.sum(mask) > 0:
            per_class_accuracy[f"class_{c}"] = np.mean(predictions[mask] == y_test[mask])
    
    return {
        'accuracy': float(accuracy),
        'per_class_accuracy': per_class_accuracy
    }


def compare_teacher_student(
    teacher: tf.keras.Model,
    student: tf.keras.Model,
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Compare teacher and student performance.
    
    Args:
        teacher: Teacher model
        student: Student model
        test_data: Tuple of (x_test, y_test)
        batch_size: Batch size for evaluation
    
    Returns:
        Comparison dictionary
    """
    teacher_metrics = evaluate_distilled_model(teacher, test_data, batch_size)
    student_metrics = evaluate_distilled_model(student, test_data, batch_size)
    
    teacher_size = get_model_size_kb(teacher)
    student_size = get_model_size_kb(student)
    
    accuracy_drop = teacher_metrics['accuracy'] - student_metrics['accuracy']
    size_ratio = teacher_size / student_size if student_size > 0 else 0
    
    return {
        'teacher': {
            'accuracy': teacher_metrics['accuracy'],
            'size_kb': teacher_size,
            'per_class_accuracy': teacher_metrics['per_class_accuracy']
        },
        'student': {
            'accuracy': student_metrics['accuracy'],
            'size_kb': student_size,
            'per_class_accuracy': student_metrics['per_class_accuracy']
        },
        'comparison': {
            'accuracy_drop': accuracy_drop,
            'size_ratio': size_ratio,
            'compression_ratio': size_ratio,
            'accuracy_retention': student_metrics['accuracy'] / teacher_metrics['accuracy'] if teacher_metrics['accuracy'] > 0 else 0
        }
    }


def save_distillation_results(
    results: Dict[str, Any],
    save_path: str,
    model_name: str,
    student_variant: str,
    num_classes: int
) -> None:
    """
    Save distillation results to JSON file.
    
    Args:
        results: Results dictionary
        save_path: Path to save results
        model_name: Name of the model
        student_variant: Student variant name
        num_classes: Number of classes
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    output = {
        'model_name': model_name,
        'student_variant': student_variant,
        'num_classes': num_classes,
        'timestamp': str(tf.timestamp().numpy()),
        **results
    }
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {save_path}")


def load_distilled_model(
    model_path: str,
    is_tflite: bool = False
) -> Union[tf.keras.Model, tf.lite.Interpreter]:
    """
    Load distilled model for inference.
    
    Args:
        model_path: Path to model file
        is_tflite: Whether model is in TFLite format
    
    Returns:
        Loaded model or interpreter
    """
    if is_tflite:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        return tf.keras.models.load_model(model_path)


def create_temperature_schedule(
    schedule_type: str = "linear",
    initial_temp: float = 8.0,
    final_temp: float = 2.0,
    total_epochs: int = 50
) -> Callable:
    """
    Create temperature schedule function.
    
    Args:
        schedule_type: Type of schedule ('linear', 'exponential', 'cosine')
        initial_temp: Initial temperature
        final_temp: Final temperature
        total_epochs: Total training epochs
    
    Returns:
        Temperature schedule function
    """
    if schedule_type == "linear":
        def linear_schedule(epoch):
            progress = min(1.0, epoch / total_epochs)
            return initial_temp * (1 - progress) + final_temp * progress
        return linear_schedule
    
    elif schedule_type == "exponential":
        def exponential_schedule(epoch):
            progress = min(1.0, epoch / total_epochs)
            return initial_temp * np.exp(-progress * np.log(initial_temp / final_temp))
        return exponential_schedule
    
    elif schedule_type == "cosine":
        def cosine_schedule(epoch):
            progress = min(1.0, epoch / total_epochs)
            return final_temp + (initial_temp - final_temp) * (1 + np.cos(np.pi * progress)) / 2
        return cosine_schedule
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def create_alpha_schedule(
    schedule_type: str = "linear",
    initial_alpha: float = 0.3,
    final_alpha: float = 0.8,
    total_epochs: int = 50
) -> Callable:
    """
    Create alpha schedule function.
    
    Args:
        schedule_type: Type of schedule ('linear', 'exponential', 'cosine')
        initial_alpha: Initial alpha
        final_alpha: Final alpha
        total_epochs: Total training epochs
    
    Returns:
        Alpha schedule function
    """
    if schedule_type == "linear":
        def linear_schedule(epoch):
            progress = min(1.0, epoch / total_epochs)
            return initial_alpha * (1 - progress) + final_alpha * progress
        return linear_schedule
    
    elif schedule_type == "exponential":
        def exponential_schedule(epoch):
            progress = min(1.0, epoch / total_epochs)
            return initial_alpha * np.exp(progress * np.log(final_alpha / initial_alpha))
        return exponential_schedule
    
    elif schedule_type == "cosine":
        def cosine_schedule(epoch):
            progress = min(1.0, epoch / total_epochs)
            return final_alpha - (final_alpha - initial_alpha) * (1 + np.cos(np.pi * progress)) / 2
        return cosine_schedule
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")