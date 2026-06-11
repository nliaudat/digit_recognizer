"""
benchmark/__init__.py
=====================
Benchmark package for evaluating TFLite models on digit recognition datasets.

Provides:
  - TFLiteDigitPredictor — pure TFLite inference (with ESP32 simulation)
  - Test data loading and model discovery (with _full_integer_quant priority)
  - Result analysis, confusion matrices, plots
  - CSV generation and summary tables
  - IoT model recommendation analysis
"""

from .predictor import (
    TFLiteDigitPredictor,
    get_model_metadata,
    get_model_parameters_count,
    get_model_output_type,
    is_valid_tflite_model,
)

from .data import (
    load_image_from_path,
    find_model_path,
    get_all_models,
    load_test_dataset_with_labels,
    configure_parameters_for_model,
    _select_best_tflite_files,
)

from .results import (
    test_model_on_dataset,
    test_all_models,
    generate_confusion_matrix,
    generate_comparison_graphs,
    save_results_to_csv,
    generate_markdown_report,
    calculate_best_iot_model,
    generate_iot_recommendation_section,
    save_failed_images,
    generate_failed_predictions_csv,
    test_single_model,
)

__all__ = [
    "TFLiteDigitPredictor",
    "get_model_metadata",
    "get_model_parameters_count",
    "get_model_output_type",
    "is_valid_tflite_model",
    "load_image_from_path",
    "find_model_path",
    "get_all_models",
    "load_test_dataset_with_labels",
    "configure_parameters_for_model",
    "_select_best_tflite_files",
    "test_model_on_dataset",
    "test_all_models",
    "generate_confusion_matrix",
    "generate_comparison_graphs",
    "save_results_to_csv",
    "generate_markdown_report",
    "calculate_best_iot_model",
    "generate_iot_recommendation_section",
    "save_failed_images",
    "generate_failed_predictions_csv",
    "test_single_model",
]