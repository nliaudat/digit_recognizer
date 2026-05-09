"""
benchmark/__init__.py
=====================
Benchmark package for evaluating TFLite models on digit recognition datasets.

Provides:
  - TFLiteDigitPredictor — pure TFLite inference
  - Test data loading and discovery
  - Result analysis, confusion matrices, plots
  - CSV generation and summary tables
"""

from .predictor import TFLiteDigitPredictor
from .data import (
    load_test_dataset_with_labels,
    find_model_path,
    get_all_models,
    configure_parameters_for_model,
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
    "load_test_dataset_with_labels",
    "find_model_path",
    "get_all_models",
    "configure_parameters_for_model",
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
