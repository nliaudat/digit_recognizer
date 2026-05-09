"""
config/data_sources.py — Data source definitions.

DATA_SOURCES is dynamically built based on NB_CLASSES so that label files
and dataset paths use the correct class count.
"""

import sys

# Get NB_CLASSES from the parent config module (avoids circular import)
_config_mod = sys.modules.get('config')
if _config_mod is not None and hasattr(_config_mod, 'NB_CLASSES'):
    NB_CLASSES = _config_mod.NB_CLASSES
else:
    # Fallback: read from environment
    import os
    NB_CLASSES = int(os.environ.get("DIGIT_NB_CLASSES", "100"))
del _config_mod

# ---------------------------------------------------------------------------
# Data source weights — effective image counts (audited 2026-05-09)
#
# Dataset                        Raw     Weight   Effective   Category
# ------------------------------ ------- -------- ----------- --------------------
# Tenth-of-step-of-a-meter-digit  22 653   x 2.5    56 633     Real / baseline
# real_integra_bad_predictions     1 867   x12.0    22 404     Real / corrected
# real_integra                     1 873   x 8.0    14 984     Real
# failed_predictions_{N}           4 371   x12.0    52 452     Real / corrected
# static_augmentation             70 632   x 0.6    42 379     Synthetic
# static_augmentation_mixup       11 301   x 0.4     4 520     Synthetic
# GWF_watermeter                     832   x 8.0     6 656     Real
# ------------------------------ ------- -------- ----------- --------------------
# TOTAL                                           200 028
#
# Target balance:
#   Real corrected (bad_pred + failed + GWF) → ~82k  (41 %)
#   Real baseline  (Tenth + real_integra)    → ~72k  (36 %)
#   Synthetic                                → ~47k  (23 %)
# ---------------------------------------------------------------------------
DATA_SOURCES = [
    {
        'name': 'Tenth-of-step-of-a-meter-digit',
        'type': 'label_file', 
        'labels': f'labels_{NB_CLASSES}_shuffle.txt',
        'path': 'datasets/Tenth-of-step-of-a-meter-digit', 
        'weight': 2.5,  # 22 653 raw → ~56 633 effective (Real baseline)
    },
    {
        'name': 'real_integra_bad_predictions',
        'type': 'label_file', 
        'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
        'path': 'datasets/real_integra_bad_predictions', 
        'weight': 12.0,  # 1 867 raw → ~22 404 effective (Corrected hard cases)
    },
    {
        'name': 'real_integra',
        'type': 'label_file', 
        'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
        'path': 'datasets/real_integra', 
        'weight': 8.0,  # 1 873 raw → ~14 984 effective (Real data)
    },
    {
        'name': f'failed_predictions_{NB_CLASSES}',
        'type': 'label_file', 
        'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
        'path': f'datasets/failed_predictions_{NB_CLASSES}', 
        'weight': 12.0,  # 4 371 raw → ~52 452 effective (Corrected hard cases)
    },
    {
        'name': 'static_augmentation',
        'type': 'label_file', 
        'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
        'path': 'datasets/static_augmentation', 
        'weight': 0.6,  # 70 632 raw → ~42 379 effective (Synthetic — down-weighted to avoid dominance)
        'is_synthetic': True,
    },
    {
        'name': 'static_augmentation_mixup',
        'type': 'label_file', 
        'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
        'path': 'datasets/static_augmentation_mixup', 
        'weight': 0.4,  # 11 301 raw → ~4 520 effective (Synthetic mixup)
        'is_synthetic': True,
    },
    {
        'name': f'GWF_watermeter',
        'type': 'label_file', 
        'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
        'path': f'datasets/GWF_watermeter', 
        'weight': 8.0,  # 832 raw → ~6 656 effective (Small real-world domain)
    },
]
