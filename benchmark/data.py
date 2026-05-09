"""
benchmark/data.py
=================
Test data loading and model discovery for benchmarking.

Extracted from bench_predict.py.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

import parameters as params
from utils.cache import load_disk_cache, save_disk_cache, dataset_cache_path
from utils.multi_source_loader import clear_cache
from utils.preprocess import preprocess_for_inference

logger = logging.getLogger(__name__)

# In-memory cache for test data
_cached_test_data = None
_cached_test_data_params = None


def load_image_from_path(image_path, input_channels):
    """Load image from specified path based on model's input requirements"""
    if input_channels == 1:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def find_model_path(model_name=None, input_dir=None):
    """Find the model path based on model name"""
    if input_dir is None:
        input_dir = params.OUTPUT_DIR

    if model_name:
        # Search for model with given name
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                if f.endswith('.tflite') and model_name in f:
                    return os.path.join(root, f)
        logger.warning(f"Model '{model_name}' not found in {input_dir}")
        return None

    # Find latest training directory
    training_dirs = [d for d in os.listdir(input_dir)
                     if os.path.isdir(os.path.join(input_dir, d))]
    if not training_dirs:
        logger.error(f"No training directories found in {input_dir}")
        return None

    latest = sorted(training_dirs)[-1]
    latest_dir = os.path.join(input_dir, latest)
    for f in os.listdir(latest_dir):
        if f.endswith('.tflite'):
            return os.path.join(latest_dir, f)
    return None


def get_all_models(quantized_only=False, subfolder=None, input_dir=None,
                   exclude_model=None, debug=False, model_list=None,
                   iot_compat=True):
    """Get all available models with metadata."""
    from .predictor import get_model_metadata

    if input_dir is None:
        input_dir = params.OUTPUT_DIR

    models = []
    seen_paths = set()

    # Collect model paths
    search_root = os.path.join(input_dir, subfolder) if subfolder else input_dir
    if not os.path.exists(search_root):
        logger.warning(f"Search path does not exist: {search_root}")
        return models

    for root, dirs, files in os.walk(search_root):
        for f in sorted(files):
            if not f.endswith('.tflite'):
                continue
            if quantized_only and '_float' in f:
                continue
            if exclude_model and exclude_model in f:
                continue

            full_path = os.path.join(root, f)
            if full_path in seen_paths:
                continue
            seen_paths.add(full_path)

            meta = get_model_metadata(full_path)
            if meta is None:
                continue

            models.append({
                'path': full_path,
                'name': f,
                'folder': os.path.basename(root),
                **meta,
            })

    # Filter by model_list if specified
    if model_list:
        models = [m for m in models if any(name in m['name'] for name in model_list)]

    return models


def configure_parameters_for_model(model_name_or_dir, override_classes=None,
                                    override_color=None):
    """Configure parameters for a specific model by setting env vars."""
    if override_classes:
        os.environ['DIGIT_NB_CLASSES'] = str(override_classes)
    if override_color:
        os.environ['DIGIT_INPUT_CHANNELS'] = str(override_color)

    # Clear cache so data reloads with new parameters
    clear_cache()
    logger.info(f"Reconfigured for {params.NB_CLASSES} classes, "
                f"{'Grayscale' if params.USE_GRAYSCALE else 'RGB'}")


def _decode_bench_image(image_path, label, fname, target_h, target_w,
                        grayscale, is_augmented=False):
    """Decode and resize a single benchmark image."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(image_path, flag)
    if img is None or img.size == 0:
        return None
    img = cv2.resize(img, (target_w, target_h))
    if grayscale and img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    elif not grayscale and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img, label, fname, is_augmented


def load_test_dataset_with_labels(num_samples=0, use_all_datasets=True):
    """
    Load test dataset with labels for benchmarking.

    Uses in-memory cache and disk cache for performance.
    """
    global _cached_test_data, _cached_test_data_params

    current_params = (params.NB_CLASSES, params.INPUT_CHANNELS,
                      params.INPUT_WIDTH, params.INPUT_HEIGHT,
                      tuple(json.dumps(s, sort_keys=True) for s in params.DATA_SOURCES))

    # Check in-memory cache
    if _cached_test_data is not None and _cached_test_data_params == current_params:
        test_data = list(_cached_test_data)
        if num_samples > 0:
            test_data = test_data[:num_samples]
            logger.info(f"📊 Using cached test dataset ({len(test_data)} test samples, limited by --test_images)")
        else:
            logger.info(f"📊 Using cached test dataset (ALL {len(_cached_test_data)} available test samples)")
        return test_data

    # Generate cache key
    sources_fingerprint = json.dumps(
        [{k: v for k, v in s.items()} for s in params.DATA_SOURCES],
        sort_keys=True,
    )
    cache_str = (f"{sources_fingerprint}|classes={params.NB_CLASSES}"
                 f"|channels={params.INPUT_CHANNELS}"
                 f"|w={params.INPUT_WIDTH}|h={params.INPUT_HEIGHT}"
                 f"|all={use_all_datasets}")
    cache_key = hashlib.md5(cache_str.encode()).hexdigest()[:12]
    cache_dir = getattr(params, "DATASET_CACHE_DIR", ".dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"bench_cache_{cache_key}.npz")

    # Try disk cache
    if os.path.exists(cache_path):
        try:
            logger.info(f"📊 Loading dataset from fast disk cache ({cache_path})...")
            data = np.load(cache_path, allow_pickle=False)
            images = data["images"]
            labels = data["labels"]
            fnames = data["fnames"]
            is_augmented = data["is_augmented"]

            test_data = list(zip(images, labels, fnames, is_augmented))
            _cached_test_data = list(test_data)
            _cached_test_data_params = current_params

            if num_samples > 0:
                test_data = test_data[:num_samples]
            return test_data
        except Exception as e:
            logger.warning(f"⚠️  Could not load disk cache: {e}. Rebuilding...")

    # Full load from sources
    from utils.multi_source_loader import MultiSourceDataLoader
    loader = MultiSourceDataLoader()
    images, labels = loader.load_all_sources()
    loader.print_detailed_stats()

    # Build test data list
    test_data = []
    for i in range(len(images)):
        fname = f"image_{i:06d}"
        test_data.append((images[i], labels[i], fname, False))

    # Save to disk cache
    try:
        is_augmented = np.array([item[3] for item in test_data])
        images_arr = np.array([item[0] for item in test_data])
        labels_arr = np.array([item[1] for item in test_data])
        fnames_arr = np.array([item[2] for item in test_data])
        np.savez_compressed(cache_path, images=images_arr, labels=labels_arr,
                            fnames=fnames_arr, is_augmented=is_augmented)
        logger.info(f"💾 Saved {len(test_data)} images to fast disk cache: {cache_path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not save disk cache: {e}")

    # Cache in memory
    _cached_test_data = list(test_data)
    _cached_test_data_params = current_params

    if num_samples > 0:
        test_data = test_data[:num_samples]
    return test_data
