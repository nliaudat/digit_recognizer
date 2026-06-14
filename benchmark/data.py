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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from tqdm import tqdm

import config as params
from utils.cache import load_disk_cache, save_disk_cache, dataset_cache_path
from utils.multi_source_loader import clear_cache
from utils.preprocess import preprocess_for_inference
from .predictor import get_model_metadata, is_valid_tflite_model

logger = logging.getLogger(__name__)

# In-memory cache for test data
_cached_test_data = None
_cached_test_data_params = None


# ── TFLite Model Selection Helper ──────────────────────────────────────────

def _select_best_tflite_files(dir_path):
    """Get TFLite model files from a directory, preferring uint8 I/O quantized models.

    Priority:
    1. *_integer_quant_uint8.tflite  (ESP32: uint8 I/O, matches raw camera bytes)
    2. *_integer_quant_float32.tflite (float32 I/O, PC benchmarking)
    3. *_full_integer_quant.tflite   (legacy ambiguous name, kept for backward compat)
    4. Any other *.tflite as fallback
    """
    all_tflite = [f for f in os.listdir(dir_path) if f.endswith('.tflite') and not f.endswith('_float.tflite')]
    # 1. uint8 I/O variant (preferred — matches ESP32 camera bytes directly)
    uint8_quant = [f for f in all_tflite if f.endswith('_integer_quant_uint8.tflite')]
    if uint8_quant:
        return uint8_quant
    # 2. float32 I/O variant (PC benchmark quality)
    float32_quant = [f for f in all_tflite if f.endswith('_integer_quant_float32.tflite')]
    if float32_quant:
        return float32_quant
    # 3. Legacy ambiguous name (backward compat)
    full_integer_quant = [f for f in all_tflite if f.endswith('_full_integer_quant.tflite')]
    if full_integer_quant:
        return full_integer_quant
    return all_tflite


# ── Model Path Finding ────────────────────────────────────────────────────

def load_image_from_path(image_path, input_channels):
    """Load image from specified path based on model's input requirements"""
    if not os.path.exists(image_path):
        return None

    if input_channels == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def find_model_path(model_name=None, input_dir=None):
    """Find the model path based on model name"""
    if input_dir is None:
        input_dir = params.OUTPUT_DIR

    all_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    # Filter out non-training directories
    training_dirs = []
    for dir_name in all_dirs:
        dir_path = os.path.join(input_dir, dir_name)
        tflite_files = _select_best_tflite_files(dir_path)
        if tflite_files and not dir_name.startswith('test_results'):
            training_dirs.append(dir_name)

    if not training_dirs:
        logger.error(f"No training directories with TFLite models found in {input_dir}.")
        return None

    if model_name:
        model_name_clean = model_name.replace('.tflite', '')

        # First, check if model_name matches a training directory
        matching_dirs = [d for d in training_dirs if model_name_clean in d]
        if matching_dirs:
            best_match = None
            for dir_name in matching_dirs:
                if dir_name == model_name_clean:
                    best_match = dir_name
                    break
            if not best_match and matching_dirs:
                best_match = matching_dirs[0]

            training_path = os.path.join(input_dir, best_match)
            tflite_files = _select_best_tflite_files(training_path)
            if tflite_files:
                return os.path.join(training_path, tflite_files[0])

        # If no directory match, search for specific model files
        for training_dir in sorted(training_dirs, reverse=True):
            training_path = os.path.join(input_dir, training_dir)
            tflite_files = _select_best_tflite_files(training_path)
            for model_file in tflite_files:
                model_file_clean = model_file.replace('.tflite', '')
                if (model_name_clean == model_file_clean or
                    model_name_clean in model_file_clean or
                    model_name == model_file):
                    return os.path.join(training_path, model_file)

        logger.warning(f"Model or directory '{model_name}' not found in any training directory.")
        return None

    else:
        # Default: latest training directory
        latest_training = sorted(training_dirs)[-1]
        latest_dir_path = os.path.join(input_dir, latest_training)
        tflite_files = _select_best_tflite_files(latest_dir_path)
        if tflite_files:
            return os.path.join(latest_dir_path, tflite_files[0])

        logger.error(f"No TFLite model found in: {latest_dir_path}")
        return None


# ── Model Discovery ───────────────────────────────────────────────────────

def get_all_models(quantized_only=False, subfolder=None, input_dir=None,
                   exclude_model=None, debug=False, model_list=None,
                   iot_compat=True):
    """Get all available models with parameters count - with error handling.
    
    Args:
        quantized_only: If True, only return quantized models
        subfolder: If specified, only look in this specific subfolder
        exclude_model: List of model names or strings to exclude from testing
        debug: Enable debug output
        model_list: Optional list of specific models to include (names or directories)
        iot_compat: If True, exclude models with 'float32' or 'dynamic_range' in name
    """
    if input_dir is None:
        input_dir = params.OUTPUT_DIR

    all_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    # Filter out non-training directories
    training_dirs = []
    for dir_name in all_dirs:
        dir_path = os.path.join(input_dir, dir_name)
        tflite_files = _select_best_tflite_files(dir_path)
        if tflite_files and not dir_name.startswith('test_results'):
            training_dirs.append(dir_name)

    # Filter by subfolder if specified
    if subfolder:
        if subfolder in training_dirs:
            training_dirs = [subfolder]
        else:
            matching_dirs = [d for d in training_dirs if subfolder in d]
            if matching_dirs:
                training_dirs = matching_dirs
            else:
                logger.warning(f"Subfolder '{subfolder}' not found in training directories")
                return []

    all_models = []

    for training_dir in training_dirs:
        training_path = os.path.join(input_dir, training_dir)
        tflite_files = _select_best_tflite_files(training_path)

        for model_file in tflite_files:
            # Skip excluded models
            if exclude_model:
                exclude_list = [exclude_model] if isinstance(exclude_model, str) else exclude_model
                if any(excl in model_file or excl in training_dir for excl in exclude_list):
                    if debug:
                        logger.info(f"Skipping excluded model: {training_dir}/{model_file}")
                    continue

            # Filter by model_list
            if model_list:
                full_model_path = f"{training_dir}/{model_file}"
                match_found = any(
                    item == model_file or item == training_dir or item in full_model_path
                    for item in model_list
                )
                if not match_found:
                    continue

            model_path = os.path.join(training_path, model_file)

            # Skip if file doesn't exist or is empty
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                continue

            # Verify the model can be loaded
            if not is_valid_tflite_model(model_path):
                continue

            try:
                model_size = os.path.getsize(model_path) / 1024
                model_type, parameters_count = get_model_metadata(model_path)

                # Filter by IoT compatibility if requested
                if iot_compat:
                    name_lower = (model_file + " " + training_dir).lower()
                    if "float32" in name_lower or "dynamic_range" in name_lower:
                        if debug:
                            logger.info(f"Skipping non-IoT model: {training_dir}/{model_file}")
                        continue

                # Filter by quantization if requested
                if quantized_only and "(quant)" not in model_type:
                    continue

                all_models.append({
                    'path': model_path,
                    'name': model_file,
                    'directory': training_dir,
                    'size_kb': model_size,
                    'type': model_type,
                    'parameters': parameters_count,
                })

            except Exception as e:
                logger.warning(f"Error processing model {training_dir}/{model_file}: {e}")
                continue

    # Remove duplicates
    unique_models = {}
    for model in all_models:
        key = f"{model['directory']}/{model['name']}"
        if key not in unique_models:
            unique_models[key] = model

    return list(unique_models.values())


# ── Dataset Loading ───────────────────────────────────────────────────────

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
    return (img, label, fname, is_augmented)


def load_test_dataset_with_labels(num_samples=0, use_all_datasets=True):
    """
    Load test dataset with proper labels, tracking original filenames.
    Returns: list of (image_array, true_label, filename_no_ext, is_augmented) tuples
    """
    global _cached_test_data, _cached_test_data_params

    current_params = (use_all_datasets, params.NB_CLASSES, params.INPUT_CHANNELS,
                      params.INPUT_WIDTH, params.INPUT_HEIGHT)

    # Check in-memory cache
    if _cached_test_data is not None and _cached_test_data_params == current_params:
        test_data = list(_cached_test_data)
        if num_samples > 0 and len(test_data) > num_samples:
            np.random.shuffle(test_data)
            test_data = test_data[:num_samples]
            logger.info(f"📊 Using cached test dataset ({len(test_data)} test samples, limited)")
        else:
            logger.info(f"📊 Using cached test dataset (ALL {len(test_data)} available test samples)")
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
            data = np.load(cache_path, allow_pickle=True)
            images, labels, fnames, is_augmented = data["images"], data["labels"], data["fnames"], data["is_augmented"]
            test_data = list(zip(images, labels, fnames, is_augmented))
            _cached_test_data = list(test_data)
            _cached_test_data_params = current_params
            if num_samples > 0 and len(test_data) > num_samples:
                np.random.shuffle(test_data)
                test_data = test_data[:num_samples]
                logger.info(f"  Using {len(test_data)} test samples (limited)")
            else:
                logger.info(f"  Using ALL {len(test_data)} available test samples")
            return test_data
        except Exception as e:
            logger.warning(f"Could not load disk cache: {e}. Rebuilding...")

    logger.info("📊 Loading test dataset with labels from disk...")

    tasks = []
    h = params.INPUT_HEIGHT
    w = params.INPUT_WIDTH
    grayscale = params.USE_GRAYSCALE

    for source_config in params.DATA_SOURCES:
        is_augmented_flag = source_config.get('is_synthetic', False)
        source_type = source_config.get('type', '')
        source_path = source_config.get('path', '')

        if source_type == 'label_file':
            labels_file = source_config.get('labels', 'labels.txt')
            label_file_path = os.path.join(source_path, labels_file)
            images_dir = os.path.join(source_path, 'images')
            if not os.path.exists(label_file_path) or not os.path.exists(images_dir):
                logger.debug(f"Source {source_path} missing labels or images dir.")
                continue
            with open(label_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for raw in lines:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) < 2:
                    continue
                fname, label_str = parts[0], parts[1]
                try:
                    label = int(label_str)
                except ValueError:
                    continue
                if label < 0 or label >= params.NB_CLASSES:
                    continue
                img_path = os.path.join(images_dir, fname)
                if not os.path.exists(img_path):
                    continue
                tasks.append((img_path, label, os.path.splitext(fname)[0], is_augmented_flag))

        elif source_type == 'folder_structure':
            for class_label in range(params.NB_CLASSES):
                class_dir = os.path.join(source_path, str(class_label))
                if not os.path.exists(class_dir):
                    continue
                for fn in os.listdir(class_dir):
                    if not any(fn.lower().endswith(e) for e in ('.jpg', '.jpeg', '.png', '.bmp')):
                        continue
                    img_path = os.path.join(class_dir, fn)
                    tasks.append((img_path, class_label, os.path.splitext(fn)[0], is_augmented_flag))

    test_data = []
    if tasks:
        n_workers = min(8, os.cpu_count() or 4)
        logger.info(f"  📂 Decoding {len(tasks)} images in parallel...")

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_decode_bench_image, path, lbl, fn, h, w, grayscale, aug)
                       for path, lbl, fn, aug in tasks]
            done = 0
            for f in as_completed(futures):
                res = f.result()
                if res is not None:
                    test_data.append(res)
                done += 1
                if done % max(1, len(tasks) // 10) == 0:
                    logger.info(f"    {done}/{len(tasks)} loaded...")

    if not test_data:
        logger.error("No data loaded from any source")
        return []

    np.random.shuffle(test_data)

    # Save to disk cache
    try:
        images_arr = np.array([item[0] for item in test_data])
        labels_arr = np.array([item[1] for item in test_data])
        fnames_arr = np.array([item[2] for item in test_data])
        is_augmented_arr = np.array([item[3] for item in test_data])
        np.savez_compressed(cache_path, images=images_arr, labels=labels_arr,
                            fnames=fnames_arr, is_augmented=is_augmented_arr)
        logger.info(f"💾 Saved {len(test_data)} images to fast disk cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Could not save disk cache: {e}")

    _cached_test_data = list(test_data)
    _cached_test_data_params = current_params

    if num_samples > 0 and len(test_data) > num_samples:
        test_data = test_data[:num_samples]
        logger.info(f"  Using {len(test_data)} test samples (limited)")
    else:
        logger.info(f"  Using ALL {len(test_data)} available test samples")

    return test_data


# ── Parameter Configuration ───────────────────────────────────────────────

def configure_parameters_for_model(model_name_or_dir, override_classes=None,
                                    override_color=None):
    """
    Adjust globals in parameters.py for the specific model.
    Manual overrides from CLI take absolute precedence.
    """
    name_upper = str(model_name_or_dir).upper()
    changed = False

    # 1. Handle NB_CLASSES
    if override_classes is not None:
        if params.NB_CLASSES != override_classes:
            params.NB_CLASSES = override_classes
            changed = True
    elif '100CLS' in name_upper and params.NB_CLASSES != 100:
        params.NB_CLASSES = 100
        changed = True
    elif '10CLS' in name_upper and params.NB_CLASSES != 10:
        params.NB_CLASSES = 10
        changed = True

    # 2. Handle INPUT_CHANNELS / COLOR
    new_channels = params.INPUT_CHANNELS
    if override_color is not None:
        new_channels = 1 if override_color == 'gray' else 3
    elif 'GRAY' in name_upper:
        new_channels = 1
    elif 'RGB' in name_upper:
        new_channels = 3

    if params.INPUT_CHANNELS != new_channels:
        params.INPUT_CHANNELS = new_channels
        changed = True

    if changed:
        # Refresh derived parameters
        params.update_derived_parameters()

        # Update labels file in data sources based on new NB_CLASSES
        for source in params.DATA_SOURCES:
            if 'labels' in source or source.get('type') == 'label_file':
                source['labels'] = f'labels_{params.NB_CLASSES}_shuffle.txt'

        # Clear the dataset cache
        clear_cache()
        logger.info(f"🔄 Reconfigured test environment for {params.NB_CLASSES} classes "
                    f"in {'Grayscale' if params.USE_GRAYSCALE else 'RGB'}")