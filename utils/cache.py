"""
utils/cache.py
==============
Shared dataset caching utilities used by multi_source_loader.py and
bench_predict.py.

Provides:
  - Disk-based NPZ cache (cross-run, survives Docker restarts)
  - In-memory cache (same-process, e.g. train_all.py loops)
  - Cache key generation from source configuration fingerprints
  - Cache clearing
"""

import hashlib
import json
import os
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# In-memory cache (single-process)
# ---------------------------------------------------------------------------
_loaded_data: Optional[Tuple[np.ndarray, np.ndarray]] = None


def get_cached_in_memory() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the in-memory cached (images, labels) or None."""
    global _loaded_data
    return _loaded_data


def set_cached_in_memory(images: np.ndarray, labels: np.ndarray) -> None:
    """Store (images, labels) in the in-memory cache."""
    global _loaded_data
    _loaded_data = (images, labels)


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


def _fingerprint_sources(source_configs: list) -> str:
    """Deterministic JSON fingerprint of a list of source dicts."""
    return json.dumps(source_configs, sort_keys=True, default=str)


def dataset_cache_key(source_configs: list, nb_classes: int,
                      input_channels: int, input_width: int,
                      input_height: int, use_all_datasets: bool = True) -> str:
    """Return a deterministic MD5 cache key for the given configuration."""
    raw = (
        f"sources={_fingerprint_sources(source_configs)}"
        f"|classes={nb_classes}"
        f"|channels={input_channels}"
        f"|w={input_width}|h={input_height}"
        f"|all={use_all_datasets}"
    )
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Disk cache path helpers
# ---------------------------------------------------------------------------


def _get_cache_dir() -> str:
    """Return the dataset cache directory, creating it if needed."""
    from config import DATASET_CACHE_DIR  # late import to avoid circular deps
    os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
    return DATASET_CACHE_DIR


def dataset_cache_path(source_configs: list, nb_classes: int,
                       input_channels: int, input_width: int,
                       input_height: int, use_all_datasets: bool = True,
                       prefix: str = "dataset") -> str:
    """Return the .npz file path for the given configuration."""
    key = dataset_cache_key(source_configs, nb_classes, input_channels,
                            input_width, input_height, use_all_datasets)
    return os.path.join(_get_cache_dir(), f"{prefix}_{key}.npz")


# ---------------------------------------------------------------------------
# Load / Save helpers
# ---------------------------------------------------------------------------


def load_disk_cache(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (images, labels) from a compressed NPZ cache, or None."""
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        images, labels = data["images"], data["labels"]
        size_mb = os.path.getsize(cache_path) / 1e6
        print(f"⚡ Loaded dataset from disk cache: {cache_path} "
              f"({len(images)} images, {size_mb:.1f} MB)")
        return images, labels
    except Exception as e:
        print(f"⚠️  Disk cache exists but could not be loaded ({e}), rebuilding…")
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None


def save_disk_cache(cache_path: str, images: np.ndarray,
                    labels: np.ndarray) -> None:
    """Persist images + labels to a compressed NPZ file."""
    try:
        np.savez_compressed(cache_path, images=images, labels=labels)
        size_mb = os.path.getsize(cache_path) / 1e6
        print(f"💾 Dataset cached to disk: {cache_path} "
              f"({size_mb:.1f} MB)")
    except Exception as e:
        print(f"⚠️  Could not save disk cache: {e}")


# ---------------------------------------------------------------------------
# Convenience: load-or-build with full config fingerprint
# ---------------------------------------------------------------------------


def load_or_build_cache(
    source_configs: list,
    nb_classes: int,
    input_channels: int,
    input_width: int,
    input_height: int,
    build_fn,
    use_all_datasets: bool = True,
    prefix: str = "dataset",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Try in-memory cache first, then disk cache, then call *build_fn*.

    *build_fn* is a zero-argument callable that returns (images, labels).
    """
    # 1. In-memory
    cached = get_cached_in_memory()
    if cached is not None:
        print("📊 Using cached dataset (in-memory)...")
        return cached

    # 2. Disk cache
    path = dataset_cache_path(source_configs, nb_classes, input_channels,
                              input_width, input_height, use_all_datasets,
                              prefix=prefix)
    cached = load_disk_cache(path)
    if cached is not None:
        images, labels = cached
        set_cached_in_memory(images, labels)
        return images, labels

    # 3. Build from scratch
    images, labels = build_fn()
    save_disk_cache(path, images, labels)
    set_cached_in_memory(images, labels)
    return images, labels


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


def clear_cache() -> None:
    """Clear the in-memory dataset cache."""
    global _loaded_data
    _loaded_data = None
    print("🧹 Cleared dataset cache")
