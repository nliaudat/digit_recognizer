# Augmentation Strategy

**Date:** 2026-06-06  
**Context:** Documentation of the current dual-layer augmentation strategy (static + probability-gated inline).

---

## 1. Architecture Overview

The project uses a **dual augmentation strategy**:

| Layer | Mechanism | Volume | When Applied |
|---|---|---|---|
| **Static** | Pre-generated images via `static_augmentation.py` (OpenCV/NumPy) | 70,632 images (~40% of training data) | Generated offline, loaded from disk |
| **Static Mixup** | Pre-generated mixup variants via `static_mixup_augmentation.py` | 11,301 images (~6%) | Generated offline, loaded from disk |
| **Inline** | `tf.keras.Sequential` pipeline in `utils/augmentation.py` (TensorFlow ops) | Applied with **p=0.3** per-image probability (~53k of 176k images/epoch) | Every epoch |

### Data Flow

```
Multi-source loader (load_combined_dataset)
  ├── Real datasets (Tenth, real_integra, GWF, failed_predictions)
  ├── Static augmentation (pre-augmented images, 70,632)
  ├── Static mixup (pre-augmented mixup, 11,301)
  │
  └── ~30% of images → inline pipeline (rotation, zoom, shift, brightness, contrast, noise)
      ~70% of images → pass through unchanged
```

### Rationale for Probability Gate

The `_maybe_augment_one_image()` wrapper in `utils/augmentation.py` gates augmentation with a per-image `tf.random.uniform() < probability` check:

- **Static datasets** provide heavy diversity with 12 transform types at strong intensities
- **Inline** provides epoch-to-epoch re-randomization preventing memorization of fixed static variants
- At **p=0.3**, only ~53k of 176k images are processed per epoch → **70% overhead reduction**
- The model sees a mix of fresh re-randomizations + original images each epoch
- Backward compatible: defaults to 1.0 for existing behavior

---

## 2. Current Configuration

### Inline Augmentation (`config/augmentation.py`)

| Parameter | Value | Notes |
|---|---|---|
| `USE_DATA_AUGMENTATION` | `True` | Inline pipeline enabled |
| `AUGMENTATION_PROBABILITY` | `0.3` | 30% of images augmented per epoch |
| `AUGMENTATION_ROTATION_RANGE` | `3.0` | ±3° (increased from ±1.15° for parity with static's ±5°) |
| `AUGMENTATION_ZOOM_RANGE` | `0.1` | ±10% |
| `AUGMENTATION_WIDTH_SHIFT_RANGE` | `0.03` | ±3% (was 0.0 — newly enabled) |
| `AUGMENTATION_HEIGHT_SHIFT_RANGE` | `0.03` | ±3% (was 0.0 — newly enabled) |
| `AUGMENTATION_BRIGHTNESS_RANGE` | `[0.9, 1.1]` | ±10% |
| `AUGMENTATION_CONTRAST_RANGE` | `0.1` | ±10% (pre-contrast clip removed — now operates on full [0,1] range) |
| `AUGMENTATION_HORIZONTAL_FLIP` | `False` | Correctly disabled |
| `AUGMENTATION_VERTICAL_FLIP` | `False` | Correctly disabled |
| `AUGMENTATION_POLARITY_INVERSION` | `False` | Disabled — fully covered by static's 100% mirror twin |
| `USE_MIXUP` | `False` | Covered by static_mixup dataset |
| `USE_CUTMIX` | `False` | Not relevant for digit classification |
| `USE_RANDOM_ERASING` | `False` | Not relevant for digit classification |

### Pre-Contrast Clip Removal

The `pre_contrast_clip` Lambda layer that clamped input to [0.1, 0.9] before contrast was **removed** from `utils/augmentation.py`. It was weakening the contrast transform by compressing dynamic range. The existing `final_value_clamp` at the pipeline end guarantees valid [0, 1] output.

### Polarity Inversion Disabled Inline

`AUGMENTATION_POLARITY_INVERSION = False` because the static augmentation already provides 100% coverage via mirror inversion (every original and every augmented variant gets an inverted twin saved with `_inv` suffix). Inline polarity inversion would be redundant and add computational cost.

### Gaussian Noise

Kept at σ=0.001 (negligible). The static dataset already provides noise at σ=0.05, p=0.5. The inline noise serves only as a numerical stability measure to prevent dead neurons.

---

## 3. Static Augmentation (`datasets/tools/static_augmentation.py`)

### Default Configuration (from `augmentation_params_Tenth-of-step-of-a-meter-digit.json`)

| Transform | Enabled | Probability | Range |
|---|---|---|---|
| Rotation | ✅ | 0.3 | ±5° |
| Zoom | ✅ | 0.3 | ±10% |
| Shift | ✅ | 0.4 | ±5% |
| Shear | ✅ | 0.3 | ±10% |
| Brightness | ✅ | 0.3 | [0.9, 1.1] |
| Contrast | ✅ | 0.3 | [0.9, 1.1] |
| Color Jitter | ✅ | 0.9 | ±0.2 |
| Gaussian Noise | ✅ | 0.5 | σ=0.05 |
| Random Crop | ✅ | 0.2 | 90% |
| Perspective | ✅ | 0.3 | scale=0.1 |
| Flashlight | ✅ | 0.3 | intensity=0.8 |
| Polarity Inversion | ✅ | 0.5 | flip 1.0-x |

### Mirror Inversion (Critical Feature)

When `mirror_inversion=True` (always enabled), **every** original image and **every** augmented variant gets an inverted twin:

```python
# Line 904 — every augmented image gets an inverted twin
inverted_augmented = self.apply_polarity_inversion(augmented)
self.save_augmented_image(inverted_augmented, ..., is_inverted=True)

# Line 978 — every original also gets an inverted twin
futures.append(executor.submit(self._save_inverted_original, img, label, path))
```

This effectively **doubles** the static dataset output and ensures 100% polarity coverage.

### Per-Source Multipliers

| Source | Multiplier | Notes |
|---|---|---|
| Tenth-of-step-of-a-meter-digit | 1x | Standard |
| real_integra | **2x** | Gets extra augmentation passes (diverse variations) |
| real_integra_bad_predictions | 1x | Standard |
| failed_predictions | 1x | Standard |

---

## 4. Static Mixup (`datasets/tools/static_mixup_augmentation.py`)

### Default Configuration

| Parameter | Value |
|---|---|
| Alpha | 0.2 |
| Max label distance | 0.1 (only adjacent classes can mix) |
| Circular labels | True (9.9 ↔ 0.0) |
| Input source | Tenth-of-step-of-a-meter-digit only |
| Adaptive tiers: <90% acc | 15× variants |
| 90-95% acc | 5× variants |
| >95% acc | 0× variants (base) |

---

## 5. Effective Image Counts — Verified

From `config/data_sources.py` — all calculations verified:

```
# Dataset                        Raw     Weight   Effective   Verified?
# ------------------------------ ------- -------- ----------- --------------------
# Tenth-of-step-of-a-meter-digit  22 653   x 2.5    56 633     ✅ 22,653 × 2.5 = 56,632.5 → 56,633
# real_integra_bad_predictions     1 867   x 5.0     9 335     ✅ 1,867 × 5.0 = 9,335
# real_integra                     1 873   x 3.0     5 619     ✅ 1,873 × 3.0 = 5,619
# failed_predictions_{N}           4 371   x 5.0    21 855     ✅ 4,371 × 5.0 = 21,855
# static_augmentation             70 632   x 1.0    70 632     ✅
# static_augmentation_mixup       11 301   x 0.8     9 041     ✅ 11,301 × 0.8 = 9,040.8 → 9,041
# GWF_watermeter                     832   x 4.0     3 328     ✅ 832 × 4.0 = 3,328
# ------------------------------ ------- -------- ----------- --------------------
# TOTAL                                           176 443     ✅ Sum = 176,443
```

**Note on weight semantics:** Weights > 1.0 use `np.random.choice` with replacement (multi_source_loader.py lines 72-77), meaning the "effective" count is an **expected value**, not a fixed number. Each training run gets a slightly different random sample. For `failed_predictions` (4,371 raw × 5.0 = 21,855), the per-run variance is approximately ±√(4371 × 5 × 0.2) ≈ ±66 samples.

---

## 6. Known Caveats (Not Yet Fixed)

These are documented for future improvement. None are critical, but addressing them would improve augmentation quality.

### static_augmentation.py

| # | Caveat | Location | Severity |
|---|---|---|---|
| 1 | **Stochastic zero-augmentation.** At most 2 spatial + 1 color augmentation per pass, each with sub-1.0 probabilities. Some passes apply **zero** augmentations and are silently skipped (line 898). With `multiplier=1`, some originals produce no usable augmented output. | Lines 847-898 | **Medium** — silent sample loss |
| 2 | **Per-source multipliers inconsistent.** `real_integra.json` uses `multiplier=2`, others use `1`. Combined with caveat #1, effective yield < 2x. | JSON configs | **Low** |
| 3 | **Float→uint8 truncation.** `(image * 255).astype(np.uint8)` truncates (not rounds) fractional parts. | Line 941 | **Low** — imperceptible at JPEG Q=95 |
| 4 | **`random.seed()` in every function.** Uses system time entropy, reducing randomness quality in threaded execution. | All augmentation functions | **Low** |
| 5 | **Contrast in BGR space.** Per-channel mean on BGR, not perceptual luminance. | Lines 560-581 | **Very Low** |

### static_mixup_augmentation.py

| # | Caveat | Location | Severity |
|---|---|---|---|
| 6 | **Only applied to Tenth dataset.** Other sources (real_integra, GWF, failed_predictions) get **zero mixup variants**. | Line 49 | **Medium** — missed opportunity |
| 7 | **No polarity inversion on mixup outputs.** The entire `static_augmentation_mixup` dataset contains only light-on-dark digits. | No inversion call | **Medium** — inverted mixup never seen |
| 8 | **Extreme adaptive tier multipliers.** Low-accuracy classes (<90%) get **15×** variants, causing synthetic class imbalance in the mixup dataset. | Line 66 | **Medium** — synthetic imbalance |
| 9 | **Label distance 0.1 is very tight.** Only adjacent classes can mix. With β(0.2, 0.2), most mixed labels are ~90%/10% weighted, reducing regularization benefit. | Line 57 | **Low** |
| 10 | **No validation of compatibility before generation.** Rare classes with no close neighbors produce zero mixup variants. | Lines 452-454 | **Low** |

---

## 7. Files Modified During This Review

| File | Change |
|---|---|
| `config/augmentation.py` | Added `AUGMENTATION_PROBABILITY = 0.3`. Shift enabled (±3%). Rotation increased (→±3°). Polarity inversion disabled (handled by static). |
| `utils/augmentation.py` | Added `_maybe_augment_one_image()` with probability gate. Removed pre-contrast clip. Wired into `setup_augmentation_for_training()`. |
| `config/data_sources.py` | Added docstring explaining the probability-gated inline augmentation strategy. |