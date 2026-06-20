# Novel Architectural Practices

## A Catalogue of Technical Innovations in the digit_recognizer Project

*A survey of novel techniques across models v3 through v34, the 6 Architecture Decision Records, and the augmentation / preprocessing pipeline. These practices represent the project's core engineering contributions.*

---

## Part 1: Preprocessing-as-a-Layer (The "Embedded Intelligence" Pipeline)

The project's deepest contribution is a **systematic evolution from MCU-side preprocessing code to self-contained model graph preprocessing**, using only standard TFLite Micro operators. Each model version critiques the previous one and adds principled intelligence.

### 1.1 Luminance Grayscale via Frozen Conv2D

**Introduced**: v23 | **Preserved in**: v23, v24, v27, v28, v29

**The Problem**: Standard RGB→grayscale conversion requires separate C++ code on the MCU. This causes 3× pre-processing overhead in memory and time.

**The Innovation**: A `trainable=False Conv2D(1×1, filters=1)` at the network entry with fixed BT.601 luminance weights `[0.299, 0.587, 0.114]`:

```python
luminance_weights = tf.constant([[[[0.299], [0.587], [0.114]]]])
```

This is a **standard TFLite CONV_2D operator**. No pre-processing code needed. No JPEG decode → convert → normalize pipeline. The camera's RGB bytes go directly into the model's first convolution.

```
RGB input (32×20×3) → [Frozen Conv2D 1×1] → Grayscale (32×20×1) → Backbone
```

**Why it's novel**: Every TinyML paper assumes conversion happens before the model. This embeds it inside, eliminating the 3× overhead entirely.

**Evolution**: v23 stored weights as `tf.constant` in the initializer. v24+ stores them via `self.add_weight()` in `build()`, ensuring proper Keras serialization and QAT compatibility.

---

### 1.2 Adaptive Contrast Normalization — Three Approaches

**Introduced**: v24 | **Design lineage**: v24 → v27 (soft replacement) → v28 (hard binary replacement) → v29 (hybrid fusion)

The project explored three fundamentally different approaches to handling lighting variation within the model graph:

#### Method A: Hard Min-Max Normalization (v24, ESP32-compatible)
```
AdaptiveContrastNormalization:
  1. Compute per-image mean intensity
  2. If mean < 0.5: invert = 1 - x (assumes dark-on-light)
  3. Stretch contrast: (x - min) / (max - min + ε)
```
All operations: `reduce_mean`, `sub`, `div`, `add` — standard TFLite.

#### Method B: Robust Percentile Normalization (v24, PC-only)
```
RobustContrastNormalization:
  1. Compute 10th/90th percentiles via tf.nn.quantile
  2. Stretch: (x - P10) / (P90 - P10 + ε)
```
**Not ESP32-compatible** — `tf.nn.quantile` uses sorting operations absent from TFLite. Explicitly documented as such in the code.

#### Method C: Combined Luminance + Contrast (v24, ESP32-compatible)
```
ContrastAwareInput:
  1. Luminance conversion via learned weights (0.299, 0.587, 0.114)
  2. Mean-based polarity detection
  3. Conditional inversion
  4. Contrast stretching in one fused layer
```

---

### 1.3 Soft Contrast Normalization (v27 — Improvement over v24)

**The Problem with v24**: Hard min-max contrast stretching compresses dynamic range when outliers are present. For water meter images with specular highlights, a single bright pixel can collapse the entire contrast stretch.

**The Innovation**: Z-score normalization + sigmoid squash, applied per-image:

```
SoftContrastNormalization:
  1. μ = mean(x), σ = std(x)
  2. z = (x - μ) / (σ + ε), clip to [-5, +5]
  3. output = sigmoid(z)
```

This is **fully differentiable, numerically stable, and TFLite Micro compatible**: `reduce_mean`, `reduce_std`, `sub`, `div`, `sigmoid` — all are standard ops.

**Why sigmoid(x) where sigmoid saturates**: Clipping at ±5 preserves gradients within that range; beyond ±5, the sigmoid is effectively 0 or 1 with negligible gradient — which is fine because contrast extremes are already handled.

---

### 1.4 Learnable Soft Binarization (v27 — Novel)

**Introduced**: v27 (also present in rejected v26)

**The Innovation**: A trainable threshold for foreground/background separation.

```
LearnableSoftBinarization:
  threshold  = trainable weight (initialized 0.5, clipped [0.1, 0.9])
  sharpness  = 10 (10-class) or 5 (100-class) — fixed by task
  output     = sigmoid(sharpness × (x - threshold))
```

**Two key design insights**:

1. **Task-specific sharpness**: 10-class digit recognition wants near-binary separation (sharpness=10). 100-class transitional digit reading needs to preserve gradient information for sub-digit state estimation (sharpness=5). This is domain knowledge baked into architecture.

2. **Learnable threshold**: Rather than assuming mean-based or fixed thresholds, the model learns the optimal split point from the data. The weight clips to [0.1, 0.9] to prevent degenerate solutions.

**TFLite ops**: `sub`, `mul`, `sigmoid` — all standard.

---

### 1.5 Adaptive Mean Binarization with STE (v28 — Novel)

**Introduced**: v28

**The Innovation**: Replaces v27's multi-stage preprocessing (polarity augment → soft contrast norm → learnable binarization = 3 layers) with a single deterministic binarization using the Straight-Through Estimator:

```
AdaptiveMeanBinarization:
  Forward:  binary = float(x > mean(x))        # hard threshold
  Backward: ∂L/∂x ≈ ∂L/∂binary (identity)       # STE trick
  Code:     inputs + stop_gradient(binary - inputs)
```

**Why STE matters**: Hard thresholding (`x > mean`) has zero gradient almost everywhere. The STE tricks the gradient computation into flowing through as if the operation was the identity, enabling end-to-end training through a fundamentally non-differentiable operation.

**Deterministic inference**: No randomness, no learned parameters. Every image of the same digit produces exactly the same binary output.

---

### 1.6 Difference of Gaussians Edge Detection via Frozen Conv2D (v28 — Novel)

**Introduced**: v28 (optional `use_edge_fusion=True`)

**The Innovation**: A frozen, pre-computed Difference of Gaussians (DoG) edge detector using only standard Conv2D operations:

```
DoGEdgeDetection:
  G₁ = Gaussian blur (σ=0.8, kernel=3×3)    — fine detail
  G₂ = Gaussian blur (σ=1.6, kernel=5×5)    — coarse detail
  edge_map = |G₁(input) - G₂(input)|         — edges between scales
  normalized = edge_map / (max_edge + ε)     — per-image [0, 1]
```

**Implementation details**: 
- Gaussian kernels computed in `build()` via `_make_gaussian_kernel()`
- Stored as `self.add_weight(trainable=False)` for proper serialization
- Applied via `tf.nn.conv2d` with manually constructed kernels
- **Isotropic**: Unlike Sobel (axis-aligned), DoG detects edges equally in all directions

**WARNING**: Uses `tf.nn.conv2d` directly rather than `keras.layers.Conv2D`. This may cause TFLite conversion issues — the code's own header says "TFLite Micro compatible" but this hasn't been validated.

---

### 1.7 Adaptive Hybrid Binarization — 2-Channel Output (v29 — Most Novel)

**Introduced**: v29

**The Problem with v28**: Hard binarization destroys transitional gradient information. For a 100-class water meter model, differentiating 5.5 from 5.6 relies on subtle sub-digit shadow gradients. Hard binarization turns a 50% blend into a rigid 0 or 1.

**The Innovation**: A **2-channel preprocessing output**, with explicit design rationale documented in the code:

```
AdaptiveHybridBinarization(preserve_gradient=True):
  Channel 0: hard_binary = STE(x > mean(x))    → shape features
  Channel 1: soft_gradient = sigmoid((x - μ)/σ) → transitional cues
  Output: concat([binary, soft], axis=-1) → [B, H, W, 2]
```

**Why 2 channels is principled**:

| Channel | Information | Purpose | Gradients |
|---------|------------|---------|-----------|
| Ch 0 | Binary (0 or 1) | Lighting-invariant topology | STE identity |
| Ch 1 | Continuous [0, 1] | Subpixel wheel position | Full diff thru sigmoid |

The CNN's first Conv2D receives 2 input channels and learns to weight them optimally. The backbone architecture is identical to v4/v23/v24 — Keras handles the channel count change transparently.

**PolarityNormalization2Channel** simultaneously flips both channels based on the binary channel's mean, maintaining channel consistency.

---

## Part 2: Training Innovations

### 2.1 Probability-Gated Augmentation (augmentation.py)

**The Problem with Static Augmentation**: ADR-005 documents ~82K pre-generated augmented images, stored on disk with `weight: 0.5`. But training on the same static variants every epoch leads to memorization.

**The Innovation**: `_maybe_augment_one_image()` applies the augmentation pipeline with `AUGMENTATION_PROBABILITY` (controllable per run):

- If `probability < 1.0`: only a fraction of images get re-randomized each epoch
- The rest pass through unchanged
- Implemented via `tf.cond` with explicit `set_shape()` restoration to avoid TensorFlow graph mode shape errors

**Benefit**: Prevents overfitting to fixed augmented variants while maintaining epoch-to-epoch variation.

### 2.2 Augmentation Safety Monitor

**Installed**: `AugmentationSafetyMonitor` Keras callback

Auto-detects:
- Catastrophic validation loss (> `AUG_SAFETY_THRESHOLD`, default 100.0)
- Model not learning (val_acc < `AUG_LEARNING_THRESHOLD`, default 0.10, after `AUG_PATIENCE_EPOCHS`, default 10)
- NaN/Inf in validation data
- Unnormalized data (max > 5.0 for expected [0, 1] range)
- Train/val loss divergence (ratio > 5.0)

All thresholds are configurable via `config.py`.

### 2.3 Polarity Inversion as Training-Only Layer (v27 — Design Fix)

**The Problem with v24**: v24's `AdaptiveContrastNormalization` does inference-time statistical polarity detection. This adds an unreliable decision point: what if the image is 51% white? The flip decision flips erratically.

**The Fix (v27)**: Move polarity robustness entirely into the training data via `PolarityInversionAugmentation(probability=0.5)`. At inference, the layer is a no-op. The model weights learn contrast invariance naturally through exposure to both polarities during training. No inference-time conditional logic needed.

This is documented as "Fix 2" in the v27 header:
> "Removes unreliable inference-time detection from v24. Weights learn contrast invariance naturally — no inference-time flip ever needed."

---

## Part 3: Architecture Evolution as Bug-Fix Lineage

The model files form an explicit design narrative, with each version critiquing its predecessor:

| Model | Self-Described Purpose | File Header Quote |
|-------|----------------------|-------------------|
| **v23** | "Luminance Grayscale with Fixed Conv2D Weights" | "Add perceptual luminance grayscale conversion as an entry layer" |
| **v24** | "Adaptive Contrast Normalization" | "Handle digits written on both light-on-dark and dark-on-light backgrounds" |
| **v27** | "Improved 100-class Adaptive Contrast Model" | "4 fixes based on benchmark analysis (v23 -9.2pp, v24 -10.5pp drop at 100cls)" |
| **v28** | "Direct Binarization + Polarity Normalization" | "Replaces v27's multi-stage preprocessing... with a direct, deterministic binary pipeline" |
| **v29** | "Adaptive Hard Binarization + Transition Preservation" | "The Critique of v28: destroys transition information... v29 outputs TWO channels" |

Each header explicitly names what the previous version got wrong and proposes a principled fix. This is **architectural research as a dialogue with yourself** — a form of scientific record-keeping rarely seen in ML codebases.

---

## Part 4: Adaptive Architecture Scaling (v27, v28, v29)

**Innovation**: Filter counts and dense layer sizes scale with `NB_CLASSES` via a formula:

```python
scale = max(1.0, (NB_CLASSES / 10) ** 0.45)
filters = [max(int(f * scale), f) for f in [20, 36, 48, 56]]
dense = max(int(64 * scale), 64)
```

**Effect**:

| Task | Filters | Dense | Params |
|------|---------|-------|--------|
| 10-class | [20, 36, 48, 56] | 64 | ~91K |
| 100-class | [32, 58, 77, 90] | 102 | ~195K |

The 0.45 exponent means capacity grows sub-linearly with class count — reflecting the intuition that 100 classes require more capacity than 10, but not 10× more.

---

## Part 5: Knowledge Distillation Pipeline

### 5.1 Multi-Teacher Ensemble Distillation

The project contains a complete distillation framework in `train_super_student.py` and `distill_best.py`:

- **v32_xl**: A large teacher (2.5M params, 99.44% accuracy) for training smaller models
- **v33**: ConvNeXt-Tiny (10-15M params, 10-class) — modern pure-convolutional design with LayerNorm, GELU, 7×7 depthwise kernels, DropPath stochastic depth, and LayerScale
- **v34**: ConvNeXt-Tiny (100-class variant)

The v33 model contains a correct `DropPath` implementation using `tf.cond` (with explicit static shape restoration to avoid TensorFlow graph compilation errors) and `LayerScale` implemented as a `DepthwiseConv2D(1×1)` for proper weight tracking.

### 5.2 Distillation from RGB Teachers to Grayscale Students

The README claims: "Mixed Input Support: Distill from RGB teachers into efficient Grayscale students" — enabling knowledge transfer from high-capacity color models to tiny single-channel IoT models.

---

## Part 6: Quantization Maturity

### 6.1 Master Quantization Configuration

A single `QUANTIZATION_MODE` switch in `config/quantization.py` controls all underlying flags:

```python
QUANTIZATION_MODE = "tqt"   # Options: "none", "ptq", "qat", "tqt", "auto"
```

### 6.2 TQT — Trainable Quantization Thresholds (DEPRECATES QAT)

Per ADR-001: TQT fine-tunes quantization scales post-training, recovering 1-3% accuracy vs QAT. Chip-specific hyperparameters:

| Chip | Steps | LR | Block Size | Int Lambda |
|------|-------|----|------------|------------|
| ESP32 | 200 | 1e-6 | 2 | 0.10 |
| ESP32-S3 | 200 | 1e-6 | 2 | 0.05 |
| ESP32-P4 | 200 | 1e-6 | 2 | 0.0 |

Multi-target export: Set `TQT_EXPORT_ALL_TARGETS = True` to produce optimized models for all three chips from a single float model.

### 6.3 uint8 I/O Contract

TFLite Micro on ESP32 expects raw camera bytes in [0, 255] (uint8). The pipeline defaults to `USE_TFLITE_BUILTINS_UINT8_ONLY = True`. For ESP-DL SDK (expects int8), set `USE_TFLITE_BUILTINS_INT8_ONLY = True` — the model then requires `pixel_int8 = pixel_uint8 - 128` in C++.

---

## Part 7: Domain-Specific Design Decisions (ADRs)

The 6 Architecture Decision Records document explicit trade-off reasoning:

| ADR | Decision | Key Insight |
|-----|----------|------------|
| ADR-001 | **TQT over QAT** | TQT recovers 1-3% accuracy, no training-time overhead |
| ADR-002 | **GhostNet family primary** | v18 achieves >90% at <100KB — first family to cross this threshold |
| ADR-003 | **Grayscale default** | Water meter digits are inherently grayscale. RGB adds no information |
| ADR-004 | **RMSprop for 100-class cold-start** | Handles QAT gradient noise better than Adam |
| ADR-005 | **Static augmentation over online** | Deterministic reproducibility, 2-5× epoch speedup |
| ADR-006 | **10-class primary use case** | 100-class is research stress test, not deployment target |

---

## Summary: 16 Novel Practices Across 7 Categories

| Part | Category | Practices Covered |
|------|----------|-------------------|
| Part 1 | Preprocessing-as-a-Layer | Luminance Conv2D, 3 contrast methods, soft normalization, learnable binarization, STE binarization, DoG edges, 2-channel hybrid |
| Part 2 | Training Innovations | Probability-gated augmentation, safety monitor, polarity inversion training |
| Part 3 | Bug-Fix Architecture Lineage | v23→v24→v27→v28→v29 explicit design evolution |
| Part 4 | Adaptive Scaling | Sub-linear filter/dense scaling with NB_CLASSES |
| Part 5 | Distillation Pipeline | Multi-teacher ensemble, ConvNeXt super-students, cross-color-space distillation |
| Part 6 | Quantization Maturity | TQT pipeline, master config, uint8 I/O contract |
| Part 7 | Architecture Decision Records | 6 ADRs documenting key trade-off decisions |

These 16 practices span the full depth of the project: from input preprocessing to model architecture to training methodology to hardware deployment. Each represents a principled engineering decision with documented rationale, and collectively they form a cohesive system for TinyML digit recognition on ESP32-class hardware.

---

## References

1. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. — ConvNeXt architecture used in v33/v34 super-students.

2. Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv:1704.04861*. — Depthwise separable convolutions foundation.

3. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. — Inverted residual blocks used in v16 teacher.

4. Han, K., et al. (2020). GhostNet: More features from cheap operations. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. — Ghost module design used in v17.

5. Tang, Y., et al. (2022). GhostNetV2: Enhance cheap operation with long-range attention. *Advances in Neural Information Processing Systems*, 35, 9969–9982. — GhostNet evolution.

6. Marr, D., & Hildreth, E. (1980). Theory of edge detection. *Proceedings of the Royal Society of London. Series B*, 207(1167), 187–217. — Difference of Gaussians edge detection (v28).

7. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91–110. — DoG for scale-space feature detection (v28).

8. Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep networks with stochastic depth. *European Conference on Computer Vision*. — DropPath/Stochastic depth for ConvNeXt training.

9. Touvron, H., et al. (2021). Going deeper with image transformers. *Proceedings of the IEEE/CVF International Conference on Computer Vision*. — LayerScale initialization used in ConvNeXt blocks.

10. Yin, P., et al. (2019). Understanding and improving straight-through estimators. *International Conference on Learning Representations*. — STE theory for hard binarization (v28).

11. Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or augmenting gradients in binary hierarchical networks with l0-regularization. *arXiv:1308.3432*. — Original STE formulation.

12. Jacob, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. — QAT foundation.

13. Li, S., et al. (2021). PPQ: A practical platform for post-training quantization. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. — PPQ framework underlying TQT.

14. Espressif Systems. (2025). ESP-PPQ: Post-Training Quantization Toolkit for Espressif Chips. GitHub: \url{https://github.com/espressif/esp-ppq}. — TQT implementation for ESP32.

15. David, R., et al. (2021). TensorFlow Lite Micro: Embedded machine learning on TinyML systems. *Proceedings of Machine Learning and Systems*, 3, 800–811. — TFLite Micro framework.

16. ITU-R. (2011). Studio encoding parameters of digital television for standard 4:3 and wide-screen 16:9 aspect ratios. Recommendation BT.601-7. — Luminance weights [0.299, 0.587, 0.114].

17. Lin, T.-Y., et al. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision*. — Focal loss foundation.

18. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NIPS Deep Learning and Representation Learning Workshop*. — Knowledge distillation foundation.

19. Lin, J., et al. (2021). MCUNetV2: Memory-efficient patch-based inference for tiny deep learning. *Advances in Neural Information Processing Systems*, 34. — TinyML memory optimization.

20. Lin, J., et al. (2020). MCUNet: Tiny deep learning on IoT devices. *Advances in Neural Information Processing Systems*, 33, 11711–11722. — TinyML architecture search.
