# Digit Recognizer — Model Enhancement Proposals

## Background

This project recognizes rotating digits (0–9 / 0–99) from 32×20-pixel images and deploys to ESP32 microcontrollers via INT8-quantized TFLite Micro / ESP-DL.  Current state-of-the-art results:

| Target | Best model | Accuracy | Size |
|--------|-----------|----------|------|
| 10-class RGB | v16 TQT | **99.5%** | 128 KB |
| 10-class RGB (balanced) | v24 TQT | **98.9%** | 69 KB |
| 100-class RGB | v16 | **93.4%** | 140 KB |
| 100-class gray | v19 | **91.6%** | 146 KB |

Key bottleneck for 10-class: already saturating; every 0.1% matters.  
Key bottleneck for 100-class: large gap (~6%) vs 10-class — structured prediction and better features needed.

---

## Open Questions

> [!IMPORTANT]
> **Q1 — Primary objective:** Is the focus on pushing 10-class accuracy above 99.5%, improving 100-class (currently 93.4%), or reducing model size while preserving accuracy?

> [!IMPORTANT]
> **Q2 — Dataset availability:** The Haverland dataset is marked unavailable as of 20.06.2026. Are the locally augmented 82K images still available for training experiments?

> [!IMPORTANT]
> **Q3 — Implementation priority:** Should enhancements be proposed as *architecture changes only* (new model versions you implement later), or should code be written and integrated into `train.py` now?

> [!NOTE]
> **Q4 — Hardware budget:** Is the 100–150 KB INT8 limit a hard constraint, or would 200 KB be acceptable for a high-accuracy 100-class model?

---

## Proposed Enhancements

Organized by category and effort level (🟢 Low / 🟡 Medium / 🔴 High).

---

### 1. New Model Architectures

---

#### 1a. v25 — EfficientNet-Lite Nano (compound scaling, TFLite-safe) 🟡

**Motivation:** EfficientNet-Lite uses compound depth/width/resolution scaling and explicitly targets TFLite Micro (no SE blocks, ReLU6 everywhere). No version of this exists in the project.

**Architecture concept** (scaled for 32×20 input):
```
Input
→ Conv2D(16, 3×3, stride=1) + BN + ReLU6          [stem]
→ MBConv1(out=16, k=3, stride=1)                   [stage 1 — no expansion]
→ MBConv6(out=24, k=3, stride=2)                   [stage 2 — spatial /2]
→ MBConv6(out=24, k=3, stride=1)                   [stage 2 residual]
→ MBConv6(out=40, k=5, stride=2)                   [stage 3 — wider receptive field]
→ MBConv6(out=48, k=5, stride=1)                   [stage 3 residual]
→ Conv2D(112, 1×1) + BN + ReLU6                    [head expansion]
→ GlobalAveragePooling2D → Dense(NB_CLASSES)
```

**MBConv block:** `1×1 expand → 3×3 or 5×5 DepthwiseConv(stride) → 1×1 project`, all BN+ReLU6. 5×5 depthwise captures wider digit context for less cost than two 3×3 convs.

**Why better than v16:**
- 5×5 DW convolutions provide larger receptive field per block → better for rotated digit shapes
- Compound-scaled: width and depth grow together rather than ad-hoc
- Estimated params: ~120K, estimated accuracy: ≥99.3% (RGB TQT)

**TFLite safety:** All ops (Conv2D, DepthwiseConv2D, BN, ReLU6, GAP, Dense) are built-in. No custom ops.

---

#### 1b. v26 — MixDepthwise CNN (multi-scale receptive fields) 🟡

**Motivation:** Rotated digits present strokes at all angles. A single 3×3 kernel captures only one spatial scale. Mixing kernel sizes in the same depthwise layer (MixConv) captures multi-scale context without extra depth.

**MixDW block concept:**
```python
# Split channels into groups, apply different kernel sizes, concat
ch_3x3 = DepthwiseConv2D(3×3)(x[:, :, :, :c//2])
ch_5x5 = DepthwiseConv2D(5×5)(x[:, :, :, c//2:])
out = Concatenate([ch_3x3, ch_5x5])  # same channel count
```
All ops TFLite-safe. Concatenate + DepthwiseConv2D are built-in ops.

**Full architecture:**
```
Input → Conv2D(24, 3×3) + BN + ReLU6 + MaxPool(2×2)
      → MixDW(48 ch: 24 × 3×3 + 24 × 5×5) + Pointwise(48, 1×1) + BN + ReLU6
      → MaxPool(2×2)
      → MixDW(64 ch: 32 × 3×3 + 32 × 5×5) + Pointwise(64, 1×1) + BN + ReLU6
      → GlobalAveragePooling2D → Dense(64) + ReLU6 → Dense(NB_CLASSES)
```

Estimated size: ~60 KB. Rotation-robust by design.

---

#### 1c. v30 — Hierarchical Dual-Head for 100-class Recognition 🔴

**Motivation:** 100-class output encodes a structured domain: tens digit + units digit. A flat softmax over 100 classes ignores this structure. Digits like "07" vs "70" differ only in digit ordering — the current architecture has no inductive bias for this. A hierarchical head explicitly separates the two sub-problems.

**Architecture:**
```
Shared backbone (e.g. v16 or v19 backbone)
         ↓ GlobalAveragePooling2D
    Feature vector (128-dim)
      /              \
Dense(32) + ReLU6   Dense(32) + ReLU6
      |                    |
Dense(10, softmax)   Dense(10, softmax)
  [tens digit]        [units digit]
      \              /
      outer_product  → reshape to (100,) → log_softmax
```

**Loss:** Sum of two cross-entropy losses (one per head) + optional combined cross-entropy on the 100-class product.

**Why it works:** The backbone still learns 100-class features, but the hierarchical head forces separate learning of position-invariant digit identity. Expected gain: +1–3% on 100-class over flat softmax, especially for hard pairs like (06,60), (17,71), (69,96).

**TFLite export:** Both Dense heads export cleanly. Outer product via `Reshape` + `einsum` may need to be replaced with `MatMul` or pre-flattened; the outer product operation can alternatively be replaced with `Multiply` + `Reshape`.

---

#### 1d. v31 — Channel Attention (CBAM-Lite) Augmented Model 🟡

**Motivation:** Channel attention (squeeze-and-excite) re-weights feature maps by learned importance. The existing v32 teacher has SE blocks, but no deployable IoT student does. A lightweight version can be TFLite Micro safe.

**Lightweight Channel Attention (CA) block:**
```python
# Input: (H, W, C)
gap = GlobalAveragePooling2D(keepdims=True)  # → (1,1,C)
fc1 = Conv2D(max(C//8, 4), 1, activation='relu6')(gap)  # squeeze
fc2 = Conv2D(C, 1, activation='sigmoid')(fc1)           # excite
out = Multiply()([x, fc2])                              # recalibrate
```

All ops are Conv2D, Sigmoid, Multiply — TFLite built-ins. No GlobalAveragePooling1D or custom ops needed.

**Placement:** After the final Conv2D block in v17/v18/v19 GhostNet variants, before GAP. Cost: +~500 params per CA block, estimated accuracy gain: +0.2–0.5%.

---

### 2. Training Enhancements

---

#### 2a. SAM Optimizer (Sharpness-Aware Minimization) 🟡

**Motivation:** SAM finds flatter loss minima which generalize better AND are more robust to quantization (INT8 quantization introduces perturbations that flat minima tolerate better). This is a direct path to better TQT post-quantization accuracy without changing the model.

**Implementation:**
```python
class SAM(tf.keras.optimizers.Optimizer):
    def __init__(self, base_optimizer, rho=0.05, **kwargs):
        # First step: compute gradient, perturb weights by rho
        # Second step: compute gradient at perturbed point, restore weights, apply
```

Or use the well-maintained `keras-sam` package. SAM doubles the forward/backward passes per step — training is ~2× slower, but typically recovers 0.1–0.5% accuracy.

**Recommended config:** `SAM(Nadam(lr=1e-3), rho=0.05)`. Add `'sam'` to the optimizer list in `config/models.py`.

---

#### 2b. EMA Weights (Exponential Moving Average) 🟢

**Motivation:** EMA of weights during training smooths out high-frequency weight fluctuations. The EMA copy is used only at evaluation/export time. Well-established technique — used in MobileNetV3, EfficientNet training recipes.

**Implementation:**
```python
class EMACallback(tf.keras.callbacks.Callback):
    def __init__(self, decay=0.9999):
        self.ema_weights = None
        self.decay = decay
    def on_batch_end(self, batch, logs=None):
        if self.ema_weights is None:
            self.ema_weights = [w.numpy() for w in self.model.weights]
        else:
            for i, w in enumerate(self.model.weights):
                self.ema_weights[i] = self.decay * self.ema_weights[i] + (1-self.decay) * w.numpy()
    def on_epoch_end(self, epoch, logs=None):
        # Temporarily swap to EMA weights for val accuracy computation
```

**Expected gain:** +0.05–0.2% validation accuracy, especially beneficial at the 99%+ regime where training fluctuations are the limiting factor.

---

#### 2c. Stochastic Depth (Drop Path) for Residual Models 🟢

**Motivation:** Randomly dropping entire residual blocks during training forces earlier layers to be more capable. Proven regularizer for ResNet/MobileNetV2 style models. Currently only Dropout is used.

**Implementation:**
```python
def drop_path(x, drop_prob, training):
    if not training or drop_prob == 0.0:
        return x
    keep = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) > drop_prob
    return tf.cast(keep, x.dtype) * x / (1.0 - drop_prob)
```

**Target models:** v15 (ResBlocks), v16 (InvRes blocks), v17–v19 (GhostBlocks). Drop rates: linearly scale from 0.0 (first block) to 0.1–0.15 (last block). Note: Drop Path is training-only, has no effect at inference — 100% TFLite Micro safe.

---

#### 2d. Cosine Annealing with Warm Restarts (SGDR) 🟢

**Motivation:** The current scheduler uses ReduceLROnPlateau → cosine transition. SGDR (cosine annealing with warm restarts) escapes local minima more aggressively by periodically resetting LR to a warm value.

**Implementation:**
```python
def cosine_annealing_with_restarts(epoch, T_0=50, T_mult=2, eta_max=1e-3, eta_min=1e-7):
    t = epoch % T_0  # reset period
    T_0 *= T_mult    # double period after each restart
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_0))
```

Add as a `LearningRateSchedule` option in `config/training.py` alongside the existing dynamic scheduler.

---

#### 2e. Mixup / CutMix Augmentation 🟡

**Motivation:** Mixup (linear interpolation of image+label pairs) and CutMix (paste crop from one image into another) are the most effective standard augmentations for classification beyond data-level transforms. Neither is currently implemented.

**Mixup:**
```python
lam = np.random.beta(alpha=0.2, size=batch_size)
x_mix = lam * x1 + (1-lam) * x2
y_mix = lam * y1 + (1-lam) * y2  # soft labels
```

**CutMix:** Replace a random rectangle of `x1` with pixels from `x2`, mix labels proportionally to area.

**Caution for digit recognition:** Mixup is more appropriate than CutMix here — CutMix can produce ambiguous crops that corrupt the digit structure (e.g. mixing top of "7" with bottom of "1"). Use `alpha=0.1–0.2` to keep mixing mild.

---

### 3. Augmentation Improvements

---

#### 3a. Rotation-Specific Hard Augmentation 🟢

**Motivation:** Hard classes noted in v18 are 6, 7, 9. Digits "6" and "9" differ only by 180° rotation; "7" vs "1" differs by slight rotation+perspective. Specifically targeting these with heavier rotation augmentation could reduce confusion.

**Implementation:** Class-conditional augmentation probability in the static augmentation pipeline:
```python
HARD_CLASSES = {6, 7, 9}
HARD_CLASS_EXTRA_ROTATION = 10  # additional ±10° for hard classes
HARD_CLASS_EXTRA_PERSPECTIVE = True
```

Add to `config/augmentation.py` as `HARD_CLASS_AUGMENTATION=True`.

---

#### 3b. Quantization Noise Injection During Training 🟡

**Motivation:** Post-training quantization introduces rounding errors. Training with simulated INT8 noise in the forward pass forces the model to learn representations robust to this degradation — a regularization technique complementary to TQT.

**Implementation (after float training, before TQT):**
```python
class QuantNoiseDense(tf.keras.layers.Dense):
    """Dense layer with simulated INT8 quantization noise during training."""
    def call(self, inputs, training=None):
        w = self.kernel
        if training:
            scale = tf.reduce_max(tf.abs(w)) / 127.0
            w_quant = tf.round(w / scale) * scale  # simulate INT8 rounding
            w = w + tf.stop_gradient(w_quant - w)  # STE
        return tf.matmul(inputs, w) + self.bias
```

This is effectively LSQ (Learned Step Size Quantization) applied selectively to dense layers. Expected gain: +0.1–0.3% TQT accuracy vs standard float training → TQT pipeline.

---

#### 3c. Perspective Distortion Augmentation 🟢

**Motivation:** Digits photographed at angles undergo perspective distortion. The current augmentation includes shear but not a full perspective transform (4-point homography). This would make models more robust to real-world camera angles.

**Implementation:**
```python
import cv2
def random_perspective(img, distortion=0.1):
    h, w = img.shape[:2]
    # Perturb corners by up to distortion × dimension
    pts_src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    pts_dst = pts_src + np.random.uniform(-distortion, distortion, (4,2)) * [w, h]
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(img, M, (w, h))
```

Add to `config/augmentation.py` as `USE_PERSPECTIVE_DISTORTION=True`.

---

### 4. Quantization Improvements

---

#### 4a. Per-Layer TQT Learning Rate Scaling 🟡

**Motivation:** Currently all TQT calibration uses a uniform LR (1e-6) for all layers. First and last layers are typically more sensitive to quantization than middle layers. Per-layer LR scaling could improve TQT convergence.

**Proposed:**
```python
TQT_PER_LAYER_LR_SCALE = {
    'first_conv': 0.5,   # entry conv — more sensitive
    'dense': 0.3,        # classification head — most sensitive
    'default': 1.0       # all other layers
}
```

---

#### 4b. Mixed-Precision INT8/INT16 for 100-class Head 🟡

**Motivation:** The 100-class output Dense layer has fine-grained probability distinctions between similar classes. INT8 quantization of the final Dense layer loses resolution. ESP-DL supports INT16 for specific layers.

**Proposal:** Keep backbone INT8, quantize final Dense + one preceding Dense to INT16.  
Estimated accuracy gain on 100-class: +0.5–1.0%.  
Estimated size overhead: +4–8 KB.

---

#### 4c. Repair INT8 Bias Clamp 🟢

**Motivation:** The existing `repair_int8.py` suggests there are known issues with INT8 bias values being clamped incorrectly. This should be generalized and integrated into the standard export pipeline rather than being a separate repair script.

---

### 5. Architecture Comparison Summary

| Model | Type | Est. Params | Est. Size (10cls RGB INT8) | Est. Accuracy | Key Advantage |
|-------|------|------------|--------------------------|---------------|---------------|
| **Current v16** | MobileNetV2 | 262K | 128 KB | 99.5% | Highest accuracy |
| **Current v24** | Adaptive CNN | 158K | 69 KB | 98.9% | Best size/accuracy |
| **Proposed v25** | EfficientNet-Lite Nano | ~120K | ~65 KB | ≥99.3% | Compound scaling, 5×5 DW |
| **Proposed v26** | MixDepthwise CNN | ~80K | ~55 KB | ~99.0% | Multi-scale receptive field |
| **Proposed v31** | v17+CA attention | ~85K | ~75 KB | ~99.2% | Channel attention, GhostNet |
| **Proposed v30** | Hierarchical dual-head | ~280K | ~145 KB | ~95–96% (100cls) | Structured 100-class output |

---

## Verification Plan

### Per Enhancement
1. **New architecture (v25, v26, v30, v31):** Train on 10-class gray first (fastest feedback), compare val accuracy vs v4/v7 baseline. Then RGB. Then TQT export and compare TQT accuracy delta.
2. **Training enhancements (SAM, EMA, drop path, SGDR):** A/B train v16 or v24 with vs without, compare 5-fold convergence curves.
3. **Augmentation (mixup, perspective):** Compare best val_accuracy on same model (v16) with/without.
4. **Quantization improvements (per-layer TQT LR):** Compare TQT accuracy on v16 with default uniform LR vs scaled LR.

### Automated
- All new model versions register in `AVAILABLE_MODELS` in `config/models.py`
- Run `bench_predict.py` for inference throughput after TQT export
- Run `generate_summary_csv.py` to compare against `model_comparison.csv` baseline

### Manual
- Verify TFLite I/O type is `uint8` (not `int8`) using `check_model_type.py` before hardware deployment
- Flash the best new model to ESP32 and verify inference confidence is ≥ existing models

---

## Recommended Priority Order

| Priority | Enhancement | Effort | Expected Gain |
|----------|------------|--------|---------------|
| 🥇 1 | **v25 EfficientNet-Lite Nano** architecture | 🟡 Medium | +accuracy at same size as v24 |
| 🥈 2 | **EMA Weights callback** | 🟢 Low | +0.1–0.2% free at end of training |
| 🥉 3 | **Stochastic Depth** for v15/v16/v17–v19 | 🟢 Low | +0.1–0.3% regularization |
| 4 | **v30 Hierarchical head** for 100-class | 🔴 High | +1–3% on 100-class |
| 5 | **Mixup augmentation** | 🟡 Medium | +0.1–0.2% |
| 6 | **SAM optimizer** | 🟡 Medium | Better quantization robustness |
| 7 | **v26 MixDepthwise** | 🟡 Medium | Alternative efficiency/accuracy tradeoff |
| 8 | **Channel Attention v31** | 🟡 Medium | +0.2–0.5% GhostNet variant |
| 9 | **Quantization noise injection** | 🟡 Medium | +0.1–0.3% TQT accuracy |
| 10 | **Per-layer TQT LR scaling** | 🟡 Medium | Fine-tuning quantization |
