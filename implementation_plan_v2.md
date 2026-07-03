# Implementation Plan — 4 New Model Architectures

## Goal
Build 4 new model files that integrate directly into `train.py` via the existing factory pattern:
- **2 IoT models** (<200 KB INT8, deployable to ESP32)
- **1 teacher model** (PC-only, large capacity, distillation source)
- **1 reparameterizable IoT model** (stronger training, zero inference overhead)

All models follow the existing conventions:
- Functional Keras API (`tf.keras.Input → tf.keras.Model`)
- `params.INPUT_SHAPE`, `params.NB_CLASSES`, `params.USE_LOGITS` from `config as params`
- `create_digit_recognizer_vXX()` entry point
- `create_qat_model()` wrapper
- `ReLU(max_value=6.0)` (not `Activation(tf.nn.relu6)`)
- All TFLite Micro built-in ops for IoT models: Conv2D, DepthwiseConv2D, BN, ReLU6, Add, Concat, Multiply, GAP, Dense

---

## Proposed Changes

### New Files

---

#### [NEW] digit_recognizer_v35.py
**MobileNetV2 + Squeeze-and-Excitation (IoT, best accuracy <200 KB)**

Upgrade of v16 — the current best IoT model (99.5% at 128 KB RGB). Adds SE channel
attention and wider channels to push past 99.6%.

**SE block** (TFLite-safe, same technique as v32 teacher):
```
GAP(keepdims=True) → Conv2D(ch//8, 1×1) + ReLU6 → Conv2D(ch, 1×1) + Sigmoid → Multiply
```

**InvRes+SE block:**
```
expand(1×1) + BN + ReLU6
→ DW(3×3, stride) + BN + ReLU6
→ SE(reduction=8)              ← NEW vs v16
→ project(1×1) + BN            ← linear bottleneck
→ Add (skip if stride==1, ch_in==ch_out)
```

**Full architecture:**
```
Input (32×20×C)
→ Conv2D(20, 3×3) + BN + ReLU6                          [entry, wider than v16's 16ch]
→ InvRes+SE(out=28,  expand=4, stride=2)                 [spatial /2]
→ InvRes+SE(out=28,  expand=4, stride=1)                 [residual]
→ InvRes+SE(out=48,  expand=4, stride=2)                 [spatial /4]
→ InvRes+SE(out=48,  expand=6, stride=1)                 [residual]
→ InvRes+SE(out=64,  expand=6, stride=1)                 [extra depth vs v16]
→ InvRes+SE(out=64,  expand=6, stride=1)                 [extra block for 100cls capacity]
→ Conv2D(128, 1×1) + BN + ReLU6                         [head expansion]
→ GAP(keepdims=True) → Flatten
→ Dense(NB_CLASSES, softmax)
```

**Estimates:** ~220K params / ~175 KB INT8 (10cls RGB) — within 200 KB budget.
**Target accuracy:** 10cls RGB ≥99.6% TQT | 100cls RGB ≥94.5% TQT

---

#### [NEW] digit_recognizer_v36.py
**GhostNet + Squeeze-and-Excitation (IoT, efficient, ~100 KB)**

Upgrade of v19 — the best GhostNet (91.6% at 145 KB). SE attention added inside
each Ghost Bottleneck.

**Full architecture:**
```
Input (32×20×C)
→ Conv2D(20, 3×3) + BN + ReLU6
→ GhostBlock+SE(out=32, stride=2)
→ GhostBlock+SE(out=40, stride=1)
→ GhostBlock+SE(out=56, stride=2)
→ GhostBlock+SE(out=56, stride=1)
→ GhostBlock+SE(out=72, stride=1)
→ Conv2D(120, 1×1) + BN + ReLU6
→ GAP → Dense(NB_CLASSES, softmax)
```

**Estimates:** ~130K params / ~105 KB INT8 (10cls RGB)
**Target accuracy:** 10cls RGB ≥99.2% TQT | 100cls RGB ≥93% TQT

---

#### [NEW] digit_recognizer_v37_teacher.py
**Wide MobileNetV3-style Teacher (PC-only, distillation, ~5M params)**

Large, high-accuracy teacher for distillation into v35/v36/v16:
- 11 InvRes+SE blocks, very wide channels (up to 192ch)
- Dense head: GAP → Dense(1024) + Dropout(0.4) → Dense(512) → Dense(NB_CLASSES)
- Not quantized for ESP32, FP32/FP16 only

**Estimates:** ~5.2M params / ~20 MB FP32
**Target accuracy:** 10cls RGB ≥99.8% | 100cls RGB ≥96%

---

#### [NEW] digit_recognizer_v38.py
**RepVGG-style Reparameterizable CNN (IoT, multi-branch training → single-branch inference)**

Reference: Ding et al. "RepVGG" CVPR 2021.

During **training**: each block is 3 parallel branches (richer gradients):
```
x → Conv2D(3×3)+BN → y1
x → Conv2D(1×1)+BN → y2
x → Identity+BN    → y3   (only if ch_in == ch_out)
output = ReLU6(y1 + y2 + y3)
```

At **export**: `reparameterize()` folds all branches into a **single Conv2D(3×3)** + bias.
The exported TFLite model is a plain VGG-style net with zero multi-branch overhead.

**Full architecture (training mode):**
```
Input (32×20×C)
→ RepBlock(24, stride=1)  [entry]
→ RepBlock(32, stride=2)  [spatial /2]
→ RepBlock(32, stride=1)  [residual]
→ RepBlock(48, stride=2)  [spatial /4]
→ RepBlock(48, stride=1)  [residual]
→ RepBlock(64, stride=1)  [deeper]
→ Conv2D(96, 1×1) + BN + ReLU6
→ GAP → Dense(NB_CLASSES, softmax)
```

**Estimates (training):** ~270K params | **(after reparameterize):** ~90K / ~72 KB INT8
**Target accuracy:** 10cls RGB ≥99.2% | Better dynamics than v7/v6

> [!IMPORTANT]
> `model.reparameterize()` must be called before TFLite export. I will add detection
> of v38 models in `export_tflite.py` as part of this work.

---

### Modified Files

#### [MODIFY] [config/models.py](file:///c:/Users/nl/Dropbox/home_automation/digit_recognizer/config/models.py)
Add 4 new entries to `AVAILABLE_MODELS`.

---

## Verification Plan

### Build checks
```powershell
python models\digit_recognizer_v35.py
python models\digit_recognizer_v36.py
python models\digit_recognizer_v37_teacher.py
python models\digit_recognizer_v38.py
```
Each prints param count and estimated INT8 KB. Verify against targets.

### Factory integration check
```powershell
python -c "from models.model_factory import create_model_by_name; [print(create_model_by_name(n, num_classes=10, input_shape=(32,20,3)).output_shape) for n in ['digit_recognizer_v35','digit_recognizer_v36','digit_recognizer_v37_teacher','digit_recognizer_v38']]"
```

### v38 reparameterize check
```powershell
python -c "
import sys; sys.path.insert(0,'c:/Users/nl/Dropbox/home_automation/digit_recognizer')
from models.digit_recognizer_v38 import create_digit_recognizer_v38
import tensorflow as tf, numpy as np
m = create_digit_recognizer_v38()
x = tf.random.uniform((2,32,20,3))
y_before = m(x, training=False).numpy()
m.reparameterize()
y_after = m(x, training=False).numpy()
print('Max diff after reparameterize:', np.max(np.abs(y_before - y_after)))
"
```
Max diff should be < 1e-4 (floating point rounding only).
