# Implementation Walkthrough — 4 New Architectures

I've implemented the 4 new model architectures proposed in the plan and integrated them directly into your training pipeline.

## 1. Model Details

### [digit_recognizer_v35.py](file:///c:/Users/nl/Dropbox/home_automation/digit_recognizer/models/digit_recognizer_v35.py) (MobileNetV2 + SE)
- **Target:** IoT deployment (<200KB INT8).
- **Architecture:** Builds on the successful MobileNetV2 inverted residual blocks from `v16`, but adds a lightweight, TFLite-safe Squeeze-and-Excitation (SE) channel attention module. It also uses slightly wider channels (`entry_conv` is 20 instead of 16).
- **Estimated Size:** ~275KB INT8.

### [digit_recognizer_v36.py](file:///c:/Users/nl/Dropbox/home_automation/digit_recognizer/models/digit_recognizer_v36.py) (GhostNet + SE)
- **Target:** Extreme efficiency for IoT (~100KB).
- **Architecture:** Builds on the `v19` GhostNet model which generates 50% of its features using cheap depthwise operations. I've integrated the SE channel attention block inside the `_ghost_block` to reweight the features before projection.
- **Estimated Size:** ~75KB INT8.

### [digit_recognizer_v37_teacher.py](file:///c:/Users/nl/Dropbox/home_automation/digit_recognizer/models/digit_recognizer_v37_teacher.py) (Wide MobileNetV3-style)
- **Target:** High-capacity PC-only teacher for knowledge distillation (train using `train_distill.py`).
- **Architecture:** An extremely deep and wide version of the MobileNetV2+SE architecture. It ends in a massive 1024 → 512 dense head with Dropout to maximize 100-class accuracy before being distilled into `v35` or `v36`.
- **Estimated Size:** ~5MB (Float32).

### [digit_recognizer_v38.py](file:///c:/Users/nl/Dropbox/home_automation/digit_recognizer/models/digit_recognizer_v38.py) (RepVGG-style)
- **Target:** IoT deployment.
- **Architecture:** Uses structural reparameterization. During training, each block contains three parallel branches (`3x3`, `1x1`, and `identity`). This allows for much richer gradient flow. Before exporting to TFLite, calling `model.reparameterize()` mathematically folds these three branches into a single `3x3` convolution layer, resulting in **zero** multi-branch inference overhead.

---

## 2. Integration

I've updated `AVAILABLE_MODELS` in [config/models.py](file:///c:/Users/nl/Dropbox/home_automation/digit_recognizer/config/models.py) to include all four new models.

They are fully integrated and ready to be run via:
```powershell
# In config/models.py, set MODEL_ARCHITECTURE to "digit_recognizer_v35"
python train.py
```

> **Important for v38:** Because `digit_recognizer_v38.py` uses structural reparameterization, it must be put into deployment mode before quantization or export. You can call `model.reparameterize()` after loading the trained `.keras` weights and before calling `tfmot.quantization.keras.quantize_model()`.

---

## 3. Training & Augmentation Enhancements

As part of the plan to push accuracy even higher on 10-class and 100-class tasks, several advanced training techniques have been implemented:

### Optimizer Improvements
- **Exponential Moving Average (EMA) Weights:** Added `USE_EMA` and `EMA_MOMENTUM` to `config/training.py`. When enabled, the optimizer maintains a moving average of the weights. This smooths out parameter trajectories, leading to better generalization and significantly more stable Quantization-Aware Training (QAT).
- **Sharpness-Aware Minimization (SAM):** Implemented `SAMModelWrapper` in `utils/optimizers.py` and added `USE_SAM` to `config/training.py`. SAM forces the model to find flatter minima, which dramatically improves generalization on small IoT models that are prone to overfitting.
- **Cosine Annealing (SGDR):** The `CosineDecayRestarts` LR scheduler is already built-in and can be activated by setting `LR_SCHEDULER_TYPE = "cosine"`.

### Data Augmentation
Three advanced augmentations were added to `config/augmentation.py` and the pipeline in `utils/augmentation.py`:
1. **Targeted Hard Rotation (`OccasionalHardRotation`):** Periodically applies rotations up to ±45° (configurable via `AUGMENTATION_ROTATION_HARD_RANGE`). This forces the model to learn subtle rotational invariants without permanently confusing rotation-sensitive digits like 6 and 9.
2. **Quantization Noise Injection (`QuantizationNoiseAugmentation`):** Adds uniform noise scaled to half an INT8 step `[-1/510, 1/510]` during training. This prepares the float32 model to be robust against rounding errors, bridging the gap before actual QAT.
3. **Perspective Distortion (`PerspectiveDistortionAugmentation`):** An augmentation stub to simulate camera angle viewing. (Currently passes through safely, ready for `tfa` integration or custom image mapping if needed).
