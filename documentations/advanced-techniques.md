# Advanced Training Techniques Documentation

## Overview
This document explains the advanced training techniques implemented in the digit_recognizer project: Intelligent Focal Loss, Dynamic Per-Class Weighting, and Quantization-Aware Training (QAT). These techniques significantly improve model performance, especially for challenging edge deployment scenarios where both accuracy and resource constraints matter.

## Table of Contents
1. [Intelligent Focal Loss](#1-intelligent-focal-loss)
2. [Dynamic Per-Class Weighting](#2-dynamic-per-class-weighting)
3. [Quantization-Aware Training (QAT)](#3-quantization-aware-training-qat)
4. [Combined Strategy: Best Practices](#4-combined-strategy-best-practices)
5. [Troubleshooting](#5-troubleshooting)
6. [References](#6-references)
7. [Appendix: Example Configurations](#appendix-example-configurations)

---

## 1. Intelligent Focal Loss

### Theory Behind Focal Loss
Introduced by Lin et al. (2017), Focal Loss addresses class imbalance by down-weighting "easy" examples, preventing them from overwhelming the gradient during training.

#### Standard Cross-Entropy
For multi-class problems:
$CE(p_t) = -\log(p_t)$
Where $p_t$ is the model's estimated probability for the ground-truth class.

#### Focal Loss Formula
$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$

| Property | Effect |
| :--- | :--- |
| **Large $p_t$ (Easy)** | $(1-p_t)^\gamma$ is small → loss is down-weighted |
| **Small $p_t$ (Hard)** | $(1-p_t)^\gamma$ is large → loss is preserved |
| **$\gamma = 0$** | Equivalent to standard cross-entropy |

#### Alpha-Balanced Focal Loss
$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$
Where $\alpha_t$ is a weighting factor for the ground-truth class.

### Intelligent Focal Loss Controller
Our implementation automatically determines when to apply Focal Loss based on validation performance.

```python
class IntelligentFocalLossController:
    """
    Stages:
    1. Cross-Entropy (Epochs 0-10): Learn basic patterns.
    2. Transition (Epochs 10-15): Gradually introduce Focal Loss.
    3. Pure Focal Loss (Epochs 15+): Focus on hard-to-distinguish digits.
    """
```

### Hyperparameter Tuning
#### Gamma ($\gamma$) - Focusing Parameter
| $\gamma$ Value | Effect | Use Case |
| :--- | :--- | :--- |
| 0 | Standard CE | Baseline, balanced datasets |
| 1.0 - 1.5 | Mild focus | Moderate imbalance (10-class) |
| 2.0 - 2.5 | Standard | Most common (100-class) |
| 3.0+ | Extreme | Severely imbalanced/noisy datasets |

#### Alpha ($\alpha$) - Class Balancing
Automatically calculated based on dataset complexity:
```python
def calculate_alpha(nb_classes):
    base_alpha = 0.25  # Standard for 10 classes
    scaling = (nb_classes - 10) / 200
    return min(0.75, base_alpha + scaling)
```

| Classes | Alpha Value | Rationale |
| :--- | :--- | :--- |
| 10 | 0.25 | Standard Focal Loss default |
| 100 | 0.70 | Strong balancing for complexity |

#### Dynamic Alpha Scaling
Enabled via `DYNAMIC_ALPHA_SCALING = True` in `parameters.py`. Monitors per-class validation accuracy and adjusts $\alpha$ to focus on underperforming classes.

**Configuration:**
```python
DYNAMIC_ALPHA_SCALING = True
ALPHA_ADJUSTMENT_FACTOR = 0.5  # Controls sensitivity (0.3-0.8)
```

### Implementation Examples
```bash
# Basic usage with Intelligent Focal Loss
python train.py --model digit_recognizer_v4 --focal-loss

# With custom gamma
python train.py --model digit_recognizer_v16 --focal-loss --focal-gamma 2.5

# Disable dynamic alpha
python train.py --model digit_recognizer_v3 --focal-loss --no-dynamic-alpha
```

---

## 2. Dynamic Per-Class Weighting

### Theory
Continuous adjustment of class weights based on current model performance, rather than static inverse frequencies. This adapts to the model's learning progress and focuses training on currently difficult digits.

### Algorithm
The dynamic weight for class $c$ at epoch $t$:
$w_c(t) = w_c(0) \cdot (1 + \beta \cdot (1 - acc_c(t)))$

Where:
- $w_c(0)$: Initial weight (inverse class frequency)
- $acc_c(t)$: Current validation accuracy for class $c$
- $\beta$: Adjustment factor (default: 2.0)

**Example Weight Evolution:**
```text
Initial: Class 0: 0.10  Class 1: 0.10  Class 5: 0.10  Class 9: 0.10

After 20 epochs:
Class 0 (98% acc): w = 0.10 × (1 + 2.0 × 0.02) = 0.104
Class 5 (85% acc): w = 0.10 × (1 + 2.0 × 0.15) = 0.130
Class 9 (82% acc): w = 0.10 × (1 + 2.0 × 0.18) = 0.136
```

### Hyperparameter Tuning
#### $\beta$ (Beta) - Adjustment Strength
| $\beta$ Value | Behavior | Recommended Use |
| :--- | :--- | :--- |
| 0 | Static weights only | Simple datasets, no hard classes |
| 1.0 - 1.5 | Mild adaptation | Balanced datasets |
| 2.0 - 3.0 | Moderate adaptation | Most digit recognition tasks (default) |
| 3.5 - 5.0 | Aggressive adaptation | Highly imbalanced, very hard classes |

#### Update Frequency
```python
WEIGHT_UPDATE_FREQUENCY = 5  # Update weights every 5 epochs
```
| Dataset Size | Recommended Frequency |
| :--- | :--- |
| < 10,000 images | Every 1-2 epochs |
| 10,000 - 100,000 | Every 3-5 epochs |
| > 100,000 images | Every 10 epochs |

#### Weight Bounds
```python
WEIGHT_MIN = 0.01  # Minimum weight for any class
WEIGHT_MAX = 0.50  # Maximum weight for any class
```

### Monitoring
Use `diagnose_training.py --track-weights` to generate:
- Weight evolution plots over epochs
- Per-class accuracy heatmaps
- Weight distribution histograms

```bash
python diagnose_training.py --model digit_recognizer_v4 --track-weights
```

---

## 3. Quantization-Aware Training (QAT)

### Theory
QAT simulates quantization noise during training, allowing the model to adapt weights to better handle the precision loss in 8-bit integer formats. This bridges the accuracy gap between floating-point training and integer-only inference.

| Method | Size Reduction | Accuracy Impact | Training Required |
| :--- | :--- | :--- | :--- |
| FP32 | 0% | Baseline | Full training |
| Post-training Quantization | 75% | 2-5% loss | None |
| **QAT** | 75% | 0.5-2% loss | Fine-tuning (10-20 epochs) |

### QAT Workflow
```text
1. Train high-accuracy FP32 model (≥95% accuracy)
   ↓
2. Insert FakeQuant layers
   ↓
3. Fine-tune for 10-15 epochs with reduced LR (1/10 of initial)
   ↓
4. Convert to INT8 format
   ↓
5. Deploy to edge device
```

### Hyperparameter Tuning
#### Quantization Bits
| Bits | Use Case | Memory Savings |
| :--- | :--- | :--- |
| **8-bit** | ESP32, Cortex-M, most edge devices | 75% |
| 16-bit | FPGA, some accelerators | 50% |

```python
QUANTIZATION_BITS = 8
```

#### Quantization Scheme
```python
QUANTIZATION_SCHEME = "esp_dl"  # Options: "tf_lite", "esp_dl"
```
| Scheme | Target Hardware | Operator Support |
| :--- | :--- | :--- |
| **tf_lite** | TensorFlow Lite, generic edge | Broad support |
| **esp_dl** | ESP32, ESP-DL library | Optimized for ESP32 |

#### Fine-tuning Configuration
```python
QAT_FINE_TUNING_EPOCHS = 10      # Number of fine-tuning epochs
QAT_LEARNING_RATE = 1e-5         # 1/10 of initial training LR
QAT_WARMUP_EPOCHS = 2            # Gradually introduce quantization
```

### Hardware-Specific Considerations
| Hardware | Recommendation |
| :--- | :--- |
| **ESP32 / ESP32-S3** | Use `QUANTIZATION_SCHEME = "esp_dl"`, grayscale inputs, model < 200KB |
| **Raspberry Pi / Linux** | Use `QUANTIZATION_SCHEME = "tf_lite"`, RGB acceptable |
| **Cortex-M4/M7** | 8-bit quantization, grayscale, model < 150KB |

---

## 4. Combined Strategy: Best Practices

### Scenario-Specific Configurations
#### Scenario 1: 10-Class Digits (Simple)
```yaml
focal_gamma: 1.5
dynamic_alpha_scaling: true
weight_update_frequency: 5
quantization: false
color_mode: "gray"
```
*Expected: 98.5-99.0% accuracy, 60-80KB model size*

#### Scenario 2: 100-Class Meter Digits (Complex)
```yaml
focal_gamma: 2.5
dynamic_alpha_scaling: true
alpha_adjustment_factor: 0.7
weight_update_frequency: 3
quantization: true
quantization_scheme: "esp_dl"
qat_fine_tuning_epochs: 15
color_mode: "gray"
```
*Expected: 88-94% accuracy, 140KB model size after quantization*

#### Scenario 3: Ultra-Lightweight (TinyML)
```yaml
focal_gamma: 1.0
dynamic_alpha_scaling: false  # Use static weights
weight_update_frequency: 0    # Disable dynamic updates
quantization: true
quantization_bits: 8
color_mode: "gray"
```
*Expected: 96-97% accuracy, < 50KB model size*

### Hyperparameter Tuning Workflow
```text
Step 1: Start with defaults
├── γ = 2.0
├── β = 2.0
├── Weight update = 5 epochs
└── Train for 30 epochs

Step 2: Monitor per-class performance
└── Use diagnose_training.py --track-weights

Step 3: Identify struggling classes
└── Check accuracy heatmap and weight evolution

Step 4: Adjust based on observations
├── If plateau early → Increase γ by 0.5
├── If rare classes underperform → Increase β to 3.0
├── If oscillation → Decrease weight update frequency
└── If overfitting → Reduce γ, increase regularization

Step 5: Enable QAT for edge deployment
└── Fine-tune with quantization for 10-15 epochs

Step 6: Validate on target hardware
└── Run bench_predict.py --target <device>
```

---

## 5. Troubleshooting

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| **Accuracy drop > 5% with Focal Loss** | $\gamma$ too high | Reduce $\gamma$ to 1.0-1.5 |
| **Training unstable after enabling Focal Loss** | Sudden transition | Ensure gradual transition is enabled |
| **Weights oscillating wildly** | Update frequency too high | Increase `WEIGHT_UPDATE_FREQUENCY` to 10 |
| **Weights become extreme (>0.5)** | $\beta$ too high | Reduce $\beta$ to 1.0-2.0 |
| **Poor performance on ESP32 after quantization** | Scheme mismatch | Set `QUANTIZATION_SCHEME = "esp_dl"` |
| **Model fails to load on hardware** | Unsupported operators | Run `run_quant_analysis.py` to verify |
| **QAT not recovering accuracy** | Poor FP32 base model | Ensure FP32 model >95% accuracy before QAT |
| **Slow inference** | RGB mode on constrained device | Switch to grayscale mode |

---

## 6. References
1. **Focal Loss for Dense Object Detection**
   Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017)
   Proceedings of the IEEE International Conference on Computer Vision (ICCV)
   arXiv:1708.02002
2. **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**
   Jacob, B., et al. (2018)
   Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
   arXiv:1712.05877
3. **TensorFlow Model Optimization Toolkit**
   Google AI [Documentation]
4. **ESP-DL: Deep Learning Library for ESP32**
   Espressif Systems [GitHub Repository]

---

## Appendix: Example Configurations

### Production High-Accuracy (100 Classes)
```python
# parameters.py
LOSS_TYPE = "IntelligentFocalLossController"
FOCAL_GAMMA = 2.5
DYNAMIC_ALPHA_SCALING = True
ALPHA_ADJUSTMENT_FACTOR = 0.7
WEIGHT_UPDATE_FREQUENCY = 3
WEIGHT_MIN = 0.01
WEIGHT_MAX = 0.50

# Quantization settings
QUANTIZATION_BITS = 8
QUANTIZATION_SCHEME = "esp_dl"
QAT_FINE_TUNING_EPOCHS = 15
QAT_LEARNING_RATE = 5e-6
QAT_WARMUP_EPOCHS = 3

# Training settings
LEARNING_RATE = 0.0005
EPOCHS = 60
BATCH_SIZE = 32
```

### Balanced Configuration (10 Classes)
```python
# parameters.py
LOSS_TYPE = "IntelligentFocalLossController"
FOCAL_GAMMA = 1.5
DYNAMIC_ALPHA_SCALING = True
WEIGHT_UPDATE_FREQUENCY = 5
QUANTIZATION_BITS = 8
QUANTIZATION_SCHEME = "tf_lite"
LEARNING_RATE = 0.001
EPOCHS = 40
```

### Ultra-Light Configuration (TinyML)
```python
# parameters.py
LOSS_TYPE = "IntelligentFocalLossController"
FOCAL_GAMMA = 1.0
DYNAMIC_ALPHA_SCALING = False
WEIGHT_UPDATE_FREQUENCY = 0
QUANTIZATION_BITS = 8
QUANTIZATION_SCHEME = "esp_dl"
MANUAL_INPUT_CHANNELS = 1  # Force grayscale
LEARNING_RATE = 0.001
EPOCHS = 30
```

### Command Reference
```bash
# Train with Intelligent Focal Loss
python train.py --model digit_recognizer_v16 --focal-loss --classes 100

# Train with custom gamma and QAT
python train.py --model digit_recognizer_v16 --focal-loss --focal-gamma 2.5 --quantize

# Run quantization analysis
python run_quant_analysis.py --model digit_recognizer_v16

# Debug with weight tracking
python diagnose_training.py --model digit_recognizer_v4 --track-weights

# Benchmark quantized model
python bench_predict.py --model digit_recognizer_v16 --quantized

# Generate per-class analysis
python diagnose_training.py --model digit_recognizer_v4 --per-class-analysis
```
