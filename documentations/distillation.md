# Knowledge Distillation Pipeline

Knowledge distillation is a powerful technique for transferring the deep knowledge representations of large, complex models (teachers) into smaller, highly efficient edge models (students). This is essential for deploying highly accurate Digit Recognition models on constrained devices like the ESP32.

This repository features a comprehensive distillation pipeline that supports standard (single teacher) and ensemble (multi-teacher) knowledge distillation.

## Core Concepts

*   **Teacher Model:** A large, heavyweight model (e.g., EfficientNet, SE-ResNet) trained purely for maximum accuracy. It is rarely intended for deployment on ESP32 due to its size.
*   **Student Model:** An extreme-edge IoT architecture (e.g., v4, v15, v16) designed specifically to run quickly and fit within the memory limits of the ESP32.
*   **Distillation Process:** The student model learns by imitating the "soft targets" (predicted probability distributions) produced by the teacher model(s). Soft targets provide rich information about the relationships between different digits, allowing the student to train faster and reach a higher accuracy ceiling.

## Training Entry Point: `train_distill.py`

The script `train_distill.py` acts as the primary orchestrator for the distillation workflow.

### Phases
The pipeline operates in three distinct phases, controlled by the `--phase` argument:
1.  **`teacher`**: Train only the teacher model(s) from scratch and save them to disk.
2.  **`student`**: Perform the actual distillation. Assumes that the prepared teacher models exist, loads them, and trains the lightweight student.
3.  **`all`**: Runs the entire process sequentially — trains the teachers, then immediately distills the student.

### Key Arguments

*   `--phase`: `teacher`, `student`, or `all`
*   `--student`: Student architecture to train (e.g., `v30_medium`, `v15`).
*   `--teachers`: One or more teacher architectures to act as teachers (default: `v16`). E.g., `--teachers v30 v31`.
*   `--load-teachers`: Provide the explicit file paths to pre-trained teacher `.keras` checkpoints if you wish to skip teacher training.
*   `--teacher-weights`: If using multiple teachers, assign importance weights to each (e.g., `--teacher-weights 0.7 0.3`). By default, teachers are weighted equally.
*   `--temperature`: Softening parameter for the KL Divergence loss. Higher values (e.g., 4.0 - 8.0) make the probability distributions "softer", revealing more secondary class information.
*   `--alpha`: Balances the distillation loss against the standard cross-entropy loss against the true labels (0.0 means 100% teacher guidance; 1.0 means 100% hard label guidance).
*   `--classes`: Defines how many output classes to train against (10 or 100).
*   `--color`: Defines the color space (`rgb` or `gray`).

## Example Workflows

### 1. Standard (Single-Teacher) Distillation

Train a standard, large `v30` teacher and immediately distill its knowledge into a very small `v15` student:

```bash
python train_distill.py --phase all --teachers v30 --student digit_recognizer_v15 --temperature 4.0 --alpha 0.7 --classes 100
```

### 2. Loading a Pre-trained Teacher

If you have already spent hours training a great teacher model and saved its `.keras` file, you can utilize it instantly to train new students without retraining:

```bash
python train_distill.py --phase student \
    --teachers v30_teacher \
    --load-teachers checkpoints/best_teacher_100cls.keras \
    --student digit_recognizer_v16 \
    --classes 100 \
    --color rgb
```

### 3. Ensemble Distillation (Multi-Teacher)

Ensemble distillation allows the student model to learn from the aggregated wisdom of multiple different teacher architectures. This prevents the student from memorizing the specific biases of a single teacher architecture and frequently improves final generalization.

Our pipeline features an **`EnsembleTeacher`** that automatically freezes all contributing teachers and seamlessly combines their outputs. It supports computing weighted averages of either probabilities or logits to create a consolidated "soft target" for the student.

To run ensemble distillation, specify multiple teacher types. Optionally, provide custom weights indicating their relative importance:

```bash
python train_distill.py --phase student \
    --teachers v30 v31 \
    --load-teachers checkpoints/v30_teacher.keras checkpoints/v31_teacher.keras \
    --teacher-weights 0.6 0.4 \
    --student digit_recognizer_v4 \
    --temperature 5.0 \
    --alpha 0.6
```

In the background, the pipeline wraps these models inside the `EnsembleTeacher` class dynamically, ensuring smooth integration with Keras loss routines.

*Note: You can also utilize pre-built "super-teacher" models, such as the `v32` family (`v32_small`, `v32_medium`, `v32_large`, `v32_xl`), which themselves were trained via ensemble distillation.*

### 4. Retraining Existing Models with a Teacher

You can retroactively improve the accuracy of existing baseline edge models (e.g., `v4` or `v16`) by fine-tuning them under the supervision of a powerful teacher. This preserves the optimized edge architectures while mapping rich knowledge into them.

```bash
# Retrain v4 with a v30 teacher
python train_distill.py --retrain-existing --existing-model v4 \
    --teacher v30 --classes 10 --color gray --epochs 30

# Retrain v16 from an existing checkpoint with progressive distillation
python train_distill.py --retrain-existing --existing-model v16 \
    --load-model-checkpoint checkpoints/v16_best.keras --teacher v30 --progressive
```

### 5. Advanced Distillation Strategies

The pipeline automatically employs advanced distillation mechanisms when requested or needed:

*   **Mixed Input Distillation:** Allows distilling knowledge from an RGB teacher (3 channels) into a lightweight Grayscale student (1 channel). The `MixedInputDistiller` automatically transforms inputs so both models receive appropriate channel structures transparently.
*   **Progressive Distillation (`--progressive`):** Instead of fixed hyperparameters, progressively anneals the temperature (from 8.0 to 2.0) and transitions the `alpha` weighting from teacher-heavy (0.3) to more hard-labels (0.8) across the epochs.

## Global Parameters (`parameters.py`)

Several default values for testing and fallback distillation exist within `parameters.py`:

```python
# Core distillation hyperparameters
DISTILLATION_TEMPERATURE = 4.0    
DISTILLATION_ALPHA = 0.7          
DISTILLATION_MODE = "soft"        

# Teacher training defaults
DISTILLATION_TEACHER_EPOCHS = 60       
DISTILLATION_TEACHER_LR = 1e-3        
DISTILLATION_TEACHER_PRETRAINED = True 
DISTILLATION_FREEZE_BACKBONE = False   

# Export targets
DISTILLATION_TARGET_HARDWARE = "esp32"  
DISTILLATION_EXPORT_QUANTIZED = True   
```

## Reviewing Results

Upon completion, `train_distill.py` exports:
1.  **Saved Model Checkpoints**: Available in the `exported_models/` tracking directory.
2.  **.tflite File**: If exporting for `esp32`, the student will be immediately quantized.
3.  **JSON Summary**: Contains detailed metrics comparing the teacher's accuracy/size directly with the student's final accuracy/size.

You can then pass the output evaluation `.tflite` model into `bench_predict.py` perfectly seamlessly!
