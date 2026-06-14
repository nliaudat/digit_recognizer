# utils/train_checkpoint.py
"""
Keras callback that handles:

* Saving the *best* TFLite model (using ``TFLiteModelManager.save_best_model``)
  whenever the validation accuracy improves.
* Periodically storing a trainable ``.keras`` checkpoint (default every 10 epochs).
* Saving / loading full training state (optimizer weights, controller states)
  for perfect resume support.
"""

import json
import os
import numpy as np
import tensorflow as tf
from utils.train_qat_helper import create_qat_representative_dataset


class TFLiteCheckpoint(tf.keras.callbacks.Callback):
    """
    Parameters
    ----------
    tflite_manager : TFLiteModelManager
        Instance responsible for converting and persisting TFLite models.
    representative_data : callable or None
        Calibration generator for PTQ (passed through to ``save_best_model``).
    x_train_raw : np.ndarray or None
        Raw training data for creating representative dataset if needed.
    save_frequency : int, default 5 (set in parameters.py CHECKPOINT_FREQUENCY)
        How often (in epochs) to write a regular Keras checkpoint.
    """

    def __init__(self, tflite_manager, representative_data, x_train_raw=None, save_frequency: int = 5):
        super().__init__()
        self.tflite_manager = tflite_manager
        self.representative_data = representative_data
        self.x_train_raw = x_train_raw
        self.save_frequency = save_frequency
        self.last_save_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy", 0.0)

        # -------------------------------------------------------------
        #  Save the best ever TFLite model
        # -------------------------------------------------------------
        if val_acc > getattr(self.tflite_manager, "best_accuracy", 0.0):
            try:
                # If we have raw data but no representative data, create it
                representative_data_to_use = self.representative_data
                if representative_data_to_use is None and self.x_train_raw is not None:
                    # from utils.train_qat_helper import create_qat_representative_dataset
                    representative_data_to_use = create_qat_representative_dataset(self.x_train_raw)
                
                self.tflite_manager.save_best_model(
                    self.model, val_acc, representative_data_to_use
                )
            except Exception as exc:
                if self.tflite_manager.debug:
                    print(f"⚠️  TFLite save failed: {exc}")

        # -------------------------------------------------------------
        #  Periodic Keras checkpoint (trainable .keras file)
        # -------------------------------------------------------------
        if epoch % self.save_frequency == 0 and epoch != self.last_save_epoch:
            try:
                ckpt_path = self.tflite_manager.save_trainable_checkpoint(
                    self.model, val_acc, epoch
                )
                if ckpt_path and self.tflite_manager.debug:
                    print(f"💾 Saved checkpoint: {ckpt_path}")
                self.last_save_epoch = epoch
            except Exception as exc:
                if self.tflite_manager.debug:
                    print(f"⚠️  Checkpoint save failed: {exc}")


# --------------------------------------------------------------------------- #
#  Training State Save / Load  (used by StateCheckpointCallback)
# --------------------------------------------------------------------------- #

def save_training_state(filepath, controller_callbacks, model):
    """
    Save optimizer weights + controller callback states to disk.

    Parameters
    ----------
    filepath : str
        Path for the JSON file (e.g. ``training_state.json``).  The optimizer
        weights are saved alongside as ``<stem>_optimizer.npy``.
    controller_callbacks : list of tf.keras.callbacks.Callback
        Callbacks that implement ``get_state()`` (e.g. IntelligentFocalLossController,
        DynamicSchedulerController).
    model : tf.keras.Model
        The compiled model whose optimizer weights will be saved.
    """
    state = {'version': 1}

    # 1. Optimizer weights
    if model is not None and model.optimizer is not None:
        try:
            opt_weights = np.array([v.numpy() for v in model.optimizer.variables], dtype=object)
            opt_path = filepath.replace('.json', '_optimizer.npy')
            np.save(opt_path, opt_weights, allow_pickle=True)
            state['optimizer_weights_file'] = os.path.basename(opt_path)
        except Exception as exc:
            print(f"⚠️  Could not save optimizer weights: {exc}")
            state['optimizer_weights_file'] = None
    else:
        state['optimizer_weights_file'] = None

    # 2. Controller callback states
    states = {}
    for cb in controller_callbacks:
        if hasattr(cb, 'get_state') and callable(cb.get_state):
            name = type(cb).__name__
            try:
                states[name] = cb.get_state()
            except Exception as exc:
                print(f"⚠️  Could not save state for {name}: {exc}")
    state['controller_states'] = states

    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)


def load_training_state(filepath, controller_callbacks, model):
    """
    Restore optimizer weights + controller callback states from disk.

    Parameters
    ----------
    filepath : str
        Path to the JSON file previously written by :func:`save_training_state`.
    controller_callbacks : list of tf.keras.callbacks.Callback
        Callbacks that implement ``set_state()``.
    model : tf.keras.Model
        The compiled model whose optimizer weights will be restored.

    Returns
    -------
    bool
        True if state was successfully restored (or no file existed), False on error.
    """
    if not os.path.exists(filepath):
        print("   ⚠️  No training_state.json found — starting fresh")
        return True  # Not an error

    try:
        with open(filepath) as f:
            state = json.load(f)
    except Exception as exc:
        print(f"   ❌ Failed to read training_state.json: {exc}")
        return False

    # 1. Optimizer weights
    opt_file = state.get('optimizer_weights_file')
    if opt_file and model is not None and model.optimizer is not None:
        opt_dir = os.path.dirname(filepath)
        opt_path = os.path.join(opt_dir, opt_file)
        if os.path.exists(opt_path):
            try:
                saved_weights = np.load(opt_path, allow_pickle=True)
                # Force init of optimizer variables if lazily not yet created
                # (TF/Keras delays variable creation until first training step)
                if not model.optimizer.variables:
                    if hasattr(model.optimizer, 'build'):
                        model.optimizer.build(model.trainable_variables)
                    elif hasattr(model.optimizer, '_create_all_weights'):
                        model.optimizer._create_all_weights(model.trainable_variables)
                # Validate count matches before restoring
                if len(saved_weights) != len(model.optimizer.variables):
                    print(f"   ⚠️  Optimizer variable count mismatch: "
                          f"saved={len(saved_weights)}, current={len(model.optimizer.variables)} — skipping restore")
                else:
                    for var, w in zip(model.optimizer.variables, saved_weights):
                        var.assign(w)
                    print(f"   ✅ Optimizer weights restored from {opt_file}")
            except Exception as exc:
                print(f"   ⚠️  Could not restore optimizer weights: {exc}")
        else:
            print(f"   ⚠️  Optimizer weights file not found: {opt_file}")

    # 2. Controller states
    controller_states = state.get('controller_states', {})
    if not controller_states:
        print("   ⚠️  No controller states found in checkpoint")
        return True

    for cb in controller_callbacks:
        name = type(cb).__name__
        if name in controller_states and hasattr(cb, 'set_state') and callable(cb.set_state):
            try:
                # Attach model before restoring state so set_state() can sync
                # restored gamma/alpha values back to model.loss if needed.
                cb.model = model
                cb.set_state(controller_states[name])
                print(f"   ✅ {name} state restored")
            except Exception as exc:
                print(f"   ⚠️  Could not restore {name}: {exc}")

    return True