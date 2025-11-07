# utils/train_checkpoint.py
"""
Keras callback that handles:

* Saving the *best* TFLite model (using ``TFLiteModelManager.save_best_model``)
  whenever the validation accuracy improves.
* Periodically storing a trainable ``.keras`` checkpoint (default every 10 epochs).
"""

import os
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
    save_frequency : int, default 10
        How often (in epochs) to write a regular Keras checkpoint.
    """

    def __init__(self, tflite_manager, representative_data, x_train_raw=None, save_frequency: int = 10):
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