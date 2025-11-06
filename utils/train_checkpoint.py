# utils/train_checkpoint.py
"""
Keras callback that handles:

* Saving the *best* TFLite model (using ``TFLiteModelManager.save_best_model``)
  whenever the validation accuracy improves.
* Periodically storing a trainable ``.keras`` checkpoint (default every 10 epochs).
"""

import os
import tensorflow as tf


class TFLiteCheckpoint(tf.keras.callbacks.Callback):
    """
    Parameters
    ----------
    tflite_manager : TFLiteModelManager
        Instance responsible for converting and persisting TFLite models.
    representative_data : callable or None
        Calibration generator for PTQ (passed through to ``save_best_model``).
    save_frequency : int, default 10
        How often (in epochs) to write a regular Keras checkpoint.
    """

    def __init__(self, tflite_manager, representative_data, save_frequency: int = 10):
        super().__init__()
        self.tflite_manager = tflite_manager
        self.representative_data = representative_data
        self.save_frequency = save_frequency
        self.last_save_epoch = -1

    # -----------------------------------------------------------------
    #  Called at the end of every epoch
    # -----------------------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy", 0.0)

        # -------------------------------------------------------------
        #  Save the best ever TFLite model
        # -------------------------------------------------------------
        if val_acc > getattr(self.tflite_manager, "best_accuracy", 0.0):
            try:
                self.tflite_manager.save_best_model(
                    self.model, val_acc, self.representative_data
                )
            except Exception as exc:  # pragma: no cover
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
            except Exception as exc:  # pragma: no cover
                if self.tflite_manager.debug:
                    print(f"⚠️  Checkpoint save failed: {exc}")