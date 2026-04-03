# utils/train_progressbar.py
"""
A lightweight tqdm based progress bar for Keras training.
Shows epoch number, loss/accuracy, validation loss/accuracy and an ETA.
"""

import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
import tensorflow as tf
from utils.keras_helper import keras

class TQDMProgressBar(keras.callbacks.Callback):
    """
    Parameters
    ----------
    total_epochs : int
        Number of epochs the training will run.
    monitor : TrainingMonitor
        Instance that records loss/accuracy/LR (used to update the bar).
    debug : bool, default False
        If True, the bar prints extra information.
    """

    def __init__(self, total_epochs: int, monitor, debug: bool = False):
        super().__init__()
        self.total_epochs = total_epochs
        self.monitor = monitor
        self.debug = debug
        self.epoch_times = []
        self.main_pbar = None
        self.epoch_pbar = None

    # -----------------------------------------------------------------
    #  Called once before training starts
    # -----------------------------------------------------------------
    def on_train_begin(self, logs=None):
        self.main_pbar = tqdm(
            total=self.total_epochs,
            desc="Total Progress",
            unit="epoch",
            bar_format="{l_bar}{bar:20}{r_bar}",
            position=0,
            leave=True,
        )

    # -----------------------------------------------------------------
    #  Called at the start of each epoch
    # -----------------------------------------------------------------
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = datetime.now()
        
        # Get steps per epoch from params
        steps = self.params.get('steps')
        if steps:
            self.epoch_pbar = tqdm(
                total=steps,
                desc=f"Epoch {epoch+1}/{self.total_epochs}",
                unit="batch",
                bar_format="{l_bar}{bar:20}{r_bar}",
                position=1,
                leave=False
            )

    # -----------------------------------------------------------------
    #  Called at the end of each batch
    # -----------------------------------------------------------------
    def on_train_batch_end(self, batch, logs=None):
        if self.epoch_pbar:
            # Update epoch bar with current batch metrics if available
            metrics = ""
            if logs:
                loss = logs.get('loss', 0.0)
                acc = logs.get('accuracy', 0.0)
                metrics = f" - loss: {loss:.4f} - acc: {acc:.4f}"
            
            self.epoch_pbar.set_postfix_str(metrics)
            self.epoch_pbar.update(1)

    # -----------------------------------------------------------------
    #  Called at the end of each epoch
    # -----------------------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = datetime.now() - self.epoch_start
        self.epoch_times.append(epoch_time)
        avg_time = np.mean(self.epoch_times) if self.epoch_times else epoch_time

        # Close epoch pbar
        if self.epoch_pbar:
            self.epoch_pbar.close()
            self.epoch_pbar = None

        # Forward the metrics to the TrainingMonitor (so it can store them)
        if self.monitor:
            self.monitor.on_epoch_end(epoch, logs)

        # Extract metrics for display
        logs = logs or {}
        train_loss = logs.get("loss", 0.0)
        train_acc = logs.get("accuracy", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)

        remaining = self.total_epochs - epoch - 1
        eta = avg_time * remaining

        # Update main bar with last epoch results
        status = (
            f"loss:{train_loss:.4f} acc:{train_acc:.4f} | "
            f"val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} | "
            f"ETA:{str(eta).split('.')[0]}"
        )
        self.main_pbar.set_postfix_str(status)
        self.main_pbar.update(1)

    # -----------------------------------------------------------------
    #  Called once after training finishes
    # -----------------------------------------------------------------
    def on_train_end(self, logs=None):
        """Close the tqdm progress bar and optionally print a final ETA."""
        if self.main_pbar:
            self.main_pbar.close()
            if self.debug:
                print("✅ Training progress bar closed.")
