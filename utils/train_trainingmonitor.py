# utils/train_trainingmonitor.py
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 

import parameters as params


class TrainingMonitor:
    """Collects loss/accuracy/LR during training and saves a plot."""

    def __init__(self, output_dir, debug=False):
        self.output_dir = output_dir
        self.debug = debug
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.lr_history = []
        self.model = None  # Initialize model as None

    # -----------------------------------------------------------------
    #  Called by Keras after each epoch
    # -----------------------------------------------------------------
    def on_epoch_end(self, epoch, logs):
        self.train_loss.append(logs.get("loss", 0.0))
        self.val_loss.append(logs.get("val_loss", 0.0))
        self.train_acc.append(logs.get("accuracy", 0.0))
        self.val_acc.append(logs.get("val_accuracy", 0.0))

        # Learning‑rate extraction with robust error handling
        lr = 0.0
        try:
            if self.model and hasattr(self.model, 'optimizer') and self.model.optimizer:
                # Try different ways to get the learning rate
                if hasattr(self.model.optimizer, 'learning_rate'):
                    lr_value = self.model.optimizer.learning_rate
                    if hasattr(lr_value, 'numpy'):  # For tf.Variable
                        lr = float(lr_value.numpy())
                    else:
                        lr = float(lr_value)
                elif hasattr(self.model.optimizer, 'lr'):
                    lr_value = self.model.optimizer.lr
                    if hasattr(lr_value, 'numpy'):
                        lr = float(lr_value.numpy())
                    else:
                        lr = float(lr_value)
                else:
                    # Fallback: try to get from Keras backend
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except (AttributeError, TypeError, ValueError) as e:
            if self.debug:
                print(f"⚠️  Could not get learning rate: {e}")
            lr = params.LEARNING_RATE  # Fallback to default learning rate
        
        self.lr_history.append(lr)

    # -----------------------------------------------------------------
    #  Keras gives the model to the callback via ``set_model``
    # -----------------------------------------------------------------
    def set_model(self, model):
        self.model = model

    # -----------------------------------------------------------------
    #  Save a three‑panel plot (loss, accuracy, LR)
    # -----------------------------------------------------------------
    def save_training_plots(self):
        if not params.SAVE_TRAINING_PLOTS:
            return

        # Only save if we have data
        if len(self.train_loss) == 0:
            if self.debug:
                print("⚠️  No training data to plot")
            return

        epochs = range(1, len(self.train_loss) + 1)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Loss
        ax1.plot(epochs, self.train_loss, "b-", label="Training loss", alpha=0.7)
        ax1.plot(epochs, self.val_loss, "r-", label="Validation loss", alpha=0.7)
        ax1.set_title("Training & Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Accuracy
        ax2.plot(epochs, self.train_acc, "b-", label="Training acc", alpha=0.7)
        ax2.plot(epochs, self.val_acc, "r-", label="Validation acc", alpha=0.7)
        ax2.set_title("Training & Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Learning Rate (only if we have LR data)
        if any(lr > 0 for lr in self.lr_history):
            ax3.plot(epochs, self.lr_history, "g-", label="Learning rate", alpha=0.7)
            ax3.set_title("Learning Rate Schedule")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning rate")
            # Use log scale only if LR values vary significantly
            if max(self.lr_history) / min(self.lr_history) > 10:
                ax3.set_yscale("log")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No LR data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes)
            ax3.set_title("Learning Rate Schedule")

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training plots saved to: {plot_path}")