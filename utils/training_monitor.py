import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import parameters as params
from tqdm.auto import tqdm

class TrainingMonitor:
    def __init__(self, output_dir, debug=False):
        self.output_dir = output_dir
        self.debug = debug
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.lr_history = []
    def on_epoch_end(self, epoch, logs):
        self.train_loss.append(logs.get('loss', 0))
        self.val_loss.append(logs.get('val_loss', 0))
        self.train_acc.append(logs.get('accuracy', 0))
        self.val_acc.append(logs.get('val_accuracy', 0))
        try:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except AttributeError:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.lr_history.append(current_lr)
    def set_model(self, model):
        self.model = model
    def save_training_plots(self):
        if not params.SAVE_TRAINING_PLOTS:
            return
        epochs = range(1, len(self.train_loss) + 1)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.plot(epochs, self.train_loss, 'b-', label='Training Loss', alpha=0.7)
        ax1.plot(epochs, self.val_loss, 'r-', label='Validation Loss', alpha=0.7)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, self.train_acc, 'b-', label='Training Accuracy', alpha=0.7)
        ax2.plot(epochs, self.val_acc, 'r-', label='Validation Accuracy', alpha=0.7)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3.plot(epochs, self.lr_history, 'g-', label='Learning Rate', alpha=0.7)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training plots saved to: {plot_path}")

class TFLiteCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, tflite_manager, representative_data, save_frequency=10):
        super().__init__()
        self.tflite_manager = tflite_manager
        self.representative_data = representative_data
        self.save_frequency = save_frequency
        self.last_save_epoch = -1
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy', 0)
        if val_accuracy > getattr(self.tflite_manager, 'best_accuracy', 0):
            try:
                self.tflite_manager.save_best_model(
                    self.model, 
                    val_accuracy, 
                    self.representative_data
                )
            except Exception as e:
                if self.tflite_manager.debug:
                    print(f"⚠️  TFLite save failed: {e}")
        if epoch % self.save_frequency == 0 and epoch != self.last_save_epoch:
            try:
                checkpoint_path = self.tflite_manager.save_trainable_checkpoint(self.model, val_accuracy, epoch)
                if checkpoint_path and self.tflite_manager.debug:
                    print(f"💾 Saved checkpoint: {checkpoint_path}")
                self.last_save_epoch = epoch
            except Exception as e:
                if self.tflite_manager.debug:
                    print(f"⚠️  Checkpoint save failed: {e}")

class TQDMProgressBar(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, monitor, tflite_manager, debug=False):
        super().__init__()
        self.total_epochs = total_epochs
        self.monitor = monitor
        self.tflite_manager = tflite_manager
        self.debug = debug
        self.epoch_times = []
        self.pbar = None
    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.total_epochs, desc='Training', unit='epoch',
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                        position=0, leave=True)
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = datetime.now()
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = datetime.now() - self.epoch_start
        self.epoch_times.append(epoch_time)
        avg_time = np.mean(self.epoch_times) if self.epoch_times else epoch_time
        self.monitor.on_epoch_end(epoch, logs)
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        remaining_epochs = self.total_epochs - epoch - 1
        remaining_time = avg_time * remaining_epochs
        desc = (f"Epoch {epoch+1}/{self.total_epochs} | "
                f"loss: {train_loss:.4f} | acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
                f"ETA: {str(remaining_time).split('.')[0]}")
        self.pbar.set_description(desc)
        self.pbar.update(1)
    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()


