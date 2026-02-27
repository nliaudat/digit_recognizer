# utils/dynamic_weighting.py
import tensorflow as tf
import numpy as np
import parameters as params

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, label_smoothing=0.05,
                 class_weights=None, nb_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.nb_classes = nb_classes
        if class_weights is not None:
            self.cw = tf.Variable(
                [class_weights[i] for i in range(self.nb_classes)],
                trainable=False, dtype=tf.float32)
        else:
            self.cw = None

    def update_weights(self, new_weights):
        """Update the dynamic class weights based on recent validation accuracy."""
        if self.cw is not None:
            self.cw.assign(new_weights)

    def call(self, y_true, y_pred):
        # y_true: one-hot (B, C)   y_pred: probs (B, C)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Handle labels of different shapes and convert to one-hot (batch_size, nb_classes)
        if len(y_true.shape) <= 1 or (len(y_true.shape) > 1 and y_true.shape[-1] != self.nb_classes):
            # If shape is (batch_size, 1), squeeze it to (batch_size,)
            if len(y_true.shape) == 2 and y_true.shape[-1] == 1:
                y_true = tf.squeeze(y_true, axis=-1)
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), self.nb_classes)
             
        y_true = tf.cast(y_true, tf.float32)

        # Label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + \
                     self.label_smoothing / self.nb_classes

        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        ce      = -y_true * tf.math.log(y_pred)
        pt      = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_w = tf.pow(1.0 - pt, self.gamma)
        loss    = focal_w * ce

        if self.cw is not None:
            loss = loss * self.cw  # per-class weight broadcast

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'gamma': self.gamma,
            'label_smoothing': self.label_smoothing,
            'nb_classes': self.nb_classes
        })
        return cfg


class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
    """Prints per-class accuracy on validation set every N epochs and updates dynamic loss weights."""

    def __init__(self, val_ds, loss_fn=None, every_n_epochs=5, nb_classes=10):
        super().__init__()
        self.val_ds      = val_ds
        self.loss_fn     = loss_fn
        self.every_n     = every_n_epochs
        self.nb_classes  = nb_classes

    def on_train_begin(self, logs=None):
        """Find the loss function if not explicitly provided."""
        if self.loss_fn is None:
            # Try to find FocalLoss in model's loss
            if hasattr(self.model, 'loss'):
                # Many Keras models store the loss function here
                maybe_loss = self.model.loss
                # Check for our FocalLoss which has a 'cw' attribute
                if hasattr(maybe_loss, 'cw'):
                    self.loss_fn = maybe_loss
                    print("🎯 PerClassAccuracyCallback: Linked to FocalLoss for dynamic weighting.")
                else:
                    print("⚠️ PerClassAccuracyCallback: Could not find FocalLoss in model. Dynamic weighting disabled.")
        elif hasattr(self, 'loss_fn') and self.loss_fn is not None:
             print("🎯 PerClassAccuracyCallback: Using explicitly provided FocalLoss.")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n != 0:
            return
        
        y_true_all, y_pred_all = [], []
        
        # Determine if val_ds is a Dataset or a tuple (x, y)
        if isinstance(self.val_ds, tf.data.Dataset):
            for imgs, labels in self.val_ds:
                preds = self.model(imgs, training=False)
                # Handle one-hot labels
                if len(labels.shape) > 1:
                    labels = tf.argmax(labels, axis=-1)
                y_true_all.append(labels.numpy())
                y_pred_all.append(tf.argmax(preds, axis=-1).numpy())
        else:
            # Assume tuple (x, y)
            x_val, y_val = self.val_ds
            # Process in batches to avoid OOM
            batch_size = 32
            for i in range(0, len(x_val), batch_size):
                imgs = x_val[i:i+batch_size]
                labels = y_val[i:i+batch_size]
                preds = self.model(imgs, training=False)
                # Handle one-hot labels
                if len(labels.shape) > 1:
                    labels = np.argmax(labels, axis=-1)
                y_true_all.append(labels)
                y_pred_all.append(np.argmax(preds, axis=-1))

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)

        per_class = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            mask = y_true == c
            if mask.sum() > 0:
                per_class[c] = (y_pred[mask] == c).mean()

        overall = (y_true == y_pred).mean()
        worst = np.argsort(per_class)[:10]
        print(f"\n[Epoch {epoch+1}] Overall val accuracy: {overall:.4f}")
        print(f"  Hardest 10 classes (lowest accuracy):")
        for c in worst:
            print(f"    class {c:3d}: {per_class[c]:.3f}")

        # --- Dynamic Weight Update ---
        if self.loss_fn is not None and hasattr(self.loss_fn, 'cw') and self.loss_fn.cw is not None:
            # 1. Calculate ideal new weights (inverse accuracy)
            # Add epsilon to prevent division by zero or huge spikes if acc is 0
            acc_safe = np.maximum(per_class, 0.01)
            raw_new_weights = 1.0 / acc_safe
            
            # 2. Normalize so mean is 1.0
            raw_new_weights /= raw_new_weights.mean()
            
            # 3. Smooth update (momentum) with current weights
            current_cw = self.loss_fn.cw.numpy()
            smoothed_weights = 0.5 * current_cw + 0.5 * raw_new_weights
            
            # 4. Explicit extra boost for the 10 hardest classes
            for c in worst:
                smoothed_weights[c] *= 1.2
            
            # 5. Re-normalize to ensure mean is exactly 1.0
            smoothed_weights /= smoothed_weights.mean()
            
            # 6. Apply to Focal Loss
            self.loss_fn.update_weights(smoothed_weights)
            
            # 7. Print top boosted classes (showing 10) in a readable table
            boosts = smoothed_weights - current_cw
            top_boosted = np.argsort(boosts)[-10:][::-1]
            print("\n  🔄 Dynamic Focal Loss weights updated! Boosting classes with low accuracy:")
            print("  ┌───────┬──────────┬────────────┬────────────┬─────────┐")
            print("  │ Class │ Accuracy │ Old Weight │ New Weight │ Change  │")
            print("  ├───────┼──────────┼────────────┼────────────┼─────────┤")
            for c in top_boosted:
                acc_pc = per_class[c] * 100.0
                print(f"  │ {c:5d} │ {acc_pc:7.1f}% │ {current_cw[c]:10.2f} │ {smoothed_weights[c]:10.2f} │ +{boosts[c]:.2f}  │")
            print("  └───────┴──────────┴────────────┴────────────┴─────────┘")
