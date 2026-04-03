import os
import sys

# Ensure project root is in sys.path when running from within utils/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import json
import argparse

from models.model_factory import create_model, compile_model
from utils.preprocess import preprocess_for_inference
from utils.keras_helper import keras


# ============================================================================
# CUSTOM LAYERS (Copied from models to avoid dependencies)
# ============================================================================

class AdaptiveHybridBinarization(keras.layers.Layer):
    """
    Hybrid approach: outputs both binary (for shape) AND soft gradient (for transition).
    Copied from models.digit_recognizer_v29 to avoid circular or missing dependencies.
    """
    def __init__(self, preserve_gradient=True, **kwargs):
        super().__init__(**kwargs)
        self.preserve_gradient = preserve_gradient

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
        binary_hard = tf.cast(inputs > mean, tf.float32)
        binary_ste  = inputs + tf.stop_gradient(binary_hard - inputs)
        
        if not self.preserve_gradient:
            return binary_ste
            
        std = tf.math.reduce_std(inputs, axis=[1, 2, 3], keepdims=True)
        soft_gradient = tf.sigmoid((inputs - mean) / (std + 1e-2))
        return tf.concat([binary_ste, soft_gradient], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"preserve_gradient": self.preserve_gradient})
        return config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes which regions of the image the model focuses on for prediction.
    """
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.grad_model = None
        
        self.keras = keras
        print(f"GradCAM using engine: {self.keras.__name__}")

        self.target_layer = self._find_target_layer(layer_name)
        self._build_gradient_model()
    
    def _find_target_layer(self, layer_name):
        """Find the last convolutional layer for Grad-CAM in an engine-agnostic way."""
        if layer_name:
            try:
                return self.model.get_layer(layer_name)
            except ValueError:
                print(f"Layer {layer_name} not found, searching in nested layers")
                for layer in self.model.layers:
                    if hasattr(layer, 'layer') and layer.layer.name == layer_name:
                        return layer
                    if layer.name == layer_name:
                        return layer
        
        # Default: last conv-like layer
        for layer in reversed(self.model.layers):
            # Engine-agnostic check via class name and properties
            cls_name = layer.__class__.__name__
            
            # Direct Conv2D or SeparableConv2D
            if "Conv2D" in cls_name:
                print(f"Using layer: {layer.name} ({cls_name})")
                return layer
                
            # Wrapped layer (QAT)
            if hasattr(layer, 'layer'):
                wrapped_cls_name = layer.layer.__class__.__name__
                if "Conv2D" in wrapped_cls_name:
                    print(f"Using layer: {layer.name} (wrapped {layer.layer.name} {wrapped_cls_name})")
                    return layer
        
        # Fallback: find by name if class name check failed
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                print(f"Using layer by name: {layer.name}")
                return layer
        
        raise ValueError("No convolutional layer found in model")
    
    def _build_gradient_model(self):
        """Create model that outputs feature maps and gradients."""
        self.grad_model = self.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.target_layer.output,
                self.model.output
            ]
        )
    
    def compute_heatmap(self, image, class_idx=None):
        """
        Compute Grad-CAM heatmap for given image.
        
        Args:
            image: Preprocessed input image (batch, height, width, channels)
            class_idx: Target class index (uses predicted class if None)
        
        Returns:
            heatmap: 2D numpy array of attention weights
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), axis=-1
        )
        
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (height, width, channels) in [0, 255]
            heatmap: Heatmap from compute_heatmap()
            alpha: Transparency factor
        
        Returns:
            overlayed_image: RGB image with heatmap overlay
        """
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed


class SaliencyMap:
    """Compute saliency maps to visualize pixel importance."""
    
    def __init__(self, model):
        self.model = model
    
    def compute(self, image, class_idx=None):
        """
        Compute saliency map using gradient of predicted class w.r.t input.
        
        Returns:
            saliency: 2D array of pixel importance
        """
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            predictions = self.model(image_tensor)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, image_tensor)[0]
        saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()
        
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        
        return saliency


def load_model_and_weights(model_arch, weights_path=None):
    """
    Load model architecture and weights.
    Extremely resilient version for Keras 2/3 and QAT.
    """
    import tensorflow as tf
    import parameters as params
    
    engines = [('keras', keras)]
        
    params.MODEL_ARCHITECTURE = model_arch
    params.update_derived_parameters()
    
    # If weights_path is a full .keras model, load it directly
    if weights_path and ('.keras' in weights_path.lower() or '.h5' in weights_path.lower()):
        print(f"Attempting to load full model from {weights_path}")
        
        # Prepare custom objects
        custom_objects = {}
        try:
            from utils.losses import DynamicSparseFocalLoss, sparse_focal_loss
            # Use the local class definition to avoid dependency on v29 model file
            custom_objects['AdaptiveHybridBinarization'] = AdaptiveHybridBinarization
            custom_objects['DynamicSparseFocalLoss'] = DynamicSparseFocalLoss
            custom_objects['sparse_focal_loss'] = sparse_focal_loss
            
            # QAT objects
            try:
                import tensorflow_model_optimization as tfmot
                custom_objects.update(tfmot.quantization.keras.get_quantize_objects())
            except:
                pass
        except ImportError:
            pass

        # Try each engine
        for name, engine in engines:
            try:
                print(f"Trying to load via {name}...")
                model = engine.models.load_model(weights_path, custom_objects=custom_objects, compile=False)
                print(f"✅ Successfully loaded full model via {name}")
                return model
            except Exception as e:
                print(f"⚠️ {name} load failed: {str(e)[:100]}...")
        
        print("Falling back to manual construction because load_model failed.")

    # Fallback: Factory creation
    print(f"Building model from factory: {model_arch}")
    model = create_model()
    
    # Apply QAT wrapper if weights suggest QAT
    if weights_path and 'QAT' in weights_path:
        print("Applying QAT wrapper to base model...")
        try:
            import tensorflow_model_optimization as tfmot
            if model_arch == 'digit_recognizer_v4':
                from models.digit_recognizer_v4 import create_qat_model
                model = create_qat_model(model)
            else:
                model = tfmot.quantization.keras.quantize_model(model)
            print("✅ Successfully applied QAT wrapper")
        except Exception as e:
            print(f"⚠️ Could not apply QAT wrapper: {e}")

    if weights_path and os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"✅ Loaded weights from {weights_path} via load_weights")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            print("Try checking if the architecture matches the weights.")
    else:
        print("No weights loaded, using initialized weights")
        
    return model


def get_binarization_overlap(model, image_input):
    """
    Get the intermediate binarized image from the model's preprocessing layers.
    Works for v28 (1ch) and v29 (2ch).
    """
    try:
        # Try to find a normalization layer (v28/v29)
        norm_layer_name = 'polarity_norm'
        if norm_layer_name in [l.name for l in model.layers]:
            intermediate_model = keras.Model(model.input, model.get_layer(norm_layer_name).output)
            return intermediate_model.predict(image_input, verbose=0)[0]
    except Exception as e:
        print(f"Could not extract binarization: {e}")
    return None


def find_best_weights(model_name):
    """
    Search for the best weights (.keras or .h5) for a given model architecture.
    Looks in exported_models/ and finds the latest training directory.
    """
    import parameters as params
    
    # We want to check both the class-specific output dir and the general one
    search_dirs = [params.OUTPUT_DIR, "exported_models"]
    
    found_weights = []
    
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
            
        # List all subdirectories
        try:
            subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))]
        except Exception:
            continue
        
        for subdir in subdirs:
            # Check if model_name is in the directory name
            if model_name in os.path.basename(subdir):
                # Look for best_model.keras or best_model.h5
                for f in ['best_model.keras', 'best_model.h5']:
                    f_path = os.path.join(subdir, f)
                    if os.path.exists(f_path):
                        # Use modification time to find the newest
                        mtime = os.path.getmtime(f_path)
                        found_weights.append((f_path, mtime))
                        
    if not found_weights:
        return None
        
    # Sort by modification time descending
    found_weights.sort(key=lambda x: x[1], reverse=True)
    return found_weights[0][0]


def visualize_attention(image_path, model, class_names=None, save_path=None):
    """
    Generate comprehensive attention visualization.
    """
    import parameters as params
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Use standard preprocessing pipeline
    img_preprocessed = preprocess_for_inference(original_img_rgb)
    
    # Keras Grad-CAM requires float32 [0, 1] regardless of quantization settings
    if img_preprocessed.dtype == np.uint8:
        img_preprocessed = img_preprocessed.astype(np.float32) / 255.0
    elif img_preprocessed.dtype == np.int8:
        # Map ESP-DL int8 [-128, 127] back to [0, 1]
        img_preprocessed = (img_preprocessed.astype(np.float32) + 128.0) / 255.0
    
    # Ensure batch dimension
    if len(img_preprocessed.shape) == 3:
        img_input = np.expand_dims(img_preprocessed, axis=0)
    else:
        img_input = img_preprocessed

    
    # Model prediction
    pred = model.predict(img_input, verbose=0)[0]
    pred_class = np.argmax(pred)
    confidence = pred[pred_class]
    
    # Grad-CAM heatmap
    gradcam = GradCAM(model)
    heatmap = gradcam.compute_heatmap(img_input, class_idx=pred_class)
    overlayed = gradcam.overlay_heatmap(original_img_rgb, heatmap)
    
    # Saliency Map
    saliency = SaliencyMap(model)
    saliency_map = saliency.compute(img_input, class_idx=pred_class)
    
    # Intermediate Binarization
    binarized = get_binarization_overlap(model, img_input)
    
    # Plotting
    cols = 3 if binarized is None else 4
    fig, axes = plt.subplots(2, cols, figsize=(2.5 * cols, 5))
    
    # Row 1: Original, Heatmap, Overlay, [Binarized CH0/CH1]
    axes[0, 0].imshow(original_img_rgb)
    axes[0, 0].set_title(f'Original\nPred: {pred_class} ({confidence:.2%})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(heatmap, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(overlayed)
    axes[0, 2].set_title('Grad-CAM Overlay')
    axes[0, 2].axis('off')
    
    if binarized is not None:
        ch0 = binarized[..., 0]
        axes[0, 3].imshow(ch0, cmap='gray')
        title = 'Binarized (v28/v29 CH0)'
        axes[0, 3].set_title(title)
        axes[0, 3].axis('off')

    # Row 2: Saliency, [Soft Gradient for v29], Top-K
    axes[1, 0].imshow(saliency_map, cmap='hot')
    axes[1, 0].set_title('Saliency Map')
    axes[1, 0].axis('off')
    
    if binarized is not None and binarized.shape[-1] == 2:
        ch1 = binarized[..., 1]
        axes[1, 1].imshow(ch1, cmap='gray')
        axes[1, 1].set_title('Soft Gradient (v29 CH1)')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off') # Spacer
    
    # Top-K Predictions
    top_k = 5
    top_indices = np.argsort(pred)[-top_k:][::-1]
    top_probs = pred[top_indices]
    top_labels = [class_names[i] if class_names else str(i) for i in top_indices]
    
    axes[1, cols-1].barh(range(top_k), top_probs, color='skyblue')
    axes[1, cols-1].set_yticks(range(top_k))
    axes[1, cols-1].set_yticklabels(top_labels)
    axes[1, cols-1].set_xlabel('Probability')
    axes[1, cols-1].set_title(f'Top {top_k} Predictions')
    axes[1, cols-1].set_xlim([0, 1])
    axes[1, cols-1].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'prediction': int(pred_class),
        'confidence': float(confidence),
        'top_predictions': list(zip(top_labels, top_probs.tolist()))
    }


def main():
    parser = argparse.ArgumentParser(description='Debug and visualize model attention')
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture to visualize')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to custom weights (optional)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--class-names', type=str, default=None,
                       help='JSON file with class names mapping')
    
    args = parser.parse_args()
    
    class_names = None
    if args.class_names:
        with open(args.class_names, 'r') as f:
            class_names = json.load(f)
    
    print(f"Loading model: {args.model}")
    
    weights_path = args.weights
    if not weights_path:
        print(f"No weights provided, searching for best weights for {args.model}...")
        weights_path = find_best_weights(args.model)
        if weights_path:
            print(f"Auto-discovered weights: {weights_path}")
        else:
            print("No weights found in exported_models/. Using initialized weights.")
            
    model = load_model_and_weights(args.model, weights_path)
    
    result = visualize_attention(
        args.image, model, class_names, args.save
    )
    
    if result:
        print(f"\nPrediction: {result['prediction']} ({result['confidence']:.2%})")
        print("\nTop predictions:")
        for label, prob in result['top_predictions']:
            print(f"  {label}: {prob:.2%}")


if __name__ == '__main__':
    main()


# python utils/gradcam.py --model digit_recognizer_v4 --image "datasets/sample.jpg"