# diagnose_training.py
import tensorflow as tf
import numpy as np
import os
import cv2
from utils.multi_source_loader import load_combined_dataset
import parameters as params

def check_dataset_structure():
    """Check the actual dataset structure and class distribution"""
    print("=== DATASET STRUCTURE ANALYSIS ===")
    
    # Check the main dataset path
    dataset_path = params.DATA_SOURCES[0]['path']
    print(f"Checking dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Check each digit folder
    class_counts = {}
    total_images = 0
    
    for digit in range(10):
        digit_folder = os.path.join(dataset_path, str(digit))
        if os.path.exists(digit_folder):
            images = [f for f in os.listdir(digit_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            class_counts[digit] = count
            total_images += count
            print(f"Digit {digit}: {count} images")
        else:
            class_counts[digit] = 0
            print(f"Digit {digit}: FOLDER MISSING")
    
    print(f"\nTotal images: {total_images}")
    
    # Check for imbalance
    if total_images > 0:
        avg_per_class = total_images / 10
        print(f"Average per class: {avg_per_class:.1f}")
        
        # Check if only digit 1 has images
        if class_counts[1] > 0 and all(class_counts.get(d, 0) == 0 for d in range(10) if d != 1):
            print("🚨 CRITICAL ISSUE: Only digit '1' has images!")
            print("This is why the model always predicts '1'!")
            return False
        elif class_counts[1] / avg_per_class > 3.0:
            print("⚠️  WARNING: Severe class imbalance - digit '1' is overrepresented")
        else:
            print("✅ Dataset appears balanced")
    
    return True

def check_actual_data_loading():
    """Check what data actually gets loaded"""
    print("\n=== DATA LOADING ANALYSIS ===")
    
    try:
        images, labels = load_combined_dataset()
        
        print(f"Loaded {len(images)} total images")
        print(f"Loaded {len(labels)} total labels")
        
        # Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\nLabel distribution in loaded data:")
        for label, count in zip(unique_labels, counts):
            print(f"  Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        # Check if only label 1 exists
        if len(unique_labels) == 1 and unique_labels[0] == 1:
            print("🚨 CRITICAL: Only label '1' exists in loaded data!")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def check_preprocessing():
    """Check if preprocessing works correctly"""
    print("\n=== PREPROCESSING CHECK ===")
    
    from utils.preprocess import preprocess_single_image
    
    # Create test images for different digits
    for digit in [0, 1, 2]:
        # Create a simple test image
        test_image = np.zeros((params.INPUT_HEIGHT, params.INPUT_WIDTH), dtype=np.uint8)
        
        # Add some pattern that resembles the digit
        center_x, center_y = params.INPUT_WIDTH // 2, params.INPUT_HEIGHT // 2
        
        if digit == 0:
            cv2.circle(test_image, (center_x, center_y), 8, 255, -1)
        elif digit == 1:
            cv2.line(test_image, (center_x, 5), (center_x, params.INPUT_HEIGHT-5), 255, 3)
        elif digit == 2:
            # Simple curve for 2
            cv2.ellipse(test_image, (center_x, center_y), (8, 12), 0, 0, 180, 255, 3)
        
        processed = preprocess_single_image(test_image)
        print(f"Digit {digit}: original range [{test_image.min()}, {test_image.max()}], "
              f"processed range [{processed.min():.3f}, {processed.max():.3f}]")

def check_model_predictions_on_real_data():
    """Test the model on actual dataset images"""
    print("\n=== MODEL PREDICTION TEST ON REAL DATA ===")
    
    # Find the latest model
    training_dirs = [d for d in os.listdir(params.OUTPUT_DIR) if d.startswith('training_')]
    if not training_dirs:
        print("No trained models found")
        return
    
    latest_training = sorted(training_dirs)[-1]
    model_path = os.path.join(params.OUTPUT_DIR, latest_training, params.TFLITE_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test on a few real images from each class
    dataset_path = params.DATA_SOURCES[0]['path']
    
    for digit in range(3):  # Test first 3 digits
        digit_folder = os.path.join(dataset_path, str(digit))
        if not os.path.exists(digit_folder):
            continue
            
        images = [f for f in os.listdir(digit_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if images:
            # Test first image of this digit
            test_image_path = os.path.join(digit_folder, images[0])
            test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
            
            if test_image is not None:
                from utils.preprocess import preprocess_single_image
                processed = preprocess_single_image(test_image)
                
                # Prepare input
                input_data = np.expand_dims(processed, axis=0).astype(np.uint8)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                
                # Handle quantization if needed
                if output_details[0]['dtype'] in [np.uint8, np.int8]:
                    output_scale, output_zero_point = output_details[0]['quantization']
                    output = (output.astype(np.float32) - output_zero_point) * output_scale
                
                prediction = np.argmax(output[0])
                confidence = np.max(output[0])
                
                print(f"True digit: {digit}, Predicted: {prediction}, Confidence: {confidence:.4f}")
                print(f"  All outputs: {[f'{x:.4f}' for x in output[0]]}")

def main():
    """Main diagnosis function"""
    print("🔍 DIAGNOSING TRAINING ISSUES")
    print("=" * 60)
    
    # 1. Check dataset structure
    dataset_ok = check_dataset_structure()
    
    # 2. Check data loading
    if dataset_ok:
        loading_ok = check_actual_data_loading()
    
    # 3. Check preprocessing
    check_preprocessing()
    
    # 4. Test model on real data
    check_model_predictions_on_real_data()
    
    print("\n=== RECOMMENDED ACTIONS ===")
    if not dataset_ok:
        print("1. 🚨 FIX YOUR DATASET FIRST!")
        print("   - Ensure you have folders 0, 1, 2, ..., 9 in datasets/meterdigits/")
        print("   - Each folder should contain images of that digit")
        print("   - Delete the current trained model in exported_models/")
        print("   - Retrain after fixing the dataset")
    else:
        print("1. ✅ Dataset structure looks good")
        print("2. 🔄 Delete current trained models and retrain")
        print("3. 📊 Monitor training with: python train.py --debug")

if __name__ == "__main__":
    main()