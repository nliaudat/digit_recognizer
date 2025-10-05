import os
import shutil
import random
from sklearn.model_selection import train_test_split
import parameters as params

def create_folder_structure(source_dir, output_dir):
    """
    Convert flat image directory to folder structure
    Assumes filenames contain class information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class folders
    for i in range(params.NB_CLASSES):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
    
    # Move images to appropriate folders
    for filename in os.listdir(source_dir):
        if any(filename.lower().endswith(ext) for ext in params.IMAGE_EXTENSIONS):
            # Extract class from filename (adjust this logic as needed)
            # Example: "digit_0_001.jpg" -> class 0
            try:
                # This depends on your filename format - adjust accordingly
                if '0' in filename: class_label = 0
                elif '1' in filename: class_label = 1
                # ... add more rules based on your filenames
                else:
                    print(f"Could not determine class for {filename}, skipping")
                    continue
                
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(output_dir, str(class_label), filename)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {filename} to class {class_label}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def create_label_file(image_dir, output_file):
    """
    Create labels.txt from images in a directory
    """
    with open(output_file, 'w') as f:
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in params.IMAGE_EXTENSIONS):
                # Extract class from filename (adjust as needed)
                try:
                    if '0' in filename: label = 0
                    elif '1' in filename: label = 1
                    elif '2' in filename: label = 2
                    elif '3' in filename: label = 3
                    elif '4' in filename: label = 4
                    elif '5' in filename: label = 5
                    elif '6' in filename: label = 6
                    elif '7' in filename: label = 7
                    elif '8' in filename: label = 8
                    elif '9' in filename: label = 9
                    else:
                        print(f"Could not determine class for {filename}, skipping")
                        continue
                    
                    f.write(f"{filename} {label}\n")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Example usage:
    # create_folder_structure("raw_images", "dataset")
    # create_label_file("raw_images", "dataset/labels.txt")
    print("Dataset preparation utilities")