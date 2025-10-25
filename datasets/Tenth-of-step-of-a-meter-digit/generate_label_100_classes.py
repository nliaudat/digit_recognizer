import os

# Function to extract the class label from the filename
def extract_label_from_filename(filename):
    # Split the filename by underscores and take the first part
    rotation_part = filename.split("_")[0]
    
    # Handle cases where the rotation value is a float (e.g., "0.0", "7.2")
    if "." in rotation_part:
        # Convert to float and then multiply by 10 to get integer class (0-99)
        rotation_value = float(rotation_part)
        class_label = int(rotation_value * 10)
    else:
        # If it's already an integer, use it directly
        class_label = int(rotation_part)
    
    # Ensure the class label is within the valid range (0-99)
    if class_label < 0 or class_label > 99:
        raise ValueError(f"Class label {class_label} is out of range (0 to 99)")
    
    return class_label

# Function to generate labels.txt for a folder of images
def generate_labels_file(image_folder, output_file="labels_100.txt"):
    # Open the output file for writing
    with open(output_file, "w", encoding="utf-8") as labels_file:
        # Iterate over all files in the folder
        for filename in os.listdir(image_folder):
            if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
                # Extract the class label from the filename
                try:
                    label = extract_label_from_filename(filename)
                except (ValueError, IndexError) as e:
                    print(f"Skipping invalid filename: {filename} ({e})")
                    continue

                # Write the filename and label to the labels file
                # labels_file.write(f"{filename} {label}\n")
                labels_file.write(f"{filename}\t{label}\n") # use tab since many files has space

    print(f"Labels file generated: {output_file}")

# Main function
if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Generate a labels.txt file for a folder of images with 100 classes.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--output", type=str, default="labels_100.txt", help="Output file name for labels. Default: 'labels.txt'.")
    args = parser.parse_args()

    # Generate the labels file
    generate_labels_file(args.folder, args.output)