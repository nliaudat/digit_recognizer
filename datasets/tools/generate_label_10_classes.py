import os

# Define the mapping array
# Each index corresponds to a rotation value (0.0 to 9.9), and the value is the label
mapping_array = [
    #0  1  2  3  4  5  6  7  8  9
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1,  # 0.0 to 0.9
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2,  # 1.0 to 1.9
    2, 2, 2, 2, 2, 2, 2, 3, 3, 3,  # 2.0 to 2.9
    3, 3, 3, 3, 3, 3, 3, 4, 4, 4,  # 3.0 to 3.9
    4, 4, 4, 4, 4, 4, 4, 5, 5, 5,  # 4.0 to 4.9
    5, 5, 5, 5, 5, 5, 5, 6, 6, 6,  # 5.0 to 5.9
    6, 6, 6, 6, 6, 6, 6, 7, 7, 7,  # 6.0 to 6.9
    7, 7, 7, 7, 7, 7, 7, 8, 8, 8,  # 7.0 to 7.9
    8, 8, 8, 8, 8, 8, 8, 9, 9, 9,  # 8.0 to 8.9
    9, 9, 9, 9, 9, 9, 9, 0, 0, 0   # 9.0 to 9.9
]

# Function to map rotation value (0.0 to 9.9) to integer label (0 to 9)
def map_rotation_to_label(rotation):
    # Ensure the rotation value is within the valid range
    if rotation < 0.0 or rotation > 9.9:
        raise ValueError(f"Rotation value {rotation} is out of range (0.0 to 9.9)")

    # Convert rotation value (0.0 to 9.9) to an index (0 to 99)
    index = int(rotation * 10)
    
    # Clamp the index to ensure it stays within the bounds of the mapping_array
    index = max(0, min(index, 99))
    
    return mapping_array[index]

# Function to extract the rotation value from the filename
def extract_rotation(filename):
    # Split the filename by underscores and take the first part
    rotation_part = filename.split("_")[0]
    # Handle cases where the rotation value is an integer (e.g., "0") or a float (e.g., "0.0")
    if "." in rotation_part:
        return float(rotation_part)  # If it contains a decimal, treat it as a float
    else:
        return float(rotation_part)  # If it's an integer, convert it to float (e.g., "0" -> 0.0)

# Function to generate labels.txt for a folder of images
def generate_labels_file(image_folder, output_file="labels_10.txt"):
    # Open the output file for writing
    with open(output_file, "w", encoding="utf-8") as labels_file:
        # Iterate over all files in the folder
        for filename in os.listdir(image_folder):
            if filename.endswith(".jpeg") or filename.endswith(".jpg"):
                # Extract the rotation value from the filename
                try:
                    rotation = extract_rotation(filename)
                except ValueError as e:
                    print(f"Skipping invalid filename: {filename} ({e})")
                    continue

                # Map the rotation value to a label
                try:
                    label = map_rotation_to_label(rotation)
                except ValueError as e:
                    print(f"Skipping invalid rotation value in filename: {filename} ({e})")
                    continue

                # Write the filename and label to the labels file
                # labels_file.write(f"{filename} {label}\n")
                labels_file.write(f"{filename}\t{label}\n") # use tab since many files has space

    print(f"Labels file generated: {output_file}")

# Main function
if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Generate a labels.txt file for a folder of images.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--output", type=str, default="labels_10.txt", help="Output file name for labels. Default: 'labels.txt'.")
    args = parser.parse_args()

    # Generate the labels file
    generate_labels_file(args.folder, args.output)