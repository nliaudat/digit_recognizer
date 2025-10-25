import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def flatten_all_images(source_dir, target_dir):
    """
    Flatten all images from all subfolders into the target directory
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Check if source directory exists
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # Find all image files first
    print("Scanning for image files...")
    image_files = []
    for file_path in source_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} image files")
    print("Copying files...")
    
    # Statistics
    total_files = len(image_files)
    duplicated_files = 0
    copied_files = 0
    
    # Track used filenames for duplicate detection
    used_filenames = set()
    
    # Process files with progress bar
    with tqdm(total=total_files, desc="Flattening", unit="file") as pbar:
        for image_file in image_files:
            original_filename = image_file.name
            target_file = target_path / original_filename
            
            # Check for duplicates
            if original_filename in used_filenames:
                duplicated_files += 1
                # Handle duplicate filenames by adding suffix
                counter = 1
                while target_file.exists() or target_file.name in used_filenames:
                    stem = Path(original_filename).stem
                    suffix = Path(original_filename).suffix
                    new_filename = f"{stem}_{counter}{suffix}"
                    target_file = target_path / new_filename
                    counter += 1
            else:
                # Handle case where file exists but not in our tracking yet
                counter = 1
                original_target = target_file
                while target_file.exists():
                    duplicated_files += 1
                    stem = original_target.stem
                    suffix = original_target.suffix
                    new_filename = f"{stem}_{counter}{suffix}"
                    target_file = target_path / new_filename
                    counter += 1
            
            # Copy the file
            shutil.copy2(image_file, target_file)
            used_filenames.add(target_file.name)
            copied_files += 1
            pbar.update(1)
    
    # Print statistics
    print("\n" + "=" * 50)
    print("FLATTENING COMPLETED!")
    print("=" * 50)
    print(f"Total files found:    {total_files}")
    print(f"Files copied:         {copied_files}")
    print(f"Duplicates handled:   {duplicated_files}")
    print(f"Target directory:     {target_dir}")
    print("=" * 50)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Image Flattening Script - Copy all images from subfolders to a single folder")
    parser.add_argument("--source_dir", type=str, default="./haverland_original",
                       help="Source directory containing images in subfolders")
    parser.add_argument("--target_dir", type=str, default="./flattened_images",
                       help="Target directory for all flattened images")
    
    args = parser.parse_args()
    
    print("Image Flattening Script")
    print("=" * 40)
    print(f"Source directory: {args.source_dir}")
    print(f"Target directory: {args.target_dir}")
    print()
    print("This will copy all images from all subfolders into a single folder.")
    print("Duplicate filenames will be automatically renamed.")
    print()
    
    # Confirm with user
    response = input("Do you want to proceed? (y/n): ").lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    try:
        flatten_all_images(args.source_dir, args.target_dir)
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()