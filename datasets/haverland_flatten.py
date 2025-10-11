import os
import shutil
import argparse
from pathlib import Path

def flatten_haverland_folders(source_dir, target_dir=None, class100=False):
    """
    Flatten folders from 0.0 to 9.9 into folders 0 to 9 according to the rules:
    - x.0 to x.5 -> folder x
    - x.6 to x.9 -> folder x+1
    - SPECIAL CASE: 9.6 to 9.9 -> folder 0
    
    If class100 is True, simply copy folders while renaming by multiplying by 10 (1.1 -> 11)
    
    Args:
        source_dir (str): Source directory containing folders 0.0 to 9.9
        target_dir (str): Target directory (defaults to source_dir if None)
        class100 (bool): If True, use class100 mode (simple copy with *10 renaming)
    """
    if target_dir is None:
        target_dir = source_dir
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if class100:
        # CLASS100 MODE: Simple copy with *10 renaming
        print("Using class100 mode: copying folders with *10 renaming")
        
        # Process each source folder from 0.0 to 9.9
        for x in range(10):
            for y in range(10):
                source_folder_name = f"{x}.{y}"
                source_folder_path = source_path / source_folder_name
                
                if not source_folder_path.exists():
                    print(f"Warning: Source folder {source_folder_name} not found, skipping...")
                    continue
                
                # Calculate target folder number (x.y -> x*10 + y)
                target_folder_num = x * 10 + y
                target_folder_path = target_path / str(target_folder_num)
                target_folder_path.mkdir(parents=True, exist_ok=True)
                
                # Copy all images from source to target
                image_files = [f for f in source_folder_path.iterdir() 
                              if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']]
                
                files_copied = 0
                for image_file in image_files:
                    target_file = target_folder_path / image_file.name
                    
                    # Handle duplicate filenames by adding suffix
                    counter = 1
                    while target_file.exists():
                        stem = image_file.stem
                        suffix = image_file.suffix
                        target_file = target_folder_path / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.copy2(image_file, target_file)
                    files_copied += 1
                
                print(f"Processed {source_folder_name} -> {target_folder_num}/ ({files_copied} files)")
    
    else:
        # ORIGINAL MODE: Flattening according to the rules
        # Create target folders 0-9
        for i in range(10):
            folder_path = target_path / str(i)
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Created target folder: {folder_path}")
        
        # Process each source folder from 0.0 to 9.9
        for x in range(10):
            for y in range(10):
                source_folder_name = f"{x}.{y}"
                source_folder_path = source_path / source_folder_name
                
                if not source_folder_path.exists():
                    print(f"Warning: Source folder {source_folder_name} not found, skipping...")
                    continue
                
                # Determine target folder based on the rules
                if y <= 5:  # x.0 to x.5 -> folder x
                    target_folder_num = x
                else:  # x.6 to x.9
                    if x == 9:  # SPECIAL CASE: 9.6 to 9.9 -> folder 0
                        target_folder_num = 0
                    else:  # x.6 to x.9 -> folder x+1 (for x < 9)
                        target_folder_num = x + 1
                
                target_folder_path = target_path / str(target_folder_num)
                
                # Copy all images from source to target
                image_files = [f for f in source_folder_path.iterdir() 
                              if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']]
                
                files_copied = 0
                for image_file in image_files:
                    target_file = target_folder_path / image_file.name
                    
                    # Handle duplicate filenames by adding suffix
                    counter = 1
                    while target_file.exists():
                        stem = image_file.stem
                        suffix = image_file.suffix
                        target_file = target_folder_path / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.copy2(image_file, target_file)
                    files_copied += 1
                
                print(f"Processed {source_folder_name} -> {target_folder_num}/ ({files_copied} files)")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Haverland Folder Flattening Script")
    parser.add_argument("--source_dir", type=str, default="./haverland_original",
                       help="Source directory containing folders 0.0 to 9.9")
    parser.add_argument("--target_dir", type=str, default="./meterdigits",
                       help="Target directory for output")
    parser.add_argument("--class100", action="store_true", 
                       help="Use class100 mode: simple copy with *10 renaming (1.1 -> 11)")
    
    args = parser.parse_args()
    
    print("Haverland Folder Flattening Script")
    print("=" * 40)
    
    if args.class100:
        print("MODE: Class100 (simple copy with *10 renaming)")
        print("Rules: x.y -> folder (x*10 + y)")
    else:
        print("MODE: Original flattening")
        print("Rules:")
        print("  x.0 to x.5 -> folder x")
        print("  x.6 to x.9 -> folder x+1")
        print("  SPECIAL CASE: 9.6 to 9.9 -> folder 0")
    
    print()
    print(f"Source directory: {args.source_dir}")
    print(f"Target directory: {args.target_dir}")
    print()
    
    # Confirm with user
    response = input("Do you want to proceed? (y/n): ").lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    try:
        flatten_haverland_folders(args.source_dir, args.target_dir, class100=args.class100)
        print("\nOperation completed successfully!")
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()