import argparse
import os
import sys
from PIL import Image
import imagehash
from collections import defaultdict
import hashlib
import shutil

def calculate_phash(image_path):
    """Calculate perceptual hash for an image."""
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_duplicates_comprehensive(image_folder, hash_threshold=5):
    """
    Find exact and near-duplicate images in a folder.
    
    Args:
        image_folder (str): Path to folder containing images
        hash_threshold (int): Threshold for perceptual hash similarity (0-64)
    
    Returns:
        tuple: (exact_duplicates, similar_duplicates, all_files)
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    exact_duplicates = defaultdict(list)
    similar_groups = defaultdict(list)
    all_files = []
    
    print(f"Scanning folder: {image_folder}")
    
    # First pass: collect all image files
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(image_extensions):
            path = os.path.join(image_folder, filename)
            all_files.append(path)
    
    print(f"Found {len(all_files)} image files")
    
    # Second pass: calculate hashes
    print("Calculating hashes...")
    phash_dict = {}
    
    for i, path in enumerate(all_files):
        if i % 100 == 0:
            print(f"Processed {i}/{len(all_files)} images...")
        
        # Exact hash (MD5) - FIXED: hexdigest() not hexdig
        try:
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()  # FIXED HERE
            exact_duplicates[file_hash].append(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
        
        # Perceptual hash
        img_phash = calculate_phash(path)
        if img_phash is not None:
            phash_dict[path] = img_phash
    
    print("Finding similar images...")
    
    # Group similar images by perceptual hash
    processed_paths = set()
    for path1, phash1 in phash_dict.items():
        if path1 in processed_paths:
            continue
            
        similar_group = [path1]
        processed_paths.add(path1)
        
        for path2, phash2 in phash_dict.items():
            if path2 not in processed_paths and path1 != path2:
                if phash1 - phash2 <= hash_threshold:
                    similar_group.append(path2)
                    processed_paths.add(path2)
        
        if len(similar_group) > 1:
            # Use first path as the group key
            similar_groups[path1] = similar_group
    
    return exact_duplicates, similar_groups, all_files

def delete_duplicates(exact_duplicates, similar_groups, keep_original='oldest', dry_run=False):
    """
    Delete duplicate files based on the specified strategy.
    
    Args:
        exact_duplicates: Dict of exact duplicate groups
        similar_groups: Dict of similar image groups
        keep_original (str): Strategy for which file to keep
            'oldest' - keep the oldest file (by modification time)
            'newest' - keep the newest file
            'first' - keep the first file alphabetically
        dry_run (bool): If True, only show what would be deleted without actually deleting
    """
    files_to_delete = []
    
    # Process exact duplicates
    for file_hash, file_list in exact_duplicates.items():
        if len(file_list) > 1:
            # Determine which file to keep
            if keep_original == 'oldest':
                keep_file = min(file_list, key=lambda x: os.path.getmtime(x))
            elif keep_original == 'newest':
                keep_file = max(file_list, key=lambda x: os.path.getmtime(x))
            else:  # 'first'
                keep_file = min(file_list)
            
            # Mark others for deletion
            for file_path in file_list:
                if file_path != keep_file:
                    files_to_delete.append(file_path)
    
    # Process similar images (near-duplicates)
    for group_key, file_list in similar_groups.items():
        if len(file_list) > 1:
            # Determine which file to keep
            if keep_original == 'oldest':
                keep_file = min(file_list, key=lambda x: os.path.getmtime(x))
            elif keep_original == 'newest':
                keep_file = max(file_list, key=lambda x: os.path.getmtime(x))
            else:  # 'first'
                keep_file = min(file_list)
            
            # Mark others for deletion
            for file_path in file_list:
                if file_path != keep_file:
                    files_to_delete.append(file_path)
    
    # Remove duplicates from the delete list (in case a file appears in both exact and similar)
    files_to_delete = list(set(files_to_delete))
    
    # Delete files
    if dry_run:
        print(f"\n--- DRY RUN - Would delete {len(files_to_delete)} files ---")
    else:
        print(f"\n--- Deleting {len(files_to_delete)} files ---")
    
    total_saved_space = 0
    
    for file_path in files_to_delete:
        try:
            file_size = os.path.getsize(file_path)
            if dry_run:
                print(f"WOULD DELETE: {file_path} ({file_size} bytes)")
            else:
                print(f"DELETING: {file_path} ({file_size} bytes)")
                os.remove(file_path)
            total_saved_space += file_size
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return len(files_to_delete), total_saved_space

def print_duplicate_report(exact_duplicates, similar_groups):
    """Print a detailed report of found duplicates."""
    print("\n" + "="*60)
    print("DUPLICATE REPORT")
    print("="*60)
    
    exact_dup_count = sum(len(files) - 1 for files in exact_duplicates.values() if len(files) > 1)
    similar_dup_count = sum(len(files) - 1 for files in similar_groups.values())
    
    print(f"Exact duplicates found: {exact_dup_count} files can be removed")
    print(f"Similar images found: {similar_dup_count} files can be removed")
    print(f"Total potential space savings: {exact_dup_count + similar_dup_count} files")
    
    if exact_duplicates:
        print(f"\n=== EXACT DUPLICATES ({len([x for x in exact_duplicates.values() if len(x) > 1])} groups) ===")
        for hash_val, files in exact_duplicates.items():
            if len(files) > 1:
                print(f"\nExact duplicate group ({len(files)} files):")
                for f in files:
                    file_size = os.path.getsize(f)
                    mtime = os.path.getmtime(f)
                    print(f"  - {f} ({file_size} bytes, modified: {mtime})")
    
    if similar_groups:
        print(f"\n=== SIMILAR IMAGES ({len(similar_groups)} groups) ===")
        for group_key, files in similar_groups.items():
            if len(files) > 1:
                print(f"\nSimilar image group ({len(files)} files):")
                for f in files:
                    file_size = os.path.getsize(f)
                    mtime = os.path.getmtime(f)
                    print(f"  - {f} ({file_size} bytes, modified: {mtime})")

def main():
    parser = argparse.ArgumentParser(description='Find and remove duplicate images')
    parser.add_argument('--folder', required=True, help='Folder containing images to scan')
    parser.add_argument('--threshold', type=int, default=5, 
                       help='Perceptual hash threshold (0-64, lower=more strict, default=3)') #0-5: Nearly identical images | 5-10: Very similar images | 10+: Different images
    parser.add_argument('--keep', choices=['oldest', 'newest', 'first'], default='oldest',
                       help='Which file to keep when duplicates found (default: oldest)')
    parser.add_argument('--delete', action='store_true',
                       help='Actually delete duplicates (without this flag, only shows report)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    
    args = parser.parse_args()
    
    # Validate folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist")
        sys.exit(1)
    
    # Validate threshold
    if args.threshold < 0 or args.threshold > 64:
        print("Error: Threshold must be between 0 and 64")
        sys.exit(1)
    
    # Find duplicates
    exact_duplicates, similar_groups, all_files = find_duplicates_comprehensive(
        args.folder, args.threshold
    )
    
    # Print report
    print_duplicate_report(exact_duplicates, similar_groups)
    
    # Handle deletion
    if args.delete or args.dry_run:
        deleted_count, saved_space = delete_duplicates(
            exact_duplicates, similar_groups, args.keep, args.dry_run
        )
        
        print(f"\n{'DRY RUN: ' if args.dry_run else ''}Summary:")
        print(f"Files marked for deletion: {deleted_count}")
        print(f"Estimated space savings: {saved_space / (1024*1024):.2f} MB")
        
        if args.dry_run:
            print("\nThis was a dry run. Use --delete to actually remove files.")
    else:
        print(f"\nTo remove these duplicates, run with --delete flag")
        print(f"To see what would be deleted without removing, use --dry-run")

if __name__ == "__main__":
    main()