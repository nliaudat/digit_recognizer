# train_cleaning.py
import os
import shutil
import argparse

def cleanup_training_directory(output_dir, debug=False):
    """
    Clean up training checkpoints and intermediate files from a specific training directory
    
    Args:
        output_dir: Directory containing training files to clean
        debug: If True, show what would be deleted without actually deleting
    """
    if not os.path.exists(output_dir):
        if debug:
            print(f"🔍 Directory doesn't exist: {output_dir}")
        return 0, 0
    
    files_deleted = 0
    space_freed = 0
    
    print(f"🧹 Cleaning up: {output_dir}")
    
    # Delete checkpoint directories
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(checkpoints_dir):
        try:
            if debug:
                size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                          for dirpath, dirnames, filenames in os.walk(checkpoints_dir) 
                          for filename in filenames)
                print(f"📁 WOULD DELETE: {checkpoints_dir} ({size/1024/1024:.1f} MB)")
            else:
                size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                          for dirpath, dirnames, filenames in os.walk(checkpoints_dir) 
                          for filename in filenames)
                shutil.rmtree(checkpoints_dir)
                print(f"🗑️  Deleted checkpoints directory ({size/1024/1024:.1f} MB)")
                files_deleted += 1
                space_freed += size
        except Exception as e:
            print(f"⚠️  Failed to delete checkpoints directory: {e}")
    
    # Delete individual checkpoint files (both formats)
    checkpoint_files = []
    for file in os.listdir(output_dir):
        if (file.startswith("checkpoint_epoch_") and 
            (file.endswith(".keras") or file.endswith(".h5"))):
            checkpoint_files.append(file)
    
    for file in checkpoint_files:
        file_path = os.path.join(output_dir, file)
        try:
            if debug:
                size = os.path.getsize(file_path)
                print(f"📄 WOULD DELETE: {file} ({size/1024:.1f} KB)")
            else:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                print(f"🗑️  Deleted checkpoint: {file} ({size/1024:.1f} KB)")
                files_deleted += 1
                space_freed += size
        except Exception as e:
            print(f"⚠️  Failed to delete {file}: {e}")
    
    # PRESERVE essential files - DO NOT DELETE these
    preserve_files = [
        'training_config.txt', 'model_summary.txt', 'training_log.csv',
        'best_model.keras', 'digit_recognizer_v4.tflite', 'digit_recognizer_v4_float.tflite',
        'training_history.png', 'quantization_report.txt'
    ]
    
    if debug:
        print(f"💾 PRESERVING essential files: {preserve_files}")
    
    return files_deleted, space_freed

def cleanup_multiple_training_runs(results_dict, debug=False):
    """
    Clean up multiple training directories from train_specific_models results
    
    Args:
        results_dict: Results dictionary from train_specific_models containing output_dir
        debug: If True, show what would be deleted without actually deleting
    """
    total_files = 0
    total_space = 0
    
    for model_name, metrics in results_dict.items():
        if 'output_dir' in metrics and os.path.exists(metrics['output_dir']):
            files, space = cleanup_training_directory(metrics['output_dir'], debug)
            total_files += files
            total_space += space
    
    return total_files, total_space

def cleanup_all_training_directories(base_dir=None, debug=False):
    """
    Clean up all training directories in the output folder
    
    Args:
        base_dir: Base directory to search for training directories
        debug: If True, show what would be deleted without actually deleting
    """
    import parameters as params
    
    if base_dir is None:
        base_dir = getattr(params, 'OUTPUT_DIR', 'output')
    
    if not os.path.exists(base_dir):
        if debug:
            print(f"🔍 Base directory doesn't exist: {base_dir}")
        return 0, 0
    
    total_files = 0
    total_space = 0
    
    print(f"🔍 Searching for training directories in: {base_dir}")
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if this looks like a training directory
            training_artifacts = any(
                os.path.exists(os.path.join(item_path, artifact))
                for artifact in ['training_log.csv', 'model_summary.txt', 'training_config.txt', 'best_model.keras']
            )
            
            if training_artifacts:
                files, space = cleanup_training_directory(item_path, debug)
                total_files += files
                total_space += space
    
    return total_files, total_space

def cleanup_old_training_runs(days_old=7, debug=False):
    """
    Clean up training directories older than specified days
    
    Args:
        days_old: Delete directories older than this many days
        debug: If True, show what would be deleted without actually deleting
    """
    import parameters as params
    from datetime import datetime, timedelta
    
    print(f"🧹 CLEANING UP TRAINING RUNS OLDER THAN {days_old} DAYS")
    print("=" * 50)
    
    cutoff_time = datetime.now() - timedelta(days=days_old)
    base_output_dir = getattr(params, 'OUTPUT_DIR', 'output')
    
    if not os.path.exists(base_output_dir):
        print("❌ Output directory not found")
        return 0, 0
    
    files_deleted = 0
    space_freed = 0
    
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        
        if os.path.isdir(item_path):
            # Check if this is a training directory (contains training artifacts)
            training_artifacts = any(
                os.path.exists(os.path.join(item_path, artifact))
                for artifact in ['training_log.csv', 'model_summary.txt', 'training_config.txt']
            )
            
            if training_artifacts:
                # Get directory modification time
                dir_time = datetime.fromtimestamp(os.path.getmtime(item_path))
                
                if dir_time < cutoff_time:
                    try:
                        # Calculate directory size
                        total_size = 0
                        for dirpath, dirnames, filenames in os.walk(item_path):
                            for filename in filenames:
                                filepath = os.path.join(dirpath, filename)
                                total_size += os.path.getsize(filepath)
                        
                        if debug:
                            print(f"📁 WOULD DELETE: {item_path} ({total_size/1024/1024:.1f} MB) - {dir_time.strftime('%Y-%m-%d')}")
                        else:
                            shutil.rmtree(item_path)
                            print(f"🗑️  DELETED: {item_path} ({total_size/1024/1024:.1f} MB) - {dir_time.strftime('%Y-%m-%d')}")
                            files_deleted += 1
                            space_freed += total_size
                            
                    except Exception as e:
                        print(f"⚠️  Failed to delete {item_path}: {e}")
    
    print("\n" + "=" * 50)
    if debug:
        print(f"🔍 DRY RUN COMPLETED - No directories were actually deleted")
        print(f"📊 Would delete {files_deleted} directories")
        print(f"💾 Would free approximately {space_freed/1024/1024:.1f} MB")
    else:
        print(f"✅ OLD TRAINING RUNS CLEANUP COMPLETED")
        print(f"📊 Deleted {files_deleted} directories")
        print(f"💾 Freed approximately {space_freed/1024/1024:.1f} MB")
    
    return files_deleted, space_freed

def main():
    """Main entry point for standalone cleaning script"""
    parser = argparse.ArgumentParser(description='Clean up training checkpoints and old files')
    parser.add_argument('--directory', '-d', type=str, 
                       help='Specific directory to clean (default: all training directories)')
    parser.add_argument('--days', type=int, default=7,
                       help='Clean directories older than X days (default: 7)')
    parser.add_argument('--debug', action='store_true',
                       help='Dry run - show what would be deleted without actually deleting')
    parser.add_argument('--old-only', action='store_true',
                       help='Only clean old training runs, not current checkpoints')
    parser.add_argument('--current-only', action='store_true',
                       help='Only clean current checkpoints, not old training runs')
    parser.add_argument('--all', action='store_true',
                       help='Clean all training directories in output folder')
    
    args = parser.parse_args()
    
    print("🚀 TRAINING CLEANUP TOOL")
    print("=" * 60)
    
    total_files = 0
    total_space = 0
    
    try:
        # Clean specific directory
        if args.directory:
            files, space = cleanup_training_directory(args.directory, args.debug)
            total_files += files
            total_space += space
        
        # Clean all training directories
        elif args.all:
            files, space = cleanup_all_training_directories(debug=args.debug)
            total_files += files
            total_space += space
        
        # Clean current checkpoints (default behavior)
        elif not args.old_only:
            files, space = cleanup_all_training_directories(debug=args.debug)
            total_files += files
            total_space += space
        
        # Clean old training runs
        if not args.current_only:
            files, space = cleanup_old_training_runs(args.days, args.debug)
            total_files += files
            total_space += space
        
        # Final summary
        print("\n" + "=" * 60)
        print("🏁 CLEANUP SUMMARY")
        print("=" * 60)
        
        if args.debug:
            print(f"🔍 DRY RUN COMPLETED - No files were actually deleted")
        else:
            print(f"✅ CLEANUP COMPLETED SUCCESSFULLY")
        
        print(f"📊 Total items: {total_files}")
        print(f"💾 Total space: {total_space/1024/1024:.1f} MB")
        
        if total_files == 0:
            print("💡 No files needed cleaning - everything is already tidy! 🎉")
            
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()