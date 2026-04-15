import os
import glob
import subprocess
import shutil
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

def run_command(cmd_list, cwd=None):
    """Run a command and allow it to stream output to stdout/stderr"""
    print(f"🚀 Running in {cwd or 'root'}: {' '.join(cmd_list)}")
    # We don't capture_output so we can see the progress bars (tqdm)
    result = subprocess.run(cmd_list, cwd=cwd)
    return result.returncode == 0

def generate_augmented_dataset():
    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    tools_dir = project_root / "datasets" / "tools"
    output_base = project_root / "datasets" / "static_augmentation"
    output_images = output_base / "images"
    temp_output = output_base / "temp_aug"

    print("🛠️  Starting Augmented Dataset Generation...")

    # 1. Initialize Directories
    if output_images.exists():
        print(f"🧹 Cleaning existing image directory: {output_images}")
        shutil.rmtree(output_images)
    output_images.mkdir(parents=True, exist_ok=True)
    
    if temp_output.exists():
        shutil.rmtree(temp_output)
    temp_output.mkdir(parents=True, exist_ok=True)

    # 2. Find JSON Configs
    json_configs = glob.glob(str(tools_dir / "*.json"))
    if not json_configs:
        print("❌ No JSON configuration files found in datasets/tools/")
        return

    # 3. Process Each Configuration
    for config_path in json_configs:
        config_name = os.path.basename(config_path)
        print(f"\n📁 Processing Configuration: {config_name}")
        
        # Run static_augmentation.py
        # Note: We run from tools_dir so that "../" paths in JSON resolve correctly
        cmd = [
            sys.executable, 
            "static_augmentation.py",
            "--config", config_name,
            "--output", str(temp_output.resolve()), # absolute output
            "--inverted", "true",
            "--yes"
        ]
        
        if run_command(cmd, cwd=str(tools_dir)):
            # Flatten: Move all images from temp_output recursively to output_images
            print(f"📦 Flattening results from {config_name}...")
            image_count = 0
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for img_path in Path(temp_output).rglob(ext):
                    # We use the existing filename (which includes codes and IDs)
                    dest_path = output_images / img_path.name
                    
                    # Handle potential collisions if any
                    if dest_path.exists():
                        # Prepend config name if collision occurs
                        new_name = f"{Path(config_path).stem}_{img_path.name}"
                        dest_path = output_images / new_name
                    
                    shutil.move(str(img_path), str(dest_path))
                    image_count += 1
            print(f"✅ Moved {image_count} images to flattened folder.")
            
            # Clean up temp folder for next configuration
            shutil.rmtree(temp_output)
            temp_output.mkdir(parents=True, exist_ok=True)
        else:
            print(f"⚠️  Failed to process {config_name}, skipping flattening.")

    # Remove final temp directory
    if temp_output.exists():
        shutil.rmtree(temp_output)

    # 4. Generate Labels
    print("\n🏷️  Generating Labels...")
    
    # 10 Classes
    label_10_file = output_base / "labels_10.txt"
    run_command([
        sys.executable, 
        str(tools_dir / "generate_label_10_classes.py"),
        "--folder", str(output_images),
        "--output", str(label_10_file)
    ])

    # 100 Classes
    label_100_file = output_base / "labels_100.txt"
    run_command([
        sys.executable, 
        str(tools_dir / "generate_label_100_classes.py"),
        "--folder", str(output_images),
        "--output", str(label_100_file)
    ])

    # 5. Shuffle Labels
    print("\n🔀 Shuffling Labels...")
    
    # Shuffle 10
    run_command([
        sys.executable, 
        str(tools_dir / "shuffle_labels.py"),
        str(label_10_file),
        "--output_file", str(output_base / "labels_10_shuffle.txt")
    ])
    
    # Shuffle 100
    run_command([
        sys.executable, 
        str(tools_dir / "shuffle_labels.py"),
        str(label_100_file),
        "--output_file", str(output_base / "labels_100_shuffle.txt")
    ])

    print("\n✨ Augmented Dataset Generation Complete!")
    print(f"📍 Files available in: {output_base}")

if __name__ == "__main__":
    generate_augmented_dataset()
