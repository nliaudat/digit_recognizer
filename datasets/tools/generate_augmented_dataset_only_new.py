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
    result = subprocess.run(cmd_list, cwd=cwd)
    return result.returncode == 0

def generate_augmented_dataset_only_new():
    project_root = Path(__file__).resolve().parent.parent.parent
    tools_dir = project_root / "datasets" / "tools"
    output_base = project_root / "datasets" / "static_augmentation"
    output_images = output_base / "images"
    temp_output = output_base / "temp_aug"

    print("🛠️  Starting Augmented Dataset Generation (NEW augmentations only)…")
    print("   (existing images preserved, only rotation_hard + quantization_noise added)")

    # 1. Ensure directories exist — do NOT clean existing images
    output_images.mkdir(parents=True, exist_ok=True)
    
    if temp_output.exists():
        shutil.rmtree(temp_output)
    temp_output.mkdir(parents=True, exist_ok=True)

    # 2. Find JSON Configs
    json_configs = glob.glob(str(tools_dir / "*.json"))
    if not json_configs:
        print("❌ No JSON configuration files found in datasets/tools/")
        return

    # 3. Process Each Configuration — only the 2 new augmentations
    for config_path in json_configs:
        config_name = os.path.basename(config_path)
        print(f"\n📁 Processing Configuration: {config_name}")
        
        cmd = [
            sys.executable, 
            "static_augmentation.py",
            "--config", config_name,
            "--output", str(temp_output.resolve()),
            "--augmentations", "rotation_hard", "quantization_noise",
            "--yes"
        ]
        
        if run_command(cmd, cwd=str(tools_dir)):
            # Flatten: Move all images from temp_output recursively to output_images
            print(f"📦 Flattening results from {config_name}…")
            image_count = 0
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for img_path in Path(temp_output).rglob(ext):
                    dest_path = output_images / img_path.name
                    
                    # Handle potential collisions
                    if dest_path.exists():
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

    # 3.5 Deduplicate Images (preserves originals, removes byte-identical dupes)
    print("\n🧹 Deduplicating Images…")
    run_command([
        sys.executable, 
        str(tools_dir / "deduplicate.py"),
        "--folder", str(output_images),
        "--delete"
    ])

    # 4. Regenerate labels (re-reads full folder, includes new images)
    print("\n🏷️  Regenerating Labels (appending new entries)…")
    
    label_10_file = output_base / "labels_10.txt"
    run_command([
        sys.executable, 
        str(tools_dir / "generate_label_10_classes.py"),
        "--folder", str(output_images),
        "--output", str(label_10_file)
    ])

    label_100_file = output_base / "labels_100.txt"
    run_command([
        sys.executable, 
        str(tools_dir / "generate_label_100_classes.py"),
        "--folder", str(output_images),
        "--output", str(label_100_file)
    ])

    # 5. Shuffle labels (deterministic ordering for training)
    print("\n🔀 Shuffling Labels…")
    
    run_command([
        sys.executable, 
        str(tools_dir / "shuffle_labels.py"),
        str(label_10_file),
        "--output_file", str(output_base / "labels_10_shuffle.txt")
    ])
    
    run_command([
        sys.executable, 
        str(tools_dir / "shuffle_labels.py"),
        str(label_100_file),
        "--output_file", str(output_base / "labels_100_shuffle.txt")
    ])

    print("\n✨ Augmented Dataset Update Complete!")
    print(f"📍 Files available in: {output_base}")
    print(f"   (new rotation_hard and quantization_noise variants added)")

if __name__ == "__main__":
    generate_augmented_dataset_only_new()