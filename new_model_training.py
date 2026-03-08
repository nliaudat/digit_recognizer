import argparse
import sys
import os
import subprocess
import platform

# PREVENT INTERACTIVE PROMPTS: Set defaults immediately before any other imports
os.environ.setdefault("DIGIT_NB_CLASSES", "10")
os.environ.setdefault("DIGIT_INPUT_CHANNELS", "3")

# Ensure stderr can handle emojis/unicode on Windows
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Launch training for a specific model across all 4 combinations")
    parser.add_argument("model_name", type=str, help="Name of the model architecture to train (e.g. digit_recognizer_v19)")
    args = parser.parse_args()
    
    model_name = args.model_name
    
    # Import parameters ONLY after environment defaults are set
    import parameters as params
    
    if model_name not in params.AVAILABLE_MODELS:
        print(f"Error: '{model_name}' is not a valid model architecture.")
        print(f"Available choices: {', '.join(params.AVAILABLE_MODELS)}")
        sys.exit(1)
        
    combinations = [
        (10, 1, "10-class Grayscale"),
        (10, 3, "10-class RGB"),
        (100, 1, "100-class Grayscale"),
        (100, 3, "100-class RGB")
    ]
    
    is_windows = platform.system() == "Windows"
    
    print(f"\n{'='*80}")
    print(f"🚀 LAUNCHING TRAINING PIPELINE: {model_name}")
    if is_windows:
        print(f"   Mode: Concurrent (4 separate CMD windows)")
    else:
        print(f"   Mode: Sequential (Foreground)")
    print(f"{'='*80}\n")
    
    for nb_classes, channels, desc in combinations:
        print(f"Preparing {desc}...")
        
        # Prepare environment
        env = os.environ.copy()
        env["DIGIT_NB_CLASSES"] = str(nb_classes)
        env["DIGIT_INPUT_CHANNELS"] = str(channels)
        
        if is_windows:
            # On Windows, spawn new windows
            window_title = f"{model_name} - {desc}"
            # Use 'set' without spaces around '&&' to avoid trailing space issues
            cmd_string = f'set DIGIT_NB_CLASSES={nb_classes}&&set DIGIT_INPUT_CHANNELS={channels}&&title {window_title}&&python train.py --train {model_name}'
            subprocess.Popen(f'start "{window_title}" cmd /K "{cmd_string}"', shell=True)
        else:
            # On Linux/Docker, run sequentially in foreground
            print(f"   → Training {desc}...")
            try:
                subprocess.run(
                    [sys.executable, "train.py", "--train", model_name],
                    env=env,
                    check=True
                )
                print(f"   ✓ Finished {desc}")
            except subprocess.CalledProcessError as e:
                print(f"   ✗ Error during {desc}: {e}")
            
    print(f"\n{'='*80}")
    if is_windows:
        print(f"🎉 All 4 training sessions have been launched in separate windows.")
    else:
        print(f"🎉 All training combinations have been processed.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
