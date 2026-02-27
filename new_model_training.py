import argparse
import sys
import os
import subprocess

# Ensure stdout can print emojis without throwing cp1252 charmap errors on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Launch separate CMD windows to train a specific model across all 4 combinations concurrently")
    parser.add_argument("model_name", type=str, help="Name of the model architecture to train (e.g. digit_recognizer_v17)")
    args = parser.parse_args()
    
    model_name = args.model_name
    
    # We briefly import params to validate the name
    import parameters as params
    if model_name not in params.AVAILABLE_MODELS:
        print(f"Error: '{model_name}' is not in params.AVAILABLE_MODELS. Available choices are:")
        for m in params.AVAILABLE_MODELS:
            print(f"  - {m}")
        sys.exit(1)
        
    combinations = [
        # (nb_classes, channels, title)
        (10, 1, "10-class Grayscale"),
        (10, 3, "10-class RGB"),
        (100, 1, "100-class Grayscale"),
        (100, 3, "100-class RGB")
    ]
    
    print(f"\n{'='*80}")
    print(f"🚀 LAUNCHING CONCURRENT TRAINING WINDOWS: {model_name}")
    print(f"   Spawning {len(combinations)} separate CMD windows...")
    print(f"{'='*80}\n")
    
    for nb_classes, channels, desc in combinations:
        print(f"Launching {desc}...")
        
        # Build the command to be executed in the new CMD window.
        # We set the environment variables, then run the normal train script.
        # The /K switch keeps the CMD window open after the script ends so you can view the results.
        window_title = f"{model_name} - {desc}"
        cmd_string = f'set DIGIT_NB_CLASSES={nb_classes}&& set DIGIT_INPUT_CHANNELS={channels}&& title {window_title}&& python train.py --train {model_name}'
        
        # Use Windows 'start' to pop open a new command processor
        subprocess.Popen(f'start "{window_title}" cmd /K "{cmd_string}"', shell=True)
            
    print(f"\n{'='*80}")
    print(f"🎉 All 4 training sessions have been launched in separate windows.")
    print(f"   You can monitor their progress individually.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
