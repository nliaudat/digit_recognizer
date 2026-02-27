import argparse
import sys
import os
import subprocess

# Ensure stdout can print emojis without throwing cp1252 charmap errors on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Launch training for all active models in parameters.py across all 4 combinations (10/100 classes, Grayscale/RGB) sequentially or concurrently.")
    parser.add_argument("--concurrent", action="store_true", help="Launch all trainings at once in separate CMD windows (WARNING: Requires massive RAM/GPU resources). If not set, trains sequentially in this window.")
    args = parser.parse_args()
    
    # Import params to get the list of active models
    import parameters as params
    
    active_models = params.AVAILABLE_MODELS
    
    if not active_models:
        print("Error: No models found in params.AVAILABLE_MODELS. Please uncomment models in parameters.py.")
        sys.exit(1)
        
    combinations = [
        # (nb_classes, channels, title)
        (10, 1, "10-class Grayscale"),
        (10, 3, "10-class RGB"),
        (100, 1, "100-class Grayscale"),
        (100, 3, "100-class RGB")
    ]
    
    total_trainings = len(active_models) * len(combinations)
    
    print(f"\n{'='*80}")
    print(f"🚀 LAUNCHING FULL TRAINING SUITE")
    print(f"   Models to train ({len(active_models)}): {', '.join(active_models)}")
    print(f"   Combinations per model: {len(combinations)}")
    print(f"   Total training sessions: {total_trainings}")
    print(f"   Mode: {'Concurrent (Separate CMDs)' if args.concurrent else 'Sequential (Single Output)'}")
    print(f"{'='*80}\n")
    
    if args.concurrent:
        # Launching everything concurrently (Caution: High Resource usage)
        for model_name in active_models:
            for nb_classes, channels, desc in combinations:
                print(f"Launching {model_name} - {desc}...")
                
                window_title = f"{model_name} - {desc}"
                cmd_string = f'set DIGIT_NB_CLASSES={nb_classes}&& set DIGIT_INPUT_CHANNELS={channels}&& title {window_title}&& python train.py --train {model_name}'
                
                # Use Windows 'start' to pop open a new command processor
                subprocess.Popen(f'start "{window_title}" cmd /c "{cmd_string}"', shell=True)
                
        print(f"\n{'='*80}")
        print(f"🎉 All {total_trainings} training sessions have been launched in separate windows.")
        print(f"   You can monitor their progress individually.")
        print(f"{'='*80}\n")
        
    else:
        # Sequential Execution (Safer for system resources)
        completed = 0
        for model_name in active_models:
            for nb_classes, channels, desc in combinations:
                completed += 1
                print(f"\n{'*'*80}")
                print(f"Training [{completed}/{total_trainings}]: {model_name} - {desc}")
                print(f"{'*'*80}\n")
                
                # Set environment variables for the subprocess
                env = os.environ.copy()
                env["DIGIT_NB_CLASSES"] = str(nb_classes)
                env["DIGIT_INPUT_CHANNELS"] = str(channels)
                
                cmd = [sys.executable, "train.py", "--train", model_name]
                
                try:
                    # Run the process sequentially and pipe output to original console
                    process = subprocess.Popen(cmd, env=env)
                    process.wait()
                    
                    if process.returncode != 0:
                        print(f"⚠️ Warning: Training for {model_name} - {desc} exited with code {process.returncode}")
                except KeyboardInterrupt:
                    print("\n🛑 Sequential training aborted by user.")
                    sys.exit(1)
                    
        print(f"\n{'='*80}")
        print(f"✅ Sequential training suite completed successfully.")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
