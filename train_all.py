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
    parser.add_argument("--classes", type=str, choices=["10", "100", "all"], default="all", help="Limit training to specific number of classes (10 or 100). Default is all.")
    parser.add_argument("--color", type=str, choices=["rgb", "gray", "all"], default="all", help="Limit training to specific color space (rgb or gray). Default is all.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--focal-gamma", type=float, default=4, help="Override Focal Loss Gamma.")
    parser.add_argument("--no-mixup", action="store_true", help="Disable Mixup augmentation.")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Override learning rate warmup epochs.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay parameter.")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Override label smoothing factor.")
    parser.add_argument("--rotation-range", type=float, default=None, help="Override maximum rotation augmentation in degrees.")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision.")
    args = parser.parse_args()
    
    # Set default environment variables strictly before importing parameters.py
    # This prevents parameters.py from triggering the interactive input prompt
    if "DIGIT_NB_CLASSES" not in os.environ:
        os.environ["DIGIT_NB_CLASSES"] = "10"
    if "DIGIT_INPUT_CHANNELS" not in os.environ:
        os.environ["DIGIT_INPUT_CHANNELS"] = "1"
        
    # Import params to get the list of active models
    import parameters as params
    
    active_models = params.AVAILABLE_MODELS
    
    if not active_models:
        print("Error: No models found in params.AVAILABLE_MODELS. Please uncomment models in parameters.py.")
        sys.exit(1)
        
    # Build combinations based on user arguments
    combinations = []
    
    classes_to_test = [10, 100] if args.classes == "all" else [int(args.classes)]
    colors_to_test = [(1, "Grayscale"), (3, "RGB")]
    if args.color == "gray":
        colors_to_test = [(1, "Grayscale")]
    elif args.color == "rgb":
        colors_to_test = [(3, "RGB")]
        
    for nb_classes in classes_to_test:
        for channels, color_desc in colors_to_test:
            combinations.append((nb_classes, channels, f"{nb_classes}-class {color_desc}"))
            
    if not combinations:
        print("Error: No combinations to run based on provided filters.")
        sys.exit(1)
    
    total_trainings = len(active_models) * len(combinations)
    
    print(f"\n{'='*80}")
    print(f"🚀 LAUNCHING FULL TRAINING SUITE")
    print(f"   Models to train ({len(active_models)}): {', '.join(active_models)}")
    print(f"   Combinations per model: {len(combinations)}")
    print(f"   Total training sessions: {total_trainings}")
    print(f"   Mode: {'Concurrent (Separate CMDs)' if args.concurrent else 'Sequential (Single Output)'}")
    print(f"{'='*80}\n")
    
    if args.concurrent:
        # Build extra arguments string
        extra_args = []
        if args.epochs is not None: extra_args.extend(["--epochs", str(args.epochs)])
        if args.batch is not None: extra_args.extend(["--batch", str(args.batch)])
        if args.lr is not None: extra_args.extend(["--lr", str(args.lr)])
        if args.warmup_epochs is not None: extra_args.extend(["--warmup-epochs", str(args.warmup_epochs)])
        if args.weight_decay is not None: extra_args.extend(["--weight-decay", str(args.weight_decay)])
        if args.label_smoothing is not None: extra_args.extend(["--label-smoothing", str(args.label_smoothing)])
        if args.focal_gamma is not None: extra_args.extend(["--focal-gamma", str(args.focal_gamma)])
        if args.rotation_range is not None: extra_args.extend(["--rotation-range", str(args.rotation_range)])
        if args.no_mixup: extra_args.append("--no-mixup")
        if args.no_mixed_precision: extra_args.append("--no-mixed-precision")
        extra_args_str = " ".join(extra_args)

        # Launching everything concurrently (Caution: High Resource usage)
        for model_name in active_models:
            for nb_classes, channels, desc in combinations:
                print(f"Launching {model_name} - {desc}...")
                
                window_title = f"{model_name} - {desc}"
                cmd_string = f'set DIGIT_NB_CLASSES={nb_classes}&& set DIGIT_INPUT_CHANNELS={channels}&& title {window_title}&& python train.py --train {model_name} {extra_args_str}'
                
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
                if args.epochs is not None: cmd.extend(["--epochs", str(args.epochs)])
                if args.batch is not None: cmd.extend(["--batch", str(args.batch)])
                if args.lr is not None: cmd.extend(["--lr", str(args.lr)])
                if args.warmup_epochs is not None: cmd.extend(["--warmup-epochs", str(args.warmup_epochs)])
                if args.weight_decay is not None: cmd.extend(["--weight-decay", str(args.weight_decay)])
                if args.label_smoothing is not None: cmd.extend(["--label-smoothing", str(args.label_smoothing)])
                if args.focal_gamma is not None: cmd.extend(["--focal-gamma", str(args.focal_gamma)])
                if args.rotation_range is not None: cmd.extend(["--rotation-range", str(args.rotation_range)])
                if args.no_mixup: cmd.append("--no-mixup")
                if args.no_mixed_precision: cmd.append("--no-mixed-precision")
                
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
