import argparse
import sys
import os
import subprocess

# Ensure stdout can print emojis without throwing cp1252 charmap errors on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set default environment variables strictly before importing parameters.py
# This prevents parameters.py from triggering the interactive input prompt
if "DIGIT_NB_CLASSES" not in os.environ:
    os.environ["DIGIT_NB_CLASSES"] = "10"
if "DIGIT_INPUT_CHANNELS" not in os.environ:
    os.environ["DIGIT_INPUT_CHANNELS"] = "1"

def main():
    parser = argparse.ArgumentParser(description="Launch training for all active models in parameters.py across all 4 combinations (10/100 classes, Grayscale/RGB) sequentially or concurrently.")
    
    # Mode
    parser.add_argument("--concurrent", action="store_true", help="Launch each training in a separate command window (Concurrent)")
    
    # Combinations filters
    parser.add_argument("--classes", type=str, default=None, choices=["10", "100", "all"], help="Classes: '10', '100', or 'all'")
    parser.add_argument("--color", type=str, default=None, choices=["gray", "rgb", "all"], help="Color mode: 'gray', 'rgb' or 'all'")
    
    # Hyperparameter overrides (passed to train.py)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--rotation-range", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--lr-scheduler", type=str, default=None)
    
    # Flags (passed to train.py)
    parser.add_argument("--cutmix", action="store_true", default=False)
    parser.add_argument("--no-mixup", action="store_true", default=False)
    parser.add_argument("--no-random-erasing", action="store_true", default=False)
    parser.add_argument("--no-dynamic-weights", action="store_true", default=False)
    parser.add_argument("--no-mixed-precision", action="store_true", default=False)

    args = parser.parse_args()
        
    # Import params to get the list of active models
    import parameters as params
    
    active_models = params.AVAILABLE_MODELS
    
    if not active_models:
        print("Error: No models found in params.AVAILABLE_MODELS. Please uncomment models in parameters.py.")
        sys.exit(1)
        
    # ── Interactive Prompts (fallback if not provided on CLI) ─────────────────
    if args.classes is None:
        if sys.stdin.isatty():
            while True:
                _ui = input("Enter number of classes [10, 100, or all]: ").strip().lower()
                if _ui in ["10", "100", "all"]:
                    args.classes = _ui
                    break
        else:
            args.classes = "all"
            
    if args.color is None:
        if sys.stdin.isatty():
            while True:
                _ui = input("Enter color mode [gray, rgb, or all]: ").strip().lower()
                if _ui in ["gray", "rgb", "all"]:
                    args.color = _ui
                    break
        else:
            args.color = "all"

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
        if args.optimizer is not None: extra_args.extend(["--optimizer", args.optimizer])
        if args.lr_scheduler is not None: extra_args.extend(["--lr-scheduler", args.lr_scheduler])
        
        if args.cutmix: extra_args.append("--cutmix")
        if args.no_mixup: extra_args.append("--no-mixup")
        if args.no_random_erasing: extra_args.append("--no-random-erasing")
        if args.no_dynamic_weights: extra_args.append("--no-dynamic-weights")
        if args.no_mixed_precision: extra_args.append("--no-mixed-precision")
        
        extra_args_str = " ".join(extra_args)

        # Launching everything concurrently (Caution: High Resource usage)
        for model_name in active_models:
            for nb_classes, channels, desc in combinations:
                print(f"Launching {model_name} - {desc}...")
                
                env = os.environ.copy()
                env["DIGIT_NB_CLASSES"] = str(nb_classes)
                env["DIGIT_INPUT_CHANNELS"] = str(channels)
                
                full_cmd = [sys.executable, "train.py", "--train", model_name] + extra_args
                subprocess.Popen(full_cmd, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
                
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
                if args.optimizer is not None: cmd.extend(["--optimizer", args.optimizer])
                if args.lr_scheduler is not None: cmd.extend(["--lr-scheduler", args.lr_scheduler])
                if args.cutmix: cmd.append("--cutmix")
                if args.no_mixup: cmd.append("--no-mixup")
                if args.no_random_erasing: cmd.append("--no-random-erasing")
                if args.no_dynamic_weights: cmd.append("--no-dynamic-weights")
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
