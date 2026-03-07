"""
retrain_all.py
==============
Discovers interrupted training runs under exported_models/ and resumes each
one from its best_model.keras checkpoint.

Auto-detects:
  - The training directory (NB_CLASSES, color) from the folder name
  - The initial_epoch from the last row of training_log.csv
  - DIGIT_NB_CLASSES and DIGIT_INPUT_CHANNELS env-vars from the folder name

Usage:
  python retrain_all.py                              # Resume all incomplete runs
  python retrain_all.py --concurrent                 # Launch in separate windows
  python retrain_all.py --classes 10 --color rgb     # Filter scope
  python retrain_all.py --epochs 300                 # Override epoch target
  python retrain_all.py --dir exported_models\\10cls_RGB\\digit_recognizer_v17_10cls_RGB
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_last_epoch(log_path: Path) -> int | None:
    """Return epoch number from last data row of training_log.csv, or None."""
    try:
        with open(log_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return int(float(rows[-1]['epoch']))
    except Exception as e:
        print(f"  ⚠️  Could not read {log_path}: {e}")
        return None


def _check_convergence(log_path: Path, patience: int) -> bool:
    """
    Check if the training has converged based on val_accuracy.
    Returns True if no improvement in val_accuracy for 'patience' epochs.
    """
    try:
        with open(log_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) < patience:
            return False
        
        # Determine monitor key
        monitor_key = 'val_accuracy'
        if monitor_key not in rows[0]:
            # Fallback to accuracy if val_accuracy is missing
            monitor_key = 'accuracy' if 'accuracy' in rows[0] else None
            
        if not monitor_key:
            return False

        # Get values for monitoring
        values = []
        for r in rows:
            try:
                values.append(float(r[monitor_key]))
            except (ValueError, KeyError):
                continue
                
        if len(values) < patience:
            return False
            
        # Check if the best value happened more than 'patience' epochs ago
        best_val = max(values)
        # Find index of last occurrence of best value
        # We search from the end to find the most recent 'best'
        last_best_idx = len(values) - 1 - values[::-1].index(best_val)
        
        epochs_since_best = (len(values) - 1) - last_best_idx
        return epochs_since_best >= patience
        
    except Exception as e:
        print(f"  ⚠️  Convergence check failed for {log_path}: {e}")
        return False


def _parse_folder(folder_name: str) -> tuple[str, int, int] | None:
    """
    Parse model name, nb_classes, and n_channels from a folder name.
    E.g. 'digit_recognizer_v17_10cls_RGB' -> ('digit_recognizer_v17', 10, 3)
    Returns None if pattern not matched.
    """
    m = re.search(r'^(.+?)_(\d+)cls_(RGB|GRAY)$', folder_name)
    if not m:
        return None
    model_name = m.group(1)
    nb_classes  = int(m.group(2))
    channels    = 3 if m.group(3).upper() == 'RGB' else 1
    return model_name, nb_classes, channels


def _discover_runs(base_dir: Path, classes_filter=None, color_filter=None) -> list[dict]:
    """Walk exported_models/ and collect resumable training directories."""
    runs = []
    for run_dir in sorted(base_dir.rglob('*')):
        if not run_dir.is_dir():
            continue
        keras_path = run_dir / 'best_model.keras'
        log_path   = run_dir / 'training_log.csv'
        if not keras_path.exists() or not log_path.exists():
            continue
        parsed = _parse_folder(run_dir.name)
        if parsed is None:
            continue
        model_name, nb_classes, channels = parsed
        # Apply filters
        if classes_filter is not None and nb_classes != classes_filter:
            continue
        color_str = 'RGB' if channels == 3 else 'GRAY'
        if color_filter is not None and color_str.lower() != color_filter.lower():
            continue
        last_epoch = _read_last_epoch(log_path)
        if last_epoch is None:
            continue
        initial_epoch = last_epoch + 1
        runs.append({
            'run_dir':       run_dir,
            'keras_path':    keras_path,
            'log_path':      log_path,
            'model_name':    model_name,
            'nb_classes':    nb_classes,
            'channels':      channels,
            'initial_epoch': initial_epoch,
        })
    return runs


def _build_cmd(run: dict, args) -> list[str]:
    """Build the train.py subprocess command for one run."""
    cmd = [
        sys.executable, 'train.py',
        '--train', run['model_name'],
        '--resume', str(run['keras_path']),
        '--initial-epoch', str(run['initial_epoch']),
    ]
    if args.epochs is not None:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.batch is not None:
        cmd.extend(['--batch', str(args.batch)])
    if args.lr is not None:
        cmd.extend(['--lr', str(args.lr)])
    if args.no_analysis:
        cmd.append('--no_analysis')
    if args.no_cleanup:
        cmd.append('--no_cleanup')
    if args.debug:
        cmd.append('--debug')
    return cmd


def _build_env(run: dict) -> dict:
    """Build subprocess environment with correct DIGIT_ vars."""
    env = os.environ.copy()
    env['DIGIT_NB_CLASSES']     = str(run['nb_classes'])
    env['DIGIT_INPUT_CHANNELS'] = str(run['channels'])
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Resume interrupted train.py runs from best_model.keras checkpoints.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--dir', type=str, default='',
        help='Resume a single specific training directory, e.g.\n'
             '  exported_models/100cls_GRAY/digit_recognizer_v3_100cls_GRAY\n'
             'Tip: use forward slashes or quote the path to avoid shell escaping issues.')
    parser.add_argument('--concurrent', action='store_true',
        help='Launch all resumes simultaneously in separate CMD windows.')
    parser.add_argument('--classes', type=str, choices=['10', '100', 'all'], default='all',
        help='Filter by number of classes (10, 100, or all). Default: all.')
    parser.add_argument('--color', type=str, choices=['rgb', 'gray', 'all'], default='all',
        help='Filter by color space (rgb, gray, or all). Default: all.')
    parser.add_argument('--epochs', type=int, default=None,
        help='Override total epoch target for resumed runs.')
    parser.add_argument('--batch', type=int, default=None,
        help='Override batch size.')
    parser.add_argument('--lr', type=float, default=None,
        help='Override learning rate.')
    parser.add_argument('--no_analysis', action='store_true',
        help='Skip post-training analysis.')
    parser.add_argument('--no_cleanup', action='store_true',
        help='Skip checkpoint cleanup after training.')
    parser.add_argument('--force', action='store_true',
        help='Force resume even if training appears to have converged.')
    parser.add_argument('--patience', type=int, default=None,
        help='Convergence patience (epochs without improvement). Defaults to param value.')
    parser.add_argument('--debug', action='store_true',
        help='Enable debug logs.')

    args = parser.parse_args()

    base_dir = Path('exported_models')

    # Load parameters to get default patience
    default_patience = 30
    try:
        import parameters as p
        default_patience = getattr(p, 'EARLY_STOPPING_PATIENCE', 30)
    except Exception:
        pass
    
    patience = args.patience if args.patience is not None else default_patience

    # ------------------------------------------------------------------
    # Determine runs to resume
    # ------------------------------------------------------------------
    if args.dir:
        # Normalize path: strip wrapping quotes, accept both slash styles
        dir_str = args.dir.strip().strip('"\'')
        dir_str = dir_str.replace('\\', os.sep).replace('/', os.sep)
        run_dir = Path(dir_str)
        if not run_dir.exists():
            print(f"❌ Directory not found: {run_dir}")
            print(f"   Tip: use forward slashes or quote the path, e.g.:")
            print(f"   python retrain_all.py --dir \"exported_models/100cls_GRAY/digit_recognizer_v3_100cls_GRAY\"")
            sys.exit(1)
        keras_path = run_dir / 'best_model.keras'
        log_path   = run_dir / 'training_log.csv'
        if not keras_path.exists():
            print(f"❌ No best_model.keras found in {run_dir}")
            sys.exit(1)
        if not log_path.exists():
            print(f"❌ No training_log.csv found in {run_dir}")
            sys.exit(1)
        parsed = _parse_folder(run_dir.name)
        if parsed is None:
            print(f"❌ Cannot parse folder name: {run_dir.name}\n"
                  f"   Expected pattern: <model>_<N>cls_<RGB|GRAY>")
            sys.exit(1)
        model_name, nb_classes, channels = parsed
        last_epoch = _read_last_epoch(log_path)
        if last_epoch is None:
            print("❌ training_log.csv is empty or unreadable.")
            sys.exit(1)
        runs = [{
            'run_dir':       run_dir,
            'keras_path':    keras_path,
            'log_path':      log_path,
            'model_name':    model_name,
            'nb_classes':    nb_classes,
            'channels':      channels,
            'initial_epoch': last_epoch + 1,
        }]
    else:
        # Discovery mode
        if not base_dir.exists():
            print(f"❌ Base directory not found: {base_dir}")
            sys.exit(1)
        classes_filter = None if args.classes == 'all' else int(args.classes)
        color_filter   = None if args.color   == 'all' else args.color
        runs = _discover_runs(base_dir, classes_filter, color_filter)

    if not runs:
        print("ℹ️  No resumable training runs found.")
        sys.exit(0)

    # Filter out already-finished runs
    target_epochs = args.epochs
    if target_epochs is None:
        # Try to read from parameters.py
        try:
            import parameters as p
            target_epochs = p.EPOCHS
        except Exception:
            target_epochs = 0

    pending = []
    skipped = []
    converged = []
    for run in runs:
        if target_epochs > 0 and run['initial_epoch'] >= target_epochs:
            skipped.append(run)
        elif not args.force and _check_convergence(run['log_path'], patience):
            converged.append(run)
        else:
            pending.append(run)

    print(f"\n{'='*70}")
    print(f"♻️  RETRAIN ALL — Resume Interrupted Training Runs")
    print(f"   Target epochs  : {target_epochs or '(not set)'}")
    print(f"   Convergence pat.: {patience}")
    print(f"   Runs found     : {len(runs)}")
    print(f"   Already done   : {len(skipped)}")
    print(f"   Converged      : {len(converged)}")
    print(f"   To resume      : {len(pending)}")
    print(f"   Mode           : {'Concurrent' if args.concurrent else 'Sequential'}")
    print(f"{'='*70}\n")

    for run in skipped:
        print(f"  ✅ Finished (epoch {run['initial_epoch']}): {run['run_dir'].name}")
    for run in converged:
        print(f"  ⏭️  Skipped (converged): {run['run_dir'].name}")
    if skipped or converged:
        print()

    if not pending:
        print("Nothing to do. All runs are complete.")
        return

    for i, run in enumerate(pending, 1):
        color = 'RGB' if run['channels'] == 3 else 'GRAY'
        print(f"  [{i}/{len(pending)}] {run['model_name']} — {run['nb_classes']}cls {color}  "
              f"(resume from epoch {run['initial_epoch']})")

    print()

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------
    if args.concurrent:
        for run in pending:
            cmd = _build_cmd(run, args)
            env = _build_env(run)
            color = 'RGB' if run['channels'] == 3 else 'GRAY'
            env = _build_env(run)
            # Use CREATE_NEW_CONSOLE to launch in a separate window on Windows without shell=True
            subprocess.Popen(cmd, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
        print(f"🚀 Launched {len(pending)} resume sessions in separate windows.")
    else:
        for i, run in enumerate(pending, 1):
            color = 'RGB' if run['channels'] == 3 else 'GRAY'
            print(f"\n{'*'*70}")
            print(f"Resuming [{i}/{len(pending)}]: {run['model_name']} — "
                  f"{run['nb_classes']}cls {color}  (epoch {run['initial_epoch']})")
            print(f"{'*'*70}\n")
            cmd = _build_cmd(run, args)
            env = _build_env(run)
            try:
                process = subprocess.Popen(cmd, env=env)
                process.wait()
                if process.returncode != 0:
                    print(f"⚠️  Exited with code {process.returncode}: {run['run_dir'].name}")
            except KeyboardInterrupt:
                print("\n🛑 Retrain aborted by user.")
                sys.exit(1)

        print(f"\n{'='*70}")
        print(f"✅ Retrain suite complete. {len(pending)} run(s) resumed.")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()


# python retrain_all.py                          # Resume all incomplete runs sequentially
# python retrain_all.py --concurrent             # Resume all in separate CMD windows
# python retrain_all.py --classes 100 --color rgb # Filter by scope
# python retrain_all.py --epochs 300             # Override total epoch target
# python retrain_all.py --dir "exported_models\100cls_RGB\super_high_accuracy_validator_100cls_RGB" --epochs 300 # Resume a single specific run
