"""
Interactive input utility for Digit Recognition project.
Ensures project configuration (NB_CLASSES, INPUT_CHANNELS) is present by asking the user if needed.
"""

import os
import sys

def interactive_digit_config(override_classes=None, override_color=None):
    """
    Ensures environment variables and project configuration are set.
    Asks the user interactively if they are missing from CLI/Env.
    
    Args:
        override_classes: Optional int override from CLI
        override_color: Optional str override from CLI ('gray' or 'rgb')
    
    Returns:
        tuple (nb_classes, input_channels)
    """
    # 1. Classes handling
    nb_classes_env = os.environ.get("DIGIT_NB_CLASSES")
    if override_classes:
        nb_classes = str(override_classes)
        os.environ["DIGIT_NB_CLASSES"] = nb_classes
    elif nb_classes_env:
        nb_classes = nb_classes_env
    else:
        # Prompt only if no CLI and no Env
        while True:
            try:
                print("\n🔢 Digit Classification Configuration")
                val = input("   Enter number of classes [10 or 100]: ").strip()
                if val in ("10", "100"):
                    nb_classes = val
                    os.environ["DIGIT_NB_CLASSES"] = val
                    break
                print("   ❌ Error: Please enter 10 or 100.")
            except (KeyboardInterrupt, EOFError):
                print("\n⚠️ Interrupted. Using default 100 classes.")
                nb_classes = "100"
                os.environ["DIGIT_NB_CLASSES"] = "100"
                break

    # 2. Color Mode handling
    channels_env = os.environ.get("DIGIT_INPUT_CHANNELS")
    if override_color:
        channels = "1" if override_color.lower() == "gray" else "3"
        os.environ["DIGIT_INPUT_CHANNELS"] = channels
    elif channels_env:
        channels = channels_env
    else:
        # Prompt only if no CLI and no Env
        while True:
            try:
                val = input("🎨 Enter color mode [gray or rgb]: ").strip().lower()
                if val in ("gray", "rgb"):
                    channels = "1" if val == "gray" else "3"
                    os.environ["DIGIT_INPUT_CHANNELS"] = channels
                    break
                print("   ❌ Error: Please enter 'gray' or 'rgb'.")
            except (KeyboardInterrupt, EOFError):
                print("\n⚠️ Interrupted. Using default Grayscale mode.")
                channels = "1"
                os.environ["DIGIT_INPUT_CHANNELS"] = "1"
                break
                
    return int(nb_classes), int(channels)
