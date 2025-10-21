import argparse

def parse_arguments():
    """Parse CLI arguments for training script."""
    parser = argparse.ArgumentParser(description="Digit Recognizer Training")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--clear-cache', action='store_true', help='Clear dataset cache after training')
    parser.add_argument('--test-all-models', action='store_true', help='Test all available model architectures')

    return parser.parse_args()
