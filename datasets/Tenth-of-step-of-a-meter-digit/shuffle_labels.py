import random
import argparse

def shuffle_file(input_file, output_file):
    # Read lines from the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Shuffle the lines while keeping pairs intact
    random.shuffle(lines)

    # Write the shuffled lines to the output file
    with open(output_file, 'w') as f:
        f.writelines(lines)

    print(f"Shuffled file saved to: {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Shuffle lines in a file while keeping pairs intact.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("--output_file", help="Path to the output file (default: {input}_shuffle.txt)", default=None)

    # Parse arguments
    args = parser.parse_args()

    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = f"{args.input_file.rsplit('.', 1)[0]}_shuffle.txt"

    # Shuffle the file
    shuffle_file(args.input_file, args.output_file)
    
    
## usage : 

# python shuffle_labels.py haverland_list.txt #--output_file custom_output.txt