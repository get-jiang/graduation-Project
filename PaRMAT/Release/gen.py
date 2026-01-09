#!/usr/bin/env python3
"""
Script to randomize edge weights in a Matrix Market format graph file.
Usage: python randomize_weights.py <input_file> <output_file> [min_weight] [max_weight]
"""

import sys
import random

def randomize_weights(input_file, output_file, min_weight=1, max_weight=100):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for i, line in enumerate(fin):
            # Keep header lines (starting with % or the dimension line)
            if line.startswith('%'):
                fout.write(line)
            elif i == 1:  # Dimension line (second line after header)
                fout.write(line)
            else:
                parts = line.strip().split()
                if len(parts) >= 2:
                    src, dst = parts[0], parts[1]
                    new_weight = random.randint(min_weight, max_weight)
                    fout.write(f"{src} {dst} {new_weight}\n")
                else:
                    fout.write(line)

    print(f"Done! Wrote randomized weights to {output_file}")
    print(f"Weight range: [{min_weight}, {max_weight}]")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python randomize_weights.py <input_file> <output_file> [min_weight] [max_weight]")
        print("Example: python randomize_weights.py rmat3.txt rmat3_random.txt 1 100")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    min_weight = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_weight = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    randomize_weights(input_file, output_file, min_weight, max_weight)
