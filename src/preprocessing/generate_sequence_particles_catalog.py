import os
import glob
import argparse
import pandas as pd
import re

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """Helper function to create a sorting key that handles numbers in strings."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def generate_data_catalog(data_dir, output_csv):
    """
    Generates a data catalog CSV file containing file paths of particle data files,
    sorted by filename in natural numerical order.

    Args:
        data_dir (str): Directory containing the particle data files.
        output_csv (str): Path to the output CSV file.
    """
    # Find all particle data files in the directory
    particle_files = glob.glob(os.path.join(data_dir, '*_particle_data.pt'))
    if not particle_files:
        raise ValueError(f"No particle data files found in directory: {data_dir}")

    # Sort the particle files by filename using natural sort
    particle_files.sort(key=lambda s: natural_sort_key(os.path.basename(s)))

    # Create a DataFrame with one column 'filepath'
    df = pd.DataFrame({'filepath': particle_files})

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Data catalog saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a data catalog containing file paths of particle data files.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the particle data files.")
    parser.add_argument('--output_csv', type=str, default='data_catalog.csv', help="Path to the output CSV file.")

    args = parser.parse_args()

    generate_data_catalog(args.data_dir, args.output_csv)
