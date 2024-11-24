# dir = "/sdf/scratch/users/t/tiffan/electrons_PMES_51"

import os
import pandas as pd
import argparse

def filter_catalog(directory_path, catalog_file_path, output_file_path):
    """
    Filters the file catalog to only include rows corresponding to files in the directory.

    Parameters:
        directory_path (str): Path to the directory containing files.
        catalog_file_path (str): Path to the CSV file containing the file catalog.
        output_file_path (str): Path to save the filtered CSV file.
    """
    # Get base filenames from the directory
    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return
    
    directory_files = [os.path.basename(f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    directory_basenames = set(os.path.splitext(f)[0] for f in directory_files)

    # Load the catalog
    if not os.path.exists(catalog_file_path):
        print(f"Error: The file catalog '{catalog_file_path}' does not exist.")
        return
    
    catalog = pd.read_csv(catalog_file_path)

    # Ensure 'filepath' column exists in the catalog
    if 'filepath' not in catalog.columns:
        print(f"Error: The file catalog must contain a 'filepath' column.")
        return
    
    # Extract base filenames from the catalog
    catalog['base_filename'] = catalog['filepath'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # Filter catalog rows to only include those with files in the directory
    filtered_catalog = catalog[catalog['base_filename'].isin(directory_basenames)]

    # Drop the 'base_filename' column from the filtered catalog for output
    filtered_catalog = filtered_catalog.drop(columns=['base_filename'])

    # Save the filtered catalog
    filtered_catalog.to_csv(output_file_path, index=False)
    print(f"Filtered catalog saved to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a file catalog based on existing files in a directory.")
    parser.add_argument("--directory", required=True, help="Path to the directory containing files.")
    parser.add_argument("--catalog", required=True, help="Path to the CSV file containing the file catalog.")
    parser.add_argument("--output", required=True, help="Path to save the filtered CSV file.")

    args = parser.parse_args()

    filter_catalog(args.directory, args.catalog, args.output)

