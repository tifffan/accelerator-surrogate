import pandas as pd
import argparse

def update_filepaths(catalog_file_path, old_path, new_path, output_file_path):
    """
    Replaces occurrences of an old path with a new path in the 'filepath' column of the catalog.

    Parameters:
        catalog_file_path (str): Path to the CSV file containing the file catalog.
        old_path (str): The string to be replaced in the 'filepath' column.
        new_path (str): The new string to replace the old path.
        output_file_path (str): Path to save the updated CSV file.
    """
    # Load the catalog
    if not catalog_file_path or not catalog_file_path.endswith('.csv'):
        print(f"Error: Invalid catalog file '{catalog_file_path}'. Ensure it is a valid CSV file.")
        return

    try:
        catalog = pd.read_csv(catalog_file_path)
    except Exception as e:
        print(f"Error reading the catalog file: {e}")
        return

    # Ensure 'filepath' column exists
    if 'filepath' not in catalog.columns:
        print(f"Error: The catalog file must contain a 'filepath' column.")
        return

    # Replace the old path with the new path in the 'filepath' column
    catalog['filepath'] = catalog['filepath'].str.replace(old_path, new_path, regex=False)

    # Save the updated catalog
    try:
        catalog.to_csv(output_file_path, index=False)
        print(f"Updated catalog saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving the updated catalog: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace path strings in the 'filepath' column of a CSV catalog.")
    parser.add_argument("--catalog", required=True, help="Path to the CSV file containing the file catalog.")
    parser.add_argument("--old_path", required=True, help="Old path string to replace.")
    parser.add_argument("--new_path", required=True, help="New path string to replace with.")
    parser.add_argument("--output", required=True, help="Path to save the updated CSV file.")

    args = parser.parse_args()

    update_filepaths(args.catalog, args.old_path, args.new_path, args.output)
