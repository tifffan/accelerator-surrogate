import pandas as pd
import argparse

def create_test_catalog(train_catalog_path, all_catalog_path, output_test_catalog_path):
    """
    Creates a test catalog by excluding train catalog entries from the all catalog.

    Parameters:
        train_catalog_path (str): Path to the train catalog CSV file.
        all_catalog_path (str): Path to the all catalog CSV file.
        output_test_catalog_path (str): Path to save the generated test catalog CSV file.
    """
    try:
        # Load the train and all catalogs
        train_catalog = pd.read_csv(train_catalog_path)
        all_catalog = pd.read_csv(all_catalog_path)

        # Ensure 'filepath' column exists in both catalogs
        if 'filepath' not in train_catalog.columns or 'filepath' not in all_catalog.columns:
            print("Error: Both catalogs must contain a 'filepath' column.")
            return

        # Identify test catalog entries
        train_filepaths = set(train_catalog['filepath'])
        test_catalog = all_catalog[~all_catalog['filepath'].isin(train_filepaths)]

        # Save the test catalog
        test_catalog.to_csv(output_test_catalog_path, index=False)
        print(f"Test catalog created and saved to: {output_test_catalog_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a test catalog by excluding train entries from the all catalog.")
    parser.add_argument("--train", required=True, help="Path to the train catalog CSV file.")
    parser.add_argument("--all", required=True, help="Path to the all catalog CSV file.")
    parser.add_argument("--output", required=True, help="Path to save the test catalog CSV file.")

    args = parser.parse_args()

    create_test_catalog(args.train, args.all, args.output)
