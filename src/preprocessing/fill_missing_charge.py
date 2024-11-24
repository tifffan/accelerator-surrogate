import pandas as pd
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python fill_missing_charge.py first_catalog.csv second_catalog.csv output_catalog.csv")
        sys.exit(1)

    first_catalog_file = sys.argv[1]
    second_catalog_file = sys.argv[2]
    output_catalog_file = sys.argv[3]

    # Read the CSV files into pandas DataFrames
    first_df = pd.read_csv(first_catalog_file)
    second_df = pd.read_csv(second_catalog_file)

    # Check if 'distgen_total_charge' exists in the first catalog
    if 'distgen_total_charge' in first_df.columns:
        print("The 'distgen_total_charge' column already exists in the first catalog.")
        sys.exit(1)

    # Merge the DataFrames on 'filename' to add 'distgen_total_charge' to the first catalog
    merged_df = pd.merge(first_df, second_df[['filename', 'distgen_total_charge']], on='filename', how='left')

    # Handle missing matches
    missing_filenames = merged_df[merged_df['distgen_total_charge'].isnull()]['filename']
    if not missing_filenames.empty:
        print("Warning: The following filenames were not found in the second catalog:")
        print(missing_filenames.to_list())
    
    # Remove columns whose names start with "Unnamed"
    merged_df = merged_df.loc[:, ~merged_df.columns.str.startswith('Unnamed')]

    # Save the updated catalog to a new file
    merged_df.to_csv(output_catalog_file, index=False)
    print(f"Updated catalog saved to {output_catalog_file}")

if __name__ == "__main__":
    main()






# /sdf/home/t/tiffan/repo/accelerator-surrogate/src/preprocessing/Archive_0_n241_match_filtered_total_charge_51_catalog.csv
# /sdf/home/t/tiffan/repo/accelerator-surrogate/Archive_0_n241_match.csv