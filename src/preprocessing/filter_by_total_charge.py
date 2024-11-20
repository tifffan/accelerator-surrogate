import pandas as pd
import h5py
from impact import Impact
import numpy as np

def filter_files_by_total_charge(data_catalog, output_csv, total_charge_range):
    """
    Filter the dataset based on the total charge range and create a new CSV file.
    Append the computed `distgen_total_charge` to the input data catalog.

    Parameters:
        data_catalog (str): Path to the original data catalog CSV file.
        output_csv (str): Path to save the new filtered data catalog CSV file.
        total_charge_range (tuple): A tuple (min_charge, max_charge) specifying the range of total charge.
    """
    # Load the original data catalog
    data = pd.read_csv(data_catalog)

    # Ensure the `distgen_total_charge` column exists
    if 'distgen_total_charge' not in data.columns:
        data['distgen_total_charge'] = np.nan

    # List to store filtered rows
    filtered_data = []

    # Loop through each file in the catalog
    for idx, row in data.iterrows():
        filepath = row['filename']  # Adjust column name as needed
        
        # Check if total charge is already present
        if not pd.isna(row['distgen_total_charge']):
            total_charge = row['distgen_total_charge']
        else:
            # Open the .h5 file and compute total charge if not present
            with h5py.File(filepath, 'r') as f:
                if 'distgen_total_charge' in f:
                    total_charge = f['distgen_total_charge'][()]
                else:
                    I = Impact()
                    I.load_archive(filepath)
                    total_charge = np.sum(I['particles']['initial_particles']['weight'])
                    print(f'Computed distgen_total_charge for {filepath}')
            
            # Update the `distgen_total_charge` column in the DataFrame
            data.loc[idx, 'distgen_total_charge'] = total_charge

        # Check if the total charge is within the specified range
        if total_charge_range[0] <= total_charge <= total_charge_range[1]:
            # Add the row to the filtered data
            filtered_data.append(row)

    # Create a new DataFrame with the filtered data
    filtered_df = pd.DataFrame(filtered_data)

    # Save the filtered DataFrame as a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data catalog saved to {output_csv}")

    # Save the updated original data catalog with the appended `distgen_total_charge` column
    data.to_csv(data_catalog, index=False)
    print(f"Updated data catalog with `distgen_total_charge` saved to {data_catalog}")

    # Print summary
    print(f"Number of files in the original catalog: {len(data)}")
    print(f"Number of files in the filtered catalog: {len(filtered_df)}")


if __name__ == "__main__":
    # Define paths and total charge range
    # data_catalog = '/global/homes/t/tiffan/slac-point/data/electrons_vary_distributions_vary_settings_catalog.csv'
    data_catalog = '/sdf/data/ad/ard/u/tiffan/Archive_0_n241_match.csv'
    # output_csv = '/global/homes/t/tiffan/slac-point/data/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog.csv'
    output_csv = '/sdf/data/ad/ard/u/tiffan/Archive_0_n241_match_filtered_total_charge_51_catalog.csv'
    total_charge_range = (0.5e-9, 0.51e-9)  # Example range for total charge

    # Call the filtering function
    filter_files_by_total_charge(data_catalog, output_csv, total_charge_range)
