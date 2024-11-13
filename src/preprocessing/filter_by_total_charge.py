import pandas as pd
import h5py

def filter_files_by_total_charge(data_catalog, output_csv, total_charge_range):
    """
    Filter the dataset based on the total charge range and create a new CSV file.

    Parameters:
        data_catalog (str): Path to the original data catalog CSV file.
        output_csv (str): Path to save the new filtered data catalog CSV file.
        total_charge_range (tuple): A tuple (min_charge, max_charge) specifying the range of total charge.
    """
    # Load the original data catalog
    data = pd.read_csv(data_catalog)

    # List to store filtered rows
    filtered_data = []

    # Loop through each file in the catalog
    for idx, row in data.iterrows():
        filepath = row['filepath']
        
        # Open the .h5 file and read the 'distgen_total_charge' value
        with h5py.File(filepath, 'r') as f:
            if 'distgen_total_charge' in f:
                total_charge = f['distgen_total_charge'][()]
            else:
                continue  # Skip if total charge is not available

            # Check if the total charge is within the specified range
            if total_charge_range[0] <= total_charge <= total_charge_range[1]:
                filtered_data.append(row)

    # Create a new DataFrame with the filtered data
    filtered_df = pd.DataFrame(filtered_data)

    # Save the filtered DataFrame as a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data catalog saved to {output_csv}")
    print(f"Number of files in the original catalog: {len(data)}")
    print(f"Number of files in the filtered catalog: {len(filtered_df)}")

if __name__ == "__main__":
    # Define paths and total charge range
    data_catalog = '/global/homes/t/tiffan/slac-point/data/electrons_vary_distributions_vary_settings_catalog.csv'
    output_csv = '/global/homes/t/tiffan/slac-point/data/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog.csv'
    total_charge_range = (0.5e-9, 0.51e-9)  # Example range for total charge

    # Call the filtering function
    filter_files_by_total_charge(data_catalog, output_csv, total_charge_range)
