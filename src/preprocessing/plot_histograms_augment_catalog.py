import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from impact import Impact

# Helper function to safely check if a value is NaN
def is_nan(value):
    return isinstance(value, (float, int, np.float64, np.int64)) and np.isnan(value)

# Function to extract data from a .h5 file using the Impact class
def extract_data_from_h5(file, default_total_charge=None):
    try:
        # Load the HDF5 file using Impact
        I = Impact.from_archive(file)
        
        # Extract norm_emit values directly from the Impact object
        norm_emit_x = I['output']['stats']['norm_emit_x'][-1] if 'norm_emit_x' in I['output']['stats'] else float('NaN')
        norm_emit_y = I['output']['stats']['norm_emit_y'][-1] if 'norm_emit_y' in I['output']['stats'] else float('NaN')
        norm_emit_z = I['output']['stats']['norm_emit_z'][-1] if 'norm_emit_z' in I['output']['stats'] else float('NaN')

        # Extract settings directly with try-except blocks
        settings = {}
        try:
            settings['SOL10111:solenoid_field_scale'] = I['SOL10111:solenoid_field_scale']
        except KeyError:
            settings['SOL10111:solenoid_field_scale'] = float('NaN')

        try:
            settings['CQ10121:b1_gradient'] = I['CQ10121:b1_gradient']
        except KeyError:
            settings['CQ10121:b1_gradient'] = float('NaN')

        try:
            settings['SQ10122:b1_gradient'] = I['SQ10122:b1_gradient']
        except KeyError:
            settings['SQ10122:b1_gradient'] = float('NaN')

        try:
            settings['GUNF:rf_field_scale'] = I['GUNF:rf_field_scale']
        except KeyError:
            settings['GUNF:rf_field_scale'] = float('NaN')

        try:
            settings['GUNF:theta0_deg'] = I['GUNF:theta0_deg']
        except KeyError:
            settings['GUNF:theta0_deg'] = float('NaN')

        try:
            settings['distgen:total_charge'] = I['distgen:total_charge']
                       
        except KeyError:
            if settings['distgen:total_charge'] is None:
                if default_total_charge is None:
                    settings['distgen:total_charge'] = np.sum(I['initial_particles']['weight'])
                else:
                    settings['distgen:total_charge'] = default_total_charge

        return norm_emit_x, norm_emit_y, norm_emit_z, settings
    
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        return float('NaN'), float('NaN'), float('NaN'), {}

# Function to plot histograms and create a cleaned catalog CSV file
def plot_histograms_and_clean_catalog(catalog_file, output_dir, cleaned_catalog_file, default_total_charge):
    # Load the catalog CSV
    df = pd.read_csv(catalog_file)

    # Prepare lists to store all extracted values for plotting
    norm_emit_x_list = []
    norm_emit_y_list = []
    norm_emit_z_list = []
    solenoid_field_scale_list = []
    b1_gradient_cq10121_list = []
    b1_gradient_sq10122_list = []
    rf_field_scale_gunf_list = []
    theta0_deg_gunf_list = []
    total_charge_list = []

    # Prepare a list to store rows with valid data
    valid_rows = []

    # Loop through each file in the catalog
    for index, row in df.iterrows():
        file = row['filepath']
        norm_emit_x, norm_emit_y, norm_emit_z, settings = extract_data_from_h5(file, default_total_charge)

        # Print a message every 10 files processed
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1} files")

        # Updated condition to check for NaN values using the is_nan function
        if not (is_nan(norm_emit_x) or is_nan(norm_emit_y) or is_nan(norm_emit_z) or
                any(is_nan(value) for value in settings.values())):
            # Append extracted values to the respective lists for plotting
            norm_emit_x_list.append(norm_emit_x)
            norm_emit_y_list.append(norm_emit_y)
            norm_emit_z_list.append(norm_emit_z)
            solenoid_field_scale_list.append(settings['SOL10111:solenoid_field_scale'])
            b1_gradient_cq10121_list.append(settings['CQ10121:b1_gradient'])
            b1_gradient_sq10122_list.append(settings['SQ10122:b1_gradient'])
            rf_field_scale_gunf_list.append(settings['GUNF:rf_field_scale'])
            theta0_deg_gunf_list.append(settings['GUNF:theta0_deg'])
            total_charge_list.append(settings['distgen:total_charge'])

            # Add the valid row data to the list
            valid_row = {
                'filename': file,
                'norm_emit_x': norm_emit_x,
                'norm_emit_y': norm_emit_y,
                'norm_emit_z': norm_emit_z,
                'SOL10111:solenoid_field_scale': settings['SOL10111:solenoid_field_scale'],
                'CQ10121:b1_gradient': settings['CQ10121:b1_gradient'],
                'SQ10122:b1_gradient': settings['SQ10122:b1_gradient'],
                'GUNF:rf_field_scale': settings['GUNF:rf_field_scale'],
                'GUNF:theta0_deg': settings['GUNF:theta0_deg'],
                'distgen:total_charge': settings['distgen:total_charge']
            }
            valid_rows.append(valid_row)

    # Create DataFrame for the valid rows and save to CSV
    df_cleaned = pd.DataFrame(valid_rows)
    df_cleaned.to_csv(cleaned_catalog_file, index=False)
    print(f"Cleaned catalog saved to {cleaned_catalog_file}")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Function to plot and save histogram for a given dataset
    def plot_histogram(data, title, filename, color='b'):
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=50, color=color, alpha=0.7)
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Histogram saved to {os.path.join(output_dir, filename)}")

    # Plot individual histograms for each quantity
    plot_histogram(norm_emit_x_list, 'norm_emit_x', 'norm_emit_x_histogram.png', 'b')
    plot_histogram(norm_emit_y_list, 'norm_emit_y', 'norm_emit_y_histogram.png', 'g')
    plot_histogram(norm_emit_z_list, 'norm_emit_z', 'norm_emit_z_histogram.png', 'r')
    plot_histogram(solenoid_field_scale_list, 'SOL10111:solenoid_field_scale', 'solenoid_field_scale_histogram.png', 'c')
    plot_histogram(b1_gradient_cq10121_list, 'CQ10121:b1_gradient', 'b1_gradient_cq10121_histogram.png', 'm')
    plot_histogram(b1_gradient_sq10122_list, 'SQ10122:b1_gradient', 'b1_gradient_sq10122_histogram.png', 'y')
    plot_histogram(rf_field_scale_gunf_list, 'GUNF:rf_field_scale', 'rf_field_scale_gunf_histogram.png', 'orange')
    plot_histogram(theta0_deg_gunf_list, 'GUNF:theta0_deg', 'theta0_deg_gunf_histogram.png', 'purple')
    plot_histogram(total_charge_list, 'distgen:total_charge', 'total_charge_histogram.png', 'brown')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process catalog CSV file, plot histograms, and clean catalog.")
    parser.add_argument('--catalog_file', type=str, help="Path to the catalog CSV file")
    parser.add_argument('--output_dir', type=str, help="Directory where the plot will be saved")
    parser.add_argument('--cleaned_catalog_file', type=str, help="Output file for the cleaned catalog CSV")
    parser.add_argument('--default_total_charge', type=float, default=None, help="Default value for distgen:total_charge")

    args = parser.parse_args()

    # Plot histograms and create cleaned catalog using the provided catalog file, output directory, and cleaned catalog file
    plot_histograms_and_clean_catalog(args.catalog_file, args.output_dir, args.cleaned_catalog_file, args.default_total_charge)
