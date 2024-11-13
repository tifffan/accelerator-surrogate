# plot_histograms_from_catalog.py

import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os

# Function to extract data from a .h5 file
def extract_data_from_h5(file):
    try:
        with h5py.File(file, 'r') as f:
            # Extract the relevant data
            norm_emit_x = np.array(f['output']['stats']['norm_emit_x'])[-1] if 'output' in f and 'stats' in f['output'] and 'norm_emit_x' in f['output']['stats'] else float('NaN')
            norm_emit_y = np.array(f['output']['stats']['norm_emit_y'])[-1] if 'output' in f and 'stats' in f['output'] and 'norm_emit_y' in f['output']['stats'] else float('NaN')
            norm_emit_z = np.array(f['output']['stats']['norm_emit_z'])[-1] if 'output' in f and 'stats' in f['output'] and 'norm_emit_z' in f['output']['stats'] else float('NaN')

            # Extract the settings from the file (assuming these settings are in the 'settings' group)
            settings = {
                'solenoid_field_scale': f['settings']['SOL10111:solenoid_field_scale'][()] if 'settings' in f and 'SOL10111:solenoid_field_scale' in f['settings'] else float('NaN'),
                'b1_gradient_cq10121': f['settings']['CQ10121:b1_gradient'][()] if 'settings' in f and 'CQ10121:b1_gradient' in f['settings'] else float('NaN'),
                'b1_gradient_sq10122': f['settings']['SQ10122:b1_gradient'][()] if 'settings' in f and 'SQ10122:b1_gradient' in f['settings'] else float('NaN'),
                'rf_field_scale_gunf': f['settings']['GUNF:rf_field_scale'][()] if 'settings' in f and 'GUNF:rf_field_scale' in f['settings'] else float('NaN'),
                'theta0_deg_gunf': f['settings']['GUNF:theta0_deg'][()] if 'settings' in f and 'GUNF:theta0_deg' in f['settings'] else float('NaN'),
                'total_charge': f['settings']['distgen:total_charge'][()] if 'settings' in f and 'distgen:total_charge' in f['settings'] else float('NaN')
            }
            return norm_emit_x, norm_emit_y, norm_emit_z, settings
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        return float('NaN'), float('NaN'), float('NaN'), {}


# Function to plot histograms for the data
def plot_histograms(catalog_file, output_dir):
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

    # Loop through each file in the catalog
    for index, row in df.iterrows():
        file = row['filename']
        norm_emit_x, norm_emit_y, norm_emit_z, settings = extract_data_from_h5(file)

        # Append extracted values to the respective lists
        norm_emit_x_list.append(norm_emit_x)
        norm_emit_y_list.append(norm_emit_y)
        norm_emit_z_list.append(norm_emit_z)
        solenoid_field_scale_list.append(settings['solenoid_field_scale'])
        b1_gradient_cq10121_list.append(settings['b1_gradient_cq10121'])
        b1_gradient_sq10122_list.append(settings['b1_gradient_sq10122'])
        rf_field_scale_gunf_list.append(settings['rf_field_scale_gunf'])
        theta0_deg_gunf_list.append(settings['theta0_deg_gunf'])
        total_charge_list.append(settings['total_charge'])

    # Create subplots for histograms
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Plot histograms for norm_emit_x, norm_emit_y, and norm_emit_z
    axs[0, 0].hist(norm_emit_x_list, bins=50, color='b', alpha=0.7)
    axs[0, 0].set_title('norm_emit_x')
    axs[0, 1].hist(norm_emit_y_list, bins=50, color='g', alpha=0.7)
    axs[0, 1].set_title('norm_emit_y')
    axs[0, 2].hist(norm_emit_z_list, bins=50, color='r', alpha=0.7)
    axs[0, 2].set_title('norm_emit_z')

    # Plot histograms for the six settings
    axs[1, 0].hist(solenoid_field_scale_list, bins=50, color='c', alpha=0.7)
    axs[1, 0].set_title('solenoid_field_scale')
    axs[1, 1].hist(b1_gradient_cq10121_list, bins=50, color='m', alpha=0.7)
    axs[1, 1].set_title('b1_gradient_cq10121')
    axs[1, 2].hist(b1_gradient_sq10122_list, bins=50, color='y', alpha=0.7)
    axs[1, 2].set_title('b1_gradient_sq10122')

    # Adjust the layout
    plt.tight_layout()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot to the output directory
    output_path = os.path.join(output_dir, "histograms.png")
    plt.savefig(output_path)
    print(f"Histogram saved to {output_path}")
    plt.close(fig)  # Close the plot after saving


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process catalog CSV file and plot histograms.")
    parser.add_argument('catalog_file', type=str, help="Path to the catalog CSV file")
    parser.add_argument('output_dir', type=str, help="Directory where the plot will be saved")
    args = parser.parse_args()

    # Plot histograms using the provided catalog file and output directory
    plot_histograms(args.catalog_file, args.output_dir)
