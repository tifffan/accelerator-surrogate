import pandas as pd
import torch
import h5py
import numpy as np
import os

def compute_global_statistics(data_catalog, output_file='global_statistics.txt', epsilon=1e-6):
    # Load data catalog
    df = pd.read_csv(data_catalog)
    
    # Initialize accumulators for sum and sum of squares with higher precision
    sum_channels = torch.zeros(18, dtype=torch.float64)
    sum_squares_channels = torch.zeros(18, dtype=torch.float64)
    total_count = 0

    # Loop over all .h5 files
    for i, row in df.iterrows():
        filepath = row['filepath']
        with h5py.File(filepath, 'r') as f:
            # Load initial state (6 channels)
            initial_state = torch.stack([
                torch.tensor(f['initial_position_x'][()], dtype=torch.float64),
                torch.tensor(f['initial_position_y'][()], dtype=torch.float64),
                torch.tensor(f['initial_position_z'][()], dtype=torch.float64),
                torch.tensor(f['initial_momentum_px'][()], dtype=torch.float64),
                torch.tensor(f['initial_momentum_py'][()], dtype=torch.float64),
                torch.tensor(f['initial_momentum_pz'][()], dtype=torch.float64)
            ], dim=1)  # Shape: (num_particles, 6)
            
            # Load final state (6 channels)
            final_state = torch.stack([
                torch.tensor(f['pr10241_position_x'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_position_y'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_position_z'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_momentum_px'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_momentum_py'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_momentum_pz'][()], dtype=torch.float64)
            ], dim=1)  # Shape: (num_particles, 6)
            
            # Load settings (5 channels)
            settings_keys = [
                'CQ10121_b1_gradient', 
                'GUNF_rf_field_scale', 
                'GUNF_theta0_deg', 
                'SOL10111_solenoid_field_scale', 
                'SQ10122_b1_gradient',
                'distgen_total_charge',
            ]
            try:
                settings = torch.tensor([f[key][()] for key in settings_keys], dtype=torch.float64)  # Shape: (6,)
            except KeyError as e:
                print("Missing key {} in file {}. Skipping this file.".format(e, filepath))
                continue  # Skip files with missing keys
            
            # Check if settings are scalars
            if settings.ndimension() != 1 or settings.shape[0] != 6:
                print("Settings have unexpected shape {} in file {}. Skipping this file.".format(settings.shape, filepath))
                continue  # Skip files with unexpected settings shape

            # Concatenate initial, final, and settings into a single tensor of shape (num_particles, 17)
            num_particles = initial_state.shape[0]
            if num_particles == 0:
                print("No particles found in file {}. Skipping this file.".format(filepath))
                continue  # Skip files with no particles

            # Expand settings to match the number of particles
            settings_expanded = settings.unsqueeze(0).expand(num_particles, -1)
            full_data = torch.cat([initial_state, final_state, settings_expanded], dim=1)  # Shape: (num_particles, 17)

            # Accumulate sum and sum of squares for mean/std computation
            sum_channels += full_data.sum(dim=0)
            sum_squares_channels += (full_data ** 2).sum(dim=0)
            total_count += num_particles

            # Optional: Print progress every 100 files
            if (i + 1) % 100 == 0:
                print("Processed {}/{} files.".format(i + 1, len(df)))

    # Check if any data was processed
    if total_count == 0:
        print("No data processed. Exiting.")
        return

    # Compute global mean and variance
    global_mean = sum_channels / total_count
    global_var = (sum_squares_channels / total_count) - (global_mean ** 2)

    # Identify channels with negative variance
    negative_var_indices = torch.nonzero(global_var < 0).squeeze().tolist()
    if isinstance(negative_var_indices, int):
        negative_var_indices = [negative_var_indices]
    if negative_var_indices:
        print("Channels with negative variance detected:")
        for idx in negative_var_indices:
            print("  Channel {}:".format(idx))
            print("    Sum: {}".format(sum_channels[idx].item()))
            print("    Sum of squares: {}".format(sum_squares_channels[idx].item()))
            print("    Mean: {}".format(global_mean[idx].item()))
            print("    Variance: {}".format(global_var[idx].item()))
            print("    Setting variance to epsilon squared to avoid NaN in std.")
            global_var[idx] = torch.tensor(epsilon ** 2, dtype=torch.float64)
    
    # Compute global standard deviation
    global_std = torch.sqrt(global_var)

    # Save the results to a file using scientific notation for better precision
    with open(output_file, 'w') as f:
        f.write("Global Mean:\n")
        f.write(", ".join([f"{m.item():.12e}" for m in global_mean]) + "\n")
        f.write("Global Std:\n")
        f.write(", ".join([f"{s.item():.12e}" for s in global_std]) + "\n")

    print("Global statistics saved to {}".format(output_file))

# Usage example:
if __name__ == "__main__":
    data_catalog = '/global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog.csv'
    compute_global_statistics(
        data_catalog, 
        output_file='/global/homes/t/tiffan/slac-point/data/catalogs/global_statistics_filtered_total_charge_51.txt',
        epsilon=1e-6  # Assigning epsilon value here
    )
