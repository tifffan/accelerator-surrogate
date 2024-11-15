import numpy as np
import h5py
import os
from pmd_beamphysics import ParticleGroup
import glob
from impact import Impact
import re
import torch  # Import torch for tensor operations
import argparse  # Import argparse

# Helper function to safely check if a value is NaN
def is_nan(value):
    return isinstance(value, (float, int, np.float64, np.int64)) and np.isnan(value)

# Function to extract and save particle data and settings
def extract_and_save_particle_data(h5_file_path, output_dir='data/particles', save_settings=True, settings_filename=None):
    # Open the HDF5 file using Impact
    try:
        I = Impact.from_archive(h5_file_path)
    except Exception as e:
        print(f"Error opening file {h5_file_path}: {e}")
        return

    # List all steps available in the data, sorted to ensure correct order
    unsorted_steps = list(I.particles)

    def sort_key(step):
        if step == 'initial_particles':
            return (0, 0)  # Ensure this is first
        elif step.startswith('write_beam_'):
            num = int(re.search(r'\d+', step).group())  # Extract the number part
            return (1, num)  # Sort these in ascending order after 'initial_particles'
        elif step == 'PR10241':
            return (2, 0)  # Ensure 'PR10241' is last
        return (3, 0)  # Any other steps go to the end

    # Sort the list using the custom key
    steps = sorted(unsorted_steps, key=sort_key)

    print(f"Processing file: {h5_file_path}")
    print(f"Sorted steps ({len(steps)}): {steps}")

    # Verify that the Impact object contains exactly 77 particle groups
    expected_num_steps = 77
    if len(steps) != expected_num_steps:
        print(f"Warning: File {h5_file_path} contains {len(steps)} particle groups instead of {expected_num_steps}. Skipping this file.")
        return  # Skip this file

    num_steps = len(steps)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List to store particle IDs that are common across all steps
    common_particle_ids = None

    # List to store particle data at each time step
    particle_data_list = []

    # Loop over time steps
    for idx, step_name in enumerate(steps):
        # Load particle data at the current step
        try:
            P = I.particles[step_name]
        except Exception as e:
            print(f"Error loading particles at step {step_name} in file {h5_file_path}: {e}")
            return

        # Drift particles to their average time
        t0 = P.avg('t')
        P.drift_to_t(t0)

        # Get particle IDs at the current step
        particle_ids = P.id

        # Initialize common_particle_ids with the IDs from the first step
        if common_particle_ids is None:
            common_particle_ids = particle_ids
        else:
            # Find intersection of particle IDs to ensure consistency
            common_particle_ids = np.intersect1d(common_particle_ids, particle_ids)

        # Store the particle data
        variables = ['x', 'y', 'z', 'px', 'py', 'pz']
        data = {var: getattr(P, var) for var in variables}
        data['id'] = particle_ids  # Also store IDs to align particles later
        particle_data_list.append(data)

    # Now that we have data from all steps, we can align particles using common_particle_ids
    if len(common_particle_ids) == 0:
        print(f"No common particles found across all steps in file {h5_file_path}.")
        return

    print(f"Number of common particles: {len(common_particle_ids)}")

    # Prepare tensors to store the data
    num_common_particles = len(common_particle_ids)
    num_features = 6  # x, y, z, px, py, pz
    num_time_steps = len(particle_data_list)

    # Initialize tensor to hold particle data
    particle_tensor = torch.zeros((num_time_steps, num_common_particles, num_features))

    # Map from variable names to indices in the feature dimension
    var_to_idx = {'x': 0, 'y': 1, 'z': 2, 'px': 3, 'py': 4, 'pz': 5}

    # Loop over time steps and fill the tensor
    for t, data in enumerate(particle_data_list):
        # Find indices of common particles in the current step
        ids = data['id']
        idx_in_current = np.array([np.where(ids == pid)[0][0] for pid in common_particle_ids])

        # For each variable, extract data for common particles and store in tensor
        for var in variables:
            values = data[var][idx_in_current]
            feature_idx = var_to_idx[var]
            particle_tensor[t, :, feature_idx] = torch.from_numpy(values)

    # Save the particle tensor to disk
    tensor_filename = os.path.join(output_dir, f"{os.path.basename(h5_file_path).replace('.h5', '')}_particle_data.pt")
    torch.save(particle_tensor, tensor_filename)
    print(f"Saved particle tensor to {tensor_filename}")

    # Now extract settings using the method from your provided script
    if save_settings:
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
            # Set total charge to 1e-9 if it's None
            if settings['distgen:total_charge'] is None:
                settings['distgen:total_charge'] = 1e-9
        except KeyError:
            settings['distgen:total_charge'] = 1e-9

        # Convert settings to tensors where applicable
        settings_tensor = {}
        for key, value in settings.items():
            if isinstance(value, (int, float, np.float64, np.int64)):
                if is_nan(value):
                    settings_tensor[key] = torch.tensor(float('nan'))
                else:
                    settings_tensor[key] = torch.tensor(value)
            else:
                # Handle non-numeric settings if needed
                settings_tensor[key] = value

        # Save the settings tensor to disk
        if settings_filename is None:
            settings_filename = os.path.join(output_dir, f"{os.path.basename(h5_file_path).replace('.h5', '')}_settings.pt")
        else:
            settings_filename = os.path.join(output_dir, settings_filename)

        torch.save(settings_tensor, settings_filename)
        print(f"Saved settings tensor to {settings_filename}")

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save particle data and settings from HDF5 files.")
    parser.add_argument('--archive_dir', type=str, default='/sdf/data/ad/ard/u/tiffan/Archive_4', help="Directory containing the HDF5 files.")
    parser.add_argument('--output_dir', type=str, default='/sdf/data/ad/ard/u/tiffan/data/time_series_particle_data_archive_4', help="Directory to save the output tensors.")
    parser.add_argument('--identical_settings', action='store_true', help="Flag indicating whether settings are identical across files.")

    args = parser.parse_args()

    # Search for HDF5 files in the archive directory
    h5_files = glob.glob(os.path.join(args.archive_dir, '*.h5'))

    # Process each HDF5 file
    if h5_files:
        os.makedirs(args.output_dir, exist_ok=True)

        # Variable to track if settings have been saved
        settings_saved = False

        for h5_file_path in h5_files:
            # Determine whether to save settings
            if args.identical_settings:
                if settings_saved:
                    # Do not save settings again
                    save_settings = False
                else:
                    save_settings = True
                    settings_filename = 'settings.pt'  # Save without filename prefix
                    settings_saved = True
            else:
                save_settings = True
                settings_filename = None  # Use default filename with prefix

            extract_and_save_particle_data(h5_file_path, output_dir=args.output_dir,
                                           save_settings=save_settings, settings_filename=settings_filename)
    else:
        raise FileNotFoundError(f"No HDF5 files found in directory: {args.archive_dir}")
