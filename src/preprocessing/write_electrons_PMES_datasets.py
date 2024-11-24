import csv
import os
import h5py
import numpy as np
from impact import Impact
from pmd_beamphysics import ParticleGroup


def process_raw_h5(input_file, output_file, settings, catalog_entry, compare_counter):
    """
    Process the raw .h5 file to extract specified quantities and store them in a new .h5 file.

    Parameters:
    input_file (str): Path to the input .h5 file.
    output_file (str): Path to the output .h5 file.
    settings (list): List of setting variables to extract.
    catalog_entry (dict): Catalog data for the current file.
    compare_counter (dict): Counter for comparing total charge values.
    """
    try:
        # Check file size
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)  # Size in MB
        if file_size_mb > 8:
            print(f"Skipping {input_file}: File size exceeds 8MB limit.")
            return

        # Load the archive using the Impact library
        I = Impact()
        I.load_archive(input_file)

        # Extract required data
        initial_x = I['particles']['initial_particles']['x']
        initial_y = I['particles']['initial_particles']['y']
        initial_z = I['particles']['initial_particles']['z']
        initial_px = I['particles']['initial_particles']['px']
        initial_py = I['particles']['initial_particles']['py']
        initial_pz = I['particles']['initial_particles']['pz']

        pr10241_x = I['particles']['PR10241']['x']
        pr10241_y = I['particles']['PR10241']['y']
        pr10241_z = I['particles']['PR10241']['z']
        pr10241_px = I['particles']['PR10241']['px']
        pr10241_py = I['particles']['PR10241']['py']
        pr10241_pz = I['particles']['PR10241']['pz']

        if check_empty_initial_position_z(input_file):
            initial_z = get_initial_z(input_file)

        norm_emit_x = I['output']['stats']['norm_emit_x'][-1]
        norm_emit_y = I['output']['stats']['norm_emit_y'][-1]
        norm_emit_z = I['output']['stats']['norm_emit_z'][-1]

        with h5py.File(output_file, 'a') as out_f:
            out_f.create_dataset('initial_position_x', data=initial_x)
            out_f.create_dataset('initial_position_y', data=initial_y)
            out_f.create_dataset('initial_position_z', data=initial_z)
            out_f.create_dataset('initial_momentum_px', data=initial_px)
            out_f.create_dataset('initial_momentum_py', data=initial_py)
            out_f.create_dataset('initial_momentum_pz', data=initial_pz)

            out_f.create_dataset('pr10241_position_x', data=pr10241_x)
            out_f.create_dataset('pr10241_position_y', data=pr10241_y)
            out_f.create_dataset('pr10241_position_z', data=pr10241_z)
            out_f.create_dataset('pr10241_momentum_px', data=pr10241_px)
            out_f.create_dataset('pr10241_momentum_py', data=pr10241_py)
            out_f.create_dataset('pr10241_momentum_pz', data=pr10241_pz)

            out_f.create_dataset('norm_emit_x', data=norm_emit_x)
            out_f.create_dataset('norm_emit_y', data=norm_emit_y)
            out_f.create_dataset('norm_emit_z', data=norm_emit_z)

            print(f"Processed: initial_position_x, initial_position_y, initial_position_z, initial_momentum_px, initial_momentum_py, initial_momentum_pz")
            print(f"Processed: pr10241_position_x, pr10241_position_y, pr10241_position_z, pr10241_momentum_px, pr10241_momentum_py, pr10241_momentum_pz")
            print(f"Processed: norm_emit_x, norm_emit_y, norm_emit_z")

            # Extract and store the settings variables
            for setting in settings:
                try:
                    if setting == 'distgen:total_charge':
                        catalog_charge_str = catalog_entry.get('distgen_total_charge')
                        if catalog_charge_str is not None:
                            try:
                                catalog_charge = float(catalog_charge_str)
                            except ValueError:
                                print(f"Invalid 'distgen_total_charge' value in catalog for {input_file}")
                                raise ValueError(f"Invalid 'distgen_total_charge' value in catalog for {input_file}")
                            # For the first 5 times, compare with computed value
                            if compare_counter['count'] < 5:
                                # Compute total_charge
                                total_charge = np.sum(I['particles']['initial_particles']['weight'])
                                if not np.isclose(catalog_charge, total_charge):
                                    print(f"Mismatch in total_charge for {input_file}. Catalog: {catalog_charge}, Computed: {total_charge}")
                                    raise ValueError(f"Mismatch in total_charge for {input_file}.")
                                compare_counter['count'] += 1
                            setting_value = catalog_charge
                        else:
                            # Compute total_charge via np.sum
                            setting_value = np.sum(I['particles']['initial_particles']['weight'])
                        out_f.create_dataset(setting.replace(':', '_'), data=setting_value)
                        print(f"Extracted setting: {setting} with value: {setting_value}")
                    else:
                        setting_value = I[setting]
                        out_f.create_dataset(setting.replace(':', '_'), data=setting_value)
                        print(f"Extracted setting: {setting} with value: {setting_value}")
                except KeyError:
                    print(f"Setting {setting} not found in {input_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        raise  # Re-raise the exception to stop the script


def check_empty_initial_position_z(input_file):
    """
    Check if the 'impact/initial_particles/position/z' group is empty in the HDF5 file.

    Parameters:
    input_file (str): Path to the HDF5 file.
    """
    # Open the HDF5 file
    with h5py.File(input_file, 'r') as file:
        # Check if the group exists
        if 'impact/initial_particles/position/z' in file:
            dataset = file['impact/initial_particles/position/z']
            # Check if dataset is empty
            if len(dataset) == 0:
                return True
        else:
            print("The group 'impact/initial_particles/position/z' does not exist in the file.")
    return False


def get_initial_z(input_file):
    try:
        a = Impact.from_archive(input_file)
        P1 = a.particles['initial_particles']
        t0 = P1.avg('t')
        P1.drift_to_t(t0)
        return P1.z
    except Exception as e:
        print(f"Error encountered while processing {input_file}: {e}")
        return None


def check_norm_emit_threshold(input_file, threshold=3.0e-5):
    """
    Check if the final value in 'impact/output/stats/norm_emit_x' or 'norm_emit_y' group is below a certain threshold in the HDF5 file.

    Parameters:
    input_file (str): Path to the HDF5 file.
    threshold (float): Threshold value for the final norm_emit_x or norm_emit_y value.

    Returns:
    bool: True if the final value is below the threshold, False otherwise.
    """
    # Open the HDF5 file
    with h5py.File(input_file, 'r') as file:
        # Check if the group exists for norm_emit_x
        if 'impact/output/stats/norm_emit_x' in file:
            dataset_x = file['impact/output/stats/norm_emit_x']
            # Check if dataset is not empty
            if len(dataset_x) > 0:
                final_value_x = dataset_x[-1]  # Get the last value
                if final_value_x >= threshold:
                    return False
            else:
                print("Dataset 'impact/output/stats/norm_emit_x' is empty.")
                return False
        else:
            print("The group 'impact/output/stats/norm_emit_x' does not exist in the file.")
            return False

        # Check if the group exists for norm_emit_y
        if 'impact/output/stats/norm_emit_y' in file:
            dataset_y = file['impact/output/stats/norm_emit_y']
            # Check if dataset is not empty
            if len(dataset_y) > 0:
                final_value_y = dataset_y[-1]  # Get the last value
                if final_value_y >= threshold:
                    return False
            else:
                print("Dataset 'impact/output/stats/norm_emit_y' is empty.")
                return False
        else:
            print("The group 'impact/output/stats/norm_emit_y' does not exist in the file.")
            return False

        return True


def process_all_h5_files(input_dir, output_dir, catalog, settings):
    """
    Process all .h5 files specified in the catalog and save them in the output directory.

    Parameters:
    input_dir (str): Path to the directory containing .h5 files.
    output_dir (str): Path to the directory where processed .h5 files will be saved.
    catalog (dict): Catalog data mapping filenames to their data.
    settings (list): List of setting variables to extract.
    """
    total_file_count = 0
    empty_z_count = 0
    valid_emit_count = 0
    compare_counter = {'count': 0}  # Initialize counter for comparisons

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over files in the catalog
        for filename, catalog_entry in catalog.items():
            # Construct the full input file path
            filename_base = os.path.basename(filename)
            input_file = os.path.join(input_dir, filename_base)
            output_file = os.path.join(output_dir, filename_base)

            if os.path.exists(input_file):
                total_file_count += 1

                if check_empty_initial_position_z(input_file):
                    empty_z_count += 1
                    if check_norm_emit_threshold(input_file):
                        valid_emit_count += 1
                        process_raw_h5(input_file, output_file, settings, catalog_entry, compare_counter)

                print(f"Processed: {filename}")
            else:
                print(f"File {input_file} does not exist.")

        print(f"Total HDF5 files processed: {total_file_count}")
        print(f"HDF5 files with empty 'impact/initial_particles/position/z': {empty_z_count}")
        print(f"HDF5 files with valid 'impact/output/stats/norm_emit_x' and 'norm_emit_y': {valid_emit_count}")

    except Exception as e:
        print(f"Error processing files: {e}")
        raise  # Re-raise the exception to stop the script


def read_data_catalog(data_catalog_directory):
    catalog_data = {}  # Initialize an empty dict

    # Open the CSV file in read mode
    with open(data_catalog_directory, 'r', newline='') as csvfile:
        # Create a CSV reader object with headers
        csvreader = csv.DictReader(csvfile)

        # Iterate over each row in the CSV file
        for row in csvreader:
            filename = row.get('filename') or row.get('file_name') or row.get('filepath') or row.get('File Name') or row.get('File') or list(row.values())[0]
            if not filename:
                continue  # Skip if no filename
            catalog_data[filename] = row  # Store the entire row

    return catalog_data

if __name__ == "__main__":

    # Specify input and output directories
    input_directory = '/sdf/data/ad/ard-online/Simulations/FACET-II_Injector_Gaussian/Archive/'
    output_directory = '/sdf/scratch/users/t/tiffan/electrons_PMES_51/'
    catalog_directory = 'Archive_0_n241_match_filtered_total_charge_51_catalog.csv'

    settings = [
        'SOL10111:solenoid_field_scale',
        'CQ10121:b1_gradient',
        'SQ10122:b1_gradient',
        'GUNF:rf_field_scale',
        'GUNF:theta0_deg',
        'distgen:total_charge'
    ]

    catalog = read_data_catalog(catalog_directory)

    # Process all .h5 files specified in the catalog
    process_all_h5_files(input_directory, output_directory, catalog, settings)
