#!/usr/bin/env python3
# compute_emittance_histograms.py

import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Import the ElectronBeamDataLoaders class
from electron_beam_dataloaders import ElectronBeamDataLoaders

def compute_normalized_emittance(position, momentum):
    """
    Computes the normalized emittance for a given direction.

    Args:
        position (numpy.ndarray): Positions in the given direction (x, y, or z).
        momentum (numpy.ndarray): Momenta in the given direction (px, py, or pz).

    Returns:
        float: The normalized emittance.
    """
    mean_pos2 = np.mean(position ** 2)
    mean_mom2 = np.mean(momentum ** 2)
    mean_pos_mom = np.mean(position * momentum)
    emittance = np.sqrt(mean_pos2 * mean_mom2 - mean_pos_mom ** 2)
    return emittance

def plot_histograms(emittance_data, save_dir, bins, figsize, title_suffix=''):
    """
    Plots histograms for norm emittance dimensions for both initial and final states.

    Args:
        emittance_data (dict): Dictionary with emittance names as keys and numpy arrays of their values.
        save_dir (str): Directory to save the histogram images.
        bins (int): Number of bins for the histograms.
        figsize (tuple): Figure size for the histograms.
        title_suffix (str): Suffix to add to the plot title (e.g., 'Filtered').
    """
    os.makedirs(save_dir, exist_ok=True)
    num_attrs = len(emittance_data)
    num_rows = 2  # Initial and Final
    num_cols = 3  # x, y, z
    plt.figure(figsize=figsize)

    for i, (attr, data) in enumerate(emittance_data.items(), 1):
        plt.subplot(num_rows, num_cols, i)
        plt.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{attr.replace("_", " ").title()} {title_suffix}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'norm_emittance_histograms_{title_suffix.lower()}.png') if title_suffix else os.path.join(save_dir, 'norm_emittance_histograms_filtered.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Histograms saved to {save_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute and Plot Norm Emittance Histograms (Initial & Final States)")
    parser.add_argument('--catalog_csv', type=str, required=True,
                        help="Path to the catalog CSV file containing data sample paths.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the histogram images and filtered catalog CSV.")
    parser.add_argument('--statistics_file', type=str, default=None,
                        help="Path to the global statistics file for normalization (optional).")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for data loading.")
    parser.add_argument('--n_train', type=int, default=0,
                        help="Number of training samples (optional).")
    parser.add_argument('--n_val', type=int, default=0,
                        help="Number of validation samples (optional).")
    parser.add_argument('--n_test', type=int, default=0,
                        help="Number of test samples (optional).")
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for data splitting.")
    parser.add_argument('--bins', type=int, default=50,
                        help="Number of bins for the histograms.")
    parser.add_argument('--figsize', type=int, nargs=2, default=(15, 5),
                        help="Figure size for the histograms (width height).")
    parser.add_argument('--task', type=str, default='predict_n6d',
                        choices=['predict_n6d', 'predict_n4d', 'predict_n2d'],
                        help="Task identifier to determine target features.")
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Logging level.")
    # Emittance attribute configuration
    # Assuming that positions are in the first 3 features and momenta in the next 3
    # Adjust indices if your dataset has a different structure
    parser.add_argument('--position_indices', type=int, nargs=3, default=[0, 1, 2],
                        help="Indices of position coordinates in node features (e.g., x y z).")
    parser.add_argument('--momentum_indices', type=int, nargs=3, default=[3, 4, 5],
                        help="Indices of momentum components in node features (e.g., px py pz).")
    # Filtering parameters
    parser.add_argument('--max_emittance_x', type=float, required=True,
                        help="Maximum allowed initial emittance in the x-direction.")
    parser.add_argument('--max_emittance_y', type=float, required=True,
                        help="Maximum allowed initial emittance in the y-direction.")

    args = parser.parse_args()

    # Setup logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Initializing ElectronBeamDataLoaders...")
    data_loaders = ElectronBeamDataLoaders(
        data_catalog=args.catalog_csv,
        statistics_file=args.statistics_file,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        random_seed=args.random_seed
    )

    logging.info("Retrieving DataLoader for all data...")
    all_loader = data_loaders.get_all_data_loader()

    # Extract the subset indices to map to filepaths
    # Assuming ElectronBeamDataLoaders uses Subset
    # Adjust if ElectronBeamDataLoaders class structure is different
    try:
        subset_indices = data_loaders.dataset.indices
    except AttributeError:
        # If dataset is not a Subset, use all indices
        subset_indices = range(len(data_loaders.dataset))

    # Read the original catalog CSV
    original_catalog = pd.read_csv(args.catalog_csv)
    # Extract filepaths corresponding to the subset
    subset_filepaths = original_catalog.iloc[subset_indices]['filepath'].tolist()

    # Initialize empty lists to collect emittance values
    emittance_data_all = {
        'Initial Emittance X': [],
        'Initial Emittance Y': [],
        'Initial Emittance Z': [],
        'Final Emittance X': [],
        'Final Emittance Y': [],
        'Final Emittance Z': []
    }

    # Initialize a list to collect filtered sample indices
    filtered_indices = []

    logging.info("Starting to compute normalized emittance for each sample...")
    for batch_idx, (initial_state, final_state, settings) in enumerate(tqdm(all_loader, desc="Processing Samples")):
        # initial_state: Tensor of shape [batch_size, num_particles, 6]
        # final_state: Tensor of shape [batch_size, num_particles, 6]
        # settings: Tensor of shape [batch_size, 6]

        initial_state_np = initial_state.numpy()  # Shape: [batch_size, num_particles, 6]
        final_state_np = final_state.numpy()      # Shape: [batch_size, num_particles, 6]
        batch_size = initial_state_np.shape[0]

        for sample_idx in range(batch_size):
            # Compute the absolute index of the sample in the entire dataset
            absolute_idx = batch_idx * args.batch_size + sample_idx
            if absolute_idx >= len(subset_filepaths):
                # In case of overflows
                continue
            filepath = subset_filepaths[absolute_idx]

            sample_initial_np = initial_state_np[sample_idx]  # Shape: [num_particles, 6]
            sample_final_np = final_state_np[sample_idx]      # Shape: [num_particles, 6]

            # Compute final emittance for x and y
            emittance_final_x = compute_normalized_emittance(
                sample_final_np[:, args.position_indices[0]],
                sample_final_np[:, args.momentum_indices[0]]
            )
            emittance_final_y = compute_normalized_emittance(
                sample_final_np[:, args.position_indices[1]],
                sample_final_np[:, args.momentum_indices[1]]
            )

            # Apply filters on final emittance x and y
            if (emittance_final_x <= args.max_emittance_x) and (emittance_final_y <= args.max_emittance_y):
                # If sample passes the filters, store emittance values

                # Initial Emittance
                emittance_initial_x = compute_normalized_emittance(
                    sample_initial_np[:, args.position_indices[0]],
                    sample_initial_np[:, args.momentum_indices[0]]
                )
                emittance_initial_y = compute_normalized_emittance(
                    sample_initial_np[:, args.position_indices[1]],
                    sample_initial_np[:, args.momentum_indices[1]]
                )
                emittance_initial_z = compute_normalized_emittance(
                    sample_initial_np[:, args.position_indices[2]],
                    sample_initial_np[:, args.momentum_indices[2]]
                )

                # Final Emittance
                emittance_final_z = compute_normalized_emittance(
                    sample_final_np[:, args.position_indices[2]],
                    sample_final_np[:, args.momentum_indices[2]]
                )

                # Append to emittance_data_all
                emittance_data_all['Initial Emittance X'].append(emittance_initial_x)
                emittance_data_all['Initial Emittance Y'].append(emittance_initial_y)
                emittance_data_all['Initial Emittance Z'].append(emittance_initial_z)
                emittance_data_all['Final Emittance X'].append(emittance_final_x)
                emittance_data_all['Final Emittance Y'].append(emittance_final_y)
                emittance_data_all['Final Emittance Z'].append(emittance_final_z)

                # Record the index of the filtered sample
                filtered_indices.append(absolute_idx)

    logging.info("Completed computing normalized emittance.")

    # Create a DataFrame for filtered samples
    filtered_filepaths = [subset_filepaths[idx] for idx in filtered_indices]
    filtered_catalog = pd.DataFrame({'filepath': filtered_filepaths})

    # Generate the new catalog CSV filename
    original_catalog_filename = os.path.splitext(os.path.basename(args.catalog_csv))[0]
    filtered_catalog_filename = f"{original_catalog_filename}_filter.csv"
    filtered_catalog_path = os.path.join(args.output_dir, filtered_catalog_filename)

    # Save the filtered catalog CSV
    filtered_catalog.to_csv(filtered_catalog_path, index=False)
    logging.info(f"Filtered catalog CSV saved to {filtered_catalog_path}")

    # Convert lists to numpy arrays for plotting
    for key in emittance_data_all:
        emittance_data_all[key] = np.array(emittance_data_all[key])

    # Plot histograms for the filtered dataset
    logging.info("Generating histograms for the filtered dataset...")
    plot_histograms(
        emittance_data=emittance_data_all,
        save_dir=args.output_dir,
        bins=args.bins,
        figsize=tuple(args.figsize),
        title_suffix='Filtered'
    )

    logging.info("Emittance histogram plotting for filtered data completed successfully.")

if __name__ == "__main__":
    main()
