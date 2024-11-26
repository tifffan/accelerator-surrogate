#!/usr/bin/env python3
# compute_emittance_histograms.py

import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
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

def plot_histograms(emittance_data, save_dir, bins, figsize):
    """
    Plots histograms for norm emittance dimensions for both initial and final states.

    Args:
        emittance_data (dict): Dictionary with emittance names as keys and numpy arrays of their values.
        save_dir (str): Directory to save the histogram images.
        bins (int): Number of bins for the histograms.
        figsize (tuple): Figure size for the histograms.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_attrs = len(emittance_data)
    num_rows = 2  # Initial and Final
    num_cols = 3  # x, y, z
    plt.figure(figsize=figsize)

    for i, (attr, data) in enumerate(emittance_data.items(), 1):
        plt.subplot(num_rows, num_cols, i)
        plt.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{attr.replace("_", " ").title()}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'norm_emittance_histograms.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Histograms saved to {save_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute and Plot Norm Emittance Histograms (x, y, z)")
    parser.add_argument('--catalog_csv', type=str, required=True,
                        help="Path to the catalog CSV file containing data sample paths.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the histogram images.")
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

    args = parser.parse_args()

#     # Setup logging
#     numeric_level = getattr(logging, args.log_level.upper(), None)
#     if not isinstance(numeric_level, int):
#         raise ValueError(f'Invalid log level: {args.log_level}')
#     logging.basicConfig(level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s")

#     logging.info("Initializing ElectronBeamDataLoaders...")
#     data_loaders = ElectronBeamDataLoaders(
#         data_catalog=args.catalog_csv,
#         statistics_file=args.statistics_file,
#         batch_size=args.batch_size,
#         n_train=args.n_train,
#         n_val=args.n_val,
#         n_test=args.n_test,
#         random_seed=args.random_seed
#     )

#     logging.info("Retrieving DataLoader for all data...")
#     all_loader = data_loaders.get_all_data_loader()

#     # Initialize empty lists to collect emittance values
#     emittance_data = {
#         'emittance_x': [],
#         'emittance_y': [],
#         'emittance_z': []
#     }

#     logging.info("Starting to compute normalized emittance for each sample...")

#     for batch_idx, (initial_state, final_state, settings) in enumerate(tqdm(all_loader, desc="Processing Samples")):
#         # initial_state: Tensor of shape [batch_size, num_particles, 6]
#         # final_state: Tensor of shape [batch_size, num_particles, 6]
#         # settings: Tensor of shape [batch_size, 6]

#         initial_state_np = initial_state.numpy()  # Shape: [batch_size, num_particles, 6]
#         batch_size = initial_state_np.shape[0]

#         for sample_idx in range(batch_size):
#             sample_initial_np = initial_state_np[sample_idx]  # Shape: [num_particles, 6]

#             for dim, pos_idx, mom_idx in zip(['x', 'y', 'z'], args.position_indices, args.momentum_indices):
#                 print("pos_idx:",  pos_idx)
#                 pos = sample_initial_np[:, pos_idx]
#                 mom = sample_initial_np[:, mom_idx]
#                 emittance = compute_normalized_emittance(pos, mom)
#                 emittance_data[f'emittance_{dim}'].append(emittance)

#     logging.info("Completed computing normalized emittance.")

#     # Convert lists to numpy arrays
#     for key in emittance_data:
#         emittance_data[key] = np.array(emittance_data[key])

#     # Plot histograms
#     logging.info("Generating histograms...")
#     plot_histograms(
#         emittance_data=emittance_data,
#         save_dir=args.output_dir,
#         bins=args.bins,
#         figsize=tuple(args.figsize)
#     )

#     logging.info("Emittance histogram plotting completed successfully.")

# if __name__ == "__main__":
#     main()


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

    # Initialize empty lists to collect emittance values
    emittance_data = {
        'Initial Emittance X': [],
        'Initial Emittance Y': [],
        'Initial Emittance Z': [],
        'Final Emittance X': [],
        'Final Emittance Y': [],
        'Final Emittance Z': []
    }

    logging.info("Starting to compute normalized emittance for each sample...")

    for batch_idx, (initial_state, final_state, settings) in enumerate(tqdm(all_loader, desc="Processing Samples")):
        # initial_state: Tensor of shape [batch_size, num_particles, 6]
        # final_state: Tensor of shape [batch_size, num_particles, 6]
        # settings: Tensor of shape [batch_size, 6]

        initial_state_np = initial_state.numpy()  # Shape: [batch_size, num_particles, 6]
        final_state_np = final_state.numpy()      # Shape: [batch_size, num_particles, 6]
        batch_size = initial_state_np.shape[0]

        for sample_idx in range(batch_size):
            sample_initial_np = initial_state_np[sample_idx]  # Shape: [num_particles, 6]
            sample_final_np = final_state_np[sample_idx]      # Shape: [num_particles, 6]

            for dim_label, pos_idx, mom_idx in zip(['x', 'y', 'z'], args.position_indices, args.momentum_indices):
                # Ensure pos_idx and mom_idx are integers
                if not isinstance(pos_idx, int) or not isinstance(mom_idx, int):
                    logging.error(f"Position index and Momentum index must be integers. Received pos_idx={pos_idx}, mom_idx={mom_idx}")
                    continue

                # Check if indices are within bounds
                if pos_idx >= sample_initial_np.shape[1] or mom_idx >= sample_initial_np.shape[1]:
                    logging.error(f"Index out of bounds for sample {batch_idx}, {sample_idx}. pos_idx={pos_idx}, mom_idx={mom_idx}")
                    continue

                # Compute emittance for initial state
                pos_initial = sample_initial_np[:, pos_idx]
                mom_initial = sample_initial_np[:, mom_idx]
                emittance_initial = compute_normalized_emittance(pos_initial, mom_initial)
                emittance_data[f'Initial Emittance {dim_label.upper()}'].append(emittance_initial)

                # Compute emittance for final state
                pos_final = sample_final_np[:, pos_idx]
                mom_final = sample_final_np[:, mom_idx]
                emittance_final = compute_normalized_emittance(pos_final, mom_final)
                emittance_data[f'Final Emittance {dim_label.upper()}'].append(emittance_final)

    logging.info("Completed computing normalized emittance.")

    # Convert lists to numpy arrays
    for key in emittance_data:
        emittance_data[key] = np.array(emittance_data[key])

    # Plot histograms
    logging.info("Generating histograms...")
    plot_histograms(
        emittance_data=emittance_data,
        save_dir=args.output_dir,
        bins=args.bins,
        figsize=tuple(args.figsize)
    )

    logging.info("Emittance histogram plotting completed successfully.")

if __name__ == "__main__":
    main()
