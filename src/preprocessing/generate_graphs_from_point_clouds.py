#!/usr/bin/env python3

import argparse
import json
import pandas as pd
import torch
import h5py
import os
import logging
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import dense_to_sparse
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_global_statistics(statistics_file):
    """Load global mean and standard deviation from a file."""
    with open(statistics_file, 'r') as f:
        lines = f.readlines()
        # Assuming the file format is:
        # Line 1: 'Global Mean'
        # Line 2: comma-separated mean values
        # Line 3: 'Global Std'
        # Line 4: comma-separated std values

        # Extract and parse Global Mean
        mean_line = lines[1].strip()
        mean_values = [float(x) for x in mean_line.split(',')]
        global_mean = torch.tensor(mean_values, dtype=torch.float32)

        # Extract and parse Global Std
        std_line = lines[3].strip()
        std_values = [float(x) for x in std_line.split(',')]
        global_std = torch.tensor(std_values, dtype=torch.float32)

    return global_mean, global_std


def process_data_catalog(
    data_catalog,
    output_base_dir,
    k=5,
    distance_threshold=float('inf'),
    edge_method='knn',
    weighted_edge=False,
    global_mean=None,
    global_std=None,
    subsample_size=128,
):
    """Process the data catalog and save the graph data to files."""
    data = pd.read_csv(data_catalog)

    # Subsample the data to only process the first 'subsample_size' rows
    data = data.head(subsample_size)

    # Generate output directories for initial state, final state, and settings
    graph_data_dir_initial = generate_graph_data_dir(
        keyword="initial",
        edge_method=edge_method,
        weighted_edge=weighted_edge,
        k=k,
        distance_threshold=distance_threshold,
        base_dir=output_base_dir,
    )

    graph_data_dir_final = generate_graph_data_dir(
        keyword="final",
        edge_method=edge_method,
        weighted_edge=weighted_edge,
        k=k,
        distance_threshold=distance_threshold,
        base_dir=output_base_dir,
    )

    settings_data_dir = generate_graph_data_dir(
        keyword="settings",
        edge_method=edge_method,
        weighted_edge=weighted_edge,
        k=k,
        distance_threshold=distance_threshold,
        base_dir=output_base_dir,
    )

    # Ensure the output directories exist
    Path(graph_data_dir_initial).mkdir(parents=True, exist_ok=True)
    Path(graph_data_dir_final).mkdir(parents=True, exist_ok=True)
    Path(settings_data_dir).mkdir(parents=True, exist_ok=True)

    # Split the global mean and std into initial, final, and settings
    # Assuming the global_mean and global_std are concatenated as follows:
    # [initial_state (6 features), final_state (6 features), settings (5 features)]
    global_mean_initial = global_mean[:6]
    global_std_initial = global_std[:6]

    global_mean_final = global_mean[6:12]
    global_std_final = global_std[6:12]

    global_mean_settings = global_mean[12:]
    global_std_settings = global_std[12:]

    epsilon = 1e-6  # Small constant to prevent division by zero

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing data catalog"):
        filepath = row['filepath']

        # Use the row index from the CSV as the unique identifier
        unique_id = idx

        # Load the data
        with h5py.File(filepath, 'r') as f:
            # Load initial state
            initial_state = _load_initial_state(f)
            # Load final state
            final_state = _load_final_state(f)
            # Load settings
            settings = _load_settings(f)

        # Apply normalization to initial state
        if global_mean is not None and global_std is not None:
            # Normalize initial_state (shape: (6, num_particles))
            initial_state = (initial_state - global_mean_initial[:, None]) / (global_std_initial[:, None] + epsilon)

            # Normalize final_state (shape: (6, num_particles))
            final_state = (final_state - global_mean_final[:, None]) / (global_std_final[:, None] + epsilon)

            # Normalize settings (shape: (5,))
            settings = (settings - global_mean_settings) / (global_std_settings + epsilon)

        # Create the graph from the point cloud for initial state
        if edge_method == 'dist':
            edge_index_initial, edge_weight_initial = _build_edges_by_distance(initial_state, distance_threshold, weighted_edge)
            edge_index_final, edge_weight_final = _build_edges_by_distance(final_state, distance_threshold, weighted_edge)
        else:
            edge_index_initial, edge_weight_initial = _build_edges_knn(initial_state, k, weighted_edge)
            edge_index_final, edge_weight_final = _build_edges_knn(final_state, k, weighted_edge)

        # Save the initial state graph data to a file
        graph_data_initial = Data(x=initial_state.T, edge_index=edge_index_initial, edge_weight=edge_weight_initial)
        save_path_initial = os.path.join(graph_data_dir_initial, f"graph_{unique_id}.pt")
        torch.save(graph_data_initial, save_path_initial)

        # Save the final state graph data to a file
        graph_data_final = Data(x=final_state.T, edge_index=edge_index_final, edge_weight=edge_weight_final)
        save_path_final = os.path.join(graph_data_dir_final, f"graph_{unique_id}.pt")
        torch.save(graph_data_final, save_path_final)

        # Save the normalized settings vector to a file
        save_path_settings = os.path.join(settings_data_dir, f"settings_{unique_id}.pt")
        torch.save(settings, save_path_settings)

    # Save the metadata for initial state
    metadata_initial = {
        'edge_method': edge_method,
        'weighted_edge': weighted_edge,
        'k': k,
        'distance_threshold': distance_threshold,
        'global_mean': global_mean_initial.tolist(),
        'global_std': global_std_initial.tolist(),
    }
    metadata_path_initial = os.path.join(graph_data_dir_initial, 'metadata.json')
    with open(metadata_path_initial, 'w') as f:
        json.dump(metadata_initial, f, indent=4)

    # Save the metadata for final state
    metadata_final = {
        'edge_method': edge_method,
        'weighted_edge': weighted_edge,
        'k': k,
        'distance_threshold': distance_threshold,
        'global_mean': global_mean_final.tolist(),
        'global_std': global_std_final.tolist(),
    }
    metadata_path_final = os.path.join(graph_data_dir_final, 'metadata.json')
    with open(metadata_path_final, 'w') as f:
        json.dump(metadata_final, f, indent=4)

    # Save the metadata for settings
    settings_keys = [
        'CQ10121_b1_gradient',
        'GUNF_rf_field_scale',
        'GUNF_theta0_deg',
        'SOL10111_solenoid_field_scale',
        'SQ10122_b1_gradient',
        'distgen_total_charge',
    ]
    metadata_settings = {
        'settings_keys': settings_keys,
        'global_mean': global_mean_settings.tolist(),
        'global_std': global_std_settings.tolist(),
    }
    metadata_path_settings = os.path.join(settings_data_dir, 'metadata.json')
    with open(metadata_path_settings, 'w') as f:
        json.dump(metadata_settings, f, indent=4)


def _load_initial_state(file):
    """Load the initial state data from the HDF5 file."""
    initial_position_x = torch.tensor(file['initial_position_x'][()]).float()
    initial_position_y = torch.tensor(file['initial_position_y'][()]).float()
    initial_position_z = torch.tensor(file['initial_position_z'][()]).float()
    initial_momentum_px = torch.tensor(file['initial_momentum_px'][()]).float()
    initial_momentum_py = torch.tensor(file['initial_momentum_py'][()]).float()
    initial_momentum_pz = torch.tensor(file['initial_momentum_pz'][()]).float()
    initial_state = torch.stack([
        initial_position_x, initial_position_y, initial_position_z,
        initial_momentum_px, initial_momentum_py, initial_momentum_pz
    ])
    return initial_state


def _load_final_state(file):
    """Load the final state data from the HDF5 file."""
    pr10241_position_x = torch.tensor(file['pr10241_position_x'][()]).float()
    pr10241_position_y = torch.tensor(file['pr10241_position_y'][()]).float()
    pr10241_position_z = torch.tensor(file['pr10241_position_z'][()]).float()
    pr10241_momentum_px = torch.tensor(file['pr10241_momentum_px'][()]).float()
    pr10241_momentum_py = torch.tensor(file['pr10241_momentum_py'][()]).float()
    pr10241_momentum_pz = torch.tensor(file['pr10241_momentum_pz'][()]).float()
    final_state = torch.stack([
        pr10241_position_x, pr10241_position_y, pr10241_position_z,
        pr10241_momentum_px, pr10241_momentum_py, pr10241_momentum_pz
    ])
    return final_state


def _load_settings(file):
    """Load the settings data from the HDF5 file."""
    settings_keys = [
        'CQ10121_b1_gradient',
        'GUNF_rf_field_scale',
        'GUNF_theta0_deg',
        'SOL10111_solenoid_field_scale',
        'SQ10122_b1_gradient',
        'distgen_total_charge',
    ]
    settings_values = []
    for key in settings_keys:
        if key in file:
            settings_values.append(file[key][()])
        else:
            settings_values.append(0.0)  # Assuming default value as 0.0 if key not found
    settings_tensor = torch.tensor(settings_values, dtype=torch.float32)
    return settings_tensor


def _build_edges_knn(node_features, k, weighted_edge):
    """Build k-NN graph edges and weight by applying Gaussian kernel on Euclidean distance."""
    # Create k-NN graph
    edge_index = knn_graph(node_features.T, k=k)

    if weighted_edge:
        # Calculate Euclidean distances between connected nodes
        position_coords = node_features[:3, :].T  # Use only the x, y, z coordinates
        dist_matrix = torch.cdist(position_coords, position_coords, p=2)  # Pairwise distances

        # Extract distances for the selected edges and apply Gaussian kernel
        edge_distances = dist_matrix[edge_index[0], edge_index[1]]
        edge_weight = gaussian_kernel(edge_distances)
    else:
        edge_weight = None  # No edge weights if weighted_edge is False

    return edge_index, edge_weight


def _build_edges_by_distance(node_features, distance_threshold, weighted_edge):
    """Build graph edges based on Euclidean distance and weight by applying Gaussian kernel."""
    position_coords = node_features[:3, :].T  # Use only the x, y, z coordinates
    dist_matrix = torch.cdist(position_coords, position_coords, p=2)  # Compute pairwise distances

    # Create adjacency matrix (0 if distance > threshold, 1 otherwise)
    adj_matrix = (dist_matrix < distance_threshold).float()
    edge_index, _ = dense_to_sparse(adj_matrix)

    if weighted_edge:
        # Apply Gaussian kernel to the distances for the edges
        edge_distances = dist_matrix[edge_index[0], edge_index[1]]
        edge_weight = gaussian_kernel(edge_distances)
    else:
        # Binary edge weights (1 for connected edges)
        edge_weight = torch.ones(edge_index.size(1))

    return edge_index, edge_weight


def gaussian_kernel(distances, sigma=1.0):
    """Gaussian kernel function for weighting edges."""
    return torch.exp(-distances ** 2 / (2 * sigma ** 2))


def generate_graph_data_dir(keyword, edge_method, weighted_edge, k=None, distance_threshold=None, base_dir=None):
    """Helper function to generate graph_data_dir based on arguments."""
    weighted_str = "weighted" if weighted_edge else "unweighted"

    if base_dir is None:
        base_dir = "./graph_data"

    if edge_method == 'knn':
        dir_name = "{}_{}_k{}_{}_graphs".format(keyword, edge_method, k, weighted_str)
    elif edge_method == 'dist':
        if distance_threshold is not None and not np.isinf(distance_threshold):
            dir_name = "{}_{}_d{}_{}_graphs".format(keyword, edge_method, distance_threshold, weighted_str)
        else:
            dir_name = "{}_{}_{}_graphs".format(keyword, edge_method, weighted_str)
    else:
        dir_name = "{}_{}_{}_graphs".format(keyword, edge_method, weighted_str)

    return os.path.join(base_dir, dir_name)


def parse_args():
    """Argument parser for command-line inputs."""
    parser = argparse.ArgumentParser(description="Process data catalog and generate graph data")
    parser.add_argument('--data_catalog', type=str, required=True, help="Path to the data catalog CSV (required)")
    parser.add_argument('--statistics_file', type=str, required=True, help="Path to the global statistics file (required)")
    parser.add_argument('--k', type=int, default=5, help="Number of nearest neighbors for graph construction")
    parser.add_argument('--distance_threshold', type=float, default=float('inf'),
                        help="Euclidean distance threshold for connecting nodes (if not using k-nearest neighbors)")
    parser.add_argument('--edge_method', type=str, default='knn', choices=['knn', 'dist'],
                        help="Edge construction method: 'knn' or 'dist'")
    parser.add_argument('--weighted_edge', action='store_true',
                        help="Use Gaussian kernel on edge weights instead of binary edges")
    parser.add_argument('--output_base_dir', type=str, default="./graph_data/",
                        help="Base directory to save the processed graph data")
    parser.add_argument('--subsample_size', type=int, default=128,
                        help="Number of data catalog entries to process (default: 128)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Ensure the output base directory exists
    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)

    # Load global mean and std from the statistics file
    global_mean, global_std = load_global_statistics(args.statistics_file)
    logging.info(f"Global mean: {global_mean}")
    logging.info(f"Global std: {global_std}")

    # Process the data catalog and save the graph data to files
    process_data_catalog(
        data_catalog=args.data_catalog,
        output_base_dir=args.output_base_dir,
        k=args.k,
        distance_threshold=args.distance_threshold,
        edge_method=args.edge_method,
        weighted_edge=args.weighted_edge,
        global_mean=global_mean,
        global_std=global_std,
        subsample_size=args.subsample_size,
    )

    # Print the generated graph_data_dirs at the end of processing
    print(f"Initial state graph data saved to: {generate_graph_data_dir('initial', args.edge_method, args.weighted_edge, args.k, args.distance_threshold, args.output_base_dir)}")
    print(f"Final state graph data saved to: {generate_graph_data_dir('final', args.edge_method, args.weighted_edge, args.k, args.distance_threshold, args.output_base_dir)}")
    print(f"Settings data saved to: {generate_graph_data_dir('settings', args.edge_method, args.weighted_edge, args.k, args.distance_threshold, args.output_base_dir)}")
