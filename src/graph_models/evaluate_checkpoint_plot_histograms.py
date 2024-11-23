# Filename: evaluate_checkpoint.py

import argparse
import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader  # Use PyTorch Geometric's DataLoader
from tqdm import tqdm
from pmd_beamphysics import ParticleGroup  # Import ParticleGroup

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Import your models and utilities
from src.datasets.datasets import GraphDataset
from utils import generate_data_dirs, set_random_seed
from src.graph_models.models.graph_networks import (
    GraphConvolutionNetwork,
    GraphAttentionNetwork,
    GraphTransformer,
    MeshGraphNet
)
from src.graph_models.models.graph_autoencoders import (
    GraphConvolutionalAutoEncoder,
    GraphAttentionAutoEncoder,
    GraphTransformerAutoEncoder,
    MeshGraphAutoEncoder
)
from src.graph_models.models.intgnn.models import GNN_TopK
from src.graph_models.models.multiscale.gnn import (
    SinglescaleGNN,
    MultiscaleGNN,
    TopkMultiscaleGNN
)

def parse_hyperparameters_from_folder_name(folder_name):
    """
    Parses hyperparameters from the folder name and returns them as a dictionary.
    """
    hyperparams = {}
    parts = folder_name.split('_')
    known_prefixes = {
        'r', 'nt', 'b', 'lr', 'h', 'ly', 'pr', 'ep', 'sch',
        'heads', 'concat', 'dropout', 'mlph', 'mmply', 'mply'
    }
    idx = 0
    data_keyword_parts = []
    while idx < len(parts) and not any(parts[idx].startswith(prefix) for prefix in known_prefixes):
        data_keyword_parts.append(parts[idx])
        idx += 1
    hyperparams['data_keyword'] = '_'.join(data_keyword_parts)
    while idx < len(parts):
        part = parts[idx]
        matched = False
        for prefix in known_prefixes:
            if part.startswith(prefix):
                value = part[len(prefix):]
                if prefix == 'pr':  # pool_ratios
                    ratios = value.split('_')
                    hyperparams['pool_ratios'] = [float(r) for r in ratios]
                elif prefix == 'concat':
                    hyperparams['gtr_concat'] = value.lower() == 'true'
                elif prefix == 'sch':
                    hyperparams['lr_scheduler'] = value
                    if value.startswith('lin'):
                        lin_params = parts[idx + 1:idx + 4]
                        if len(lin_params) >= 3:
                            hyperparams['lin_start_epoch'] = int(lin_params[0])
                            hyperparams['lin_end_epoch'] = int(lin_params[1])
                            hyperparams['lin_final_lr'] = float(lin_params[2])
                            idx += 3  # Skip the parameters we just consumed
                else:
                    param_name = {
                        'r': 'random_seed',
                        'nt': 'ntrain',
                        'b': 'batch_size',
                        'lr': 'lr',
                        'h': 'hidden_dim',
                        'ly': 'num_layers',
                        'ep': 'nepochs',
                        'heads': 'heads',
                        'dropout': 'gtr_dropout',
                        'mlph': 'multiscale_n_mlp_hidden_layers',
                        'mmply': 'multiscale_n_mmp_layers',
                        'mply': 'multiscale_n_message_passing_layers',
                    }.get(prefix, prefix)
                    hyperparams[param_name] = type_cast(param_name, value)
                matched = True
                break
        if not matched:
            logging.warning(f"Unrecognized hyperparameter part: {part}")
        idx += 1
    return hyperparams

def extract_hyperparameters_from_checkpoint(checkpoint_path):
    """
    Extracts hyperparameters from the checkpoint path.
    """
    path_parts = [part for part in checkpoint_path.split(os.sep) if part]
    if 'checkpoints' not in path_parts:
        logging.error("'checkpoints' directory not found in the checkpoint path.")
        sys.exit(1)
    checkpoint_idx = path_parts.index('checkpoints')
    if checkpoint_idx < 4:
        logging.error("Checkpoint path is too short to extract hyperparameters.")
        sys.exit(1)
    folder_name = path_parts[checkpoint_idx - 1]
    task = path_parts[checkpoint_idx - 2]
    dataset = path_parts[checkpoint_idx - 3]
    model = path_parts[checkpoint_idx - 4]
    base_results_dir = os.sep + os.sep.join(path_parts[:checkpoint_idx - 4])
    hyperparams = parse_hyperparameters_from_folder_name(folder_name)
    hyperparams.update({
        'model': model,
        'dataset': dataset,
        'task': task,
        'base_results_dir': base_results_dir
    })
    return hyperparams

def type_cast(key, value):
    """
    Helper function to cast hyperparameter values to appropriate types.
    """
    int_params = {
        'random_seed', 'ntrain', 'batch_size', 'hidden_dim', 'num_layers', 'nepochs',
        'heads', 'multiscale_n_mlp_hidden_layers', 'multiscale_n_mmp_layers',
        'multiscale_n_message_passing_layers'
    }
    float_params = {'lr', 'gtr_dropout'}
    if key in int_params:
        return int(value)
    elif key in float_params:
        return float(value)
    else:
        return value

def check_missing_hyperparameters(hyperparams, required_params):
    missing_params = [param for param in required_params if param not in hyperparams]
    if missing_params:
        logging.error(f"Missing hyperparameters: {missing_params}")
        sys.exit(1)

def transform_to_particle_group(data):
    """
    Converts data tensor to ParticleGroup.

    Args:
        data (torch.Tensor): Tensor of shape [num_nodes, 6]

    Returns:
        ParticleGroup
    """
    num_particles = data.shape[0]
    particle_dict = {
        'x': data[:, 0].numpy(),
        'y': data[:, 1].numpy(),
        'z': data[:, 2].numpy(),
        'px': data[:, 3].numpy(),
        'py': data[:, 4].numpy(),
        'pz': data[:, 5].numpy(),
        'species': 'electron',
        'weight': np.full(num_particles, 2.e-17),
        't': np.zeros(num_particles),
        'status': np.ones(num_particles, dtype=int)
    }
    particle_group = ParticleGroup(data=particle_dict)
    return particle_group

def compute_normalized_emittance_x(particle_group):
    """
    Computes the normalized emittance in x direction for a ParticleGroup.

    Args:
        particle_group (ParticleGroup): The ParticleGroup for which to compute the emittance.

    Returns:
        float: The normalized emittance in x direction.
    """
    x = particle_group['x']
    px = particle_group['px']
    mean_x2 = np.mean(x**2)
    mean_px2 = np.mean(px**2)
    mean_xpx = np.mean(x * px)
    norm_emit_x = np.sqrt(mean_x2 * mean_px2 - mean_xpx**2)
    return norm_emit_x

def compute_normalized_emittance_y(particle_group):
    """
    Computes the normalized emittance in y direction for a ParticleGroup.
    """
    y = particle_group['y']
    py = particle_group['py']
    mean_y2 = np.mean(y**2)
    mean_py2 = np.mean(py**2)
    mean_ypy = np.mean(y * py)
    norm_emit_y = np.sqrt(mean_y2 * mean_py2 - mean_ypy**2)
    return norm_emit_y

def compute_normalized_emittance_z(particle_group):
    """
    Computes the normalized emittance in z direction for a ParticleGroup.
    """
    z = particle_group['z']
    pz = particle_group['pz']
    mean_z2 = np.mean(z**2)
    mean_pz2 = np.mean(pz**2)
    mean_zpz = np.mean(z * pz)
    norm_emit_z = np.sqrt(mean_z2 * mean_pz2 - mean_zpz**2)
    return norm_emit_z

def plot_particle_groups(pred_pg, target_pg, idx, error_type, emittance_direction, results_folder):
    """
    Plots and saves figures for predicted and target ParticleGroups.

    Args:
        pred_pg (ParticleGroup): Predicted ParticleGroup
        target_pg (ParticleGroup): Target ParticleGroup
        idx (int): Sample index
        error_type (str): 'min', 'median', or 'max'
        emittance_direction (str): 'x', 'y', or 'z'
        results_folder (str): Folder to save figures
    """
    var_map = {'x': ('x', 'px'), 'y': ('y', 'py'), 'z': ('z', 'pz')}
    x_var, p_var = var_map[emittance_direction]

    # Plot predicted
    plt.figure(figsize=(6, 6))
    pred_pg.plot(x_var, p_var, label='Predicted', alpha=0.6)
    plt.title(f'Predicted {x_var} vs. {p_var}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_folder, f'{error_type}_relative_error_{emittance_direction}_sample_{idx}_pred_{x_var}_{p_var}.png'))
    plt.close()
    # Plot target
    plt.figure(figsize=(6, 6))
    target_pg.plot(x_var, p_var, label='Target', alpha=0.6)
    plt.title(f'Target {x_var} vs. {p_var}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_folder, f'{error_type}_relative_error_{emittance_direction}_sample_{idx}_target_{x_var}_{p_var}.png'))
    plt.close()

def initialize_model(hyperparams, sample):
    # [Existing initialization code remains unchanged]
    # ...

    return model

def evaluate_model(model, dataloader, device, metadata_final_path, results_folder):
    model.eval()
    all_errors = []
    all_predictions = []
    all_targets = []
    all_relative_errors_x = []
    all_relative_errors_y = []
    all_relative_errors_z = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating Model"):
            data = data.to(device)
            x_pred = model_forward(model, data)
            mse = F.mse_loss(x_pred, data.y, reduction='none').mean(dim=1)
            batch_indices = data.batch.cpu().numpy()
            graph_indices = np.unique(batch_indices)
            for idx in graph_indices:
                mask = (batch_indices == idx)
                graph_mse = mse[mask].mean().item()
                all_errors.append(graph_mse)
                all_predictions.append(x_pred[mask].cpu())
                all_targets.append(data.y[mask].cpu())

    all_errors = np.array(all_errors)
    global_mean, global_std = load_global_statistics(metadata_final_path)

    def inverse_normalize(normalized_data):
        return normalized_data * global_std + global_mean

    # Compute relative errors in emittance
    for pred, target in zip(all_predictions, all_targets):
        pred_original = inverse_normalize(pred)
        target_original = inverse_normalize(target)

        pred_pg = transform_to_particle_group(pred_original)
        target_pg = transform_to_particle_group(target_original)

        pred_norm_emit_x = compute_normalized_emittance_x(pred_pg)
        target_norm_emit_x = compute_normalized_emittance_x(target_pg)
        relative_error_x = abs(pred_norm_emit_x - target_norm_emit_x) / abs(target_norm_emit_x)
        all_relative_errors_x.append(relative_error_x)

        # Compute norm emittance y
        pred_norm_emit_y = compute_normalized_emittance_y(pred_pg)
        target_norm_emit_y = compute_normalized_emittance_y(target_pg)
        relative_error_y = abs(pred_norm_emit_y - target_norm_emit_y) / abs(target_norm_emit_y)
        all_relative_errors_y.append(relative_error_y)

        # Compute norm emittance z
        pred_norm_emit_z = compute_normalized_emittance_z(pred_pg)
        target_norm_emit_z = compute_normalized_emittance_z(target_pg)
        relative_error_z = abs(pred_norm_emit_z - target_norm_emit_z) / abs(target_norm_emit_z)
        all_relative_errors_z.append(relative_error_z)

    # Plot histograms of relative errors
    plot_relative_error_histogram(all_relative_errors_x, 'x', results_folder)
    plot_relative_error_histogram(all_relative_errors_y, 'y', results_folder)
    plot_relative_error_histogram(all_relative_errors_z, 'z', results_folder)

    # Analyze samples for top, median, and bottom relative errors in x, y, z
    analyze_samples(all_predictions, all_targets, all_relative_errors_x, 'x', results_folder, inverse_normalize)
    analyze_samples(all_predictions, all_targets, all_relative_errors_y, 'y', results_folder, inverse_normalize)
    analyze_samples(all_predictions, all_targets, all_relative_errors_z, 'z', results_folder, inverse_normalize)

    # Compute overall average relative errors
    avg_relative_error_x = np.mean(all_relative_errors_x)
    avg_relative_error_y = np.mean(all_relative_errors_y)
    avg_relative_error_z = np.mean(all_relative_errors_z)

    logging.info(f"Average Relative Error in norm_emittance_x: {avg_relative_error_x:.4f}")
    logging.info(f"Average Relative Error in norm_emittance_y: {avg_relative_error_y:.4f}")
    logging.info(f"Average Relative Error in norm_emittance_z: {avg_relative_error_z:.4f}")

def plot_relative_error_histogram(relative_errors, emittance_direction, results_folder):
    """
    Plots and saves histogram of relative errors for a given emittance direction.

    Args:
        relative_errors (list): List of relative errors
        emittance_direction (str): 'x', 'y', or 'z'
        results_folder (str): Folder to save the histogram
    """
    plt.figure(figsize=(8, 6))
    plt.hist(relative_errors, bins=50, alpha=0.7, color='blue')
    plt.xlabel(f'Relative Error in {emittance_direction}-emittance')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Relative Error in {emittance_direction}-emittance')
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, f'hist_relative_error_{emittance_direction}_emittance.png'))
    plt.close()

def analyze_samples(all_predictions, all_targets, all_relative_errors, emittance_direction, results_folder, inverse_normalize):
    """
    Analyzes samples with top 1, median, and bottom 1 relative errors.

    Args:
        all_predictions (list): List of predictions
        all_targets (list): List of targets
        all_relative_errors (list): List of relative errors
        emittance_direction (str): 'x', 'y', or 'z'
        results_folder (str): Folder to save plots
        inverse_normalize (function): Function to inverse normalize data
    """
    all_relative_errors = np.array(all_relative_errors)
    sorted_indices = np.argsort(all_relative_errors)
    min_error_idx = sorted_indices[0]
    max_error_idx = sorted_indices[-1]
    median_idx = sorted_indices[len(sorted_indices) // 2]

    sample_indices = {
        'min': min_error_idx,
        'median': median_idx,
        'max': max_error_idx
    }

    for error_type, idx in sample_indices.items():
        pred = all_predictions[idx]
        target = all_targets[idx]
        pred_original = inverse_normalize(pred)
        target_original = inverse_normalize(target)

        pred_pg = transform_to_particle_group(pred_original)
        target_pg = transform_to_particle_group(target_original)

        plot_particle_groups(pred_pg, target_pg, idx, error_type, emittance_direction, results_folder)

def model_forward(model, data):
    """
    Forward pass for the model, handling different model types.
    """
    # [Existing model_forward code remains unchanged]
    # ...

    return x_pred

def load_global_statistics(metadata_final_path):
    """Load global mean and standard deviation from a file."""
    with open(metadata_final_path, 'r') as f:
        metadata_final = json.load(f)
    global_mean_final = torch.tensor(metadata_final['global_mean'])
    global_std_final = torch.tensor(metadata_final['global_std'])

    return global_mean_final, global_std_final

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--cpu_only', action='store_true', help="Force the script to use CPU even if GPU is available")
    parser.add_argument('--results_folder', type=str, default='evaluation_results', help="Folder to save evaluation results")
    parser.add_argument('--subsample_size', type=int, default=None, help="Number of samples to use for evaluation (if None, use all)")
    args = parser.parse_args()

    # Set device
    device = torch.device('cpu') if args.cpu_only else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Extract hyperparameters from the checkpoint path
    hyperparams = extract_hyperparameters_from_checkpoint(args.checkpoint)
    logging.info(f"Extracted hyperparameters: {hyperparams}")

    # Define required hyperparameters
    required_hyperparams = [
        'model', 'dataset', 'task', 'data_keyword', 'random_seed', 'batch_size',
        'hidden_dim', 'num_layers', 'pool_ratios'
    ]

    # Check for missing hyperparameters
    check_missing_hyperparameters(hyperparams, required_params=required_hyperparams)

    # Set random seed
    set_random_seed(hyperparams['random_seed'])

    # Generate data directories
    initial_graph_dir, final_graph_dir, settings_dir = generate_data_dirs(
        hyperparams.get('base_data_dir', '/sdf/data/ad/ard/u/tiffan/data/'),
        hyperparams['dataset'],
        hyperparams['data_keyword']
    )
    logging.info(f"Initial graph directory: {initial_graph_dir}")
    logging.info(f"Final graph directory: {final_graph_dir}")
    logging.info(f"Settings directory: {settings_dir}")

    # Initialize dataset
    use_edge_attr = hyperparams['model'].lower() in [
        'intgnn', 'gtr', 'mgn', 'gtr-ae', 'mgn-ae',
        'singlescale', 'multiscale', 'multiscale-topk'
    ]
    logging.info(f"Model '{hyperparams['model']}' requires edge_attr: {use_edge_attr}")

    dataset = GraphDataset(
        initial_graph_dir=initial_graph_dir,
        final_graph_dir=final_graph_dir,
        settings_dir=settings_dir,
        task=hyperparams['task'],
        use_edge_attr=use_edge_attr
    )

    total_dataset_size = len(dataset)
    logging.info(f"Total dataset size: {total_dataset_size}")

    # Subset dataset if subsample_size is specified
    if args.subsample_size is not None:
        np.random.seed(hyperparams['random_seed'])  # For reproducibility
        if args.subsample_size >= total_dataset_size:
            logging.warning(f"Requested subsample_size {args.subsample_size} is greater than or equal to total dataset size {total_dataset_size}. Using full dataset.")
        else:
            indices = np.random.permutation(total_dataset_size)[:args.subsample_size]
            dataset = Subset(dataset, indices)
            logging.info(f"Using a subsample of {args.subsample_size} samples for evaluation.")

    dataloader = DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    # Get a sample data for model initialization
    sample = dataset[0]

    # Initialize the model
    model = initialize_model(hyperparams, sample)
    model.to(device)
    logging.info(f"Model moved to {device}.")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded model state dict from {args.checkpoint}")

    # Create results folder
    results_folder = args.results_folder
    os.makedirs(results_folder, exist_ok=True)
    logging.info(f"Results will be saved to {results_folder}")

    # Path to metadata_final.json
    metadata_final_path = os.path.join(final_graph_dir, 'metadata.json')
    if not os.path.exists(metadata_final_path):
        logging.error(f"metadata.json not found at {metadata_final_path}")
        sys.exit(1)

    # Evaluate model
    evaluate_model(model, dataloader, device, metadata_final_path, results_folder)

if __name__ == "__main__":
    main()
