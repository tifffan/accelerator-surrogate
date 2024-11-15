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

def plot_particle_groups(pred_pg, target_pg, idx, error_type, results_folder):
    """
    Plots and saves figures for predicted and target ParticleGroups.

    Args:
        pred_pg (ParticleGroup): Predicted ParticleGroup
        target_pg (ParticleGroup): Target ParticleGroup
        idx (int): Sample index
        error_type (str): 'min' or 'max'
        results_folder (str): Folder to save figures
    """
    for x_var, p_var in [('x', 'px'), ('y', 'py'), ('z', 'pz')]:
        # Plot predicted
        plt.figure(figsize=(6, 6))
        pred_pg.plot(x_var, p_var, label='Predicted', alpha=0.6)
        plt.grid(True)
        plt.savefig(os.path.join(results_folder, f'{error_type}_mse_sample_{idx}_pred_{x_var}_{p_var}.png'))
        plt.close()
        # Plot target
        plt.figure(figsize=(6, 6))
        target_pg.plot(x_var, p_var, label='Target', alpha=0.6)
        plt.grid(True)
        plt.savefig(os.path.join(results_folder, f'{error_type}_mse_sample_{idx}_target_{x_var}_{p_var}.png'))
        plt.close()

def initialize_model(hyperparams, sample):
        model_name = hyperparams['model'].lower()

        def is_autoencoder_model(model_name):
            return model_name.lower().endswith('-ae') or model_name.lower() in ['multiscale-topk']

        if is_autoencoder_model(model_name):
            # Autoencoder models
            num_layers = hyperparams['num_layers']
            if num_layers % 2 != 0:
                raise ValueError(f"For autoencoder models, 'num_layers' must be an even number. Received: {num_layers}")
            depth = num_layers // 2
            logging.info(f"Autoencoder selected. Using depth: {depth} (num_layers: {num_layers})")
            required_pool_ratios = depth - 1
            pool_ratios = hyperparams.get('pool_ratios', [])
            current_pool_ratios = len(pool_ratios)

            if required_pool_ratios <= 0:
                pool_ratios = []
                logging.info(f"No pooling layers required for depth {depth}.")
            elif current_pool_ratios < required_pool_ratios:
                pool_ratios += [1.0] * (required_pool_ratios - current_pool_ratios)
                logging.warning(f"Pool ratios were padded with 1.0 to match required_pool_ratios: {required_pool_ratios}")
            elif current_pool_ratios > required_pool_ratios:
                pool_ratios = pool_ratios[:required_pool_ratios]
                logging.warning(f"Pool ratios were trimmed to match required_pool_ratios: {required_pool_ratios}")

            if model_name == 'gcn-ae':
                in_channels = sample.x.shape[1]
                hidden_dim = hyperparams['hidden_dim']
                out_channels = sample.y.shape[1]

                model = GraphConvolutionalAutoEncoder(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    out_channels=out_channels,
                    depth=depth,
                    pool_ratios=pool_ratios
                )
                logging.info("Initialized GraphConvolutionalAutoEncoder.")

            elif model_name == 'gat-ae':
                in_channels = sample.x.shape[1]
                hidden_dim = hyperparams['hidden_dim']
                out_channels = sample.y.shape[1]
                heads = hyperparams.get('gat_heads', 1)

                model = GraphAttentionAutoEncoder(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    out_channels=out_channels,
                    depth=depth,
                    pool_ratios=pool_ratios,
                    heads=heads
                )
                logging.info("Initialized GraphAttentionAutoEncoder.")

            elif model_name == 'gtr-ae':
                in_channels = sample.x.shape[1]
                hidden_dim = hyperparams['hidden_dim']
                out_channels = sample.y.shape[1]
                num_heads = hyperparams.get('gtr_heads', 4)
                concat = hyperparams.get('gtr_concat', True)
                dropout = hyperparams.get('gtr_dropout', 0.0)
                edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else None

                model = GraphTransformerAutoEncoder(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    out_channels=out_channels,
                    depth=depth,
                    pool_ratios=pool_ratios,
                    num_heads=num_heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
                logging.info("Initialized GraphTransformerAutoEncoder.")

            elif model_name == 'mgn-ae':
                node_in_dim = sample.x.shape[1]
                edge_in_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
                node_out_dim = sample.y.shape[1]
                hidden_dim = hyperparams['hidden_dim']

                model = MeshGraphAutoEncoder(
                    node_in_dim=node_in_dim,
                    edge_in_dim=edge_in_dim,
                    node_out_dim=node_out_dim,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    pool_ratios=pool_ratios
                )
                logging.info("Initialized MeshGraphAutoEncoder.")

            elif model_name == 'multiscale-topk':
                input_node_channels = sample.x.shape[1]
                input_edge_channels = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
                hidden_channels = hyperparams['hidden_dim']
                output_node_channels = sample.y.shape[1]
                n_mlp_hidden_layers = hyperparams.get('multiscale_n_mlp_hidden_layers', 2)
                n_mmp_layers = hyperparams.get('multiscale_n_mmp_layers', 4)
                n_messagePassing_layers = hyperparams.get('multiscale_n_message_passing_layers', 2)
                max_level_mmp = num_layers // 2 - 1
                max_level_topk = num_layers // 2 - 1

                # Compute l_char (characteristic length scale)
                edge_index = sample.edge_index
                pos = sample.pos
                edge_lengths = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
                l_char = edge_lengths.mean().item()
                logging.info(f"Computed l_char (characteristic length scale): {l_char}")

                name = 'topk_multiscale_gnn'

                model = TopkMultiscaleGNN(
                    input_node_channels=input_node_channels,
                    input_edge_channels=input_edge_channels,
                    hidden_channels=hidden_channels,
                    output_node_channels=output_node_channels,
                    n_mlp_hidden_layers=n_mlp_hidden_layers,
                    n_mmp_layers=n_mmp_layers,
                    n_messagePassing_layers=n_messagePassing_layers,
                    max_level_mmp=max_level_mmp,
                    max_level_topk=max_level_topk,
                    pool_ratios=pool_ratios,
                    l_char=l_char,
                    name=name
                )
                logging.info("Initialized TopkMultiscaleGNN model.")

            else:
                raise ValueError(f"Unknown autoencoder model {model_name}")

        else:
            # Non-autoencoder models
            if model_name == 'intgnn':
                in_channels_node = sample.x.shape[1]
                in_channels_edge = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
                hidden_channels = hyperparams['hidden_dim']
                out_channels = sample.y.shape[1]
                n_mlp_encode = 3
                n_mlp_mp = 2
                n_mp_down_topk = [1, 1]
                n_mp_up_topk = [1, 1]
                pool_ratios = hyperparams.get('pool_ratios', [])
                n_mp_down_enc = [4]
                n_mp_up_enc = []
                lengthscales_enc = []
                n_mp_down_dec = [2, 2, 4]
                n_mp_up_dec = [2, 2]
                lengthscales_dec = [0.5, 1.0]
                interp = 'learned'
                act = F.elu
                param_sharing = False

                # Create bounding box if needed
                bounding_box = []
                if len(lengthscales_dec) > 0:
                    x_lo = sample.pos[:, 0].min() - lengthscales_dec[0] / 2
                    x_hi = sample.pos[:, 0].max() + lengthscales_dec[0] / 2
                    y_lo = sample.pos[:, 1].min() - lengthscales_dec[0] / 2
                    y_hi = sample.pos[:, 1].max() + lengthscales_dec[0] / 2
                    z_lo = sample.pos[:, 2].min() - lengthscales_dec[0] / 2
                    z_hi = sample.pos[:, 2].max() + lengthscales_dec[0] / 2
                    bounding_box = [
                        x_lo.item(), x_hi.item(),
                        y_lo.item(), y_hi.item(),
                        z_lo.item(), z_hi.item()
                    ]

                model = GNN_TopK(
                    in_channels_node,
                    in_channels_edge,
                    hidden_channels,
                    out_channels,
                    n_mlp_encode,
                    n_mlp_mp,
                    n_mp_down_topk,
                    n_mp_up_topk,
                    pool_ratios,
                    n_mp_down_enc,
                    n_mp_up_enc,
                    n_mp_down_dec,
                    n_mp_up_dec,
                    lengthscales_enc,
                    lengthscales_dec,
                    bounding_box,
                    interp,
                    act,
                    param_sharing,
                    name='gnn_topk'
                )
                logging.info("Initialized GNN_TopK model.")

            elif model_name == 'singlescale':
                input_node_channels = sample.x.shape[1]
                input_edge_channels = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
                hidden_channels = hyperparams['hidden_dim']
                output_node_channels = sample.y.shape[1]
                n_mlp_hidden_layers = 0  # As per MeshGraphNet
                n_messagePassing_layers = hyperparams['num_layers']
                name = 'singlescale_gnn'

                model = SinglescaleGNN(
                    input_node_channels=input_node_channels,
                    input_edge_channels=input_edge_channels,
                    hidden_channels=hidden_channels,
                    output_node_channels=output_node_channels,
                    n_mlp_hidden_layers=n_mlp_hidden_layers,
                    n_messagePassing_layers=n_messagePassing_layers,
                    name=name
                )
                logging.info("Initialized SinglescaleGNN model.")

            elif model_name == 'multiscale':
                input_node_channels = sample.x.shape[1]
                input_edge_channels = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
                hidden_channels = hyperparams['hidden_dim']
                output_node_channels = sample.y.shape[1]
                n_mlp_hidden_layers = hyperparams.get('multiscale_n_mlp_hidden_layers', 2)
                n_mmp_layers = hyperparams.get('multiscale_n_mmp_layers', 4)
                n_messagePassing_layers = hyperparams.get('multiscale_n_message_passing_layers', 2)
                num_layers = hyperparams['num_layers']
                max_level = num_layers // 2 - 1

                # Compute l_char (characteristic length scale)
                edge_index = sample.edge_index
                pos = sample.pos
                edge_lengths = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
                l_char = edge_lengths.mean().item()
                logging.info(f"Computed l_char (characteristic length scale): {l_char}")

                name = 'multiscale_gnn'

                model = MultiscaleGNN(
                    input_node_channels=input_node_channels,
                    input_edge_channels=input_edge_channels,
                    hidden_channels=hidden_channels,
                    output_node_channels=output_node_channels,
                    n_mlp_hidden_layers=n_mlp_hidden_layers,
                    n_mmp_layers=n_mmp_layers,
                    n_messagePassing_layers=n_messagePassing_layers,
                    max_level=max_level,
                    l_char=l_char,
                    name=name
                )
                logging.info("Initialized MultiscaleGNN model.")

            elif model_name == 'gcn':
                num_layers = hyperparams['num_layers']
                required_pool_ratios = num_layers - 2
                pool_ratios = hyperparams.get('pool_ratios', [])
                current_pool_ratios = len(pool_ratios)

                if required_pool_ratios <= 0:
                    pool_ratios = []
                    logging.info(f"No pooling layers required for num_layers {num_layers}.")
                elif current_pool_ratios < required_pool_ratios:
                    pool_ratios += [1.0] * (required_pool_ratios - current_pool_ratios)
                    logging.warning(f"Pool ratios were padded with 1.0 to match required_pool_ratios: {required_pool_ratios}")
                elif current_pool_ratios > required_pool_ratios:
                    pool_ratios = pool_ratios[:required_pool_ratios]
                    logging.warning(f"Pool ratios were trimmed to match required_pool_ratios: {required_pool_ratios}")

                in_channels = sample.x.shape[1]
                hidden_dim = hyperparams['hidden_dim']
                out_channels = sample.y.shape[1]

                model = GraphConvolutionNetwork(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    out_channels=out_channels,
                    num_layers=num_layers,
                    pool_ratios=pool_ratios,
                )
                logging.info("Initialized GraphConvolutionNetwork model.")

            elif model_name == 'gat':
                num_layers = hyperparams['num_layers']
                required_pool_ratios = num_layers - 2
                pool_ratios = hyperparams.get('pool_ratios', [])
                current_pool_ratios = len(pool_ratios)

                if required_pool_ratios <= 0:
                    pool_ratios = []
                    logging.info(f"No pooling layers required for num_layers {num_layers}.")
                elif current_pool_ratios < required_pool_ratios:
                    pool_ratios += [1.0] * (required_pool_ratios - current_pool_ratios)
                    logging.warning(f"Pool ratios were padded with 1.0 to match required_pool_ratios: {required_pool_ratios}")
                elif current_pool_ratios > required_pool_ratios:
                    pool_ratios = pool_ratios[:required_pool_ratios]
                    logging.warning(f"Pool ratios were trimmed to match required_pool_ratios: {required_pool_ratios}")

                in_channels = sample.x.shape[1]
                hidden_dim = hyperparams['hidden_dim']
                out_channels = sample.y.shape[1]
                heads = hyperparams.get('gat_heads', 1)

                model = GraphAttentionNetwork(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    out_channels=out_channels,
                    num_layers=num_layers,
                    pool_ratios=pool_ratios,
                    heads=heads,
                )
                logging.info("Initialized GraphAttentionNetwork model.")

            elif model_name == 'gtr':
                num_layers = hyperparams['num_layers']
                required_pool_ratios = num_layers - 2
                pool_ratios = hyperparams.get('pool_ratios', [])
                current_pool_ratios = len(pool_ratios)

                if required_pool_ratios <= 0:
                    pool_ratios = []
                    logging.info(f"No pooling layers required for num_layers {num_layers}.")
                elif current_pool_ratios < required_pool_ratios:
                    pool_ratios += [1.0] * (required_pool_ratios - current_pool_ratios)
                    logging.warning(f"Pool ratios were padded with 1.0 to match required_pool_ratios: {required_pool_ratios}")
                elif current_pool_ratios > required_pool_ratios:
                    pool_ratios = pool_ratios[:required_pool_ratios]
                    logging.warning(f"Pool ratios were trimmed to match required_pool_ratios: {required_pool_ratios}")

                in_channels = sample.x.shape[1]
                hidden_dim = hyperparams['hidden_dim']
                out_channels = sample.y.shape[1]
                num_heads = hyperparams.get('gtr_heads', 4)
                concat = hyperparams.get('gtr_concat', True)
                dropout = hyperparams.get('gtr_dropout', 0.0)
                edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else None

                model = GraphTransformer(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    out_channels=out_channels,
                    num_layers=num_layers,
                    pool_ratios=pool_ratios,
                    num_heads=num_heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
                logging.info("Initialized GraphTransformer model.")

            elif model_name == 'mgn':
                node_in_dim = sample.x.shape[1]
                edge_in_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
                node_out_dim = sample.y.shape[1]
                hidden_dim = hyperparams['hidden_dim']
                num_layers = hyperparams['num_layers']

                model = MeshGraphNet(
                    node_in_dim=node_in_dim,
                    edge_in_dim=edge_in_dim,
                    node_out_dim=node_out_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers
                )
                logging.info("Initialized MeshGraphNet model.")

            else:
                logging.error(f"Unknown model '{model_name}'.")
                sys.exit(1)

        return model

def evaluate_model(model, dataloader, device, metadata_final_path, results_folder):
    model.eval()
    all_errors = []
    all_predictions = []
    all_targets = []
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
    sorted_indices = np.argsort(all_errors)
    min_mse_indices = sorted_indices[:5]
    max_mse_indices = sorted_indices[-5:]

    global_mean, global_std = load_global_statistics(metadata_final_path)

    def inverse_normalize(normalized_data):
        return normalized_data * global_std + global_mean

    # For min MSE samples
    for idx in min_mse_indices:
        pred = all_predictions[idx]
        target = all_targets[idx]
        pred_original = inverse_normalize(pred)
        target_original = inverse_normalize(target)
        pred_pg = transform_to_particle_group(pred_original)
        target_pg = transform_to_particle_group(target_original)
        plot_particle_groups(pred_pg, target_pg, idx, 'min', results_folder)
        pred_norm_emit_x = compute_normalized_emittance_x(pred_pg)
        target_norm_emit_x = compute_normalized_emittance_x(target_pg)
        relative_error = abs(pred_norm_emit_x - target_norm_emit_x) / abs(target_norm_emit_x)
        print(f"Sample {idx} (min MSE): Relative Error in norm_emit_x: {relative_error}")

    # For max MSE samples
    for idx in max_mse_indices:
        pred = all_predictions[idx]
        target = all_targets[idx]
        pred_original = inverse_normalize(pred)
        target_original = inverse_normalize(target)
        pred_pg = transform_to_particle_group(pred_original)
        target_pg = transform_to_particle_group(target_original)
        plot_particle_groups(pred_pg, target_pg, idx, 'max', results_folder)
        pred_norm_emit_x = compute_normalized_emittance_x(pred_pg)
        target_norm_emit_x = compute_normalized_emittance_x(target_pg)
        relative_error = abs(pred_norm_emit_x - target_norm_emit_x) / abs(target_norm_emit_x)
        print(f"Sample {idx} (max MSE): Relative Error in norm_emit_x: {relative_error}")

    # Compute overall test error
    test_error = all_errors.mean()
    print(f"Test Error (MSE): {test_error}")

def model_forward(model, data):
    """
    Forward pass for the model, handling different model types.
    """
    if isinstance(model, GNN_TopK):
        x_pred, _ = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.pos,
            batch=data.batch
        )
    elif isinstance(model, TopkMultiscaleGNN):
        x_pred, _ = model(
            data.x,
            data.edge_index,
            data.pos,
            data.edge_attr,
            data.batch
        )
    elif isinstance(model, (SinglescaleGNN, MultiscaleGNN)):
        x_pred = model(
            data.x,
            data.edge_index,
            data.pos,
            data.edge_attr,
            data.batch
        )
    elif isinstance(model, (MeshGraphNet, MeshGraphAutoEncoder)):
        x_pred = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )
    elif isinstance(model, (GraphTransformer, GraphTransformerAutoEncoder)):
        x_pred = model(
            data.x,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None,
            data.batch
        )
    else:  # GraphConvolutionNetwork, GraphAttentionNetwork
        x_pred = model(
            data.x,
            data.edge_index,
            data.batch
        )
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

