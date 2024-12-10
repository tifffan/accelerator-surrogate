# train.py

import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch_geometric
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
from src.graph_models.models.multiscale.gnn import (
    SinglescaleGNN,
    MultiscaleGNN,
    TopkMultiscaleGNN
)

from trainers import GraphPredictionTrainer
from utils import (
    generate_data_dirs,
    generate_results_folder_name,
    save_metadata,
    get_scheduler,
    set_random_seed
)
from config import parse_args

# Import the data loaders
from dataloaders import GraphDataLoaders

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def is_autoencoder_model(model_name: str) -> bool:
    """
    Determines if the given model name corresponds to an autoencoder.

    Args:
        model_name (str): Name of the model.

    Returns:
        bool: True if it's an autoencoder model, False otherwise.
    """
    return model_name.lower().endswith('-ae') or model_name.lower() in ['multiscale-topk']


def adjust_pool_ratios(args, required_pool_ratios: int) -> list:
    """
    Adjusts the pool_ratios list to match the required number of pooling layers.

    Args:
        args: Parsed command-line arguments containing pool_ratios.
        required_pool_ratios (int): Number of pooling layers required.

    Returns:
        list: Adjusted pool_ratios.
    """
    current_pool_ratios = len(args.pool_ratios)

    if required_pool_ratios <= 0:
        adjusted_pool_ratios = []
        logging.info(f"No pooling layers required for the given configuration.")
    elif current_pool_ratios < required_pool_ratios:
        # Pad pool_ratios with 1.0 to match required_pool_ratios
        adjusted_pool_ratios = args.pool_ratios + [1.0] * (required_pool_ratios - current_pool_ratios)
        logging.warning(f"Pool ratios were padded with 1.0 to match required_pool_ratios: {required_pool_ratios}")
    elif current_pool_ratios > required_pool_ratios:
        # Trim pool_ratios to match required_pool_ratios
        adjusted_pool_ratios = args.pool_ratios[:required_pool_ratios]
        logging.warning(f"Pool ratios were trimmed to match required_pool_ratios: {required_pool_ratios}")
    else:
        adjusted_pool_ratios = args.pool_ratios

    return adjusted_pool_ratios


def initialize_model(args, sample: torch_geometric.data.Data) -> torch.nn.Module:
    """
    Initializes the model based on the provided arguments and sample data.

    Args:
        args: Parsed command-line arguments.
        sample (torch_geometric.data.Data): A sample data point from the dataset.

    Returns:
        torch.nn.Module: Initialized model.
    """
    model_name = args.model.lower()

    if is_autoencoder_model(model_name):
        # Autoencoder models: 'gcn-ae', 'gat-ae', 'gtr-ae', 'mgn-ae', 'multiscale-topk'
        if args.num_layers % 2 != 0:
            raise ValueError(f"For autoencoder models, 'num_layers' must be an even number. Received: {args.num_layers}")

        depth = args.num_layers // 2
        logging.info(f"Autoencoder selected. Using depth: {depth} (num_layers: {args.num_layers})")

        pool_ratios = adjust_pool_ratios(args, required_pool_ratios=depth - 1)

        if model_name == 'gcn-ae':
            model = GraphConvolutionalAutoEncoder(
                in_channels=sample.x.shape[1],
                hidden_dim=args.hidden_dim,
                out_channels=sample.y.shape[1],
                depth=depth,
                pool_ratios=pool_ratios
            )
            logging.info("Initialized GraphConvolutionalAutoEncoder.")

        elif model_name == 'gat-ae':
            model = GraphAttentionAutoEncoder(
                in_channels=sample.x.shape[1],
                hidden_dim=args.hidden_dim,
                out_channels=sample.y.shape[1],
                depth=depth,
                pool_ratios=pool_ratios,
                heads=args.gat_heads
            )
            logging.info("Initialized GraphAttentionAutoEncoder.")

        elif model_name == 'gtr-ae':
            edge_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else None
            model = GraphTransformerAutoEncoder(
                in_channels=sample.x.shape[1],
                hidden_dim=args.hidden_dim,
                out_channels=sample.y.shape[1],
                depth=depth,
                pool_ratios=pool_ratios,
                num_heads=args.gtr_heads,
                concat=args.gtr_concat,
                dropout=args.gtr_dropout,
                edge_dim=edge_dim
            )
            logging.info("Initialized GraphTransformerAutoEncoder.")

        elif model_name == 'mgn-ae':
            edge_in_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0
            model = MeshGraphAutoEncoder(
                node_in_dim=sample.x.shape[1],
                edge_in_dim=edge_in_dim,
                node_out_dim=sample.y.shape[1],
                hidden_dim=args.hidden_dim,
                depth=depth,
                pool_ratios=pool_ratios
            )
            logging.info("Initialized MeshGraphAutoEncoder.")

        elif model_name == 'multiscale-topk':
            edge_in_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0
            edge_lengths = torch.norm(sample.pos[sample.edge_index[0]] - sample.pos[sample.edge_index[1]], dim=1)
            l_char = edge_lengths.mean().item()
            logging.info(f"Computed l_char (characteristic length scale): {l_char}")

            model = TopkMultiscaleGNN(
                input_node_channels=sample.x.shape[1],
                input_edge_channels=edge_in_dim,
                hidden_channels=args.hidden_dim,
                output_node_channels=sample.y.shape[1],
                n_mlp_hidden_layers=args.multiscale_n_mlp_hidden_layers,
                n_mmp_layers=args.multiscale_n_mmp_layers,
                n_messagePassing_layers=args.multiscale_n_message_passing_layers,
                max_level_mmp=args.num_layers // 2 - 1,
                max_level_topk=args.num_layers // 2 - 1,
                pool_ratios=pool_ratios,
                l_char=l_char,
                name='topk_multiscale_gnn'
            )
            logging.info("Initialized TopkMultiscaleGNN model.")

        else:
            raise ValueError(f"Unknown autoencoder model {args.model}")

    else:
        # Non-autoencoder models: 'gcn', 'gat', 'gtr', 'mgn', 'singlescale', 'multiscale'
        if model_name == 'singlescale':
            model = SinglescaleGNN(
                input_node_channels=sample.x.shape[1],
                input_edge_channels=sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0,
                hidden_channels=args.hidden_dim,
                output_node_channels=sample.y.shape[1],
                n_mlp_hidden_layers=0,
                n_messagePassing_layers=args.num_layers,
                name='singlescale_gnn'
            )
            logging.info("Initialized SinglescaleGNN model.")

        elif model_name == 'multiscale':
            edge_lengths = torch.norm(sample.pos[sample.edge_index[0]] - sample.pos[sample.edge_index[1]], dim=1)
            l_char = edge_lengths.mean().item()
            logging.info(f"Computed l_char (characteristic length scale): {l_char}")

            model = MultiscaleGNN(
                input_node_channels=sample.x.shape[1],
                input_edge_channels=sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0,
                hidden_channels=args.hidden_dim,
                output_node_channels=sample.y.shape[1],
                n_mlp_hidden_layers=args.multiscale_n_mlp_hidden_layers,
                n_mmp_layers=args.multiscale_n_mmp_layers,
                n_messagePassing_layers=args.multiscale_n_message_passing_layers,
                max_level=args.num_layers // 2 - 1,
                l_char=l_char,
                name='multiscale_gnn'
            )
            logging.info("Initialized MultiscaleGNN model.")

        elif model_name == 'gcn':
            pool_ratios = adjust_pool_ratios(args, required_pool_ratios=args.num_layers - 2)
            model = GraphConvolutionNetwork(
                in_channels=sample.x.shape[1],
                hidden_dim=args.hidden_dim,
                out_channels=sample.y.shape[1],
                num_layers=args.num_layers,
                pool_ratios=pool_ratios,
            )
            logging.info("Initialized GraphConvolutionNetwork model.")

        elif model_name == 'gat':
            pool_ratios = adjust_pool_ratios(args, required_pool_ratios=args.num_layers - 2)
            model = GraphAttentionNetwork(
                in_channels=sample.x.shape[1],
                hidden_dim=args.hidden_dim,
                out_channels=sample.y.shape[1],
                num_layers=args.num_layers,
                pool_ratios=pool_ratios,
                heads=args.gat_heads
            )
            logging.info("Initialized GraphAttentionNetwork model.")

        elif model_name == 'gtr':
            pool_ratios = adjust_pool_ratios(args, required_pool_ratios=args.num_layers - 2)
            edge_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else None
            model = GraphTransformer(
                in_channels=sample.x.shape[1],
                hidden_dim=args.hidden_dim,
                out_channels=sample.y.shape[1],
                num_layers=args.num_layers,
                pool_ratios=pool_ratios,
                num_heads=args.gtr_heads,
                concat=args.gtr_concat,
                dropout=args.gtr_dropout,
                edge_dim=edge_dim,
            )
            logging.info("Initialized GraphTransformer model.")

        elif model_name == 'mgn':
            model = MeshGraphNet(
                node_in_dim=sample.x.shape[1],
                edge_in_dim=sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0,
                node_out_dim=sample.y.shape[1],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers
            )
            logging.info("Initialized MeshGraphNet model.")

        else:
            raise ValueError(f"Unknown model {args.model}")

    return model


def handle_checkpoint(args: Optional[object], results_folder: str) -> Optional[str]:
    """
    Handles the checkpoint logic based on the provided arguments.

    Args:
        args: Parsed command-line arguments.
        results_folder (str): Path to the results folder.

    Returns:
        Optional[str]: Path to the checkpoint file if set, else None.
    """
    checkpoint_path = args.checkpoint

    if checkpoint_path is None and args.checkpoint_epoch is not None:
        results_folder_ = re.sub(r'ep\d+', f'ep{args.checkpoint_epoch}', results_folder)
        checkpoint_path = os.path.join(results_folder_, 'checkpoints', f'model-{args.checkpoint_epoch - 1}.pth')
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint for epoch {args.checkpoint_epoch} not found at {checkpoint_path}. Exiting.")
            exit(1)
        else:
            logging.info(f"Checkpoint set to: {checkpoint_path}")

    elif checkpoint_path is not None and args.checkpoint_epoch is None:
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found at {checkpoint_path}. Exiting.")
            exit(1)
        else:
            logging.info(f"Checkpoint set to: {checkpoint_path}")

    return checkpoint_path


def main():
    args = parse_args()

    # Set device
    device = torch.device('cpu') if args.cpu_only else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Generate data directories
    initial_graph_dir, final_graph_dir, settings_dir = generate_data_dirs(
        args.base_data_dir, args.dataset, args.data_keyword
    )
    logging.info(f"Initial graph directory: {initial_graph_dir}")
    logging.info(f"Final graph directory: {final_graph_dir}")
    logging.info(f"Settings directory: {settings_dir}")

    # Generate results folder name
    results_folder = args.results_folder or generate_results_folder_name(args)
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    logging.info(f"Results will be saved to {results_folder}")

    # Set random seed
    set_random_seed(args.random_seed)

    # Determine if the model requires edge_attr
    models_requiring_edge_attr = [
        'gtr', 'mgn', 'gtr-ae', 'mgn-ae', 'singlescale', 'multiscale', 'multiscale-topk'
    ]
    use_edge_attr = args.model.lower() in models_requiring_edge_attr
    logging.info(f"Model '{args.model}' requires edge_attr: {use_edge_attr}")

    # Initialize data loaders
    data_loaders = GraphDataLoaders(
        initial_graph_dir=initial_graph_dir,
        final_graph_dir=final_graph_dir,
        settings_dir=settings_dir,
        task=args.task,
        use_edge_attr=use_edge_attr,
        edge_attr_method=args.edge_attr_method,
        preload_data=args.preload_data,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test
    )
    logging.info(f"Initialized GraphDataLoaders with n_train={args.n_train}, n_val={args.n_val}, n_test={args.n_test}")

    # Get data loaders
    train_loader = data_loaders.get_train_loader()
    val_loader = data_loaders.get_val_loader()
    test_loader = data_loaders.get_test_loader()

    # Get a sample data for model initialization
    try:
        sample_batch = next(iter(train_loader))
    except StopIteration:
        logging.error("Training loader is empty. Exiting.")
        exit(1)

    # Move sample to device
    if isinstance(sample_batch, list) or isinstance(sample_batch, tuple):
        sample = sample_batch[0].to(device)
    else:
        sample = sample_batch.to(device)

    # Initialize the model
    model = initialize_model(args, sample)

    model.to(device)
    logging.info(f"Model moved to {device}.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    logging.info(f"Initialized Adam optimizer with learning rate: {args.lr} and weight decay: {args.wd}")

    # Scheduler
    scheduler = get_scheduler(args, optimizer)
    if scheduler:
        logging.info("Initialized learning rate scheduler.")

    # Handle checkpoint
    checkpoint_path = handle_checkpoint(args, results_folder)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Initialize trainer with the custom loss function
    trainer = GraphPredictionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        nepochs=args.nepochs,
        save_checkpoint_every=args.save_checkpoint_every,
        results_folder=results_folder,
        checkpoint=checkpoint_path,
        random_seed=args.random_seed,
        device=device,
        verbose=args.verbose,
        criterion=criterion  # Pass the custom loss function here
    )
    logging.info("Initialized GraphPredictionTrainer with custom loss function.")

    # Save metadata
    save_metadata(args, model, results_folder)

    # Run train or evaluate
    if args.mode == 'train':
        trainer.train()
    else:
        logging.info("Evaluation mode is not implemented yet.")
        # TODO: Implement evaluation functionality


if __name__ == "__main__":
    main()
