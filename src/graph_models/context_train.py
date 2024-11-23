# train.py

import torch
import torch.nn.functional as F
from src.datasets.context_datasets import GraphSettingsDataset  # Updated import
from src.graph_models.context_models.context_graph_networks import (
    ConditionalGraphNetwork,
    AttentionConditionalGraphNetwork,
    GeneralGraphNetwork
)
# Removed imports of outdated models and autoencoders
from context_trainers import GraphPredictionTrainer
from utils import (
    generate_data_dirs,
    generate_results_folder_name,
    save_metadata,
    get_scheduler,
    set_random_seed
)
from context_config import parse_args
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import numpy as np
import logging
import re
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_autoencoder_model(model_name):
    """
    Determines if the given model name corresponds to an autoencoder.

    Args:
        model_name (str): Name of the model.

    Returns:
        bool: True if it's an autoencoder model, False otherwise.
    """
    # Adjusted to reflect new models
    return model_name.lower().endswith('-ae')

if __name__ == "__main__":
    args = parse_args()
    
    # Prepare WandB config
    wandb_config = {
        "project": "graph-learning",
        "config": vars(args),
        "name": args.results_folder,
    }

    # Set device
    if args.cpu_only:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Generate data directories
    initial_graph_dir, final_graph_dir, settings_dir = generate_data_dirs(
        args.base_data_dir, args.dataset, args.data_keyword
    )
    logging.info(f"Initial graph directory: {initial_graph_dir}")
    logging.info(f"Final graph directory: {final_graph_dir}")
    logging.info(f"Settings directory: {settings_dir}")

    # Generate results folder name
    if args.results_folder is not None:
        results_folder = args.results_folder
    else:
        results_folder = generate_results_folder_name(args)
    logging.info(f"Results will be saved to {results_folder}")

    # Set random seed
    set_random_seed(args.random_seed)

    # Map shorter strings to actual model class names
    model_name_mapping = {
        'ggn': 'GeneralGraphNetwork',
        'cgn': 'ConditionalGraphNetwork',
        'acgn': 'AttentionConditionalGraphNetwork',
        # Add other model mappings here if necessary
    }

    # Determine if the model requires edge_attr
    models_requiring_edge_attr = ['ggn', 'cgn', 'acgn']
    use_edge_attr = args.model.lower() in models_requiring_edge_attr
    logging.info(f"Model '{args.model}' requires edge_attr: {use_edge_attr}")

    # Initialize dataset
    dataset = GraphSettingsDataset(
        initial_graph_dir=initial_graph_dir,
        final_graph_dir=final_graph_dir,
        settings_dir=settings_dir,
        task=args.task,
        use_edge_attr=use_edge_attr
    )

    # Subset dataset if ntrain is specified
    total_dataset_size = len(dataset)
    if args.ntrain is not None:
        np.random.seed(args.random_seed)  # For reproducibility
        indices = np.random.permutation(total_dataset_size)[:args.ntrain]
        dataset = Subset(dataset, indices)
        
    # After initializing the dataset and subset
    for i in range(min(1, len(dataset))):
        sample = dataset[i]
        if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
            logging.info(f"Sample {i} edge_attr shape: {sample.edge_attr.shape}")
        else:
            logging.warning(f"Sample {i} is missing edge_attr.")

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Get a sample data for model initialization
    sample = dataset[0]

    # Model initialization
    if is_autoencoder_model(args.model):
        # Handle autoencoder models if any
        raise NotImplementedError("Autoencoder models are not defined in the current context.")
    else:
        # Non-autoencoder models
        model_key = args.model.lower()
        if model_key in model_name_mapping:
            model_class_name = model_name_mapping[model_key]
            # Dynamically get the model class from globals()
            model_class = globals().get(model_class_name)
            if model_class is None:
                raise ValueError(f"Model class '{model_class_name}' not found.")
            logging.info(f"Initializing model '{model_class_name}'.")

            if model_class_name == 'GeneralGraphNetwork':
                # Initialize GeneralGraphNetwork
                node_in_dim = sample.x.shape[1]
                edge_in_dim = sample.edge_attr.shape[1] if use_edge_attr else 0
                global_in_dim = sample.set.shape[0]  # Using 'set' field from dataset
                node_out_dim = sample.y.shape[1]
                edge_out_dim = edge_in_dim  # Assuming edge output dimensions are same as input
                global_out_dim = global_in_dim
                hidden_dim = args.hidden_dim
                num_layers = args.num_layers

                model = GeneralGraphNetwork(
                    node_in_dim=node_in_dim,
                    edge_in_dim=edge_in_dim,
                    global_in_dim=global_in_dim,
                    node_out_dim=node_out_dim,
                    edge_out_dim=edge_out_dim,
                    global_out_dim=global_out_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers
                )
                logging.info("Initialized GeneralGraphNetwork model.")

            elif model_class_name == 'ConditionalGraphNetwork':
                # Initialize ConditionalGraphNetwork
                node_in_dim = sample.x.shape[1]
                edge_in_dim = sample.edge_attr.shape[1] if use_edge_attr else 0
                cond_in_dim = sample.set.shape[0]
                node_out_dim = sample.y.shape[1]
                hidden_dim = args.hidden_dim
                num_layers = args.num_layers

                model = ConditionalGraphNetwork(
                    node_in_dim=node_in_dim,
                    edge_in_dim=edge_in_dim,
                    cond_in_dim=cond_in_dim,
                    node_out_dim=node_out_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers
                )
                logging.info("Initialized ConditionalGraphNetwork model.")

            elif model_class_name == 'AttentionConditionalGraphNetwork':
                # Initialize AttentionConditionalGraphNetwork
                node_in_dim = sample.x.shape[1]
                edge_in_dim = sample.edge_attr.shape[1] if use_edge_attr else 0
                cond_in_dim = sample.set.shape[0]
                node_out_dim = sample.y.shape[1]
                hidden_dim = args.hidden_dim
                num_layers = args.num_layers

                model = AttentionConditionalGraphNetwork(
                    node_in_dim=node_in_dim,
                    edge_in_dim=edge_in_dim,
                    cond_in_dim=cond_in_dim,
                    node_out_dim=node_out_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers
                )
                logging.info("Initialized AttentionConditionalGraphNetwork model.")

            else:
                raise ValueError(f"Unknown model class '{model_class_name}'")
        else:
            raise ValueError(f"Unknown model '{args.model}'. Available options are: {list(model_name_mapping.keys())}")

    model.to(device)
    logging.info(f"Model moved to {device}.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logging.info(f"Initialized Adam optimizer with learning rate: {args.lr}")

    # Scheduler
    scheduler = get_scheduler(args, optimizer)
    if scheduler:
        logging.info("Initialized learning rate scheduler.")

    # Handle checkpoint
    if args.checkpoint is None and args.checkpoint_epoch is not None:
        results_folder_ = re.sub(r'ep\d+', f'ep{args.checkpoint_epoch}', results_folder)
        checkpoint_path = os.path.join(results_folder_, 'checkpoints', f'model-{args.checkpoint_epoch - 1}.pth')
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint for epoch {args.checkpoint_epoch} not found at {checkpoint_path}. Exiting.")
            exit(1)
        else:
            args.checkpoint = checkpoint_path
            logging.info(f"Checkpoint set to: {args.checkpoint}")

    if args.checkpoint is not None and args.checkpoint_epoch is None:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found at {checkpoint_path}. Exiting.")
            exit(1)  
        else:
            logging.info(f"Checkpoint set to: {args.checkpoint}")

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Initialize trainer with the custom loss function
    trainer = GraphPredictionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        nepochs=args.nepochs,
        save_checkpoint_every=args.save_checkpoint_every,
        results_folder=results_folder,
        checkpoint=args.checkpoint,
        random_seed=args.random_seed,
        device=device,
        verbose=args.verbose,
        criterion=criterion,
        wandb_config=wandb_config,  # Pass WandB config to the trainer
    )
    logging.info("Initialized GraphPredictionTrainer with custom loss function.")
    
    # Save metadata
    save_metadata(args, model, results_folder)

    # Run train or evaluate
    if args.mode == 'train':
        trainer.train()
    else:
        # Implement evaluation if needed
        logging.info("Evaluation mode is not implemented yet.")
        pass
    
    trainer.finalize()  # Finalize WandB after training
