# step_pair_train.py

import torch
import torch.nn.functional as F
from src.datasets.datasets import StepPairGraphDataset
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

from trainers_accelerate import GraphPredictionTrainer
from step_pair_utils import (
    generate_results_folder_name,
    save_metadata,
    get_scheduler,
    set_random_seed
)
from step_pair_config import parse_args
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
    return model_name.lower().endswith('-ae') or model_name.lower() in ['multiscale-topk']

if __name__ == "__main__":
    args = parse_args()

    # Set device
    if args.cpu_only:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Generate data directory
    graph_data_dir = os.path.join(args.base_data_dir, args.dataset, f"{args.data_keyword}_graphs")
    logging.info(f"Graph data directory: {graph_data_dir}")

    # Generate results folder name
    if args.results_folder is not None:
        results_folder = args.results_folder
    else:
        results_folder = generate_results_folder_name(args)
    logging.info(f"Results will be saved to {results_folder}")

    # Set random seed
    set_random_seed(args.random_seed)

    # Determine if the model requires edge_attr
    models_requiring_edge_attr = [
        'intgnn', 'gtr', 'mgn', 'gtr-ae', 'mgn-ae', 
        'singlescale', 'multiscale', 'multiscale-topk'
    ]  # Adjusted list
    use_edge_attr = args.model.lower() in models_requiring_edge_attr
    logging.info(f"Model '{args.model}' requires edge_attr: {use_edge_attr}")

    # Initialize dataset
    dataset = StepPairGraphDataset(
        graph_data_dir=graph_data_dir,
        initial_step=args.initial_step,
        final_step=args.final_step,
        task=args.task,
        use_settings=args.use_settings,
        identical_settings=args.identical_settings,
        settings_file=args.settings_file,
        use_edge_attr=use_edge_attr,
        subsample_size=args.subsample_size
    )

    # Subset dataset if ntrain is specified
    total_dataset_size = len(dataset)
    if args.ntrain is not None:
        np.random.seed(args.random_seed)  # For reproducibility
        indices = np.random.permutation(total_dataset_size)[:args.ntrain]
        dataset = Subset(dataset, indices)
        
    # After initializing the dataset and subset
    for i in range(min(5, len(dataset))):
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
    # Reuse your model initialization code from your old train.py, adjusting as necessary.

    # Example placeholder code (you need to replace this with your actual model initialization code):
    model = GraphConvolutionNetwork(
        in_channels=sample.x.shape[1],
        hidden_dim=args.hidden_dim,
        out_channels=sample.y.shape[1],
        num_layers=args.num_layers,
        pool_ratios=args.pool_ratios,
    )
    logging.info(f"Initialized model: {args.model}")

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
            logging.info(f"Checkpoint set to: {checkpoint_path}")

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Initialize trainer with the loss function
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
        criterion=criterion
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
