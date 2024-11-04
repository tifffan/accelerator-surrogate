# train.py

import torch
import torch.nn.functional as F
from datasets import GraphDataset
from models import GraphConvolutionNetwork, GraphAttentionNetwork, GraphTransformer, MeshGraphNet
from intgnn.models import GNN_TopK
from trainers import GraphPredictionTrainer
from utils import generate_data_dirs, generate_results_folder_name, save_metadata, get_scheduler, set_random_seed
from config import parse_args
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import numpy as np
import logging
import re
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    args = parse_args()

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

    # Initialize dataset
    dataset = GraphDataset(initial_graph_dir, final_graph_dir, settings_dir, task=args.task)

    # Subset dataset if ntrain is specified
    total_dataset_size = len(dataset)
    if args.ntrain is not None:
        np.random.seed(args.random_seed)  # For reproducibility
        indices = np.random.permutation(total_dataset_size)[:args.ntrain]
        dataset = Subset(dataset, indices)

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Get a sample data for model initialization
    sample = dataset[0]

    # Model initialization
    if args.model == 'intgnn':
        # Model parameters specific to GNN_TopK
        in_channels_node = sample.x.shape[1]
        in_channels_edge = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') else 0
        hidden_channels = args.hidden_dim
        out_channels = sample.y.shape[1]
        n_mlp_encode = 3
        n_mlp_mp = 2
        n_mp_down_topk = [1, 1]
        n_mp_up_topk = [1, 1]
        pool_ratios = args.pool_ratios
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

        # Initialize GNN_TopK model
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

    elif args.model == 'gcn':
        # Ensure pool_ratios length matches num_layers - 2
        if len(args.pool_ratios) < args.num_layers - 2:
            args.pool_ratios += [1.0] * (args.num_layers - 2 - len(args.pool_ratios))
        elif len(args.pool_ratios) > args.num_layers - 2:
            args.pool_ratios = args.pool_ratios[:args.num_layers - 2]

        # Model parameters specific to GraphConvolutionNetwork
        in_channels = sample.x.shape[1]
        hidden_dim = args.hidden_dim
        out_channels = sample.y.shape[1]
        num_layers = args.num_layers
        pool_ratios = args.pool_ratios

        # Initialize GraphConvolutionNetwork model
        model = GraphConvolutionNetwork(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers,
            pool_ratios=pool_ratios,
        )
        
    elif args.model == 'gat':
        # Ensure pool_ratios length matches num_layers - 2 (since we don't pool after the last layer)
        if len(args.pool_ratios) < args.num_layers - 2:
            # Pad pool_ratios with 1.0 to match num_layers - 2
            args.pool_ratios += [1.0] * (args.num_layers - 2 - len(args.pool_ratios))
        elif len(args.pool_ratios) > args.num_layers - 2:
            # Trim pool_ratios to match num_layers - 2
            args.pool_ratios = args.pool_ratios[:args.num_layers - 2]

        # Model parameters specific to GraphAttentionNetwork
        in_channels = sample.x.shape[1]
        hidden_dim = args.hidden_dim
        out_channels = sample.y.shape[1]
        num_layers = args.num_layers
        pool_ratios = args.pool_ratios
        heads = args.gat_heads

        # Initialize GraphAttentionNetwork model
        model = GraphAttentionNetwork(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers,
            pool_ratios=pool_ratios,
            heads=heads,
        )
    
    elif args.model == 'gtr':
        # Model parameters specific to GraphTransformer
        in_channels = sample.x.shape[1]
        hidden_dim = args.hidden_dim
        out_channels = sample.y.shape[1]
        num_layers = args.num_layers
        pool_ratios = args.pool_ratios
        num_heads = args.gtr_heads
        concat = args.gtr_concat
        dropout = args.gtr_dropout
        edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else None

        # Adjust pool_ratios to match num_layers - 2
        if len(pool_ratios) < num_layers - 2:
            pool_ratios += [1.0] * (num_layers - 2 - len(pool_ratios))
        elif len(pool_ratios) > num_layers - 2:
            pool_ratios = pool_ratios[:num_layers - 2]

        # Initialize GraphTransformer model
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

    
    elif args.model == 'meshgraphnet':
        # Model parameters specific to MeshGraphNet
        node_in_dim = sample.x.shape[1]
        edge_in_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
        node_out_dim = sample.y.shape[1]
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers

        # Initialize MeshGraphNet model
        model = MeshGraphNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            node_out_dim=node_out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    else:
        raise ValueError(f"Unknown model {args.model}")

    model.to(device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = get_scheduler(args, optimizer)

    # Handle checkpoint
    if args.checkpoint is None and args.checkpoint_epoch is not None:
        results_folder_ = re.sub(r'ep\d+', f'ep{args.checkpoint_epoch}', results_folder)
        checkpoint_path = os.path.join(results_folder_, 'checkpoints', f'model-{args.checkpoint_epoch - 1}.pth')
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint for epoch {args.checkpoint_epoch} not found at {checkpoint_path}. Exiting.")
            exit(1)
        else:
            args.checkpoint = checkpoint_path

    if args.checkpoint is not None and args.checkpoint_epoch is None:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found at {checkpoint_path}. Exiting.")
            exit(1)  

    # Initialize trainer
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
    )
    
    # Save metadata
    save_metadata(args, model, results_folder)

    # Run train or evaluate
    if args.mode == 'train':
        trainer.train()
    else:
        # Implement evaluation if needed
        pass
