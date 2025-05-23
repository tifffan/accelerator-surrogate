# step_pair_train.py

import torch
import torch.nn.functional as F
from src.graph_models.dataloaders import StepPairGraphDataset
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

from trainers import GraphPredictionTrainer
from step_pair_utils import (
    generate_results_folder_name,
    save_metadata,
    get_scheduler,
    set_random_seed
)
from step_pair_config import parse_args

# Import the data loaders
from dataloaders import StepPairGraphDataLoaders

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

    # Initialize data loaders
    data_loaders = StepPairGraphDataLoaders(
        graph_data_dir=graph_data_dir,
        initial_step=args.initial_step,
        final_step=args.final_step,
        task='predict_n6d',
        use_settings=args.use_settings,
        identical_settings=args.identical_settings,
        settings_file=args.settings_file,
        use_edge_attr=use_edge_attr,
        edge_attr_method=args.edge_attr_method,
        preload_data=args.preload_data,
        batch_size=args.batch_size,
        n_train=args.ntrain,
        n_val=args.nval,
        n_test=args.ntest,
    )
    logging.info(f"Initialized StepPairGraphDataLoaders with n_train={args.ntrain}, n_val={args.nval}, n_test={args.ntest}")

    # Get data loaders
    train_loader = data_loaders.get_train_loader()
    val_loader = data_loaders.get_val_loader()
    test_loader = data_loaders.get_test_loader()

    # Get a sample data for model initialization
    sample_batch = next(iter(train_loader))  # Get a batch
    sample = sample_batch.to(device)
    # Since sample is a batch of data objects, get the first element
    if isinstance(sample, list) or isinstance(sample, tuple):
        sample = sample[0]
    else:
        sample = sample

    # Model initialization
    # Model initialization begins
    if is_autoencoder_model(args.model):
        # Autoencoder models: 'gcn-ae', 'gat-ae', 'gtr-ae', 'mgn-ae', 'multiscale-topk'
        # Assert that args.num_layers is even
        if args.num_layers % 2 != 0:
            raise ValueError(f"For autoencoder models, 'num_layers' must be an even number. Received: {args.num_layers}")

        # Calculate depth as half of num_layers
        depth = args.num_layers // 2
        logging.info(f"Autoencoder selected. Using depth: {depth} (num_layers: {args.num_layers})")

        # Adjust pool_ratios to match depth - 1 (since pooling layers = depth -1)
        required_pool_ratios = depth - 1
        current_pool_ratios = len(args.pool_ratios)

        if required_pool_ratios <= 0:
            args.pool_ratios = []
            logging.info(f"No pooling layers required for depth {depth}.")
        elif current_pool_ratios < required_pool_ratios:
            # Pad pool_ratios with 1.0 to match required_pool_ratios
            args.pool_ratios += [1.0] * (required_pool_ratios - current_pool_ratios)
            logging.warning(f"Pool ratios were padded with 1.0 to match required_pool_ratios: {required_pool_ratios}")
        elif current_pool_ratios > required_pool_ratios:
            # Trim pool_ratios to match required_pool_ratios
            args.pool_ratios = args.pool_ratios[:required_pool_ratios]
            logging.warning(f"Pool ratios were trimmed to match required_pool_ratios: {required_pool_ratios}")

        # Initialize the corresponding autoencoder model
        if args.model.lower() == 'gcn-ae':
            in_channels = sample.x.shape[1]
            hidden_dim = args.hidden_dim
            out_channels = sample.y.shape[1]
            pool_ratios = args.pool_ratios

            model = GraphConvolutionalAutoEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                out_channels=out_channels,
                depth=depth,
                pool_ratios=pool_ratios
            )
            logging.info("Initialized GraphConvolutionalAutoEncoder.")

        elif args.model.lower() == 'gat-ae':
            in_channels = sample.x.shape[1]
            hidden_dim = args.hidden_dim
            out_channels = sample.y.shape[1]
            pool_ratios = args.pool_ratios
            heads = args.gat_heads  # Ensure this argument exists

            model = GraphAttentionAutoEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                out_channels=out_channels,
                depth=depth,
                pool_ratios=pool_ratios,
                heads=heads
            )
            logging.info("Initialized GraphAttentionAutoEncoder.")

        elif args.model.lower() == 'gtr-ae':
            in_channels = sample.x.shape[1]
            hidden_dim = args.hidden_dim
            out_channels = sample.y.shape[1]
            pool_ratios = args.pool_ratios
            num_heads = args.gtr_heads  # Ensure this argument exists
            concat = args.gtr_concat    # Ensure this argument exists
            dropout = args.gtr_dropout  # Ensure this argument exists
            edge_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else None

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

        elif args.model.lower() == 'mgn-ae':
            node_in_dim = sample.x.shape[1]
            edge_in_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0
            node_out_dim = sample.y.shape[1]
            hidden_dim = args.hidden_dim
            pool_ratios = args.pool_ratios

            model = MeshGraphAutoEncoder(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                node_out_dim=node_out_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                pool_ratios=pool_ratios
            )
            logging.info("Initialized MeshGraphAutoEncoder.")

        elif args.model.lower() == 'multiscale-topk':
            # Initialize TopkMultiscaleGNN
            input_node_channels = sample.x.shape[1]
            input_edge_channels = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0
            hidden_channels = args.hidden_dim
            output_node_channels = sample.y.shape[1]
            n_mlp_hidden_layers = args.multiscale_n_mlp_hidden_layers
            n_mmp_layers = args.multiscale_n_mmp_layers
            n_messagePassing_layers = args.multiscale_n_message_passing_layers
            max_level_mmp = args.num_layers // 2 - 1  # n_levels = max_level + 1
            max_level_topk = args.num_layers // 2 - 1
            pool_ratios = args.pool_ratios

            # Compute l_char (characteristic length scale)
            edge_index = sample.edge_index
            pos = sample.pos
            edge_lengths = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
            l_char = edge_lengths.mean().item()
            logging.info(f"Computed l_char (characteristic length scale): {l_char}")

            name = 'topk_multiscale_gnn'

            # Initialize model
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
            raise ValueError(f"Unknown autoencoder model {args.model}")

    else:
        # Non-autoencoder models: 'gcn', 'gat', 'gtr', 'mgn', 'singlescale', 'multiscale'
        if args.model.lower() == 'singlescale':
            # Initialize SinglescaleGNN
            input_node_channels = sample.x.shape[1]
            input_edge_channels = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0
            hidden_channels = args.hidden_dim
            output_node_channels = sample.y.shape[1]
            n_mlp_hidden_layers = 0  # Set based on MeshGraphNet
            n_messagePassing_layers = args.num_layers

            name = 'singlescale_gnn'

            # Initialize model
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

        elif args.model.lower() == 'multiscale':
            # Initialize MultiscaleGNN
            input_node_channels = sample.x.shape[1]
            input_edge_channels = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0
            hidden_channels = args.hidden_dim
            output_node_channels = sample.y.shape[1]
            n_mlp_hidden_layers = args.multiscale_n_mlp_hidden_layers
            n_mmp_layers = args.multiscale_n_mmp_layers
            n_messagePassing_layers = args.multiscale_n_message_passing_layers
            max_level = args.num_layers // 2 - 1  # n_levels = max_level + 1

            # Compute l_char (characteristic length scale)
            edge_index = sample.edge_index
            pos = sample.pos
            edge_lengths = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
            l_char = edge_lengths.mean().item()
            logging.info(f"Computed l_char (characteristic length scale): {l_char}")

            name = 'multiscale_gnn'

            # Initialize model
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

        elif args.model.lower() == 'gcn':
            # Ensure pool_ratios length matches num_layers - 2
            required_pool_ratios = args.num_layers - 2
            current_pool_ratios = len(args.pool_ratios)

            if required_pool_ratios <= 0:
                args.pool_ratios = []
                logging.info(f"No pooling layers required for num_layers {args.num_layers}.")
            elif current_pool_ratios < required_pool_ratios:
                args.pool_ratios += [1.0] * (required_pool_ratios - current_pool_ratios)
                logging.warning(f"Pool ratios were padded with 1.0 to match required_pool_ratios: {required_pool_ratios}")
            elif current_pool_ratios > required_pool_ratios:
                args.pool_ratios = args.pool_ratios[:required_pool_ratios]
                logging.warning(f"Pool ratios were trimmed to match required_pool_ratios: {required_pool_ratios}")

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
            logging.info("Initialized GraphConvolutionNetwork model.")

        elif args.model.lower() == 'gat':
            # Ensure pool_ratios length matches num_layers - 2
            required_pool_ratios = args.num_layers - 2
            current_pool_ratios = len(args.pool_ratios)

            if required_pool_ratios <= 0:
                args.pool_ratios = []
                logging.info(f"No pooling layers required for num_layers {args.num_layers}.")
            elif current_pool_ratios < required_pool_ratios:
                args.pool_ratios += [1.0] * (required_pool_ratios - current_pool_ratios)
                logging.warning(f"Pool ratios were padded with 1.0 to match required_pool_ratios: {required_pool_ratios}")
            elif current_pool_ratios > required_pool_ratios:
                args.pool_ratios = args.pool_ratios[:required_pool_ratios]
                logging.warning(f"Pool ratios were trimmed to match required_pool_ratios: {required_pool_ratios}")

            # Model parameters specific to GraphAttentionNetwork
            in_channels = sample.x.shape[1]
            hidden_dim = args.hidden_dim
            out_channels = sample.y.shape[1]
            num_layers = args.num_layers
            pool_ratios = args.pool_ratios
            heads = args.gat_heads  # Ensure this argument exists

            # Initialize GraphAttentionNetwork model
            model = GraphAttentionNetwork(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                out_channels=out_channels,
                num_layers=num_layers,
                pool_ratios=pool_ratios,
                heads=heads,
            )
            logging.info("Initialized GraphAttentionNetwork model.")

        elif args.model.lower() == 'gtr':
            # Model parameters specific to GraphTransformer
            in_channels = sample.x.shape[1]
            hidden_dim = args.hidden_dim
            out_channels = sample.y.shape[1]
            num_layers = args.num_layers
            pool_ratios = args.pool_ratios
            num_heads = args.gtr_heads
            concat = args.gtr_concat
            dropout = args.gtr_dropout
            edge_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else None

            # Adjust pool_ratios to match num_layers - 2
            required_pool_ratios = num_layers - 2
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
            logging.info("Initialized GraphTransformer model.")

        elif args.model.lower() == 'mgn':
            # Model parameters specific to MeshGraphNet
            node_in_dim = sample.x.shape[1]
            edge_in_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0
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
            logging.info("Initialized MeshGraphNet model.")

        else:
            raise ValueError(f"Unknown model {args.model}")
    logging.info(f"Initialized model: {args.model}")
    # Model initialization ends

    # # Example placeholder code (you need to replace this with your actual model initialization code):
    # model = GraphConvolutionNetwork(
    #     in_channels=sample.x.shape[1],
    #     hidden_dim=args.hidden_dim,
    #     out_channels=sample.y.shape[1],
    #     num_layers=args.num_layers,
    #     pool_ratios=args.pool_ratios,
    # )
    # logging.info(f"Initialized model: {args.model}")

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
        train_loader=train_loader,
        val_loader=val_loader,
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
