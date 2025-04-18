# context_train.py

import torch
import torch.nn.functional as F
from src.graph_models.context_dataloaders import GraphSettingsDataLoaders
from src.graph_models.context_models.context_graph_networks import (
    ConditionalGraphNetwork,
    AttentionConditionalGraphNetwork,
    GeneralGraphNetwork
)
from trainers import GraphPredictionTrainer
from utils import (
    generate_data_dirs,
    generate_results_folder_name,
    save_metadata,
    get_scheduler,
    set_random_seed
)
from context_config import parse_args
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
    
    # # Prepare WandB config (if using WandB)
    # wandb_config = {
    #     "project": "graph-training",
    #     "config": vars(args),
    #     "name": args.results_folder,
    # }

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

    # Set random seed for reproducibility
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

    # Initialize DataLoaders with new parameters
    data_loaders = GraphSettingsDataLoaders(
        initial_graph_dir=initial_graph_dir,
        final_graph_dir=final_graph_dir,
        settings_dir=settings_dir,
        task=args.task,
        use_edge_attr=use_edge_attr,
        edge_attr_method=args.edge_attr_method if hasattr(args, 'edge_attr_method') else 'v1',
        preload_data=args.preload_data if hasattr(args, 'preload_data') else False,
        batch_size=args.batch_size,
        n_train=args.n_train if hasattr(args, 'n_train') else 1000,
        n_val=args.n_val if hasattr(args, 'n_val') else 200,
        n_test=args.n_test if hasattr(args, 'n_test') else 200
    )
    logging.info(f"Initialized GraphSettingsDataLoaders with {len(data_loaders.train_set)} train, "
                 f"{len(data_loaders.val_set)} val, and {len(data_loaders.test_set)} test samples.")

    # Optionally, log sample edge_attr shapes (only for a few samples)
    num_samples_to_log = min(3, len(data_loaders.train_set))
    for i in range(num_samples_to_log):
        sample = data_loaders.train_set[i]
        if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
            logging.info(f"Train Sample {i} edge_attr shape: {sample.edge_attr.shape}")
        else:
            logging.warning(f"Train Sample {i} is missing edge_attr.")

    # Similarly, log a few validation samples
    for i in range(num_samples_to_log):
        sample = data_loaders.val_set[i]
        if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
            logging.info(f"Validation Sample {i} edge_attr shape: {sample.edge_attr.shape}")
        else:
            logging.warning(f"Validation Sample {i} is missing edge_attr.")

    # Retrieve DataLoaders
    train_loader = data_loaders.get_train_loader()
    val_loader = data_loaders.get_val_loader()
    test_loader = data_loaders.get_test_loader()

    # Get a sample data for model initialization
    sample = data_loaders.dataset[0]

    # Model initialization
    if is_autoencoder_model(args.model):
        # Handle autoencoder models if any
        raise NotImplementedError("Autoencoder models are not defined in the current context.")
    else:
        # Non-autoencoder models
        model_key = args.model.lower()
        if model_key in model_name_mapping:
            model_class_name = model_name_mapping[model_key]
            # Dynamically get the model class from the imported modules
            model_class = globals().get(model_class_name)
            if model_class is None:
                # Attempt to retrieve the class from the imported modules
                try:
                    model_class = getattr(__import__('src.graph_models.context_models.context_graph_networks', fromlist=[model_class_name]), model_class_name)
                except AttributeError:
                    raise ValueError(f"Model class '{model_class_name}' not found in the imported modules.")
            logging.info(f"Initializing model '{model_class_name}'.")

            if model_class_name == 'GeneralGraphNetwork':
                # Initialize GeneralGraphNetwork
                node_in_dim = sample.x.shape[1]
                edge_in_dim = sample.edge_attr.shape[1] if use_edge_attr else 0
                global_in_dim = sample.set.shape[1]  # Using 'set' field from dataset
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
                cond_in_dim = sample.set.shape[1]
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
                cond_in_dim = sample.set.shape[1]
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    logging.info(f"Initialized Adam optimizer with learning rate: {args.lr} and weight decay: {args.wd}")

    # Scheduler
    scheduler = get_scheduler(args, optimizer)
    if scheduler:
        logging.info("Initialized learning rate scheduler.")

    # Handle checkpoint
    if args.checkpoint is None and hasattr(args, 'checkpoint_epoch') and args.checkpoint_epoch is not None:
        results_folder_ = re.sub(r'ep\d+', f'ep{args.checkpoint_epoch}', results_folder)
        checkpoint_path = os.path.join(results_folder_, 'checkpoints', f'model-{args.checkpoint_epoch - 1}.pth')
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint for epoch {args.checkpoint_epoch} not found at {checkpoint_path}. Exiting.")
            exit(1)
        else:
            args.checkpoint = checkpoint_path
            logging.info(f"Checkpoint set to: {args.checkpoint}")

    if args.checkpoint is not None and (not hasattr(args, 'checkpoint_epoch') or args.checkpoint_epoch is None):
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found at {checkpoint_path}. Exiting.")
            exit(1)  
        else:
            logging.info(f"Checkpoint set to: {checkpoint_path}")

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Initialize trainer with the custom loss function and the new DataLoaders
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
        criterion=criterion,
        # wandb_config=wandb_config,  # Pass WandB config to the trainer
    )
    logging.info("Initialized GraphPredictionTrainer with custom loss function and DataLoaders.")
    
    # Save metadata
    save_metadata(args, model, results_folder)

    # Run train or evaluate
    if args.mode == 'train':
        trainer.train()
    else:
        # Implement evaluation if needed
        logging.info("Evaluation mode is not implemented yet.")
        pass
    
    # # Finalize (e.g., finish WandB run)
    # if hasattr(trainer, 'finalize'):
    #     trainer.finalize()
