import os

# Adjust NCCL settings to mitigate kernel compatibility issues
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_TIMEOUT"] = "20"

# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import logging

from dataloaders import PointsDataLoaders
from trainers import PointsTrainer
from config import parse_args
from utils import generate_results_folder_name, save_metadata, set_random_seed, get_scheduler
from models import (
    PointNet1,
    PointNet2,
    PointNet3,
    PointNetRegression,
    PointNetRegression_SettingsAtStart,
    PointNetRegression_SettingsAtMiddle,
    PointNetRegression_SettingsAtGlobal
)

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Set up logging with timestamp, log level, and message
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Set random seed for reproducibility
    set_random_seed(args.random_seed)
    logging.info(f"Random seed set to {args.random_seed}")

    # Determine the device to run on (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Generate a unique results folder name based on configuration
    results_folder = generate_results_folder_name(args)
    logging.info(f"Results will be saved to {results_folder}")

    # Prepare WandB (Weights & Biases) configuration for experiment tracking
    wandb_config = {
        "project": "points-training",
        "config": vars(args),
        "name": results_folder,
    }

    # Initialize the DataLoaders for training, validation, and testing
    data_loaders = PointsDataLoaders(
        data_catalog=args.data_catalog,
        statistics_file=args.statistics_file,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        random_seed=args.random_seed
    )

    # Retrieve DataLoaders for training and validation
    train_loader = data_loaders.get_train_loader()
    val_loader = data_loaders.get_val_loader()
    # Optionally, retrieve test_loader if needed for evaluation after training
    # test_loader = data_loaders.get_test_loader()

    # Log the number of samples in each DataLoader
    logging.info(f"Number of training batches: {len(train_loader)}")
    logging.info(f"Number of validation batches: {len(val_loader)}")
    # logging.info(f"Number of test batches: {len(test_loader)}")

    # Initialize the model based on the specified architecture
    if args.model == 'pn1':
        model = PointNet1(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.model == 'pn2':
        model = PointNet2(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.model == 'pn3':
        model = PointNet3(hidden_dim=args.hidden_dim)
    elif args.model == 'pn0':
        model = PointNetRegression(
            input_dim=6,
            output_dim=6,
            hidden_dim=args.hidden_dim,
            STN_dim=args.STN_dim,
            feature_transform=False
        )
    elif args.model == 'pn0-start':
        model = PointNetRegression_SettingsAtStart(
            input_dim=6,
            output_dim=6,
            hidden_dim=args.hidden_dim,
            feature_transform=False
        )
    elif args.model == 'pn0-middle':
        model = PointNetRegression_SettingsAtMiddle(
            input_dim=6,
            output_dim=6,
            hidden_dim=args.hidden_dim,
            feature_transform=False
        )
    elif args.model == 'pn0-global':
        model = PointNetRegression_SettingsAtGlobal(
            input_dim=6,
            output_dim=6,
            hidden_dim=args.hidden_dim,
            feature_transform=False
        )
    else:
        raise ValueError(f"Unknown model {args.model}")
    
    # Move the model to the specified device (GPU/CPU)
    model.to(device)
    logging.info(f"Model {args.model} initialized and moved to {device}")

    # Define the loss function (criterion)
    criterion = nn.MSELoss()

    # Initialize optimizer with weight decay (regularization)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    logging.info(f"Optimizer: Adam with learning rate {args.learning_rate} and weight decay {args.weight_decay}")

    # Initialize learning rate scheduler if specified
    scheduler = get_scheduler(args, optimizer)
    if scheduler:
        logging.info(f"Scheduler {args.lr_scheduler} initialized")

    # Initialize the trainer with all necessary components
    trainer = PointsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        save_checkpoint_every=args.save_checkpoint_every,
        results_folder=results_folder,
        checkpoint=args.checkpoint,
        random_seed=args.random_seed,
        verbose=args.verbose,
        wandb_config=wandb_config,
        criterion=criterion
    )
    logging.info("Initialized PointsTrainer.")

    # Save experiment metadata for future reference
    save_metadata(args, model, results_folder)
    logging.info("Metadata saved.")

    # Start the training process
    trainer.train()
    logging.info("Training process completed.")

    # Finalize the trainer (e.g., finish WandB run)
    trainer.finalize()
    logging.info("Trainer finalized.")
