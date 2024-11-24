# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import logging

from dataloaders import PointsDataLoaders
from trainers import PointsTrainer
from config import parse_args
from utils import generate_results_folder_name, save_metadata, set_random_seed, get_scheduler
from models import PointNet1 

if __name__ == "__main__":
    args = parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Set random seed
    set_random_seed(args.random_seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Generate results folder name
    results_folder = generate_results_folder_name(args)
    logging.info(f"Results will be saved to {results_folder}")

    # Prepare WandB config
    wandb_config = {
        "project": "points-training",
        "config": vars(args),
        "name": results_folder,
    }

    # Initialize the DataLoaders
    data_loaders = PointsDataLoaders(
        data_catalog=args.data_catalog,
        statistics_file=args.statistics_file,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        random_seed=args.random_seed
    )

    # Retrieve DataLoaders
    # train_loader = data_loaders.get_train_loader()
    # val_loader = data_loaders.get_val_loader()
    # test_loader = data_loaders.get_test_loader()
    train_loader = data_loaders.get_all_data_loader()
    val_loader = None

    # Initialize the model and loss function
    if args.model == 'pn1':
        model = PointNet1(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    else:
        raise ValueError(f"Unknown model {args.model}")
    model.to(device)
    criterion = nn.MSELoss()
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Get scheduler
    scheduler = get_scheduler(args, optimizer)

    # Initialize trainer
    trainer = PointsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        results_folder=results_folder,
        checkpoint=args.checkpoint,
        random_seed=args.random_seed,
        verbose=args.verbose,
        wandb_config=wandb_config,
        criterion=criterion
    )
    logging.info("Initialized ElectronBeamTrainer.")

    # Save metadata
    save_metadata(args, model, results_folder)

    # Run training
    trainer.train()

    trainer.finalize()  # Finalize WandB after training
