# trainers.py

import torch
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from pathlib import Path

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

# Removed accelerate import
# from accelerate import Accelerator

def identify_model_type(model):
    """
    Identifies the type of the model and returns a string identifier.
    """
    if isinstance(model, GNN_TopK):
        return 'GNN_TopK'
    elif isinstance(model, TopkMultiscaleGNN):
        return 'TopkMultiscaleGNN'
    elif isinstance(model, SinglescaleGNN):
        return 'SinglescaleGNN'
    elif isinstance(model, MultiscaleGNN):
        return 'MultiscaleGNN'
    elif isinstance(model, MeshGraphNet):
        return 'MeshGraphNet'
    elif isinstance(model, MeshGraphAutoEncoder):
        return 'MeshGraphAutoEncoder'
    elif isinstance(model, GraphTransformer):
        return 'GraphTransformer'
    elif isinstance(model, GraphTransformerAutoEncoder):
        return 'GraphTransformerAutoEncoder'
    else:
        return 'UnknownModel'

def move_data_to_device(data, device):
    """
    Recursively moves data to the specified device.
    Supports dictionaries, lists, tuples, and torch.Tensor objects.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_data_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_data_to_device(v, device) for v in data)
    else:
        return data

class BaseTrainer:
    def __init__(self, model, dataloader, optimizer, scheduler=None, device=None, wandb_logger=None, **kwargs):
        # Identify device manually
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Identify and store the model type
        self.model_type = identify_model_type(model)
        logging.info(f"Identified model type: {self.model_type}")
        
        # Initialize attributes
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        # Device is already set
        # self.device = self.accelerator.device  # Removed
        
        # Trainer configurations
        self.start_epoch = 0
        self.nepochs = kwargs.get('nepochs', 100)
        self.save_checkpoint_every = kwargs.get('save_checkpoint_every', 10)
        self.results_folder = Path(kwargs.get('results_folder', './results'))
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.loss_history = []
        self.verbose = kwargs.get('verbose', False)
        self.wandb_logger = None
        
        # Create 'checkpoints' subfolder under results_folder
        self.checkpoints_folder = self.results_folder / 'checkpoints'
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

        self.random_seed = kwargs.get('random_seed', 42)

        # Checkpoint
        self.checkpoint = kwargs.get('checkpoint', None)
        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)
        
        # Initialize WandB only if main process (always True in single process)
        self.init_wandb(kwargs.get("wandb_config", None))
        
        # Watch the model if WandB is enabled
        if self.wandb_logger:
            self.wandb_watch_model()

        # Move scheduler to device if it exists
        if self.scheduler:
            self.scheduler = self.scheduler

    def init_wandb(self, wandb_config):
        """Initialize WandB logging if enabled."""
        if wandb_config is not None:
            import wandb
            self.wandb_logger = wandb.init(
                project=wandb_config.get("project", "default-project"),
                config=wandb_config.get("config", {}),
                name=wandb_config.get("name", "default-run"),
            )
            self.wandb_logger.config.update({"results_folder": str(self.results_folder)}, allow_val_change=True)
    
    def wandb_watch_model(self):
        """Watch the model using WandB."""
        if self.wandb_logger:
            import wandb
            wandb.watch(
                self.model,
                log="all",  # Log gradients and parameter updates
                log_freq=100  # Adjust log frequency as needed
            )
            logging.info("WandB is now watching the model for gradients and parameter updates.")
    
    def train(self):
        logging.info("Starting training...")
        for epoch in range(self.start_epoch, self.nepochs):
            self.model.train()
            total_loss = 0
            # Adjust progress bar
            if self.verbose:
                progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.nepochs}")
            else:
                progress_bar = self.dataloader
            for data in progress_bar:
                # Move data to device
                data = move_data_to_device(data, self.device)
                
                self.optimizer.zero_grad()
                loss = self.train_step(data)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if self.verbose:
                    progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
                    
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.verbose:
                    logging.info(f"Epoch {epoch+1}: Learning rate adjusted to {current_lr}")

            # Save loss history
            avg_loss = total_loss / len(self.dataloader)
            self.loss_history.append(avg_loss)
            logging.info(f'Epoch {epoch+1}/{self.nepochs}, Loss: {avg_loss:.4e}')
                
            # Log metrics to WandB
            if self.wandb_logger:
                self.wandb_logger.log({"epoch": epoch + 1, "loss": avg_loss})
                if self.scheduler:
                    self.wandb_logger.log({"learning_rate": current_lr})

            # Save checkpoint
            if (epoch + 1) % self.save_checkpoint_every == 0 or (epoch + 1) == self.nepochs:
                self.save_checkpoint(epoch)

        # Plot loss convergence
        self.plot_loss_convergence()
        logging.info("Training complete!")

    def train_step(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

    def save_checkpoint(self, epoch):
        checkpoint_path = self.checkpoints_folder / f'model-{epoch}.pth'
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch + 1,  # Save the next epoch to resume from
            'random_seed': self.random_seed,
            'loss_history': self.loss_history,
        }
        # Save using torch.save
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Model checkpoint saved to {checkpoint_path}")
        if self.wandb_logger:
            self.wandb_logger.save(str(checkpoint_path))

    def load_checkpoint(self, checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        if 'random_seed' in checkpoint:
            self.random_seed = checkpoint['random_seed']
            logging.info(f"Using random seed from checkpoint: {self.random_seed}")
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        logging.info(f"Resumed training from epoch {self.start_epoch}")

    def plot_loss_convergence(self):
        if True:  # Always true since single process
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_history, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Convergence")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.results_folder / "loss_convergence.png")
            plt.close()

    def finalize(self):
        """Finish the WandB run if it was initialized."""
        if self.wandb_logger:
            self.wandb_logger.finish()

class GraphPredictionTrainer(BaseTrainer):
    def __init__(self, criterion=None, wandb_logger=None, **kwargs):
        # Pop out values from kwargs to prevent duplication
        model = kwargs.pop('model', None)
        dataloader = kwargs.pop('dataloader', None)
        optimizer = kwargs.pop('optimizer', None)
        scheduler = kwargs.pop('scheduler', None)
        device = kwargs.pop('device', None)  # Changed default to None to use automatic detection
        
        # Pass the remaining kwargs to the parent class
        super().__init__(model=model,
                         dataloader=dataloader,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         device=device,
                         wandb_logger=wandb_logger,
                         **kwargs)
        
        # Set the loss function
        if criterion is not None:
            self.criterion = criterion
        else:
            # Default loss function for regression tasks
            self.criterion = torch.nn.MSELoss()
        
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")

    def train_step(self, data):
        # x_pred = self.model_forward(data)
        x_pred = self.model_forward(data)
        loss = self.criterion(x_pred, data['y'])  # Assuming data is a dict-like object with 'y'
        
        # Log step-wise loss to WandB
        if self.wandb_logger:
            self.wandb_logger.log({"step_loss": loss.item()})
            
        return loss

    def model_forward(self, data):
        """
        Calls the model's forward method based on the identified model type.
        """
        model_type = self.model_type
        if model_type == 'GNN_TopK':
            x_pred, _ = self.model(
                x=data['x'],
                edge_index=data['edge_index'],
                edge_attr=data['edge_attr'],
                pos=data['pos'],
                batch=data['batch']
            )
        elif model_type == 'TopkMultiscaleGNN':
            x_pred, mask = self.model(
                x=data['x'],
                edge_index=data['edge_index'],
                pos=data['pos'],
                edge_attr=data['edge_attr'],
                batch=data['batch']
            )
        elif model_type in ['SinglescaleGNN', 'MultiscaleGNN']:
            x_pred = self.model(
                x=data['x'],
                edge_index=data['edge_index'],
                pos=data['pos'],
                edge_attr=data['edge_attr'],
                batch=data['batch']
            )
        elif model_type in ['MeshGraphNet', 'MeshGraphAutoEncoder']:
            # MeshGraphNet and MeshGraphAutoEncoder use edge attributes
            x_pred = self.model(
                x=data['x'],
                edge_index=data['edge_index'],
                edge_attr=data['edge_attr'],
                batch=data['batch']  # Removed 'u=None'
            )
        elif model_type in ['GraphTransformer', 'GraphTransformerAutoEncoder']:
            x_pred = self.model(
                x=data['x'],
                edge_index=data['edge_index'],
                edge_attr=data.get('edge_attr', None),
                batch=data['batch']
            )
        else:  # GraphConvolutionNetwork, GraphAttentionNetwork, etc. TODO: Add specific model types here
            x_pred = self.model(
                x=data['x'],
                edge_index=data['edge_index'],
                batch=data['batch']
            )
        return x_pred
