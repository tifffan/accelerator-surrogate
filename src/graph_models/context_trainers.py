# trainers.py

import torch
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer

from src.graph_models.context_models.context_graph_networks import (
    GeneralGraphNetwork,
    ConditionalGraphNetwork,
    AttentionConditionalGraphNetwork
)
# Removed imports of outdated models

# Import accelerate library
from accelerate import Accelerator

def identify_model_type(model):
    """
    Identifies the type of the model and returns a string identifier.
    """
    if isinstance(model, GeneralGraphNetwork):
        return 'GeneralGraphNetwork'
    elif isinstance(model, ConditionalGraphNetwork):
        return 'ConditionalGraphNetwork'
    elif isinstance(model, AttentionConditionalGraphNetwork):
        return 'AttentionConditionalGraphNetwork'
    else:
        return 'UnknownModel'

class BaseTrainer:
    def __init__(self, model, dataloader, optimizer, scheduler=None, device='cpu', **kwargs):
        # Initialize the accelerator
        self.accelerator = Accelerator()
        
        # Identify and store the model type
        self.model_type = identify_model_type(model)
        logging.info(f"Identified model type: {self.model_type}")
        
        # Initialize attributes
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = self.accelerator.device  # Use accelerator's device
        
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
        
        # Initialize WandB only if main process
        self.init_wandb(kwargs.get("wandb_config", None))
        
        # Watch the model if WandB is enabled
        if self.wandb_logger:
            self.wandb_watch_model()

        # Prepare the model, optimizer, scheduler, and dataloader
        if self.scheduler:
            self.model, self.optimizer, self.scheduler, self.dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.dataloader)
        else:
            self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.dataloader)

    def init_wandb(self, wandb_config):
        """Initialize WandB logging if the current process is the main one."""
        if self.accelerator.is_main_process and wandb_config:
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
        self.accelerator.wait_for_everyone()
        for epoch in range(self.start_epoch, self.nepochs):
            self.model.train()
            total_loss = 0
            # Adjust progress bar for distributed training
            if self.verbose and self.accelerator.is_main_process:
                progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.nepochs}")
            else:
                progress_bar = self.dataloader
            for data in progress_bar:
                # No need to move data to device; accelerator handles it
                self.optimizer.zero_grad()
                loss = self.train_step(data)
                # Use accelerator's backward method
                self.accelerator.backward(loss)
                self.optimizer.step()
                total_loss += loss.item()
                if self.verbose and self.accelerator.is_main_process:
                    progress_bar.set_postfix(loss=total_loss / len(self.dataloader))
                    
            self.accelerator.wait_for_everyone()
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.verbose:
                    logging.info(f"Epoch {epoch+1}: Learning rate adjusted to {current_lr}")

            # Save loss history
            avg_loss = total_loss / len(self.dataloader)
            self.loss_history.append(avg_loss)
            if self.accelerator.is_main_process:
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
        if self.accelerator.is_main_process:
            self.plot_loss_convergence()
            logging.info("Training complete!")

    def train_step(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

    def save_checkpoint(self, epoch):
        # Unwrap the model to get the original model (not wrapped by accelerator)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        checkpoint_path = self.checkpoints_folder / f'model-{epoch}.pth'
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch + 1,  # Save the next epoch to resume from
            'random_seed': self.random_seed,
            'loss_history': self.loss_history,
        }
        # Use accelerator's save method
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save(checkpoint_data, checkpoint_path)
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
        if self.accelerator.is_main_process:
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
        if self.accelerator.is_main_process and self.wandb_logger:
            self.wandb_logger.finish()

class GraphPredictionTrainer(BaseTrainer):
    def __init__(self, criterion=None, **kwargs):
        # Pop out values from kwargs to prevent duplication
        model = kwargs.pop('model', None)
        dataloader = kwargs.pop('dataloader', None)
        optimizer = kwargs.pop('optimizer', None)
        scheduler = kwargs.pop('scheduler', None)
        device = kwargs.pop('device', 'cpu')
        
        # Pass the remaining kwargs to the parent class
        super().__init__(model=model,
                         dataloader=dataloader,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         device=device,
                         **kwargs)
        
        # Set the loss function
        if criterion is not None:
            self.criterion = criterion
        else:
            # Default loss function for regression tasks
            self.criterion = torch.nn.MSELoss()
        
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")

    def train_step(self, data):
        x_pred = self.model_forward(data)
        loss = self.criterion(x_pred, data.y)
        
        # Log step-wise loss to WandB
        if self.wandb_logger and self.accelerator.is_main_process:
            self.wandb_logger.log({"step_loss": loss.item()})
            
        return loss

    def model_forward(self, data):
        """
        Calls the model's forward method based on the identified model type.
        """
        model_type = self.model_type
        if model_type == 'GeneralGraphNetwork':
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                u=data.set,
                batch=data.batch
            )
        elif model_type == 'ConditionalGraphNetwork':
            
            print("data.set.shape", data.set.shape)
            
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                conditions=data.set,
                batch=data.batch
            )
        elif model_type == 'AttentionConditionalGraphNetwork':
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                conditions=data.set,
                batch=data.batch
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return x_pred
