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

# Import accelerate library
from accelerate import Accelerator

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

class BaseTrainer:
    def __init__(self, model, dataloader, optimizer, scheduler=None, device='cpu', **kwargs):
        # Initialize the accelerator
        self.accelerator = Accelerator()
        
        # Identify and store the model type
        self.model_type = identify_model_type(model)
        logging.info(f"Identified model type: {self.model_type}")
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = self.accelerator.device  # Use accelerator's device
        
        self.start_epoch = 0
        self.nepochs = kwargs.get('nepochs', 100)
        self.save_checkpoint_every = kwargs.get('save_checkpoint_every', 10)
        self.results_folder = Path(kwargs.get('results_folder', './results'))
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.loss_history = []
        self.verbose = kwargs.get('verbose', False)

        # Create 'checkpoints' subfolder under results_folder
        self.checkpoints_folder = self.results_folder / 'checkpoints'
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

        self.random_seed = kwargs.get('random_seed', 42)

        # Checkpoint
        self.checkpoint = kwargs.get('checkpoint', None)
        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)

        # Prepare the model, optimizer, scheduler, and dataloader
        if self.scheduler:
            self.model, self.optimizer, self.scheduler, self.dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.dataloader)
        else:
            self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.dataloader)

    def train(self):
        logging.info("Starting training...")
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
        # # Check edge_attr
        # if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        #     logging.error("data.edge_attr is missing or None.")
        #     raise ValueError("data.edge_attr is missing or None.")
        # else:
        #     logging.info(f"data.edge_attr shape: {data.edge_attr.shape}")
        
        x_pred = self.model_forward(data)
        loss = self.criterion(x_pred, data.y)
        return loss

    def model_forward(self, data):
        """
        Calls the model's forward method based on the identified model type.
        """
        model_type = self.model_type
        if model_type == 'GNN_TopK':
            x_pred, _ = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                pos=data.pos,
                batch=data.batch
            )
        elif model_type == 'TopkMultiscaleGNN':
            x_pred, mask = self.model(
                x=data.x,
                edge_index=data.edge_index,
                pos=data.pos,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        elif model_type in ['SinglescaleGNN', 'MultiscaleGNN']:
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                pos=data.pos,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        elif model_type in ['MeshGraphNet', 'MeshGraphAutoEncoder']:
            # MeshGraphNet and MeshGraphAutoEncoder use edge attributes
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch  # Removed 'u=None'
            )
        elif model_type in ['GraphTransformer', 'GraphTransformerAutoEncoder']:
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
                batch=data.batch
            )
        else:  # GraphConvolutionNetwork, GraphAttentionNetwork, etc. TODO: Add specific model types here
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch
            )
        return x_pred
