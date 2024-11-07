# trainers.py

import torch
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from intgnn.models import GNN_TopK
from src.graph_models.models.graph_networks import MeshGraphNet, GraphTransformer
from src.graph_models.models.graph_autoencoders import MeshGraphAutoEncoder, GraphTransformerAutoEncoder

class BaseTrainer:
    def __init__(self, model, dataloader, optimizer, scheduler=None, device='cpu', **kwargs):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
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

    def train(self):
        logging.info("Starting training...")
        for epoch in range(self.start_epoch, self.nepochs):
            self.model.train()
            total_loss = 0
            if self.verbose:
                progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.nepochs}")
            else:
                progress_bar = self.dataloader
            for data in progress_bar:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                loss = self.train_step(data)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if self.verbose:
                    progress_bar.set_postfix(loss=total_loss / len(self.dataloader))

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch+1}: Learning rate adjusted to {current_lr}")

            # Save loss history
            avg_loss = total_loss / len(self.dataloader)
            self.loss_history.append(avg_loss)
            logging.info(f'Epoch {epoch+1}/{self.nepochs}, Loss: {avg_loss:.4f}')

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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch + 1,  # Save the next epoch to resume from
            'random_seed': self.random_seed,
            'loss_history': self.loss_history,
        }, checkpoint_path)
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
        super().__init__(**kwargs)
        
        # Set the loss function
        if criterion is not None:
            self.criterion = criterion
        else:
            # Default loss function for classification tasks
            self.criterion = torch.nn.MSELoss()
        
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")

    def train_step(self, data):
        x_pred = self.model_forward(data)
        loss = self.criterion(x_pred, data.y)
        return loss

    def model_forward(self, data):
        if isinstance(self.model, GNN_TopK):
            x_pred, _ = self.model(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.pos,
                batch=data.batch
            )
        elif isinstance(self.model, MeshGraphNet) or isinstance(self.model, MeshGraphAutoEncoder):
            # MeshGraphNet uses edge attributes
            x_pred = self.model(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch
            )
        elif isinstance(self.model, GraphTransformer) or isinstance(self.model, GraphTransformerAutoEncoder):
            x_pred = self.model(
                data.x,
                data.edge_index,
                data.edge_attr if hasattr(data, 'edge_attr') else None,
                data.batch
            )
        else: # GraphConvolutionNetwork, GraphAttentionNetwork
            x_pred = self.model(
                data.x,
                data.edge_index,
                data.batch
            )
        return x_pred
