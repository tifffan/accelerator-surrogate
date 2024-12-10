# trainers.py

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from tqdm import tqdm

from src.graph_models.context_models.context_graph_networks import (
    AttentionConditionalGraphNetwork,
    ConditionalGraphNetwork,
    GeneralGraphNetwork,
)
from src.graph_models.models.graph_autoencoders import (
    GraphAttentionAutoEncoder,
    GraphConvolutionalAutoEncoder,
    GraphTransformerAutoEncoder,
    MeshGraphAutoEncoder,
)
from src.graph_models.models.graph_networks import (
    GraphAttentionNetwork,
    GraphConvolutionNetwork,
    GraphTransformer,
    MeshGraphNet,
)
from src.graph_models.models.intgnn.models import GNN_TopK
from src.graph_models.models.multiscale.gnn import (
    MultiscaleGNN,
    SinglescaleGNN,
    TopkMultiscaleGNN,
)


def identify_model_type(model) -> str:
    """
    Identifies the type of the model and returns a string identifier.
    """
    model_types = {
        GNN_TopK: 'GNN_TopK',
        TopkMultiscaleGNN: 'TopkMultiscaleGNN',
        SinglescaleGNN: 'SinglescaleGNN',
        MultiscaleGNN: 'MultiscaleGNN',
        MeshGraphNet: 'MeshGraphNet',
        MeshGraphAutoEncoder: 'MeshGraphAutoEncoder',
        GraphTransformer: 'GraphTransformer',
        GraphTransformerAutoEncoder: 'GraphTransformerAutoEncoder',
        GeneralGraphNetwork: 'GeneralGraphNetwork',
        ConditionalGraphNetwork: 'ConditionalGraphNetwork',
        AttentionConditionalGraphNetwork: 'AttentionConditionalGraphNetwork',
        GraphConvolutionNetwork: 'GraphConvolutionNetwork',
        GraphAttentionNetwork: 'GraphAttentionNetwork',
    }

    for cls, name in model_types.items():
        if isinstance(model, cls):
            return name
    return 'UnknownModel'


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = 'cpu',
        **kwargs,
    ):
        # Initialize the accelerator for distributed training
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Identify and store the model type
        self.model_type = identify_model_type(model)
        logging.info(f"Identified model type: {self.model_type}")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.start_epoch = 0
        self.nepochs = kwargs.get('nepochs', 100)
        self.save_checkpoint_every = kwargs.get('save_checkpoint_every', 10)
        self.results_folder = Path(kwargs.get('results_folder', './results'))
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.checkpoints_folder = self.results_folder / 'checkpoints'
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

        self.loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1

        self.verbose = kwargs.get('verbose', False)
        self.random_seed = kwargs.get('random_seed', 42)

        # Handle checkpoint loading
        checkpoint_path = kwargs.get('checkpoint')
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # Prepare the model, optimizer, scheduler, and dataloaders with Accelerator
        prepare_args = [self.model, self.optimizer]
        if self.scheduler:
            prepare_args.append(self.scheduler)
        prepare_args.extend([self.train_loader, self.val_loader])

        prepared = self.accelerator.prepare(*prepare_args)
        self.model, self.optimizer = prepared[:2]
        if self.scheduler:
            self.scheduler = prepared[2]
            self.train_loader, self.val_loader = prepared[3], prepared[4]
        else:
            self.train_loader, self.val_loader = prepared[2], prepared[3]

    def train(self):
        logging.info("Starting training...")
        for epoch in range(self.start_epoch, self.nepochs):
            self.model.train()
            total_loss = 0.0

            # Setup progress bar
            if self.verbose and self.accelerator.is_main_process:
                progress_bar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.nepochs}",
                    disable=not self.verbose,
                )
            else:
                progress_bar = self.train_loader

            for batch_idx, data in enumerate(progress_bar):
                self.optimizer.zero_grad()
                loss = self.train_step(data)
                self.accelerator.backward(loss)
                self.optimizer.step()
                total_loss += loss.item()

                if self.verbose and self.accelerator.is_main_process:
                    current_loss = total_loss / (batch_idx + 1)
                    progress_bar.set_postfix(loss=f"{current_loss:.4e}")

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.verbose:
                    logging.info(f"Epoch {epoch + 1}: Learning rate adjusted to {current_lr}")

            # Record training loss
            avg_loss = total_loss / len(self.train_loader)
            self.loss_history.append(avg_loss)

            # Validation
            val_loss = None
            if self.val_loader:
                val_loss = self.validate()
                self.val_loss_history.append(val_loss)

            # Logging
            if self.accelerator.is_main_process:
                if val_loss is not None:
                    logging.info(
                        f"Epoch {epoch + 1}/{self.nepochs}, "
                        f"Loss: {avg_loss:.4e}, Val Loss: {val_loss:.4e}"
                    )
                else:
                    logging.info(f"Epoch {epoch + 1}/{self.nepochs}, Loss: {avg_loss:.4e}")

            # Save checkpoint
            if (epoch + 1) % self.save_checkpoint_every == 0 or (epoch + 1) == self.nepochs:
                self.save_checkpoint(epoch)

            # Save the best model based on validation loss
            if val_loss is not None and val_loss < self.best_val_loss:
                logging.info(
                    f"New best model found at epoch {epoch + 1} with Val Loss: {val_loss:.4e}"
                )
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, best=True)

        # Finalize training
        if self.accelerator.is_main_process:
            self.plot_loss_convergence()
            logging.info("Training complete!")
            logging.info(f"Best Val Loss: {self.best_val_loss:.4e} at epoch {self.best_epoch}")

    def validate(self) -> float:
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in self.val_loader:
                loss = self.validate_step(data)
                val_loss += loss.item()
                num_batches += 1

        # Aggregate losses across all processes
        total_val_loss = torch.tensor(val_loss, device=self.accelerator.device)
        total_num_batches = torch.tensor(num_batches, device=self.accelerator.device)

        total_val_loss = self.accelerator.gather(total_val_loss).sum()
        total_num_batches = self.accelerator.gather(total_num_batches).sum()

        avg_val_loss = total_val_loss / total_num_batches
        return avg_val_loss.item()

    def train_step(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

    def validate_step(self, data):
        self.model.eval()
        with torch.no_grad():
            return self.train_step(data)

    def save_checkpoint(self, epoch: int, best: bool = False):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        checkpoint_filename = 'best_model.pth' if best else f'model-{epoch + 1}.pth'
        checkpoint_path = self.checkpoints_folder / checkpoint_filename

        checkpoint_data = {
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch + 1,
            'random_seed': self.random_seed,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save(checkpoint_data, checkpoint_path)
            logging.info(f"Model checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0)
        self.random_seed = checkpoint.get('random_seed', self.random_seed)
        self.loss_history = checkpoint.get('loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', -1)

        logging.info(f"Resumed training from epoch {self.start_epoch}")

    def plot_loss_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label="Training Loss")
        if self.val_loss_history:
            plt.plot(self.val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Convergence")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.results_folder / "loss_convergence.png")
        plt.close()


class GraphPredictionTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        criterion: torch.nn.Module = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = 'cpu',
        **kwargs,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            **kwargs,
        )

        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")

    def train_step(self, data):
        outputs = self.model_forward(data)
        loss = self.criterion(outputs, data.y)
        return loss

    def validate_step(self, data):
        outputs = self.model_forward(data)
        loss = self.criterion(outputs, data.y)
        return loss

    def model_forward(self, data):
        model_type = self.model_type
        kwargs = {
            'x': data.x,
            'edge_index': data.edge_index,
            'batch': data.batch,
        }

        # Common attributes
        if hasattr(data, 'edge_attr'):
            kwargs['edge_attr'] = data.edge_attr
            
        # Additional attributes based on model type
        if model_type in ['GNN_TopK', 'TopkMultiscaleGNN', 'SinglescaleGNN', 'MultiscaleGNN']:
            kwargs['pos'] = data.pos
            return self.model(**kwargs)
        elif model_type in ['MeshGraphNet', 'MeshGraphAutoEncoder']:
            return self.model(**kwargs)
        elif model_type in ['GraphTransformer', 'GraphTransformerAutoEncoder']:
            kwargs['edge_attr'] = data.edge_attr if hasattr(data, 'edge_attr') else None
            return self.model(**kwargs)
        elif model_type in ['GraphConvolutionNetwork', 'GraphAttentionNetwork']:
            return self.model(**kwargs)
        elif model_type == 'GeneralGraphNetwork':
            kwargs['u'] = data.set
            return self.model(**kwargs)
        elif model_type in ['ConditionalGraphNetwork', 'AttentionConditionalGraphNetwork']:
            kwargs['conditions'] = data.set
            return self.model(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
