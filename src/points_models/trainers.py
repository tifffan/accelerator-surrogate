# trainers.py

import torch
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from accelerate import Accelerator

class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None, device='cpu', wandb_logger=None, **kwargs):
        # Initialize the accelerator
        self.accelerator = Accelerator()
        
        # Initialize attributes
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = self.accelerator.device  # Use accelerator's device
        
        # Trainer configurations
        self.start_epoch = 0
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.save_checkpoint_every = kwargs.get('save_checkpoint_every', 10)
        self.results_folder = Path(kwargs.get('results_folder', './results'))
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.loss_history = []
        self.val_loss_history = []
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
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader)
        else:
            self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader)

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
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            running_loss = 0.0
            # Adjust progress bar for distributed training
            if self.verbose and self.accelerator.is_main_process:
                progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            else:
                progress_bar = self.train_loader
            for batch_idx, batch in enumerate(progress_bar):
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss
                loss = self.train_step(batch)

                # Backward pass and optimization
                self.accelerator.backward(loss)
                self.optimizer.step()

                # Accumulate loss
                running_loss += loss.item()

                if self.verbose and self.accelerator.is_main_process:
                    progress_bar.set_postfix(loss=running_loss / (batch_idx + 1))

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.verbose:
                    logging.info(f"Epoch {epoch+1}: Learning rate adjusted to {current_lr}")

            # Save loss history
            avg_loss = running_loss / len(self.train_loader)
            self.loss_history.append(avg_loss)

            # Validation
            if self.val_loader:
                val_loss = self.validate()
                self.val_loss_history.append(val_loss)
            else:
                val_loss = None  # No validation loss

            if self.accelerator.is_main_process:
                if val_loss is not None:
                    logging.info(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
                else:
                    logging.info(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')

                # Log metrics to WandB
                if self.wandb_logger:
                    log_dict = {"epoch": epoch + 1, "train_loss": avg_loss}
                    if val_loss is not None:
                        log_dict["val_loss"] = val_loss
                    if self.scheduler:
                        log_dict["learning_rate"] = current_lr
                    self.wandb_logger.log(log_dict)

            # Save checkpoint
            if (epoch + 1) % self.save_checkpoint_every == 0 or (epoch + 1) == self.num_epochs:
                self.save_checkpoint(epoch)

        # Plot loss convergence
        if self.accelerator.is_main_process:
            self.plot_loss_convergence()
            logging.info("Training complete!")

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.validate_step(batch)
                val_loss += loss.item()
        val_loss /= len(self.val_loader)
        return val_loss

    def train_step(self, batch):
        raise NotImplementedError

    def validate_step(self, batch):
        raise NotImplementedError

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
            'val_loss_history': self.val_loss_history,
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
        if 'val_loss_history' in checkpoint:
            self.val_loss_history = checkpoint['val_loss_history']
        logging.info(f"Resumed training from epoch {self.start_epoch}")

    def plot_loss_convergence(self):
        if self.accelerator.is_main_process:
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
    
    def finalize(self):
        """Finish the WandB run if it was initialized."""
        if self.accelerator.is_main_process and self.wandb_logger:
            self.wandb_logger.finish()

class PointsTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion=None, **kwargs):
        super().__init__(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, **kwargs)
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = torch.nn.MSELoss()
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")

    def train_step(self, batch):
        initial_state, final_state, settings = batch
        outputs = self.model(initial_state, settings)
        loss = self.criterion(outputs, final_state)
        return loss

    def validate_step(self, batch):
        initial_state, final_state, settings = batch
        outputs = self.model(initial_state, settings)
        loss = self.criterion(outputs, final_state)
        return loss
