import torch
import torch.nn as nn
import torch.optim as optim
import logging

from dataloaders import PointsDataLoaders
from trainers import PointsTrainer
from config import parse_args
from utils import generate_results_folder_name, save_metadata, set_random_seed, get_scheduler
from models import PointNet1, PointNet2, PointNet3, PointNetRegression

def test_models():
    import torch

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define hidden_dim values to test
    hidden_dims = [32, 64, 128]

    for hidden_dim in hidden_dims:
        print(f"\nTesting models with hidden_dim={hidden_dim}")
        
        # Initialize pn0 (PointNetRegression)
        pn0_model = PointNetRegression(
            output_dim=6,
            feature_transform=False,
            input_dim=6,
            settings_dim=6,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Initialize pn3 (PN3Model)
        pn3_model = PointNet3(
            num_input_channels=6,
            num_settings=6,
            num_output_channels=6,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Create dummy data
        batch_size = 2
        num_points = 1024
        initial_state = torch.randn(batch_size, num_points, 6).to(device)  # (batch_size, num_input_channels, n_pts)
        settings = torch.randn(batch_size, 6).to(device)  # (batch_size, num_settings)
        final_state = torch.randn(batch_size, num_points, 6).to(device)  # (batch_size, num_particles, 6)
        
        # Test pn0
        pn0_model.eval()
        with torch.no_grad():
            pn0_outputs = pn0_model(initial_state, settings)
            print(f"pn0_output shape: {pn0_outputs.shape}")  # Expected: (batch_size, num_points, 6)
        
        # Test pn3
        pn3_model.eval()
        with torch.no_grad():
            pn3_outputs = pn3_model(initial_state, settings)
            print(f"pn3_output shape: {pn3_outputs.shape}")  # Expected: (batch_size, num_points, 6)

# Call the testing function
test_models()
