# model.py

import torch
import torch.nn as nn

# Residual MLP Block
class ResidualMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.relu2 = nn.ReLU()
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out += identity
        out = self.relu2(out)
        return out

# PointNet Model with Configurable Hidden Dimension and Layers
class PointNet1(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3):
        super(PointNet1, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initialize the list of residual blocks
        layers = []
        input_dim = 6  # Initial input feature dimension

        # Create residual layers
        for i in range(num_layers):
            if i == 0:
                layers.append(ResidualMLP(input_dim, hidden_dim))
            else:
                layers.append(ResidualMLP(hidden_dim, hidden_dim))
        self.res_blocks = nn.ModuleList(layers)

        # Global feature MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # Per-point MLPs for final prediction
        self.mlp4 = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.mlp5 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 6)  # Output per-point features of size 6
        )

    def forward(self, initial_state, settings):
        # initial_state: (batch_size, num_points, 6)
        # settings: (batch_size, 6)
        x = initial_state  # (B, N, 6)

        # Apply residual MLP blocks
        for res_block in self.res_blocks:
            x = res_block(x)  # (B, N, hidden_dim)

        # Obtain global features via max pooling
        x_global = torch.max(x, dim=1)[0]  # (B, hidden_dim)

        # Concatenate settings to global features
        x_global = torch.cat([x_global, settings], dim=1)  # (B, hidden_dim + 6)

        # Process global features
        x_global = self.global_mlp(x_global)  # (B, hidden_dim)

        # Expand and concatenate global features with per-point features
        x_global_expanded = x_global.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, N, hidden_dim)
        x_concat = torch.cat([x, x_global_expanded], dim=2)  # (B, N, hidden_dim + hidden_dim)

        # Further per-point processing
        x = self.mlp4(x_concat)  # (B, N, hidden_dim // 2)
        x = self.mlp5(x)  # (B, N, 6)

        return x
