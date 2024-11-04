# datasets.py

import os
import torch
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, initial_graph_dir, final_graph_dir, settings_dir, task='predict_n6d'):
        self.initial_graph_dir = initial_graph_dir
        self.final_graph_dir = final_graph_dir
        self.settings_dir = settings_dir
        self.task = task

        # Get list of graph files and ensure they correspond
        self.initial_files = sorted([f for f in os.listdir(initial_graph_dir) if f.endswith('.pt')])
        self.final_files = sorted([f for f in os.listdir(final_graph_dir) if f.endswith('.pt')])
        self.settings_files = sorted([f for f in os.listdir(settings_dir) if f.endswith('.pt')])

        assert len(self.initial_files) == len(self.final_files) == len(self.settings_files), \
            "Mismatch in number of initial graphs, final graphs, and settings files."

    def __len__(self):
        return len(self.initial_files)

    def __getitem__(self, idx):
        # Load initial graph
        initial_filepath = os.path.join(self.initial_graph_dir, self.initial_files[idx])
        initial_data = torch.load(initial_filepath)

        # Load final graph
        final_filepath = os.path.join(self.final_graph_dir, self.final_files[idx])
        final_data = torch.load(final_filepath)

        # Load settings
        settings_filepath = os.path.join(self.settings_dir, self.settings_files[idx])
        settings = torch.load(settings_filepath)  # Shape: [settings_dim]

        # Concatenate settings to each node's feature in the initial graph
        num_nodes = initial_data.num_nodes
        settings_expanded = settings.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, settings_dim]
        initial_data.x = torch.cat([initial_data.x, settings_expanded], dim=1)  # Concatenate settings to node features

        # Adjust target features based on the task
        if self.task == 'predict_n6d':
            initial_data.y = final_data.x[:, :6]  # Use first 6 features
        elif self.task == 'predict_n4d':
            initial_data.y = final_data.x[:, :4]  # Use first 4 features
        elif self.task == 'predict_n2d':
            initial_data.y = final_data.x[:, :2]  # Use first 2 features
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Include 'batch' attribute if not present
        if not hasattr(initial_data, 'batch'):
            initial_data.batch = torch.zeros(initial_data.num_nodes, dtype=torch.long)

        # Return the initial_data object with input features and target
        return initial_data
