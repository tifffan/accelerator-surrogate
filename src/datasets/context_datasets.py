from torch_geometric.data import Data  # Import Data class
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import logging
import glob
import re
import os

class GraphSettingsDataset(Dataset):
    def __init__(self, initial_graph_dir, final_graph_dir, settings_dir, task='predict_n6d', use_edge_attr=False):
        """
        Initializes the GraphSettingsDataset.

        Args:
            initial_graph_dir (str): Directory containing initial graph .pt files.
            final_graph_dir (str): Directory containing final graph .pt files.
            settings_dir (str): Directory containing settings .pt files.
            task (str, optional): Prediction task. One of ['predict_n6d', 'predict_n4d', 'predict_n2d'].
                                  Defaults to 'predict_n6d'.
            use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
                                            Defaults to False.
        """
        self.initial_graph_dir = initial_graph_dir
        self.final_graph_dir = final_graph_dir
        self.settings_dir = settings_dir
        self.task = task
        self.use_edge_attr = use_edge_attr

        # Get list of graph files and ensure they correspond
        self.initial_files = sorted([f for f in os.listdir(initial_graph_dir) if f.endswith('.pt')])
        self.final_files = sorted([f for f in os.listdir(final_graph_dir) if f.endswith('.pt')])
        self.settings_files = sorted([f for f in os.listdir(settings_dir) if f.endswith('.pt')])

        assert len(self.initial_files) == len(self.final_files) == len(self.settings_files), \
            "Mismatch in number of initial graphs, final graphs, and settings files."

        logging.info(f"Initialized GraphSettingsDataset with {len(self)} samples. Use edge_attr: {self.use_edge_attr}")

    def __len__(self):
        return len(self.initial_files)

    def __getitem__(self, idx):
        """
        Retrieves the graph data at the specified index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            torch_geometric.data.Data: Graph data object containing node features, edge indices,
                                       edge attributes (if required), target labels, positions,
                                       set, and batch information.
        """
        # Load initial graph
        initial_filepath = os.path.join(self.initial_graph_dir, self.initial_files[idx])
        initial_data = torch.load(initial_filepath, weights_only=False)

        # Load final graph
        final_filepath = os.path.join(self.final_graph_dir, self.final_files[idx])
        final_data = torch.load(final_filepath, weights_only=False)

        # Load settings
        settings_filepath = os.path.join(self.settings_dir, self.settings_files[idx])
        settings = torch.load(settings_filepath, weights_only=False)  # Expected shape: [num_settings]

        # Store settings as a separate field in initial_data
        initial_data.set = settings.unsqueeze(0)  # Shape: [1, num_settings]

        # Extract positions (assuming the first 3 features are x, y, z coordinates)
        initial_data.pos = initial_data.x[:, :3]  # Shape: [num_nodes, 3]

        # Compute edge attributes manually if required
        if self.use_edge_attr:
            if hasattr(initial_data, 'edge_index') and initial_data.edge_index is not None:
                row, col = initial_data.edge_index
                pos_diff = initial_data.pos[row] - initial_data.pos[col]  # Shape: [num_edges, 3]
                distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)  # Shape: [num_edges, 1]
                edge_attr = torch.cat([pos_diff, distance], dim=1)  # Shape: [num_edges, 4]

                # Standardize edge attributes
                eps = 1e-10
                edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
                edge_attr_std = edge_attr.std(dim=0, keepdim=True)
                edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)  # Shape: [num_edges, 4]

                initial_data.edge_attr = edge_attr  # Assign the standardized edge attributes
                logging.debug(f"Sample {idx}: Computed and standardized edge_attr with shape {initial_data.edge_attr.shape}")
            else:
                raise ValueError(f"Sample {idx} is missing 'edge_index', cannot compute 'edge_attr'.")
        else:
            initial_data.edge_attr = None  # Explicitly set to None if not used
            logging.debug(f"Sample {idx}: edge_attr not computed (use_edge_attr=False)")

        # Assign target node features based on the task
        if self.task == 'predict_n6d':
            initial_data.y = final_data.x[:, :6]  # Shape: [num_nodes, 6]
        elif self.task == 'predict_n4d':
            initial_data.y = final_data.x[:, :4]  # Shape: [num_nodes, 4]
        elif self.task == 'predict_n2d':
            initial_data.y = final_data.x[:, :2]  # Shape: [num_nodes, 2]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Include 'batch' attribute if not present (useful for batching in PyTorch Geometric)
        if not hasattr(initial_data, 'batch') or initial_data.batch is None:
            initial_data.batch = torch.zeros(initial_data.num_nodes, dtype=torch.long)
            logging.debug(f"Sample {idx}: Initialized 'batch' attribute with zeros.")

        return initial_data
