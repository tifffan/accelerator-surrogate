# sequence_datasets.py

import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import logging
import glob
import re  # Import regular expressions

class SequenceGraphDataset(Dataset):
    def __init__(self, graph_data_dir, initial_step, final_step, task='predict_n6d',
                 identical_settings=False, settings_file=None, use_edge_attr=False, subsample_size=None):
        """
        Args:
            graph_data_dir (str): Base directory containing the graph data organized by sequence steps.
            initial_step (int): Index of the initial sequence step.
            final_step (int): Index of the final sequence step.
            task (str, optional): Prediction task. One of ['predict_n6d', 'predict_n4d', 'predict_n2d'].
            identical_settings (bool, optional): Whether settings are identical across samples.
            settings_file (str, optional): Path to the settings file (used if identical_settings is True).
            use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
            subsample_size (int, optional): Number of samples to use from the dataset. Use all data if not specified.
        """
        self.graph_data_dir = graph_data_dir
        self.initial_step = initial_step
        self.final_step = final_step
        self.task = task
        self.identical_settings = identical_settings
        self.settings_file = settings_file
        self.use_edge_attr = use_edge_attr

        # Build file paths for initial and final graphs
        initial_graph_dir = os.path.join(graph_data_dir, f"sequence_step_{initial_step}")
        final_graph_dir = os.path.join(graph_data_dir, f"sequence_step_{final_step}")

        # Function to extract the graph number from filenames
        def extract_graph_number(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'graph_(\d+)\.pt', filename)
            if match:
                return int(match.group(1))
            else:
                return -1  # Or raise an exception if filenames don't match the pattern

        # Get list of initial and final graph files and sort them by graph number
        self.initial_graph_files = sorted(
            glob.glob(os.path.join(initial_graph_dir, 'graph_*.pt')),
            key=extract_graph_number
        )
        self.final_graph_files = sorted(
            glob.glob(os.path.join(final_graph_dir, 'graph_*.pt')),
            key=extract_graph_number
        )

        # Subsample the dataset if subsample_size is specified
        if subsample_size is not None:
            self.initial_graph_files = self.initial_graph_files[:subsample_size]
            self.final_graph_files = self.final_graph_files[:subsample_size]

        if len(self.initial_graph_files) != len(self.final_graph_files):
            raise ValueError("Mismatch in number of initial and final graph files.")

        # Load settings if identical
        if self.identical_settings:
            if settings_file is None:
                raise ValueError("Settings file must be provided when identical_settings is True.")
            self.settings = torch.load(settings_file)
        else:
            # Load settings per sample if necessary
            self.settings_files = [f.replace(f"sequence_step_{initial_step}", "settings").replace('graph_', 'settings_') for f in self.initial_graph_files]
            if subsample_size is not None:
                self.settings_files = self.settings_files[:subsample_size]
            if not all(os.path.isfile(f) for f in self.settings_files):
                raise ValueError("Some settings files are missing.")

        logging.info(f"Initialized SequenceGraphDataset with {len(self)} samples.")

    def __len__(self):
        return len(self.initial_graph_files)

    def __getitem__(self, idx):
        # Load initial graph
        initial_graph = torch.load(self.initial_graph_files[idx])
        # Load final graph
        final_graph = torch.load(self.final_graph_files[idx])

        # Load settings
        if self.identical_settings:
            settings = self.settings
        else:
            settings = torch.load(self.settings_files[idx])

        # Optionally concatenate settings to node features (commented out)
        # Uncomment and modify if you wish to include settings
        # num_nodes = initial_graph.num_nodes
        # settings_expanded = settings.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, num_settings]
        # initial_graph.x = torch.cat([initial_graph.x, settings_expanded], dim=1)  # New shape: [num_nodes, original_x_dim + num_settings]

        # Extract positions (assuming the first 3 features are x, y, z coordinates)
        initial_graph.pos = initial_graph.x[:, :3]  # Shape: [num_nodes, 3]

        # Compute edge attributes manually if required
        if self.use_edge_attr:
            if hasattr(initial_graph, 'edge_index') and initial_graph.edge_index is not None:
                row, col = initial_graph.edge_index
                pos_diff = initial_graph.pos[row] - initial_graph.pos[col]  # Shape: [num_edges, 3]
                distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)  # Shape: [num_edges, 1]
                edge_attr = torch.cat([pos_diff, distance], dim=1)  # Shape: [num_edges, 4]

                # Standardize edge attributes
                eps = 1e-10
                edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
                edge_attr_std = edge_attr.std(dim=0, keepdim=True)
                edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)  # Shape: [num_edges, 4]

                initial_graph.edge_attr = edge_attr  # Assign the standardized edge attributes
                logging.debug(f"Sample {idx}: Computed and standardized edge_attr with shape {initial_graph.edge_attr.shape}")
            else:
                raise ValueError(f"Sample {idx} is missing 'edge_index', cannot compute 'edge_attr'.")
        else:
            initial_graph.edge_attr = None  # Explicitly set to None if not used
            logging.debug(f"Sample {idx}: edge_attr not computed (use_edge_attr=False)")

        # Assign target node features based on the task
        if self.task == 'predict_n6d':
            initial_graph.y = final_graph.x[:, :6]  # Shape: [num_nodes, 6]
        elif self.task == 'predict_n4d':
            initial_graph.y = final_graph.x[:, :4]  # Shape: [num_nodes, 4]
        elif self.task == 'predict_n2d':
            initial_graph.y = final_graph.x[:, :2]  # Shape: [num_nodes, 2]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Include 'batch' attribute if not present (useful for batching in PyTorch Geometric)
        if not hasattr(initial_graph, 'batch') or initial_graph.batch is None:
            initial_graph.batch = torch.zeros(initial_graph.num_nodes, dtype=torch.long)
            logging.debug(f"Sample {idx}: Initialized 'batch' attribute with zeros.")

        return initial_graph