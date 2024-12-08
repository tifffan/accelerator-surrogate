# datasets.py

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import logging
import glob
import re
import os

# class GraphDataset(Dataset):
#     def __init__(self, initial_graph_dir, final_graph_dir, settings_dir, task='predict_n6d', use_edge_attr=False):
#         """
#         Initializes the GraphDataset.

#         Args:
#             initial_graph_dir (str): Directory containing initial graph .pt files.
#             final_graph_dir (str): Directory containing final graph .pt files.
#             settings_dir (str): Directory containing settings .pt files.
#             task (str, optional): Prediction task. One of ['predict_n6d', 'predict_n4d', 'predict_n2d'].
#                                   Defaults to 'predict_n6d'.
#             use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
#                                             Defaults to False.
#         """
#         self.initial_graph_dir = initial_graph_dir
#         self.final_graph_dir = final_graph_dir
#         self.settings_dir = settings_dir
#         self.task = task
#         self.use_edge_attr = use_edge_attr

#         # Get list of graph files and ensure they correspond
#         self.initial_files = sorted([f for f in os.listdir(initial_graph_dir) if f.endswith('.pt')])
#         self.final_files = sorted([f for f in os.listdir(final_graph_dir) if f.endswith('.pt')])
#         self.settings_files = sorted([f for f in os.listdir(settings_dir) if f.endswith('.pt')])

#         assert len(self.initial_files) == len(self.final_files) == len(self.settings_files), \
#             "Mismatch in number of initial graphs, final graphs, and settings files."

#         logging.info(f"Initialized GraphDataset with {len(self)} samples. Use edge_attr: {self.use_edge_attr}")

#     def __len__(self):
#         return len(self.initial_files)

#     def __getitem__(self, idx):
#         """
#         Retrieves the graph data at the specified index.

#         Args:
#             idx (int): Index of the data sample.

#         Returns:
#             torch_geometric.data.Data: Graph data object containing node features, edge indices,
#                                        edge attributes (if required), target labels, positions,
#                                        and batch information.
#         """
#         # Load initial graph
#         initial_filepath = os.path.join(self.initial_graph_dir, self.initial_files[idx])
#         # initial_data = torch.load(initial_filepath, weights_only=True)
#         initial_data = torch.load(initial_filepath, weights_only=False)

#         # Load final graph
#         final_filepath = os.path.join(self.final_graph_dir, self.final_files[idx])
#         # final_data = torch.load(final_filepath, weights_only=True)
#         final_data = torch.load(final_filepath, weights_only=False)

#         # Load settings
#         settings_filepath = os.path.join(self.settings_dir, self.settings_files[idx])
#         # settings = torch.load(settings_filepath, weights_only=True)  # Expected shape: [num_settings]
#         settings = torch.load(settings_filepath, weights_only=False)

#         # Concatenate settings to each node's feature in the initial graph
#         num_nodes = initial_data.num_nodes
#         settings_expanded = settings.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, num_settings]
#         initial_data.x = torch.cat([initial_data.x, settings_expanded], dim=1)  # New shape: [num_nodes, original_x_dim + num_settings]

#         # Extract positions (assuming the first 3 features are x, y, z coordinates)
#         initial_data.pos = initial_data.x[:, :3]  # Shape: [num_nodes, 3]

#         # Compute edge attributes manually if required
#         if self.use_edge_attr:
#             if hasattr(initial_data, 'edge_index') and initial_data.edge_index is not None:
#                 row, col = initial_data.edge_index
#                 pos_diff = initial_data.pos[row] - initial_data.pos[col]  # Shape: [num_edges, 3]
#                 distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)  # Shape: [num_edges, 1]
#                 edge_attr = torch.cat([pos_diff, distance], dim=1)  # Shape: [num_edges, 4]

#                 # Standardize edge attributes
#                 eps = 1e-10
#                 edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
#                 edge_attr_std = edge_attr.std(dim=0, keepdim=True)
#                 edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)  # Shape: [num_edges, 4]

#                 initial_data.edge_attr = edge_attr  # Assign the standardized edge attributes
#                 logging.debug(f"Sample {idx}: Computed and standardized edge_attr with shape {initial_data.edge_attr.shape}")
#             else:
#                 raise ValueError(f"Sample {idx} is missing 'edge_index', cannot compute 'edge_attr'.")
#         else:
#             initial_data.edge_attr = None  # Explicitly set to None if not used
#             logging.debug(f"Sample {idx}: edge_attr not computed (use_edge_attr=False)")

#         # Assign target node features based on the task
#         if self.task == 'predict_n6d':
#             initial_data.y = final_data.x[:, :6]  # Shape: [num_nodes, 6]
#         elif self.task == 'predict_n4d':
#             initial_data.y = final_data.x[:, :4]  # Shape: [num_nodes, 4]
#         elif self.task == 'predict_n2d':
#             initial_data.y = final_data.x[:, :2]  # Shape: [num_nodes, 2]
#         else:
#             raise ValueError(f"Unknown task: {self.task}")

#         # Include 'batch' attribute if not present (useful for batching in PyTorch Geometric)
#         if not hasattr(initial_data, 'batch') or initial_data.batch is None:
#             initial_data.batch = torch.zeros(initial_data.num_nodes, dtype=torch.long)
#             logging.debug(f"Sample {idx}: Initialized 'batch' attribute with zeros.")

#         return initial_data

class GraphDataset(Dataset):
    def __init__(self, initial_graph_dir, final_graph_dir, settings_dir, task='predict_n6d', 
                 use_edge_attr=False, edge_attr_method="v0", preload_data=False):
        """
        Initializes the GraphDataset.

        Args:
            initial_graph_dir (str): Directory containing initial graph .pt files.
            final_graph_dir (str): Directory containing final graph .pt files.
            settings_dir (str): Directory containing settings .pt files.
            task (str, optional): Prediction task. Defaults to 'predict_n6d'.
            use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
                                            Defaults to False.
            edge_attr_method (str, optional): Method to compute edge attributes ('v0', 'v0n', 'v1', 'v1n', 'v2', 'v2n', 'v3').
                                              Defaults to 'v0'.
            preload_data (bool, optional): If True, preload all data into memory. Defaults to False.
        """
        self.initial_graph_dir = initial_graph_dir
        self.final_graph_dir = final_graph_dir
        self.settings_dir = settings_dir
        self.task = task
        self.use_edge_attr = use_edge_attr
        self.edge_attr_method = edge_attr_method
        self.preload_data = preload_data

        # Get list of graph files and ensure they correspond
        self.initial_files = sorted(
            [f for f in os.listdir(initial_graph_dir) if f.endswith('.pt')],
            key=lambda f: int(re.search(r'graph_(\d+)', f).group(1))
        )
        self.final_files = sorted(
            [f for f in os.listdir(final_graph_dir) if f.endswith('.pt')],
            key=lambda f: int(re.search(r'graph_(\d+)', f).group(1))
        )
        self.settings_files = sorted(
            [f for f in os.listdir(settings_dir) if f.endswith('.pt')],
            key=lambda f: int(re.search(r'settings_(\d+)', f).group(1))
        )

        assert len(self.initial_files) == len(self.final_files) == len(self.settings_files), \
            "Mismatch in number of initial graphs, final graphs, and settings files."

        # Preload data if enabled
        if self.preload_data:
            self.initial_graphs = [torch.load(os.path.join(initial_graph_dir, f)) for f in self.initial_files]
            self.final_graphs = [torch.load(os.path.join(final_graph_dir, f)) for f in self.final_files]
            self.settings = [torch.load(os.path.join(settings_dir, f)) for f in self.settings_files]
        else:
            self.initial_graphs = None
            self.final_graphs = None
            self.settings = None

        logging.info(f"Initialized GraphDataset with {len(self)} samples. Preload data: {self.preload_data}")

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
                                       and batch information.
        """
        # Load preloaded data if available
        if self.preload_data:
            initial_data = self.initial_graphs[idx]
            final_data = self.final_graphs[idx]
            settings = self.settings[idx]
        else:
            # Load data from disk
            initial_filepath = os.path.join(self.initial_graph_dir, self.initial_files[idx])
            final_filepath = os.path.join(self.final_graph_dir, self.final_files[idx])
            settings_filepath = os.path.join(self.settings_dir, self.settings_files[idx])

            initial_data = torch.load(initial_filepath)
            final_data = torch.load(final_filepath)
            settings = torch.load(settings_filepath)

        # Concatenate settings to each node's feature in the initial graph
        num_nodes = initial_data.num_nodes
        settings_expanded = settings.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, num_settings]
        initial_data.x = torch.cat([initial_data.x, settings_expanded], dim=1)  # New shape: [num_nodes, original_x_dim + num_settings]

        # Extract positions (assuming the first 3 features are x, y, z coordinates)
        initial_data.pos = initial_data.x[:, :3]  # Shape: [num_nodes, 3]

        # Compute edge attributes manually if required
        if self.use_edge_attr:
            self._compute_edge_attr(initial_data)

        # Assign target node features based on the task
        if self.task == 'predict_n6d':
            initial_data.y = final_data.x[:, :6]  # Shape: [num_nodes, 6]
        elif self.task == 'predict_n4d':
            initial_data.y = final_data.x[:, :4]  # Shape: [num_nodes, 4]
        elif self.task == 'predict_n2d':
            initial_data.y = final_data.x[:, :2]  # Shape: [num_nodes, 2]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return initial_data

    def _compute_edge_attr(self, graph):
        row, col = graph.edge_index
        pos_diff = graph.pos[row] - graph.pos[col]  # Shape: [num_edges, 3]

        if self.edge_attr_method == "v0":
            edge_attr = pos_diff  # Use pos_diff only
        elif self.edge_attr_method == "v0n":
            edge_attr = pos_diff / torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
        elif self.edge_attr_method == "v1":
            distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)
            edge_attr = torch.cat([pos_diff, distance], dim=1)
        elif self.edge_attr_method == "v1n":
            norm = torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
            edge_attr = torch.cat([pos_diff / norm, norm], dim=1)
        elif self.edge_attr_method == "v2":
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)
            edge_attr = torch.cat([pos_diff, 1 / squared_distance], dim=1)
        elif self.edge_attr_method == "v2n":
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)
            edge_attr = torch.cat([pos_diff / torch.sqrt(squared_distance), 1 / squared_distance], dim=1)
        elif self.edge_attr_method == "v3":
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)
            edge_attr = pos_diff / squared_distance
        else:
            raise ValueError(f"Invalid edge_attr_method: {self.edge_attr_method}")

        if self.edge_attr_method in ["v0", "v1", "v2", "v3"]:
            eps = 1e-10
            edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr.std(dim=0, keepdim=True)
            edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)
        elif self.edge_attr_method in ["v1n", "v2n"]:
            eps = 1e-10
            magnitude_indices = [-1]  # Standardize only the last column
            edge_attr_mean = edge_attr[:, magnitude_indices].mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr[:, magnitude_indices].std(dim=0, keepdim=True)
            edge_attr[:, magnitude_indices] = (edge_attr[:, magnitude_indices] - edge_attr_mean) / (edge_attr_std + eps)

        graph.edge_attr = edge_attr


class StepPairGraphDataset(Dataset):
    def __init__(self, graph_data_dir, initial_step, final_step, task='predict_n6d', use_settings=False,
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
        self.use_settings = use_settings
        self.identical_settings = identical_settings
        self.settings_file = settings_file
        self.use_edge_attr = use_edge_attr

        # Build file paths for initial and final graphs
        initial_graph_dir = os.path.join(graph_data_dir, f"step_{initial_step}")
        final_graph_dir = os.path.join(graph_data_dir, f"step_{final_step}")

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
        if self.use_settings:
            if self.identical_settings:
                if settings_file is None:
                    raise ValueError("Settings file must be provided when identical_settings is True.")
                self.settings = torch.load(settings_file)
            else:
                # Load settings per sample if necessary
                self.settings_files = [f.replace(f"step_{initial_step}", "settings").replace('graph_', 'settings_') for f in self.initial_graph_files]
                if subsample_size is not None:
                    self.settings_files = self.settings_files[:subsample_size]
                if not all(os.path.isfile(f) for f in self.settings_files):
                    raise ValueError("Some settings files are missing.")

        logging.info(f"Initialized StepPairGraphDataset with {len(self)} samples.")

    def __len__(self):
        return len(self.initial_graph_files)

    def __getitem__(self, idx):
        # Load initial graph
        initial_graph = torch.load(self.initial_graph_files[idx])
        # Load final graph
        final_graph = torch.load(self.final_graph_files[idx])

        # Optionally concatenate settings to node features
        if self.use_settings:
            # Load settings
            if self.identical_settings:
                settings = self.settings
            else:
                settings = torch.load(self.settings_files[idx])
            
            num_nodes = initial_graph.num_nodes
            settings_expanded = settings.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, num_settings]
            initial_graph.x = torch.cat([initial_graph.x, settings_expanded], dim=1)  # New shape: [num_nodes, original_x_dim + num_settings]

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

