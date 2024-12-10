# context_dataloaders.py: DataLoaders for the GraphSettingsDataset.

import os
import re
import torch
import logging
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader


class GraphSettingsDataset(Dataset):
    def __init__(self, initial_graph_dir, final_graph_dir, settings_dir, task='predict_n6d',
                 use_edge_attr=False, edge_attr_method="v0", preload_data=False):
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
            edge_attr_method (str, optional): Method for edge attribute computation.
                                              Options:
                                                - 'v0': pos_diff only, standardized
                                                - 'v0n': normalized pos_diff, no standardization
                                                - 'v1': pos_diff + distance, all standardized
                                                - 'v1n': normalized pos_diff + distance, only distance standardized
                                                - 'v2': pos_diff + 1/squared distance, all standardized
                                                - 'v2n': normalized pos_diff + 1/squared distance, only 1/squared distance standardized
                                                - 'v3': pos_diff / squared distance, all standardized
                                              Defaults to 'v0'.
            preload_data (bool, optional): If True, preloads all data into memory.
                                           Defaults to False.
        """
        self.initial_graph_dir = initial_graph_dir
        self.final_graph_dir = final_graph_dir
        self.settings_dir = settings_dir
        self.task = task
        self.use_edge_attr = use_edge_attr
        self.edge_attr_method = edge_attr_method
        self.preload_data = preload_data

        # Load and sort initial, final, and settings files
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

        logging.info(f"Initialized GraphSettingsDataset with {len(self)} samples. "
                     f"Preload data: {self.preload_data}, Use edge_attr: {self.use_edge_attr}, "
                     f"Edge attr method: {self.edge_attr_method}")

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
        if self.preload_data:
            initial_data = self.initial_graphs[idx]
            final_data = self.final_graphs[idx]
            settings = self.settings[idx]
        else:
            initial_data = torch.load(os.path.join(self.initial_graph_dir, self.initial_files[idx]))
            final_data = torch.load(os.path.join(self.final_graph_dir, self.final_files[idx]))
            settings = torch.load(os.path.join(self.settings_dir, self.settings_files[idx]))

        # Store settings as a separate field in initial_data
        initial_data.set = settings.unsqueeze(0)  # Shape: [1, num_settings]

        # Extract positions (assuming the first 3 features are x, y, z coordinates)
        initial_data.pos = initial_data.x[:, :3]  # Shape: [num_nodes, 3]

        # Compute edge attributes manually if required
        if self.use_edge_attr:
            self._compute_edge_attr(initial_data)
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

    def _compute_edge_attr(self, graph):
        """
        Computes and assigns edge attributes based on the selected method.

        Args:
            graph (torch_geometric.data.Data): The graph data object.
        """
        row, col = graph.edge_index
        pos_diff = graph.pos[row] - graph.pos[col]  # Shape: [num_edges, 3]

        if self.edge_attr_method == "v0":
            # Use pos_diff only, with standardization
            edge_attr = pos_diff  # Shape: [num_edges, 3]
        elif self.edge_attr_method == "v0n":
            # Use normalized pos_diff, without standardization
            norm = torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
            edge_attr = pos_diff / norm  # Shape: [num_edges, 3]
        elif self.edge_attr_method == "v1":
            # Use pos_diff + distance, all standardized
            distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)  # Shape: [num_edges, 1]
            edge_attr = torch.cat([pos_diff, distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v1n":
            # Use normalized pos_diff + distance, only distance standardized
            norm = torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
            normalized_pos_diff = pos_diff / norm  # Shape: [num_edges, 3]
            edge_attr = torch.cat([normalized_pos_diff, norm], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v2":
            # Use pos_diff + 1/squared distance, all standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            inverse_squared_distance = 1 / squared_distance  # Shape: [num_edges, 1]
            edge_attr = torch.cat([pos_diff, inverse_squared_distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v2n":
            # Use normalized pos_diff + 1/squared distance, only 1/squared distance standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            normalized_pos_diff = pos_diff / torch.sqrt(squared_distance)  # Shape: [num_edges, 3]
            inverse_squared_distance = 1 / squared_distance  # Shape: [num_edges, 1]
            edge_attr = torch.cat([normalized_pos_diff, inverse_squared_distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v3":
            # Use pos_diff / squared_distance, all standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            edge_attr = pos_diff / squared_distance  # Shape: [num_edges, 3]
        else:
            raise ValueError(f"Invalid edge_attr_method: {self.edge_attr_method}")

        # Standardization logic
        if self.edge_attr_method in ["v0", "v1", "v2", "v3"]:
            # Standardize all features
            eps = 1e-10
            edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr.std(dim=0, keepdim=True)
            edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)  # Shape: [num_edges, ...]
        elif self.edge_attr_method in ["v1n", "v2n"]:
            # Standardize only the magnitude-related feature (last column)
            eps = 1e-10
            magnitude_indices = [-1]  # Last column contains the magnitude feature
            edge_attr_mean = edge_attr[:, magnitude_indices].mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr[:, magnitude_indices].std(dim=0, keepdim=True)
            edge_attr[:, magnitude_indices] = (edge_attr[:, magnitude_indices] - edge_attr_mean) / (edge_attr_std + eps)

        graph.edge_attr = edge_attr
        logging.debug(f"Computed edge_attr with shape {graph.edge_attr.shape} using method {self.edge_attr_method}")



class GraphSettingsDataLoaders:
    def __init__(
        self,
        initial_graph_dir,
        final_graph_dir,
        settings_dir,
        task='predict_n6d',
        use_edge_attr=False,
        edge_attr_method="v0",
        preload_data=False,
        batch_size=32,
        n_train=1000,
        n_val=200,
        n_test=200
    ):
        """
        Initializes the GraphSettingsDataLoaders.

        Args:
            initial_graph_dir (str): Directory containing initial graph .pt files.
            final_graph_dir (str): Directory containing final graph .pt files.
            settings_dir (str): Directory containing settings .pt files.
            task (str, optional): Prediction task. One of ['predict_n6d', 'predict_n4d', 'predict_n2d'].
                                  Defaults to 'predict_n6d'.
            use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
                                            Defaults to False.
            edge_attr_method (str, optional): Method for edge attribute computation.
                                              Options:
                                                - 'v0': pos_diff only, standardized
                                                - 'v0n': normalized pos_diff, no standardization
                                                - 'v1': pos_diff + distance, all standardized
                                                - 'v1n': normalized pos_diff + distance, only distance standardized
                                                - 'v2': pos_diff + 1/squared distance, all standardized
                                                - 'v2n': normalized pos_diff + 1/squared distance, only 1/squared distance standardized
                                                - 'v3': pos_diff / squared distance, all standardized
                                              Defaults to 'v0'.
            preload_data (bool, optional): If True, preloads all data into memory.
                                           Defaults to False.
            batch_size (int, optional): Batch size for the DataLoaders. Defaults to 32.
            n_train (int, optional): Number of training samples. Defaults to 1000.
            n_val (int, optional): Number of validation samples. Defaults to 200.
            n_test (int, optional): Number of testing samples. Defaults to 200.
        """
        # Initialize the dataset with new parameters
        self.dataset = GraphSettingsDataset(
            initial_graph_dir=initial_graph_dir,
            final_graph_dir=final_graph_dir,
            settings_dir=settings_dir,
            task=task,
            use_edge_attr=use_edge_attr,
            edge_attr_method=edge_attr_method,
            preload_data=preload_data
        )

        # Sort the dataset indices based on the integer extracted from filenames
        sorted_indices = sorted(
            range(len(self.dataset)),
            key=lambda idx: int(re.search(r'graph_(\d+)', self.dataset.initial_files[idx]).group(1))
        )

        # Total samples required
        total_samples = len(self.dataset)
        n_total = n_train + n_val + n_test

        if n_total > total_samples:
            raise ValueError(f"n_train + n_val + n_test ({n_total}) exceeds dataset size ({total_samples}).")

        # Select test indices as the last n_test samples
        test_indices = sorted_indices[-n_test:]
        remaining_indices = sorted_indices[:-n_test]

        # Split remaining_indices into train and val
        n_remaining = len(remaining_indices)
        if n_train + n_val > n_remaining:
            raise ValueError(f"n_train + n_val ({n_train + n_val}) exceeds remaining dataset size ({n_remaining}) after excluding test samples.")

        train_indices = remaining_indices[:n_train]
        val_indices = remaining_indices[n_train:n_train + n_val]

        # Create subsets
        self.train_set = Subset(self.dataset, train_indices)
        self.val_set = Subset(self.dataset, val_indices)
        self.test_set = Subset(self.dataset, test_indices)

        self.batch_size = batch_size

        # Initialize DataLoaders as None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._all_data_loader = None

        logging.info(f"Initialized GraphSettingsDataLoaders with {n_train} train, {n_val} val, and {n_test} test samples.")

    def get_train_loader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True  # Shuffle only the training data
            )
        return self._train_loader

    def get_val_loader(self):
        if self._val_loader is None:
            self._val_loader = DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False
            )
        return self._val_loader

    def get_test_loader(self):
        if self._test_loader is None:
            self._test_loader = DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                shuffle=False
            )
        return self._test_loader

    def get_all_data_loader(self):
        """
        Returns a DataLoader for the entire dataset as a single batch, without splitting.
        """
        if self._all_data_loader is None:
            self._all_data_loader = DataLoader(
                self.dataset,
                batch_size=len(self.dataset),
                shuffle=False
            )
        return self._all_data_loader
