# Filename: sequence_graph_dataset.py

import os
import torch
from torch_geometric.data import Dataset, Data

class SequenceGraphDataset(Dataset):
    def __init__(self, graph_data_dir, initial_step=0, final_step=10, max_prediction_horizon=3, transform=None, pre_transform=None):
        self.graph_data_dir = graph_data_dir
        self.initial_step = initial_step
        self.final_step = final_step
        self.max_prediction_horizon = max_prediction_horizon
        self.sequence_length = final_step - initial_step + 1
        super(SequenceGraphDataset, self).__init__(graph_data_dir, transform, pre_transform)
        self.graph_paths = self._load_graph_paths()

    def _load_graph_paths(self):
        # Assuming the graphs are stored in directories named 'step_0', 'step_1', etc.
        graph_dirs = [os.path.join(self.graph_data_dir, f'step_{i}') for i in range(self.initial_step, self.final_step + 1)]
        graph_paths = []

        for dir in graph_dirs:
            files = sorted(os.listdir(dir))
            files = [os.path.join(dir, f) for f in files if f.endswith('.pt')]
            graph_paths.append(files)

        # Transpose to get list of sequences
        graph_paths = list(zip(*graph_paths))
        return graph_paths

    def len(self):
        return len(self.graph_paths)

    def get(self, idx):
        # Returns the initial graph and a list of target graphs up to max_prediction_horizon steps ahead
        sequence = self.graph_paths[idx]
        data_list = [torch.load(path) for path in sequence]

        # Prepare data for training
        data_sequences = []
        for t in range(len(data_list) - self.max_prediction_horizon):
            initial_graph = data_list[t]
            target_graphs = data_list[t+1:t+1+self.max_prediction_horizon]
            seq_length = len(target_graphs)
            data_sequences.append((initial_graph, target_graphs, seq_length))

        return data_sequences  # List of (initial_graph, target_graphs, seq_length)
