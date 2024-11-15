# sequence_trainer_accelerate.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import logging
import os

from trainers import BaseTrainer

class SequenceTrainerAccelerate(BaseTrainer):
    def __init__(self, **kwargs):
        # Pop out values from kwargs to prevent duplication
        criterion = kwargs.pop('criterion', None)
        discount_factor = kwargs.pop('discount_factor', 0.9)
        
        # Initialize the parent class
        super().__init__(**kwargs)
        
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
        self.discount_factor = discount_factor
        
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")
        logging.info(f"Using discount factor: {self.discount_factor}")
        
    def train_step(self, batch):
        initial_graphs, target_graphs_list, seq_lengths = batch
        batch_size = len(initial_graphs)
        loss = 0

        for i in range(batch_size):
            initial_graph = initial_graphs[i]
            target_graphs = target_graphs_list[i]
            seq_length = seq_lengths[i]

            # Initialize prediction with the initial graph
            prediction = initial_graph.x

            # Iterate over prediction horizon
            for k, target_graph in enumerate(target_graphs, start=1):
                # Apply the model to get the next prediction
                data_input = Data(x=prediction, edge_index=initial_graph.edge_index, batch=initial_graph.batch)
                prediction = self.model_forward(data_input)

                # Compute loss between prediction and target
                step_loss = self.criterion(prediction, target_graph.x)

                # Apply discount factor
                discounted_loss = (self.discount_factor ** (seq_length - k)) * step_loss

                # Accumulate loss
                loss += discounted_loss / batch_size  # Normalize by batch size

                # Optional: Update initial_graph if graph structure changes
                # initial_graph = target_graph  # Uncomment if needed

        return loss

    def model_forward(self, data):
        """
        Calls the model's forward method based on the identified model type.
        """
        model_type = self.model_type
        if model_type == 'GNN_TopK':
            x_pred, _ = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                pos=data.pos,
                batch=data.batch
            )
        elif model_type == 'TopkMultiscaleGNN':
            x_pred, _ = self.model(
                x=data.x,
                edge_index=data.edge_index,
                pos=data.pos,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        elif model_type in ['SinglescaleGNN', 'MultiscaleGNN']:
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                pos=data.pos,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        elif model_type in ['MeshGraphNet', 'MeshGraphAutoEncoder']:
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        elif model_type in ['GraphTransformer', 'GraphTransformerAutoEncoder']:
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
                batch=data.batch
            )
        else:  # GraphConvolutionNetwork, GraphAttentionNetwork, etc.
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch
            )
        return x_pred

    # def save_checkpoint(self, epoch):
    #     # Save checkpoint as in BaseTrainer
    #     super().save_checkpoint(epoch)
