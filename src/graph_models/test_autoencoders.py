# test_autoencoders.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from autoencoders import (
    GraphConvolutionalAutoEncoder,
    GraphAttentionAutoEncoder,
    GraphTransformerAutoEncoder,
    MeshGraphAutoEncoder,
)

def test_graph_convolutional_autoencoder():
    # Create a dummy graph
    num_nodes = 100
    num_node_features = 16
    num_edge_features = 4

    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, batch=batch)

    # Initialize the model
    model = GraphConvolutionalAutoEncoder(
        in_channels=num_node_features,
        hidden_dim=32,
        out_channels=num_node_features,
        depth=3,
        pool_ratios=[0.5, 0.5]
    )

    # Forward pass
    model.train()
    x_reconstructed = model(data.x, data.edge_index, data.batch)

    # Check output shape
    assert x_reconstructed.shape == data.x.shape, "Output shape mismatch"

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(x_reconstructed, data.x)

    # Backward pass
    loss.backward()

    print("GraphConvolutionalAutoEncoder unit test passed.")

def test_graph_attention_autoencoder():
    # Create a dummy graph
    num_nodes = 100
    num_node_features = 16

    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, batch=batch)

    # Initialize the model
    model = GraphAttentionAutoEncoder(
        in_channels=num_node_features,
        hidden_dim=32,
        out_channels=num_node_features,
        depth=3,
        pool_ratios=[0.5, 0.5],
        heads=2
    )

    # Forward pass
    model.train()
    x_reconstructed = model(data.x, data.edge_index, data.batch)

    # Check output shape
    assert x_reconstructed.shape == data.x.shape, "Output shape mismatch"

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(x_reconstructed, data.x)

    # Backward pass
    loss.backward()

    print("GraphAttentionAutoEncoder unit test passed.")

def test_graph_transformer_autoencoder():
    # Create a dummy graph
    num_nodes = 100
    num_node_features = 16
    num_edge_features = 4

    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    edge_attr = torch.randn(edge_index.size(1), num_edge_features)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    # Initialize the model
    model = GraphTransformerAutoEncoder(
        in_channels=num_node_features,
        hidden_dim=32,
        out_channels=num_node_features,
        depth=3,
        pool_ratios=[0.5, 0.5],
        num_heads=2,
        concat=True,
        edge_dim=num_edge_features
    )

    # Forward pass
    model.train()
    x_reconstructed = model(data.x, data.edge_index, data.edge_attr, data.batch)

    # Check output shape
    assert x_reconstructed.shape == data.x.shape, "Output shape mismatch"

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(x_reconstructed, data.x)

    # Backward pass
    loss.backward()

    print("GraphTransformerAutoEncoder unit test passed.")

def test_mesh_graph_autoencoder():
    # Create a dummy graph
    num_nodes = 100
    num_node_features = 16
    num_edge_features = 4

    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    edge_attr = torch.randn(edge_index.size(1), num_edge_features)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    # Initialize the model
    model = MeshGraphAutoEncoder(
        node_in_dim=num_node_features,
        edge_in_dim=num_edge_features,
        node_out_dim=num_node_features,
        hidden_dim=32,
        depth=3,
        pool_ratios=[0.5, 0.5, 0.5]
    )

    # Forward pass
    model.train()
    x_reconstructed = model(data.x, data.edge_index, data.edge_attr, data.batch)

    # Check output shape
    assert x_reconstructed.shape == data.x.shape, "Output shape mismatch"

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(x_reconstructed, data.x)

    # Backward pass
    loss.backward()

    print("MeshGraphAutoEncoder unit test passed.")

def main():
    test_graph_convolutional_autoencoder()
    test_graph_attention_autoencoder()
    test_graph_transformer_autoencoder()
    test_mesh_graph_autoencoder()

if __name__ == '__main__':
    main()
