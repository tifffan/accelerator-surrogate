# test_training.py

import torch
from datasets import GraphDataset
from models import GraphConvolutionNetwork
from trainers import GraphPredictionTrainer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from utils import set_random_seed

def main():
    # Set random seed
    set_random_seed(42)

    # Create a dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            num_nodes = 50
            x = torch.randn(num_nodes, 10)
            edge_index = torch.randint(0, num_nodes, (2, 100))
            y = torch.randn(num_nodes, 5)
            data = Data(x=x, edge_index=edge_index, y=y)
            data.batch = torch.zeros(num_nodes, dtype=torch.long)
            return data

        def __len__(self):
            return 10  # Small dataset

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=2)

    # Initialize model
    model = GraphConvolutionNetwork(
        in_channels=10,
        hidden_dim=16,
        out_channels=5,
        num_layers=3,
        pool_ratios=[1.0]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Trainer
    trainer = GraphPredictionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        nepochs=2,  # Short training
        save_checkpoint_every=1,
        results_folder='./test_results',
        device='cpu',
        verbose=True
    )

    trainer.train()

if __name__ == '__main__':
    main()
