import os
import shutil
import torch
import torch_geometric
from pathlib import Path
from sequence_train_accelerate import main
from sequence_config import parse_args

def test_sequence_train_accelerate():
    # Temporary directory setup for the test
    temp_data_dir = Path("./test_graph_data")
    temp_results_dir = Path("./test_results")
    temp_data_dir.mkdir(exist_ok=True)
    temp_results_dir.mkdir(exist_ok=True)

    try:
        # Generate mock graph data for testing
        generate_mock_graph_data(temp_data_dir, num_steps=3, num_graphs=10)

        # Set arguments for testing
        args = parse_args([
            "--model", "gcn",
            "--dataset", "test_dataset",
            "--data_keyword", "test_keyword",
            "--task", "predict_n6d",
            "--initial_step", "0",
            "--final_step", "2",
            "--max_prediction_horizon", "2",
            "--base_data_dir", str(temp_data_dir),
            "--results_folder", str(temp_results_dir),
            "--batch_size", "2",
            "--nepochs", "2",
            "--lr", "0.001",
            "--hidden_dim", "16",
            "--num_layers", "3",
            "--discount_factor", "0.9",
            "--random_seed", "42",
            "--verbose"
        ])

        # Run the main function
        main(args)

        # Check if results are saved
        assert temp_results_dir.exists(), "Results directory was not created."
        checkpoint_dir = temp_results_dir / "checkpoints"
        assert checkpoint_dir.exists(), "Checkpoints directory was not created."
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "No checkpoints were saved."

        print("Test passed: sequence_train_accelerate.py ran successfully.")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        # Cleanup temporary files
        if temp_data_dir.exists():
            shutil.rmtree(temp_data_dir)
        if temp_results_dir.exists():
            shutil.rmtree(temp_results_dir)

def generate_mock_graph_data(base_dir, num_steps=3, num_graphs=10):
    """
    Generate mock graph data for testing.
    Args:
        base_dir (Path): Base directory to store the graph data.
        num_steps (int): Number of sequence steps.
        num_graphs (int): Number of graphs per step.
    """
    for step in range(num_steps):
        step_dir = base_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_graphs):
            # Mock graph data with 5 nodes and 3 features
            x = torch.randn(5, 3)
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
            graph = torch_geometric.data.Data(x=x, edge_index=edge_index)
            torch.save(graph, step_dir / f"graph_{i}.pt")

if __name__ == "__main__":
    test_sequence_train_accelerate()
