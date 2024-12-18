# context_config.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate graph-based models.")

    # Common arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['ggn', 'cgn', 'acgn'],
                        help="Model to train.")
    parser.add_argument('--data_keyword', type=str, required=True,
                        help="Common keyword to infer data directories")
    parser.add_argument('--base_data_dir', type=str, default="/sdf/data/ad/ard/u/tiffan/data/",
                        help="Base directory where the data is stored")
    parser.add_argument('--base_results_dir', type=str, default="/sdf/data/ad/ard/u/tiffan/results/",
                        help="Base directory where the results are stored.")

    parser.add_argument('--dataset', type=str, required=True, help="Dataset identifier")
    parser.add_argument('--preload_data', action='store_true', help="Preload the dataset into memory")
    parser.add_argument('--edge_attr_method', type=str, default='v1', help='Method for generating edge attributes')
    parser.add_argument('--task', type=str, required=True, choices=['predict_n6d', 'predict_n4d', 'predict_n2d'],
                        help="Task to perform")
    parser.add_argument('--n_train', type=int, default=800, help="Number of training samples")
    parser.add_argument('--n_val', type=int, default=100, help="Number of validation samples")
    parser.add_argument('--n_test', type=int, default=100, help="Number of test samples")
    parser.add_argument('--nepochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--save_checkpoint_every', type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument('--wd', type=float, default=0.0, help="Weight decay (L2 regularization) factor")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden layer dimension size")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of layers in the model")
    parser.add_argument('--pool_ratios', type=float, nargs='+', default=[1.0], help="Pooling ratios for TopKPooling layers")

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help="Mode to run")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint to resume training")
    parser.add_argument('--checkpoint_epoch', type=int, default=None, help="Epoch of the checkpoint to load")
    parser.add_argument('--results_folder', type=str, default=None, help="Directory to save results and checkpoints")
    parser.add_argument('--random_seed', type=int, default=63, help="Random seed for reproducibility")
    parser.add_argument('--cpu_only', action='store_true', help="Force the script to use CPU even if GPU is available")
    parser.add_argument('--verbose', action='store_true', help="Display progress bar while training")

    # Learning Rate Scheduler Arguments
    parser.add_argument('--lr_scheduler', type=str, choices=['none', 'exp', 'lin'], default='none',
                        help="Learning rate scheduler type: 'none', 'exp', or 'lin'")
    # Exponential Scheduler Parameters
    parser.add_argument('--exp_decay_rate', type=float, default=0.001, help="Decay rate for exponential scheduler")
    parser.add_argument('--exp_start_epoch', type=int, default=0, help="Start epoch for exponential scheduler")
    # Linear Scheduler Parameters
    parser.add_argument('--lin_start_epoch', type=int, default=100, help="Start epoch for linear scheduler")
    parser.add_argument('--lin_end_epoch', type=int, default=1000, help="End epoch for linear scheduler")
    parser.add_argument('--lin_final_lr', type=float, default=1e-5, help="Final learning rate for linear scheduler")
    
    # AttentionConditionalGraphNetwork-specific arguments (now referred to as 'acgnn')
    parser.add_argument('--num_heads', type=int, default=4,
                        help="Number of attention heads for ACGNN (AttentionConditionalGraphNetwork)")

    return parser.parse_args()
