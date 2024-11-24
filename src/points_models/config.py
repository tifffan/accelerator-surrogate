# config.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training ElectronBeamPointNet')

    parser.add_argument('--data_catalog', type=str, default='catalogs/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog_train_sdf.csv', help='Path to data catalog')
    parser.add_argument('--statistics_file', type=str, default='catalogs/global_statistics_filtered_total_charge_51_train.txt', help='Path to statistics file')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--n_train', type=int, default=4156, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=0, help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=0, help='Number of test samples')
    parser.add_argument('--random_seed', type=int, default=123, help='Random seed')

    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of residual layers')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')

    # Learning rate scheduler options
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['exp', 'lin', None], help='Learning rate scheduler type')
    # Scheduler-specific arguments
    parser.add_argument('--exp_decay_rate', type=float, default=0.001, help='Decay rate for exponential scheduler')
    parser.add_argument('--exp_start_epoch', type=int, default=0, help='Start epoch for exponential scheduler')
    parser.add_argument('--lin_start_epoch', type=int, default=10, help='Start epoch for linear scheduler')
    parser.add_argument('--lin_end_epoch', type=int, default=100, help='End epoch for linear scheduler')
    parser.add_argument('--lin_final_lr', type=float, default=1e-6, help='Final learning rate for linear scheduler')

    parser.add_argument('--model', type=str, default='PointNet1', help='Model name and version')
    parser.add_argument('--base_results_dir', type=str, default='./results', help='Base directory for results')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training')

    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()
    return args
