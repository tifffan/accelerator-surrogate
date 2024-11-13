import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, Subset, random_split

class ElectronBeamDataset(Dataset):
    def __init__(self, data_catalog, statistics_file=None):
        self.data = pd.read_csv(data_catalog)

        if statistics_file is not None:
            # Load global statistics
            self.global_mean, self.global_std = self.load_global_statistics(statistics_file)
            # Build the data transform function
            self.transform = self._build_transform()
        else:
            self.global_mean = None
            self.global_std = None
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filepath = self.data.iloc[idx]['filepath']
        with h5py.File(filepath, 'r') as f:
            initial_state = self._load_initial_state(f)
            final_state = self._load_final_state(f)
            settings = self._load_settings(f)

        if self.transform:
            initial_state, settings, final_state = self.transform(initial_state, settings, final_state)

        return initial_state, final_state, settings

    def _load_initial_state(self, file):
        initial_position_x = torch.tensor(file['initial_position_x'][()]).float()
        initial_position_y = torch.tensor(file['initial_position_y'][()]).float()
        initial_position_z = torch.tensor(file['initial_position_z'][()]).float()
        initial_momentum_px = torch.tensor(file['initial_momentum_px'][()]).float()
        initial_momentum_py = torch.tensor(file['initial_momentum_py'][()]).float()
        initial_momentum_pz = torch.tensor(file['initial_momentum_pz'][()]).float()

        # Combine into a point cloud of shape (num_particles, num_features=6)
        initial_state = torch.stack([
            initial_position_x, initial_position_y, initial_position_z,
            initial_momentum_px, initial_momentum_py, initial_momentum_pz
        ], dim=1)
        return initial_state

    def _load_final_state(self, file):
        pr10241_position_x = torch.tensor(file['pr10241_position_x'][()]).float()
        pr10241_position_y = torch.tensor(file['pr10241_position_y'][()]).float()
        pr10241_position_z = torch.tensor(file['pr10241_position_z'][()]).float()
        pr10241_momentum_px = torch.tensor(file['pr10241_momentum_px'][()]).float()
        pr10241_momentum_py = torch.tensor(file['pr10241_momentum_py'][()]).float()
        pr10241_momentum_pz = torch.tensor(file['pr10241_momentum_pz'][()]).float()

        # Combine into a point cloud of shape (num_particles, num_features=6)
        final_state = torch.stack([
            pr10241_position_x, pr10241_position_y, pr10241_position_z,
            pr10241_momentum_px, pr10241_momentum_py, pr10241_momentum_pz
        ], dim=1)
        return final_state

    def _load_settings(self, file):
        settings_keys = [
            'CQ10121_b1_gradient',
            'GUNF_rf_field_scale',
            'GUNF_theta0_deg',
            'SOL10111_solenoid_field_scale',
            'SQ10122_b1_gradient',
            'distgen_total_charge',
        ]
        settings_values = [file[key][()] if key in file else 0.0 for key in settings_keys]
        settings_tensor = torch.tensor(settings_values, dtype=torch.float32)
        return settings_tensor

    def load_global_statistics(self, statistics_file):
        with open(statistics_file, 'r') as f:
            lines = f.readlines()
            # Extract and parse Global Mean
            mean_line = lines[1].strip()
            mean_values = [float(x) for x in mean_line.split(',')]
            global_mean = torch.tensor(mean_values, dtype=torch.float32)

            # Extract and parse Global Std
            std_line = lines[3].strip()
            std_values = [float(x) for x in std_line.split(',')]
            global_std = torch.tensor(std_values, dtype=torch.float32)

        return global_mean, global_std

    def _build_transform(self):
        # Define a transform function that normalizes data using global statistics
        def transform(initial_state, settings, final_state):
            # Concatenate initial_state and final_state along features
            initial_final_state = torch.cat([initial_state, final_state], dim=1)  # Shape: (num_particles, 12)

            # Normalize initial and final states
            epsilon = 1e-6
            initial_final_state_normalized = (initial_final_state - self.global_mean[:12]) / (self.global_std[:12] + epsilon)

            # Split back into initial_state and final_state
            initial_state_normalized = initial_final_state_normalized[:, :6]
            final_state_normalized = initial_final_state_normalized[:, 6:]

            # Normalize settings
            settings_normalized = (settings - self.global_mean[12:]) / (self.global_std[12:] + epsilon)

            return initial_state_normalized, settings_normalized, final_state_normalized

        return transform

class ElectronBeamDataLoaders:
    def __init__(
        self,
        data_catalog,
        statistics_file=None,
        batch_size=32,
        n_train=1000,
        n_val=200,
        n_test=200,
        random_seed=42
    ):
        self.dataset = ElectronBeamDataset(data_catalog, statistics_file=statistics_file)

        torch.manual_seed(random_seed)

        total_samples = len(self.dataset)
        n_total = n_train + n_val + n_test

        if n_total > total_samples:
            raise ValueError(f"n_train + n_val + n_test ({n_total}) exceeds total dataset size ({total_samples}).")

        if n_total < total_samples:
            print(f"Using a subset of {n_total} samples out of {total_samples}.")
            indices = torch.randperm(total_samples)[:n_total].tolist()
            subset_dataset = Subset(self.dataset, indices)
        else:
            subset_dataset = self.dataset

        self.train_set, self.val_set, self.test_set = random_split(
            subset_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(random_seed)
        )

        self.batch_size = batch_size
        self.random_seed = random_seed

        # Initialize DataLoaders as None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._all_data_loader = None

    def get_train_loader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True
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
            # Load the entire dataset in a single batch
            self._all_data_loader = DataLoader(
                self.dataset,
                batch_size=len(self.dataset),  # Single batch for the entire dataset
                shuffle=False
            )
        return self._all_data_loader


if __name__ == "__main__":
    data_catalog = '/global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv'
    statistics_file = '/global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt'

    # Parameters
    batch_size = 64
    n_train = 800
    n_val = 100
    n_test = 100
    random_seed = 123

    # Initialize the DataLoaders
    data_loaders = ElectronBeamDataLoaders(
        data_catalog=data_catalog,
        statistics_file=statistics_file,
        batch_size=batch_size,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        random_seed=random_seed
    )

    # Retrieve each DataLoader individually
    train_loader = data_loaders.get_train_loader()
    val_loader = data_loaders.get_val_loader()
    test_loader = data_loaders.get_test_loader()

    # Example: Iterate through the training DataLoader
    for batch_idx, (initial_state, final_state, settings) in enumerate(train_loader):
        # Your training code here
        print(f"Batch {batch_idx}:")
        print("Initial State Shape:", initial_state.shape)
        print("Final State Shape:", final_state.shape)
        print("Settings Shape:", settings.shape)
        # Break after one batch for demonstration purposes
        break
