import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset

class ElectronBeamDataset(Dataset):
    def __init__(self, data_catalog, transform=None):
        self.data = pd.read_csv(data_catalog)
        self.transform = transform

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
        initial_state = torch.stack([initial_position_x, initial_position_y, initial_position_z, 
                                     initial_momentum_px, initial_momentum_py, initial_momentum_pz], dim=1)
        return initial_state

    def _load_final_state(self, file):
        pr10241_position_x = torch.tensor(file['pr10241_position_x'][()]).float()
        pr10241_position_y = torch.tensor(file['pr10241_position_y'][()]).float()
        pr10241_position_z = torch.tensor(file['pr10241_position_z'][()]).float()
        pr10241_momentum_px = torch.tensor(file['pr10241_momentum_px'][()]).float()
        pr10241_momentum_py = torch.tensor(file['pr10241_momentum_py'][()]).float()
        pr10241_momentum_pz = torch.tensor(file['pr10241_momentum_pz'][()]).float()
        
        # Combine into a point cloud of shape (num_particles, num_features=6)
        final_state = torch.stack([pr10241_position_x, pr10241_position_y, pr10241_position_z,
                                   pr10241_momentum_px, pr10241_momentum_py, pr10241_momentum_pz], dim=1)
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
