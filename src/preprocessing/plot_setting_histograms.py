import matplotlib.pyplot as plt
import numpy as np
from electron_beam_dataloaders import ElectronBeamDataLoaders


data_catalog = "/sdf/home/t/tiffan/repo/accelerator-surrogate/src/points_models/catalogs/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog_test_sdf.csv"

# Initialize the DataLoaders
data_loaders = ElectronBeamDataLoaders(
    data_catalog=data_catalog,
    n_train=5790,
    n_val=0, 
    n_test=0
    
)

# Retrieve the training DataLoader
loader = data_loaders.get_all_data_loader()

# Initialize empty lists to collect settings values
CQ10121_b1_gradient = []
GUNF_rf_field_scale = []
GUNF_theta0_deg = []
SOL10111_solenoid_field_scale = []
SQ10122_b1_gradient = []
distgen_total_charge = []

# Loop over the dataset and collect the settings
for _, _, settings in loader:
    CQ10121_b1_gradient.append(settings[:, 0].numpy())
    GUNF_rf_field_scale.append(settings[:, 1].numpy())
    GUNF_theta0_deg.append(settings[:, 2].numpy())
    SOL10111_solenoid_field_scale.append(settings[:, 3].numpy())
    SQ10122_b1_gradient.append(settings[:, 4].numpy())
    distgen_total_charge.append(settings[:, 5].numpy())

# Concatenate the settings values into a single array
CQ10121_b1_gradient = np.concatenate(CQ10121_b1_gradient)
GUNF_rf_field_scale = np.concatenate(GUNF_rf_field_scale)
GUNF_theta0_deg = np.concatenate(GUNF_theta0_deg)
SOL10111_solenoid_field_scale = np.concatenate(SOL10111_solenoid_field_scale)
SQ10122_b1_gradient = np.concatenate(SQ10122_b1_gradient)
distgen_total_charge = np.concatenate(distgen_total_charge)

# Plot histograms for each setting
settings_names = [
    'CQ10121_b1_gradient',
    'GUNF_rf_field_scale',
    'GUNF_theta0_deg',
    'SOL10111_solenoid_field_scale',
    'SQ10122_b1_gradient',
    'distgen_total_charge',
]

settings_data = [
    CQ10121_b1_gradient,
    GUNF_rf_field_scale,
    GUNF_theta0_deg,
    SOL10111_solenoid_field_scale,
    SQ10122_b1_gradient,
    distgen_total_charge,
]

plt.figure(figsize=(12, 10))
for i, (setting_name, data) in enumerate(zip(settings_names, settings_data)):
    plt.subplot(3, 2, i+1)
    plt.hist(data, bins=50, alpha=0.7)
    plt.title(setting_name)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('settings_histograms_filtered_total_charge_51_test.png')
