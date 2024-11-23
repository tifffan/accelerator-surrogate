import torch
import torch.nn as nn
import torch.optim as optim

# Residual MLP Block
class ResidualMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.relu2 = nn.ReLU()
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out += identity
        out = self.relu2(out)
        return out

# ElectronBeamPointNet Model with Configurable Hidden Dimension and Layers
class ElectronBeamPointNet(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3):
        super(ElectronBeamPointNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initialize the list of residual blocks
        layers = []
        input_dim = 6  # Initial input feature dimension

        # Create residual layers
        for i in range(num_layers):
            if i == 0:
                layers.append(ResidualMLP(input_dim, hidden_dim))
            else:
                layers.append(ResidualMLP(hidden_dim, hidden_dim))
        self.res_blocks = nn.ModuleList(layers)

        # Global feature MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # Per-point MLPs for final prediction
        self.mlp4 = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.mlp5 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 6)  # Output per-point features of size 6
        )

    def forward(self, initial_state, settings):
        # initial_state: (batch_size, num_points, 6)
        # settings: (batch_size, 6)
        x = initial_state  # (B, N, 6)

        # Apply residual MLP blocks
        for res_block in self.res_blocks:
            x = res_block(x)  # (B, N, hidden_dim)

        # Obtain global features via max pooling
        x_global = torch.max(x, dim=1)[0]  # (B, hidden_dim)

        # Concatenate settings to global features
        x_global = torch.cat([x_global, settings], dim=1)  # (B, hidden_dim + 6)

        # Process global features
        x_global = self.global_mlp(x_global)  # (B, hidden_dim)

        # Expand and concatenate global features with per-point features
        x_global_expanded = x_global.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, N, hidden_dim)
        x_concat = torch.cat([x, x_global_expanded], dim=2)  # (B, N, hidden_dim + hidden_dim)

        # Further per-point processing
        x = self.mlp4(x_concat)  # (B, N, hidden_dim // 2)
        x = self.mlp5(x)  # (B, N, 6)

        return x

# Main code to utilize the model with your DataLoader
if __name__ == "__main__":
    data_catalog = '/global/homes/t/tiffan/slac-point/data/catalogs/electrons_vary_distributions_vary_settings_catalog.csv'
    statistics_file = '/global/homes/t/tiffan/slac-point/data/catalogs/global_statistics.txt'

    # Parameters
    batch_size = 64
    n_train = 800
    n_val = 100
    n_test = 100
    random_seed = 123
    num_epochs = 10  # Set the number of training epochs
    hidden_dim = 128  # Hidden dimension size
    num_layers = 3    # Number of residual layers

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

    # Retrieve DataLoaders
    train_loader = data_loaders.get_train_loader()
    val_loader = data_loaders.get_val_loader()
    test_loader = data_loaders.get_test_loader()

    # Initialize the model, loss function, and optimizer
    model = ElectronBeamPointNet(hidden_dim=hidden_dim, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (initial_state, final_state, settings) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(initial_state, settings)

            # Compute loss
            loss = criterion(outputs, final_state)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Print average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Evaluation on the validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for initial_state, final_state, settings in val_loader:
            outputs = model(initial_state, settings)
            loss = criterion(outputs, final_state)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Example: Iterate through the test DataLoader
    test_loss = 0.0
    with torch.no_grad():
        for initial_state, final_state, settings in test_loader:
            outputs = model(initial_state, settings)
            loss = criterion(outputs, final_state)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
