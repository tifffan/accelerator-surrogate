# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# PointNet Model with Configurable Hidden Dimension and Layers
class PointNet1(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3):
        super(PointNet1, self).__init__()
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

# PointNet Model for Node-Level Regression with Residual Blocks in mlp4 and mlp5
class PointNet2(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3, output_dim=6):
        super(PointNet2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initialize the list of residual blocks for per-point feature extraction
        layers = []
        input_dim = 6  # Initial input feature dimension

        # Create residual layers for initial per-point processing
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

        # Replace mlp4 and mlp5 with residual blocks

        # Define mlp4 residual blocks
        self.mlp4_resblocks = nn.ModuleList([
            ResidualMLP(2 * hidden_dim, hidden_dim),
            ResidualMLP(hidden_dim, hidden_dim // 2),
        ])

        # Define mlp5 residual blocks
        self.mlp5_resblocks = nn.ModuleList([
            ResidualMLP(hidden_dim // 2, hidden_dim // 4),
        ])

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, initial_state, settings):
        # initial_state: (batch_size, num_points, 6)
        # settings: (batch_size, 6)
        x = initial_state  # (B, N, 6)

        # Apply residual MLP blocks to per-point features
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
        x_concat = torch.cat([x, x_global_expanded], dim=2)  # (B, N, 2*hidden_dim)

        # Pass through mlp4 residual blocks
        x = x_concat
        for res_block in self.mlp4_resblocks:
            x = res_block(x)

        # Pass through mlp5 residual blocks
        for res_block in self.mlp5_resblocks:
            x = res_block(x)

        # Final output layer
        x = self.output_layer(x)  # (B, N, output_dim)

        return x

# ==============================================================================================================

# Spatial Transformer Network with parameterized hidden_dim
class STNkd(nn.Module):
    def __init__(self, k=6, hidden_dim=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, hidden_dim // 4, 1)
        self.conv2 = torch.nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 1)
        self.conv3 = torch.nn.Conv1d(hidden_dim // 2, hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim // 2)

        self.k = k
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 4, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)
        x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim, n_pts)
        x = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim, 1)
        x = x.view(-1, self.hidden_dim)  # (batchsize, hidden_dim)

        x = F.relu(self.bn4(self.fc1(x)))  # (batchsize, hidden_dim)
        x = F.relu(self.bn5(self.fc2(x)))  # (batchsize, hidden_dim // 2)
        x = self.fc3(x)  # (batchsize, k*k)

        iden = torch.eye(self.k).flatten().view(1, self.k * self.k).repeat(batchsize, 1).to(x.device)
        x = x + iden  # Add identity to the transformation matrix
        x = x.view(-1, self.k, self.k)  # (batchsize, k, k)
        return x

# # PointNet feature extractor with parameterized hidden_dim
# class PointNetfeat(nn.Module):
#     def __init__(self, global_feat=True, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64):
#         super(PointNetfeat, self).__init__()
#         self.stn = STNkd(k=input_dim, hidden_dim=hidden_dim)  # Spatial transformer network
#         self.conv1 = torch.nn.Conv1d(input_dim, hidden_dim // 4, 1)
#         self.conv2 = torch.nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 1)
#         # Adjust input channels of conv3 to account for settings concatenation
#         self.conv3 = torch.nn.Conv1d((hidden_dim // 2) + settings_dim, hidden_dim, 1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
#         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
#         self.bn3 = nn.BatchNorm1d(hidden_dim)
#         self.global_feat = global_feat
#         self.feature_transform = feature_transform
#         if self.feature_transform:
#             self.fstn = STNkd(k=hidden_dim // 2, hidden_dim=hidden_dim)
            
#         self.hidden_dim = hidden_dim
#         self.settings_dim = settings_dim

#     def forward(self, x, settings):
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
#         trans = self.stn(x)  # (batchsize, k, k)
#         x = x.transpose(2, 1)  # (batchsize, n_pts, k)
#         x = torch.bmm(x, trans)  # Apply the transformation
#         x = x.transpose(2, 1)  # (batchsize, k, n_pts)
#         x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 4, n_pts)

#         if self.feature_transform:
#             trans_feat = self.fstn(x)  # (batchsize, hidden_dim // 2, hidden_dim // 2)
#             x = x.transpose(2, 1)  # (batchsize, n_pts, hidden_dim // 4)
#             x = torch.bmm(x, trans_feat)  # Apply feature transformation
#             x = x.transpose(2, 1)  # (batchsize, hidden_dim // 4, n_pts)
#         else:
#             trans_feat = None

#         pointfeat = x  # (batchsize, hidden_dim // 4, n_pts)
#         x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)

#         # Expand and concatenate settings with per-point features
#         settings_expanded = settings.unsqueeze(2).repeat(1, 1, n_pts)  # (batchsize, settings_dim, n_pts)
#         x = torch.cat([x, settings_expanded], dim=1)  # (batchsize, hidden_dim // 2 + settings_dim, n_pts)

#         x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim, n_pts)
#         x = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim, 1)
#         x = x.view(-1, self.hidden_dim)  # (batchsize, hidden_dim)
#         if self.global_feat:
#             return x, trans, trans_feat
#         else:
#             x = x.view(-1, self.hidden_dim, 1).repeat(1, 1, n_pts)  # (batchsize, hidden_dim, n_pts)
#             return torch.cat([x, pointfeat], 1), trans, trans_feat  # (batchsize, hidden_dim + hidden_dim // 4, n_pts)

# PointNet feature extractor with parameterized hidden_dim and STN_dim
class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64, STN_dim=3):
        """
        PointNet Feature Extractor.

        Args:
            global_feat (bool): Whether to return global features.
            feature_transform (bool): Whether to apply feature transformation.
            input_dim (int): Number of input dimensions per point.
            settings_dim (int): Number of settings dimensions.
            hidden_dim (int): Dimension of hidden layers.
            STN_dim (int): Number of dimensions to apply STN on (3 or 6).
        """
        super(PointNetfeat, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.hidden_dim = hidden_dim
        self.settings_dim = settings_dim
        self.STN_dim = STN_dim

        # Spatial transformer network applied to the first STN_dim dimensions
        self.stn = STNkd(k=STN_dim, hidden_dim=hidden_dim)

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim // 4, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 1)
        # Adjust input channels of conv3 to account for settings concatenation
        self.conv3 = nn.Conv1d((hidden_dim // 2) + settings_dim, hidden_dim, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Feature transformer if enabled
        if self.feature_transform:
            self.fstn = STNkd(k=hidden_dim // 2, hidden_dim=hidden_dim)

    def forward(self, x, settings):
        """
        Forward pass of PointNetfeat.

        Args:
            x (torch.Tensor): Input tensor of shape (batchsize, input_dim, n_pts).
            settings (torch.Tensor): Settings tensor of shape (batchsize, settings_dim).

        Returns:
            tuple: Features, transformation matrix, and feature transformation matrix (if feature_transform=True).
        """
        batchsize = x.size(0)
        n_pts = x.size(2)

        # Apply spatial transformer to the first STN_dim dimensions
        trans = self.stn(x[:, :self.STN_dim, :])  # (batchsize, STN_dim, STN_dim)
        x_trans = x[:, :self.STN_dim, :].transpose(2, 1)  # (batchsize, n_pts, STN_dim)
        x_trans = torch.bmm(x_trans, trans)  # (batchsize, n_pts, STN_dim)
        x_trans = x_trans.transpose(2, 1)  # (batchsize, STN_dim, n_pts)
        x = torch.cat([x_trans, x[:, self.STN_dim:, :]], dim=1)  # (batchsize, input_dim, n_pts)

        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 4, n_pts)

        if self.feature_transform:
            trans_feat = self.fstn(x)  # (batchsize, hidden_dim // 2, hidden_dim // 2)
            x = x.transpose(2, 1)      # (batchsize, n_pts, hidden_dim // 4)
            x = torch.bmm(x, trans_feat)  # Apply feature transformation
            x = x.transpose(2, 1)      # (batchsize, hidden_dim // 4, n_pts)
        else:
            trans_feat = None

        pointfeat = x  # (batchsize, hidden_dim // 4, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)

        # Expand and concatenate settings with per-point features
        settings_expanded = settings.unsqueeze(2).repeat(1, 1, n_pts)  # (batchsize, settings_dim, n_pts)
        x = torch.cat([x, settings_expanded], dim=1)  # (batchsize, hidden_dim // 2 + settings_dim, n_pts)

        x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim, n_pts)
        x = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim, 1)
        x = x.view(-1, self.hidden_dim)      # (batchsize, hidden_dim)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.hidden_dim, 1).repeat(1, 1, n_pts)  # (batchsize, hidden_dim, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # (batchsize, hidden_dim + hidden_dim // 4, n_pts)

# # PointNetRegression with parameterized hidden_dim
# class PointNetRegression(nn.Module):
#     def __init__(self, output_dim=6, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64):
#         super(PointNetRegression, self).__init__()
#         self.output_dim = output_dim
#         self.feature_transform = feature_transform
#         self.input_dim = input_dim
#         self.settings_dim = settings_dim
#         self.hidden_dim = hidden_dim
#         self.feat = PointNetfeat(
#             global_feat=False,
#             feature_transform=feature_transform,
#             input_dim=input_dim,
#             settings_dim=settings_dim,
#             hidden_dim=hidden_dim
#         )
#         # Adjust input channels of conv1 due to concatenation
#         self.conv1 = torch.nn.Conv1d(hidden_dim + hidden_dim // 4, hidden_dim, 1)
#         self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim // 2, 1)
#         self.conv3 = torch.nn.Conv1d(hidden_dim // 2, hidden_dim // 4, 1)
#         self.conv4 = torch.nn.Conv1d(hidden_dim // 4, self.output_dim, 1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
#         self.bn3 = nn.BatchNorm1d(hidden_dim // 4)

#     def forward(self, x, settings):
#         # x: (batchsize, input_dim, n_pts)
        
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
        
#         x = x.permute(0, 2, 1)
#         x, trans, trans_feat = self.feat(x, settings)  # x: (batchsize, hidden_dim + hidden_dim // 4, n_pts)
#         x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim, n_pts)
#         x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)
#         x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim // 4, n_pts)
#         x = self.conv4(x)  # (batchsize, output_dim, n_pts)
#         x = x.transpose(2, 1).contiguous()  # (batchsize, n_pts, output_dim)
#         return x  # Return outputs for regression tasks

# PointNetRegression with parameterized hidden_dim and STN_dim
class PointNetRegression(nn.Module):
    def __init__(self, output_dim=6, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64, STN_dim=3):
        """
        PointNet Regression Model.

        Args:
            output_dim (int): Number of output dimensions per point.
            feature_transform (bool): Whether to apply feature transformation.
            input_dim (int): Number of input dimensions per point.
            settings_dim (int): Number of settings dimensions.
            hidden_dim (int): Dimension of hidden layers.
            STN_dim (int): Number of dimensions to apply STN on (3 or 6).
        """
        super(PointNetRegression, self).__init__()
        self.output_dim = output_dim
        self.feature_transform = feature_transform
        self.input_dim = input_dim
        self.settings_dim = settings_dim
        self.hidden_dim = hidden_dim
        self.STN_dim = STN_dim

        # Initialize the feature extractor with STN_dim
        self.feat = PointNetfeat(
            global_feat=False,
            feature_transform=feature_transform,
            input_dim=input_dim,
            settings_dim=settings_dim,
            hidden_dim=hidden_dim,
            STN_dim=STN_dim
        )

        # Adjust input channels of conv1 due to concatenation
        # The PointNetfeat returns (hidden_dim, n_pts)
        self.conv1 = nn.Conv1d(hidden_dim + hidden_dim // 4, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim // 4, 1)
        self.conv4 = nn.Conv1d(hidden_dim // 4, self.output_dim, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)

    def forward(self, x, settings):
        """
        Forward pass of PointNetRegression.

        Args:
            x (torch.Tensor): Input tensor of shape (batchsize, input_dim, n_pts).
            settings (torch.Tensor): Settings tensor of shape (batchsize, settings_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batchsize, n_pts, output_dim).
        """
        # x: (batchsize, input_dim, n_pts)
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        # Permute to (batchsize, n_pts, input_dim) for PointNetfeat
        x = x.permute(0, 2, 1)  # (batchsize, n_pts, input_dim)
        x, trans, trans_feat = self.feat(x, settings)  # x: (batchsize, hidden_dim, n_pts)

        # Per-point MLP layers
        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)
        x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim // 4, n_pts)
        x = self.conv4(x)                     # (batchsize, output_dim, n_pts)

        # Permute back to (batchsize, n_pts, output_dim)
        x = x.transpose(2, 1).contiguous()    # (batchsize, n_pts, output_dim)
        return x  # Return outputs for regression tasks
    
# Regularization loss remains the same
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :].to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


# ==============================================================================================================

class SimpleEncoder(nn.Module):
    def __init__(self, num_input_channels=6, hidden_dim=64):
        super(SimpleEncoder, self).__init__()
        # Define the convolutional layers with dimensions based on hidden_dim
        self.conv1 = torch.nn.Conv1d(num_input_channels, hidden_dim // 2, 1)
        self.conv2 = torch.nn.Conv1d(hidden_dim // 2, hidden_dim, 1)
        self.conv3 = torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.global_feat = True

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.hidden_dim = hidden_dim

    def forward(self, input_points):
        hidden_dim = self.hidden_dim
        # Apply the convolutional layers with ReLU activation
        x = F.relu(self.bn1(self.conv1(input_points)))  # (batch_size, hidden_dim // 2, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, hidden_dim, n_pts)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch_size, hidden_dim, n_pts)
        # Sum across the length dimension and reshape
        x = torch.sum(x, 2, keepdim=True)  # (batch_size, hidden_dim, 1)
        x = x.view(-1, hidden_dim)  # (batch_size, hidden_dim)
        return x

class PointNet3(nn.Module):
    def __init__(self, num_input_channels=6, num_settings=6, num_output_channels=6, hidden_dim=64):
        super(PointNet3, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_input_channels = num_input_channels
        self.num_settings = num_settings
        self.num_output_channels = num_output_channels

        # Instantiate the SimpleEncoder with parameterized hidden_dim
        self.feat = SimpleEncoder(num_input_channels=self.num_input_channels, hidden_dim=self.hidden_dim)
                
        # Fully connected layers with dimensions based on hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        
        # Convolutional layers with dimensions based on hidden_dim
        self.convs1 = torch.nn.Conv1d(hidden_dim // 2 + self.num_input_channels + self.num_settings, hidden_dim // 4, 1)
        self.convs2 = torch.nn.Conv1d(hidden_dim // 4, hidden_dim // 8, 1)
        self.convs3 = torch.nn.Conv1d(hidden_dim // 8, self.num_output_channels, 1)
        
        # Batch normalization layers
        self.bns1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bns2 = nn.BatchNorm1d(hidden_dim // 8)
        
        self.relu = nn.ReLU()

    def forward(self, input_points, settings):
        # Extract features using the encoder
        input_points = input_points.permute(0, 2, 1)
        x = self.feat(input_points)  # (batch_size, n_pts, hidden_dim)
        
        # Apply fully connected layers with ReLU activation and residual connections
        x = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x = x + x2
        x3 = self.fc3(x)
        global_feature = x + x3  # (batch_size, hidden_dim // 2)
        
        # Expand the global feature and settings to match the input_points size
        global_feature_expanded = global_feature.unsqueeze(2).expand(-1, -1, input_points.size(2))  # (batch_size, hidden_dim // 2, n_pts)
        settings_expanded = settings.unsqueeze(2).expand(-1, -1, input_points.size(2))  # (batch_size, num_settings, n_pts)
        
        # Concatenate input points, global feature, and settings along the channel dimension
        x = torch.cat([input_points, global_feature_expanded, settings_expanded], dim=1)  # (batch_size, hidden_dim // 2 + num_input_channels + num_settings, n_pts)
        
        # Apply the convolutional and batch normalization layers with ReLU activation
        x = F.relu(self.bns1(self.convs1(x)))  # (batch_size, hidden_dim // 4, n_pts)
        x = F.relu(self.bns2(self.convs2(x)))  # (batch_size, hidden_dim // 8, n_pts)
        x = self.convs3(x)  # (batch_size, num_output_channels, n_pts)
        
        x = x.transpose(2, 1).contiguous()  # (batch_size, n_pts, num_output_channels)
        return x  # Return outputs for regression tasks
    
# ==============================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# Variation 1: Settings at Start
# =============================

# Spatial Transformer Network for SettingsAtStart
class STNkd_SettingsAtStart(nn.Module):
    def __init__(self, k=3, hidden_dim=64):
        super(STNkd_SettingsAtStart, self).__init__()
        self.conv1 = nn.Conv1d(k, hidden_dim // 4, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim // 2)

        self.k = k
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batchsize = x.size(0)
        # print("STNkd_SettingsAtStart Shape:", x.shape)
        
        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 4, n_pts)
        
        # print("STNkd_SettingsAtStart Shape:", x.shape)
        
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)
        x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim, n_pts)
        x = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim, 1)
        x = x.view(-1, self.hidden_dim)      # (batchsize, hidden_dim)

        x = F.relu(self.bn4(self.fc1(x)))    # (batchsize, hidden_dim)
        x = F.relu(self.bn5(self.fc2(x)))    # (batchsize, hidden_dim // 2)
        x = self.fc3(x)                       # (batchsize, k*k)

        iden = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(batchsize, 1)
        x = x + iden                          # Add identity to the transformation matrix
        x = x.view(-1, self.k, self.k)       # (batchsize, k, k)
        
        # print("STNkd_SettingsAtStart Shape:", x.shape)
        return x

# PointNet Feature Extractor with Settings Concatenated at Start
class PointNetfeat_SettingsAtStart(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64):
        super(PointNetfeat_SettingsAtStart, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.hidden_dim = hidden_dim
        self.settings_dim = settings_dim

        # Spatial transformer for first three dimensions
        self.stn = STNkd_SettingsAtStart(k=3, hidden_dim=hidden_dim)  # Only first three dimensions

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim + settings_dim, hidden_dim // 4, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Feature transformer if enabled
        if self.feature_transform:
            self.fstn = STNkd_SettingsAtStart(k=hidden_dim // 2, hidden_dim=hidden_dim)

    def forward(self, x, settings):
        """
        x: (batchsize, input_dim, n_pts)
        settings: (batchsize, settings_dim)
        """
        batchsize, input_dim, n_pts = x.size()
        
        # print("Input Shape line 600:", x.shape)

        # Concatenate settings to each node feature
        settings_expanded = settings.unsqueeze(2).repeat(1, 1, n_pts)  # (batchsize, settings_dim, n_pts)
        x = torch.cat([x, settings_expanded], dim=1)  # (batchsize, input_dim + settings_dim, n_pts)
        
        # print("Input Shape:", x.shape)

        # Split spatial and other features
        spatial_features = x[:, :3, :]  # (batchsize, 3, n_pts)
        other_features = x[:, 3:, :]    # (batchsize, input_dim + settings_dim - 3, n_pts)

        # Apply spatial transformer on spatial features
        trans = self.stn(spatial_features)  # (batchsize, 3, 3)
        spatial_features = spatial_features.transpose(2, 1)  # (batchsize, n_pts, 3)
        spatial_features = torch.bmm(spatial_features, trans)  # (batchsize, n_pts, 3)
        spatial_features = spatial_features.transpose(2, 1)  # (batchsize, 3, n_pts)

        # print("Spatial Features Shape:", spatial_features.shape)
        
        # print("Other Features Shape:", other_features.shape)
        
        # Concatenate transformed spatial features with other features
        x = torch.cat([spatial_features, other_features], dim=1)  # (batchsize, input_dim + settings_dim, n_pts)
        
        # print("Concatenated Features Shape before first convolution:", x.shape)

        # First convolution
        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 4, n_pts)
        
        # print("Concatenated Features Shape after first convolution:", x.shape)

        if self.feature_transform:
            trans_feat = self.fstn(x)  # (batchsize, hidden_dim // 2, hidden_dim // 2)
            x = x.transpose(2, 1)      # (batchsize, n_pts, hidden_dim // 4)
            x = torch.bmm(x, trans_feat)  # Apply feature transformation
            x = x.transpose(2, 1)      # (batchsize, hidden_dim // 4, n_pts)
        else:
            trans_feat = None

        # print("x Shape before assign pointfeat:", x.shape)
        
        pointfeat = x  # (batchsize, hidden_dim // 4, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)

        # Third convolution
        x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim, n_pts)

        x = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim, 1)
        x = x.view(-1, self.hidden_dim)      # (batchsize, hidden_dim)

        # print("x shape before return:", x.shape)
        
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.hidden_dim, 1).repeat(1, 1, n_pts)  # (batchsize, hidden_dim, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # (batchsize, hidden_dim + hidden_dim // 4, n_pts)

# PointNet Regression Model with Settings Concatenated at Start
class PointNetRegression_SettingsAtStart(nn.Module):
    def __init__(self, output_dim=6, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64):
        super(PointNetRegression_SettingsAtStart, self).__init__()
        self.output_dim = output_dim
        self.feature_transform = feature_transform
        self.input_dim = input_dim
        self.settings_dim = settings_dim
        self.hidden_dim = hidden_dim

        self.feat = PointNetfeat_SettingsAtStart(
            global_feat=False,
            feature_transform=feature_transform,
            input_dim=input_dim,
            settings_dim=settings_dim,
            hidden_dim=hidden_dim
        )

        # Additional convolutional layers for regression
        self.conv1 = nn.Conv1d(hidden_dim + hidden_dim // 4, hidden_dim // 2, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim // 4, 1)
        self.conv3 = nn.Conv1d(hidden_dim // 4, self.output_dim, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)

    def forward(self, x, settings):
        """
        x: (batchsize, input_dim, n_pts)
        settings: (batchsize, settings_dim)
        """
        # x: (batchsize, n_pts, input_dim)
        
        x = x.permute(0, 2, 1)
        
        batchsize, input_dim, n_pts = x.size()

        
        x, trans, trans_feat = self.feat(x, settings)  # x: (batchsize, hidden_dim, n_pts)
        # x = x.permute(0, 2, 1)  # (batchsize, n_pts, input_dim)
        
        # print("Shape before conv1:", x.shape)

        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 2, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 4, n_pts)
        x = self.conv3(x)                     # (batchsize, output_dim, n_pts)
        
        # print("Shape before output:", x.shape)

        x = x.transpose(2, 1).contiguous()    # (batchsize, n_pts, output_dim)
        
        # print("Output Shape:", x.shape)
        return x  # Return outputs for regression tasks

# Example Usage for SettingsAtStart
if __name__ == "__main__":
    print("===== Testing PointNetRegression_SettingsAtStart =====")
    batch_size = 32
    n_pts = 1024
    input_dim = 6
    settings_dim = 6
    output_dim = 6

    # Sample input data
    x = torch.randn(batch_size, n_pts, input_dim)        # (batchsize, n_pts, input_dim)    
    # x = torch.randn(batch_size, input_dim, n_pts)        # (batchsize, input_dim, n_pts)
    settings = torch.randn(batch_size, settings_dim)      # (batchsize, settings_dim)

    # Instantiate the model
    model_start = PointNetRegression_SettingsAtStart(
        output_dim=output_dim,
        feature_transform=False,
        input_dim=input_dim,
        settings_dim=settings_dim,
        hidden_dim=64
    )

    # Forward pass
    output_start = model_start(x, settings)  # (batchsize, n_pts, output_dim)
    print("Output Shape (Settings at Start):", output_start.shape)

# =============================
# Variation 2: Settings at Global
# =============================

# Spatial Transformer Network for SettingsAtGlobal
class STNkd_SettingsAtGlobal(nn.Module):
    def __init__(self, k=3, hidden_dim=64):
        super(STNkd_SettingsAtGlobal, self).__init__()
        self.conv1 = nn.Conv1d(k, hidden_dim // 4, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim // 2)

        self.k = k
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batchsize = x.size(0)
        
        # print("STNkd_SettingsAtGlobal Shape line 773:", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 4, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)
        x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim, n_pts)
        x = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim, 1)
        x = x.view(-1, self.hidden_dim)      # (batchsize, hidden_dim)

        x = F.relu(self.bn4(self.fc1(x)))    # (batchsize, hidden_dim)
        x = F.relu(self.bn5(self.fc2(x)))    # (batchsize, hidden_dim // 2)
        x = self.fc3(x)                       # (batchsize, k*k)

        iden = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(batchsize, 1)
        x = x + iden                          # Add identity to the transformation matrix
        x = x.view(-1, self.k, self.k)       # (batchsize, k, k)
        return x

# PointNet Feature Extractor with Settings Concatenated at Global Features
class PointNetfeat_SettingsAtGlobal(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64):
        super(PointNetfeat_SettingsAtGlobal, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.hidden_dim = hidden_dim
        self.settings_dim = settings_dim

        # Spatial transformer for first three dimensions
        self.stn = STNkd_SettingsAtGlobal(k=3, hidden_dim=hidden_dim)  # Only first three dimensions

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim // 4, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim + settings_dim, hidden_dim, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Feature transformer if enabled
        if self.feature_transform:
            self.fstn = STNkd_SettingsAtGlobal(k=hidden_dim // 2, hidden_dim=hidden_dim)

    def forward(self, x, settings):
        """
        x: (batchsize, input_dim, n_pts)
        settings: (batchsize, settings_dim)
        """
        batchsize, input_dim, n_pts = x.size()

        # Split spatial and other features
        spatial_features = x[:, :3, :]  # (batchsize, 3, n_pts)
        other_features = x[:, 3:, :]    # (batchsize, input_dim - 3, n_pts)

        # Apply spatial transformer on spatial features
        trans = self.stn(spatial_features)  # (batchsize, 3, 3)
        spatial_features = spatial_features.transpose(2, 1)  # (batchsize, n_pts, 3)
        spatial_features = torch.bmm(spatial_features, trans)  # (batchsize, n_pts, 3)
        spatial_features = spatial_features.transpose(2, 1)  # (batchsize, 3, n_pts)

        # print("Spatial Features Shape:", spatial_features.shape)
        # print("Other Features Shape:", other_features.shape)
        
        # Concatenate transformed spatial features with other features
        x = torch.cat([spatial_features, other_features], dim=1)  # (batchsize, input_dim, n_pts)

        # First convolution
        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 4, n_pts)
        
        # print("x Shape before assign pointfeat:", x.shape)

        if self.feature_transform:
            trans_feat = self.fstn(x)  # (batchsize, hidden_dim // 2, hidden_dim // 2)
            x = x.transpose(2, 1)      # (batchsize, n_pts, hidden_dim // 4)
            x = torch.bmm(x, trans_feat)  # Apply feature transformation
            x = x.transpose(2, 1)      # (batchsize, hidden_dim // 4, n_pts)
        else:
            trans_feat = None

        pointfeat = x  # (batchsize, hidden_dim // 4, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)
        
        # print("x Shape before global features and settings:", x.shape)

        # Global feature
        global_feature = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim // 2, 1)
        settings = settings.unsqueeze(2)  # (batchsize, settings_dim, 1)
        global_feature = torch.cat([global_feature, settings], dim=1)  # (batchsize, hidden_dim // 2 + settings_dim, 1)
        global_feature = global_feature.repeat(1, 1, n_pts)  # (batchsize, hidden_dim // 2 + settings_dim, n_pts)
        
        # print("Global Features Shape:", global_feature.shape)

        # Concatenate with point features
        x = torch.cat([x, global_feature], dim=1)  # (batchsize, hidden_dim // 2 + settings_dim, n_pts)

        # Third convolution
        x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim, n_pts)

        x = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim, 1)
        x = x.view(-1, self.hidden_dim)      # (batchsize, hidden_dim)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.hidden_dim, 1).repeat(1, 1, n_pts)  # (batchsize, hidden_dim, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # (batchsize, hidden_dim + hidden_dim // 4, n_pts)

# PointNet Regression Model with Settings Concatenated at Global Features
class PointNetRegression_SettingsAtGlobal(nn.Module):
    def __init__(self, output_dim=6, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64):
        super(PointNetRegression_SettingsAtGlobal, self).__init__()
        self.output_dim = output_dim
        self.feature_transform = feature_transform
        self.input_dim = input_dim
        self.settings_dim = settings_dim
        self.hidden_dim = hidden_dim

        self.feat = PointNetfeat_SettingsAtGlobal(
            global_feat=False,
            feature_transform=feature_transform,
            input_dim=input_dim,
            settings_dim=settings_dim,
            hidden_dim=hidden_dim
        )

        # Additional convolutional layers for regression
        self.conv1 = nn.Conv1d(hidden_dim + hidden_dim // 4, hidden_dim // 2, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim // 4, 1)
        self.conv3 = nn.Conv1d(hidden_dim // 4, self.output_dim, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)

    def forward(self, x, settings):
        """
        x: (batchsize, input_dim, n_pts)
        settings: (batchsize, settings_dim)
        """
        
        # x: (batchsize, n_pts, input_dim)
        
        x = x.permute(0, 2, 1)
        batchsize, input_dim, n_pts = x.size()

        # x = x.permute(0, 2, 1)  # (batchsize, n_pts, input_dim)
        x, trans, trans_feat = self.feat(x, settings)  # x: (batchsize, hidden_dim, n_pts)

        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 2, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 4, n_pts)
        x = self.conv3(x)                     # (batchsize, output_dim, n_pts)

        x = x.transpose(2, 1).contiguous()    # (batchsize, n_pts, output_dim)
        return x  # Return outputs for regression tasks

# Example Usage for SettingsAtGlobal
if __name__ == "__main__":
    print("===== Testing PointNetRegression_SettingsAtGlobal =====")
    batch_size = 32
    n_pts = 1024
    input_dim = 6
    settings_dim = 6
    output_dim = 6

    # Sample input data
    x = torch.randn(batch_size, n_pts, input_dim)        # (batchsize, n_pts, input_dim)    
    # x = torch.randn(batch_size, input_dim, n_pts)        # (batchsize, input_dim, n_pts)
    settings = torch.randn(batch_size, settings_dim)      # (batchsize, settings_dim)

    # Instantiate the model
    model_global = PointNetRegression_SettingsAtGlobal(
        output_dim=output_dim,
        feature_transform=False,
        input_dim=input_dim,
        settings_dim=settings_dim,
        hidden_dim=64
    )

    # Forward pass
    output_global = model_global(x, settings)  # (batchsize, n_pts, output_dim)
    print("Output Shape (Settings at Global):", output_global.shape)

# =============================
# Variation 3: Settings at Current
# =============================

# Spatial Transformer Network for SettingsAtMiddle
class STNkd_SettingsAtMiddle(nn.Module):
    def __init__(self, k=3, hidden_dim=64):
        super(STNkd_SettingsAtMiddle, self).__init__()
        self.conv1 = nn.Conv1d(k, hidden_dim // 4, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)        
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim // 2)

        self.k = k
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batchsize = x.size(0) # (batchsize, k, n_pts)
        # print("x Shape before conv1:", x.shape)
        
        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 4, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)
        x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim, n_pts)
        
        # print("x Shape before max:", x.shape)
        x = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim, 1)
        x = x.view(-1, self.hidden_dim)      # (batchsize, hidden_dim)
        
        # print("x Shape before fc1:", x.shape)
        x = F.relu(self.bn4(self.fc1(x)))    # (batchsize, hidden_dim)
        x = F.relu(self.bn5(self.fc2(x)))    # (batchsize, hidden_dim // 2)
        x = self.fc3(x)                       # (batchsize, k*k)

        iden = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(batchsize, 1)
        x = x + iden                          # Add identity to the transformation matrix
        x = x.view(-1, self.k, self.k)       # (batchsize, k, k)
        return x

# PointNet Feature Extractor with Settings Concatenated at Current Position
class PointNetfeat_SettingsAtMiddle(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64):
        super(PointNetfeat_SettingsAtMiddle, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.hidden_dim = hidden_dim
        self.settings_dim = settings_dim

        # Spatial transformer for first three dimensions
        self.stn = STNkd_SettingsAtMiddle(k=3, hidden_dim=hidden_dim)  # Only first three dimensions

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim // 4, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim // 2 + settings_dim, hidden_dim, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Feature transformer if enabled
        if self.feature_transform:
            self.fstn = STNkd_SettingsAtMiddle(k=hidden_dim // 2, hidden_dim=hidden_dim)

    def forward(self, x, settings):
        """
        x: (batchsize, input_dim, n_pts)
        settings: (batchsize, settings_dim)
        """
        batchsize, input_dim, n_pts = x.size()

        # Split spatial and other features
        spatial_features = x[:, :3, :]  # (batchsize, 3, n_pts)
        other_features = x[:, 3:, :]    # (batchsize, input_dim - 3, n_pts)

        # Apply spatial transformer on spatial features
        trans = self.stn(spatial_features)  # (batchsize, 3, 3)
        spatial_features = spatial_features.transpose(2, 1)  # (batchsize, n_pts, 3)
        spatial_features = torch.bmm(spatial_features, trans)  # (batchsize, n_pts, 3)
        spatial_features = spatial_features.transpose(2, 1)  # (batchsize, 3, n_pts)

        # Concatenate transformed spatial features with other features
        x = torch.cat([spatial_features, other_features], dim=1)  # (batchsize, input_dim, n_pts)

        # First convolution
        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 4, n_pts)

        if self.feature_transform:
            trans_feat = self.fstn(x)  # (batchsize, hidden_dim // 2, hidden_dim // 2)
            x = x.transpose(2, 1)      # (batchsize, n_pts, hidden_dim // 4)
            x = torch.bmm(x, trans_feat)  # Apply feature transformation
            x = x.transpose(2, 1)      # (batchsize, hidden_dim // 4, n_pts)
        else:
            trans_feat = None

        pointfeat = x  # (batchsize, hidden_dim // 4, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 2, n_pts)

        # Concatenate settings with per-point features
        settings_expanded = settings.unsqueeze(2).repeat(1, 1, n_pts)  # (batchsize, settings_dim, n_pts)
        x = torch.cat([x, settings_expanded], dim=1)  # (batchsize, hidden_dim // 2 + settings_dim, n_pts)

        # Third convolution
        x = F.relu(self.bn3(self.conv3(x)))  # (batchsize, hidden_dim, n_pts)

        x = torch.max(x, 2, keepdim=True)[0]  # (batchsize, hidden_dim, 1)
        x = x.view(-1, self.hidden_dim)      # (batchsize, hidden_dim)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.hidden_dim, 1).repeat(1, 1, n_pts)  # (batchsize, hidden_dim, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # (batchsize, hidden_dim + hidden_dim // 4, n_pts)

# PointNet Regression Model with Settings Concatenated at Current Position
class PointNetRegression_SettingsAtMiddle(nn.Module):
    def __init__(self, output_dim=6, feature_transform=False, input_dim=6, settings_dim=6, hidden_dim=64):
        super(PointNetRegression_SettingsAtMiddle, self).__init__()
        self.output_dim = output_dim
        self.feature_transform = feature_transform
        self.input_dim = input_dim
        self.settings_dim = settings_dim
        self.hidden_dim = hidden_dim

        self.feat = PointNetfeat_SettingsAtMiddle(
            global_feat=False,
            feature_transform=feature_transform,
            input_dim=input_dim,
            settings_dim=settings_dim,
            hidden_dim=hidden_dim
        )

        # Additional convolutional layers for regression
        self.conv1 = nn.Conv1d(hidden_dim + hidden_dim // 4, hidden_dim // 2, 1)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim // 4, 1)
        self.conv3 = nn.Conv1d(hidden_dim // 4, self.output_dim, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)

    def forward(self, x, settings):
        """
        x: (batchsize, input_dim, n_pts)
        settings: (batchsize, settings_dim)
        """
        # x: (batchsize, n_pts, input_dim)
        
        x = x.permute(0, 2, 1)
        batchsize, input_dim, n_pts = x.size()
        
        # print("x Shape before feat:", x.shape)

        # x = x.permute(0, 2, 1)  # (batchsize, n_pts, input_dim)
        x, trans, trans_feat = self.feat(x, settings)  # x: (batchsize, hidden_dim, n_pts)

        x = F.relu(self.bn1(self.conv1(x)))  # (batchsize, hidden_dim // 2, n_pts)
        x = F.relu(self.bn2(self.conv2(x)))  # (batchsize, hidden_dim // 4, n_pts)
        x = self.conv3(x)                     # (batchsize, output_dim, n_pts)

        x = x.transpose(2, 1).contiguous()    # (batchsize, n_pts, output_dim)
        return x  # Return outputs for regression tasks

# Example Usage for SettingsAtMiddle
if __name__ == "__main__":
    print("===== Testing PointNetRegression_SettingsAtMiddle =====")
    batch_size = 32
    n_pts = 1024
    input_dim = 6
    settings_dim = 6
    output_dim = 6

    # Sample input data
    x = torch.randn(batch_size, n_pts, input_dim)        # (batchsize, n_pts, input_dim)    
    # x = torch.randn(batch_size, input_dim, n_pts)        # (batchsize, input_dim, n_pts)
    settings = torch.randn(batch_size, settings_dim)      # (batchsize, settings_dim)

    # Instantiate the model
    model_current = PointNetRegression_SettingsAtMiddle(
        output_dim=output_dim,
        feature_transform=False,
        input_dim=input_dim,
        settings_dim=settings_dim,
        hidden_dim=64
    )

    # Forward pass
    output_current = model_current(x, settings)  # (batchsize, n_pts, output_dim)
    print("Output Shape (Settings at Current):", output_current.shape)
