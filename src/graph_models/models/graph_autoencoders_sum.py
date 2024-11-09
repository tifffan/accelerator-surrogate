I'll create a modified version focusing on sum-based skip connections instead of concatenation. This will change the input dimensions and simplify some parts of the architecture.



```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, MetaLayer, TopKPooling
from torch_geometric.data import Data


class GraphConvolutionalAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, depth, pool_ratios):
        """
        Initializes the GraphConvolutionalAutoEncoder with sum-based skip connections.

        Args:
            in_channels (int): Number of input node features.
            hidden_dim (int): Number of hidden dimensions.
            out_channels (int): Number of output node features.
            depth (int): Depth of the encoder and decoder (number of layers).
            pool_ratios (list of float): Pooling ratios for each encoder layer.
        """
        super(GraphConvolutionalAutoEncoder, self).__init__()
        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.pool_ratios = pool_ratios

        # Encoder
        self.encoder_convs.append(GCNConv(in_channels, hidden_dim))
        self.activations.append(nn.ReLU())
        for i in range(1, depth):
            self.encoder_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.ReLU())

            # Pooling layer
            pool_ratio = pool_ratios[i - 1] if (i - 1) < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))
            else:
                self.pools.append(None)

        # Decoder
        for i in range(depth - 1):
            self.decoder_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.ReLU())
        # Final layer
        self.decoder_convs.append(GCNConv(hidden_dim, out_channels))
        self.activations.append(None)

    def forward(self, x, edge_index, batch):
        x_enc = []
        edge_indices = []
        batches = []
        perms = []

        # Encoder
        for i in range(len(self.encoder_convs)):
            x = self.encoder_convs[i](x, edge_index)
            if self.activations[i] is not None:
                x = self.activations[i](x)
            x_enc.append(x)
            edge_indices.append(edge_index)
            batches.append(batch)

            if i < len(self.pools) and self.pools[i] is not None:
                x, edge_index, _, batch, perm, _ = self.pools[i](x, edge_index, batch=batch)
                perms.append(perm)
            else:
                perms.append(None)

        # Decoder
        for i in range(len(self.decoder_convs)):
            idx = len(self.decoder_convs) - i - 1  # Reverse order

            if perms[idx] is not None:
                # Unpooling
                perm = perms[idx]
                x_unpooled = torch.zeros(x_enc[idx].size(0), x.size(1), device=x.device)
                x_unpooled[perm] = x
                x = x_unpooled
                edge_index = edge_indices[idx]
                batch = batches[idx]
            else:
                # No unpooling needed
                x = x
                edge_index = edge_indices[idx]
                batch = batches[idx]

            # Skip connection using sum
            x = x + x_enc[idx]

            x = self.decoder_convs[i](x, edge_index)
            act_idx = len(self.encoder_convs) + i
            if act_idx < len(self.activations) and self.activations[act_idx] is not None:
                x = self.activations[act_idx](x)
        return x


class GraphAttentionAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, depth, pool_ratios, heads=1):
        """
        Initializes the GraphAttentionAutoEncoder with sum-based skip connections.

        Args:
            in_channels (int): Number of input node features.
            hidden_dim (int): Number of hidden dimensions.
            out_channels (int): Number of output node features.
            depth (int): Depth of the encoder and decoder (number of layers).
            pool_ratios (list of float): Pooling ratios for each encoder layer.
            heads (int, optional): Number of attention heads. Defaults to 1.
        """
        super(GraphAttentionAutoEncoder, self).__init__()
        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.heads = heads
        self.pool_ratios = pool_ratios

        # Ensure hidden_dim is divisible by heads for proper dimension matching
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by number of heads"
        self.head_dim = hidden_dim // heads

        # Encoder
        self.encoder_convs.append(GATConv(in_channels, self.head_dim, heads=self.heads, concat=True))
        self.activations.append(nn.ELU())

        for i in range(1, depth):
            # Pooling layer
            pool_ratio = pool_ratios[i - 1] if (i - 1) < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))
            else:
                self.pools.append(None)

            self.encoder_convs.append(GATConv(hidden_dim, self.head_dim, heads=self.heads, concat=True))
            self.activations.append(nn.ELU())

        # Decoder
        for i in range(depth - 1):
            self.decoder_convs.append(GATConv(hidden_dim, self.head_dim, heads=self.heads, concat=True))
            self.activations.append(nn.ELU())

        # Final layer
        self.decoder_convs.append(GATConv(hidden_dim, out_channels, heads=1, concat=False))
        self.activations.append(None)

    def forward(self, x, edge_index, batch):
        x_enc = []
        edge_indices = []
        batches = []
        perms = []

        # Encoder
        for i in range(len(self.encoder_convs)):
            x = self.encoder_convs[i](x, edge_index)
            if self.activations[i] is not None:
                x = self.activations[i](x)
            x_enc.append(x)
            edge_indices.append(edge_index)
            batches.append(batch)

            if i < len(self.pools) and self.pools[i] is not None:
                x, edge_index, _, batch, perm, _ = self.pools[i](x, edge_index, batch=batch)
                perms.append(perm)
            else:
                perms.append(None)

        # Decoder
        for i in range(len(self.decoder_convs)):
            idx = len(self.decoder_convs) - i - 1

            if perms[idx] is not None:
                # Unpooling
                perm = perms[idx]
                x_unpooled = torch.zeros(x_enc[idx].size(0), x.size(1), device=x.device)
                x_unpooled[perm] = x
                x = x_unpooled
                edge_index = edge_indices[idx]
                batch = batches[idx]
            else:
                x = x
                edge_index = edge_indices[idx]
                batch = batches[idx]

            # Skip connection using sum
            x = x + x_enc[idx]

            x = self.decoder_convs[i](x, edge_index)
            act_idx = len(self.encoder_convs) + i
            if act_idx < len(self.activations) and self.activations[act_idx] is not None:
                x = self.activations[act_idx](x)
        return x


class GraphTransformerAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        out_channels,
        depth,
        pool_ratios,
        num_heads=4,
        dropout=0.0,
        edge_dim=None,
    ):
        """
        Initializes the GraphTransformerAutoEncoder with sum-based skip connections.

        Args:
            in_channels (int): Number of input node features.
            hidden_dim (int): Number of hidden dimensions.
            out_channels (int): Number of output node features.
            depth (int): Depth of the encoder and decoder (number of layers).
            pool_ratios (list of float): Pooling ratios for each encoder layer.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            edge_dim (int, optional): Dimension of edge features. Defaults to None.
        """
        super(GraphTransformerAutoEncoder, self).__init__()
        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.pool_ratios = pool_ratios
        self.num_heads = num_heads

        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        # Encoder
        self.encoder_convs.append(
            TransformerConv(
                in_channels,
                self.head_dim,
                heads=self.num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
                beta=True,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.activations.append(nn.ReLU())

        for i in range(1, depth):
            pool_ratio = pool_ratios[i - 1] if (i - 1) < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))
            else:
                self.pools.append(None)

            self.encoder_convs.append(
                TransformerConv(
                    hidden_dim,
                    self.head_dim,
                    heads=self.num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    beta=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.activations.append(nn.ReLU())

        # Decoder
        for i in range(depth - 1):
            self.decoder_convs.append(
                TransformerConv(
                    hidden_dim,
                    self.head_dim,
                    heads=self.num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    beta=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.activations.append(nn.ReLU())

        # Final layer
        self.decoder_convs.append(
            TransformerConv(
                hidden_dim,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
                beta=True,
            )
        )
        self.bns.append(None)
        self.activations.append(None)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_enc = []
        edge_indices = []
        edge_attrs = []
        batches = []
        perms = []

        # Encoder
        for i in range(len(self.encoder_convs)):
            x = self.encoder_convs[i](x, edge_index, edge_attr)
            if self.bns[i] is not None:
                x = self.bns[i](x)
            if self.activations[i] is not None:
                x = self.activations[i](x)
            x_enc.append(x)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            batches.append(batch)

            if i < len(self.pools) and self.pools[i] is not None:
                x, edge_index, edge_attr, batch, perm, _ = self.pools[i](
                    x, edge_index, edge_attr=edge_attr, batch=batch
                )
                perms.append(perm)
            else:
                perms.append(None)

        # Decoder
        for i in range(len(self.decoder_convs)):
            idx = len(self.decoder_convs) - i - 1
            act_idx = len(self.encoder_convs) + i

            if perms[idx] is not None:
                perm = perms[idx]
                x_unpooled = torch.zeros(x_enc[idx].size(0), x.size(1), device=x.device)
                x_unpooled[perm] = x
                x = x_unpooled
                edge_index = edge_indices[idx]
                edge_attr = edge_attrs[idx]
                batch = batches[idx]
            else:
                x = x
                edge_index = edge_indices[idx]
                edge_attr = edge_attrs[idx]
                batch = batches[idx]

            # Skip connection using sum
            x = x + x_enc[idx]

            x = self.decoder_convs[i](x, edge_index, edge_attr)
            if self.bns[act_idx] is not None:
                x = self.bns[act_idx](x)
            if self.activations[act_idx] is not None:
                x = self.activations[act_idx](x)
        return x


class EdgeModel(nn.Module):
    def __init__(self, edge_in_dim, node_in_dim, edge_out_dim, hidden_dim):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim + 2 * node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_out_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch
```