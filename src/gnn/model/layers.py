# gnn_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GATConv,
    SAGEConv,
    MessagePassing,
)
from torch_geometric.utils import softmax
from omegaconf import OmegaConf
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_activation_function(name: str):
    """Return a callable activation function by name."""
    activation_dict = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "hardswish": nn.Hardswish(),
        "hardtanh": nn.Hardtanh(),
    }
    if name not in activation_dict:
        logger.warning(f"Unknown activation function: {name}. Using ReLU instead.")
        return nn.ReLU()
    return activation_dict[name]


def build_mlp(input_dim: int, output_dim: int, hidden_dim: Optional[int] = None,
              dropout: float = 0.0, activation: str = "relu"):
    """Simple helper to build a 2-layer MLP with ReLU and optional dropout."""
    hidden_dim = hidden_dim if hidden_dim is not None else output_dim
    layers = [nn.Linear(input_dim, hidden_dim), get_activation_function(activation)]
    if dropout and dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class CoordinateEmbedder(nn.Module):
    """Embed 2D/3D coordinates with multiple possible strategies."""

    def __init__(
        self,
        coord_dim: int = 3,
        embed_dim: int = 64,
        method: str = "mlp",
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.method = method
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        
        if method == "mlp":
            self.embedder = build_mlp(coord_dim, embed_dim, dropout=dropout, activation=activation)
        elif method == "sinusoidal":
            # keep a small linear layer and add sinusoidal positional encoding
            self.embedder = nn.Linear(coord_dim, embed_dim)
        elif method == "distance_based":
            # append distance + angle (2 extra values)
            self.embedder = build_mlp(coord_dim + 2, embed_dim, dropout=dropout, activation=activation)
        elif method == "euclidean_distance":
            # *** ADDED: append euclidean distance (1 extra value) ***
            self.embedder = build_mlp(coord_dim + 1, embed_dim, dropout=dropout, activation=activation)
        elif method == "spherical":
            self.embedder = build_mlp(coord_dim, embed_dim, dropout=dropout, activation=activation)
        else:
            self.embedder = build_mlp(coord_dim, embed_dim, dropout=dropout, activation=activation)

    def forward(self, coords: torch.Tensor):
        """
        coords: [N, coord_dim]
        returns: [N, embed_dim]
        """
        if self.method == "sinusoidal":
            pos_enc = torch.zeros(coords.size(0), self.embed_dim,
                                  device=coords.device)
            for i in range(self.coord_dim):
                scale = 10000 ** (2 * i / max(1, self.embed_dim))
                # guard against index overflow when embed_dim < coord_dim*2
                idx_sin = min(i * 2, self.embed_dim - 1)
                idx_cos = min(i * 2 + 1, self.embed_dim - 1)
                pos_enc[:, idx_sin] = torch.sin(coords[:, i] / scale)
                pos_enc[:, idx_cos] = torch.cos(coords[:, i] / scale)
            return self.embedder(coords) + pos_enc[:, : self.embed_dim]

        if self.method == "distance_based":
            distances = torch.norm(coords, dim=1, keepdim=True)
            angles = torch.atan2(coords[:, 1:2], coords[:, 0:1])  # simple 2D angle
            enhanced = torch.cat([coords, distances, angles], dim=1)
            return self.embedder(enhanced)

        if self.method == "euclidean_distance":
            distances = torch.norm(coords, dim=1, keepdim=True) # [N, 1]
            enhanced = torch.cat([coords, distances], dim=1) # [N, coord_dim + 1]
            return self.embedder(enhanced)

        if self.method == "spherical":
            r = torch.norm(coords, dim=1, keepdim=True)
            theta = torch.acos(coords[:, 2:3] / (r + 1e-8))
            phi = torch.atan2(coords[:, 1:2], coords[:, 0:1])
            spherical = torch.cat([r, theta, phi], dim=1)
            return self.embedder(spherical)

        # default mlp
        return self.embedder(coords)


class EdgeFeatureProcessor(nn.Module):
    """Simple MLP to project raw edge attributes to a learned edge embedding."""

    def __init__(self, edge_attr_dim: int = 2, hidden_dim: int = 32,
                 dropout: float = 0.0, activation="relu"):
        super().__init__()
        self.edge_mlp = build_mlp(edge_attr_dim, hidden_dim,
                                  hidden_dim, dropout=dropout, activation=activation)

    def forward(self, edge_attr: torch.Tensor):
        """edge_attr: [E, edge_attr_dim] -> [E, hidden_dim]"""
        return self.edge_mlp(edge_attr)


class DistanceAwareGCNConv(MessagePassing):
    """
    A message-passing layer that incorporates per-edge weights computed from
    edge attributes (distance-aware). Outputs `out_channels` per node.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 2, activation="relu"):
        super().__init__(aggr="add")
        self.linear = nn.Linear(in_channels, out_channels)
        # Map raw edge attributes -> multiplicative weights per output channel
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_channels),
            get_activation_function(activation),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor):
        """
        x: [N, in_channels]
        edge_attr: [E, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor):
        """
        x_j: [E, in_channels] (source node features for each edge)
        edge_attr: [E, edge_dim]
        returns: [E, out_channels]
        """
        x_j_proj = self.linear(x_j)              # [E, out_channels]
        edge_w = self.edge_mlp(edge_attr)        # [E, out_channels]
        return edge_w * x_j_proj                 # element-wise modulation


class MultiHeadGATWithEdge(MessagePassing):
    """
    Multi-head GAT variant that also takes per-edge attributes into attention
    computation. This class inherits MessagePassing and performs per-head
    attention normalization.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 4, edge_dim: int = 2):
        # aggregate by sum
        super().__init__(aggr="add")
        self.heads = heads
        # out_channels is final output size (heads * out_per_head)
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.out_per_head = out_channels // heads

        # Linear to map node features -> (heads * out_per_head)
        self.W = nn.Linear(in_channels, self.heads * self.out_per_head,
                           bias=False)

        # Small MLP to map raw edge_attr -> heads-sized vector (one scalar per head)
        self.edge_mlp = nn.Linear(edge_dim, heads)

        # Attention scoring: input per head will be [2*out_per_head + 1] (x_i, x_j, edge_score)
        self.att = nn.Linear(2 * self.out_per_head + 1, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]
        """
        N = x.size(0)
        # project and reshape per head: [N, heads, out_per_head]
        x_proj = self.W(x).view(N, self.heads, self.out_per_head)

        row, col = edge_index  # row: source idx, col: target idx
        x_i = x_proj[row]      # [E, heads, out_per_head]
        x_j = x_proj[col]      # [E, heads, out_per_head]

        # edge per-head scores: [E, heads]
        edge_scores = self.edge_mlp(edge_attr)

        # build attention input per head: [E, heads, 2*out_per_head + 1]
        att_input = torch.cat(
            [x_i, x_j, edge_scores.unsqueeze(-1)], dim=-1
        )  # last dim = 2*out_per_head + 1

        # compute raw attention coefficients per edge per head: [E, heads]
        alpha_raw = self.att(att_input).squeeze(-1)
        alpha_raw = F.leaky_relu(alpha_raw)

        # normalize (softmax) per target node, for each head separately
        alpha = []
        for h in range(self.heads):
            alpha_h = softmax(alpha_raw[:, h], col, num_nodes=N)
            alpha.append(alpha_h)
        alpha = torch.stack(alpha, dim=1)  # [E, heads]

        # aggregate per head using propagate; message uses alpha for weighting
        out = torch.zeros(N, self.heads, self.out_per_head, device=x.device)
        for h in range(self.heads):
            out[:, h, :] = self.propagate(edge_index, x=x_proj[:, h, :],
                                          alpha=alpha[:, h])
        # flatten heads: [N, heads * out_per_head]
        return out.view(N, -1)

    def message(self, x_j: torch.Tensor, alpha: torch.Tensor):
        """Message: scale neighbor features by attention weights."""
        return alpha.unsqueeze(-1) * x_j


def get_conv_layer(gnn_type: str,
                   in_channels: int,
                   out_channels: int,
                   edge_dim: int = 2,
                   activation: str = "relu",
                   special_config: OmegaConf = None
    ):
    """
    Factory to produce convolution layers given a type string.
    gnn_type: 'gcn', 'gin', 'gat', 'sage', 'distance_aware', 'gat_edge'
    """
    gnn_type = gnn_type.lower()
    if gnn_type == "gcn":
        return GCNConv(in_channels, out_channels)
    if gnn_type == "gin":
        # use a lightweight MLP inside GIN
        gin_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        return GINConv(gin_nn, eps=special_config.eps, train_eps=special_config.train_eps)
    if gnn_type == "gat":
        # produce out_channels as final dim: set head out dim accordingly
        assert special_config.heads > 0
        return GATConv(in_channels, out_channels // special_config.heads, heads=special_config.heads)
    if gnn_type == "sage":
        return SAGEConv(in_channels, out_channels)
    if gnn_type == "distance_aware":
        return DistanceAwareGCNConv(in_channels, out_channels, edge_dim=edge_dim, activation=activation)
    if gnn_type == "gat_edge":
        return MultiHeadGATWithEdge(in_channels, out_channels, heads=special_config.heads, edge_dim=edge_dim)
    raise ValueError(f"Unknown gnn_type: {gnn_type}")


__all__ = [
    "get_activation_function",
    "build_mlp",
    "CoordinateEmbedder",
    "EdgeFeatureProcessor",
    "DistanceAwareGCNConv",
    "MultiHeadGATWithEdge",
    "get_conv_layer",
]
