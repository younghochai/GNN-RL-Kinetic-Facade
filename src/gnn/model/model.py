import logging
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    GINConv,
)
from torch_geometric.utils import softmax

from .layers import (
    build_mlp,
    get_activation_function,
    CoordinateEmbedder,
    EdgeFeatureProcessor,
    get_conv_layer,
    DistanceAwareGCNConv,
    MultiHeadGATWithEdge,
)

logger = logging.getLogger(__name__)


def get_activation_f(activation: str):
    return {
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "elu": F.elu,
        "gelu": F.gelu,
        "swish": F.silu,
        "silu": F.silu,
        "tanh": F.tanh,
        "sigmoid": F.sigmoid,
        "hardswish": F.hardswish,
        "hardtanh": F.hardtanh,
    }[activation]


class GraphRegressorV2(nn.Module):
    """
    Graph-level regressor.

    - Input: a PyG `data` object (data.x, data.edge_index, optional data.edge_attr,
      optional data.global_x, data.batch)
    - Supports multiple GNN layer types.
    - Edge attributes may be used in two ways:
        * For special convs that accept edge_attr directly (distance_aware, gat_edge)
          the raw edge_attr is passed to that conv.
        * For general convs, edge attributes are processed into edge embeddings,
          aggregated to target nodes and added to node embeddings before convolution.
    - Global features (data.global_x) are optional.
    """

    def __init__(
        self,
        gnn_type: str,
        config: OmegaConf,
    ):
        super().__init__()
        self.config = config

        in_channels = config.in_channels
        hidden_channels = config.hidden_channels
        global_dim = config.global_dim
        out_dim = config.out_dim
        num_layers = config.num_layers
        use_edge_attr = config.use_edge_attr
        use_global_features = config.use_global_features
        user_global_coord_embedder = config.user_global_coord_embedder
        coord_embed_method = config.coord_embed_method
        pooling_method = config.pooling_method
        activation_function = config.activation_function
        dropout = config.dropout
        edge_dim = config.edge_dim
        edge_embedding_dim = config.edge_embedding_dim
        use_node_coord = config.use_node_coord if 'use_node_coord' in config else True

        self.gnn_type = gnn_type.lower()
        self.use_edge_attr = use_edge_attr
        self.use_global_features = use_global_features
        self.user_global_coord_embedder = user_global_coord_embedder
        self.use_node_coord = use_node_coord
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.activation_fn = get_activation_f(activation_function)
        # coordinate embedder (we choose half of hidden for coords, half for other node features)
        coord_embed_dim = hidden_channels // 2

        if use_node_coord:
            self.coord_embedder = CoordinateEmbedder(
                coord_dim=3,
                embed_dim=coord_embed_dim,
                method=coord_embed_method,
                dropout=dropout,
                activation=activation_function,
            )
        if user_global_coord_embedder:
            self.global_coord_embedder = CoordinateEmbedder(
                coord_dim=3,
                embed_dim=3,
                method=coord_embed_method,
                dropout=dropout,
                activation=activation_function,
            )

        # node MLP for non-coordinate node features
        other_in = max(0, in_channels - 3) if use_node_coord else in_channels
        output_channels = hidden_channels // 2 if use_node_coord else hidden_channels
        self.node_mlp = nn.Sequential(
            nn.Linear(other_in, output_channels),
            get_activation_function(activation_function),
            nn.Dropout(dropout),
        )

        # Build GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = get_conv_layer(
                gnn_type=self.gnn_type,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                activation=activation_function,
                special_config=config.special_config,
            )
            self.gnn_layers.append(conv)

        # Edge attribute processing for the "general convs" path.
        if use_edge_attr:
            # edge_processor maps raw edge_attr -> edge_embedding_dim
            self.edge_processor = EdgeFeatureProcessor(
                edge_attr_dim=edge_dim,
                hidden_dim=edge_embedding_dim,
                dropout=dropout,
                activation=activation_function,
            )
            # If the produced edge embedding size differs from node hidden size,
            # project to node hidden size before aggregation
            if edge_embedding_dim != hidden_channels:
                self.edge_proj = nn.Linear(edge_embedding_dim, hidden_channels)
            else:
                self.edge_proj = None

        # Global MLP if requested
        if use_global_features:
            self.global_mlp = nn.Sequential(
                nn.Linear(global_dim, hidden_channels),
                get_activation_function(activation_function),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
        else:
            logger.info("Global features disabled for GraphRegressorV2")

        # Pooling selection
        if pooling_method == "mean":
            self.pooling = global_mean_pool
        elif pooling_method == "max":
            self.pooling = global_max_pool
        elif pooling_method == "add":
            self.pooling = global_add_pool
        elif pooling_method == "attention":
            self.attention_pool = nn.Linear(hidden_channels, 1)
            self.pooling = self._attention_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

        # Final readout
        input_dim = hidden_channels
        if use_global_features:
            input_dim += hidden_channels

        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_dim),
        )

    def _attention_pool(self, x: torch.Tensor, batch: torch.Tensor):
        """
        Attention-based pooling.
        Normalize attention per-graph using torch_geometric.utils.softmax with
        index=batch so every graph's nodes are normalized separately.
        """
        scores = self.attention_pool(x).squeeze(-1)  # [N]
        attn = softmax(scores, batch)  # normalized per graph (using batch indices)
        return global_add_pool(x * attn.unsqueeze(-1), batch)

    def forward(self, data: Data):
        """
        data: PyG data object. Required fields:
          - data.x: [N, in_channels] where first 3 dims are coordinates
          - data.edge_index: [2, E]
          - data.batch: [N]
        Optional:
          - data.edge_attr: [E, edge_dim] (used if use_edge_attr=True)
          - data.global_x: [num_graphs, global_dim] (used if use_global_features=True)
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = getattr(data, "edge_attr", None) if self.use_edge_attr else None
        global_x = getattr(data, "global_x", None) if self.use_global_features else None

        # split coordinates and other node features
        if self.use_node_coord:
            coords = x[:, :3]  # [Node_X, Node_Y, Node_Z]
            other_feats = x[:, 3:]  # [열림각도]
            coord_emb = self.coord_embedder(coords)                 # [N, hidden/2]
            other_emb = self.node_mlp(other_feats)                 # [N, hidden/2]
            x_emb = torch.cat([coord_emb, other_emb], dim=1)       # [N, hidden]
        else:
            x_emb = self.node_mlp(x)                 # [N, hidden]

        # GNN message passing — handle edge_attr according to conv capability
        for i, conv in enumerate(self.gnn_layers):
            # If edge attributes are provided and conv expects them directly
            if self.use_edge_attr and edge_attr is not None and \
               isinstance(conv, (DistanceAwareGCNConv, MultiHeadGATWithEdge)):
                x_emb = conv(x_emb, edge_index, edge_attr)
            elif self.use_edge_attr and edge_attr is not None:
                # General convs do not take edge_attr directly:
                # 1) project raw edge_attr -> edge embeddings
                # 2) aggregate edge embeddings to the *target* node and add to node features
                edge_emb = self.edge_processor(edge_attr)  # [E, edge_emb_dim]
                if self.edge_proj is not None:
                    edge_emb = self.edge_proj(edge_emb)     # -> [E, hidden]

                # aggregate edge embeddings to target nodes
                row, col = edge_index
                node_msg = torch.zeros_like(x_emb)
                node_msg.index_add_(0, col, edge_emb)  # accumulate into target node rows

                x_input = x_emb + node_msg
                x_emb = conv(x_input, edge_index)
            else:
                # no edge attributes used
                x_emb = conv(x_emb, edge_index)

            # apply activation for all but the last GNN layer
            if i < len(self.gnn_layers) - 1:
                x_emb = self.activation_fn(x_emb)

        # graph-level pooling
        graph_emb = self.pooling(x_emb, batch)

        # optional global features
        if self.use_global_features and global_x is not None:
            if self.user_global_coord_embedder:
                global_coord_emb = self.global_coord_embedder(global_x[:, :3])
                global_x = torch.cat([global_coord_emb, global_x[:, 3:]], dim=-1)
            global_x = self.global_mlp(global_x)
            h = torch.cat([graph_emb, global_x], dim=-1)
        else:
            h = graph_emb

        out = self.fc_out(h)
        return out

    def get_hidden_state(self, data: Data):
        """
        data: PyG data object. Required fields:
          - data.x: [N, in_channels] where first 3 dims are coordinates
          - data.edge_index: [2, E]
          - data.batch: [N]
        Optional:
          - data.edge_attr: [E, edge_dim] (used if use_edge_attr=True)
          - data.global_x: [num_graphs, global_dim] (used if use_global_features=True)
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = getattr(data, "edge_attr", None) if self.use_edge_attr else None
        global_x = getattr(data, "global_x", None) if self.use_global_features else None

        # split coordinates and other node features
        if self.use_node_coord:
            coords = x[:, :3]  # [Node_X, Node_Y, Node_Z]
            other_feats = x[:, 3:]  # [열림각도]
            coord_emb = self.coord_embedder(coords)                 # [N, hidden/2]
            other_emb = self.node_mlp(other_feats)                 # [N, hidden/2]
            x_emb = torch.cat([coord_emb, other_emb], dim=1)       # [N, hidden]
        else:
            x_emb = self.node_mlp(x)                 # [N, hidden]

        # GNN message passing — handle edge_attr according to conv capability
        for i, conv in enumerate(self.gnn_layers):
            # If edge attributes are provided and conv expects them directly
            if self.use_edge_attr and edge_attr is not None and \
               isinstance(conv, (DistanceAwareGCNConv, MultiHeadGATWithEdge)):
                x_emb = conv(x_emb, edge_index, edge_attr)
            elif self.use_edge_attr and edge_attr is not None:
                # General convs do not take edge_attr directly:
                # 1) project raw edge_attr -> edge embeddings
                # 2) aggregate edge embeddings to the *target* node and add to node features
                edge_emb = self.edge_processor(edge_attr)  # [E, edge_emb_dim]
                if self.edge_proj is not None:
                    edge_emb = self.edge_proj(edge_emb)     # -> [E, hidden]

                # aggregate edge embeddings to target nodes
                row, col = edge_index
                node_msg = torch.zeros_like(x_emb)
                node_msg.index_add_(0, col, edge_emb)  # accumulate into target node rows

                x_input = x_emb + node_msg
                x_emb = conv(x_input, edge_index)
            else:
                # no edge attributes used
                x_emb = conv(x_emb, edge_index)

            # apply activation for all but the last GNN layer
            if i < len(self.gnn_layers) - 1:
                x_emb = self.activation_fn(x_emb)

        # graph-level pooling
        graph_emb = self.pooling(x_emb, batch)

        # optional global features
        if self.use_global_features and global_x is not None:
            if self.user_global_coord_embedder:
                global_coord_emb = self.global_coord_embedder(global_x[:, :3])
                global_x = torch.cat([global_coord_emb, global_x[:, 3:]], dim=-1)
            global_x = self.global_mlp(global_x)
            h = torch.cat([graph_emb, global_x], dim=-1)
        else:
            h = graph_emb

        return h


class MolecularInspiredGNN(nn.Module):
    """
    Molecular-inspired GNN that builds edge features from distances and
    angles between node coordinates, integrates them into node representations,
    applies GIN convolution, and produces a graph-level regression output.

    Input:
        data: PyG data object with fields:
            - data.x: [N, in_channels] (first 3 dims should be coords if present)
            - data.edge_index: [2, E]
            - (optional) data.edge_attr: [E, edge_attr_dim]
            - data.batch: [N]
    """
    EPS = 1e-8

    def __init__(
        self,
        config: OmegaConf,
    ):
        super().__init__()
        self.config = config
        in_channels = config.in_channels
        hidden_channels = config.hidden_channels
        out_dim = config.out_dim
        activation_function = config.activation_function
        use_edge_attr = config.use_edge_attr
        edge_attr_dim = config.edge_dim
        pooling = config.pooling_method
        dropout = config.dropout
        eps = config.special_config.eps
        train_eps = config.special_config.train_eps

        self.gnn_type = 'molecular_inspired'
        self.use_edge_attr = use_edge_attr
        self.activation = get_activation_f(activation_function)
        self.hidden_channels = hidden_channels

        # Node feature MLP
        self.node_mlp = build_mlp(in_channels, hidden_channels, dropout=dropout, activation=activation_function)

        # Per-edge small MLPs for distance and angle (project to quarter-size each)
        per_edge_dim = hidden_channels // 4 if hidden_channels >= 4 else 1
        self.distance_mlp = build_mlp(1, per_edge_dim, dropout=dropout, activation=activation_function)
        self.angle_mlp = build_mlp(1, per_edge_dim, dropout=dropout, activation=activation_function)

        # If external edge_attr exists and you want to consume it, we provide a small projector
        if use_edge_attr:
            self.raw_edge_proj = build_mlp(edge_attr_dim, per_edge_dim, dropout=dropout, activation=activation_function)

        # Combine edge features (distance + angle + optional raw) -> edge_feature_dim
        edge_feature_dim = per_edge_dim * 2 + (per_edge_dim if use_edge_attr else 0)

        # If edge_feature_dim != hidden_channels//2, project later
        self.edge_feature_dim = edge_feature_dim
        if edge_feature_dim != hidden_channels // 2:
            self.edge_to_node = nn.Linear(edge_feature_dim, hidden_channels // 2)
        else:
            self.edge_to_node = None

        # GIN conv expects an MLP inside; sizes match concatenated node + edge contribution
        gin_input_dim = hidden_channels + (hidden_channels // 2)
        gin_nn = nn.Sequential(
            nn.Linear(gin_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.gin_conv = GINConv(gin_nn, eps=eps, train_eps=train_eps)

        # Pooling and final readout
        self.pooling = global_mean_pool if pooling == "mean" else global_mean_pool
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _compute_edge_vectors(coords: torch.Tensor, edge_index: torch.Tensor):
        """
        Vectorized: given coords [N, 3] and edge_index [2, E],
        return pair vectors (src->tgt) and norms.
        """
        row, col = edge_index
        vec_ij = coords[row] - coords[col]         # [E, 3]
        norm_i = torch.norm(coords[row], dim=1, keepdim=True)  # [E,1]
        norm_j = torch.norm(coords[col], dim=1, keepdim=True)  # [E,1]
        return vec_ij, norm_i, norm_j

    def _compute_angles(self, coords: torch.Tensor, edge_index: torch.Tensor):
        """
        Compute angle between vectors of two nodes for each edge:
        angle = arccos( (u·v) / (|u||v| + eps) )
        Here we treat u = coord[src], v = coord[tgt] as position vectors
        relative to origin. This matches original behavior in provided code.
        """
        row, col = edge_index
        vec_i = coords[row]    # [E, 3]
        vec_j = coords[col]    # [E, 3]

        dot = (vec_i * vec_j).sum(dim=1, keepdim=True)   # [E,1]
        norm_i = torch.norm(vec_i, dim=1, keepdim=True)
        norm_j = torch.norm(vec_j, dim=1, keepdim=True)
        denom = norm_i * norm_j + self.EPS
        cos_angle = dot / denom
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angles = torch.acos(cos_angle)  # [E,1]
        return angles

    def forward(self, data: Data):
        """
        Forward using a PyG data object.
        """
        x = data.x                            # [N, in_channels]
        edge_index = data.edge_index          # [2, E]
        batch = data.batch                    # [N]
        coords = x[:, :3] if x.size(1) >= 3 else None
        raw_edge_attr = getattr(data, "edge_attr", None)

        # Node embedding
        node_emb = self.node_mlp(x)           # [N, hidden]

        # Build edge features: distances and angles
        # If raw edge_attr includes distances at a particular column originally,
        # we still compute distances from coords for consistency.
        if coords is not None:
            # distances: Euclidean distance between nodes for each edge
            row, col = edge_index
            dist_vec = coords[row] - coords[col]
            distances = torch.norm(dist_vec, dim=1, keepdim=True)  # [E,1]
            dist_emb = self.distance_mlp(distances)                 # [E, per_edge_dim]

            # angles: using positions (vectorized)
            angles = self._compute_angles(coords, edge_index)      # [E,1]
            angle_emb = self.angle_mlp(angles)                     # [E, per_edge_dim]
        else:
            # fallback zero tensors if coords not provided
            num_edges = edge_index.size(1)
            device = x.device
            dist_emb = torch.zeros((num_edges, self.distance_mlp[-1].out_features),
                                   device=device)
            angle_emb = torch.zeros_like(dist_emb)

        # optional use of raw edge_attr
        if self.use_edge_attr and raw_edge_attr is not None:
            raw_proj = self.raw_edge_proj(raw_edge_attr)             # [E, per_edge_dim]
            edge_features = torch.cat([dist_emb, angle_emb, raw_proj], dim=-1)
        else:
            edge_features = torch.cat([dist_emb, angle_emb], dim=-1)  # [E, edge_feature_dim]

        # project edge features to match node aggregation size if necessary
        if self.edge_to_node is not None:
            edge_features = self.edge_to_node(edge_features)         # [E, hidden//2]

        # aggregate edge features to target nodes (col indicates target in edge_index)
        row, col = edge_index
        num_nodes = x.size(0)
        device = x.device
        edge_dim = edge_features.size(1)
        node_edge_agg = torch.zeros((num_nodes, edge_dim), device=device)
        node_edge_agg.index_add_(0, col, edge_features)  # sum contributions into target nodes

        # Concatenate node embedding and aggregated edge info
        enhanced_x = torch.cat([node_emb, node_edge_agg], dim=-1)  # [N, hidden + hidden//2]

        # GIN convolution
        out = self.gin_conv(enhanced_x, edge_index)   # [N, hidden]
        out = self.activation(out)
        out = self.dropout(out)

        # Graph pooling and readout
        graph_emb = self.pooling(out, batch)
        return self.fc_out(graph_emb)
