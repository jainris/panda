from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import coalesce, remove_self_loops, softmax, to_undirected
from torch_scatter import scatter
from torch_sparse import spspmm

from models.mlp import MLP


def _empty_edge_index(device: torch.device) -> Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _sparse_product(
    left_index: Tensor,
    right_index: Tensor,
    left_size: int,
    inner_size: int,
    right_size: int,
) -> Tensor:
    if left_index.numel() == 0 or right_index.numel() == 0:
        return _empty_edge_index(left_index.device)

    dtype = torch.float32
    left_value = torch.ones(left_index.size(1), device=left_index.device, dtype=dtype)
    right_value = torch.ones(
        right_index.size(1), device=right_index.device, dtype=dtype
    )
    product_index, _ = spspmm(
        left_index,
        left_value,
        right_index,
        right_value,
        left_size,
        inner_size,
        right_size,
        coalesced=False,
    )
    return product_index


def _make_undirected(edge_index: Tensor, num_nodes: int) -> Tensor:
    if edge_index.numel() == 0:
        return _empty_edge_index(edge_index.device)

    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes, reduce="add")
    return coalesce(edge_index, num_nodes=num_nodes)


def incidence_to_undirected_graph(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    num_nodes: int,
) -> Tensor:
    """Projects node-hyperedge incidences to an undirected node graph.

    Complex incidence weights use their imaginary and real components to mark
    source and target memberships, respectively. Directed hyperedges are
    projected by connecting every source to every target. Hyperedges without
    source markers are projected using their ordinary two-section.
    """

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("Incidence edge_index must have shape [2, num_incidences].")
    if edge_index.numel() == 0:
        return _empty_edge_index(edge_index.device)
    if edge_index[0].min() < 0 or edge_index[0].max() >= num_nodes:
        raise ValueError("Incidence node indices fall outside [0, num_nodes).")

    num_hyperedges = int(edge_index[1].max()) + 1
    if edge_index[1].min() < 0:
        raise ValueError("Hyperedge indices must be non-negative.")

    if edge_weight is None or not torch.is_complex(edge_weight):
        incidence = edge_index
        clique_edges = _sparse_product(
            incidence,
            incidence.flip(0),
            num_nodes,
            num_hyperedges,
            num_nodes,
        )
        return _make_undirected(clique_edges, num_nodes)

    if edge_weight.dim() != 1 or edge_weight.numel() != edge_index.size(1):
        raise ValueError(
            "edge_weight must contain one complex value for every incidence."
        )

    hyperedge = edge_index[1]
    source_mask = edge_weight.imag != 0
    target_mask = edge_weight.real != 0
    source_count = scatter(
        source_mask.to(torch.long),
        hyperedge,
        dim=0,
        dim_size=num_hyperedges,
        reduce="sum",
    )
    target_count = scatter(
        target_mask.to(torch.long),
        hyperedge,
        dim=0,
        dim_size=num_hyperedges,
        reduce="sum",
    )

    directed_hyperedge = (source_count > 0) & (target_count > 0)
    role_free_hyperedge = source_count == 0

    directed_source = source_mask & directed_hyperedge[hyperedge]
    directed_target = target_mask & directed_hyperedge[hyperedge]
    directed_edges = _sparse_product(
        edge_index[:, directed_source],
        edge_index[:, directed_target].flip(0),
        num_nodes,
        num_hyperedges,
        num_nodes,
    )

    role_free_incidence = role_free_hyperedge[hyperedge]
    role_free_index = edge_index[:, role_free_incidence]
    role_free_edges = _sparse_product(
        role_free_index,
        role_free_index.flip(0),
        num_nodes,
        num_hyperedges,
        num_nodes,
    )

    projected = torch.cat([directed_edges, role_free_edges], dim=1)
    return _make_undirected(projected, num_nodes)


def build_undirected_graph(data) -> Tensor:
    """Returns the graph used by GCN + PANDA, preferring original graph edges."""

    num_nodes = int(data.x.size(0))
    graph_edge_index = getattr(data, "graph_edge_index", None)
    if graph_edge_index is not None:
        if graph_edge_index.dim() != 2 or graph_edge_index.size(0) != 2:
            raise ValueError("data.graph_edge_index must have shape [2, num_edges].")
        if graph_edge_index.numel() and (
            graph_edge_index.min() < 0 or graph_edge_index.max() >= num_nodes
        ):
            raise ValueError(
                "data.graph_edge_index contains indices outside [0, num_nodes)."
            )
        return _make_undirected(graph_edge_index.long(), num_nodes)

    return incidence_to_undirected_graph(
        data.edge_index,
        getattr(data, "edge_weight", None),
        num_nodes,
    )


class GraphMainPathConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        concat: bool = True,
        dropout: float = 0.0,
        attention_type: str = "gcn+att",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination: str = "only-src",
        residual: bool = False,
        bias: bool = True,
        reset_parameters_at_call: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        kwargs.setdefault("node_dim", 0)
        super().__init__(**kwargs)

        if heads < 1:
            raise ValueError("heads must be positive.")
        if attention_version not in {"v1", "v2", "sigmoid"}:
            raise ValueError(f"Unknown attention version: {attention_version}.")
        if attention_combination not in {"only-src", "src+dst", "mul"}:
            raise ValueError(f"Unknown attention combination: {attention_combination}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.concat = concat
        self.dropout = dropout
        self.attention_version = attention_version
        self.shared_att_weights = shared_att_weights
        self.attention_combination = attention_combination

        self.lin_proj = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att = None
        self.att_src = None
        self.att_dst = None
        self.lin_att = None
        self.lin_att_src = None
        self.lin_att_dst = None

        if shared_att_weights:
            self.att = Parameter(torch.empty(1, heads, out_channels))
            self.lin_att = nn.Linear(out_channels, out_channels, bias=False)
        elif attention_version == "v1":
            self.att_src = Parameter(torch.empty(1, heads, out_channels))
            self.att_dst = Parameter(torch.empty(1, heads, out_channels))
            self.lin_att = nn.Linear(out_channels, out_channels, bias=False)
        else:
            self.att = Parameter(torch.empty(1, heads, out_channels))
            self.lin_att_src = nn.Linear(out_channels, out_channels, bias=False)
            self.lin_att_dst = nn.Linear(out_channels, out_channels, bias=False)

        self.attention_types, self.attention_weights = self._parse_attention_type(
            attention_type
        )
        unknown_types = set(self.attention_types) - {"gcn", "att"}
        if unknown_types:
            raise ValueError(
                f"Unknown graph attention type(s): {sorted(unknown_types)}."
            )

        self.att_wt_lin1 = None
        self.att_wt_lin2 = None
        if self.attention_weights is None and self.attention_combination != "mul":
            self.att_wt_lin1 = nn.Linear(out_channels, 8, bias=False)
            self.att_wt_lin2 = nn.Linear(8, len(self.attention_types), bias=False)

        total_out_channels = out_channels * (heads if concat else 1)
        self.res = (
            nn.Linear(in_channels, total_out_channels, bias=False) if residual else None
        )
        self.bias = Parameter(torch.empty(total_out_channels)) if bias else None

        if reset_parameters_at_call:
            self.reset_parameters()

    @staticmethod
    def _parse_attention_type(attention_type: str):
        parts = attention_type.split("+")
        if not parts or any(not part for part in parts):
            raise ValueError(f"Invalid graph attention type: {attention_type!r}.")

        has_weight = ["=" in part for part in parts]
        if any(has_weight) and not all(has_weight):
            raise ValueError(
                "Either all graph attention components must have weights or none."
            )
        if all(has_weight):
            parsed = [part.split("=", maxsplit=1) for part in parts]
            names = [name for name, _ in parsed]
            try:
                weights = [float(weight) for _, weight in parsed]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid graph attention weights: {attention_type!r}."
                ) from exc
            return names, weights
        if len(parts) == 1:
            return parts, [1.0]
        return parts, None

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.lin_proj.weight)
        if self.att is not None:
            glorot(self.att)
        if self.att_src is not None:
            glorot(self.att_src)
        if self.att_dst is not None:
            glorot(self.att_dst)
        if self.lin_att is not None:
            glorot(self.lin_att.weight)
        if self.lin_att_src is not None:
            glorot(self.lin_att_src.weight)
        if self.lin_att_dst is not None:
            glorot(self.lin_att_dst.weight)
        if self.att_wt_lin1 is not None:
            glorot(self.att_wt_lin1.weight)
        if self.att_wt_lin2 is not None:
            glorot(self.att_wt_lin2.weight)
        if self.res is not None:
            glorot(self.res.weight)
        if self.bias is not None:
            zeros(self.bias)

    def _calculate_attention_alpha(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        if self.lin_att is not None:
            x_src = self.lin_att(x_src)
            x_dst = self.lin_att(x_dst)
        else:
            x_src = self.lin_att_src(x_src)
            x_dst = self.lin_att_dst(x_dst)

        if self.attention_version == "v1":
            if self.shared_att_weights:
                alpha_src = (x_src * self.att).sum(dim=-1)
                alpha_dst = (x_dst * self.att).sum(dim=-1)
            else:
                alpha_src = (x_src * self.att_src).sum(dim=-1)
                alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
            alpha = F.leaky_relu(alpha_src + alpha_dst, self.negative_slope)
        else:
            score = x_src + x_dst
            if self.attention_version == "sigmoid":
                score = F.relu(score)
                return torch.sigmoid((score * self.att).sum(dim=-1))
            score = F.leaky_relu(score, self.negative_slope)
            alpha = (score * self.att).sum(dim=-1)

        alpha = softmax(alpha, index=dst)
        return F.dropout(alpha, p=self.dropout, training=self.training)

    def calculate_alphas(
        self,
        x: Tensor,
        edge_index: Tensor,
        gcn_alpha: Tensor,
    ) -> Tensor:
        alphas = []
        for attention_type in self.attention_types:
            if attention_type == "gcn":
                alphas.append(gcn_alpha.view(-1, 1))
            else:
                alphas.append(self._calculate_attention_alpha(x, edge_index))

        if self.attention_combination == "mul":
            alpha = torch.ones_like(alphas[0])
            for component in alphas:
                alpha = alpha * component
            return alpha

        if self.attention_weights is not None:
            alpha = torch.zeros_like(alphas[0])
            for weight, component in zip(self.attention_weights, alphas):
                alpha = alpha + weight * component
            return alpha

        src, dst = edge_index
        src_features = self.att_wt_lin1(x[src])
        if self.attention_combination == "src+dst":
            mix_features = src_features + self.att_wt_lin1(x[dst])
        else:
            mix_features = src_features
        mix_features = F.leaky_relu(mix_features, self.negative_slope)
        mix_weights = F.softmax(self.att_wt_lin2(mix_features), dim=-1)

        alpha = torch.zeros_like(alphas[0])
        for index, component in enumerate(alphas):
            alpha = alpha + mix_weights[:, :, index] * component
        return alpha

    def _prepare(
        self, x: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        edge_index, gcn_alpha = gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=x.size(0),
            improved=False,
            add_self_loops=True,
            flow=self.flow,
            dtype=x.dtype,
        )
        projected = self.lin_proj(x).view(-1, self.heads, self.out_channels)
        alpha = self.calculate_alphas(projected, edge_index, gcn_alpha)
        return projected, edge_index, alpha, gcn_alpha

    def _finish(self, out: Tensor, residual: Optional[Tensor]) -> Tensor:
        if self.concat:
            out = out.reshape(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if residual is not None:
            out = out + residual
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        residual = self.res(x) if self.res is not None else None
        projected, edge_index, alpha, _ = self._prepare(x, edge_index)
        out = self.propagate(edge_index, x=projected, alpha=alpha)
        return self._finish(out, residual)

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j


class GraphDualPathConv(GraphMainPathConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aux_dim: int = 16,
        aux_type: str = "diff-mlp",
        symmetric_aux: bool = False,
        **kwargs,
    ):
        kwargs["reset_parameters_at_call"] = False
        super().__init__(in_channels, out_channels, **kwargs)

        self.reduction_src = nn.Linear(out_channels, aux_dim)
        self.reduction_dst = nn.Linear(out_channels, aux_dim)
        self.reduction_out = nn.Linear(aux_dim, 1)
        if aux_type == "diff-mlp":
            self.reduction_lin = nn.Linear(out_channels, out_channels)
        elif aux_type == "same-mlp":
            self.reduction_lin = None
        else:
            raise ValueError(f"Unknown auxiliary type: {aux_type}.")
        self.symmetric_aux = symmetric_aux
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.reduction_src.reset_parameters()
        self.reduction_dst.reset_parameters()
        self.reduction_out.reset_parameters()
        if self.reduction_lin is not None:
            self.reduction_lin.reset_parameters()

    def calculate_auxiliary_gate(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        score = self.reduction_src(x[src]) + self.reduction_dst(x[dst])
        score = self.reduction_out(F.relu(score))
        if self.symmetric_aux:
            return torch.tanh(score)
        return torch.sigmoid(score)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        residual = self.res(x) if self.res is not None else None
        projected, edge_index, alpha, gcn_alpha = self._prepare(x, edge_index)
        sigma = self.calculate_auxiliary_gate(projected, edge_index)
        x_neg = (
            self.reduction_lin(projected)
            if self.reduction_lin is not None
            else -projected
        )
        out = self.propagate(
            edge_index,
            x=projected,
            x_neg=x_neg,
            alpha=alpha.unsqueeze(-1),
            sigma=sigma,
            gcn_alpha=gcn_alpha.view(-1, 1, 1),
        )
        return self._finish(out, residual)

    def message(
        self,
        x_j: Tensor,
        x_neg_j: Tensor,
        alpha: Tensor,
        sigma: Tensor,
        gcn_alpha: Tensor,
    ) -> Tensor:
        if self.symmetric_aux:
            if self.reduction_lin is None:
                reduced = sigma * x_j
            else:
                reduced = F.relu(sigma) * x_j + F.relu(-sigma) * x_neg_j
            return alpha * reduced

        if self.attention_combination == "mul":
            return alpha * x_j + (gcn_alpha - alpha) * sigma * x_j
        return alpha * x_j + (1.0 - alpha) * sigma * x_neg_j


def build_graph_conv(conv_name: str, **kwargs) -> GraphMainPathConv:
    if conv_name == "att":
        kwargs.pop("aux_type", None)
        kwargs.pop("symmetric_aux", None)
        return GraphMainPathConv(**kwargs)
    if conv_name == "aux":
        return GraphDualPathConv(**kwargs)
    raise ValueError(f"Unknown graph convolution name: {conv_name}.")


class GCNMainPath(nn.Module):
    """GCN with PANDA's main path, also used for the GCN/GAT baselines."""
    def __init__(
        self,
        num_features: int,
        hidden: int,
        label_dim: int,
        layer: int,
        dropout: float,
        heads: int = 1,
        attention_type: str = "gcn+att",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination: str = "only-src",
        att_dropout: Optional[float] = None,
        conv_name: str = "att",
        aux_type: str = "diff-mlp",
        symmetric_aux: bool = False,
        args=None,
    ):
        super().__init__()
        if layer < 1:
            raise ValueError("GCNMainPath requires at least one convolution layer.")

        attention_dropout = dropout if att_dropout is None else att_dropout
        convs = []
        for index in range(layer):
            in_channels = num_features if index == 0 else heads * hidden
            convs.append(
                build_graph_conv(
                    conv_name,
                    in_channels=in_channels,
                    out_channels=hidden,
                    heads=heads,
                    concat=True,
                    dropout=attention_dropout,
                    attention_type=attention_type,
                    attention_version=attention_version,
                    shared_att_weights=shared_att_weights,
                    attention_combination=attention_combination,
                    residual=index > 0,
                    aux_type=aux_type,
                    symmetric_aux=symmetric_aux,
                )
            )
        self.convs = nn.ModuleList(convs)
        self.classifier = MLP(
            in_channels=heads * hidden,
            hidden_channels=args.Classifier_hidden,
            out_channels=label_dim,
            num_layers=args.Classifier_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )
        self.dropout = dropout
        self._cached_edge_index = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def _get_edge_index(self, data) -> Tensor:
        if (
            self._cached_edge_index is None
            or self._cached_edge_index.device != data.x.device
        ):
            self._cached_edge_index = build_undirected_graph(data)
        return self._cached_edge_index

    def forward(self, data) -> Tensor:
        x = data.x
        edge_index = self._get_edge_index(data)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GCNPANDA(GCNMainPath):
    """GCN with both PANDA paths (the GCN + PANDA row of Table 2)."""
    def __init__(self, *args, **kwargs):
        kwargs["conv_name"] = "aux"
        super().__init__(*args, **kwargs)
