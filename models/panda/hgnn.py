from typing import Optional
import numpy as np

import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    scatter,
)

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.conv import MessagePassing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP


class MainPathConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        concat: bool = True,
        mask_self_loops: bool = False,
        dropout: float = 0.0,
        attention_type: str = "hgcn",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination: str = "only-src",
        reset_parameters_at_call: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        kwargs.setdefault("node_dim", 0)
        super(MainPathConv, self).__init__(**kwargs)

        kwargs.setdefault("flow", "target_to_source")

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Attention
        # self.att = Parameter(torch.empty(1, heads, out_channels))
        self.att, self.att_src, self.att_dst = None, None, None
        self.lin, self.lin_src, self.lin_dst = None, None, None
        self.att2, self.att2_src, self.att2_dst = None, None, None
        self.shared_att_weights = shared_att_weights
        if shared_att_weights:
            self.att = Parameter(torch.empty(1, heads, in_channels))
            self.lin = torch.nn.Linear(in_channels, in_channels, bias=False)

            if "diff" in attention_version:
                self.att2 = Parameter(torch.empty(1, heads, in_channels))
        else:
            if "v1" in attention_version:
                self.att_src = Parameter(torch.empty(1, heads, in_channels))
                self.att_dst = Parameter(torch.empty(1, heads, in_channels))
                self.lin = torch.nn.Linear(in_channels, in_channels, bias=False)

                if "diff" in attention_version:
                    self.att2_src = Parameter(torch.empty(1, heads, in_channels))
                    self.att2_dst = Parameter(torch.empty(1, heads, in_channels))
            elif "v2" in attention_version or "mul" in attention_version:
                self.att = Parameter(torch.empty(1, heads, in_channels))
                self.lin_src = torch.nn.Linear(in_channels, in_channels, bias=False)
                self.lin_dst = torch.nn.Linear(in_channels, in_channels, bias=False)

                if "diff" in attention_version:
                    self.att2 = Parameter(torch.empty(1, heads, in_channels))
        self.heads = heads
        self.negative_slope = negative_slope
        self.concat = concat
        self.dropout = dropout

        self.mask_self_loops = mask_self_loops
        # self.attention_type = attention_type
        attention_types = attention_type.split("+")
        attention_weights = [1.0]
        if len(attention_types) == 1:
            self.attention_types = attention_types
            self.attention_weights = attention_weights
        else:
            if len(attention_types[0].split("=")) == 2:
                attention_types, attention_weights = zip(
                    *[x.split("=") for x in attention_types]
                )
                attention_weights = [float(x) for x in attention_weights]
            else:
                attention_weights = None
                attention_types = attention_types
            self.attention_types = attention_types
            self.attention_weights = attention_weights

        self.att_wt_lin1, self.att_wt_lin2 = None, None
        if self.attention_weights is None:
            # Dynamic weighting
            self.att_wt_lin1 = torch.nn.Linear(in_channels, 8, bias=False)
            self.att_wt_lin2 = torch.nn.Linear(8, len(self.attention_types), bias=False)

        self.attention_version = attention_version
        self.attention_combination = attention_combination

        self.gate_lin = None

        if reset_parameters_at_call:
            self.reset_parameters()

    def reset_parameters(self):
        # self.att.reset_parameters()
        if self.att is not None:
            glorot(self.att)
        if self.att_src is not None:
            glorot(self.att_src)
        if self.att_dst is not None:
            glorot(self.att_dst)
        if self.lin is not None:
            glorot(self.lin.weight)
        if self.lin_src is not None:
            glorot(self.lin_src.weight)
        if self.lin_dst is not None:
            glorot(self.lin_dst.weight)
        if self.att_wt_lin1 is not None:
            glorot(self.att_wt_lin1.weight)
        if self.att_wt_lin2 is not None:
            glorot(self.att_wt_lin2.weight)
        if self.att2 is not None:
            glorot(self.att2)
        if self.att2_src is not None:
            glorot(self.att2_src)
        if self.att2_dst is not None:
            glorot(self.att2_dst)
        if self.gate_lin is not None:
            glorot(self.gate_lin.weight)

    def calculate_alphas(self, x_src, x_dst, edge_index, edge_weight):
        alphas = []
        for attention_type in self.attention_types:
            alpha = self.calculate_alpha(
                attention_type,
                x_src,
                x_dst,
                edge_index,
                edge_weight,
            )
            alphas.append(alpha)
        if self.attention_combination == "mul":
            alpha = 1.0
            for a in alphas:
                alpha = alpha * a
        elif self.attention_weights is not None:
            alpha = 0.0
            for i, attention_type in enumerate(self.attention_types):
                alpha = alpha + self.attention_weights[i] * alphas[i]
        else:
            x_src = x_src.view(-1, self.heads, self.in_channels)  # [N, H, in_channels]
            x_src = self.att_wt_lin1(x_src)
            x_src = x_src[edge_index[0]]  # [E, H, 8]

            if self.attention_combination == "only-src":
                x = x_src
            elif self.attention_combination == "not-none":
                if x_dst is not None:
                    x_dst = self.att_wt_lin1(x_dst)
                    x_dst = x_dst[edge_index[1]]  # [E, H, 8]
                    x = x_src + x_dst
                else:
                    x = x_src
            elif self.attention_combination == "gedi-if-none":
                if x_dst is None:
                    x_dst = self.aggregate_structural_prior(x_src, edge_index, edge_weight)
                x_dst = x_dst.view(-1, self.heads, self.in_channels)
                x_dst = self.att_wt_lin1(x_dst)
                x_dst = x_dst[edge_index[1]]  # [E, H, 8]
                x = x_src + x_dst
            elif self.attention_combination == "gedi":
                x_dst = self.aggregate_structural_prior(x_src, edge_index, edge_weight)
                x_dst = x_dst.view(-1, self.heads, self.in_channels)
                x_dst = self.att_wt_lin1(x_dst)
                x_dst = x_dst[edge_index[1]]  # [E, H, 8]
                x = x_src + x_dst
            elif self.attention_combination == "src+dst":
                x_dst = x_dst.view(-1, self.heads, self.in_channels)
                x_dst = self.att_wt_lin1(x_dst)
                x_dst = x_dst[edge_index[1]]  # [E, H, 8]
                x = x_src + x_dst
            else:
                raise ValueError(
                    f"Unknown attention combination: {self.attention_combination}."
                )

            x = F.leaky_relu(x, self.negative_slope)
            att_wts = self.att_wt_lin2(x)
            att_wts = F.softmax(att_wts, dim=-1)  # [E, H, num_attention_types]

            alpha = 0.0
            for i, attention_type in enumerate(self.attention_types):
                alpha = alpha + att_wts[:, :, i] * alphas[i]
        return alpha

    def calculate_alpha(
        self,
        attention_type,
        x_src,
        x_dst,
        edge_index,
        edge_weight,
    ):
        if attention_type == "gedi":
            return self.calculate_structural_prior(edge_index, x_src.dtype)
        elif attention_type == "att":
            return self.calculate_alpha_att(x_src, x_dst, edge_index)
        elif attention_type == "hgat":
            if x_dst is None:
                return self.calculate_attention_with_prior_embeddings(x_src, edge_index, edge_weight)
            return self.calculate_alpha_att(x_src, x_dst, edge_index)
        elif attention_type == "gedi-att":
            return self.calculate_attention_with_prior_embeddings(x_src, edge_index, edge_weight)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}.")

    def calculate_structural_prior(self, edge_index, dtype):
        src = torch.ones(edge_index.shape[1], dtype=dtype, device=edge_index.device)
        degs1 = scatter(src, edge_index[0], dim=0, reduce="sum")
        degs2 = scatter(src, edge_index[1], dim=0, reduce="sum")

        degs1 = torch.pow(degs1, -0.5)
        degs2 = torch.pow(degs2, -0.5)

        degs1[degs1 == float("inf")] = 0
        degs2[degs2 == float("inf")] = 0

        alpha = degs1[edge_index[0]] * degs2[edge_index[1]]
        alpha = alpha.view(-1, 1)  # [E, 1]
        return alpha

    def calculate_alpha_att(self, x_src, x_dst, edge_index):
        if self.attention_version == "v1":
            return self.calculate_alpha_attv1(x_src, x_dst, edge_index)
        elif self.attention_version == "v2":
            return self.calculate_alpha_attv2(x_src, x_dst, edge_index)
        else:
            raise ValueError(f"Unknown attention version: {self.attention_version}.")

    def calculate_alpha_attv1(self, x_src, x_dst, edge_index):
        src, dst = edge_index

        x_src = x_src[src]  # [E, in_channels * H]
        x_dst = x_dst[dst]  # [E, in_channels * H]

        H, C = self.heads, self.in_channels
        x_src = x_src.view(-1, H, C)  # [E, H, in_channels]
        x_dst = x_dst.view(-1, H, C)  # [E, H, in_channels]

        x_src = self.lin(x_src)
        x_dst = self.lin(x_dst)

        if self.shared_att_weights:
            x_src = (x_src * self.att).sum(dim=-1)
            x_dst = (x_dst * self.att).sum(dim=-1)
        else:
            x_src = (x_src * self.att_src).sum(dim=-1)
            x_dst = (x_dst * self.att_dst).sum(dim=-1)

        x = x_src + x_dst
        alpha = F.leaky_relu(x, self.negative_slope)
        alpha = softmax(alpha, index=dst)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def calculate_alpha_attv2(self, x_src, x_dst, edge_index):
        src, dst = edge_index

        x_src = x_src[src]  # [E, in_channels * H]
        x_dst = x_dst[dst]  # [E, in_channels * H]

        H, C = self.heads, self.in_channels
        x_src = x_src.view(-1, H, C)  # [E, H, in_channels]
        x_dst = x_dst.view(-1, H, C)  # [E, H, in_channels]
        if self.shared_att_weights:
            x_src = self.lin(x_src)
            x_dst = self.lin(x_dst)
        else:
            x_src = self.lin_src(x_src)
            x_dst = self.lin_dst(x_dst)

        x = x_src + x_dst
        x = F.leaky_relu(x, self.negative_slope)

        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index=dst)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def aggregate_structural_prior(self, x_src, edge_index, edge_weight):
        gedi_alpha = self.calculate_structural_prior(edge_index, x_src.dtype)

        H, C = self.heads, self.in_channels
        x_src_temp = x_src.view(-1, H, C)

        out = self.propagate(
            edge_index,
            x=x_src_temp,
            alpha=gedi_alpha,
            gating_score=None,
            size=None,
        )

        return out

    def calculate_attention_with_prior_embeddings(self, x_src, edge_index, edge_weight):
        x_dst = self.aggregate_structural_prior(x_src, edge_index, edge_weight)

        return self.calculate_alpha_att(x_src, x_dst, edge_index)

    def calculate_gating_score(self, x):
        gating_score = F.sigmoid(self.gate_lin(F.relu(x)))
        return gating_score.view(-1, self.heads, self.in_channels)

    def forward(
        self,
        x_src: torch.FloatTensor,
        x_dst: Optional[torch.FloatTensor],
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
    ):
        alpha = self.calculate_alphas(x_src, x_dst, edge_index, edge_weight)

        gating_score = None
        if self.gate_lin is not None:
            gating_score = self.calculate_gating_score(x_src)

        H, C = self.heads, self.in_channels
        x_src = x_src.view(-1, H, C)

        out = self.propagate(
            edge_index,
            x=x_src,
            alpha=alpha,
            gating_score=gating_score,
            size=None,
        )

        return out

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

    def update(self, inputs: Tensor, gating_score: Optional[Tensor] = None):
        if gating_score is not None:
            inputs = inputs * gating_score
        if self.concat:
            inputs = inputs.view(-1, self.heads * self.in_channels)
        else:
            inputs = inputs.mean(dim=1)
        return inputs


class DualPathConv(MainPathConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        aux_dim=16,
        heads=1,
        aux_type="diff-mlp",
        symmetric_aux=False,
        **kwargs,
    ):
        reset_parameters_at_call = kwargs.get("reset_parameters_at_call", True)
        kwargs["reset_parameters_at_call"] = False
        super().__init__(
            in_channels,
            out_channels,
            heads=heads,
            **kwargs,
        )

        self.aux_src = torch.nn.Linear(in_channels, aux_dim)
        self.aux_dst = torch.nn.Linear(in_channels, aux_dim)
        self.aux_out = torch.nn.Linear(aux_dim, 1)

        if aux_type == "diff-mlp":
            self.aux_lin = torch.nn.Linear(in_channels, in_channels)
        elif aux_type == "same-mlp":
            self.aux_lin = None
        else:
            raise ValueError(f"Unknown aux type: {aux_type}.")

        self.symmetric_aux = symmetric_aux

        if reset_parameters_at_call:
            self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.aux_src.reset_parameters()
        self.aux_dst.reset_parameters()
        self.aux_out.reset_parameters()
        if self.aux_lin is not None:
            self.aux_lin.reset_parameters()

    def calculate_auxiliary_gate(self, x_src, x_dst, edge_index, edge_weight):
        src, dst = edge_index

        x_src = x_src[src]  # [E, in_channels * H]
        x_dst = x_dst[dst]  # [E, in_channels * H]

        H, C = self.heads, self.in_channels
        x_src = x_src.view(-1, H, C)  # [E, H, in_channels]
        x_dst = x_dst.view(-1, H, C)  # [E, H, in_channels]

        x_src = self.aux_src(x_src)
        x_dst = self.aux_dst(x_dst)

        x = x_src + x_dst
        x = F.relu(x)
        x = self.aux_out(x)  # [E, H, 1]

        if self.symmetric_aux:
            return F.tanh(x)  # [E, H, 1]

        return F.sigmoid(x)  # [E, H, 1]

    def aggregate_structural_prior(self, x_src, edge_index, edge_weight):
        gedi_alpha = self.calculate_structural_prior(edge_index, x_src.dtype)

        H, C = self.heads, self.in_channels
        x_src_temp = x_src.view(-1, H, C)

        out = self.propagate(
            edge_index,
            x=x_src_temp,
            alpha=gedi_alpha,
            x_aux=None,
            sigma_aux=None,
            gating_score=None,
            gedi_alpha=None,
            size=None,
        )

        return out

    def forward(
        self,
        x_src: torch.FloatTensor,
        x_dst: Optional[torch.FloatTensor],
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
    ):
        alpha = self.calculate_alphas(x_src, x_dst, edge_index, edge_weight)
        alpha = torch.unsqueeze(alpha, -1)  # [E, H, 1]

        if x_dst is None:
            x_dst = self.aggregate_structural_prior(x_src, edge_index, edge_weight)

        sigma_aux = self.calculate_auxiliary_gate(
            x_src, x_dst, edge_index, edge_weight
        )  # [E, H, 1]

        gating_score = None
        if self.gate_lin is not None:
            gating_score = self.calculate_gating_score(x_src)

        H, C = self.heads, self.in_channels
        x_src = x_src.view(-1, H, C)

        if self.aux_lin is not None:
            x_src_neg = self.aux_lin(x_src)
        else:
            x_src_neg = -x_src

        gedi_alpha = None
        if not self.symmetric_aux and self.attention_combination == "mul":
            gedi_alpha = self.calculate_structural_prior(edge_index, x_src.dtype)
            gedi_alpha = torch.unsqueeze(gedi_alpha, -1)  # [E, 1, 1]

        out = self.propagate(
            edge_index,
            x=x_src,
            x_aux=x_src_neg,
            alpha=alpha,
            sigma_aux=sigma_aux,
            gating_score=gating_score,
            gedi_alpha=gedi_alpha,
            size=None,
        )

        return out

    def message(self, x_j, alpha, sigma_aux=None, x_aux_j=None, gedi_alpha=None):
        if sigma_aux is None:
            return alpha.unsqueeze(-1) * x_j
        if self.symmetric_aux:
            if self.aux_lin is None:
                out = sigma_aux * x_j
            else:
                out = F.relu(sigma_aux) * x_j + F.relu(-sigma_aux) * x_aux_j
            return alpha * out
        if self.attention_combination == "mul":
            return alpha * x_j + (gedi_alpha - alpha) * sigma_aux * x_j
        return alpha * x_j + (1 - alpha) * sigma_aux * x_aux_j

    def update(self, inputs: Tensor, gating_score: Optional[Tensor] = None):
        if gating_score is not None:
            inputs = inputs * gating_score
        if self.concat:
            inputs = inputs.view(-1, self.heads * self.in_channels)
        else:
            inputs = inputs.mean(dim=1)
        return inputs


def build_message_conv(
    conv_name: str,
    in_channels: int,
    out_channels: int,
    heads: int = 1,
    negative_slope: float = 0.2,
    concat: bool = True,
    mask_self_loops: bool = False,
    dropout: float = 0.0,
    attention_type: str = "gedi",
    attention_version: str = "v2",
    shared_att_weights: bool = True,
    attention_combination: str = "only-src",
    aux_type: str = "diff-mlp",
    symmetric_aux: bool = False,
    **kwargs,
):
    if conv_name == "att":
        return MainPathConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            negative_slope=negative_slope,
            concat=concat,
            mask_self_loops=mask_self_loops,
            dropout=dropout,
            attention_type=attention_type,
            attention_version=attention_version,
            shared_att_weights=shared_att_weights,
            attention_combination=attention_combination,
            **kwargs,
        )
    elif conv_name == "aux":
        return DualPathConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            negative_slope=negative_slope,
            concat=concat,
            mask_self_loops=mask_self_loops,
            dropout=dropout,
            attention_type=attention_type,
            attention_version=attention_version,
            shared_att_weights=shared_att_weights,
            attention_combination=attention_combination,
            aux_type=aux_type,
            symmetric_aux=symmetric_aux,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown convolution name: {conv_name}.")


class HGNNPANDALayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        edge_index=None,
        edge_weight=None,
        heads: int = 1,
        negative_slope: float = 0.2,
        concat: bool = True,
        mask_self_loops: bool = False,
        dropout: float = 0.0,
        attention_type1: str = "gedi",
        attention_type2: str = "gedi",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination1: str = "only-src",
        attention_combination2: str = "only-src",
        conv_name: str = "aux",
        aux_type: str = "diff-mlp",
        symmetric_aux: bool = False,
        **kwargs,
    ):
        super(HGNNPANDALayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_update = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_res = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter("bias", None)

        self.edge_index = edge_index
        self.edge_weight = edge_weight

        # Attention
        self.att_conv1 = build_message_conv(
            conv_name=conv_name,
            in_channels=out_channels,
            out_channels=out_channels,
            heads=heads,
            negative_slope=negative_slope,
            concat=concat,
            mask_self_loops=mask_self_loops,
            dropout=dropout,
            attention_type=attention_type1,
            attention_version=attention_version,
            shared_att_weights=shared_att_weights,
            attention_combination=attention_combination1,
            aux_type=aux_type,
            symmetric_aux=symmetric_aux,
        )
        self.att_conv2 = build_message_conv(
            conv_name=conv_name,
            in_channels=out_channels,
            out_channels=out_channels,
            heads=heads,
            negative_slope=negative_slope,
            concat=concat,
            mask_self_loops=mask_self_loops,
            dropout=dropout,
            attention_type=attention_type2,
            attention_version=attention_version,
            shared_att_weights=shared_att_weights,
            attention_combination=attention_combination2,
            aux_type=aux_type,
            symmetric_aux=symmetric_aux,
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_update.weight)
        glorot(self.lin_res.weight)
        zeros(self.bias)
        self.att_conv1.reset_parameters()
        self.att_conv2.reset_parameters()

    def forward(self, x: torch.FloatTensor, h: Optional[torch.FloatTensor] = None):
        edge_index = self.edge_index
        edge_weight = self.edge_weight

        x_update = self.lin_update(x)
        x_res = self.lin_res(x)

        h = self.att_conv1(
            x_update,
            h,
            edge_index,
            edge_weight,
        )

        edge_index = edge_index.flip(0)
        x = self.att_conv2(
            h,
            x_update,
            edge_index,
            edge_weight,
        )

        x = x + x_res

        if self.bias is not None:
            x = x + self.bias

        return x, h


class HGNNPANDA(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden: int = 2,
        K: int = 1,
        label_dim: int = 2,
        activation: bool = True,
        layer: int = 2,
        dropout: float = 0.5,
        normalization: str = "sym",
        edge_index=None,
        edge_weight=None,
        mask_self_loops: bool = False,
        attention_type1: str = "gedi",
        attention_type2: str = "gedi",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination1: str = "only-src",
        attention_combination2: str = "only-src",
        conv_name: str = "aux",
        aux_type: str = "diff-mlp",
        symmetric_aux: bool = False,
        args=None,
    ):
        super(HGNNPANDA, self).__init__()

        att_dropout = args.att_dropout
        if att_dropout is None:
            att_dropout = dropout

        H = args.n_heads

        edge_weight = None

        chebs = nn.ModuleList()
        chebs.append(
            HGNNPANDALayer(
                in_channels=num_features,
                out_channels=hidden,
                edge_index=edge_index,
                edge_weight=edge_weight,
                mask_self_loops=mask_self_loops,
                attention_type1=attention_type1,
                attention_type2=attention_type2,
                heads=H,
                dropout=att_dropout,
                attention_version=attention_version,
                shared_att_weights=shared_att_weights,
                attention_combination1=attention_combination1,
                attention_combination2=attention_combination2,
                conv_name=conv_name,
                aux_type=aux_type,
                symmetric_aux=symmetric_aux,
            )
        )
        self.normalization = normalization
        self.activation = activation

        for _ in range(1, layer):
            chebs.append(
                HGNNPANDALayer(
                    in_channels=H * hidden,
                    out_channels=hidden,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    mask_self_loops=mask_self_loops,
                    attention_type1=attention_type1,
                    attention_type2=attention_type2,
                    heads=args.n_heads,
                    dropout=att_dropout,
                    attention_version=attention_version,
                    shared_att_weights=shared_att_weights,
                    attention_combination1=attention_combination1,
                    attention_combination2=attention_combination2,
                    conv_name=conv_name,
                    aux_type=aux_type,
                    symmetric_aux=symmetric_aux,
                )
            )

        self.Chebs = chebs
        last_dim = H * hidden
        self.classifier = MLP(
            in_channels=last_dim,
            hidden_channels=args.Classifier_hidden,
            out_channels=label_dim,
            num_layers=args.Classifier_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        """
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * data (graph as input) - Edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        x = data.x
        h = None

        for ii, cheb in enumerate(self.Chebs):
            x_prev = x  # Store previous values
            x, h = cheb(x, h)
            x = F.relu(x)

            # Add skip connection
            if ii != 0:
                x = x + x_prev

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.classifier(x)
        return x
