from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import scatter, softmax


class MainPathConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout=0.0,
        negative_slope: float = 0.2,
        heads: int = 1,
        attention_type: str = "phenom",
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
        self.concat = True  # Currently only support True
        self.dropout = dropout

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

    def calculate_alphas(
        self,
        x_src,
        x_dst,
        edge_index,
        edge_weight,
        src_degree_inv_sqrt,
        dst_degree_inv_sqrt,
    ):
        alphas = []
        for attention_type in self.attention_types:
            alpha = self.calculate_alpha(
                attention_type,
                x_src,
                x_dst,
                edge_index,
                edge_weight,
                src_degree_inv_sqrt,
                dst_degree_inv_sqrt,
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
        src_degree_inv_sqrt,
        dst_degree_inv_sqrt,
    ):
        if attention_type == "phenom":
            return self.calculate_structural_prior(
                edge_index, src_degree_inv_sqrt, dst_degree_inv_sqrt
            )
        elif attention_type == "att":
            return self.calculate_alpha_att(x_src, x_dst, edge_index)
        elif attention_type == "hgat":
            if x_dst is None:
                return self.calculate_attention_with_prior_embeddings(
                    x_src,
                    edge_index,
                    edge_weight,
                    src_degree_inv_sqrt,
                    dst_degree_inv_sqrt,
                )
            return self.calculate_alpha_att(x_src, x_dst, edge_index)
        elif attention_type == "phenom-att":
            return self.calculate_attention_with_prior_embeddings(
                x_src,
                edge_index,
                edge_weight,
                src_degree_inv_sqrt,
                dst_degree_inv_sqrt,
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}.")

    def calculate_structural_prior(
        self, edge_index, src_degree_inv_sqrt, dst_degree_inv_sqrt
    ):
        alpha = src_degree_inv_sqrt[edge_index[0]] * dst_degree_inv_sqrt[edge_index[1]]

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

    def aggregate_structural_prior(
        self, x_src, edge_index, edge_weight, src_degree_inv_sqrt, dst_degree_inv_sqrt
    ):
        phenom_alpha = self.calculate_structural_prior(
            edge_index, src_degree_inv_sqrt, dst_degree_inv_sqrt
        )

        H, C = self.heads, self.in_channels
        x_src_temp = x_src.view(-1, H, C)

        out = self.propagate(
            edge_index,
            x=x_src_temp,
            alpha=phenom_alpha,
            gating_score=None,
            size=None,
        )

        return out

    def calculate_attention_with_prior_embeddings(
        self, x_src, edge_index, edge_weight, src_degree_inv_sqrt, dst_degree_inv_sqrt
    ):
        x_dst = self.aggregate_structural_prior(
            x_src, edge_index, edge_weight, src_degree_inv_sqrt, dst_degree_inv_sqrt
        )

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
        src_degree_inv_sqrt: torch.FloatTensor,
        dst_degree_inv_sqrt: torch.FloatTensor,
        deg_src: torch.FloatTensor,
    ):
        alpha = self.calculate_alphas(
            x_src,
            x_dst,
            edge_index,
            edge_weight,
            src_degree_inv_sqrt,
            dst_degree_inv_sqrt,
        )

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

        deg_src = deg_src.view(-1, H, C)
        deg_out = self.propagate(
            edge_index,
            x=deg_src,
            alpha=alpha,
            gating_score=gating_score,
            size=None,
        )

        return out, deg_out

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

    def aggregate_structural_prior(
        self, x_src, edge_index, edge_weight, src_degree_inv_sqrt, dst_degree_inv_sqrt
    ):
        phenom_alpha = self.calculate_structural_prior(
            edge_index, src_degree_inv_sqrt, dst_degree_inv_sqrt
        )

        H, C = self.heads, self.in_channels
        x_src_temp = x_src.view(-1, H, C)

        out = self.propagate(
            edge_index,
            x=x_src_temp,
            alpha=phenom_alpha,
            x_aux=None,
            sigma_aux=None,
            gating_score=None,
            phenom_alpha=None,
            size=None,
        )

        return out

    def forward(
        self,
        x_src: torch.FloatTensor,
        x_dst: Optional[torch.FloatTensor],
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
        src_degree_inv_sqrt: torch.FloatTensor,
        dst_degree_inv_sqrt: torch.FloatTensor,
        deg_src: torch.FloatTensor,
    ):
        alpha = self.calculate_alphas(
            x_src,
            x_dst,
            edge_index,
            edge_weight,
            src_degree_inv_sqrt,
            dst_degree_inv_sqrt,
        )
        alpha = torch.unsqueeze(alpha, -1)  # [E, H, 1]

        x_dst = self.aggregate_structural_prior(
            x_src, edge_index, edge_weight, src_degree_inv_sqrt, dst_degree_inv_sqrt
        )

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

        phenom_alpha = None
        if not self.symmetric_aux and self.attention_combination == "mul":
            phenom_alpha = self.calculate_structural_prior(
                edge_index, src_degree_inv_sqrt, dst_degree_inv_sqrt
            )
            phenom_alpha = torch.unsqueeze(phenom_alpha, -1)  # [E, 1, 1]

        out = self.propagate(
            edge_index,
            x=x_src,
            x_aux=x_src_neg,
            alpha=alpha,
            sigma_aux=sigma_aux,
            gating_score=gating_score,
            phenom_alpha=phenom_alpha,
            size=None,
        )

        deg_src = deg_src.view(-1, H, C)
        deg_src_neg = -deg_src if self.aux_lin is None else deg_src
        deg_out = self.propagate(
            edge_index,
            x=deg_src,
            x_aux=deg_src_neg,
            alpha=alpha,
            sigma_aux=sigma_aux,
            gating_score=gating_score,
            phenom_alpha=phenom_alpha,
        )

        return out, deg_out

    def message(
        self,
        x_j,
        alpha,
        sigma_aux=None,
        x_aux_j=None,
        phenom_alpha=None,
    ):
        if sigma_aux is None:
            return alpha.unsqueeze(-1) * x_j
        if self.symmetric_aux:
            if self.aux_lin is None:
                out = sigma_aux * x_j
            else:
                out = F.relu(sigma_aux) * x_j + F.relu(-sigma_aux) * x_aux_j
            return alpha * out
        if self.attention_combination == "mul":
            return alpha * x_j + (phenom_alpha - alpha) * sigma_aux * x_j
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
    dropout: float = 0.0,
    attention_type: str = "phenom",
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


class PhenomNNPANDAConv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        residual=False,
        variant=False,
        dropout=0.0,
        heads: int = 1,
        negative_slope: float = 0.2,
        attention_type1: str = "phenom",
        attention_type2: str = "phenom",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination1: str = "only-src",
        attention_combination2: str = "only-src",
        conv_name: str = "aux",
        aux_type: str = "diff-mlp",
        symmetric_aux: bool = False,
        args=None,
    ):
        super(PhenomNNPANDAConv, self).__init__()
        self.variant = variant
        self.args = args
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features
        self.lam4 = args.lam4
        if self.lam4 != 0:
            print("lam4 is not zero!!!!!!! wrong")
            exit(0)
        self.lam0 = args.lam0
        self.lam1 = args.lam1
        self.alpha = (
            args.alp if args.alp != 0 else 1 / (1 + args.lam4 + args.lam0 + args.lam1)
        )
        self.num_steps = args.prop_step
        self.out_features = out_features
        self.residual = residual
        self.notresidual = args.notresidual
        self.twoHgamma = args.twoHgamma
        self.adj = None
        self.normalize_type = args.normalize_type
        if args.H:
            H = {}
            for t in ["beta", "gamma1", "gamma2"]:

                if args.notresidual:
                    H[t] = torch.rand(in_features, in_features)
                    bound = 4 / in_features
                    nn.init.normal_(H[t], 0, bound)
                    H[t] = nn.Parameter(H[t])
                else:

                    H[t] = torch.rand(in_features, in_features)
                    bound = 1 / in_features
                    nn.init.normal_(H[t], 0, bound)
                    H[t] = H[t] + torch.eye(in_features)

                    H[t] = nn.Parameter(H[t])

            self.H = nn.ParameterDict(H)

        else:
            self.H = None

        self.init_attn = None
        self.node_to_edge = build_message_conv(
            conv_name=conv_name,
            in_channels=out_features,
            out_channels=out_features,
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            attention_type=attention_type1,
            attention_version=attention_version,
            shared_att_weights=shared_att_weights,
            attention_combination=attention_combination1,
            aux_type=aux_type,
            symmetric_aux=symmetric_aux,
        )
        self.edge_to_node = build_message_conv(
            conv_name=conv_name,
            in_channels=out_features,
            out_channels=out_features,
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            attention_type=attention_type2,
            attention_version=attention_version,
            shared_att_weights=shared_att_weights,
            attention_combination=attention_combination2,
            aux_type=aux_type,
            symmetric_aux=symmetric_aux,
        )
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        if self.args.H:
            for t in ["beta", "gamma1", "gamma2"]:
                if self.args.notresidual:
                    bound = 4 / self.in_features
                    nn.init.normal_(self.H[t], 0, bound)
                else:
                    bound = 1 / self.in_features
                    nn.init.normal_(self.H[t], 0, bound)
                    self.H[t] = self.H[t] + torch.eye(
                        self.in_features, device=self.H[t].device
                    )
        self.node_to_edge.reset_parameters()
        self.edge_to_node.reset_parameters()

    def _incidence_mm(self, X, h, row, col, incidence_weight, edge_norm, node_norm):
        num_nodes = X.size(0)
        if row.numel() == 0:
            return X.new_zeros((num_nodes, X.size(1)))

        edge_index_ne = torch.stack([row, col], dim=0)
        edge_index_en = torch.stack([col, row], dim=0)

        deg_src = torch.ones_like(X)

        edge_feat, deg_out = self.node_to_edge(
            x_src=X,
            x_dst=h,
            edge_index=edge_index_ne,
            edge_weight=incidence_weight,
            src_degree_inv_sqrt=node_norm,
            dst_degree_inv_sqrt=edge_norm,
            deg_src=deg_src,
        )
        out, deg_out = self.edge_to_node(
            x_src=edge_feat,
            x_dst=X,
            edge_index=edge_index_en,
            edge_weight=incidence_weight,
            src_degree_inv_sqrt=edge_norm,
            dst_degree_inv_sqrt=node_norm,
            deg_src=deg_out,
        )
        return out, edge_feat, deg_out

    def _normalized_operator(
        self,
        X,
        h,
        row,
        col,
        incidence_weight,
        edge_norm,
        node_degree,
    ):
        deg_inv_sqrt = torch.pow(node_degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

        edge_inv_sqrt = torch.pow(edge_norm, -0.5)
        edge_inv_sqrt[torch.isinf(edge_inv_sqrt)] = 0

        out, edge_feat, deg_out = self._incidence_mm(
            X,
            h,
            row,
            col,
            incidence_weight,
            edge_inv_sqrt,
            deg_inv_sqrt,
        )
        return out, edge_feat, deg_out

    def _build_normalization(
        self,
        row,
        col,
        num_nodes,
        num_edges,
        incidence_weight,
        edge_norm,
        dtype,
        device,
    ):
        if row.numel() == 0:
            node_degree = torch.ones(num_nodes, device=device, dtype=dtype)
            return node_degree

        edge_degree = scatter(
            incidence_weight,
            col,
            dim=0,
            dim_size=num_edges,
            reduce="sum",
        )

        node_degree = scatter(
            incidence_weight * edge_norm[col] * edge_degree[col],
            row,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        node_degree = node_degree + 1.0
        return node_degree

    def forward(self, X, edge_index, edge_weight=None, num_nodes=None):
        if edge_index is None:
            raise ValueError("edge_index is required for PhenomNN propagation.")

        if num_nodes is None:
            num_nodes = X.size(0)

        edge_index = edge_index.long().to(X.device)
        row, col = edge_index
        num_edges = int(col.max().item()) + 1 if col.numel() > 0 else 0

        if edge_weight is None or edge_weight.numel() != row.numel():
            incidence_weight = torch.ones(row.numel(), device=X.device, dtype=X.dtype)
        else:
            incidence_weight = edge_weight.to(device=X.device, dtype=X.dtype).view(-1)

        if num_edges > 0:
            edge_degree = scatter(
                incidence_weight,
                col,
                dim=0,
                dim_size=num_edges,
                reduce="sum",
            )
        else:
            edge_degree = torch.zeros(0, device=X.device, dtype=X.dtype)

        edge_norm_beta = torch.ones_like(edge_degree)
        edge_norm_gamma = torch.pow(edge_degree, -1)
        edge_norm_gamma[torch.isinf(edge_norm_gamma)] = 0

        node_degree_beta = self._build_normalization(
            row,
            col,
            num_nodes,
            num_edges,
            incidence_weight,
            edge_norm_beta,
            X.dtype,
            X.device,
        )
        node_degree_gamma = self._build_normalization(
            row,
            col,
            num_nodes,
            num_edges,
            incidence_weight,
            edge_norm_gamma,
            X.dtype,
            X.device,
        )

        H = self.H
        Y = Y0 = X

        edge_feats_beta = None
        edge_feats_gamma = None

        if H is not None:
            diagD = True
            H_1 = H["beta"]
            H_2 = H["gamma1"]
            H_3 = H["gamma2"]

        for k in range(self.num_steps):
            A_beta_Y, edge_feats_beta, D_beta = self._normalized_operator(
                Y,
                edge_feats_beta,
                row,
                col,
                incidence_weight,
                edge_norm_beta,
                node_degree_beta,
            )
            A_gamma_Y, edge_feats_gamma, D_gamma = self._normalized_operator(
                Y,
                edge_feats_gamma,
                row,
                col,
                incidence_weight,
                edge_norm_gamma,
                node_degree_gamma,
            )
            D_beta = torch.mean(D_beta, dim=-1)
            D_gamma = torch.mean(D_gamma, dim=-1)

            q_tild = self.lam0 * D_beta + self.lam1 * D_gamma + 1.0

            if H is not None:
                L_gamma_Y = D_gamma.unsqueeze(-1) * Y - A_gamma_Y

                if diagD:
                    if self.twoHgamma:
                        Y_hat = (
                            self.lam0
                            * (
                                A_beta_Y @ (H_1 + H_1.T)
                                - (D_beta.unsqueeze(-1) * Y) @ H_1 @ H_1.T
                            )
                            + Y0
                            + self.lam1
                            / 2
                            * (
                                L_gamma_Y
                                + A_gamma_Y @ (H_2 + H_2.T)
                                - (D_gamma.unsqueeze(-1) * Y) @ H_2 @ H_2.T
                                + A_gamma_Y @ (H_3 + H_3.T)
                                - A_gamma_Y @ H_3 @ H_3.T
                            )
                        )
                    else:
                        if self.args.HisI:
                            Y_hat = (
                                self.lam0 * (2 * A_beta_Y - D_beta.unsqueeze(-1) * Y)
                                + Y0
                                + self.lam1 * A_gamma_Y
                            )
                        else:

                            Y_hat = (
                                self.lam0
                                * (
                                    A_beta_Y @ (H_1 + H_1.T)
                                    - (D_beta.unsqueeze(-1) * Y) @ H_1 @ H_1.T
                                )
                                + Y0
                                + self.lam1
                                * (
                                    L_gamma_Y
                                    + A_gamma_Y @ (H_2 + H_2.T)
                                    - (D_gamma.unsqueeze(-1) * Y) @ H_2 @ H_2.T
                                )
                            )
            else:

                Y_hat = self.lam0 * A_beta_Y + Y0 + self.lam1 * A_gamma_Y
            Y = (1 - self.alpha) * Y + self.alpha * (Y_hat / q_tild.unsqueeze(-1))

        return Y


class PhenomNNPANDABlock(nn.Module):
    def __init__(
        self,
        nfeat,
        nlayers,
        nhidden,
        nclass,
        dropout,
        lamda,
        alpha,
        variant,
        heads: int = 1,
        negative_slope: float = 0.2,
        attention_type1: str = "phenom",
        attention_type2: str = "phenom",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination1: str = "only-src",
        attention_combination2: str = "only-src",
        conv_name: str = "aux",
        aux_type: str = "diff-mlp",
        symmetric_aux: bool = False,
        args=None,
    ):
        super(PhenomNNPANDABlock, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(1):
            self.convs.append(
                PhenomNNPANDAConv(
                    heads * nhidden,
                    nhidden,
                    variant=variant,
                    dropout=dropout,
                    heads=heads,
                    negative_slope=negative_slope,
                    attention_type1=attention_type1,
                    attention_type2=attention_type2,
                    attention_version=attention_version,
                    shared_att_weights=shared_att_weights,
                    attention_combination1=attention_combination1,
                    attention_combination2=attention_combination2,
                    conv_name=conv_name,
                    aux_type=aux_type,
                    symmetric_aux=symmetric_aux,
                    args=args,
                )
            )
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, heads * nhidden))
        self.fcs.append(nn.Linear(heads * nhidden, nclass))
        self.in_features = nfeat
        self.out_features = nclass
        self.hiddendim = nhidden
        self.nhiddenlayer = nlayers

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, input, edge_index, edge_weight=None):
        _layers = []
        x = F.dropout(input, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        # layer_inner = input
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(
                    layer_inner,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    num_nodes=input.size(0),
                )
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        self.adj = con.adj
        return layer_inner

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s lamda=%s alpha=%s (%d - [%d:%d] > %d)" % (
            self.__class__.__name__,
            self.lamda,
            self.alpha,
            self.in_features,
            self.hiddendim,
            self.nhiddenlayer,
            self.out_features,
        )


class PhenomNNPANDA(nn.Module):
    """
    The model architecture likes:
    All options are configurable.
    """

    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        nhidlayer,
        dropout,
        baseblock="phenomnn",
        inputlayer=None,
        outputlayer=None,
        nbaselayer=0,
        heads: int = 1,
        negative_slope: float = 0.2,
        attention_type1: str = "phenom",
        attention_type2: str = "phenom",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination1: str = "only-src",
        attention_combination2: str = "only-src",
        conv_name: str = "aux",
        aux_type: str = "diff-mlp",
        symmetric_aux: bool = False,
        args=None,
    ):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "phenomnn", "phenomnn_s".
        :param nbaselayer: the number of layers in one hidden block.
        """
        super(PhenomNNPANDA, self).__init__()
        self.dropout = dropout
        self.baseblock = baseblock.lower()
        self.nbaselayer = nbaselayer
        self.args = args

        if self.baseblock == "phenomnn":
            self.BASEBLOCK = PhenomNNPANDABlock
        else:
            raise NotImplementedError(
                "Current baseblock %s is not supported." % (baseblock)
            )

        self.midlayer = nn.ModuleList()
        for i in range(nhidlayer):

            if baseblock.lower() in ["phenomnn"]:
                gcb = self.BASEBLOCK(
                    nfeat=nfeat,
                    nlayers=nbaselayer,
                    nhidden=nhid,
                    nclass=nclass,
                    dropout=dropout,
                    lamda=args.lamda,
                    alpha=args.alpha,
                    variant=args.variant,
                    heads=heads,
                    negative_slope=negative_slope,
                    attention_type1=attention_type1,
                    attention_type2=attention_type2,
                    attention_version=attention_version,
                    shared_att_weights=shared_att_weights,
                    attention_combination1=attention_combination1,
                    attention_combination2=attention_combination2,
                    conv_name=conv_name,
                    aux_type=aux_type,
                    symmetric_aux=symmetric_aux,
                    args=args,
                )

            else:  # gcn
                NotImplementedError(
                    "Current baseblock %s is not supported." % (baseblock)
                )
            self.midlayer.append(gcb)
        if baseblock.lower() in ["phenomnn"]:
            # self.ingc = nn.Linear(nfeat, nhid)
            # self.outgc = nn.Linear(nhid, nclass)
            # self.fcs = nn.ModuleList([self.ingc, self.outgc])
            self.params1 = self.midlayer[0].params1
            self.params2 = self.midlayer[0].params2
        self.reset_parameters()

    def reset_parameters(self):
        for mid in self.midlayer:
            mid.reset_parameters()

    def forward(self, data):
        fea = data.x
        edge_index = getattr(data, "edge_index", None)
        # edge_weight = getattr(data, "edge_weight", None)
        edge_weight = None

        if edge_index is None and getattr(data, "B", None) is not None:
            B = data.B.coalesce()
            edge_index = B.indices()
            edge_weight = B.values()

        if edge_index is None:
            raise ValueError("PhenomNN expects incidence edge_index in data.")

        if self.baseblock == "phenomnn":
            out = self.midlayer[0](
                input=fea,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            return out
