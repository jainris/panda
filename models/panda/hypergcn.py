import math
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax


class HyperGCNMainPathConv(MessagePassing):
    def __init__(
        self,
        a,
        b,
        reapproximate=True,
        heads=1,
        negative_slope: float = 0.2,
        concat: bool = True,
        dropout: float = 0.0,
        attention_type: str = "gcn",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination: str = "only-src",
        reset_parameters_at_call=True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        kwargs.setdefault("node_dim", 0)
        kwargs.setdefault("flow", "target_to_source")

        super(HyperGCNMainPathConv, self).__init__(**kwargs)
        self.a, self.b = a, b
        self.reapproximate = reapproximate

        self.W = Parameter(torch.FloatTensor(a, b * heads))
        self.bias = Parameter(torch.FloatTensor(b * heads if concat else b))

        in_channels = b
        self.in_channels = in_channels

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

        if reset_parameters_at_call:
            self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

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

    def calculate_alphas(self, x, edge_index, edge_weight):
        alphas = []
        for attention_type in self.attention_types:
            alpha = self.calculate_alpha(
                attention_type,
                x,
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
            x_src = x.view(-1, self.heads, self.in_channels)  # [N, H, in_channels]
            x_dst = x.view(-1, self.heads, self.in_channels)  # [N, H, in_channels]
            x_src = self.att_wt_lin1(x_src)
            x_src = x_src[edge_index[0]]  # [E, H, 8]

            if self.attention_combination == "only-src":
                x = x_src
            elif self.attention_combination == "src+dst":
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
        x,
        edge_index,
        edge_weight,
    ):
        if attention_type == "gcn":
            return edge_weight.view(-1, 1)
        elif attention_type == "att":
            return self.calculate_alpha_att(x, edge_index)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}.")

    def calculate_alpha_att(self, x, edge_index):
        if self.attention_version == "v1":
            return self.calculate_alpha_attv1(x, x, edge_index)
        elif self.attention_version == "v2":
            return self.calculate_alpha_attv2(x, x, edge_index)
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

    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else:
            A = structure

        A = A.to(H.device)
        A = Variable(A)

        edge_index = A._indices()
        edge_weight = A._values()

        alpha = self.calculate_alphas(HW, edge_index, edge_weight)

        HW = HW.view(-1, self.heads, self.in_channels)  # [N, H, in_channels]

        AHW = self.propagate(edge_index, x=HW, alpha=alpha)

        return AHW + b

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

    def update(self, inputs: Tensor):
        if self.concat:
            inputs = inputs.view(-1, self.heads * self.in_channels)
        else:
            inputs = inputs.mean(dim=1)
        return inputs

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.a) + " -> " + str(self.b) + ")"


class HyperGCNDualPathConv(HyperGCNMainPathConv):
    def __init__(
        self,
        a,
        b,
        reapproximate=True,
        heads=1,
        negative_slope: float = 0.2,
        concat: bool = True,
        dropout: float = 0.0,
        attention_type: str = "gcn",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination: str = "only-src",
        aux_dim=16,
        aux_type="diff-mlp",
        symmetric_aux=False,
        **kwargs,
    ):
        reset_parameters_at_call = kwargs.get("reset_parameters_at_call", True)
        super().__init__(
            a,
            b,
            reapproximate=reapproximate,
            heads=heads,
            negative_slope=negative_slope,
            concat=concat,
            dropout=dropout,
            attention_type=attention_type,
            attention_version=attention_version,
            shared_att_weights=shared_att_weights,
            attention_combination=attention_combination,
            reset_parameters_at_call=False,
        )

        self.aux_src = torch.nn.Linear(b, aux_dim)
        self.aux_dst = torch.nn.Linear(b, aux_dim)
        self.aux_out = torch.nn.Linear(aux_dim, 1)

        if aux_type == "diff-mlp":
            self.aux_lin = torch.nn.Linear(b, b)
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

    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else:
            A = structure

        A = A.to(H.device)
        A = Variable(A)

        edge_index = A._indices()
        edge_weight = A._values()

        alpha = self.calculate_alphas(HW, edge_index, edge_weight)

        sigma_aux = self.calculate_auxiliary_gate(
            HW, HW, edge_index, edge_weight
        )  # [E, H, 1]

        HW = HW.view(-1, self.heads, self.in_channels)  # [N, H, in_channels]

        if self.aux_lin is not None:
            x_aux = self.aux_lin(HW)
        else:
            x_aux = -HW

        AHW = self.propagate(
            edge_index,
            x=HW,
            alpha=alpha,
            sigma_aux=sigma_aux,
            x_aux=x_aux,
            edge_weight=edge_weight,
        )
        return AHW + b

    def message(self, x_j, alpha, sigma_aux, x_aux_j, edge_weight):
        if sigma_aux is None:
            return alpha.unsqueeze(-1) * x_j
        if self.symmetric_aux:
            if self.aux_lin is None:
                out = sigma_aux * x_j
            else:
                out = F.relu(sigma_aux) * x_j + F.relu(-sigma_aux) * x_aux_j
            return alpha * out
        if self.attention_combination == "mul":
            return alpha * x_j + (edge_weight - alpha) * sigma_aux * x_j
        return alpha.unsqueeze(-1) * x_j + (1 - alpha.unsqueeze(-1)) * sigma_aux * x_aux_j


def build_message_conv(
    conv_name,
    a,
    b,
    reapproximate=True,
    heads=1,
    negative_slope: float = 0.2,
    concat: bool = True,
    dropout: float = 0.0,
    attention_type: str = "gcn",
    attention_version: str = "v2",
    shared_att_weights: bool = True,
    attention_combination: str = "only-src",
    aux_dim=16,
    aux_type="diff-mlp",
    symmetric_aux=False,
    **kwargs,
):
    if conv_name == "att":
        return HyperGCNMainPathConv(
            a,
            b,
            reapproximate=reapproximate,
            heads=heads,
            negative_slope=negative_slope,
            concat=concat,
            dropout=dropout,
            attention_type=attention_type,
            attention_version=attention_version,
            shared_att_weights=shared_att_weights,
            attention_combination=attention_combination,
            reset_parameters_at_call=kwargs.get("reset_parameters_at_call", True),
        )
    elif conv_name == "aux":
        return HyperGCNDualPathConv(
            a,
            b,
            reapproximate=reapproximate,
            heads=heads,
            negative_slope=negative_slope,
            concat=concat,
            dropout=dropout,
            attention_type=attention_type,
            attention_version=attention_version,
            shared_att_weights=shared_att_weights,
            attention_combination=attention_combination,
            aux_dim=aux_dim,
            aux_type=aux_type,
            symmetric_aux=symmetric_aux,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown conv name: {conv_name}.")


class HyperGCNPANDA(nn.Module):
    def __init__(
        self,
        num_features,
        num_classses,
        args,
        heads=1,
        negative_slope: float = 0.2,
        concat: bool = True,
        dropout: float = 0.0,
        attention_type: str = "gcn",
        attention_version: str = "v2",
        shared_att_weights: bool = True,
        attention_combination: str = "only-src",
        conv_name="aux",
    ):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCNPANDA, self).__init__()
        d, l, c = num_features, args.nconv, num_classses
        h = [d]
        in_mult = [1]
        for i in range(l - 1):
            power = l - i + 2
            if args.dname == "citeseer":
                power = l - i + 4
            h.append(2**power)
            in_mult.append(heads if concat else 1)
        h.append(c)

        self.layers = nn.ModuleList(
            [
                build_message_conv(
                    conv_name,
                    h[i] * in_mult[i],
                    h[i + 1],
                    reapproximate=(not args.HyperGCN_fast),
                    heads=heads,
                    negative_slope=negative_slope,
                    concat=concat,
                    dropout=dropout,
                    attention_type=attention_type,
                    attention_version=attention_version,
                    shared_att_weights=shared_att_weights,
                    attention_combination=attention_combination,
                )
                for i in range(l)
            ]
        )
        self.do, self.l = dropout, args.nconv
        self.structure, self.m = None, args.HyperGCN_mediators
        self.fast = args.HyperGCN_fast

        self.lin_out = torch.nn.Linear(h[-1] * heads if concat else h[-1], c)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, data):
        """
        an l-layer GCN
        """
        if self.structure is None:
            print("Precomputing ...")
            data0 = data.clone().cpu()
            He_dict = get_HyperGCN_He_dict(data0)

            if self.fast:
                self.structure = Laplacian(
                    V=data0.x.shape[0], E=He_dict, X=data0.x, m=self.m
                )
            else:
                self.structure = He_dict

        do, l, m = self.do, self.l, self.m
        H = data.x

        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            if i < l - 1:
                V = H
                H = F.dropout(H, do, training=self.training)

        H = F.relu(H)
        H = self.lin_out(H)

        return H


# functions for processing/checkning the edge_index
def get_HyperGCN_He_dict(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index.cpu())
    """
    For each he, clique-expansion. Note that we allow the weighted edge.
    Note that if node pair (vi,vj) is contained in both he1, he2, we will have (vi,vj) twice in edge_index. (weighted version CE)
    We default no self loops so far.
    """
    # edge_index[1, :] = edge_index[1, :]-edge_index[1, :].min()
    He_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = list(edge_index[0, :][edge_index[1, :] == he])
        He_dict[he.item()] = nodes_in_he

    return He_dict


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns:
    updated data with 'graph' as a key and its value the approximated hypergraph
    """

    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for k in E.keys():
        hyperedge = list(E[k])

        p = np.dot(X[hyperedge], rv)  # projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2 * len(hyperedge) - 3  # normalisation constant
        if m:

            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])

            if (Se, Ie) not in weights:
                weights[(Se, Ie)] = 0
            weights[(Se, Ie)] += float(1 / c)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / c)

            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend(
                        [[Se, mediator], [Ie, mediator], [mediator, Se], [mediator, Ie]]
                    )
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se, Ie], [Ie, Se]])
            e = len(hyperedge)

            if (Se, Ie) not in weights:
                weights[(Se, Ie)] = 0
            weights[(Se, Ie)] += float(1 / e)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / e)

    return adjacency(edges, weights, V)


def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """

    if (Se, mediator) not in weights:
        weights[(Se, mediator)] = 0
    weights[(Se, mediator)] += float(1 / c)

    if (Ie, mediator) not in weights:
        weights[(Ie, mediator)] = 0
    weights[(Ie, mediator)] += float(1 / c)

    if (mediator, Se) not in weights:
        weights[(mediator, Se)] = 0
    weights[(mediator, Se)] += float(1 / c)

    if (mediator, Ie) not in weights:
        weights[(mediator, Ie)] = 0
    weights[(mediator, Ie)] += float(1 / c)

    return weights


def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """

    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]
    organised = []

    for e in edges:
        i, j = e[0], e[1]
        w = weights[(i, j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32
    )
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A


def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.0
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)


def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """

    M = M.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.0
    di = np.nan_to_num(di)
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)
