import argparse

from model_names import PANDA_METHODS, is_graph_method, prepare_model_args

from models import (
    HCHA,
    HNHN,
    LEGCN,
    EquivSetGNN,
    GeDiHNN,
    PhenomNN,
    HyperGCN,
    HyperND,
    HyperSAGE,
    SetGNN,
    UniGCNII,
)
from models.panda.hypergcn import HyperGCNPANDA
from models.panda.gedi import GeDiPANDA
from models.panda.gcn import GCNMainPath, GCNPANDA
from models.panda.phenomnn import PhenomNNPANDA
from models.panda.hgnn import HGNNPANDA
from operators import process_gedi_laplacian


def add_tuning_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--n_trials", default=500, type=int)
    parser.add_argument(
        "--tune_attention_version", type=str, action="append", default=[]
    )
    parser.add_argument("--tune_attention_types", type=str, action="append", default=[])
    parser.add_argument("--no_att_tune", action="store_true", default=False)

    return parser


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--lr_scheduler", default=None, type=str)
    parser.add_argument("--lr_end", default=5e-6, type=float)
    parser.add_argument("--to_undirected", action="store_true", default=False)
    parser.add_argument("--n_heads", default=1, type=int)
    parser.add_argument("--hidden", default=None, type=int)

    # PANDA main and auxiliary paths (Sections 3.1 and 3.2).
    parser.add_argument("--mask_self_loops", action="store_true", default=False)
    parser.add_argument("--attention_type1", type=str, default="gedi")
    parser.add_argument("--attention_type2", type=str, default="gedi")
    parser.add_argument("--att_dropout", type=float, default=None)
    parser.add_argument("--gedi2_deg1", type=str, default="dv_inv")
    parser.add_argument("--gedi2_deg2", type=str, default="dv_id_inv")
    parser.add_argument("--attention_version", type=str, default="v2")
    parser.add_argument("--distinct_att_weights", action="store_true", default=False)
    parser.add_argument("--attention_combination1", type=str, default="only-src")
    parser.add_argument("--attention_combination2", type=str, default="only-src")
    parser.add_argument("--aux_type", type=str, default="diff-mlp")
    parser.add_argument("--symmetric_aux", action="store_true", default=False)
    parser.add_argument("--shared_alphas", action="store_true", default=False)

    parser.add_argument(
        "--mean_heads", action="store_true", default=False
    )  # Currently only used for HyperGCNPANDA

    # AttEDHNN
    parser.add_argument("--use_gedi_for_out", action="store_true", default=None)

    parser.add_argument("--uncoalesced_data", action="store_true", default=False)

    parser.add_argument("--exp_name", type=str, default="unnamed")
    parser.add_argument("--hyperparam_name", type=str, default=None)
    parser.add_argument("--wandb_tag", type=str, default=None)

    parser.add_argument("--use_test", action="store_true", default=False)

    parser.add_argument("--set_max_gpu_memory", type=float, default=None)  # in MB

    parser = add_tuning_args(parser)

    return parser


def prepare_data(args, data):
    prepare_model_args(args)
    edge_index, norm_real, norm_imag = None, None, None
    if args.method in ["AllSetTransformer", "AllDeepSets"]:
        data = SetGNN.norm_contruction(data, option=args.normtype)
    elif args.method == "HNHN":
        data = HNHN.generate_norm(data, args)
    elif args.method == "HyperSAGE":
        data = HyperSAGE.generate_hyperedge_dict(data)
    elif args.method == "LEGCN":
        data = LEGCN.line_expansion(data)
    elif args.method == "PhenomNN":
        data = PhenomNN.data_creation(data, args)
    elif args.method == "GeDi":
        edge_index, norm_real, norm_imag = process_gedi_laplacian(
            edge_index=data.edge_index,
            x_real=data.x,
            edge_weight=data.edge_weight,
            normalization="sym",
            num_nodes=data.num_nodes,
            return_lambda_max=False,
        )

    return data, edge_index, norm_real, norm_imag


def build_model(data, args, edge_index, norm_real, norm_imag):
    prepare_model_args(args)
    if args.method == "AllSetTransformer":
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method == "AllDeepSets":
        args.AllSet_PMA = False
        args.aggregate = "add"
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method in ["HGNN", "HCHA"]:
        args.method = (
            "HCHA_att_sym"
            if args.HCHA_att and args.HCHA_symdegnorm
            else "HCHA_att" if args.HCHA_att else args.method
        )
        model = HCHA(data.num_features, data.num_classes, args)
    elif args.method == "HNHN":
        model = HNHN(data.num_features, data.num_classes, args)
    elif args.method == "HyperGCN":
        model = HyperGCN(data.num_features, data.num_classes, args)
    elif args.method == "HyperSAGE":
        model = HyperSAGE(data.num_features, data.num_classes, args)
    elif args.method == "LEGCN":
        model = LEGCN(data.num_features, data.num_classes, args)
    elif args.method == "UniGCNII":
        model = UniGCNII(data.num_features, data.num_classes, args)
    elif args.method == "HyperND":
        model = HyperND(data.num_features, data.num_classes, args)
    elif args.method == "EDGNN":
        model = EquivSetGNN(data.num_features, data.num_classes, args)
    elif args.method == "PhenomNN":
        model = PhenomNN(
            nfeat=data.num_features,
            nhid=args.MLP_hidden,
            nclass=data.num_classes,
            nhidlayer=args.nhidden,
            dropout=args.dropout,
            baseblock="phenomnn",
            nbaselayer=args.nbaseblocklayer,
            args=args,
        )
    elif args.method == "GeDi":
        model = GeDiHNN(
            K=1,
            num_features=data.num_features,
            hidden=args.MLP_hidden,
            label_dim=data.num_classes,  # hidden=256 dropout= 0.3
            i_complex=False,
            layer=args.nconv,
            other_complex=args.other_complex,
            edge_index=edge_index,
            norm_real=norm_real,
            norm_imag=norm_imag,
            dropout=args.dropout,
            gcn=False,
            args=args,
        )
    elif is_graph_method(args.method):
        model_cls = GCNPANDA if args.method == "GCNPANDA" else GCNMainPath
        model = model_cls(
            num_features=data.num_features,
            hidden=args.MLP_hidden,
            label_dim=data.num_classes,
            layer=args.nconv,
            dropout=args.dropout,
            heads=args.n_heads,
            attention_type=args.attention_type1,
            attention_version=args.attention_version,
            shared_att_weights=not args.distinct_att_weights,
            attention_combination=args.attention_combination1,
            att_dropout=args.att_dropout,
            aux_type=args.aux_type,
            symmetric_aux=args.symmetric_aux,
            args=args,
        )
    elif args.method in PANDA_METHODS and args.method.startswith("GeDiPANDA"):
        if args.method == "GeDiPANDA":
            conv_name = "aux"
        else:
            conv_name = "att"
        model = GeDiPANDA(
            num_features=data.num_features,
            hidden=args.MLP_hidden,
            label_dim=data.num_classes,  # hidden=256 dropout= 0.3
            i_complex=False,
            layer=args.nconv,
            other_complex=args.other_complex,
            edge_index=data.edge_index,
            edge_weight=data.edge_weight,
            dropout=args.dropout,
            args=args,
            mask_self_loops=args.mask_self_loops,
            attention_type1=args.attention_type1,
            attention_type2=args.attention_type2,
            attention_version=args.attention_version,
            shared_att_weights=not args.distinct_att_weights,
            attention_combination1=args.attention_combination1,
            attention_combination2=args.attention_combination2,
            conv_name=conv_name,
            aux_type=args.aux_type,
            symmetric_aux=args.symmetric_aux,
            shared_alphas=args.shared_alphas,
        )
    elif args.method in PANDA_METHODS and args.method.startswith("HGNNPANDA"):
        conv_name = "aux" if args.method == "HGNNPANDA" else "att"

        model = HGNNPANDA(
            num_features=data.num_features,
            hidden=args.MLP_hidden,
            label_dim=data.num_classes,  # hidden=256 dropout= 0.3
            layer=args.nconv,
            edge_index=data.edge_index,
            edge_weight=data.edge_weight,
            dropout=args.dropout,
            args=args,
            mask_self_loops=args.mask_self_loops,
            attention_type1=args.attention_type1,
            attention_type2=args.attention_type2,
            attention_version=args.attention_version,
            shared_att_weights=not args.distinct_att_weights,
            attention_combination1=args.attention_combination1,
            attention_combination2=args.attention_combination2,
            conv_name=conv_name,
            aux_type=args.aux_type,
            symmetric_aux=args.symmetric_aux,
        )
    elif args.method in PANDA_METHODS and args.method.startswith("HyperGCNPANDA"):
        conv_name = "aux" if args.method == "HyperGCNPANDA" else "att"

        model = HyperGCNPANDA(
            data.num_features,
            data.num_classes,
            args,
            heads=args.n_heads,
            concat=not args.mean_heads,
            dropout=args.dropout,
            attention_type=args.attention_type1,
            attention_version=args.attention_version,
            shared_att_weights=not args.distinct_att_weights,
            attention_combination=args.attention_combination1,
            conv_name=conv_name,
        )
    elif args.method in PANDA_METHODS and args.method.startswith("PhenomNNPANDA"):
        conv_name = "aux" if args.method == "PhenomNNPANDA" else "att"

        model = PhenomNNPANDA(
            nfeat=data.num_features,
            nhid=args.MLP_hidden,
            nclass=data.num_classes,
            nhidlayer=1,
            dropout=args.dropout,
            baseblock="phenomnn",
            nbaselayer=0,
            conv_name=conv_name,
            heads=args.n_heads,
            attention_type1=args.attention_type1,
            attention_type2=args.attention_type2,
            attention_version=args.attention_version,
            shared_att_weights=not args.distinct_att_weights,
            attention_combination1=args.attention_combination1,
            attention_combination2=args.attention_combination2,
            aux_type=args.aux_type,
            symmetric_aux=args.symmetric_aux,
            args=args,
        )
    else:
        raise ValueError(f"Undefined model name: {args.method}")

    return model
