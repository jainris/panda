import os, sys
import math, time, random
import pickle
import argparse, configargparse
import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from training_args import add_training_args, build_model, prepare_data
from model_names import is_graph_method, prepare_model_args

from tqdm import tqdm

from models import PhenomNN
from scipy.sparse import coo_matrix

import datasets_directed
import datasets_undirected

import utils

"""
Train node-classification models with dataset-provided train/validation/test splits.
"""


def in_out_degree(edge_index, size, weight=None):
    if weight is None:
        A = coo_matrix(
            (np.ones(len(edge_index)), (edge_index[0], edge_index[1])),
            shape=(size, size),
            dtype=np.float32,
        ).tocsr()
    else:
        A = coo_matrix(
            (weight, (edge_index[0], edge_index[1])),
            shape=(size, size),
            dtype=np.float32,
        ).tocsr()

    out_degree = np.sum(np.abs(A), axis=0).T
    in_degree = np.sum(np.abs(A), axis=1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(
    model,
    data,
    train_index,
    val_index,
    test_index,
    evaluator,
    loss_fn=None,
    return_out=False,
):
    model.eval()
    out = model(data)
    out = F.log_softmax(out, dim=1)

    train_acc = evaluator.eval(data.y[train_index], out[train_index])["acc"]
    valid_acc = evaluator.eval(data.y[val_index], out[val_index])["acc"]
    test_acc = evaluator.eval(data.y[test_index], out[test_index])["acc"]
    ret_list = [train_acc, valid_acc, test_acc]

    # Also keep track of losses
    if loss_fn is not None:
        train_loss = loss_fn(out[train_index], data.y[train_index])
        valid_loss = loss_fn(out[val_index], data.y[val_index])
        test_loss = loss_fn(out[test_index], data.y[test_index])
        ret_list += [train_loss, valid_loss, test_loss]

    if return_out:
        ret_list.append(out)

    return ret_list


def main(args):
    train_model(args)

def train_model(args):
    prepare_model_args(args)
    device = torch.device(
        "cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu"
    )

    if args.method not in ["HyperGCN", "HyperSAGE"]:
        transform = torch_geometric.transforms.Compose(
            [datasets_undirected.AddHypergraphSelfLoops()]
        )
    else:
        transform = None

    graph_method = is_graph_method(args.method)
    # Keep original graph edges even when the model uses an undirected graph.
    if not args.directed and not graph_method:
        print("undirected")
        data = datasets_undirected.HypergraphDataset(
            root=args.data_dir,
            name=args.dname,
            path_to_download=args.raw_data_dir,
            feature_noise=args.feature_noise,
            transform=transform,
            second_name=args.second_name,
        ).data
    else:
        print("directed")
        data = datasets_directed.HypergraphDataset(
            root=args.data_dir,
            name=args.dname,
            path_to_download=args.raw_data_dir,
            feature_noise=args.feature_noise,
            transform=transform,
            second_name=args.second_name,
            coalesce_data=not args.uncoalesced_data,
            preserve_graph=graph_method,
        ).data

    data, edge_index, norm_real, norm_imag = prepare_data(args, data)

    # Get splits already defined
    data.y = data.y.long()
    train_mask = data.train_mask.data.numpy().astype("bool_")
    val_mask = data.val_mask.data.numpy().astype("bool_")
    test_mask = data.test_mask.data.numpy().astype("bool_")
    data = data.to(device)

    # define the model

    model = build_model(
        data, args, edge_index=edge_index, norm_real=norm_real, norm_imag=norm_imag
    )
    model = model.to(device)
    num_params = count_parameters(model)
    print("# Params:", num_params)

    logger = utils.Logger(args.runs, args)

    loss_fn = nn.NLLLoss()
    evaluator = utils.NodeClsEvaluator()
    if (not args.directed) and (args.method == "GeDi"):
        args.method = "GeDi_undirected"
    runtime_list = []
    run_name = args.exp_name
    if args.hyperparam_name is not None:
        run_name += f"_{args.hyperparam_name}"

    all_best_val_vals = []

    for run in range(args.runs):
        wandb.init(
            project=args.project_name,
            entity=args.wandb_entity,
            name=run_name + f"_run-{run}",
            config=vars(args),
            tags=[args.wandb_tag] if args.wandb_tag is not None else None,
        )

        start_time = time.time()
        # fare qui un opzione per sistemare le maschere
        train_index = train_mask[:, run]
        val_index = val_mask[:, run]
        test_index = test_mask[:, run]
        model.reset_parameters()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )

        lr_scheduler = None
        if args.lr_scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr_end
            )
        elif args.lr_scheduler == "one-cycle":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                total_steps=args.epochs,
            )
        elif args.lr_scheduler is not None:
            raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}.")

        best_val = float("-inf")
        best_val_val = float("-inf")
        best_val_test = float("-inf")
        best_val_train = float("-inf")
        best_val_epoch = 0

        if args.method == "PhenomNN":
            _, A_beta, D_beta, I = PhenomNN.B2A(data.B, normalize_type="node")
            L_alpha, A_gamma, D_gamma, _ = PhenomNN.B2A(data.B, normalize_type="full")
            data.H = [A_beta, A_gamma]
            data.G = [D_beta, D_gamma, I]

        for epoch in range(args.epochs):
            # Training loop
            model.train()
            optimizer.zero_grad()
            out = model(data)

            out = F.log_softmax(out, dim=1)
            loss = loss_fn(out[train_index], data.y[train_index])
            loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            # Evaluation and logging
            result = evaluate(
                model, data, train_index, val_index, test_index, evaluator, loss_fn
            )
            logger.add_result(run, *result[:3])

            if result[1] > best_val_val:
                best_val_train = result[0]
                best_val_val = result[1]
                best_val_test = result[2]
                best_val_epoch = epoch

            wandb.log(
                {
                    "train_accuracy": result[0],
                    "valid_accuracy": result[1],
                    "test_accuracy": result[2],
                    "train_loss": result[3],
                    "valid_loss": result[4],
                    "test_loss": result[5],
                    "best_val_valid_accuracy": best_val_val,
                    "best_val_test_accuracy": best_val_test,
                    "best_val_train_accuracy": best_val_train,
                    "best_val_epoch": best_val_epoch,
                }
            )

            if (
                epoch % args.display_step == 0
                and args.display_step > 0
                and 100 * result[1] > best_val
            ):
                print(
                    f"Run: {run:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Train Loss: {loss:.4f}, "
                    f"Valid Loss: {result[4]:.4f}, "
                    f"Test Loss: {result[5]:.4f}, "
                    f"Train Acc: {100 * result[0]:.2f}%, "
                    f"Valid Acc: {100 * result[1]:.2f}%, "
                    f"Test Acc: {100 * result[2]:.2f}%"
                )
                best_val = 100 * result[1]
        end_time = time.time()
        runtime_list.append(end_time - start_time)

        all_best_val_vals.append((best_val_test, best_val_val))
        wandb.finish()

    logger.print_statistics(args=args)

    ## Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)
    #
    best_val, best_test = logger.print_statistics()
    res_root = "hyperparameter_tunning"
    if not os.path.isdir(res_root):
        os.makedirs(res_root)
    #
    filename = f"{res_root}/{args.second_name}.csv"
    print(f"Saving results to {filename}")
    with open(filename, "a+") as write_obj:
        cur_line = f"{args.method}"
        cur_line += f",{best_val.mean():.3f} ± {best_val.std():.3f}"
        cur_line += f",{best_test.mean():.3f} ± {best_test.std():.3f}"
        cur_line += f",{num_params}, {avg_time:.2f}s, {std_time:.2f}s"
        cur_line += f",{avg_time//60}min{(avg_time % 60):.2f}s"
        cur_line += f"\n"
        write_obj.write(cur_line)
    #
    all_args_file = f"{res_root}/all_args_{args.second_name}.csv"
    with open(all_args_file, "a+") as f:
        f.write(str(args))
        f.write("\n")
    #
    print("All done! Exit python code")

    return all_best_val_vals

def parse_args():
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--config", is_config_file=True)

    # Dataset specific arguments
    parser.add_argument("--dname", default="walmart-trips-100")
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--raw_data_dir", type=str, required=False)
    parser.add_argument("--train_prop", type=float, default=0.5)
    parser.add_argument("--valid_prop", type=float, default=0.25)
    parser.add_argument(
        "--feature_noise", default="1", type=str, help="std for synthetic feature noise"
    )
    parser.add_argument(
        "--normtype", default="all_one", choices=["all_one", "deg_half_sym"]
    )
    parser.add_argument("--add_self_loop", action="store_false")
    parser.add_argument(
        "--exclude_self",
        action="store_true",
        help="whether the he contain self node or not",
    )
    parser.add_argument("--second_name", type=str, default="")
    parser.add_argument("--directed", type=bool, default=False)

    # Training specific hyperparameters
    parser.add_argument("--epochs", default=500, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--cuda", default=0, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--input_dropout", default=0.2, type=float)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--display_step", type=int, default=50)

    # For saving porpuse
    parser.add_argument("--config_number", type=int, default=0)
    parser.add_argument("--save_result", action="store_true", default=False)

    # Model common hyperparameters
    parser.add_argument(
        "--method", default="EDGNN",
        help="Backbone or PANDA method, e.g. GeDiPANDA, HGNNPANDA, GCNPANDA. "
        "Append WithoutAux or WithoutPriorAux for the paper's ablations.",
    )
    parser.add_argument(
        "--All_num_layers", default=2, type=int, help="number of basic blocks"
    )
    parser.add_argument(
        "--MLP_num_layers", default=2, type=int, help="layer number of mlps"
    )
    parser.add_argument(
        "--MLP_hidden", default=64, type=int, help="hidden dimension of mlps"
    )
    parser.add_argument(
        "--Classifier_num_layers", default=2, type=int
    )  # How many layers of decoder
    parser.add_argument(
        "--Classifier_hidden", default=64, type=int
    )  # Decoder hidden units
    parser.add_argument("--aggregate", default="mean", choices=["sum", "mean"])
    parser.add_argument("--normalization", default="ln", choices=["bn", "ln", "None"])
    parser.add_argument("--activation", default="relu", choices=["Id", "relu", "prelu"])

    # Args for GeDi
    parser.add_argument("--other_complex", action="store_true", default=False)
    parser.add_argument("--nconv", default=2, type=int)

    # Args for EDGNN
    parser.add_argument(
        "--MLP2_num_layers", default=-1, type=int, help="layer number of mlp2"
    )
    parser.add_argument(
        "--MLP3_num_layers", default=-1, type=int, help="layer number of mlp3"
    )
    parser.add_argument(
        "--edconv_type",
        default="EquivSet",
        type=str,
        choices=["EquivSet", "JumpLink", "MeanDeg", "Attn", "TwoSets"],
    )
    parser.add_argument("--restart_alpha", default=0.5, type=float)

    # Args for AllSet
    parser.add_argument("--AllSet_input_norm", default=True)
    parser.add_argument("--AllSet_GPR", action="store_false")  # skip all but last dec
    parser.add_argument("--AllSet_LearnMask", action="store_false")
    parser.add_argument("--AllSet_PMA", action="store_true")
    parser.add_argument("--AllSet_num_heads", default=1, type=int)
    # Args for CEGAT
    parser.add_argument("--output_heads", default=1, type=int)  # Placeholder
    # Args for HyperGCN
    parser.add_argument("--HyperGCN_mediators", action="store_true")
    parser.add_argument("--HyperGCN_fast", action="store_true")
    # Args for HyperSAGE
    parser.add_argument("--HyperSAGE_power", default=1.0, type=float)
    parser.add_argument("--HyperSAGE_num_sample", default=100, type=int)
    # Args for HNHN
    parser.add_argument("--HNHN_alpha", default=-1.5, type=float)
    parser.add_argument("--HNHN_beta", default=-0.5, type=float)
    parser.add_argument("--HNHN_nonlinear_inbetween", default=True, type=bool)
    # Args for HCHA
    parser.add_argument("--HCHA_symdegnorm", action="store_true")
    parser.add_argument("--HCHA_att", default=False, type=bool)

    # Args for UniGNN
    parser.add_argument(
        "--UniGNN_use_norm", action="store_true", help="use norm in the final layer"
    )
    parser.add_argument("--UniGNN_degV", default=0)
    parser.add_argument("--UniGNN_degE", default=0)
    # Args for HyperND
    parser.add_argument("--HyperND_ord", default=1.0, type=float)
    parser.add_argument("--HyperND_tol", default=1e-4, type=float)
    parser.add_argument("--HyperND_steps", default=100, type=int)

    # argument for phenomnn
    parser.add_argument(
        "--LP", action="store_true", default=False, help="Label propagation"
    )
    parser.add_argument("--lam4", type=float, default=0, help="lam4.")
    parser.add_argument("--lam0", type=float, default=10, help="lam0.")
    parser.add_argument("--lam1", type=float, default=10, help="lam1.")
    parser.add_argument(
        "--normalize_type", type=str, default="full", help="normalize type for phenomnn"
    )
    parser.add_argument(
        "--H",
        action="store_true",
        default=False,
        help="whether to use compatibility matrix in phenomnn",
    )
    parser.add_argument(
        "--HisI", action="store_true", default=False, help="if using H and H is I "
    )
    parser.add_argument(
        "--notresidual",
        action="store_true",
        default=False,
        help="whether to use residual in H",
    )
    parser.add_argument(
        "--twoHgamma",
        action="store_true",
        default=False,
        help="whether to use two H for gamma matrix",
    )
    parser.add_argument(
        "--nbaseblocklayer",
        type=int,
        default=1,
        help="The number of layers in each baseblock",
    )  # same as '--layer' of gcnii
    parser.add_argument("--alp", type=float, default=1, help="alp")
    parser.add_argument("--prop_step", type=int, default=16, help="propagating steps")
    parser.add_argument("--lamda", type=float, default=0.5, help="lamda.")  # non suo
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha_l")  # non suo
    parser.add_argument(
        "--variant", action="store_true", default=False, help="GCN* model."
    )
    parser.add_argument(
        "--sigma", type=float, default=-1, help="sigma for edge degree matirx."
    )
    parser.add_argument("--nhidden", default=2, type=int)

    parser.add_argument("--project_name", default="hypergraph-attention", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)

    parser = add_training_args(parser)

    args = parser.parse_args()

    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(AllSet_GPR=False)
    parser.set_defaults(AllSet_LearnMask=False)
    parser.set_defaults(AllSet_PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)

    #     Use the line below for .py file
    args = parser.parse_args()
    #     Use the line below for notebook
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
    # main(**vars(args))
