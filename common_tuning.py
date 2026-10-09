import random

import numpy as np
import optuna
import torch

import wandb
from model_names import PANDA_METHODS, is_graph_method, prepare_model_args


def build_tuning_objective(training_func, args):
    prepare_model_args(args)
    if args.set_max_gpu_memory is not None:
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is not available, but set_max_gpu_memory is specified."
            )

        target_mem = args.set_max_gpu_memory * 1024 * 1024  # Convert MB (MiB) to bytes
        device = torch.device(f"cuda:{args.cuda}")

        max_mem = torch.cuda.get_device_properties(device).total_memory
        if target_mem > max_mem:
            raise ValueError(
                f"Requested max GPU memory ({args.set_max_gpu_memory} MB) exceeds the total available memory ({max_mem / (1024 * 1024):.2f} MB)."
            )

        torch.cuda.set_per_process_memory_fraction(target_mem / max_mem, device=device)

    att_versions = (
        args.tune_attention_version if args.tune_attention_version else ["v1", "v2"]
    )
    graph_method = is_graph_method(args.method)
    hypergcn = args.method in PANDA_METHODS and args.method.startswith("HyperGCNPANDA")
    phenomnn = args.method in PANDA_METHODS and args.method.startswith("PhenomNNPANDA")
    if args.method in PANDA_METHODS and args.method.endswith("WithoutPriorAux"):
        default_att_types = [f"{args.attention_type1}++{args.attention_type2}"]
    elif args.method in {"GCN", "GAT"}:
        default_att_types = [f"{args.attention_type1}++{args.attention_type1}"]
    elif graph_method or hypergcn:
        default_att_types = ["gcn+att++gcn+att", "att++att"]
    elif phenomnn:
        default_att_types = ["phenom+phenom-att++phenom+att", "phenom-att++att"]
    else:
        default_att_types = ["gedi+gedi-att++gedi+att", "gedi-att++att"]
    att_types = (
        args.tune_attention_types
        if args.tune_attention_types
        else default_att_types
    )
    directed = not (
        graph_method or hypergcn or phenomnn or args.method.startswith("HGNNPANDA")
    )

    def objective(trial):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Suggest hyperparameters
        args.lr = trial.suggest_float("lr", 1e-3, 2e-2, log=True)
        # args.wd = trial.suggest_float("wd", 0, 5e-3)
        args.dropout = trial.suggest_float("dropout", 0.1, 0.9, step=0.1)

        if not phenomnn:
            args.nconv = trial.suggest_int("nconv", 1, 5, step=1)

        wd_idx = trial.suggest_int("wd_idx", 0, 3)
        args.wd = [0, 5e-5, 5e-4, 5e-3][wd_idx]

        if directed:
            args.other_complex = trial.suggest_categorical(
                "other_complex", [False, True]
            )

        hname = ""
        if not hypergcn:
            mlp_hidden_exp = trial.suggest_int(
                "mlp_hidden_exp", 6, 9
            )  # 2^6=64 to 2^9=512
            args.MLP_hidden = 2**mlp_hidden_exp

            if not phenomnn:
                classifier_hidden_exp = trial.suggest_int(
                    "classifier_hidden_exp", 6, 8
                )  # 2^6=64 to 2^8=256
                args.Classifier_hidden = 2**classifier_hidden_exp
        else:
            # args.HyperGCN_fast = trial.suggest_categorical("hypergcn_fast", [False, True])
            args.HyperGCN_fast = True
            args.HyperGCN_mediators = trial.suggest_categorical(
                "hypergcn_mediators", [False, True]
            )

            hname = f"_HyperGCN_fast-{args.HyperGCN_fast}_HyperGCN_mediators-{args.HyperGCN_mediators}"

        if phenomnn:
            args.lam0 = trial.suggest_float("lam0", 0.0, 100.0, step=0.1)
            args.lam1 = trial.suggest_float("lam1", 0.0, 100.0, step=0.1)
            alp_idx = trial.suggest_int("alpha_idx", 0, 3)
            args.alp = [0, 0.05, 0.1, 1.0][alp_idx]

            H_setting = trial.suggest_int("H_setting", 0, 3)
            if H_setting == 0:
                args.H = False
                args.HisI = False
                args.twoHgamma = False
            elif H_setting == 1:
                args.H = True
                args.HisI = False
                args.twoHgamma = False
            elif H_setting == 2:
                args.H = True
                args.HisI = True
                args.twoHgamma = False
            else:
                args.H = True
                args.HisI = False
                args.twoHgamma = True

            args.prop_step = trial.suggest_categorical("prop_step", [8, 16])

            hname += f"_lam0-{args.lam0}_lam1-{args.lam1}_alp-{args.alp}_H-{args.H}_HisI-{args.HisI}_twoHgamma-{args.twoHgamma}_prop_step-{args.prop_step}"

        attention_types = trial.suggest_categorical(
            "attention_type1--attention_type2", att_types
        )
        args.attention_type1, args.attention_type2 = attention_types.split("++")

        if (
            args.tune_attention_types != "gedi++gedi"
            and args.tune_attention_types != "gcn++gcn"
            and args.tune_attention_types != "phenom++phenom"
            and not args.no_att_tune
        ):
            args.n_heads = trial.suggest_int("n_heads", 1, 8, log=True)
            args.attention_version = trial.suggest_categorical(
                "attention_version", att_versions
            )

            args.distinct_att_weights = trial.suggest_categorical(
                "distinct_att_weights", [False, True]
            )

        args.hyperparam_name = f"MLP_hidden-{args.MLP_hidden}_Classifier_hidden-{args.Classifier_hidden}_nconv-{args.nconv}_other_complex-{args.other_complex}_attention_type1-{args.attention_type1}_attention_type2-{args.attention_type2}_n_heads-{args.n_heads}_attention_version-{args.attention_version}_attention_combination1-{args.attention_combination1}_attention_combination2-{args.attention_combination2}-lr-{args.lr}_wd-{args.wd}_dropout-{args.dropout}_distinct_att_wts_{args.distinct_att_weights}{hname}"

        try:
            all_best_val_vals = training_func(args)
            all_best_val_vals = np.array(all_best_val_vals)

            all_best_val_vals = all_best_val_vals[:, 1]
        except RuntimeError as e:
            print(f"RuntimeError encountered: {e}")
            torch.cuda.empty_cache()
            wandb.finish(exit_code=1)
            raise optuna.TrialPruned()

        torch.cuda.empty_cache()

        return np.mean(all_best_val_vals)

    return objective
