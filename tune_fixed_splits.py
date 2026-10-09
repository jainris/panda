import optuna
import torch
import numpy as np
import random
from train_fixed_splits import train_model as training_func, parse_args

from common_tuning import build_tuning_objective


def main():
    args = parse_args()
    study = optuna.create_study(direction="maximize")
    objective = build_tuning_objective(training_func, args)
    study.optimize(objective, n_trials=args.n_trials)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")


if __name__ == "__main__":
    main()
