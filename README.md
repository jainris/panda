# PANDA: Prior-guided Attentional Dual-path Architecture
[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)
[![Apache 2.0 License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat)](LICENSE)

This is the code repository for the paper ["PANDA: Prior-guided Attentional Dual-path Architecture"](https://arxiv.org/abs/TODO) introducing PANDA, a plugin for two-stage message passing HNNs applicable to both undirected and directed settings, that uses a dual-path architecture to incorporate structural priors, standard attention and non-competitive attention together to enhance the model's ability to capture complex relationships in the data. The paper has been accepted at NeurIPS 2026.

## Installation

Relevant packages can be installed using `uv`.

```bash
# CPU
uv sync --extra cpu

# NVIDIA GPU, using the CUDA 12.8 PyTorch build
uv sync --extra gpu
```

The supplied extras pin PyTorch 2.8 and PyG extension wheels for **Linux x86_64**.
Other platforms require compatible extension wheels and an adjustment to the
sources in [pyproject.toml](pyproject.toml).

## Running experiments

Run commands from the repository root. The scripts differ by split protocol:

| Splits | Training | Hyperparameter tuning |
| --- | --- | --- |
| Random train/validation/test splits | `train_random_splits.py` | `tune_random_splits.py` |
| Dataset-provided splits | `train_fixed_splits.py` | `tune_fixed_splits.py` |

Random splits use `--train_prop 0.5 --valid_prop 0.25` by default. Fixed-split
training requires dataset masks; `--runs` must not exceed the number of supplied
splits.

### Train GeDi + PANDA

GeDi + PANDA is the paper's main integration. Train it on Telegram using the
provided splits:

```bash
uv run --extra cpu train_fixed_splits.py \
  --method GeDiPANDA --directed True \
  --attention_type1 gedi+gedi-att --attention_type2 gedi+att \
  --dname telegram --second_name telegram \
  --raw_data_dir ./dataset/raw \
  --data_dir ./dataset/processed/telegram \
  --runs 10 --epochs 500 --nconv 1 --n_heads 4 \
  --wandb_entity your-id --project_name panda \
  --wandb_tag PANDA \
  --exp_name GeDiPANDA
```

For directed hypergraph loading, pass `--directed True`. For undirected
hypergraph loading, omit the flag. Graph methods select their graph-preserving
loader automatically. Use a separate `--data_dir` for each dataset and
preprocessing configuration.


### Train GCN + PANDA

Use the same Telegram dataset to train the GCN + PANDA graph integration:

```bash
uv run --extra cpu train_fixed_splits.py \
  --method GCNPANDA --directed True \
  --attention_type1 gcn+att \
  --dname telegram --second_name telegram \
  --raw_data_dir ./dataset/raw \
  --data_dir ./dataset/processed/telegram-graph \
  --runs 10 --epochs 500 --nconv 1 --n_heads 4 \
  --wandb_entity your-id --project_name panda \
  --wandb_tag PANDA \
  --exp_name GCNPANDA
```

Graph models use preserved original edges when available, symmetrize the graph,
and add GCN self-loops. Without original graph edges, they project hypergraph
incidences: source-to-target connections for directed hyperedges and clique
expansion for undirected hyperedges.

### Tune hyperparameters

Tune GeDi + PANDA on Telegram. Optuna maximizes mean validation accuracy across
the provided splits:

```bash
uv run --extra cpu tune_fixed_splits.py \
  --method GeDiPANDA --directed True \
  --dname telegram --second_name telegram \
  --raw_data_dir ./dataset/raw \
  --data_dir ./dataset/processed/telegram \
  --tune_attention_types 'gedi+gedi-att++gedi+att' \
  --n_trials 100 \
  --wandb_entity your-id --project_name panda \
  --wandb_tag PANDA \
  --exp_name GeDiPANDA
```

Each `--tune_attention_types` value is `stage1++stage2`; repeat the flag to offer
multiple choices. The default search spaces are backbone-specific. The tuner
also searches learning rate, dropout, weight decay, model size, and supported
attention settings; see [common_tuning.py](common_tuning.py) for the exact ranges.

Replace `your-id` with your W&B username or team name. Training scripts append
result summaries and argument records to `hyperparameter_tunning/`. Use `--seed` to
set the run seed. Full CLI options are available with `--help`.

## Models and ablations

| Backbone | Vanilla `--method` | Full PANDA `--method` / class | PANDA implementation |
| --- | --- | --- | --- |
| GeDi-HNN | `GeDi` | `GeDiPANDA` | [gedi.py](models/panda/gedi.py) |
| HGNN | `HGNN` | `HGNNPANDA` | [hgnn.py](models/panda/hgnn.py) |
| HyperGCN | `HyperGCN` | `HyperGCNPANDA` | [hypergcn.py](models/panda/hypergcn.py) |
| PhenomNN | `PhenomNN` | `PhenomNNPANDA` | [phenomnn.py](models/panda/phenomnn.py) |
| GCN | `GCN` | `GCNPANDA` | [gcn.py](models/panda/gcn.py) |

The graph attention baseline is selected with `--method GAT` and
`--attention_version v1` or `v2`. Additional comparison models are registered in
[training_args.py](training_args.py).

For any PANDA method, append the paper's ablation suffix:

| Suffix | Main path | Auxiliary path |
| --- | --- | --- |
| None | Configured attention/prior combination | Enabled |
| `WithoutAux` | Same configured main path | Removed |
| `WithoutPriorAux` | Learned attention only | Removed |

For example, `GeDiPANDAWithoutAux` removes the auxiliary path, while
`GeDiPANDAWithoutPriorAux` also removes the structural-prior mixture. The latter
still uses vanilla aggregation to initialize featureless hyperedge
representations, as described in the paper.

### Attention configuration

For training, set the attention types explicitly. `--attention_type1` controls
node-to-hyperedge aggregation and `--attention_type2` controls hyperedge-to-node
aggregation. GCN and HyperGCN use only the first type.

| Integration | Main-path prior | Example `--attention_type1` | Example `--attention_type2` |
| --- | --- | --- | --- |
| GeDi / HGNN | `gedi` | `gedi+gedi-att` | `gedi+att` |
| PhenomNN | `phenom` | `phenom+phenom-att` | `phenom+att` |
| GCN / HyperGCN | `gcn` | `gcn+att` | Unused |

`att` denotes learned receiver-normalized attention. The `gedi-att` and
`phenom-att` forms first construct receiver representations using the backbone's
vanilla aggregation. Combining components with `+` learns their mixture;
explicit weights, such as `gcn=0.5+att=0.5`, fix the mixture instead.
`WithoutPriorAux` configures the attention-only types automatically.

Common controls are `--n_heads`, `--attention_version`, and
`--distinct_att_weights`. GCN also accepts `--att_dropout` to set attention
dropout separately from feature dropout. The default auxiliary transformation is
`--aux_type diff-mlp`; `same-mlp` uses the negated main-path representation.
`--symmetric_aux` selects a signed-gate variant. These auxiliary options are
forwarded by the GeDi, HGNN, PhenomNN, and GCN factories; HyperGCN currently uses
its constructor defaults.

## Extending to other backbones

We have implemented PANDA over four hypergraph backbones and one graph backbone.
You can refer to the existing code to extend it to your own backbone.
It is implemented through backbone-specific message-passing modules. To
integrate another backbone, its aggregation must expose:

1. **Message support:** the sender/receiver pairs over which messages are
   aggregated. For an HNN, identify both node-to-hyperedge and hyperedge-to-node
   stages. For a graph backbone, use its node-to-node support.
2. **Structural priors:** the backbone's original per-message coefficients,
   including its degree normalization or reduction weights. Preserve these
   coefficients when constructing the prior-only aggregation.
3. **Sender and receiver representations:** features for attention and local
   gating. If hyperedges start without features, initialize their representations
   with the backbone's vanilla aggregation.

These can then be used to calculate standard attentional scores and take a weighted
sum with the structural prior scores. This sum can then be used alongside
a non-competitive auxiliary attentional channel to form PANDA's node update.

For the standard additive PANDA configuration, one message has the form:

```text
alpha_main = beta * alpha_attention + (1 - beta) * alpha_prior
alpha_aux  = (1 - alpha_main) * sigmoid(auxiliary_score)
message    = alpha_main * transform_main(sender)
           + alpha_aux  * transform_aux(sender)
```

Normalize learned attention over the receiver's incoming messages. Auxiliary
coefficients are not normalized across neighbors; each incidence has its own
gate. The incidence-wise attention/prior blend need not sum to one over a
receiver's neighborhood. The complement-bound interpretation assumes real
main-path coefficients in `[0, 1]`, so preserve the required prior scaling.

Directed or complex-valued backbones must also preserve head/tail roles and
incidence phases. [GeDi's integration](models/panda/gedi.py) separates these
contributions when applying the real-valued attention and auxiliary gates.

## Repository layout

- `models/backbones/`: vanilla models and comparisons.
- `models/panda/`: PANDA integrations and message-passing layers.
- `models/mlp.py`: shared MLP components.
- `operators/`: generalized directed Laplacian construction.
- `datasets_*.py`, `data_loaders_*.py`: dataset preparation and loading.
- `training_args.py`, `model_names.py`: model construction, options, and variants.
- `common_tuning.py`: Optuna search spaces and validation objective.
- `data/original_data/`: included source datasets.

## License

Licensed under Apache 2.0; see [LICENSE](LICENSE). This repository builds on
[GeDi-HNN](https://github.com/Stefa1994/GeDi-HNN) and retains its original license.
