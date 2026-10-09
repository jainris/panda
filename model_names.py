"""Method names matching the paper's backbones and PANDA ablations."""

PANDA_BACKBONES = ("GeDi", "HGNN", "HyperGCN", "PhenomNN", "GCN")
PANDA_METHODS = frozenset(
    f"{backbone}PANDA{variant}"
    for backbone in PANDA_BACKBONES
    for variant in ("", "WithoutAux", "WithoutPriorAux")
)


def is_graph_method(method):
    return method in {"GCN", "GAT"} or (
        method in PANDA_METHODS and method.startswith("GCNPANDA")
    )


def prepare_model_args(args):
    """Configure the paper's attention-only ablation and graph baselines."""
    if args.method in PANDA_METHODS and args.method.endswith("WithoutPriorAux"):
        # Hyperedges start without features. As described in Appendix B.3,
        # initialize their representations with vanilla aggregation, then apply
        # learned attention without blending a structural-prior coefficient.
        if args.method.startswith(("GeDi", "HGNN")):
            args.attention_type1 = "gedi-att"
        elif args.method.startswith("PhenomNN"):
            args.attention_type1 = "phenom-att"
        else:
            args.attention_type1 = "att"
        args.attention_type2 = "att"
    elif args.method in {"GCN", "GAT"}:
        args.attention_type1 = "gcn" if args.method == "GCN" else "att"
    return args
