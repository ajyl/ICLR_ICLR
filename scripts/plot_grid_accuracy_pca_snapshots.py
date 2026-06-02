import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils
from scripts import get_activations


DEFAULT_TIMESTEPS = [10, 100, 400]

MODEL_NAME_BY_SIZE = {
    "1B": "meta-llama/Llama-3.2-1B",
    "8B": "meta-llama/Meta-Llama-3.1-8B",
    "70B": "meta-llama/Meta-Llama-3.1-70B",
}

N_LAYERS_BY_SIZE = {
    "1B": 16,
    "8B": 32,
    "70B": 80,
}


def get_default_layers_to_save(model_size, required_layer):
    if model_size == "1B":
        layers_to_save = list(range(0, 16, 1))
    elif model_size == "8B":
        layers_to_save = list(range(0, 32, 2))
    elif model_size == "70B":
        layers_to_save = list(range(0, 80, 5))
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

    n_layers = N_LAYERS_BY_SIZE[model_size]
    if required_layer < 0 or required_layer >= n_layers:
        raise ValueError(
            f"Layer {required_layer} is outside the valid range for {model_size}: "
            f"0-{n_layers - 1}"
        )
    if required_layer not in layers_to_save:
        layers_to_save = sorted(set(layers_to_save + [required_layer]))
    return layers_to_save


def normalize_separator(separator):
    return None if separator in (None, "", "None", "none") else separator


def get_default_exp_dir(args):
    batched = "single" if args.single_context else "batch"
    separator = normalize_separator(args.separator)
    separator_name = "sep" if separator is not None else "nosep"
    base_dir = (
        f"./results/{args.model_family}_{args.model_size}_{args.graph}_"
        f"{args.graph_size}_{args.method}_{batched}_{separator_name}_{args.wordsel}"
    )
    return os.path.join(base_dir, f"seed_{args.seed}")


def load_words(args):
    tokens = np.loadtxt(args.tokens_path, dtype=str)
    tokens = [" " + word for word in tokens]
    if args.wordsel == "16_tokens":
        return tokens[: args.graph_size]
    return tokens


def build_grid_config(args, exp_dir):
    config = copy.deepcopy(get_activations.get_default_config())
    separator = normalize_separator(args.separator)

    n_examples = max(args.n_examples, max(args.timesteps))
    if args.method == "random":
        n_examples = int(n_examples / (3 if separator is not None else 2))

    if args.single_context:
        batch_size = 1
        uniform_init = False
    else:
        batch_size = args.batch_size
        uniform_init = not args.no_uniform_init

    if uniform_init and batch_size != args.graph_size:
        raise ValueError(
            "Uniform initialization requires --batch_size to equal --graph_size."
        )

    config["overwrite"] = args.overwrite
    config["exp_dir"] = exp_dir
    config["model_params"]["model_name"] = MODEL_NAME_BY_SIZE[args.model_size]
    config["model_params"]["nnsight"] = args.nnsight
    config["model_params"]["remote"] = args.remote
    config["record_params"]["nnsight"] = args.nnsight
    config["record_params"]["remote"] = args.remote
    config["record_params"]["get_logits"] = True
    config["record_params"]["layers_to_save"] = get_default_layers_to_save(
        args.model_size, args.layer
    )
    config["dgp_params"]["graph_type"] = args.graph
    config["dgp_params"]["graph_size"] = args.graph_size
    config["dgp_params"]["sampling_kwargs"]["batch_size"] = batch_size
    config["dgp_params"]["sampling_kwargs"]["method"] = args.method
    config["dgp_params"]["sampling_kwargs"]["n_examples"] = n_examples
    config["dgp_params"]["sampling_kwargs"]["uniform_init"] = uniform_init
    config["dgp_params"]["separator"] = separator
    config["dgp_params"]["sampling_seed"] = args.seed
    config["dgp_params"]["word_seed"] = args.seed
    config["dgp_params"]["words"] = load_words(args)
    return config


def maybe_generate_experiment(args, exp_dir):
    required_files = ["config.yaml", "data.pt", "activations.pt", "rule_accs.pt"]
    missing = [
        file_name
        for file_name in required_files
        if not os.path.exists(os.path.join(exp_dir, file_name))
    ]

    if not missing and (not args.generate or not args.overwrite):
        return

    if missing and not args.generate:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required files in {exp_dir}: {missing_str}. "
            "Pass --generate to run the experiment first."
        )

    config = build_grid_config(args, exp_dir)
    model = utils.load_model(config)
    dgp = utils.get_dgp(config, model.tokenizer)
    config["record_params"]["relevant_tokens"] = dgp.word_tokens.tolist()
    utils.run(config, model)


def torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_plot_inputs(exp_dir):
    config = utils.load_config(os.path.join(exp_dir, "config.yaml"))
    data = torch_load(os.path.join(exp_dir, "data.pt"))
    activations = torch_load(os.path.join(exp_dir, "activations.pt")).to(
        dtype=torch.float32
    )
    rule_accs = torch_load(os.path.join(exp_dir, "rule_accs.pt")).to(
        dtype=torch.float32
    )

    dgp_path = os.path.join(exp_dir, "dgp.pt")
    if os.path.exists(dgp_path):
        dgp = torch_load(dgp_path)
    else:
        dgp = utils.get_dgp(config, tokenizer=None)

    if hasattr(dgp, "adjacency_matrix"):
        adjacency = dgp.adjacency_matrix
    else:
        adjacency = dgp.transition_matrix > 0

    return config, data, activations, rule_accs, adjacency


def get_accuracy_curve(config, data, rule_accs):
    if rule_accs.ndim == 2:
        accuracy = rule_accs.mean(dim=0)
    elif rule_accs.ndim == 1:
        accuracy = rule_accs
    else:
        raise ValueError(f"Expected rule_accs to be 1D or 2D, got {rule_accs.shape}")

    sampling_kwargs = config.get("dgp_params", {}).get("sampling_kwargs", {})
    method = sampling_kwargs.get("method", "traverse")
    if method == "random" and "word1_positions" in data:
        xs = data["word1_positions"]
    elif "word_positions" in data:
        xs = data["word_positions"]
    elif "eval_positions" in data:
        xs = data["eval_positions"]
    else:
        xs = torch.arange(len(accuracy))

    if torch.is_tensor(xs) and xs.ndim > 1:
        xs = xs[0]
    xs = torch.as_tensor(xs, dtype=torch.float32)
    if len(xs) != len(accuracy):
        if len(xs) > len(accuracy):
            xs = xs[: len(accuracy)]
        else:
            xs = torch.arange(len(accuracy), dtype=torch.float32)

    return xs.numpy(), accuracy.detach().cpu().numpy()


def get_layer_index(config, layer):
    layers_to_save = [
        int(layer_num) for layer_num in config["record_params"]["layers_to_save"]
    ]
    if layer not in layers_to_save:
        raise ValueError(
            f"Layer {layer} was not saved in this experiment. "
            f"Saved layers: {layers_to_save}"
        )
    return layers_to_save.index(layer)


def get_class_mean_snapshot(activations, data, timestep, layer_idx, window_size):
    seq_len = activations.shape[-2]
    if timestep < 0 or timestep >= seq_len:
        raise ValueError(
            f"Timestep {timestep} is outside the saved activation sequence length "
            f"0-{seq_len - 1}."
        )

    class_means = utils.get_class_mean_activations(
        activations,
        seq_loc=timestep,
        word_masks=data["word_masks"],
        window_size=window_size,
    )
    return class_means[layer_idx].detach().cpu().to(dtype=torch.float32)


def project_snapshots(snapshots, component_1, component_2, pca_scope):
    n_components = max(5, component_1 + 1, component_2 + 1)
    n_components = min(
        n_components,
        min(min(acts.shape) for acts in snapshots.values()),
    )
    projected = {}

    if pca_scope == "shared":
        all_acts = torch.cat(list(snapshots.values()), dim=0)
        pca = utils.get_pca(all_acts, n_components=n_components)
        for timestep, acts in snapshots.items():
            projected[timestep] = pca.transform(acts.numpy())
    elif pca_scope == "per_snapshot":
        for timestep, acts in snapshots.items():
            pca = utils.get_pca(acts, n_components=n_components)
            projected[timestep] = pca.transform(acts.numpy())
    else:
        raise ValueError(f"Unsupported pca_scope: {pca_scope}")

    return projected


def get_edges(adjacency):
    adjacency = torch.as_tensor(adjacency).bool().cpu()
    edges = []
    for edge in torch.nonzero(adjacency, as_tuple=False):
        i, j = int(edge[0]), int(edge[1])
        if i < j:
            edges.append((i, j))
    return edges


def get_colors(n_classes):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n_classes)]


def plot_pca_snapshot(ax, points, edges, component_1, component_2, colors):
    for i, j in edges:
        ax.plot(
            [points[i, component_1], points[j, component_1]],
            [points[i, component_2], points[j, component_2]],
            color="0.15",
            lw=0.8,
            alpha=0.75,
            zorder=1,
        )

    ax.scatter(
        points[:, component_1],
        points[:, component_2],
        c=colors,
        s=32,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)


def x_to_axes_fraction(x, xlim, xscale):
    if xscale == "log":
        if x <= 0 or xlim[0] <= 0 or xlim[1] <= 0:
            raise ValueError("Log x-scale requires positive timesteps and x limits.")
        return (np.log10(x) - np.log10(xlim[0])) / (
            np.log10(xlim[1]) - np.log10(xlim[0])
        )
    return (x - xlim[0]) / (xlim[1] - xlim[0])


def get_inset_lefts(timesteps, ax, width, xscale):
    xlim = ax.get_xlim()
    desired = [
        max(0.0, min(1.0, x_to_axes_fraction(timestep, xlim, xscale))) - width / 2
        for timestep in timesteps
    ]
    ordered = sorted(enumerate(desired), key=lambda item: item[1])
    lefts = [0.0] * len(timesteps)
    min_gap = 0.02
    max_left = 1.0 - width

    prev = None
    for idx, desired_left in ordered:
        left = max(0.0, min(max_left, desired_left))
        if prev is not None:
            left = max(left, prev + width + min_gap)
        lefts[idx] = left
        prev = left

    overflow = lefts[ordered[-1][0]] - max_left
    if overflow > 0:
        lefts = [max(0.0, left - overflow) for left in lefts]

    return lefts


def plot_accuracy_with_snapshots(
    output_path,
    config,
    data,
    activations,
    rule_accs,
    adjacency,
    args,
):
    layer_idx = get_layer_index(config, args.layer)
    xs, accuracy = get_accuracy_curve(config, data, rule_accs)

    if min(args.timesteps) < xs.min() or max(args.timesteps) > xs.max():
        raise ValueError(
            f"Requested timesteps {args.timesteps} are outside the accuracy curve "
            f"range {int(xs.min())}-{int(xs.max())}."
        )

    snapshots = {
        timestep: get_class_mean_snapshot(
            activations, data, timestep, layer_idx, args.window_size
        )
        for timestep in args.timesteps
    }
    projected = project_snapshots(
        snapshots, args.component_1, args.component_2, args.pca_scope
    )
    edges = get_edges(adjacency)
    colors = get_colors(next(iter(projected.values())).shape[0])

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(xs, accuracy, color="#2f8f83", lw=2.2, label="ICL task accuracy")
    ax.set_xlabel("Token timestep")
    ax.set_ylabel("ICL task accuracy")
    ax.set_xscale(args.xscale)
    ax.grid(True, axis="both", color="0.88", linewidth=0.8)
    ax.set_title(f"Layer {args.layer} grid PCA snapshots over ICL accuracy")
    ax.legend(loc="lower right", frameon=False)

    y_values = np.interp(args.timesteps, xs, accuracy)
    ax.scatter(
        args.timesteps,
        y_values,
        color="#1f5f58",
        s=28,
        zorder=4,
    )

    for timestep in args.timesteps:
        ax.axvline(timestep, color="0.55", lw=0.8, ls=":", alpha=0.6, zorder=0)

    inset_width = args.inset_width
    inset_height = args.inset_height
    inset_bottom = args.inset_bottom
    lefts = get_inset_lefts(args.timesteps, ax, inset_width, args.xscale)

    for timestep, left, y_value in zip(args.timesteps, lefts, y_values):
        inset = ax.inset_axes([left, inset_bottom, inset_width, inset_height])
        plot_pca_snapshot(
            inset,
            projected[timestep],
            edges,
            args.component_1,
            args.component_2,
            colors,
        )
        inset.set_title(f"t={timestep}", fontsize=9, pad=2)

        ax.annotate(
            "",
            xy=(timestep, y_value),
            xycoords="data",
            xytext=(left + inset_width / 2, inset_bottom),
            textcoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "-",
                "color": "0.3",
                "lw": 0.8,
                "alpha": 0.7,
            },
            zorder=3,
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot ICL task accuracy over token timesteps with overlaid layer PCA "
            "snapshots for the grid task."
        )
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Experiment directory containing config.yaml, data.pt, activations.pt, and rule_accs.pt.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path for the saved figure. Defaults inside exp_dir.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run the grid experiment first if files are missing or --overwrite is set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite exp_dir when used with --generate.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=DEFAULT_TIMESTEPS,
        help="Token timesteps to use for PCA snapshots.",
    )
    parser.add_argument("--layer", type=int, default=15, help="Layer for PCA snapshots.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=200,
        help="Backward window size for class mean activations at each snapshot.",
    )
    parser.add_argument("--component_1", type=int, default=0)
    parser.add_argument("--component_2", type=int, default=1)
    parser.add_argument(
        "--pca_scope",
        choices=["shared", "per_snapshot"],
        default="per_snapshot",
        help=(
            "Use one PCA basis across snapshots or fit each snapshot separately. "
            "The default matches the notebook PCA helper."
        ),
    )
    parser.add_argument(
        "--xscale",
        choices=["linear", "log"],
        default="log",
        help="Scale for the token timestep axis.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--inset_width", type=float, default=0.22)
    parser.add_argument("--inset_height", type=float, default=0.34)
    parser.add_argument("--inset_bottom", type=float, default=0.55)

    parser.add_argument("--model_family", type=str, default="llama")
    parser.add_argument("--model_size", choices=["1B", "8B", "70B"], default="1B")
    parser.add_argument("--graph", choices=["grid"], default="grid")
    parser.add_argument("--graph_size", type=int, default=16)
    parser.add_argument("--method", choices=["traverse", "random"], default="traverse")
    parser.add_argument("--n_examples", type=int, default=max(DEFAULT_TIMESTEPS))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--single_context", action="store_true")
    parser.add_argument("--no_uniform_init", action="store_true")
    parser.add_argument("--separator", type=str, default=None)
    parser.add_argument(
        "--wordsel", choices=["16_tokens", "all_tokens"], default="16_tokens"
    )
    parser.add_argument("--tokens_path", type=str, default="./random_tokens.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nnsight", action="store_true")
    parser.add_argument("--remote", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = args.exp_dir or get_default_exp_dir(args)
    output_path = args.output_path or os.path.join(
        exp_dir, f"accuracy_pca_snapshots_layer_{args.layer}.pdf"
    )

    maybe_generate_experiment(args, exp_dir)
    config, data, activations, rule_accs, adjacency = load_plot_inputs(exp_dir)
    plot_accuracy_with_snapshots(
        output_path, config, data, activations, rule_accs, adjacency, args
    )
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
