import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils
from scripts import get_activations


#DEFAULT_TIMESTEPS = [5, 10, 100, 400]
DEFAULT_TIMESTEPS = [5, 30, 105, 402]

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.fontsize": 7,
    "lines.linewidth": 1.25,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

ACCURACY_COLOR = "#0072B2"
MARKER_EDGE_COLOR = "#004B78"
GRAPH_EDGE_COLOR = "0.25"

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

NODE_COLORS = [
    "#d60001",
    "#028700",
    "#b500ff",
    "#05abc6",
    "#98fe02",
    "#ffa632",
    "#ff00ff",
    "#78525e",
    "#01fccf",
    "#aea4ff",
    "#8aa17a",
    "#9a6901",
    "#366962",
    "#d2008c",
    "#0000ff",
    "#f1c232",
]

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
    return NODE_COLORS[:n_classes]
    #return [cmap(i % cmap.N) for i in range(n_classes)]


def plot_pca_snapshot(ax, points, edges, component_1, component_2, colors):
    for i, j in edges:
        ax.plot(
            [points[i, component_1], points[j, component_1]],
            [points[i, component_2], points[j, component_2]],
            color=GRAPH_EDGE_COLOR,
            lw=0.8,
            alpha=0.8,
            zorder=1,
        )

    ax.scatter(
        points[:, component_1],
        points[:, component_2],
        c=colors,
        s=32,
        edgecolors="white",
        linewidths=0.35,
        zorder=2,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("0.995")
    for spine in ax.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.55)


def format_accuracy_tick(value, _pos):
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_accuracy_axis(ax, args):
    ax.set_xlabel("Token Position (Log-Scale)")
    ax.set_ylabel("Task Accuracy")
    ax.set_xscale(args.xscale)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_accuracy_tick))
    ax.grid(True, axis="y", color="0.90", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=3.0, pad=2)
    ax.tick_params(axis="both", which="minor", length=1.8)

    if args.xscale == "log":
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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


def get_inset_left_positions(args, ax, width):
    if args.inset_lefts is None:
        return get_inset_lefts(args.timesteps, ax, width, args.xscale)
    if len(args.inset_lefts) != len(args.timesteps):
        raise ValueError(
            "--inset_lefts must provide one value per timestep. "
            f"Got {len(args.inset_lefts)} left positions for "
            f"{len(args.timesteps)} timesteps."
        )
    return args.inset_lefts


def get_inset_bottoms(args):
    if args.inset_bottoms is None:
        return [args.inset_bottom] * len(args.timesteps)
    if len(args.inset_bottoms) != len(args.timesteps):
        raise ValueError(
            "--inset_bottoms must provide one value per timestep. "
            f"Got {len(args.inset_bottoms)} bottoms for {len(args.timesteps)} timesteps."
        )
    return args.inset_bottoms


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

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        ax.plot(xs, accuracy, color=ACCURACY_COLOR, lw=1.35, label="Accuracy")
        format_accuracy_axis(ax, args)
        if args.title:
            ax.set_title(args.title, pad=4)
        if args.show_legend:
            ax.legend(loc="lower right", frameon=False, handlelength=1.8)

        y_values = np.interp(args.timesteps, xs, accuracy)
        ax.scatter(
            args.timesteps,
            y_values,
            facecolors="white",
            edgecolors=MARKER_EDGE_COLOR,
            linewidths=0.8,
            s=20,
            zorder=4,
        )

        for timestep in args.timesteps:
            ax.axvline(
                timestep,
                color="0.62",
                lw=0.55,
                ls=(0, (1.5, 2.0)),
                alpha=0.75,
                zorder=0,
            )

        inset_width = args.inset_width
        inset_height = args.inset_height
        inset_bottoms = get_inset_bottoms(args)
        lefts = get_inset_left_positions(args, ax, inset_width)

        for idx, (timestep, left, y_value, inset_bottom) in enumerate(zip(
            args.timesteps, lefts, y_values, inset_bottoms
        )):
            width = inset_width
            height = inset_height
            if idx == len(lefts) - 1:
                width *= 1.5
                height *= 1.5
                left = min(max(left - (width - inset_width) / 2, 0.0), 1.0 - width)

            inset = ax.inset_axes([left, inset_bottom, width, height])
            plot_pca_snapshot(
                inset,
                projected[timestep],
                edges,
                args.component_1,
                args.component_2,
                colors,
            )
            #inset.set_title(f"$t={timestep}$", fontsize=7, pad=1.5)

            ax.annotate(
                "",
                xy=(timestep, y_value),
                xycoords="data",
                xytext=(left + width / 2, inset_bottom),
                textcoords=ax.transAxes,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.35",
                    "lw": 1.5,
                    "alpha": 0.8,
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
    parser.add_argument(
        "--fig_width",
        type=float,
        default=5.2,
        help="Figure width in inches. Default is suitable for a full-width paper figure.",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=2.8,
        help="Figure height in inches. Default is suitable for a full-width paper figure.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title. Omitted by default because paper captions usually carry this text.",
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        help="Show the accuracy legend. Hidden by default to reduce paper-figure clutter.",
    )
    parser.add_argument("--inset_width", type=float, default=0.18)
    parser.add_argument("--inset_height", type=float, default=0.30)
    parser.add_argument(
        "--inset_lefts",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional per-snapshot inset left positions in main-axis coordinates, "
            "in the same order as --timesteps. Overrides automatic x placement."
        ),
    )
    parser.add_argument("--inset_bottom", type=float, default=0.55)
    parser.add_argument(
        "--inset_bottoms",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional per-snapshot inset bottom positions, in the same order as "
            "--timesteps. Overrides --inset_bottom."
        ),
    )

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
