import sys

sys.path.append("./")

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
import yaml

COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "brown",
    "gray",
    "cyan",
    "magenta",
    "lime",
    "olive",
    "maroon",
    "navy",
    "teal",
]
# Lol.
COLORS = COLORS + COLORS + COLORS


def plot_pcas_by_layers(acts, layer_nums, adj, component_1, component_2, output_path):
    n_classes = adj.shape[0]
    projected_acts = []
    for layer_idx, layer_num in enumerate(layer_nums):
        # acts: {timestep: [layer, class, d_model]
        _acts = acts[layer_idx]
        trf = utils.get_pca(_acts, n_components=5)
        projected_acts.append(torch.tensor(trf.transform(_acts.cpu().numpy())))

    # [layer, class, n_components]
    projected_acts = torch.stack(projected_acts, dim=0)

    n_rows = 4
    n_cols = 4
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    links = {}
    for i in range(adj.shape[0]):
        row = adj[i].nonzero()
        for elem in row:
            links[(i, elem.item())] = 1
    links = list(links.keys())

    for layer_idx in range(acts.shape[0]):
        row_idx = layer_idx // n_rows
        col_idx = layer_idx % n_cols
        ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)
        for _class in range(n_classes):
            _color = COLORS[_class]
            plt.scatter(
                projected_acts[layer_idx, _class, component_1],
                projected_acts[layer_idx, _class, component_2],
                c=_color,
                marker="o",
                s=10**2,
            )
            ax.set_title(f"Layer {layer_nums[layer_idx]}")

        for sl_idx, sl in enumerate(links):
            plt.plot(
                [
                    projected_acts[layer_idx, sl[0], component_1],
                    projected_acts[layer_idx, sl[1], component_1],
                ],
                [
                    projected_acts[layer_idx, sl[0], component_2],
                    projected_acts[layer_idx, sl[1], component_2],
                ],
                c="k",
                ls="-",
            )

    plt.savefig(output_path)
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("--seq_loc", type=int, default=1000)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--component_1", type=int, default=0)
    parser.add_argument("--component_2", type=int, default=1)
    return parser.parse_args()


def main(exp_dir, tokenizer, seq_locs, window_size, component_1, component_2,suffix=""):
    config_path = os.path.join(exp_dir, "config.yaml")
    config = utils.load_config(config_path)

    dgp = utils.get_dgp(config, tokenizer)
    adj_matrix = dgp.transition_matrix > 0

    data = torch.load(os.path.join(exp_dir, "data.pt"), weights_only=False)
    activations = torch.load(
        os.path.join(exp_dir, "activations.pt"), weights_only=False
    ).to(dtype=torch.float32)
    layers_to_save = config["record_params"]["layers_to_save"]

    if isinstance(seq_locs, int):
        seq_locs = [seq_locs]

    figs = []
    for seq_loc in seq_locs:
        class_mean_activations = utils.get_class_mean_activations(
            activations,
            seq_loc=seq_loc,
            word_masks=data["word_masks"],
            window_size=window_size,
        )
        plot_save_path = os.path.join(
            exp_dir, "plot_seq_loc_{}_window_size_{}{}.png".format(seq_loc, window_size,suffix)
        )
        figs.append(
            plot_pcas_by_layers(
                acts=class_mean_activations,
                layer_nums=layers_to_save,
                adj=adj_matrix,
                component_1=component_1,
                component_2=component_2,
                output_path=plot_save_path,
            )
        )
    return figs


if __name__ == "__main__":
    args = parse_args()
    exp_dir = args.exp_dir
    seq_loc = args.seq_loc
    window_size = args.window_size
    comp1 = args.component_1
    comp2 = args.component_2

    main(exp_dir, seq_loc, window_size, comp1, comp2)
