import os
from itertools import product
import numpy as np
import torch
import argparse
from tqdm import tqdm
import utils


def get_edges(adj):
    n = adj.shape[0]
    edges = []

    for i, j in product(range(n), range(n)):
        if adj[i, j] == 1:#this is fine since the adjacency is symmetric
            edges.append((i, j))

    return edges


def compute_energy(X: torch.Tensor, edges):
    """X: (n_class, d)"""
    E = ((X[edges[:, 0]] - X[edges[:, 1]]).norm(dim=-1) ** 2).mean().item()
    return E


def get_mean_class_activations(acts, word_masks, window_size):
    """
    acts: [layer, batch, seq, d_model]
    returns: {timestep: [n_classes, batch, d_model]}
    """
    print("Computing mean class activations...")
    seq_len = acts.shape[-2]
    n_classes = word_masks.shape[1]

    class_means = {}
    for timestep in tqdm(range(1, seq_len)):

        window_mask = torch.zeros(seq_len, dtype=torch.bool)
        window_mask[max(0, timestep - window_size) : timestep + 1] = True

        _class_mean_acts = []
        valid = True
        for word_idx in range(n_classes):

            # acts: [layers, batch, seq, d_model]
            # valid: [layer, sub_seq, d_model]
            valid_acts = acts[:, window_mask * word_masks[:, word_idx], :]

            if valid_acts.shape[1] < 1:
                valid = False
                break
            _class_mean_acts.append(valid_acts.mean(dim=1))

        if valid:
            class_means[timestep] = torch.stack(_class_mean_acts, dim=1)
        valid = True

    # {timestep: [layers, n_classes, d_model]}
    return class_means


def main(config):
    """
    z
    """
    exp_dir = config["exp_dir"]

    layer_nums = config["layer_nums"]
    n_examples = config["n_examples"]
    seq_loc = config["seq_loc"]
    window_size = config["window_size"]

    data = utils.load_prompt_data(exp_dir)
    token_ids = data["token_ids"]
    word_masks = data["word_masks"]

    dgp = torch.load(os.path.join(exp_dir, "dgp.pt"))

    activations = torch.load(
        os.path.join(exp_dir, "activations.pt"), weights_only=False
    ).to(dtype=torch.float32)

    print(data["word_masks"].shape)

    class_mean_activations = get_mean_class_activations(
        activations,
        word_masks=data["word_masks"],
        window_size=window_size,
    )

    adj = dgp.adjacency_matrix
    edges = get_edges(adj)
    edges = torch.LongTensor(edges)

    print("Computing energy")
    energy = {}
    for timestep in tqdm(class_mean_activations.keys()):

        energy[timestep] = {}

        for layer_idx, layer in enumerate(layer_nums):
            _mean_class_acts = class_mean_activations[timestep][layer_idx].squeeze()
            energy[timestep][layer] = compute_energy(
                _mean_class_acts, edges
                #_mean_class_acts - _mean_class_acts.mean(dim=0), edges
            )

    return energy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_dir", type=str, help="Directory to load the data and activations"
    )
    # model side
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama-3.1-8B",
        choices=[
            "llama-3.1-8B",
            "llama-3.1-70B",
            "llama-3.1-405B",
            "llama-3.2-1B",
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
        ],
        help="Model name",
    )
    parser.add_argument("--seq_loc", type=int, default=1000)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument(
        "--n_layers_plot",
        type=int,
        default=8,
        help="Number of layers to plot (will space out equally)",
    )
    parser.add_argument(
        "--n_examples", type=int, default=1200, help="Number of examples to sample"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    exp_dir = args.exp_dir
    seq_loc = args.seq_loc
    model_name = args.model_name
    n_layers_plot = args.n_layers_plot
    n_examples = args.n_examples
    window_size = args.window_size

    full_model_name = {
        "llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
        "llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B",
        "llama-3.1-405B": "meta-llama/Meta-Llama-3.1-405B",
        "llama-3.2-1B": "meta-llama/Llama-3.2-1B",
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
    }[model_name]

    n_layers = {
        "llama-3.1-8B": 32,
        "llama-3.1-70B": 32,
        "llama-3.1-405B": 32,
        "llama-3.2-1B": 16,
        "gpt2": 12,
        "gpt2-medium": 24,
        "gpt2-large": 36,
    }[model_name]

    step = (n_layers + 1) // n_layers_plot
    layers_to_plot = list(range(0, n_layers, step))

    config = {
        "exp_dir": exp_dir,
        "model_params": {
            "model_name": full_model_name,
            "rand_init": False,
            "nnsight": False,
            "remote": False,
        },
        "layer_nums": layers_to_plot,
        "n_examples": n_examples,
        "seq_loc": seq_loc,
        "window_size": window_size,
    }
    main(config)
