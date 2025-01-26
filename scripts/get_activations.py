import sys

sys.path.append("./")

import os
import numpy as np
import torch
import argparse
import shutil
import yaml

import utils


def main(config, model=None):
    exp_dir = config["exp_dir"]
    if os.path.exists(exp_dir):
        if config["overwrite"]:
            shutil.rmtree(exp_dir)
        else:
            raise ValueError(
                f"Directory {exp_dir} already exists, use the flag --overwrite to overwrite it"
            )
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.yaml")
    utils.save_config(config, config_path)

    # load model
    if model is None:
        model = utils.load_model(config)
        tokenizer = model.tokenizer
    else:
        assert model.config._name_or_path == config["model_params"]["model_name"]
        tokenizer = model.tokenizer

    # make dgp
    dgp = utils.get_dgp(config, tokenizer)
    sampling_kwargs = config["dgp_params"]["sampling_kwargs"]
    data = dgp.get_batched_sequences(**sampling_kwargs)

    token_ids = data["token_ids"]
    record_params = config["record_params"]
    activations = utils.get_activations(model, data["token_ids"], record_params)

    data_path = os.path.join(config["exp_dir"], "data.pt")
    dgp_path = os.path.join(config["exp_dir"], "dgp.pt")
    activations_save_path = os.path.join(config["exp_dir"], "activations.pt")
    #
    torch.save(data, data_path)
    torch.save(dgp, dgp_path)
    torch.save(activations, activations_save_path)
    print("Data and activations saved at", data_path, activations_save_path)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_dir", type=str, help="Directory to save the data and activations"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the directory if it exists"
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
    parser.add_argument(
        "--n_layers_save",
        type=int,
        default=-1,
        help="Number of layers to save (will space out equally), -1 to save all",
    )
    parser.add_argument(
        "--rand_init", action="store_true", help="Use random initialization"
    )
    parser.add_argument("--nnsight", action="store_true", help="Use nnsight")
    parser.add_argument("--remote", action="store_true", help="Use remote for nnsight")

    # dgp side
    parser.add_argument("--graph_type", type=str, default="grid", help="Type of graph")
    parser.add_argument("--graph_size", type=int, default=16, help="Size of the graph")
    parser.add_argument("--self_edge", action="store_true", help="Include self edge")
    parser.add_argument(
        "--separator",
        type=str,
        default="",
        help="Separator for the sequence if using random sampling",
    )

    parser.add_argument(
        "--sampling_method",
        type=str,
        default="traverse",
        choices=["traverse", "random"],
        help="Sampling method, traverse or random",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for sampling"
    )
    parser.add_argument(
        "--n_examples", type=int, default=1200, help="Number of examples to sample"
    )
    parser.add_argument(
        "--uniform_init",
        action="store_true",
        help="Use uniform initialization to diversify batches",
    )
    parser.add_argument(
        "--sampling_seed", type=int, default=42, help="Seed for sampling"
    )

    parser.add_argument(
        "--word_file_path",
        type=str,
        default=os.path.join(utils.ROOT, "assets/custom30.txt"),
        help="Path to the word file",
    )
    parser.add_argument(
        "--word_seed", type=int, default=42, help="Seed for word sampling"
    )
    return parser.parse_args()


def get_default_config():
    config = {
        "exp_dir": None,
        "overwrite": False,
        "model_params": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B",
            "rand_init": False,
            "nnsight": False,
            "remote": False,
        },
        "dgp_params": {
            "graph_type": "grid",
            "graph_size": 16,
            "self_edge": False,
            "separator": None,
            "sampling_kwargs": {
                "batch_size": 16,
                "uniform_init": False,
                "method": "traverse",
                "n_examples": 1200,
            },
            "sampling_seed": 42,
            "word_file_path": "./words.txt",
            "word_seed": 42,
            "words": None,  # explicitly set
        },
        "record_params": {
            "layers_to_save": [0],
            "nnsight": False,
            "remote": False,
        },
    }
    return config


if __name__ == "__main__":
    args = parse_args()
    exp_dir = args.exp_dir

    model_name = args.model_name
    n_layers_save = args.n_layers_save
    rand_init = args.rand_init
    nnsight = args.nnsight
    remote = args.remote
    if nnsight and remote:
        assert not rand_init, "nnsight and remote are incompatible with rand_init"

    graph_type = args.graph_type
    graph_size = args.graph_size
    self_edge = args.self_edge
    separator = args.separator
    separator = None if separator == "" else separator

    sampling_method = args.sampling_method
    batch_size = args.batch_size
    n_examples = args.n_examples
    uniform_init = args.uniform_init
    if uniform_init:
        assert sampling_method == "traverse" and batch_size == graph_size
    sampling_seed = args.sampling_seed

    word_file_path = args.word_file_path
    word_seed = args.word_seed
    overwrite = args.overwrite

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
    assert n_layers_save <= n_layers and (n_layers_save > 0 or n_layers_save == -1)
    if n_layers_save == -1:
        n_layers_save = n_layers
    step = (n_layers + 1) // n_layers_save
    layers_to_save = list(range(0, n_layers, step))

    config = {
        "exp_dir": exp_dir,
        "overwrite": overwrite,
        "model_params": {
            "model_name": full_model_name,
            "rand_init": rand_init,
            "nnsight": nnsight,
            "remote": remote,
        },
        "dgp_params": {
            "graph_type": graph_type,
            "graph_size": graph_size,
            "self_edge": self_edge,
            "separator": separator,
            "sampling_kwargs": {
                "batch_size": batch_size,
                "uniform_init": uniform_init,
                "method": sampling_method,
                "n_examples": n_examples,
            },
            "sampling_seed": sampling_seed,
            "word_file_path": word_file_path,
            "word_seed": word_seed,
            "words": None,
        },
        "record_params": {
            "layers_to_save": layers_to_save,
            "nnsight": nnsight,
            "remote": remote,
        },
    }

    main(config)
