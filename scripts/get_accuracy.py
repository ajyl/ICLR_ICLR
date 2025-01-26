import os
import numpy as np
import torch
import argparse
import utils


@torch.no_grad()
def get_probs(llama, token_ids, word_token_ids):
    """
    return [batch, seq, n_classes]
    """
    logits = llama(token_ids.to("cuda")).logits
    probs = logits.softmax(dim=-1)[:, :, word_token_ids]
    return probs


def main(config, model=None):
    exp_dir = config["exp_dir"]
    method = config["dgp_params"]["sampling_kwargs"]["method"]


    # load model
    if model is None:
        model = utils.load_model(config)
        tokenizer = model.tokenizer
    else:
        assert model.config._name_or_path == config["model_params"]["model_name"]
        tokenizer = model.tokenizer

    # data: {
    #   "token_ids": [batch, seq],
    #   "word_idxs": [batch, seq],
    #   "prompt": list[str] (length: batch)
    #   "word_mask": [batch, graph_size, seq],
    # }
    data = utils.load_prompt_data(exp_dir)

    token_ids = data["token_ids"]
    word_masks = data["word_masks"]
    # [batch, seq, graph_size]
    _possibilities = data["possibilities"]
    word_tokens = data["word_tokens"]
    n_words = data["n_words"]

    probs = get_probs(model, token_ids, word_tokens)
    probs = probs.cpu()
    next_token_probs = probs[:, 1:, :]
    possibilities = _possibilities[:, :-1, :]

    if method == "random":
        next_token_probs = probs[:, data["eval_positions"][0], :]
        possibilities = _possibilities
    #    print(_possibilities.shape)
    #    print(data["eval_positions"][0])
    #    print(next_token_probs.shape)
    #    possibilities = _possibilities[:, data["eval_positions"][0], :]

    valid_probs = (next_token_probs[:, :, : n_words] * possibilities).sum(-1)
    accs = valid_probs.mean(dim=0)
    return accs



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
        "--word_seed", type=int, default=42, help="Seed for word sampling"
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    exp_dir = args.exp_dir
    model_name = args.model_name
    batch_size = args.batch_size
    n_examples = args.n_examples

    full_model_name = {
        "llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
        "llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B",
        "llama-3.1-405B": "meta-llama/Meta-Llama-3.1-405B",
        "llama-3.2-1B": "meta-llama/Llama-3.2-1B",
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
    }[model_name]

    config = {
        "exp_dir": exp_dir,
        "model_params": {
            "model_name": full_model_name,
            "rand_init": False,
            "nnsight": False,
            "remote": False,
        },
        "dgp": {
            "sample_args": {
                "batch_size": batch_size,
                "context_length": n_examples,
            },
        },
    }
    main(config)
