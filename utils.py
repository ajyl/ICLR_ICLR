"""
Utility functions.
"""

import os
from itertools import product
import random
import numpy as np
import torch
import transformers
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformer_lens import HookedTransformer
import networkx
import yaml
import shutil

from record_utils import record_activations


ROOT = os.path.dirname(os.path.realpath(__file__))


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_config(config, config_path):
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def seed_all(seed, deterministic_algos=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if deterministic_algos:
        torch.use_deterministic_algorithms()


def load_model(config):
    model_params = config["model_params"]
    model_name = model_params["model_name"]
    if model_params.get("nnsight", False):
        import nnsight
        from nnsight import LanguageModel

        model = LanguageModel(model_name, device_map="auto")
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto"
        )
        model.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        if model_params["rand_init"]:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    torch.nn.init.normal_(param, mean=0, std=0.02)
    return model


def load_tokenizer(config):
    model_name = config["model_params"]["model_name"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_words(n_words, words_file, shuffle_seed=42):
    all_words = np.loadtxt(words_file, dtype=str)
    all_words = np.array([" " + word for word in all_words])
    n_all = len(all_words)
    np.random.seed(shuffle_seed)
    all_words = all_words[np.random.choice(n_all, size=n_all, replace=False)]
    return all_words[:n_words]


@torch.no_grad()
def record(model, token_ids, record_params):
    """
    z
    """
    layers_to_save = record_params.get("layers_to_save", [])
    get_logits = record_params.get("get_logits", False)
    relevant_tokens = record_params.get("relevant_tokens", None)
    if relevant_tokens is not None:
        relevant_tokens = torch.tensor(relevant_tokens)
    nnsight = record_params.get("nnsight", False)
    piecewise = record_params.get("piecewise", False)

    result = {}
    if nnsight:
        import nnsight

        remote = record_params.get("remote", False)
        with model.trace(token_ids, remote=remote):
            activations = nnsight.list()
            for i_layer in layers_to_save:
                activation = model.model.layers[i_layer].output[0]
                activations.append(activation)
            if get_logits:
                if relevant_tokens is None:
                    logits = model.model.lm_head.output[0].save()
                else:
                    logits = model.model.lm_head.output[0][:, :, relevant_tokens].save()
            activations.save()
        if len(layers_to_save) > 0:
            activations = torch.stack(activations.value, dim=0)
            result["activations"] = activations
        if get_logits:
            result["logits"] = logits.value
    else:
        if model.__str__().startswith("GPT2"):
            layer_key = "transformer.h"
        elif model.__str__().startswith("Gemma2"):
            layer_key = "model.layers"
        else:
            layer_key = "model.layers"

        if len(layers_to_save) > 0:
            if not piecewise:
                with record_activations(
                    model,
                    layer_nums=layers_to_save,
                    layer_types=["decoder_block"],
                ) as recording:
                    model(token_ids.to("cuda"))

                # [layer, batch, seq, d_model]
                activations = torch.stack(
                    [
                        recording[layer_key + f".{layer_num}"][0].cpu()
                        for layer_num in layers_to_save
                    ],
                    dim=0,
                )
            else:
                # token_ids.shape: [batch, seq] ([30, 1202])
                num_splits = 8
                inner_batch_size = int(token_ids.shape[0] / num_splits)
                all_acts = []
                for batch_idx in range(0, token_ids.shape[0], inner_batch_size):
                    batched_token_ids = token_ids[
                        batch_idx : batch_idx + inner_batch_size
                    ]
                    with record_activations(
                        model,
                        layer_nums=layers_to_save,
                        layer_types=["decoder_block"],
                    ) as recording:
                        model(batched_token_ids.to("cuda"))

                    acts = torch.stack(
                        [
                            recording[layer_key + f".{layer_num}"][0].cpu()
                            for layer_num in layers_to_save
                        ],
                        dim=0,
                    )
                    all_acts.append(acts)

                activations = torch.cat(all_acts, dim=1)

            result["activations"] = activations
        if get_logits:
            if relevant_tokens is None:
                logits = model(token_ids.to("cuda"))["logits"]
            else:
                if not piecewise:
                    logits = model(token_ids.to("cuda"))["logits"][
                        :, :, relevant_tokens
                    ]
                else:
                    # token_ids.shape: [batch, seq] ([30, 1202])
                    num_splits = 8
                    inner_batch_size = int(token_ids.shape[0] / num_splits)
                    all_logits = []
                    for batch_idx in range(0, token_ids.shape[0], inner_batch_size):
                        batched_token_ids = token_ids[
                            batch_idx : batch_idx + inner_batch_size
                        ]
                        _logits = model(batched_token_ids.to("cuda"))["logits"][
                            :, :, relevant_tokens
                        ]
                        all_logits.append(_logits.cpu())

                    logits = torch.cat(all_logits, dim=0)

            logits = logits.cpu()
            result["logits"] = logits
    return result


@torch.no_grad()
def get_activations(model, token_ids, record_params):
    """
    z
    """
    layers_to_save = record_params["layers_to_save"]
    nnsight = record_params.get("nnsight", False)
    if nnsight:
        import nnsight

        remote = record_params.get("remote", False)
        with model.trace(token_ids, remote=remote):
            activations = nnsight.list()
            for i_layer in layers_to_save:
                activation = model.model.layers[i_layer].output[0]
                activations.append(activation)
            activations.save()
        return torch.stack(activations.value, dim=0)
    else:
        if model.__str__().startswith("GPT2"):
            layer_key = "transformer.h"
        else:
            layer_key = "model.layers"

        with record_activations(
            model,
            layer_nums=layers_to_save,
            layer_types=["decoder_block"],
        ) as recording:
            model(token_ids.to("cuda"))

        # [layer, batch, seq, d_model]
        activations = torch.stack(
            [
                recording[layer_key + f".{layer_num}"][0].cpu()
                for layer_num in layers_to_save
            ],
            dim=0,
        )
        return activations


@torch.no_grad()
def get_logits(model, token_ids, record_params, relevant_tokens=None):
    nnsight = record_params.get("nnsight", False)
    if nnsight:
        assert relevant_tokens is None
        import nnsight

        remote = record_params.get("remote", False)
        with model.trace(token_ids, remote=remote):
            logits = model.model.lm_head.output[0].save()
            print("NOT TESTED")
        return logits
    else:
        if relevant_tokens is None:
            logits = model(token_ids.to("cuda"))["logits"]
        else:
            logits = model(token_ids.to("cuda"))["logits"][:, :, relevant_tokens]
        return logits


def get_class_mean_activations(activations, seq_loc, word_masks, window_size):
    seq_len = activations.shape[-2]
    n_classes = word_masks.shape[1]

    window_mask = torch.zeros(seq_len, dtype=torch.bool)
    window_mask[max(0, seq_loc - window_size) : seq_loc + 1] = True
    # print(activations.shape,seq_loc,window_mask.shape,word_masks.shape)

    class_mean_activations = []
    for word_idx in range(n_classes):
        valid_acts = activations[:, window_mask * word_masks[:, word_idx], :]
        if valid_acts.shape[1] < 1:
            assert False, "Some classes not present in the window"
        class_mean_activations.append(valid_acts.mean(dim=1))
    class_mean_activations = torch.stack(class_mean_activations, dim=1)
    return class_mean_activations

def print_tokenized(tokenizer, text):
    print([tokenizer.decode(i) for i in tokenizer.encode(text)])


###############################DGP
class InContextGraphRandom:
    MAX_SEED = 2**32 - 1

    def __init__(
        self, tokenizer, words, transition_matrix, sep=None, seed=None, **kwargs
    ):
        self.tokenizer = tokenizer
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None
        self.words = words
        self.n_words = len(words)
        self.transition_matrix = transition_matrix
        assert self.transition_matrix.shape == (self.n_words, self.n_words)
        self.sep = sep
        self.i_sep = self.n_words
        self.i_bos = self.i_sep + 1
        if self.tokenizer is not None:
            self.check_word_tokenization()
        self.preprocess_matrices()

    def preprocess_matrices(self):
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(
            dim=1, keepdim=True
        )
        self.possibility_matrix = self.transition_matrix > 0
        self.adjacency_matrix = self.transition_matrix > 0

    def check_word_tokenization(self):
        assert self.tokenizer is not None
        word_tokens = []
        for word in self.words:
            token_ids = self.tokenizer.encode(word)
            assert len(token_ids) == 2, word
            word_tokens.append(token_ids[1])
        if self.sep is not None:
            token_ids = self.tokenizer.encode(self.sep)
            assert len(token_ids) == 2
            word_tokens.append(token_ids[1])
        else:
            word_tokens.append(self.tokenizer.bos_token_id)
        word_tokens.append(self.tokenizer.bos_token_id)
        self.word_tokens = torch.tensor(word_tokens)

    def get_sequence(self, *args, **kwargs):
        assert self.tokenizer is not None
        assert "method" in kwargs
        method = kwargs["method"]
        if method == "random":
            return self.get_sequence_random(*args, **kwargs)
        elif method == "traverse":
            return self.get_sequence_traverse(*args, **kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")

    def get_sequence_random(self, n_examples=100, **kwargs):
        if "i_word_start" in kwargs:
            i_word1s = torch.cat(
                [
                    torch.tensor([kwargs["i_word_start"]]),
                    torch.randint(
                        0, self.n_words, (n_examples - 1,), generator=self.generator
                    ),
                ],
            )
        else:
            i_word1s = torch.randint(
                0, self.n_words, (n_examples,), generator=self.generator
            )

        i_word2s = torch.multinomial(
            self.transition_matrix[i_word1s], num_samples=1, replacement=True
        )[:, 0]
        possibilities = self.possibility_matrix[i_word1s]

        ######
        prompt = ""
        inds = [self.i_bos]
        for i in range(n_examples):
            prompt += (
                self.words[i_word1s[i]]
                + self.words[i_word2s[i]]
                + (self.sep if self.sep is not None else "")
            )
            inds.append(i_word1s[i].item())
            inds.append(i_word2s[i].item())
            if self.sep is not None:
                inds.append(self.i_sep)
        word1_positions = torch.arange(
            1, len(inds), step=(3 if self.sep is not None else 2)
        )
        word2_positions = torch.arange(
            2, len(inds), step=(3 if self.sep is not None else 2)
        )
        inds = torch.tensor(inds)
        token_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0, :]
        assert token_ids.shape == (len(inds),), f"token_ids: {token_ids}\ninds: {inds}"
        eval_positions = word1_positions.clone()
        word_masks = torch.stack(
            [token_ids == self.word_tokens[i] for i in range(self.n_words)]
        )

        return {
            "token_ids": token_ids,
            "inds": inds,
            "prompt": prompt,
            "i_word1s": i_word1s,
            "i_word2s": i_word2s,
            "word1_positions": word1_positions,
            "word2_positions": word2_positions,
            "eval_positions": eval_positions,
            "possibilities": possibilities,
            "word_masks": word_masks,
            "word_tokens": self.word_tokens,
            "n_words": self.n_words,
        }

    def get_sequence_traverse(self, n_examples=100, **kwargs):
        assert self.sep is None
        # sequence will be n_examples+2 long +1 for bos
        if "i_word_start" in kwargs:
            i_word_start = kwargs["i_word_start"]
        else:
            i_word_start = torch.randint(
                0, self.n_words, (1,), generator=self.generator
            ).item()
        i_words = [i_word_start]
        for _ in range(n_examples):
            i_words.append(
                torch.multinomial(
                    self.transition_matrix[i_words[-1]], num_samples=1, replacement=True
                ).item()
            )
        i_words = torch.tensor(i_words)
        possibilities = self.possibility_matrix[i_words]

        ######
        prompt = ""
        inds = [self.i_bos]
        for i in range(n_examples + 1):
            prompt += self.words[i_words[i]]
            inds.append(i_words[i])
        word_positions = torch.arange(1, len(inds), step=1)
        inds = torch.tensor(inds)
        token_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0, :]
        assert token_ids.shape == (
            len(inds),
        ), f"token_ids: {len(token_ids)}\ninds: {len(inds)}"
        eval_positions = word_positions.clone()
        word_masks = torch.stack(
            [token_ids == self.word_tokens[i] for i in range(self.n_words)], dim=0
        )
        return {
            "token_ids": token_ids,
            "inds": inds,
            "prompt": prompt,
            "i_words": i_words,
            "word_positions": word_positions,
            "word_masks": word_masks,
            "eval_positions": eval_positions,
            "possibilities": possibilities,
            "word_tokens": self.word_tokens,
            "n_words": self.n_words,
        }

    def get_batched_sequences(self, batch_size=16, **kwargs):
        sequences = []
        for i_seq in range(batch_size):
            if kwargs["uniform_init"]:
                assert self.n_words == batch_size  # and kwargs["method"] == "traverse"
                i_word_start = i_seq
                kwargs["i_word_start"] = i_word_start
            sequences.append(self.get_sequence(**kwargs))
        batch = {}
        for key in sequences[0].keys():
            if key in ["word_tokens", "n_words"]:
                batch[key] = sequences[0][key]
            elif key == "prompt":
                batch[key] = [sequence[key] for sequence in sequences]
            else:
                batch[key] = torch.stack(
                    [sequence[key] for sequence in sequences], dim=0
                )
        return batch


def get_pca(vecs, n_components=None):
    vecs = vecs.to(dtype=torch.float32).cpu().numpy()
    if n_components is None:
        n_components = min(*vecs.shape)
    pca = PCA(n_components=n_components)
    pca.fit(vecs)
    return pca


# hex
def get_side_hex(n):
    side = 1
    while get_n_hex(side) < n:
        side += 1
    assert (
        get_n_hex(side) == n
    ), f"side: {side}, n: {n}, get_n_hex(side): {get_n_hex(side)}, get_n_hex(side-1): {get_n_hex(side-1)}"
    return side


def get_n_hex(side):
    g = networkx.hexagonal_lattice_graph(m=side, n=side)
    return len(g.nodes)


def get_adjacency_hex(side):
    g = networkx.hexagonal_lattice_graph(m=side, n=side)
    adjacency = networkx.adjacency_matrix(g).todense()
    return torch.from_numpy(adjacency)


# tri
def get_side_tri(n):
    side = 1
    while get_n_tri(side) < n:
        side += 1
    assert (
        get_n_tri(side) == n
    ), f"side: {side}, n: {n}, get_n_tri(side): {get_n_tri(side)}, get_n_tri(side-1): {get_n_tri(side-1)}"
    return side


def get_n_tri(side):
    g = networkx.triangular_lattice_graph(m=side, n=side)
    return len(g.nodes)


def get_adjacency_tri(side):
    g = networkx.triangular_lattice_graph(m=side, n=side)
    adjacency = networkx.adjacency_matrix(g).todense()
    return torch.from_numpy(adjacency)


def get_adjacency_ring(n):
    adjacency = torch.zeros(n, n)
    for i in range(n):
        adjacency[i, (i + 1) % n] = 1
        adjacency[i, (i - 1) % n] = 1
    return adjacency


def get_side_grid(n):
    return int(np.round(np.sqrt(n), 0))


def get_n_grid(side):
    return side**2


def get_adjacency_grid(side):
    g = networkx.grid_2d_graph(side, side)
    adjacency = networkx.adjacency_matrix(g).todense()
    return torch.from_numpy(adjacency)


def get_dgp(config, tokenizer=None):
    dgp_params = config["dgp_params"]
    graph_type = dgp_params["graph_type"]
    graph_size = dgp_params["graph_size"]
    self_edge = dgp_params.get("self_edge", False)
    separator = dgp_params.get("separator", None)
    word_file_path = dgp_params.get("word_file_path", "./assets/nouns_core.txt")
    word_seed = dgp_params.get("word_seed", 42)
    words = dgp_params.get("words", None)
    sampling_seed = dgp_params.get("sampling_seed", 42)

    assert graph_type in [
        "ring",
        "grid",
        "tri",
        "hex",
    ], f"Invalid graph type: {graph_type}"

    if graph_type == "ring":
        adjacency = get_adjacency_ring(graph_size)
    elif graph_type == "grid":
        side = get_side_grid(graph_size)
        adjacency = get_adjacency_grid(side)
    elif graph_type == "tri":
        side = get_side_tri(graph_size)
        adjacency = get_adjacency_tri(side)
    elif graph_type == "hex":
        side = get_side_hex(graph_size)
        adjacency = get_adjacency_hex(side)
    else:
        raise ValueError(f"BUG")

    if self_edge:
        adjacency[torch.eye(adjacency.shape[0]) == 1] = 1
    else:
        adjacency[torch.eye(adjacency.shape[0]) == 1] = 0
    transition_matrix = adjacency / adjacency.sum(dim=1, keepdim=True)

    if words is None:
        words = load_words(graph_size, word_file_path, word_seed)

    dgp = InContextGraphRandom(
        tokenizer, words, transition_matrix, sep=separator, seed=sampling_seed
    )

    return dgp


def load_prompt_data(exp_dir):
    """
    Load DGP from file.
    """
    return torch.load(os.path.join(exp_dir, "data.pt"))


####ndif
def check_ndif_model(model_name, remote=True):
    import nnsight
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="auto")
    # test nnsight
    with model.trace("Hi!", remote=remote):
        next_token = model.lm_head.output[0, -1, :].argmax(-1).save()
    return True


def run(config, model=None):
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
    save_config(config, config_path)

    # load model
    if model is None:
        model = load_model(config)
        tokenizer = model.tokenizer
    else:
        assert model.config._name_or_path == config["model_params"]["model_name"]
        tokenizer = model.tokenizer

    # make dgp
    dgp = get_dgp(config, tokenizer)
    sampling_kwargs = config["dgp_params"]["sampling_kwargs"]
    data = dgp.get_batched_sequences(**sampling_kwargs)

    token_ids = data["token_ids"]
    record_params = config["record_params"]
    result = record(model, token_ids, record_params)

    data_path = os.path.join(config["exp_dir"], "data.pt")
    dgp_path = os.path.join(config["exp_dir"], "dgp.pt")
    torch.save(data, data_path)
    torch.save(dgp, dgp_path)

    if "activations" in result.keys():
        activations = result["activations"]
        activations_save_path = os.path.join(config["exp_dir"], "activations.pt")
        torch.save(activations, activations_save_path)
    if "logits" in result.keys():
        logits = result["logits"]
        logits_save_path = os.path.join(config["exp_dir"], "logits.pt")
        torch.save(logits, logits_save_path)

        # rule acc and prob mse
        method = config["dgp_params"]["sampling_kwargs"]["method"]
        word_positions = (
            data["word1_positions"] if method == "random" else data["word_positions"]
        )
        possibilities = data["possibilities"]
        gt_probs = possibilities.to(torch.float32) / possibilities.to(
            torch.float32
        ).sum(-1, keepdim=True)
        probs = (
            torch.nn.functional.softmax(logits[:, word_positions[0], :-2], dim=-1)
            .cpu()
            .detach()
        )
        rule_accs = (probs * possibilities).sum(-1)
        prob_mses = ((probs - gt_probs) ** 2).mean(-1)

        rule_accs_save_path = os.path.join(config["exp_dir"], "rule_accs.pt")
        prob_mses_save_path = os.path.join(config["exp_dir"], "prob_mses.pt")
        torch.save(rule_accs, rule_accs_save_path)
        torch.save(prob_mses, prob_mses_save_path)
    print("Saved at", config["exp_dir"])
    return model


def get_edges(adj):
    n = adj.shape[0]
    edges = []

    for i, j in product(range(n), range(n)):
        if adj[i, j] == 1:  # this is fine since the adjacency is symmetric
            edges.append((i, j))

    return torch.LongTensor(edges)


def compute_energy(X: torch.Tensor, edges):
    """X: (n_class, d)"""
    E = ((X[edges[:, 0]] - X[edges[:, 1]]).norm(dim=-1) ** 2).mean()
    return E
