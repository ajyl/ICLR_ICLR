# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

import utils
from tqdm import tqdm
import copy
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

import importlib

importlib.reload(utils)


# %%

exp_dir = "results/llama_8B_grid_16_traverse_batch_nosep_16_tokens/seed_0"

config = utils.load_config(os.path.join(exp_dir, "config.yaml"))

# %%

model = utils.load_model(config)

# %%

dgp = torch.load(os.path.join(exp_dir, "dgp.pt"))
record_params = config["record_params"]
adj = dgp.transition_matrix > 0

# %%


def get_class_mean_activations(
    activations, seq_loc, word_masks, window_size, n_classes
):
    window_mask = torch.zeros(seq_loc, dtype=torch.bool)
    window_mask[max(0, seq_loc - window_size) : seq_loc + 1] = True

    class_mean_activations = []
    for word_idx in range(n_classes):
        valid_acts = activations[:, window_mask * word_masks[word_idx], :]
        if valid_acts.shape[1] < 1:
            assert False, "Some classes not present in the window"

        class_mean_activations.append(valid_acts.mean(dim=1))
    class_mean_activations = torch.stack(class_mean_activations, dim=1)

    # [classes, d_model]
    return class_mean_activations


# %%


def intervene(model, interv_vec, hook_layer):
    def patch(_vec):
        def hook(module, input, output):
            output[0][:, -1:, :] = output[0][:, -1:, :] + interv_vec
            return output

        return hook

    def hook_model():
        return model.model.layers[hook_layer].register_forward_hook(patch(interv_vec))

    hooks = hook_model()
    return hooks


# %%


@torch.no_grad()
def get_probs(model, token_ids, word_token_ids):
    zxcv = model(token_ids.to("cuda"))
    logits = zxcv.logits
    probs = logits.softmax(dim=-1)[:, :, word_token_ids]
    return probs


# %%

is_random = False
n_testcases = 1000
graph_size = adj.shape[0]
batch_size = 10
seq_len = 2000
num_comps = 2
word_token_ids = dgp.word_tokens
hook_layers = list(range(28, 32))
record_params["layers_to_save"] = hook_layers
accs = []
null_interv_probs = []
hit_at_1 = 0
hit_at_3 = 0
expl_var_ratios = []
for idx in tqdm(range(0, n_testcases, batch_size)):
    batch = dgp.get_batched_sequences(
        batch_size=batch_size, n_examples=seq_len, uniform_init=False, method="traverse"
    )
    recording = utils.record(model, batch["token_ids"], record_params)

    # [layers, batch, seq, d_model]
    acts = recording["activations"]

    curr_nodes = batch["inds"][:, -1]
    rand_offset = torch.randint(1, graph_size, (batch_size,))
    target_nodes = (curr_nodes + rand_offset) % graph_size
    assert torch.all(curr_nodes != target_nodes)

    orig_probs = get_probs(model, batch["token_ids"], word_token_ids)

    # word_mask: [batch, graph_size, seq]
    for sample_idx in range(batch_size):
        sample = acts[:, sample_idx, :, :]

        # orig_probs = get_probs(
        #    model, batch["token_ids"][sample_idx].unsqueeze(0), word_token_ids
        # )
        # [classes, d_model]
        class_mean_acts = get_class_mean_activations(
            sample,
            batch["token_ids"].shape[-1],
            batch["word_masks"][sample_idx],
            1000,
            graph_size,
        )

        all_hooks = []
        for layer_idx, layer_num in enumerate(hook_layers):
            pca = utils.get_pca(class_mean_acts[layer_idx])
            pc_comps = pca.components_
            expl_var_ratios.append(pca.explained_variance_ratio_[:num_comps].sum())

            src_acts = acts[layer_idx, sample_idx, -1].to("cuda")
            target_class = target_nodes[sample_idx]
            dst_acts = class_mean_acts[layer_idx][target_class].to("cuda")

            interv_vec = None
            for pc_idx in range(num_comps):
                pc = torch.tensor(pc_comps[pc_idx]).to("cuda")
                src_comp = torch.dot(src_acts, pc) / pc.norm()
                dst_comp = torch.dot(dst_acts, pc) / pc.norm()

                dst_token_embed = model.lm_head.weight[word_token_ids[target_class]]
                dst_token_embed = dst_token_embed / dst_token_embed.norm()

                if interv_vec is None:
                    interv_vec = dst_comp * pc
                else:
                    interv_vec = interv_vec + (dst_comp * pc)

                interv_vec -= src_comp * pc
                interv_vec -= dst_token_embed

            if is_random:
                _interv_vec = torch.randn_like(interv_vec)
                interv_vec = _interv_vec / _interv_vec.norm() * interv_vec.norm()

            _hooks = intervene(model, interv_vec, layer_num)
            all_hooks.append(_hooks)

        # [1, seq, graph_size (+2)]
        interv_probs = get_probs(
            model, batch["token_ids"][sample_idx].unsqueeze(0), word_token_ids
        )
        for _hook in all_hooks:
            _hook.remove()

        # Accumulated prob mass.
        correct_prob = interv_probs[0, -1, :-2] * adj[target_nodes[sample_idx]].to(
            "cuda"
        )
        accs.append(correct_prob.sum().item())

        # Hit @ 1
        top_3 = interv_probs[0, -1, :-2].topk(k=3).indices.cpu()
        gt_nodes = adj[target_nodes[sample_idx]]

        if gt_nodes[top_3[0]]:
            hit_at_1 += 1

        if torch.any(gt_nodes[top_3]):
            hit_at_3 += 1



# %%

print(np.mean(accs))
print(hit_at_1 / n_testcases)
print(hit_at_3 / n_testcases)

