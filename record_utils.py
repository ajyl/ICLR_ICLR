import re
from collections import defaultdict
from typing import Any, Generator, Sequence, cast, Callable, Iterable, Literal
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


RECORD_ALL = [
    "embed",
    "pos",
    "decoder_block",
    "self_attn",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm",
]


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def clear_all_forward_hooks(model: nn.Module) -> None:
    """Clear all forward hooks from the given model"""
    model._forward_hooks.clear()
    for _name, submodule in model.named_modules():
        submodule._forward_hooks.clear()


LayerMatcher = str | Callable[[nn.Module, int], str]


def collect_matching_layers(model: nn.Module, layer_matcher: LayerMatcher) -> list[str]:
    """
    Find all layers in the model that match the layer_matcher, in order by layer_num.
    layer_matcher can be a string formatted like "transformer.h.{num}.mlp" or a callable
    If layer_matcher is a callable, it should take in a model and layer_num and return
    a string representing the layer name corresponding to that layer number.
    If layer_matcher is a string, it's considered a template and MUST contain a "{num}" portion
    """
    matcher_callable = _layer_matcher_to_callable(layer_matcher)
    all_layer_names = dict(model.named_modules()).keys()
    matching_layers = []
    for layer_num in range(len(all_layer_names)):
        layer_name = matcher_callable(model, layer_num)
        if layer_name in all_layer_names:
            matching_layers.append(layer_name)
        else:
            break
    return matching_layers


def get_num_matching_layers(model: nn.Module, layer_matcher: LayerMatcher) -> int:
    """Returns the number of layers in the model that match the layer_matcher"""
    return len(collect_matching_layers(model, layer_matcher))


def get_layer_name(
    model: nn.Module, layer_matcher: LayerMatcher, layer_num: int
) -> str:
    matcher_callable = _layer_matcher_to_callable(layer_matcher)
    layer_num = fix_neg_layer_num(model, layer_matcher, layer_num)
    return matcher_callable(model, layer_num)


def fix_neg_layer_num(
    model: nn.Module, layer_matcher: LayerMatcher, layer_num: int
) -> int:
    """Helper to handle negative layer nums. If layer_num is negative, return len(layers) + layer_num"""
    if layer_num >= 0:
        return layer_num
    matching_layers = collect_matching_layers(model, layer_matcher)
    return len(matching_layers) + layer_num


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    return dict(model.named_modules())[layer_name]


def _layer_matcher_to_callable(
    layer_matcher: LayerMatcher,
) -> Callable[[nn.Module, int], str]:
    if isinstance(layer_matcher, str):
        if "wte" in layer_matcher:
            return lambda _model, layer_num: layer_matcher
        if "wpe" in layer_matcher:
            return lambda _model, layer_num: layer_matcher

        if "{num}" in layer_matcher:
            return lambda _model, layer_num: layer_matcher.format(num=layer_num)

    return layer_matcher


LAYER_GUESS_RE = r"^([^\d]+)\.([\d]+)(.*)$"


def guess_decoder_block_matcher(model: nn.Module) -> str | None:
    """
    Guess the hidden layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(dict(model.named_modules()).keys())


def guess_mlp_matcher(model: nn.Module) -> str | None:
    """
    Guess the mlp layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(), filter=lambda guess: "mlp" in guess
    )


def guess_self_attn_matcher(model: nn.Module) -> str | None:
    """
    Guess the self attention layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(),
        filter=lambda guess: "attn" in guess or "attention" in guess,
    )


def guess_input_layernorm_matcher(model: nn.Module) -> str | None:
    """
    Guess the input layernorm layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(),
        filter=lambda guess: "ln_1" in guess or "input_layernorm" in guess,
    )


def guess_post_attention_layernorm_matcher(model: nn.Module) -> str | None:
    """
    Guess the post-attention layernorm layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(),
        filter=lambda guess: "ln_2" in guess or "post_attention_layernorm" in guess,
    )


def guess_embed_matcher(model: nn.Module) -> str | None:
    for name, module in model.named_modules():
        if "wte" in name:
            return name
    return None


def guess_pos_matcher(model: nn.Module) -> str | None:
    for name, module in model.named_modules():
        if "wpe" in name:
            return name
    return None


# broken into a separate function for easier testing
def _guess_block_matcher_from_layers(
    layers: Iterable[str], filter: Callable[[str], bool] | None = None
) -> str | None:
    counts_by_guess: dict[str, int] = defaultdict(int)

    for layer in layers:
        if re.match(LAYER_GUESS_RE, layer):
            guess = re.sub(LAYER_GUESS_RE, r"\1.{num}\3", layer)
            if filter is None or filter(guess):
                counts_by_guess[guess] += 1
    if len(counts_by_guess) == 0:
        return None

    # score is higher for guesses that match more often, are and shorter in length
    guess_scores = [
        (guess, count + 1 / len(guess)) for guess, count in counts_by_guess.items()
    ]
    return max(guess_scores, key=lambda x: x[1])[0]


LayerType = Literal[
    "embed",
    "pos",
    "decoder_block",
    "self_attn",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm",
]

ModelLayerConfig = dict[LayerType, LayerMatcher]


_LAYER_TYPE_TO_GUESSER: dict[LayerType, Callable[[nn.Module], str | None]] = {
    "embed": guess_embed_matcher,
    "pos": guess_pos_matcher,
    "decoder_block": guess_decoder_block_matcher,
    "self_attn": guess_self_attn_matcher,
    "mlp": guess_mlp_matcher,
    "input_layernorm": guess_input_layernorm_matcher,
    "post_attention_layernorm": guess_post_attention_layernorm_matcher,
}


def enhance_model_config_matchers(
    model: nn.Module, config: ModelLayerConfig, layer_type: LayerType | None = None
) -> ModelLayerConfig:
    """Returns a new layer config, attempting to fill-in missing layer matchers"""
    enhanced_config: ModelLayerConfig = {**config}
    types_to_guess: Iterable[LayerType] = (
        [layer_type] if layer_type is not None else _LAYER_TYPE_TO_GUESSER.keys()
    )
    for guess_layer_type in types_to_guess:
        if guess_layer_type not in config and (
            layer_matcher := _LAYER_TYPE_TO_GUESSER[guess_layer_type](model)
        ):
            enhanced_config[guess_layer_type] = layer_matcher

    return enhanced_config


def guess_and_enhance_layer_config(
    model: nn.Module,
    layer_config: ModelLayerConfig | None = None,
    layer_type: LayerType | None = None,
) -> ModelLayerConfig:
    """
    Try to guess any missing parts of the layer config, after checking against predefined configs.
    If layer_type is provided, only guess the layer_matcher for that layer type.
    """
    layer_config = enhance_model_config_matchers(
        model, layer_config or {}, layer_type=layer_type
    )
    return layer_config


@contextmanager
def record_activations(
    model: nn.Module,
    layer_types: list[LayerType] = RECORD_ALL,
    layer_config: ModelLayerConfig | None = None,
    clone_activations: bool = True,
    layer_nums: Sequence[int] | None = None,
) -> Generator[dict[str, list[Tensor]], None, None]:
    """
    Record the model activations at each layer of type `layer_type`.
    This function will record every forward pass through the model
    at all layers of the given layer_type.

    Args:
        model: The model to record activations from
        layer_type: The type of layer to record activations from
        layer_config: A dictionary mapping layer types to layer matching functions.
            If not provided, this will be inferred automatically.
        clone_activations: If True, clone the activations before recording them. Default True.
        layer_nums: A list of layer numbers to record activations from. If None, record
            activations from all matching layers
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
        >>> with record_activations(model, layer_type="decoder_block") as recorded_activations:
        >>>     model.forward(...)
        >>> # recorded_activations is a dictionary mapping layer numbers to lists of activations
    """
    recorded_activations: dict[int, list[Tensor]] = defaultdict(list)
    layer_config = guess_and_enhance_layer_config(model, layer_config)

    matching_layers = []
    for layer_type in layer_types:
        if layer_type not in layer_config:
            raise ValueError(f"layer_type {layer_type} not provided in layer config")
        matcher = layer_config[layer_type]
        matching_layers.extend(collect_matching_layers(model, matcher))
    hooks: list[RemovableHandle] = []
    for layer_name in matching_layers:
        module = get_module(model, layer_name)
        hook_fn = _create_read_hook(
            layer_name,
            recorded_activations,
            clone_activations=clone_activations,
        )
        hooks.append(module.register_forward_hook(hook_fn))
    try:
        yield recorded_activations
    finally:
        for hook in hooks:
            hook.remove()


def _create_read_hook(
    layer_name: str, records: dict[int, list[Tensor]], clone_activations: bool
) -> Any:
    """Create a hook function that records the model activation at :layer_name:"""

    def hook_fn(_m: Any, _inputs: Any, outputs: Any) -> Any:
        activation = untuple_tensor(outputs)
        if not isinstance(cast(Any, activation), Tensor):
            raise ValueError(
                f"Expected a Tensor reading model activations, got {type(activation)}"
            )
        if clone_activations:
            activation = activation.clone().detach()
        records[layer_name].append(activation)
        return outputs

    return hook_fn


def build_activations_stack(
    recording: dict[str, list[Tensor]],
    num_layers: int,
    include_lns=True,
    use_residual=True,
) -> Tensor:
    """
    Build activations stack given recording.

    Decomps: [embed, pos, ln1_i, attn_i, ln2_i, mlp_i] for i in range(n_layers)
    """
    embed = recording["transformer.wte"][0]
    pos = recording["transformer.wpe"][0].unsqueeze(0).repeat((embed.shape[0], 1, 1))

    _input = embed + pos

    resid_streams = [_input]
    decomps = [embed, pos]

    for layer_num in range(num_layers):
        ln_1 = recording[f"transformer.h.{layer_num}.ln_1"][0]
        attn = recording[f"transformer.h.{layer_num}.attn"][0]
        ln_2 = recording[f"transformer.h.{layer_num}.ln_2"][0]
        mlp = recording[f"transformer.h.{layer_num}.mlp"][0]

        _resid_stream = recording[f"transformer.h.{layer_num}"][0]

        if include_lns:
            decomps.extend([ln_1, attn, ln_2, mlp])
        else:
            decomps.extend([attn, mlp])

        mid = resid_streams[-1] + attn
        resid_streams.extend([mid, _resid_stream])
        if use_residual:
            assert torch.equal(mid + mlp, _resid_stream)

    resid_streams = torch.stack(resid_streams, dim=1)
    decomps = torch.stack(decomps, dim=1)

    assert resid_streams.shape[1] == (2 * num_layers) + 1
    assert decomps.shape[1] == (4 * num_layers) + 2
    return resid_streams, decomps


def reshape_activations_stack(
    acts: Tensor, n_resid_streams: int, include_lns=True
) -> Tensor:
    """
    Reshape acts.
    Params
    :acts: [batch, n_layers, seq, d_model], where n_layers are ordered in
        resid_stream + decomps.
    :n_resid_streams: Number of layers (resid streams)

    Returns
    :reshaped: [batch, n_layers, seq, d_model], where n_layers are zigged
        such that the order follows the usual transformer
        (i.e., emb, ln1, attn1, ln2, mlp1, resid_stream1, ...)
    """
    acts = acts.permute(1, 0, 2, 3)

    resids = acts[:n_resid_streams]
    decomps = acts[n_resid_streams:]

    reshaped = [resids[0]]
    decomp_idx = 0
    for resid_idx in range(1, n_resid_streams):
        reshaped.append(decomps[decomp_idx])
        if include_lns:
            decomp_idx += 1
            reshaped.append(decomps[decomp_idx])

        decomp_idx += 1

        reshaped.append(resids[resid_idx])

    # [n_layers, batch, seq, d_model]
    reshaped = torch.stack(reshaped, dim=0)
    return reshaped.permute(1, 0, 2, 3)
