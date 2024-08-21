from collections import defaultdict
import re

import torch
from bitsandbytes.nn.modules import Params4bit, QuantState, LinearNF4
from bitsandbytes.functional import dequantize_nf4


RE_QUANT_STATE = re.compile(r"(.*)\.weight\.([^.]+(?:\.[^.]+)?)")


class LinearNF4(torch.nn.Module):
    """Non learnable, but differentiable, unlike bitsandbytes.nn.modules.Linear"""

    def __init__(self, weight, bias, quant_states):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.quant_states = quant_states
        self.in_features = quant_states.shape[1]
        self.out_features = quant_states.shape[0]

    def forward(self, x):
        weight = dequantize_nf4(self.weight, self.quant_states)
        return torch.nn.functional.linear(x, weight, self.bias)


def load_quantized_statedict(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict=True,
    device="cuda",
):
    # Filter the quantized states out of the state dict
    removed_keys = set()
    quant_states = defaultdict(dict)
    for k, tensor in state_dict.items():
        m = RE_QUANT_STATE.match(k)
        if m is not None:
            parent, quant_sate_name = m.groups()
            quant_states[parent][quant_sate_name] = tensor
            removed_keys.add(k)

    state_dict = {k: v for k, v in state_dict.items() if k not in removed_keys}

    # Swap the quantized layers and load the quant states
    modules_dict = dict(module.named_modules())
    removed_keys = set()
    for k, m in modules_dict.items():
        quant_state_dict = quant_states.get(k)
        if quant_state_dict is None:
            continue

        # Load the quant state from the state dict
        quant_state_dict = quant_state_dict.copy()
        quant_state_dict["quant_type"] = "nf4"
        weight_key = f"{k}.weight"
        bias_key = f"{k}.bias"
        removed_keys.add(weight_key)
        removed_keys.add(bias_key)

        layer = LinearNF4(
            state_dict.pop(weight_key),
            state_dict.pop(bias_key, None),
            QuantState.from_dict(quant_state_dict, device=device),
        )

        parent, _, name = k.rpartition(".")
        setattr(modules_dict[parent], name, layer)
        modules_dict[k] = layer

    # Load the rest of the state dict
    res = module.load_state_dict(
        {k: v.to(device=device) for k, v in state_dict.items()},
        assign=True,
        strict=False,
    )
    missing_keys = [k for k in res.missing_keys if k not in removed_keys]
    if strict:
        if missing_keys:
            raise ValueError(f"Missing keys: {missing_keys}")
        if res.unexpected_keys:
            raise ValueError(f"Unexpected keys: {res.unexpected_keys}")
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, res.unexpected_keys)
