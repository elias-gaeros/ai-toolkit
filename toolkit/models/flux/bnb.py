from collections import defaultdict
import re

import torch
from bitsandbytes.nn.modules import Params4bit, QuantState, LinearNF4


RE_QUANT_STATE = re.compile(r"(.*)\.weight\.([^.]+(?:\.[^.]+)?)")


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
    weight_keys = set()
    for k, m in modules_dict.items():
        quant_state_dict = quant_states.get(k)
        if quant_state_dict is None:
            continue

        # Create the layer
        layer = LinearNF4(
            m.in_features, m.out_features, bias=m.bias is not None, device="meta"
        )

        # Load the quant state from the state dict
        quant_state_dict = quant_state_dict.copy()
        quant_state_dict["quant_type"] = "nf4"
        weight_key = f"{k}.weight"
        weight_keys.add(weight_key)
        params = Params4bit.from_prequantized(
            state_dict.pop(weight_key), quant_state_dict, device=device
        )
        params.module = layer
        layer.weight = params
        layer.quant_state = params.quant_state

        parent, _, name = k.rpartition(".")
        setattr(modules_dict[parent], name, layer)
        modules_dict[k] = layer

    # Load the rest of the state dict
    res = module.load_state_dict(
        {k: v.to(device=device) for k, v in state_dict.items()},
        assign=True,
        strict=False,
    )
    missing_keys = [k for k in res.missing_keys if k not in weight_keys]
    if strict:
        if missing_keys:
            raise ValueError(f"Missing keys: {missing_keys}")
        if res.unexpected_keys:
            raise ValueError(f"Unexpected keys: {res.unexpected_keys}")
    return torch.nn.modules.module._IncompatibleKeys(
        missing_keys, res.unexpected_keys
    )
