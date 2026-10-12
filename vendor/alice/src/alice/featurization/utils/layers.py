# Select linear modules for LoRA without changing their traversal order.
import math
from numbers import Real
import torch
import torch.nn as nn


def get_target_layers(
    model: nn.Module,
    proportion: float = 0.25,
    from_top: bool = True
):
    if isinstance(proportion, bool) or not isinstance(proportion, Real) or not math.isfinite(proportion) or not 0 <= proportion <= 1:
        raise ValueError('[get_target_layers] proportion must be finite and in [0, 1]')
    all_layers = []
    layer_numbers = set()

    def extract_layer_number(name):
        if "block." in name: # T5 style
            parts = name.split("block.")
            if len(parts) > 1:
                num = parts[1].split(".")[0]
                return int(num) if num.isdigit() else None
        elif "layers." in name: # ModernBERT style
            parts = name.split("layers.")
            if len(parts) > 1:
                num = parts[1].split(".")[0]
                return int(num) if num.isdigit() else None
        elif "layer." in name: # BERT style
            parts = name.split("layer.")
            if len(parts) > 1:
                num = parts[1].split(".")[0]
                return int(num) if num.isdigit() else None
        return None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_num = extract_layer_number(name)
            if layer_num is not None:
                layer_numbers.add(layer_num)
                all_layers.append((layer_num, name))

    if not layer_numbers:
        return []

    num_layers = len(layer_numbers)
    num_target_layers = max(1, round(num_layers * proportion))

    sorted_layer_nums = sorted(layer_numbers, reverse=from_top)
    target_layer_nums = set(sorted_layer_nums[:num_target_layers])

    target_modules = [
        name
        for layer_num, name in all_layers
        if layer_num in target_layer_nums
    ]

    print(
        f"\nFound {len(target_modules)} linear layers "
        f"({'top' if from_top else 'bottom'} {proportion*100:.1f}% of {num_layers} layers):"
    )
    print(f"Layer numbers selected: {sorted(target_layer_nums)}")

    return target_modules

