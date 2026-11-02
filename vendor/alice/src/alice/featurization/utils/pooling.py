# Pool token embeddings while excluding padding.
import torch


def _check_shapes(hidden_states, attention_mask):
    if (hidden_states.ndim != 3 or attention_mask.ndim != 2
            or hidden_states.shape[:2] != attention_mask.shape or not hidden_states.shape[1]):
        raise ValueError('[pooling] Expected embeddings (batch, tokens, features) and a matching token mask')
    if hidden_states.device != attention_mask.device:
        raise ValueError('[pooling] Embeddings and token masks must be on the same device')


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    _check_shapes(last_hidden_states, attention_mask)
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    counts = attention_mask.sum(dim=1)[..., None]
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    return last_hidden.sum(dim=1) / counts # an entirely padded sequence gives a zero vector


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, left_padding: bool = False):
    _check_shapes(last_hidden_states, attention_mask)
    if attention_mask.is_floating_point() or attention_mask.is_complex():
        raise ValueError('[last_token_pool] Expected a boolean or integer attention mask')
    if left_padding:
        return last_hidden_states[:, -1].masked_fill(~attention_mask.any(dim=1)[..., None], 0.0)
    sequence_lengths = attention_mask.sum(dim=1) - 1
    last_hidden = last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device),
                                   sequence_lengths.clamp(min=0)]
    return last_hidden.masked_fill((sequence_lengths < 0)[..., None], 0.0)


def weighted_average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    _check_shapes(last_hidden_states, attention_mask)
    seq_length = last_hidden_states.size(1)
    weights = torch.arange(1, seq_length + 1, dtype=torch.float32).unsqueeze(0).to(last_hidden_states.device)
    weighted_mask = weights * attention_mask.float()
    weighted_hidden_states = last_hidden_states * weighted_mask.unsqueeze(-1)
    sum_weighted_embeddings = torch.sum(weighted_hidden_states, dim=1)
    sum_weights = torch.sum(weighted_mask, dim=1, keepdim=True).clamp(min=1)
    return sum_weighted_embeddings / sum_weights
