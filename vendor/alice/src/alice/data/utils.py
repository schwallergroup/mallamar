# Validate row selections and locate problematic features.
import numpy as np
import torch


def validate_indices(indices, n_rows, allow_empty=False):
    if isinstance(indices, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in indices):
        raise ValueError('[indices] Boolean values are not row indices')
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    selected = np.asarray(indices)
    if selected.ndim != 1 or (not allow_empty and selected.size == 0):
        raise ValueError('[indices] Expected a nonempty one-dimensional row selection')
    if selected.dtype.kind not in 'iuf' or not np.isfinite(selected).all():
        raise ValueError('[indices] Row indices must be finite integers')
    if selected.dtype.kind == 'f' and np.any(selected != np.floor(selected)):
        raise ValueError('[indices] Fractional row indices are invalid')
    if np.any(selected < 0) or np.any(selected >= n_rows):
        raise ValueError(f'[indices] Row indices must be in [0, {n_rows})')
    return selected.astype(np.int64, copy=False)


def find_duplicates(x: torch.Tensor):
    _, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True, dim=0)
    if not (counts > 1).any():
        return []
    order = torch.argsort(inverse, stable=True) # group duplicates while keeping their original row order
    groups = inverse[order]
    repeated = torch.cat([torch.zeros(1, dtype=torch.bool, device=x.device), groups[1:] == groups[:-1]])
    return order[repeated].tolist()


def find_nan_rows(x: torch.Tensor):
    return torch.isnan(x).any(dim=1).nonzero().flatten().tolist()


class NearestCandidates:
    """Rank numerical candidates on requested axes; callers exclude occupied rows."""

    def __init__(self, x):
        self.x = x
        self.span = x.max(axis=0) - x.min(axis=0)
        self.span[self.span == 0] = 1.0 # constant axes use their original distance

    def rank(self, point, columns):
        distances = np.linalg.norm((self.x[:, columns] - point) / self.span[columns], axis=1)
        return distances, np.argsort(distances)
