# Select the initial training rows for Bayesian optimization (modified for Mallamar).
import logging
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.quasirandom import SobolEngine
import torch.nn as nn
from alice.data.utils import validate_indices

logger = logging.getLogger(__name__)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'[initializer] {name} must be a positive integer')
    return value


def _check_features(x):
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not torch.isfinite(x).all():
        raise ValueError('[initializer] Features must be a nonempty finite matrix')


class Initializer(ABC):
    @abstractmethod
    def fit(self, x):
        pass


class SobolInitializer(Initializer):
    def __init__(self, n_clusters=96, seed=None, **kwargs):
        self.n_clusters = _positive_integer(n_clusters, 'n_clusters')
        self.seed = seed

    def fit(self, x, **kwargs):
        _check_features(x)
        if self.n_clusters > len(x):
            raise ValueError('[SobolInitializer] n_clusters exceeds the number of available rows')
        device = x.device
        x_normalized_numpy = MinMaxScaler().fit_transform(x.cpu().numpy())
        x_scaled = torch.tensor(x_normalized_numpy, dtype=torch.float64).to(device)
        sobol = SobolEngine(dimension=x_scaled.shape[1], scramble=True, seed=self.seed)
        sobol_samples = sobol.draw(self.n_clusters).to(device)
        distances = torch.cdist(sobol_samples.float(), x_scaled.float())
        selected_indices = []
        temp_distances = distances.clone()
        for _ in range(self.n_clusters):
            min_distances, min_indices = temp_distances.min(dim=1)
            best_sobol_idx = torch.argmin(min_distances)
            data_idx = min_indices[best_sobol_idx.item()]
            selected_indices.append(data_idx.item())
            temp_distances[:, data_idx] = float('inf') # each data row can be selected once
            temp_distances[best_sobol_idx, :] = float('inf')
        return selected_indices, {}


class RandomInitializer(Initializer):
    def __init__(self, n_clusters=20, seed=0, **kwargs):
        self.n_clusters = _positive_integer(n_clusters, 'n_clusters')
        self.seed = seed

    def fit(self, x, exclude=None):
        _check_features(x)
        if exclude is None:
            return np.random.default_rng(seed=self.seed).permutation(len(x))[:self.n_clusters].tolist(), {}
        excluded = validate_indices(exclude, len(x), allow_empty=True)
        available = np.delete(np.arange(len(x)), np.unique(excluded))
        if not len(available):
            raise ValueError('[RandomInitializer] No rows remain after exclusions')
        selected = np.random.default_rng(seed=self.seed).permutation(len(available))[:self.n_clusters]
        return available[selected].tolist(), {}


class PredefinedInitializer(Initializer):
    def __init__(self, predefined_points=None, **kwargs):
        self.predefined_points = predefined_points

    def fit(self, x, exclude=None, **kwargs):
        _check_features(x)
        selected = validate_indices(self.predefined_points, len(x))
        if exclude is not None:
            excluded = validate_indices(exclude, len(x), allow_empty=True)
            if np.intersect1d(selected, excluded).size:
                raise ValueError('[PredefinedInitializer] A predefined row is excluded')
        return selected.tolist(), {}


def torch_delete_rows(input_tensor, indices):
    if indices is None:
        return input_tensor
    indices = validate_indices(indices, len(input_tensor), allow_empty=True)
    mask = torch.ones(input_tensor.size(0), dtype=torch.bool, device=input_tensor.device)
    mask[torch.as_tensor(indices, device=input_tensor.device)] = False
    return input_tensor[mask]


def landscape_roughness(x, y, k=5, eps=1e-8):
    """Compute neighbour reward/input distance ratios without rescaling inputs."""
    distances = cdist(x, x, metric='euclidean')
    neighbour_count = min(k, len(x) - 1)
    roughness = np.zeros(len(x))
    for index in range(len(x)):
        neighbours = np.argsort(distances[index])[1:neighbour_count + 1]
        reward_distances = np.linalg.norm(y[index] - y[neighbours], axis=1)
        roughness[index] = np.mean(reward_distances / (distances[index, neighbours] + eps))
    return roughness, distances


def weighted_maximin(distances, weights, n_select):
    """Select rows by weight times distance from the rows already selected."""
    selected = [int(np.argmax(weights))]
    min_distances = distances[selected[0]].copy()
    for _ in range(min(n_select, len(distances)) - 1):
        scores = weights * min_distances
        scores[selected] = -np.inf
        next_index = int(np.argmax(scores))
        selected.append(next_index)
        min_distances = np.minimum(min_distances, distances[next_index])
    return selected


class RoughnessInitializer(Initializer):
    """Balance local reward changes with distance from the already selected rows.

    Roughness is mean neighbour reward distance / (input distance + eps).
    Selection maximizes roughness times distance from the current seed set.
    Input features are used directly, including token IDs when those are provided.
    """
    requires_y = True
    skip_featurizer = True

    def __init__(self, n_init: int = 24, k: int = 5, eps: float = 1e-8, **_kwargs):
        self.n_init = _positive_integer(n_init, 'n_init')
        self.k = _positive_integer(k, 'k')
        if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, (int, float, np.integer, np.floating)) or not np.isfinite(eps) or eps <= 0:
            raise ValueError('[RoughnessInitializer] eps must be finite and positive')
        self.eps = eps
        print('[RoughnessInitializer] Using reward-informed initialization')

    def fit(self, x, y=None, exclude=None):
        _check_features(x)
        if y is None:
            raise ValueError('RoughnessInitializer requires reward values (y).')
        if y.ndim != 2 or len(y) != len(x) or not y.shape[1] or not torch.isfinite(y).all():
            raise ValueError('[RoughnessInitializer] Rewards must be a finite matrix with one row per input')
        row_indices = np.arange(len(x))
        if exclude is not None:
            excluded = validate_indices(exclude, len(x), allow_empty=True)
            row_indices = np.delete(row_indices, np.unique(excluded))
        x_np = torch_delete_rows(x, exclude).cpu().numpy().astype(np.float64)
        y_np = torch_delete_rows(y, exclude).cpu().numpy().astype(np.float64)
        n = len(x_np)
        if not n:
            raise ValueError('[RoughnessInitializer] No rows remain after exclusions')
        if n == 1:
            return [int(row_indices[0])], {'roughness': [0.0]}
        roughness, distances = landscape_roughness(x_np, y_np, self.k, self.eps)
        roughness_norm = roughness / (roughness.max() + self.eps)
        print(f'[RoughnessInitializer] roughness range: [{roughness.min():.4f}, {roughness.max():.4f}] mean={roughness.mean():.4f}')
        selected = weighted_maximin(distances, roughness_norm, self.n_init)
        print(f'[RoughnessInitializer] selected {len(selected)} points')
        return row_indices[selected].tolist(), {'roughness': roughness.tolist()}


class BOInitializer:
    """Optionally embed/PCA-reduce inputs, then return sorted initial row indices."""

    def __init__(self, method: str = 'sobol', n_clusters: int = None, pca_dim: int = None,
                 seed: Optional[int] = None, featurizer: Optional[nn.Module] = None,
                 tkwargs: Optional[dict] = None, **kwargs):
        if n_clusters is not None:
            _positive_integer(n_clusters, 'n_clusters')
        if pca_dim is not None:
            _positive_integer(pca_dim, 'pca_dim')
        self.seed = seed
        self.n_clusters = n_clusters
        self.featurizer = featurizer
        self.pca_dim = pca_dim
        self.tkwargs = tkwargs if tkwargs is not None else {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'dtype': torch.float64}
        if self.featurizer:
            logger.info('BOInitializer: embedding inputs with %s before selection.', type(self.featurizer).__name__)
            self.featurizer.eval()
        else:
            logger.info('BOInitializer: no featurizer, selecting on raw inputs.')
        self.methods = {'sobol': SobolInitializer, 'true_random': RandomInitializer,
                        'predefined': PredefinedInitializer, 'roughness': RoughnessInitializer}
        if method not in self.methods:
            raise ValueError(f'Unknown init_method: {method}')
        parameters = dict(seed=seed, **kwargs)
        if n_clusters is not None:
            parameters['n_clusters'] = n_clusters # leave the concrete default in place when omitted
        self.initializer = self.methods[method](**parameters)
        self.n_clusters = getattr(self.initializer, 'n_clusters', n_clusters)
        self.selected_reactions = None
        self.clusters = None

    def fit(self, x, y=None):
        _check_features(x)
        skip_featurizer = getattr(self.initializer, 'skip_featurizer', False)
        if not skip_featurizer and self.featurizer:
            logger.debug('Embedding inputs (shape %s) with the featurizer.', tuple(x.shape))
            with torch.no_grad():
                features = self.featurizer.get_embeddings(x.to(self.tkwargs['device']))
        else:
            logger.debug('Using inputs (shape %s) directly.', tuple(x.shape))
            features = x
        x_numpy = features.cpu().numpy()
        if self.pca_dim is not None and not skip_featurizer:
            if self.pca_dim > min(x_numpy.shape):
                raise ValueError('[BOInitializer] pca_dim exceeds available rows or feature dimensions')
            x_scaled = StandardScaler().fit_transform(x_numpy)
            x_processed = PCA(n_components=self.pca_dim).fit_transform(x_scaled)
            logger.info('Reduced features %d -> %d dims (StandardScaler + PCA).', x_numpy.shape[1], self.pca_dim)
        else:
            logger.debug('No PCA applied.')
            x_processed = x_numpy
        x_for_init = torch.from_numpy(x_processed).to(**self.tkwargs)
        if getattr(self.initializer, 'requires_y', False):
            result = self.initializer.fit(x_for_init, y=y, exclude=None)
        else:
            result = self.initializer.fit(x_for_init)
        selected_indices, clusters = result if isinstance(result, tuple) else (result, {})
        selected = validate_indices(selected_indices, len(x))
        if len(np.unique(selected)) != len(selected):
            raise RuntimeError(f'Initializer returned duplicate indices: {selected}')
        self.selected_reactions = np.sort(selected)
        self.clusters = clusters
        return self.selected_reactions, self.clusters


def select_sobol_candidates(x, n_init, seed):
    """Snap seeded Sobol draws to unused rows, in draw order."""
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('[cold_start] Need a nonempty pool with finite concentrations')
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (x - lo) / span

    sob = torch.quasirandom.SobolEngine(dimension=x.shape[1], scramble=True, seed=seed)
    pts = sob.draw(min(n_init, len(x))).numpy()
    chosen, used = [], set()
    for p in pts:
        for j in np.argsort(((scaled - p) ** 2).sum(axis=1)):
            j = int(j)
            if j not in used:
                used.add(j); chosen.append(j); break

    return chosen
