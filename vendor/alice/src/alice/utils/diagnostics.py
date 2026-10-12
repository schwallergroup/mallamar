# Report model, sampling and optimization diagnostics without changing them.
import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from alice.data.module import BaseDataModule


def print_diagnostic_summary(dm: BaseDataModule, rep_short, trainable, model_name):
    if not np.array_equal(dm.data.index.values, np.arange(len(dm.data))):
        raise ValueError(f'DataFrame index is not a contiguous 0..N-1 range. Positional indices '
                         f'used by the BO loop would not align with DataFrame rows. Source: {dm.data_path}')
    print('[data] Experiment diagnostic summary')
    print(f'Features: {tuple(dm.x.shape)}; objectives: {tuple(dm.y.shape)}; rows: {len(dm.data)}')
    print(f'Target columns: {dm.target_column}; maximize: {getattr(dm, "maximize", None)}')
    for index, values in enumerate(dm.y.T):
        measured = values[torch.isfinite(values)]
        limits = f'{measured.min().item():.3f} to {measured.max().item():.3f}' if len(measured) else 'unmeasured'
        print(f'Objective {index + 1} range: {limits}')
    print(f'Representation: {rep_short}; model: {model_name}; trainable: {trainable}')
    print(f'Featurizer: {type(dm.featurizer).__name__}')
    for label, prefix, indices_name in [('Train', 'train', 'train_indexes'), ('Heldout', 'heldout', 'heldout_indices')]:
        indices = getattr(dm, indices_name, [])
        x_shape = tuple(getattr(dm, prefix + '_x', torch.empty(0)).shape)
        y_shape = tuple(getattr(dm, prefix + '_y', torch.empty(0)).shape)
        print(f'{label} features: {x_shape}; targets: {y_shape}; indices: {len(indices)}')
        if prefix == 'train':
            print(f'Sample train indices: {indices[:5]}')


def print_bo_diagnostic(dm: BaseDataModule, config, bo, model_name, acqf_name):
    parameters = config.get('acquisition', {}).get('init_args', {})
    reference = 'not specified'
    if 'ref_point' in parameters:
        try:
            tensor = torch.as_tensor(parameters['ref_point'], dtype=dm.y.dtype)
            reference = f'{tuple(tensor.cpu().numpy())}; dtype: {tensor.dtype}; shape: {tuple(tensor.shape)}'
        except (ValueError, TypeError, RuntimeError) as error:
            reference = f'conversion failed: {error}'
    print('[bo] Optimizer diagnostic summary')
    print(f'Optimizer: {type(bo).__name__}; acquisition: {acqf_name}; model: {model_name}')
    print(f'Batch size: {config.get("bo", {}).get("init_args", {}).get("batch_size", "N/A")}')
    print(f'Reference point: {reference}')
    print(f'Initial training rows: {len(dm.train_indexes)}; heldout rows: {len(dm.heldout_indices)}')


def mean_gp_lengthscale(optimizer):
    """Average each fitted objective's lengthscale, then average across objectives."""
    models = getattr(getattr(getattr(optimizer, "surrogate_model", None), "model", None), "models", None) or []
    lengthscales = [float(model.covar_module.base_kernel.lengthscale.detach().cpu().mean()) for model in models]
    return sum(lengthscales) / len(lengthscales) if lengthscales else None


def sampling_coverage(x):
    """Return PCA variance and five-bin coverage of the first component."""
    scaled = MinMaxScaler().fit_transform(x)
    pca = PCA(n_components=min(2, scaled.shape[1]))
    coordinates = pca.fit_transform(scaled)
    pc1 = coordinates[:, 0]
    counts, _ = np.histogram(pc1, bins=np.linspace(pc1.min(), pc1.max(), 6))
    return pca.explained_variance_ratio_, counts


def suggestion_diversity(x):
    """Return minimum and mean distances after scaling each feature to 0–1."""
    scaled = MinMaxScaler().fit_transform(x) if len(x) > 1 else x
    distances = cdist(scaled, scaled)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min()), float(distances[distances < np.inf].mean())


def hypervolume_2d(points, ref):
    """Compute 2-objective hypervolume dominated by `points` relative to `ref` (maximisation)"""
    if len(points) == 0:
        return 0.0
    pts = points[(points[:, 0] > ref[0]) & (points[:, 1] > ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[pts[:, 0].argsort()[::-1]]
    hv, prev_y = 0.0, ref[1]
    for p in pts:
        if p[1] > prev_y:
            hv += (p[0] - ref[0]) * (p[1] - prev_y)
            prev_y = p[1]
    return float(hv)
