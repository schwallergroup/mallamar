# Load, featurize and split CSV data for Bayesian optimization (modified for Mallamar).
import logging
from typing import List, Union, Optional
import numpy as np
import pandas as pd
import torch
from alice.data.utils import find_duplicates, find_nan_rows, validate_indices
from alice.utils.config import instantiate_class
from alice.initialization.initializers import BOInitializer
from alice.featurization.base import Featurizer

logger = logging.getLogger(__name__)


class BaseDataModule:
    """Keep raw rows and model tensors together, with positional train/heldout indices."""

    def __init__(self, data_path: str, input_column: Union[str, List[str]] = 'input',
                 target_column: Union[str, List[str]] = 'target', maximize: Union[bool, List[bool]] = True,
                 featurizer: Optional[Featurizer] = None, initializer: Optional[BOInitializer] = None,
                 normalize_input: str = 'original', dtype: torch.dtype = torch.float32):
        super().__init__()
        self.data_path = data_path
        self.input_column = input_column
        self.normalize_input = normalize_input
        self.dtype = dtype
        self.target_column = [target_column] if isinstance(target_column, str) else target_column.copy()
        if not self.target_column or not all(isinstance(column, str) and column for column in self.target_column):
            raise ValueError('[BaseDataModule] Target columns must be nonempty names')
        if isinstance(maximize, bool):
            self.maximize = [maximize] * len(self.target_column)
        else:
            if len(maximize) != len(self.target_column):
                raise ValueError("Length of 'maximize' must match number of target columns.")
            self.maximize = maximize.copy()
        self.featurizer = featurizer if featurizer is not None else Featurizer()
        self.initializer = initializer
        self.setup()

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        if self.data.empty:
            raise ValueError(f'[BaseDataModule] Empty dataset: {self.data_path}')
        inputs = self.input_column if isinstance(self.input_column, list) else [self.input_column]
        if not inputs or not all(isinstance(column, str) and column for column in inputs):
            raise ValueError('[BaseDataModule] Input columns must be nonempty names')
        missing = sorted(set(inputs + self.target_column) - set(self.data.columns))
        if missing:
            raise ValueError(f'[BaseDataModule] Missing columns {missing}: {self.data_path}')
        for column, do_maximize in zip(self.target_column, self.maximize):
            if not do_maximize:
                self.data[column] = -self.data[column]

    def featurize_data(self):
        representation = getattr(self.featurizer, 'representation', None)
        if representation == 'descriptors':
            logger.info("Featurizing in 'descriptors' mode, selecting input columns directly.")
            if not isinstance(self.input_column, list):
                raise TypeError("For 'descriptors' mode, 'input_column' must be a list of column names.")
            x = self.data[self.input_column].to_numpy()
        else:
            logger.info('Featurizing with representation %r.', representation)
            inputs = ([self.data[column] for column in self.input_column]
                      if isinstance(self.input_column, list) else self.data[self.input_column])
            x = self.featurizer.featurize(inputs)
        self.x = torch.tensor(x, dtype=self.dtype)
        self.y = torch.tensor(self.data[self.target_column].to_numpy(), dtype=self.dtype)
        if self.x.ndim != 2 or self.x.shape[0] != len(self.data) or not self.x.shape[1]:
            raise ValueError('[BaseDataModule] Features must have one nonempty row per CSV row')

    def preprocess_data(self):
        if torch.isinf(self.x).any() or torch.isinf(self.y).any():
            raise ValueError(f'[BaseDataModule] Infinite features or targets: {self.data_path}')
        problematic = np.union1d(find_nan_rows(self.x), find_duplicates(self.x))
        if len(problematic):
            raise ValueError(f'Data contains {len(problematic)} problematic rows (NaNs or duplicates). '
                             f'Please clean your data at: {self.data_path}')
        if torch.isnan(self.y).any():
            n_rows = int(torch.isnan(self.y).any(dim=1).sum())
            raise ValueError(f'Target columns contain NaN in {n_rows} rows. Source: {self.data_path}')

    def normalize_data(self):
        if self.normalize_input == 'standard_scaling':
            mean = self.x.mean(0)
            std = self.x.std(0) if len(self.x) > 1 else torch.ones_like(mean)
            self.x = (self.x - mean) / torch.where(std == 0, torch.ones_like(std), std)
        elif self.normalize_input in ('l2_max_scaling', 'l2_normalize'):
            norms = self.x.norm(dim=1, keepdim=True)
            scale = norms.max() if self.normalize_input == 'l2_max_scaling' else norms
            if not torch.isfinite(scale).all():
                raise ValueError('[BaseDataModule] Input norms overflowed; rescale the source features')
            self.x = self.x / torch.where(scale == 0, torch.ones_like(scale), scale)
        elif self.normalize_input != 'original':
            raise ValueError(f'Unknown normalize_input: {self.normalize_input!r}')
        if not torch.isfinite(self.x).all():
            raise ValueError('[BaseDataModule] Normalization produced non-finite features')

    def split_data(self):
        if not hasattr(self.initializer, 'fit'):
            raise AttributeError(f'Initializer {self.initializer.__class__.__name__} must implement fit() method')
        if getattr(getattr(self.initializer, 'initializer', None), 'requires_y', False):
            result = self.initializer.fit(self.x, y=self.y)
        else:
            result = self.initializer.fit(self.x)
        indices = validate_indices(result[0] if isinstance(result, tuple) else result, len(self.x))
        if len(np.unique(indices)) != len(indices):
            raise ValueError('[BaseDataModule] Initializer returned duplicate row indices')
        self.train_indexes = torch.as_tensor(indices, dtype=torch.long)
        heldout = sorted(set(range(len(self.x))) - set(indices))
        self.heldout_indices = torch.as_tensor(heldout, dtype=torch.long)
        self.train_x, self.train_y = self.x[self.train_indexes], self.y[self.train_indexes]
        self.heldout_x, self.heldout_y = self.x[self.heldout_indices], self.y[self.heldout_indices]

    def setup(self):
        self.load_data()
        self.featurize_data()
        self.preprocess_data()
        self.normalize_data()
        self.split_data()


class CampaignDataModule(BaseDataModule):
    """Preserve row identity and fit measured targets while keeping unknown rows available."""

    def __init__(self, *args, require_candidates=False, **kwargs):
        if not isinstance(require_candidates, bool):
            raise ValueError('[CampaignDataModule] require_candidates must be a boolean')
        self.require_candidates = require_candidates
        super().__init__(*args, **kwargs)

    def preprocess_data(self):
        missing = torch.isnan(self.y)
        if (missing.any(1) & ~missing.all(1)).any():
            raise ValueError("Partially measured objectives require explicit handling")
        self._pool_mode = bool(missing.any())

        if not self._pool_mode:
            if self.require_candidates:
                raise ValueError("Campaign pool has no unmeasured candidates remaining")
            return super().preprocess_data()
        
        observed = self.x[~missing.any(1)] # only fully observed rows
        if not len(observed):
            raise ValueError("No observations: run campaign cold-start initialization first")
        if len(find_nan_rows(self.x)) or torch.isinf(self.x).any():
            raise ValueError("Campaign features contain non-finite values")
        if len(find_duplicates(observed)):
            raise ValueError("Observed campaign features contain duplicate rows")
        if torch.isinf(self.y).any():
            raise ValueError("Campaign objectives contain infinite values")

    # Split the data into training and held-out sets, ensuring only observed rows are used for training.
    # Necessary for SDL to correctly handle partially measured campaign pools
    def split_data(self):
        if not self._pool_mode:
            return super().split_data()
        
        missing = torch.isnan(self.y).any(1)
        observed = torch.where(~missing)[0]

        # Fresh campaigns may have fewer observations than the configured PCA
        # dimension or initialization size. 

        if self.initializer.pca_dim is not None:
            self.initializer.pca_dim = min(self.initializer.pca_dim, len(observed), self.x.shape[1]) # ensure PCA dimension does not exceed available observations or feature count

        if self.initializer.n_clusters is not None:
            count = min(self.initializer.n_clusters, len(observed))
            self.initializer.n_clusters = count
            self.initializer.initializer.n_clusters = count # ensure the initializer respects the available observations

        selected = self.initializer.fit(self.x[observed], y=self.y[observed]) 
        local = validate_indices(selected[0] if isinstance(selected, tuple) else selected, len(observed)) 

        if not len(local) or len(np.unique(local)) != len(local):
            raise ValueError("Initializer must return a nonempty unique training split")

        # Observed pool indices and training/held-out splits
        self.observed_pool_indices = observed
        self.train_indexes = observed[torch.as_tensor(local, dtype=torch.long)]
        self.heldout_indices = torch.where(missing)[0]
        self.train_x, self.train_y = self.x[self.train_indexes], self.y[self.train_indexes]
        self.heldout_x, self.heldout_y = self.x[self.heldout_indices], self.y[self.heldout_indices]


def setup_data(config, device=None):
    """Build the embedding model, initializer and dataset in their fitting order."""
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = torch.float32
    tkwargs = {"device": device, "dtype": target_dtype}

    finetuning_cfg = config.get("surrogate_model", {}).get("init_args", {}).get("finetuning_model", None)
    embedding_model = instantiate_class(finetuning_cfg) if finetuning_cfg else None

    data_cfg = config["data"]["init_args"]
    initializer = instantiate_class(
        data_cfg["initializer"],
        seed=config["seed"],
        featurizer=embedding_model,
        tkwargs=tkwargs
    )
    featurizer = instantiate_class(data_cfg["featurizer"])
    dm = instantiate_class(
        config["data"],
        initializer=initializer,
        featurizer=featurizer,
        normalize_input=data_cfg["normalize_input"],
        maximize=data_cfg["maximize"],
    )
    return dm, target_dtype
