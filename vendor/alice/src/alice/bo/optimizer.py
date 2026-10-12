# Modified for Mallamar campaign compatibility.
"Module for running acquisition functions and obtaining suggestions"
from typing import Any, Dict, Optional
import logging, numpy as np

from tqdm import tqdm

import torch
from torch import Tensor
from botorch.sampling.normal import SobolQMCNormalSampler

from pytorch_lightning import seed_everything
from alice.utils.config import instantiate_class
from alice.utils.model_embd_sizes import MODEL_EMBEDDING_SIZES
from alice.data.module import setup_data
from alice.data.utils import validate_indices
from alice.bo.acqf_helper import ACQF_REGISTRY, batched_acqf_forward, MultiObjectiveUCB, RandomAcqf


logger = logging.getLogger(__name__)


class BotorchOptimizer:
    """Surrogate-model + acquisition-function optimizer for the BO loop.

    Instantiated once per run by ``setup_bo_optimizer`` in ``train_master.py``;
    ``suggest_next_experiments`` is then called once per BO iteration.

    Constructor arguments (source -> internal use):
        surrogate_model_config: ``config["surrogate_model"]``. Instantiated and
            fitted in ``train_surrogate_model`` (called per iteration).
        acq_function_config: ``config["acquisition"]``. Resolved to an acqf class
        via ``ACQF_REGISTRY``. Only ``init_args.ref_point`` is read from it, and
            only for the Noisy EHVI variants; all other ``init_args`` keys are
            ignored, since the constructor kwargs are set explicitly in ``suggest_next_experiments``.
        batch_size: ``config["bo"]["init_args"]["batch_size"]``. Number of greedy
            selection steps per iteration (``steps = min(batch_size, n_candidates)``).
        max_eval_chunk: chunk size for ``batched_acqf_forward``. Not set by any
            config or ``setup_bo_optimizer`` -> always the default (2048). Kept as a
            tuning knob, but effectively constant in the shipped pipeline.
        tkwargs: device/dtype dict, built by ``setup_bo_optimizer`` from ``dtype``.
            Used throughout for tensor placement.

    Attributes read OUTSIDE this class:
        surrogate_model: read by ``save_checkpoint`` (``utils/model_utils.py``) to
            persist the fitted featurizer/projection layer.

    Internal-only attributes:
        _shared_sampler: lazily-created ``SobolQMCNormalSampler`` reused across steps.
    """
    def __init__(
        self,
        surrogate_model_config: Optional[Dict[str, Any]] = None,
        acq_function_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        max_eval_chunk: int = 2048,
        tkwargs: Optional[Dict[str, Any]] = None,
        design_space=None,
        batch_mode="greedy",
    ):
        if tkwargs is None:
            raise ValueError(
                "tkwargs is required; setup_bo_optimizer supplies device and dtype."
            )

        if not acq_function_config:
            raise ValueError(
                "acq_function_config is required."
            )

        self.surrogate_model_config = surrogate_model_config
        self.acq_function_config = acq_function_config
        self.batch_size = batch_size
        self.tkwargs = tkwargs
        self.max_eval_chunk = max_eval_chunk

        if batch_mode != "greedy":
            raise ValueError("[BotorchOptimizer] Only greedy batch selection is supported")
        self._last_selected_indices = [] # kept for benchmark callers
        self._shared_sampler = None
        self.surrogate_model = None
        logger.info("BotorchOptimizer using device %s.", self.tkwargs["device"])

    def train_surrogate_model(self, train_x, train_y):
        "Instantiate surrogate and fit it"
        self.surrogate_model = instantiate_class(
            self.surrogate_model_config,
            train_x=train_x,
            train_y=train_y,
        )

        self.surrogate_model.model.to(**self.tkwargs)
        try:
            logger.debug("Fitting surrogate model.")
            self.surrogate_model.fit()
        except Exception:
            logger.exception("Surrogate model fit failed at this BO iteration.")
            raise

    def suggest_next_experiments(self, train_x: Tensor, train_y: Tensor, design_space: Tensor):
        """Select the next batch of experiments from ``design_space``.

        Refits the surrogate on the current data, then greedily picks
        ``min(batch_size, len(design_space))`` points, adding each selection to
        ``X_pending`` so later picks account for earlier ones.

        Args:
            train_x: Observed inputs, shape (n_train, d).
            train_y: Observed objectives, shape (n_train, n_objectives).
            design_space: Candidate pool to select from, shape (n_candidates, d).

        Returns:
            list[int]: Row indices into ``design_space`` of the selected points,
            in selection order. ``train_master`` maps these to global dataset
            indices via ``dm.heldout_indices``.
        """
        # getting the acquisition function class from the config file argument
        acqf_class_path = str(self.acq_function_config.get("class_path", ""))
        acqf_class = ACQF_REGISTRY.get(acqf_class_path) or ACQF_REGISTRY.get(acqf_class_path.split(".")[-1])
        if acqf_class is None:
            raise ValueError(f"Unknown acquisition class_path {acqf_class_path!r}. "
                            f"Known: {sorted(ACQF_REGISTRY)}")

        if acqf_class is RandomAcqf:
            indices = torch.randperm(len(design_space), device=self.tkwargs["device"])[:self.batch_size].tolist()
            self._last_selected_indices = indices
            return indices # random baseline skips model fitting

        logger.info("Re-training surrogate model with new data...")
        self.train_surrogate_model(train_x, train_y)

        if self._shared_sampler is None:
            self._shared_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        # getting featurizer from finetuned model, defining featurizer, selected X, and possible choices
        featurizer = getattr(self.surrogate_model, "finetuning_model", None)
        if featurizer is None:
            z_baseline = train_x.to(**self.tkwargs)
            z_choices  = design_space.to(**self.tkwargs)
        else:
            featurizer = featurizer.eval()
            with torch.no_grad():
                z_baseline = featurizer(train_x.to(**self.tkwargs))
                z_choices  = featurizer(design_space.to(**self.tkwargs))

        acq_name = acqf_class.__name__
        acq_args = {}

        if "Hypervolume" in acq_name or "EHVI" in acq_name:
            ref_point_arg = (self.acq_function_config.get("init_args") or {}).get("ref_point")
            if ref_point_arg is not None:
                ref_point = torch.as_tensor(ref_point_arg, **self.tkwargs)
            else:
                # Component-wise nadir of the observations so far
                ref_point = train_y.to(**self.tkwargs).min(dim=0).values - 1e-6
            acq_args["ref_point"] = ref_point
            acq_args["incremental_nehvi"] = True

        if acqf_class is MultiObjectiveUCB:
            acq_args = dict(self.acq_function_config.get("init_args") or {})
            acq_args["batch_size"] = self.batch_size
            MultiObjectiveUCB.reset_diverse_counter()

        selected_indices = []

        # train data points are dropped before evaluation in train_master.py
        choices_idx = torch.arange(design_space.size(0), device=design_space.device)

        eval_matrix = z_choices.clone() # features of the design space
        eval2global = choices_idx.clone() # index of design space from 0 to n

        steps = min(self.batch_size, eval_matrix.size(0))
        pending = torch.empty(0, eval_matrix.size(-1), **self.tkwargs)

        for step in tqdm(range(steps), desc="Greedy Batch Selection"):
            logger.debug(
                "[step %d/%d] scoring %d remaining candidates.",
                step + 1, steps, eval_matrix.size(0),
            )

            acqf_step = acqf_class(
                model=self.surrogate_model.model,
                X_baseline=z_baseline,
                X_pending=pending,
                sampler=self._shared_sampler,
                prune_baseline=True,
                **acq_args # contains ref_point if needed
            ).to(**self.tkwargs)
            acqf_step.eval()

            acq_vals = batched_acqf_forward(
                acqf=acqf_step,
                X=eval_matrix,
                max_eval_batch=self.max_eval_chunk
            ).to(**self.tkwargs)

            best_idx_local = int(acq_vals.argmax().item()) # best idx within eval_matrix
            global_idx = int(eval2global[best_idx_local].item()) # get idx of candidate within global design_space

            selected_indices.append(global_idx)

            # update search space and indexing to drop selected point
            keep_mask = (eval2global != global_idx)
            eval_matrix = eval_matrix[keep_mask]
            eval2global = eval2global[keep_mask]

            best_candidate_z = z_choices[global_idx]
            pending = torch.cat([pending, best_candidate_z.unsqueeze(0)], dim=0)

        self._last_selected_indices = selected_indices
        return selected_indices


def setup_bo_optimizer(config, dtype, batch_size=None, device=None, max_eval_chunk=2048):
    """Build the optimizer with explicit precision and optional batch overrides."""
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return BotorchOptimizer(
        surrogate_model_config=config["surrogate_model"],
        acq_function_config=config["acquisition"],
        batch_size=config["bo"]["init_args"].get("batch_size", 1) if batch_size is None else batch_size,
        max_eval_chunk=max_eval_chunk,
        tkwargs={"device": device, "dtype": dtype},
    )


class BayesianOptimization:
    """Set up a dataset once and return primary/reserve row IDs for each batch."""

    def __init__(self, config, device=None):
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config["bo"]["init_args"].get("batch_size", 5)
        self.reserve_size = config.get("reserve_size", 6)
        self.data = None
        featurizer = config["data"]["init_args"]["featurizer"]["init_args"]
        model_name = featurizer.get("model_name")
        if model_name in MODEL_EMBEDDING_SIZES:
            embedding_size = MODEL_EMBEDDING_SIZES[model_name]
            surrogate = config.get("surrogate_model", {}).get("init_args", {})
            if "finetuning_model" in surrogate:
                current = surrogate["finetuning_model"]["init_args"].get("input_dim")
                if current != embedding_size:
                    print(f"[CONFIG] Updating input_dim {current} → {embedding_size}")
                    surrogate["finetuning_model"]["init_args"]["input_dim"] = embedding_size
        acquisition = config['acquisition']['class_path']
        if acquisition not in ACQF_REGISTRY and acquisition.split('.')[-1] not in ACQF_REGISTRY:
            raise ValueError(f'[recipe] Unknown acquisition class_path {acquisition!r}')

    def setup(self):
        seed_everything(self.config.get("seed", 1), workers=True)
        self.data, self.dtype = setup_data(self.config, device=self.device)
        self.pool_mode = hasattr(self.data, 'observed_pool_indices')
        acquisition = self.config.get("acquisition", {}).get("init_args", {})
        if "ref_point" in acquisition:
            acquisition["ref_point"] = torch.as_tensor(acquisition["ref_point"], dtype=self.dtype, device=self.device)
        self.optimizer = setup_bo_optimizer(self.config, self.dtype,
            batch_size=self.batch_size + self.reserve_size, device=self.device,
            max_eval_chunk=2048) # keep the acquisition scoring chunk fixed
        return self

    def suggest(self):
        if self.data is None:
            raise RuntimeError('[BayesianOptimization] Call setup before requesting a batch')
        data = self.data
        if not len(data.heldout_indices):
            raise ValueError('[suggestions] No unmeasured candidates remain for the next iteration')
        train_x = data.train_x.clone().to(self.device, dtype=self.dtype)
        train_y = data.train_y.clone().to(self.device, dtype=self.dtype)
        design_space = data.heldout_x.clone().to(self.device, dtype=self.dtype)
        selected = self.optimizer.suggest_next_experiments(train_x, train_y, design_space)
        positions = validate_indices(selected, len(data.heldout_indices))
        row_ids = np.asarray(data.heldout_indices)[positions]
        unique_ids, unique_positions = np.unique(row_ids, return_index=True) # retain sorted row ordering before the split
        primary_ids = unique_ids[:self.batch_size]
        reserve_ids = unique_ids[self.batch_size:]
        selected_positions = positions[unique_positions]
        data.heldout_indices = np.delete(np.asarray(data.heldout_indices), selected_positions).astype(int)
        if not self.pool_mode:
            data.train_indexes = np.concatenate([np.asarray(data.train_indexes), primary_ids]).astype(int)
            data.train_x, data.train_y = data.x[data.train_indexes], data.y[data.train_indexes]
        data.heldout_x, data.heldout_y = data.x[data.heldout_indices], data.y[data.heldout_indices]
        return primary_ids, reserve_ids
