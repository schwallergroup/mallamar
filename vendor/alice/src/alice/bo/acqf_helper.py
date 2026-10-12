# Modified for Mallamar campaign compatibility.
# Acquisition scores and candidate selection for multi-objective BO.
from collections.abc import Callable
from typing import List

import torch
from torch import Tensor
import gpytorch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
    qLogExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.models.model import Model
from botorch.posteriors.fully_bayesian import MCMC_DIM
from botorch.sampling.base import MCSampler
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import is_ensemble



class qNParEGO(qNoisyExpectedImprovement, MultiObjectiveMCAcquisitionFunction):
    """Original non-log qNParEGO using qNEI after Chebyshev scalarization.

    BoTorch 0.13 exposes only the numerically improved ``qLogNParEGO`` helper.
    This class mirrors that helper exactly through construction of the
    composite objective, then delegates to the original non-log qNEI.  It is
    included solely for the requested log-vs-non-log acquisition comparison.
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        scalarization_weights: Tensor | None = None,
        sampler: MCSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        eta: Tensor | float = 1e-3,
        prune_baseline: bool = False,
        cache_root: bool = True,
    ) -> None:
        MultiObjectiveMCAcquisitionFunction.__init__(
            self,
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
        )
        original_objective = self.objective
        with torch.no_grad():
            baseline_y = original_objective(model.posterior(X_baseline).mean)
        if is_ensemble(model):
            baseline_y = torch.mean(baseline_y, dim=MCMC_DIM)
        scalarization_weights = (
            scalarization_weights
            if scalarization_weights is not None
            else sample_simplex(
                d=baseline_y.shape[-1],
                device=X_baseline.device,
                dtype=X_baseline.dtype,
            ).view(-1)
        )
        scalarization = get_chebyshev_scalarization(
            weights=scalarization_weights,
            Y=baseline_y,
        )
        composite_objective = GenericMCObjective(
            objective=lambda samples, X=None: scalarization(
                original_objective(samples=samples, X=X), X=X
            )
        )
        qNoisyExpectedImprovement.__init__(
            self,
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=composite_objective,
            X_pending=X_pending,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
            constraints=constraints,
            eta=eta,
        )
        self._org_objective = original_objective
        self.chebyshev_scalarization = scalarization
        self.scalarization_weights = scalarization_weights
        self.Y_baseline = baseline_y


class RandomAcqf(AcquisitionFunction):
    "Random acquisition function baseline"
    def __init__(self, model=None, **kwargs):
        super().__init__(model)

    def forward(self, X): #! do we need this? I don't think we need to make RandomAcqf a class object
        return torch.rand(X.shape[:-2], device=X.device, dtype=X.dtype)


class MultiObjectiveUCB(AcquisitionFunction):
    """Scalarized multi-objective Upper Confidence Bound acquisition function.

    For each candidate x and each objective i, computes:
        UCB_i(x) = mu_i(x) + beta * sigma_i(x)

    Then aggregates across objectives with a weighted sum:
        score(x) = sum_i [ w_i * UCB_i(x) ]

    Three weight strategies:
        equal   — fixed uniform weights [1/n, ..., 1/n] for all candidates
        random  — weights sampled once at construction from Uniform(simplex);
                  re-instantiated each greedy step so each step gets fresh weights
        diverse — class-level step counter sweeps weights linearly from [1,0] to
                  [0,1] across the batch_size greedy steps, ensuring each step
                  optimises a different objective trade-off

    Args:
        model:           BoTorch model (must support model.posterior(X))
        beta:            UCB exploration parameter (default 2.0)
        weight_strategy: "equal" | "random" | "diverse" (default "diverse")
        num_objectives:  number of objectives (default 2)
        batch_size:      total greedy steps; only used by "diverse" (default 24)
        **kwargs:        absorbs X_baseline, X_pending, sampler, prune_baseline
                         passed by BotorchOptimizer for compatibility
    """

    # Class-level counter used by "diverse" strategy to track which greedy step
    # is currently being constructed.  Reset to 0 at the start of each batch.
    _diverse_step: int = 0

    @classmethod
    def reset_diverse_counter(cls) -> None:
        cls._diverse_step = 0

    def __init__(
        self,
        model,
        beta: float = 2.0,
        weight_strategy: str = "diverse",
        num_objectives: int = 2,
        batch_size: int = 24,
        **kwargs,          # absorb X_baseline, X_pending, sampler, prune_baseline
    ):
        super().__init__(model)
        self.beta            = beta
        self.weight_strategy = weight_strategy
        self.num_objectives  = num_objectives

        if weight_strategy == "equal":
            w = torch.ones(num_objectives) / num_objectives

        elif weight_strategy == "random":
            w = torch.rand(num_objectives)
            w = w / w.sum()

        elif weight_strategy == "diverse":
            total = max(batch_size - 1, 1)
            t = MultiObjectiveUCB._diverse_step / total          # 0 → 1
            MultiObjectiveUCB._diverse_step += 1
            # linearly interpolate: step 0 → [1, 0], last step → [0, 1]
            if num_objectives == 2:
                w = torch.tensor([1.0 - t, t])
            else:
                # generalise: concentrate weight on objectives in round-robin order
                w = torch.ones(num_objectives) / num_objectives
                focus = MultiObjectiveUCB._diverse_step % num_objectives
                w[focus] += t * (1.0 - 1.0 / num_objectives)
                w = w / w.sum()
        else:
            raise ValueError(
                f"Unknown weight_strategy '{weight_strategy}'. "
                "Choose from: 'equal', 'random', 'diverse'."
            )

        self.register_buffer("weights", w)
        print(
            f"[MultiObjectiveUCB] strategy={weight_strategy}  "
            f"beta={beta}  weights={w.tolist()}"
        )

    def forward(self, X: Tensor) -> Tensor:
        """Score candidates.

        Args:
            X: (batch, q, d) — typically (n_candidates, 1, d) from
               batched_acqf_forward which unsqueezes before calling.

        Returns:
            scores: (batch,) scalar score per candidate
        """
        posterior = self.model.posterior(X)
        mean = posterior.mean        # (batch, q, num_objectives)
        std  = posterior.variance.clamp_min(1e-9).sqrt()

        ucb = mean + self.beta * std  # (batch, q, num_objectives)

        # aggregate over q dimension (q=1 in greedy mode) then over objectives
        ucb_flat = ucb.squeeze(-2)    # (batch, num_objectives)

        w = self.weights.to(dtype=ucb_flat.dtype, device=ucb_flat.device)
        scores = (w * ucb_flat).sum(dim=-1)   # (batch,)

        return scores


def select_random_candidates(
    design_space: Tensor,
    batch_size: int,
    device: torch.device=None
) -> tuple[List[Tensor], List[int]]:
    """
    Select random candidates from design space.
    
    Args:
        design_space: Full design space tensor (n_candidates, n_features)
        batch_size: Number of random candidates to select
        device: Device to use
    
    Returns:
        candidates: List of selected candidate tensors
        indices: List of selected indices in design_space
    """
    n = design_space.shape[0]
    q = batch_size

    # Random permutation to select batch size q indices
    idx = torch.randperm(n, device=device)[:q].tolist()

    # Extract candidates
    candidates = [design_space[i].to(device).reshape(-1) for i in idx]

    print(f"[RandomAcqf] Selected {q} random candidates from {n} total")
    print(f"[RandomAcqf] Indices: {idx[:min(10, q)]}" + 
          (f" ... ({q} total)" if q > 10 else ""))

    return candidates, idx


ACQF_REGISTRY = {
    "botorch.acquisition.multi_objective.qExpectedHypervolumeImprovement": qExpectedHypervolumeImprovement,
    "qExpectedHypervolumeImprovement": qExpectedHypervolumeImprovement,
    "botorch.acquisition.multi_objective.qNoisyExpectedHypervolumeImprovement": qNoisyExpectedHypervolumeImprovement,
    "qNoisyExpectedHypervolumeImprovement": qNoisyExpectedHypervolumeImprovement,
    "botorch.acquisition.multi_objective.qLogNoisyExpectedHypervolumeImprovement": qLogNoisyExpectedHypervolumeImprovement,
    "qLogNoisyExpectedHypervolumeImprovement": qLogNoisyExpectedHypervolumeImprovement,
    "botorch.acquisition.multi_objective.qLogExpectedHypervolumeImprovement": qLogExpectedHypervolumeImprovement,
    "qLogExpectedHypervolumeImprovement": qLogExpectedHypervolumeImprovement,
    "botorch.acquisition.multi_objective.parego.qLogNParEGO": qLogNParEGO,
    "qLogNParEGO": qLogNParEGO,
    "alice.bo.acqf_helper.qNParEGO": qNParEGO,
    "qNParEGO": qNParEGO,
    "RandomAcqf": RandomAcqf,
    "random": RandomAcqf,
    "alice.bo.acqf_helper.MultiObjectiveUCB": MultiObjectiveUCB,
    "MultiObjectiveUCB": MultiObjectiveUCB,
}

def batched_acqf_forward(
    acqf: AcquisitionFunction,
    X: Tensor,
    max_eval_batch: int=2048
) -> Tensor:
    """
    Evaluates an acquisition function on a set of candidates X in memory-safe batches.

    Splits a large tensor of candidate points `X` into smaller mini-batches,
    scoring each batch sequentially, and then concatenating the results.

    Args:
        acqf: The acquisition function to evaluate. Must be a callable
              that accepts a tensor of candidates.
        X: A tensor of candidate points to be scored.
        max_eval_batch: The number of candidates to evaluate in each batch.

    Returns:
        A tensor containing the acquisition score for each candidate in `X`
    """
    scores = []
    n_total = X.shape[0]
    n_batches = (n_total + max_eval_batch - 1) // max_eval_batch
    device = X.device
    for i in range(n_batches):
        start = i * max_eval_batch
        end = min((i + 1) * max_eval_batch, n_total)
        x_batch = X[start:end].unsqueeze(1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
            s = acqf(x_batch).squeeze(-1).detach().cpu()
        scores.append(s)

    return torch.cat(scores, dim=0).to(device)
