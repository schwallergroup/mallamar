# Check campaign inputs before enabling agent features.

from __future__ import annotations
import os, logging, argparse
from pathlib import Path
from agent_lg.config import CONC_COLS
from agent_lg.policy import resolve_improve_signal

from agent_lg.services.storage import read_csv
from agent_lg.services.literature import LocalCorpus
from agent_lg.services.validation import CampaignSafetyError
from agent_lg.services.method import CampaignRecipes
from agent_lg.tools.hypothesis import store as hypothesis_store

logger = logging.getLogger(__name__)

class PreflightError(RuntimeError):
    """An agentic axis is enabled but cannot work. Raised before anything runs."""


def corpus_dir(dataset: str = "bz"):
    """Return the explicitly configured directory of local papers."""
    return LocalCorpus().path


def _corpus_problems(dataset: str = "bz", warnings: list[str] | None = None):
    corpus = LocalCorpus()
    if corpus.path is None:
        return ["hypothesis mode is ON but OCEAN_CORPUS_DIR is unset. "
                "Set it to your local pdf/txt/md collection (e.g. references/papers), before starting the campaign."]
    if not corpus.files():
        return [f"hypothesis mode is ON but OCEAN_CORPUS_DIR={corpus.path} contains no pdf/txt/md files. "
                "Provide an existing directory containing the campaign's papers."]
    return []


def _pool_problems(db_path: str | None):
    """Reasons candidate injection could not place a point."""
    problems = []
    # The API binds this pool from db_path before checking enabled axes.
    pool = os.getenv("OCEAN_CAMPAIGN_DB")
    if not pool:
        problems.append(
            "injection is ON but OCEAN_CAMPAIGN_DB is unset"
            ".\n"
            "      fix : run with --db <campaign_pool>.csv or call run_agent(db_path=...)\n"
            "      why : proposals are snapped to unmeasured POOL rows; with no pool\n"
            "            propose_experiments returns no_op and every batch is plain BO."
        )
        return problems

    p = Path(pool)
    if not p.exists():
        problems.append(f"injection is ON but the pool does not exist: {p}")
        return problems

    try:
        df = read_csv(p, nrows=5000)
    except Exception as e:
        problems.append(f"injection is ON but the pool could not be read ({p}): "
                        f"{type(e).__name__}: {e}")
        return problems

    missing = [c for c in CONC_COLS if c not in df.columns]
    if missing:
        problems.append(
            f"injection is ON but the pool is missing concentration columns {missing}.\n"
            f"      pool : {p}\n"
            f"      why  : proposals are snapped in concentration space; without these\n"
            f"             columns there is nothing to snap to."
        )
    return problems


def _improve_signal_problems(db_path: str | None):
    """Reasons the Pareto-membership improvement signal could not fire."""
    path = os.getenv("OCEAN_CAMPAIGN_DB") or db_path
    if not path or not Path(path).exists():
        return [] # nothing to check yet; other checks cover a missing pool
    try:
        cols = read_csv(path, nrows=1).columns
    except Exception:
        return [] # unreadable is reported by the pool check, don't double up
    if "exp_id" not in cols:
        return [f"an agentic axis is ON but the dataset has no 'exp_id' column: {path}. "
                "Add experiment identifiers to the campaign pool. Pareto improvement uses the IDs of new measurements. "
                "Set OCEAN_IMPROVE_SIGNAL=hv5pct only if you deliberately want the alternative HV rule."]
    return []


def check_agentic_config(db_path: str | None = None, dataset: str = "bz",
                         verbose: bool = True):
    """Validate the enabled axes. Raises PreflightError listing everything wrong."""
    hyp_on = bool(os.getenv("OCEAN_BO_HYPOTHESIS"))
    inject_on = bool(os.getenv("OCEAN_BO_INJECT"))
    filter_on = bool(os.getenv("OCEAN_BO_FILTER"))
    errors, warnings = [], []
    try:
        CampaignRecipes().check_environment()
        CampaignRecipes().resolve(CampaignRecipes.config_key)
    except ValueError as error:
        errors.append(str(error))

    # Injection implies hypothesis — that is the only form it was validated in.
    if inject_on and not hyp_on:
        warnings.append("injection is ON without hypothesis mode; hypothesis is enabled "
                        "implicitly (proposals are meant to come from hypothesis windows).")
        hyp_on = True

    if hyp_on:
        errors.extend(_corpus_problems(dataset, warnings))
        if os.getenv("OCEAN_CAMPAIGN_DB"):
            try:
                hypothesis_store()
            except CampaignSafetyError as error:
                errors.append(str(error))

    # Only when the pareto signal is actually in play — an explicit
    # OCEAN_IMPROVE_SIGNAL=hv5pct is a deliberate choice, not a misconfiguration.
    if resolve_improve_signal() == "pareto":
        errors.extend(_improve_signal_problems(db_path))

    if inject_on:
        errors.extend(_pool_problems(db_path))
        n_agent = os.getenv("OCEAN_INJECT_N_AGENT")
        if not n_agent:
            errors.append(
                "injection is ON but OCEAN_INJECT_N_AGENT is not set.\n"
                "      fix : set it explicitly (e.g. 6 for a batch of 6 = full agent choice)\n"
                "      why : the injection count is part of the method and must be explicit."
            )
        elif not str(n_agent).isdigit():
            errors.append(f"OCEAN_INJECT_N_AGENT={n_agent!r} is not an integer.")

    if errors:
        raise PreflightError(
            "\n\nPREFLIGHT FAILED — an agentic axis is enabled but cannot work:\n\n"
            + "\n\n".join(f"  [{i}] {e}" for i, e in enumerate(errors, 1))
            + "\n\nNothing was started. Fix the above before starting the campaign.\n"
        )

    if verbose:
        on = [n for n, v in [("filter", filter_on),
                             ("hypothesis", hyp_on), ("injection", inject_on)] if v]
        print(f"[preflight] agentic axes ON: {', '.join(on)}")
        if hyp_on:
            print(f"[preflight] corpus: {corpus_dir(dataset)}")
        for w in warnings:
            print(f"[preflight] WARNING: {w}")
    return warnings


def main():
    ap = argparse.ArgumentParser(description="Validate the enabled agentic axes.")
    ap.add_argument("--db", default=None, help="Campaign pool csv.")
    ap.add_argument("--dataset", default="bz", help="Dataset name for the corpus lookup.")
    args = ap.parse_args()
    try:
        check_agentic_config(db_path=args.db, dataset=args.dataset)
    except PreflightError as e:
        print(str(e))
        return 1
    print("[preflight] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
