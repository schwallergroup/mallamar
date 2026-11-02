# Score robot results and synchronize measured rows into the candidate pool.
from __future__ import annotations
import json, logging, numpy as np, pandas as pd
from pathlib import Path
from agent_lg.services.storage import file_revision, locked_dataset, read_csv, write_csv_atomic, write_json_atomic

logger = logging.getLogger(__name__)

TIME_CAP = 3400.0

# Raw metric columns the robot must write for each finished experiment.
RAW_COLS = [
    "interp_period_mean_s", "interp_period_std_s",
    "interp_maxima_mean_V", "interp_maxima_std_V",
    "interp_amplitude_mean_V", "interp_amplitude_std_V",
    "total_time_s", "n_osc",
]

# Distinguish chemistry failures, hardware failures and missing results; successful rows have no failure label.
FAILURE_MODE_COL = "failure_mode"
FAILURE_CHEMISTRY = "chemistry" # ran, did not oscillate, real evidence
FAILURE_HW = "hw_failed" # robot reported state=failed, NOT chemistry evidence
FAILURE_NO_RESULT = "no_result" # reported finished but no row ever appeared — unknown

# Normalize five metrics, using std/mean for period variability.
_ZCOLS = [
    "interp_period_mean_s", "period_cv",
    "total_time_capped_s", "interp_maxima_mean_V", "interp_amplitude_mean_V",
]

# Use these coefficients for both reward scoring and failure penalties.
_COEF_OSC = (-22.6, -2.4) # period_mean, period_cv
_COEF_SIG = (100.0, 3.8, 2.4) # total_time_capped, maxima_mean, amplitude_mean
_NORM_OSC = sum(abs(c) for c in _COEF_OSC)
_NORM_SIG = sum(abs(c) for c in _COEF_SIG)

REWARD_COLS = ["reward_oscillation", "reward_signal"]

# Match robot results to pool rows using concentrations rounded to the grid precision.
CONC_COLS = ["conc_MA_M", "conc_NaBrO3_M", "conc_H2SO4_M", "conc_Ru_M"]
CONC_ROUND = {"conc_MA_M": 4, "conc_NaBrO3_M": 4, "conc_H2SO4_M": 4, "conc_Ru_M": 7}

# Assign both objectives a fixed -5 penalty when an attempted experiment has no oscillations.
_FAIL_PENALTY_OSC = sum(-5.0 * abs(c) for c in _COEF_OSC) / _NORM_OSC
_FAIL_PENALTY_SIG = sum(-5.0 * abs(c) for c in _COEF_SIG) / _NORM_SIG


def _attempted(df: pd.DataFrame):
    """Rows attempted by the robot, including failed experiments."""
    if "exp_id" not in df.columns:
        return pd.Series(False, index=df.index)
    s = df["exp_id"].astype(str)
    return df["exp_id"].notna() & ~s.isin(("", "nan", "NaN", "None"))


def _apply_failure_penalty(df: pd.DataFrame, attempted: pd.Series):
    """Apply the configured penalty to failed experiment rows."""
    failed = attempted & (pd.to_numeric(df["n_osc"], errors="coerce").fillna(-1) == 0)

    # Label unlabelled failures as chemistry failures while preserving hardware and missing-result labels.
    if FAILURE_MODE_COL not in df.columns:
        df[FAILURE_MODE_COL] = pd.NA
    df[FAILURE_MODE_COL] = df[FAILURE_MODE_COL].astype("object")
    unlabelled = failed & df[FAILURE_MODE_COL].isna()
    if unlabelled.any():
        df.loc[unlabelled, FAILURE_MODE_COL] = FAILURE_CHEMISTRY

    needs_penalty = failed & (df["reward_oscillation"].isna() | df["reward_signal"].isna())
    n = int(needs_penalty.sum())
    if n == 0:
        return 0
    df.loc[needs_penalty, "reward_oscillation"] = _FAIL_PENALTY_OSC
    df.loc[needs_penalty, "reward_signal"] = _FAIL_PENALTY_SIG
    return n


def _derive(df: pd.DataFrame):
    """Add the capped-time and coefficient-of-variation intermediates."""
    d = df.copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    d["total_time_capped_s"] = np.minimum(num("total_time_s"), TIME_CAP)
    d["period_cv"] = num("interp_period_std_s") / num("interp_period_mean_s").replace(0, np.nan)
    return d


def fit_reward_norm(df_observed: pd.DataFrame):
    """Fit normalization statistics on the supplied measured rows."""
    missing = [c for c in RAW_COLS if c not in df_observed.columns]
    if missing:
        raise KeyError(f"reference set is missing raw metric columns: {missing}")
    successful = pd.to_numeric(df_observed["n_osc"], errors="coerce") > 0
    d = _derive(df_observed.loc[successful])
    if len(d) < 2:
        raise ValueError('[rewards] Normalization needs at least two successful measurements; recorded results are preserved')
    values = d[_ZCOLS].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('[rewards] Successful measurements contain missing or non-finite reward metrics')
    norm = {}
    for column in _ZCOLS:
        stats = [float(values[column].mean()), float(values[column].std())]
        if not np.isfinite(stats).all():
            raise ValueError('[rewards] Cannot fit finite normalization statistics; inspect the successful raw metrics')
        norm[column] = stats
    return norm


def rewards_from_raw(df: pd.DataFrame, norm: dict):
    """Score raw robot metrics with the supplied normalization."""
    d = _derive(df)

    def z(col: str):
        mean, std = norm[col]
        if not std or std <= 0:
            return pd.Series(0.0, index=d.index)
        return (d[col] - mean) / std

    osc = (_COEF_OSC[0] * z("interp_period_mean_s")
           + _COEF_OSC[1] * z("period_cv")) / _NORM_OSC
    sig = (_COEF_SIG[0] * z("total_time_capped_s")
           + _COEF_SIG[1] * z("interp_maxima_mean_V")
           + _COEF_SIG[2] * z("interp_amplitude_mean_V")) / _NORM_SIG

    failed = pd.to_numeric(d["n_osc"], errors="coerce").fillna(0) == 0
    d["reward_oscillation"] = osc.mask(failed, np.nan)
    d["reward_signal"] = sig.mask(failed, np.nan)
    return d


@locked_dataset
def recompute_all_rewards(db_path):
    """Refit normalization and rescore all attempted rows after each batch."""
    revision = file_revision(db_path)
    df = read_csv(db_path)
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        # Leave rewards unchanged when raw metric columns are missing.
        print(f"[rewards] skip recompute — DB missing raw metric columns {missing}")
        return 0

    attempted = _attempted(df)
    n = int(attempted.sum())
    if n == 0:
        return 0

    # Failed runs may retain finite placeholder metrics; exclude them from normalization.
    osc_rows = attempted & (pd.to_numeric(df["n_osc"], errors="coerce") > 0)
    for col in REWARD_COLS:
        if col not in df.columns:
            df[col] = np.nan

    if osc_rows.any():
        norm = fit_reward_norm(df.loc[osc_rows]) # z-norm over successful only
        scored = rewards_from_raw(df.loc[attempted], norm) # score success + fail
        df.loc[attempted, REWARD_COLS] = scored[REWARD_COLS].values
    # Without successful measurements, apply failure penalties without fitting normalization.

    # Replace failed rows' missing rewards with the fixed penalty.
    _apply_failure_penalty(df, attempted)

    write_csv_atomic(df, db_path, revision)
    return n


def _rounded_concentrations(values):
    return tuple(round(float(value), CONC_ROUND[column]) for column, value in zip(CONC_COLS, values))


def _conc_key(row):
    return _rounded_concentrations(row[column] for column in CONC_COLS)


def concentration_keys(frame):
    # Keep Python rounding because NumPy can differ at half-grid values.
    return [_rounded_concentrations(values)
            for values in frame[CONC_COLS].itertuples(index=False, name=None)]


@locked_dataset
def sync_results_into_pool(pool_path, results_path):
    """Match robot results to pool rows by rounded concentrations."""
    revision = file_revision(pool_path)
    pool = read_csv(pool_path)
    res = read_csv(results_path)
    for label, frame in [("pool", pool), ("results", res)]:
        missing = [column for column in CONC_COLS + ["exp_id"] if column not in frame.columns]
        if missing:
            raise ValueError(f"[sync_results_into_pool] {label} missing columns: {missing}")

    # Synchronize all attempted experiments, including failures with missing raw metrics.
    res_attempted = res[_attempted(res)]
    if res_attempted.empty:
        return 0
    res_by_conc = {}
    for _, r in res_attempted.iterrows():
        res_by_conc.setdefault(_conc_key(r), r) # first attempted row per concentration

    # Preserve failure labels when synchronizing results into the pool.
    if FAILURE_MODE_COL in res.columns and FAILURE_MODE_COL not in pool.columns:
        pool[FAILURE_MODE_COL] = pd.NA
    fill_cols = [c for c in RAW_COLS + ["exp_id", FAILURE_MODE_COL]
                 if c in pool.columns and c in res.columns]
    for column in ("exp_id", FAILURE_MODE_COL):
        if column in fill_cols:
            pool[column] = pool[column].astype("object")
    # Skip pool rows that already have an experiment ID.
    pool_synced = _attempted(pool)
    n_filled = 0
    pending = pool.loc[~pool_synced, CONC_COLS]
    for ci, key in zip(pending.index, concentration_keys(pending)):
        rr = res_by_conc.get(key)
        if rr is not None:
            for c in fill_cols:
                pool.at[ci, c] = rr[c]
            n_filled += 1

    if n_filled:
        write_csv_atomic(pool, pool_path, revision)
    return n_filled


def save_norm(norm: dict, path):
    write_json_atomic(path, norm)


def load_norm(path):
    norm = json.loads(Path(path).read_text())
    try:
        for column in _ZCOLS:
            values = np.asarray(norm[column], dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or values[1] < 0:
                raise ValueError(column)
            norm[column] = values.tolist()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('[rewards] Invalid saved normalization; restore finite means and non-negative standard deviations before resuming') from error
    return norm


@locked_dataset
def score_with_frozen(db_path, norm_path):
    """Score new measurements using saved normalization statistics."""
    revision = file_revision(db_path)
    df = read_csv(db_path)
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        print(f"[rewards] skip freeze-score — DB missing raw metric columns {missing}")
        return 0
    for col in REWARD_COLS:
        if col not in df.columns:
            df[col] = np.nan

    attempted = _attempted(df)
    if not attempted.any():
        return 0
    osc_rows = attempted & (pd.to_numeric(df["n_osc"], errors="coerce") > 0)

    norm_path = Path(norm_path)
    if norm_path.exists():
        norm = load_norm(norm_path) # reuse the frozen constants
    elif osc_rows.any():
        norm = fit_reward_norm(df.loc[osc_rows]) # fit ONCE on first batch's successes
        save_norm(norm, norm_path)
        print(f"[rewards] froze z-norm on {int(osc_rows.sum())} oscillating rows → {norm_path}")
    else:
        # Defer normalization until successful measurements are available.
        norm = None

    n_scored = 0
    if norm is not None:
        # Score only successful rows that do not already have rewards.
        new_osc = osc_rows & df["reward_oscillation"].isna()
        if new_osc.any():
            scored = rewards_from_raw(df.loc[new_osc], norm)
            df.loc[new_osc, REWARD_COLS] = scored[REWARD_COLS].values
            n_scored = int(new_osc.sum())

    n_penalized = _apply_failure_penalty(df, attempted)

    if n_scored or n_penalized:
        write_csv_atomic(df, db_path, revision)
    return n_scored
