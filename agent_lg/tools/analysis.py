# Summarize measured data, coverage and campaign progress.

import os, logging, re, numpy as np, pandas as pd, torch
from pathlib import Path
from agent_lg.services.storage import read_campaign_csv, atomic_write, write_json_atomic
from agent_lg.services.outputs import OutputPaths
from botorch.utils.multi_objective.pareto import is_non_dominated
from alice.initialization.initializers import landscape_roughness, weighted_maximin
from alice.utils.diagnostics import sampling_coverage, suggestion_diversity, hypervolume_2d as _hv_2d
from sklearn.preprocessing import MinMaxScaler
from agent_lg.config import CONC_COLS, HERE, REF_POINT

logger = logging.getLogger(__name__)

def _tool_read_dataset(path: str):
    try:
        df = read_campaign_csv(path)
    except FileNotFoundError:
        # guarding the hallucination here, jsut in case
        campaign = os.getenv("OCEAN_CAMPAIGN_DB")
        if campaign and Path(campaign).exists() and Path(path) != Path(campaign):
            print(f"[read_dataset] path not found: {path!r} — single-pool mode, "
                  f"redirecting to OCEAN_CAMPAIGN_DB={campaign!r}")
            try:
                df = read_campaign_csv(campaign)
            except Exception as e:
                return {"error": f"redirect to {campaign} also failed: {e}"}
        else:
            return {"error": f"FileNotFoundError: {path}"}
    except Exception as e:
        return {"error": str(e)}
    reward_cols = [c for c in df.columns if c.startswith("reward_")]
    info = {"n_rows": len(df), "columns": list(df.columns)}
    if reward_cols:
        valid = df[reward_cols[0]].notna().sum()
        info["n_valid_rewards"] = int(valid)
        info["n_nan_rewards"] = int(len(df) - valid)
        info["reward_stats"] = {
            col: {"min": round(float(df[col].min()), 4), "max": round(float(df[col].max()), 4),
                  "mean": round(float(df[col].mean()), 4), "n_nan": int(df[col].isna().sum())}
            for col in reward_cols
        }
    if "n_osc" in df.columns:
        info["n_osc_zero"] = int((df["n_osc"] == 0).sum()) # keep track of how many rows have n_osc=0
    if "reward_oscillation" in df.columns and "reward_signal" in df.columns:
        df_obs = df.dropna(subset=["reward_oscillation", "reward_signal"])
        df_obs = df_obs.assign(_hv=df_obs["reward_oscillation"] + df_obs["reward_signal"])
        missing_columns = [column for column in CONC_COLS if column not in df.columns]
        if missing_columns:
            return {"error": f"Missing concentration columns: {missing_columns}"}
        top5 = df_obs.nlargest(5, "_hv")[CONC_COLS + ["reward_oscillation", "reward_signal"]].to_dict(orient="records") # current top 5 by combined reward, for quick reference
        info["top5_by_combined_reward"] = top5
    return info


def _tool_analyze_space_coverage(db_path: str):
    """Provides sampling diversity info to Agent"""
    try:
        df = read_campaign_csv(db_path)
    except Exception as e:
        return {"error": str(e)}
    rc = ["reward_oscillation"] if "reward_oscillation" in df.columns else []
    df_obs = df.dropna(subset=rc) if rc else df
    if len(df_obs) < 4:
        return {"error": "Too few observed rows for PCA"}
    X = df_obs[CONC_COLS].values.astype(float)
    variance, counts = sampling_coverage(X)
    return {
        "n_observed": len(df_obs),
        "pca_variance_explained": [round(float(v), 3) for v in variance],
        "pc1_bin_counts": counts.tolist(),
        "concentration_ranges": {col: {"min": round(float(df_obs[col].min()), 4),
                                          "max": round(float(df_obs[col].max()), 4)}
                                    for col in CONC_COLS},
    }


def _tool_estimate_roughness(db_path: str, n_select: int = 24, k: int = 5):
    """Estimate reward landscape roughness and select a diverse subset of observed points"""
    try:
        df = read_campaign_csv(db_path)
    except Exception as e:
        return {"error": str(e)}
    rc = [c for c in ["reward_oscillation", "reward_signal"] if c in df.columns]
    if not rc:
        return {"error": "No reward columns found"}
    df_obs = df.dropna(subset=rc).copy()
    if len(df_obs) < k + 1:
        return {"error": f"Need ≥{k+1} observed rows, got {len(df_obs)}"}
    X = df_obs[CONC_COLS].values.astype(float)
    Y = df_obs[rc].values.astype(float)
    X_scaled = MinMaxScaler().fit_transform(X)
    roughness, distances = landscape_roughness(X_scaled, Y, k)
    weights = roughness / (roughness.max() + 1e-8)
    selected = weighted_maximin(distances, weights, n_select)
    return {
        "n_selected": len(selected),
        "roughness_stats": {"min": round(float(roughness.min()), 4),
                            "max": round(float(roughness.max()), 4),
                            "mean": round(float(roughness.mean()), 4)},
        "selected_rows": df_obs.iloc[selected][CONC_COLS + rc].round(5).to_dict(orient="records"),
        "global_indices": selected,
    }


def _tool_check_suggestion_diversity(suggestions: list):
    """Check diversity of suggestions based on pairwise distances in input space."""
    if not suggestions:
        return {"error": "Empty suggestions list"}
    # Collect all numeric keys, excluding pure-ID columns
    skip = {"rxn_id", "batch_id", "iter", "suggestion_index"}
    all_numeric_keys = [k for k, v in suggestions[0].items()
                        if k not in skip and isinstance(v, (int, float))]
    # Prefer BZ concentration keys when they are present AND carry real variation
    bz_keys = ["MA", "NaBrO3", "H2SO4", "Ru"]
    bz_present = all(k in suggestions[0] for k in bz_keys)
    if bz_present:
        try:
            bz_vals = np.array([[s[k] for k in bz_keys] for s in suggestions], dtype=float)
            bz_has_variation = bz_vals.std(axis=0).max() > 1e-9
        except Exception:
            bz_has_variation = False
        keys = bz_keys if bz_has_variation else [k for k in all_numeric_keys if k not in set(bz_keys)]
    else:
        keys = all_numeric_keys
    # Drop any constant features (no variation across suggestions)
    if keys:
        try:
            X_all = np.array([[s[k] for k in keys] for s in suggestions], dtype=float)
            vary = X_all.std(axis=0) > 1e-9
            keys = [k for k, v in zip(keys, vary) if v]
        except Exception:
            pass
    if not keys:
        return {"note": "All numeric features are constant — cannot compute pairwise distances",
                "n_suggestions": len(suggestions), "verdict": "N/A (no variation in features)"}
    try:
        X = np.array([[s[k] for k in keys] for s in suggestions], dtype=float)
    except (KeyError, TypeError, ValueError) as e:
        return {"error": f"Could not build feature matrix: {e}. Keys tried: {keys}"}
    if len(X) == 1:
        return {
            "n_suggestions": 1,
            "min_pairwise_dist": 0.0,
            "mean_pairwise_dist": 0.0,
            "verdict": "N/A (single suggestion)",
        }
    md, mean_d = suggestion_diversity(X)
    verdict = ("POOR" if md < 0.02 else "MODERATE" if mean_d < 0.15 else "GOOD")
    return {"n_suggestions": len(suggestions),
            "min_pairwise_dist": round(md, 4),
            "mean_pairwise_dist": round(mean_d, 4),
            "verdict": verdict}

def _tool_get_current_pareto_front(db_path: str, skip_first_n: int = 0):
    """Get the current Pareto front from measured points so far"""    
    try:
        df = read_campaign_csv(db_path)
    except Exception as e:
        return {"error": str(e)}
    rc = [c for c in ["reward_oscillation", "reward_signal"] if c in df.columns]
    if not rc:
        return {"n_pareto": 0, "front": [], "note": "No reward columns — run prepare_bo_dataset first."}
    df_obs = df.dropna(subset=rc).iloc[skip_first_n:].reset_index(drop=True)
    if df_obs.empty:
        return {"n_pareto": 0, "n_total_observed": 0, "front": []}
    Y = df_obs[rc].values
    on_front = _pareto_mask(Y)
    pareto = df_obs[on_front][CONC_COLS + rc]
    return {"n_pareto": int(on_front.sum()), "n_total_observed": len(df_obs),
            "skipped_n": skip_first_n, "front": pareto.round(5).to_dict(orient="records")}



def _pareto_mask(points):
    """Keep every non-dominated experiment, including identical reward pairs."""
    return is_non_dominated(torch.as_tensor(points), deduplicate=False).numpy()



def _compute_iteration_hv(db_path: str, batch_ids=None, campaign_ids=None):
    """Read DB, compute current Pareto front and 2D HV. Its a progress tracker used by progress_tracker_node."""
    try:
        df = read_campaign_csv(db_path)
    except Exception as e:
        return {"error": str(e)}
    rc = [c for c in ["reward_oscillation", "reward_signal"] if c in df.columns]
    if not rc:
        return {"n_pareto": 0, "hv_2d": 0.0, "n_observed": 0}
    df_obs = df.dropna(subset=rc)
    if df_obs.empty:
        return {"n_pareto": 0, "hv_2d": 0.0, "n_observed": 0}

    Y = df_obs[rc].values
    on_front = _pareto_mask(Y)
    front = Y[on_front]

    out = {
        "n_observed": int(len(df_obs)),
        "n_pareto": int(on_front.sum()),
        "hv_2d": round(_hv_2d(front, REF_POINT), 4),
        "ref_point": REF_POINT,
    }

    # Which rows are on the front, by exp_id (provenance + the membership test).
    if "exp_id" in df_obs.columns:
        eids = df_obs["exp_id"].astype(str).values
        front_ids = [e for e, keep in zip(eids, on_front)
                     if keep and e not in ("", "nan", "NaN", "None")]
        out["pareto_exp_ids"] = front_ids
        if batch_ids is not None:
            batch = {str(b) for b in batch_ids}
            out["n_new_pareto"] = int(sum(1 for e in front_ids if e in batch))
        if campaign_ids is not None:
            camp = {str(c) for c in campaign_ids}
            mask = np.array([e in camp for e in eids], dtype=bool)
            if mask.any():
                Yc = Y[mask]
                out["hv_campaign"] = round(_hv_2d(Yc[_pareto_mask(Yc)], REF_POINT), 4)
                out["n_observed_campaign"] = int(mask.sum())
            else:
                out["hv_campaign"] = 0.0
                out["n_observed_campaign"] = 0
    elif batch_ids is not None:
        # No exp_id column (legacy combined file) — membership is not decidable.
        out["n_new_pareto"] = None
    return out

def _tool_analyze_final_results(db_path: str, baseline_n_obs: int = 0, **kwargs):
    """Analyze final results: reward stats, failure rate, new hits compared to baseline_n_obs."""
    # Accept common alias variants the agent has been observed to use
    if "baseline_n_observed" in kwargs:
        baseline_n_obs = int(kwargs["baseline_n_observed"])
    try:
        df = read_campaign_csv(db_path)
    except Exception as e:
        return {"error": str(e)}
    rc = [c for c in ["reward_oscillation", "reward_signal"] if c in df.columns]
    df_obs = df.dropna(subset=rc) if rc else df
    new = df_obs.iloc[baseline_n_obs:] if baseline_n_obs < len(df_obs) else df_obs
    fail_r = float((df.get("n_osc", pd.Series([1]*len(df))) == 0).sum()) / max(len(df), 1)
    stats = {"total_observed": len(df_obs), "new_in_this_run": len(new),
              "failure_rate_n_osc0": round(fail_r, 3)}
    if rc and len(new) > 0:
        stats["new_reward_stats"] = {
            col: {"min": round(float(new[col].min()), 3), "max": round(float(new[col].max()), 3),
                  "mean": round(float(new[col].mean()), 3)} for col in rc}
        new2 = new.copy(); new2["_sum"] = new2[rc].sum(axis=1)
        stats["top5_new_hits"] = new2.nlargest(5, "_sum")[CONC_COLS + rc].to_dict(orient="records")
    return stats


def _tool_compare_initializers(db_path: str, n_init: int = 24, k: int = 5,
                                out_path: str = None):
    """Compare different initialization strategies based on observed data."""
    target = OutputPaths().resolve(out_path, (".json",)) if out_path else None
    try:
        df = read_campaign_csv(db_path)
    except Exception as e:
        return {"error": str(e)}
    rc = [c for c in ["reward_oscillation", "reward_signal"] if c in df.columns]
    if not rc:
        return {"error": "No reward columns found"}
    df_obs = df.dropna(subset=rc).copy()
    n_obs = len(df_obs)
    if n_obs < k + 1:
        return {"error": f"Need ≥{k+1} observed rows"}
    X = df_obs[CONC_COLS].values.astype(float)
    Y = df_obs[rc].values.astype(float)
    X_scaled = MinMaxScaler().fit_transform(X)
    roughness, distances = landscape_roughness(X_scaled, Y, k)
    weights = roughness / (roughness.max() + 1e-8)
    rough_sel = weighted_maximin(distances, weights, n_init)
    np.random.seed(42)
    sobol_sel = weighted_maximin(distances, np.ones(n_obs), n_init)
    overlap = set(rough_sel) & set(sobol_sel)
    result = {
        "n_observed": n_obs, "n_init": n_init, "overlap_count": len(overlap),
        "roughness_stats": {
            "all_mean": round(float(roughness.mean()), 2),
            "roughness_init_mean": round(float(roughness[rough_sel].mean()), 2),
            "sobol_init_mean": round(float(roughness[sobol_sel].mean()), 2),
        },
        "interpretation": (
            f"Roughness init selects points with mean roughness "
            f"{roughness[rough_sel].mean():.0f} vs {roughness[sobol_sel].mean():.0f} for Sobol "
            f"({roughness[rough_sel].mean()/max(roughness[sobol_sel].mean(),1):.1f}x higher). "
            f"Only {len(overlap)}/{n_init} points overlap."
        ),
    }
    if out_path:
        write_json_atomic(target, result)
        result["report_saved"] = out_path
    return result


def _reports_dir():
    """Use the mailbox report directory, or the caller's report workspace."""
    root = os.getenv("OCEAN_MAILBOX_ROOT")
    return ((Path(root) / "reports") if root else (HERE / "reports")).expanduser().resolve()


def _tool_write_report(content: str, out_path: str = None):
    """Write a report atomically, redirecting attempts to overwrite campaign inputs."""
    reports = _reports_dir()
    policy = OutputPaths()
    requested = Path(out_path).expanduser() if out_path is not None else reports / "report.md"
    reason = None
    try:
        if requested.exists() and requested.suffix.lower() not in (".md", ".markdown"):
            raise ValueError(f"{requested.name} already exists and is not a report")
        target = policy.resolve(requested if requested.suffix else requested.with_suffix(".md"),
                                (".md", ".markdown"))
    except (OSError, RuntimeError, ValueError) as error:
        reason = str(error)
        safe_name = re.sub(r"[^\w.-]", "_", requested.name) or "report"
        if not safe_name.endswith(".md"):
            safe_name += ".md"
        target = policy.resolve(reports / f"redirected_{safe_name}", (".md",))
    atomic_write(target, lambda temporary: temporary.write_text(content, encoding="utf-8"))
    result = {"written_to": str(target), "n_chars": len(content)}
    if reason:
        message = (f"REFUSED to write the report to {out_path!r}: {reason}. "
                   f"Wrote it to {target} instead. Reports go in {reports}.")
        print(f"[write_report] {message}")
        result.update(redirected=True, warning=message)
    return result
