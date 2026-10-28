# Agent tools for BO configuration, dataset preparation and batch selection.
import json, os, subprocess, sys, logging, tempfile, numpy as np, pandas as pd
from pathlib import Path
from agent_lg.services.bo_process import BOProcessRunner
from agent_lg.config import AVAILABLE_CONFIGS, CONC_COLS, HERE

from agent_lg.services.storage import read_csv, read_campaign_csv, write_json_atomic, write_csv_atomic, file_revision
from agent_lg.services.protocols import CONCENTRATION_ALIASES, load_protocol, select_protocol, validate_protocol
from agent_lg.services.pools import check_campaign_pool
from agent_lg.services.method import CampaignRecipes, CampaignMethod
from agent_lg.services.outputs import OutputPaths
from agent_lg.services.validation import integer_input, positive_input
from alice.data.utils import validate_indices, NearestCandidates
from alice.initialization.initializers import select_sobol_candidates

logger = logging.getLogger(__name__)


def _n_observed(data_path):
    """Number of measured rows in a campaign pool (non-NaN reward_oscillation)."""
    try:
        df = read_csv(data_path)
        if "reward_oscillation" not in df.columns:
            return -1 # not a campaign pool (legacy combined) — caller skips cold start
        return int(pd.to_numeric(df["reward_oscillation"], errors="coerce").notna().sum())
    except Exception:
        return -1


def _cold_start_sobol(data_path, n_init: int, seed: int):
    """Cold start (zero observed): pick n_init space-filling candidates from the pool."""

    integer_input(n_init, 'n_init')
    integer_input(seed, 'seed', minimum=0, maximum=2**32 - 1)
    df = read_campaign_csv(data_path)
    cand = df[pd.to_numeric(df["reward_oscillation"], errors="coerce").isna()].reset_index(drop=True)
    X = cand[CONC_COLS].to_numpy(dtype=float)
    chosen = select_sobol_candidates(X, n_init, seed)

    suggestions = [{"conc": {"MA": float(r.conc_MA_M), "NaBrO3": float(r.conc_NaBrO3_M),
                             "H2SO4": float(r.conc_H2SO4_M), "Ru(bpy)3": float(r.conc_Ru_M)}}
                   for _, r in cand.iloc[chosen].iterrows()]

    out_dir = OutputPaths().resolve(_results_root() / 'init' / 'suggestions_init.json', ('.json',)).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(tempfile.mkdtemp(prefix='batch-', dir=out_dir)) / 'suggestions_init.json'
    write_json_atomic(json_path, [{"iter": 0, "suggestions": suggestions, "reserve": []}])

    primaries = [{"path": f"init_sobol_{i+1}", "conc": s["conc"]} for i, s in enumerate(suggestions)]
    print(f"[run_bo_iteration] COLD START — sobol-picked {len(primaries)} init candidates from the pool.")
    return {"returncode": 0, "n_primary": len(primaries), "n_reserve": 0,
            "primaries": primaries, "reserves": [], "json_path": str(json_path),
            "stderr": "", "cold_start": True}


def _results_root():
    """Where BO writes suggestion JSONs and protocols."""
    root = os.getenv("OCEAN_RESULTS_ROOT")
    return (Path(root) if root else HERE / 'results_MOBBOL').expanduser().resolve()


def _tool_list_available_configs():
    return {k: {**v, "exists": Path(v["path"]).exists()} for k, v in AVAILABLE_CONFIGS.items()}


def _tool_prepare_bo_dataset(prepare_py: str = None):
    # Single-pool campaigns score measured rows in place; candidates keep NaN rewards.
    # Rebuilding a separate dataset would discard this campaign state.
    campaign_db = os.getenv("OCEAN_CAMPAIGN_DB")
    if campaign_db:
        return {"returncode": 0,
                "stdout": (f"single-pool mode: {campaign_db} is the dataset "
                           f"(measured rows are scored in place, candidates stay NaN) "
                           f"— no regeneration needed"),
                "stderr": ""}

    if not prepare_py:
        return {"returncode": 1, "stdout": "",
                "stderr": "No campaign CSV configured. Pass --db to the campaign CLI or db_path to run_agent."}
    p = Path(prepare_py)
    if not p.is_absolute():
        p = HERE / p
    result = subprocess.run([sys.executable, str(p), "--include_candidates"],
                            capture_output=True, text=True, cwd=str(HERE))
    return {"returncode": result.returncode, "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-500:] if result.returncode != 0 else ""}


def _tool_run_bo_iteration(config: str, volume_uL: float):
    try:
        positive_input(volume_uL, 'volume_uL')
    except ValueError as error:
        return {'error': str(error), 'abort': True, 'step': 'bo_inputs'}
    config_key = CampaignRecipes.config_key
    try:
        CampaignRecipes().check_environment()
        config = str(CampaignRecipes().resolve(config))
    except ValueError as error:
        return {'error': str(error), 'valid_configs': [config_key]}
    # The campaign runner restores this explicit pool path on restart.
    campaign_db = os.getenv("OCEAN_CAMPAIGN_DB")
    if not campaign_db:
        return {"error": "No campaign CSV configured. Pass --db to the campaign CLI or db_path to run_agent.",
                "abort": True, "step": "pool_integrity", "config": config_key}
    data_path = Path(campaign_db).expanduser().resolve()
    if not data_path.is_file():
        return {"error": f"Campaign CSV not found at {data_path}.",
                "abort": True, "step": "pool_integrity", "config": config_key}

    try:
        check_campaign_pool(data_path)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {'error': str(error), 'abort': True, 'step': 'pool_integrity', 'config': config_key}

    # With no measured rows, use seeded Sobol candidates before fitting a GP.
    if _n_observed(data_path) == 0:
        method = CampaignMethod.from_environment()
        n_init = (method.batch_size if method and not method.n_agent
                  else int(os.getenv("OCEAN_N_INIT", "12")))
        # Report the fixed config even though cold-start Sobol needs no GP.
        return {**_cold_start_sobol(data_path, n_init=n_init,
                                    seed=int(os.getenv("OCEAN_SEED", "1"))),
                "config": config_key}

    return BOProcessRunner().run(config, data_path, volume_uL, _results_root(), config_key)


def _final_conc(entry, default_ru: float | None = None):
    """Normalize one suggestion into a flat {MA, NaBrO3, H2SO4, Ru(bpy)3} dict."""
    if not isinstance(entry, dict):
        return None
    c = entry.get("conc") if isinstance(entry.get("conc"), dict) else entry
    out = {}
    for key, aliases in (("MA", ("MA", "MA_M", "conc_MA_M")),
                         ("NaBrO3", ("NaBrO3", "NaBrO3_M", "conc_NaBrO3_M")),
                         ("H2SO4", ("H2SO4", "H2SO4_M", "conc_H2SO4_M")),
                         ("Ru(bpy)3", ("Ru(bpy)3", "Ru", "Ru(bpy)3_M", "conc_Ru_M"))):
        for a in aliases:
            if c.get(a) is not None:
                try:
                    out[key] = float(c[a])
                except (TypeError, ValueError):
                    return None
                break
        else:
            if key == "Ru(bpy)3" and default_ru is not None:
                out[key] = float(default_ru)
            else:
                return None
    return out


def _conc_key(conc: dict):
    """Rounded key for matching a suggestion back to a built protocol."""
    return (round(conc["MA"], 4), round(conc["NaBrO3"], 4),
            round(conc["H2SO4"], 4), round(conc["Ru(bpy)3"], 7))


def write_final_batch(json_path: str, suggestions: list):
    """Publish the agent's FINAL batch as a sibling `*_final.json` for submission."""

    p = Path(json_path or "")
    if not p.exists():
        print(f"[write_final_batch] no such suggestions JSON: {json_path!r} — skipped.")
        return None

    try:
        data, _ = load_protocol(p)
    except Exception as e:
        print(f"[write_final_batch] could not parse {p}: {type(e).__name__}: {e}")
        return None

    # A single unambiguous Ru level across BO's own candidates can fill in for entries the
    # agent re-typed without it (see _final_conc).
    _ru = set()
    for entry in (data if isinstance(data, list) else [data]):
        if isinstance(entry, dict):
            for cand in list(entry.get("suggestions") or []) + list(entry.get("reserve") or []):
                cc = cand.get("conc") if isinstance(cand, dict) else None
                if isinstance(cc, dict):
                    for a in ("Ru(bpy)3", "Ru", "conc_Ru_M"):
                        if cc.get(a) is not None:
                            _ru.add(round(float(cc[a]), 9)); break
    default_ru = _ru.pop() if len(_ru) == 1 else None

    # (normalized conc, original entry) pairs — the original is needed so an injected
    # point can be carried through verbatim rather than rebuilt.
    raw_list = list(suggestions or [])
    pairs = [(s, _final_conc(s, default_ru)) for s in raw_list]
    finals = [(c, raw) for raw, c in pairs if c]
    if not finals:
        print("[write_final_batch] no usable final suggestions — keeping BO's batch.")
        return None

    # NOTE: NEVER silently publish a shorter batch than we were handed. Dropping an
    # unparseable entry means the robot runs N-k experiments while wait_for_result polls
    # for all N — it then waits out its full timeout on results that were never
    # submitted. Refusing keeps submission and polling pointed at the same batch.
    unparsed = [i for i, (_, c) in enumerate(pairs) if not c]
    if unparsed:
        print(f"[write_final_batch] REFUSING write-back: {len(unparsed)} of {len(raw_list)} "
              f"final point(s) could not be read as concentrations (index {unparsed[:4]}"
              f"{'...' if len(unparsed) > 4 else ''}; e.g. {raw_list[unparsed[0]]!r:.160}). "
              f"Publishing a short batch would hang wait_for_result on the missing ones.")
        return None

    # Collect every candidate BO built a protocol for: {conc_key: entry}.
    iters = data if isinstance(data, list) else [data]
    built = {}
    latest_iter = 0
    for entry in iters:
        if not isinstance(entry, dict):
            continue
        latest_iter = max(latest_iter, int(entry.get("iter", 0) or 0))
        for cand in list(entry.get("suggestions") or []) + list(entry.get("reserve") or []):
            c = _final_conc(cand)
            if c:
                built.setdefault(_conc_key(c), cand)

    # Injected points (candidate INJECTION) legitimately have no BO-built protocol:
    # they were snapped to unmeasured POOL rows by _tool_propose_experiments. That is
    # safe in mailbox mode because the robot builds the protocol from concentrations
    # (io_mailbox sends `concentrations` + `builder_params`, never a protocol path) —
    # so a pool row needs nothing pre-built. Gated on OCEAN_BO_INJECT so the strict
    # subset check still applies to every non-injection campaign.

    allow_injected = bool(os.environ.get("OCEAN_BO_INJECT"))

    chosen, missing, n_injected = [], [], 0
    for c, raw in finals:
        cand = built.get(_conc_key(c))
        if cand is not None:
            chosen.append(cand)
        elif allow_injected:
            # Keep the agent's entry as-is; robot.py reads `conc` and ignores `path`.
            chosen.append(raw if isinstance(raw, dict) and raw.get("conc") else {"conc": c})
            n_injected += 1
        else:
            missing.append(c)
    if missing:
        print(f"[write_final_batch] REFUSING write-back: {len(missing)} final point(s) "
              f"have no built protocol (e.g. {missing[0]}). Submitting BO's batch instead.")
        return None
    if n_injected:
        print(f"[write_final_batch] {n_injected} injected point(s) carried through "
              f"(no BO protocol needed — HW builds from concentrations).")

    out_path = p.with_name(f"{p.stem}_final{p.suffix}")
    payload = [{**select_protocol(data), "iter": latest_iter, "suggestions": chosen, "reserve": []}]
    try:
        validate_protocol(payload)
        write_json_atomic(out_path, payload)
    except Exception as e:
        print(f"[write_final_batch] write failed ({type(e).__name__}: {e}) — "
              f"keeping BO's batch.")
        return None

    orig_n = len(next((e.get("suggestions") or [] for e in reversed(iters)
                       if isinstance(e, dict) and e.get("suggestions")), []))
    print(f"[write_final_batch] final batch = {len(chosen)} experiment(s) "
          f"(BO proposed {orig_n}) → {out_path}")
    return str(out_path)


def _batch_entries(entries, name):
    if entries is None:
        return []
    if not isinstance(entries, (list, tuple)) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f'[batch] {name} must be a list of suggestion mappings')
    return list(entries)


def _tool_filter_suggestions(suggestions: list, reasoning: str,
                             keep_indices: list = None, reserves: list = None,
                             add_reserve_indices: list = None):
    """Chemistry-aware post-hoc selection over the candidates BO already returned."""
    try:
        suggestions = _batch_entries(suggestions, 'suggestions')
        reserves = _batch_entries(reserves, 'reserves') # keep reserve entries consistent

        if keep_indices is not None:
            keep_indices = validate_indices(keep_indices, len(suggestions), allow_empty=True)
        if add_reserve_indices is not None:
            add_reserve_indices = validate_indices(add_reserve_indices, len(reserves), allow_empty=True)

    except ValueError as error:
        return {'error': str(error)}
    kept = (suggestions if keep_indices is None
            else [suggestions[i] for i in keep_indices if 0 <= i < len(suggestions)])
    added = [reserves[i] for i in ([] if add_reserve_indices is None else add_reserve_indices)]
    final = (kept + added) or list(suggestions)
    n_drop = len(suggestions) - len(kept)
    return {"suggestions": final, "n_kept": len(kept), "n_dropped_from_bo": n_drop,
            "n_added_from_reserve": len(added), "reasoning": reasoning,
            "overrode_bo": (n_drop > 0 or len(added) > 0)}

 
def _tool_propose_experiments(explore_concentrations: list, 
                              keep_bo_indices: list = None,
                              suggestions: list = None, 
                              reserves: list = None,
                              reasoning: str = ""):
    
    """CANDIDATE INJECTION: final batch = kept BO picks + agent-proposed points."""
    try:
        method = CampaignMethod.from_environment()
        if method and not method.n_agent:
            raise ValueError('[injection] LLM proposals are disabled in pure BO mode')
        
        suggestions = _batch_entries(suggestions, 'suggestions') # from BO
        reserves = _batch_entries(reserves, 'reserves') # from BO reserve pool
        requested = ([explore_concentrations] if isinstance(explore_concentrations, dict)
                     else _batch_entries(explore_concentrations, 'explore_concentrations'))
        
        if keep_bo_indices is not None:
            keep_bo_indices = validate_indices(keep_bo_indices, len(suggestions), allow_empty=True).tolist()

        if method: # ! the method dictates the number of BO and LLM proposals
            expected_indices = method.n_bo
            if len(requested) != method.n_agent:
                raise ValueError(f'[injection] Supply exactly {method.n_agent} LLM proposals')
            if keep_bo_indices is not None and len(keep_bo_indices) != expected_indices:
                raise ValueError(f'[injection] Keep exactly {expected_indices} BO primaries')
            if len(suggestions) < expected_indices:
                raise ValueError('[injection] Not enough BO primaries for this mode')
            if not reasoning.strip():
                raise ValueError('[injection] Explain the hypothesis behind the LLM proposals')
            
        for index, point in enumerate(requested):
            point = dict(point)
            for name, aliases in CONCENTRATION_ALIASES.items():
                column = aliases[-1] # CSV concentration name
                if column in point:
                    if name in point and float(point[name]) != float(point[column]):
                        raise ValueError(f'[injection] {name} aliases disagree')
                    point[name] = point.pop(column)
            requested[index] = point
            if method and any(point.get(name) is None for name in ('MA', 'NaBrO3', 'H2SO4')):
                raise ValueError('[injection] Each LLM point needs MA, NaBrO3 and H2SO4')
            
            for name in ('MA', 'NaBrO3', 'H2SO4', 'Ru', 'Ru(bpy)3'):
                if point.get(name) is not None:
                    value = point[name]
                    if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) < 0:
                        raise ValueError(f'[injection] {name} must be finite and nonnegative')
                    
            # ensure Ru and Ru(bpy)3 are consistent if both are provided
            if point.get('Ru') is not None and point.get('Ru(bpy)3') is not None and float(point['Ru']) != float(point['Ru(bpy)3']):
                raise ValueError('[injection] Ru aliases disagree')
            
    except (TypeError, ValueError, OverflowError) as error:
        return {'error': str(error), 'no_op': True}

    bsz = method.batch_size if method else (len(suggestions) or int(os.getenv("OCEAN_BATCH_SIZE", "6")))
    n_agent = max(0, min(int(os.getenv("OCEAN_INJECT_N_AGENT", str(bsz // 2))), bsz))
    n_bo = bsz - n_agent

    pool_path = os.getenv("OCEAN_CAMPAIGN_DB")
    if not pool_path or not Path(pool_path).exists():
        return {"error": "injection requires OCEAN_CAMPAIGN_DB (the candidate pool). "
                         "No injection performed; BO's batch stands.", "no_op": True}
    try:
        if method:
            check_campaign_pool(pool_path)
        pool = read_campaign_csv(pool_path)
    except Exception as e:
        return {"error": f"could not read pool {pool_path}: {e}", "no_op": True}

    # Unmeasured candidates only: no exp_id (never attempted) and no reward yet.
    unmeasured = pool
    if "exp_id" in pool.columns:
        eid = pool["exp_id"].astype(str)
        unmeasured = pool[pool["exp_id"].isna() | eid.isin(("", "nan", "NaN", "None"))]
    if "reward_oscillation" in unmeasured.columns:
        unmeasured = unmeasured[unmeasured["reward_oscillation"].isna()]
    if unmeasured.empty:
        return {"error": "no unmeasured candidates left in the pool", "no_op": True}

    # extract the concentrations for nearest neighbor search
    X = unmeasured[CONC_COLS].to_numpy(dtype=float) 
    if not np.isfinite(X).all():
        return {'error': '[injection] Candidate concentrations must be finite', 'no_op': True}
    candidates = NearestCandidates(X)

    # Concentrations already in this batch must not be proposed twice.
    def _flat(entry):
        c = entry.get("conc") if isinstance(entry, dict) and isinstance(entry.get("conc"), dict) else entry
        return c if isinstance(c, dict) else {}

    keep_idx = (list(keep_bo_indices) if keep_bo_indices is not None
                else list(range(min(n_bo, len(suggestions)))))
    kept = [suggestions[i] for i in keep_idx if 0 <= i < len(suggestions)][:n_bo]
    taken_keys = {(round(float(_flat(k).get("MA", -1)), 4),
                   round(float(_flat(k).get("NaBrO3", -1)), 4),
                   round(float(_flat(k).get("H2SO4", -1)), 4)) for k in kept
                  if _flat(k)}

    injected, matched = [], []
    key_map = {"MA": "conc_MA_M", "NaBrO3": "conc_NaBrO3_M",
               "H2SO4": "conc_H2SO4_M", "Ru": "conc_Ru_M", "Ru(bpy)3": "conc_Ru_M"}
    for req in requested[:n_agent]:
        if not isinstance(req, dict):
            continue
        target = {}
        for k, v in req.items():
            col = key_map.get(k)
            if col and v is not None:
                try:
                    target[col] = float(v)
                except (TypeError, ValueError):
                    pass
        if not target:
            continue
        # Distance over the axes the agent actually specified (usually MA/NaBrO3/H2SO4).
        cols = [c for c in CONC_COLS if c in target]
        idxs = [CONC_COLS.index(c) for c in cols]
        p = np.array([target[c] for c in cols])
        d, nearest = candidates.rank(p, idxs)
        for j in nearest:
            row = unmeasured.iloc[int(j)]
            conc = {"MA": float(row["conc_MA_M"]), "NaBrO3": float(row["conc_NaBrO3_M"]),
                    "H2SO4": float(row["conc_H2SO4_M"]), "Ru(bpy)3": float(row["conc_Ru_M"])}
            key = (round(conc["MA"], 4), round(conc["NaBrO3"], 4), round(conc["H2SO4"], 4))
            if key in taken_keys:
                continue # already in this batch — take next nearest
            taken_keys.add(key)
            injected.append({"path": f"[injected] {key[0]}_{key[1]}_{key[2]}", "conc": conc})
            matched.append({"requested": req,
                            "matched_conc": conc,
                            "snap_dist_norm": round(float(d[int(j)]), 4)})
            break

    if method and (len(kept) != method.n_bo or len(injected) != method.n_agent):
        return {'error': '[injection] Not enough distinct unmeasured pool points for the requested split', 'no_op': True}
    final = kept + injected
    # Pad from BO's remaining picks, then reserves, so the robot always gets a full batch.
    for cand in suggestions + reserves:
        if len(final) >= bsz:
            break
        c = _flat(cand)
        if not c:
            continue
        key = (round(float(c.get("MA", -1)), 4), round(float(c.get("NaBrO3", -1)), 4),
               round(float(c.get("H2SO4", -1)), 4))
        if key in taken_keys:
            continue
        taken_keys.add(key)
        final.append(cand)

    if not final:
        return {"error": "injection produced an empty batch; BO's batch stands",
                "no_op": True}

    print(f"  [INJECT] final batch = {len(kept)} BO + {len(injected)} agent-proposed"
          f"{f' + {len(final) - len(kept) - len(injected)} padded' if len(final) > len(kept) + len(injected) else ''}")
    for m in matched:
        print(f"    proposed {m['requested']} -> pool {m['matched_conc']} "
              f"(norm dist {m['snap_dist_norm']})")
    return {"suggestions": final, "n_bo_kept": len(kept), "n_agent_injected": len(injected),
            "n_padded": len(final) - len(kept) - len(injected),
            "agent_points": matched, "reasoning": reasoning,
            "overrode_bo": bool(injected)}


def _tool_override_bo_suggestions(
    working_db_path: str,
    original_rxn_ids: list,
    override_rxn_ids: list,
    reason: str,
):
    """Replace some BO-suggested candidates with agent-selected ones."""
    try:
        db = read_campaign_csv(working_db_path)
    except Exception as e:
        return {"error": str(e)}

    if "rxn_id" not in db.columns:
        return {"error": "working_db_path must have a 'rxn_id' column"}

    replaced = [r for r in original_rxn_ids if r not in override_rxn_ids]
    added = [r for r in override_rxn_ids if r not in original_rxn_ids]

    # Verify all override ids exist
    missing = [r for r in override_rxn_ids if r not in db["rxn_id"].values]
    if missing:
        return {"error": f"override_rxn_ids not found in DB: {missing}"}

    # Return the chosen IDs; execution and scoring happen at the mailbox boundary.
    print(f"  [AGENT-OVERRIDE] Replaced {len(replaced)} BO picks with {len(added)} agent picks")
    print(f"  [AGENT-OVERRIDE] Reason: {reason[:200]}")

    return {
        "status": "ok",
        "original_rxn_ids": original_rxn_ids,
        "final_rxn_ids": override_rxn_ids,
        "n_replaced": len(replaced),
        "n_added": len(added),
        "reason": reason,
    }


def _tool_generate_design_space(db_path: str, ranges: dict,
                                 n_points: int = 10_000, out_path: str = None):
    """Generate a random design space of `n_points` within `ranges`, excluding points in `db_path` if it exists."""
    if out_path is None:
        return {"error": "Provide out_path for the candidate CSV.", "n_generated": 0}
    key_map = {"MA": "conc_MA_M", "NaBrO3": "conc_NaBrO3_M",
               "H2SO4": "conc_H2SO4_M", "Ru": "conc_Ru_M", 'Ru(bpy)3': 'conc_Ru_M'}
    existing_set = set()
    if db_path:
        try:
            existing = read_campaign_csv(db_path)
            existing_set = set(map(tuple, existing[CONC_COLS].round(7).values.tolist()))
        except (OSError, TypeError, ValueError, KeyError) as error:
            return {"error": f"Cannot read observed concentrations from {db_path}: {error}", "n_generated": 0}
        if Path(out_path).expanduser().resolve() == Path(db_path).expanduser().resolve():
            return {"error": "Candidate output must differ from the observed dataset.", "n_generated": 0}
    try:
        integer_input(n_points, 'n_points')
        destination = OutputPaths().resolve(out_path, ('.csv',))
        if not isinstance(ranges, dict) or not ranges:
            raise ValueError('[design_space] Supply a nonempty mapping of concentration ranges')
        columns = [key_map.get(key, key) for key in ranges]
        if len(set(columns)) != len(columns):
            raise ValueError('[design_space] Concentration aliases must not define the same axis twice')
        for name, axis in ranges.items():
            if not isinstance(axis, dict) or any(key not in axis for key in ('min', 'max', 'step')):
                raise ValueError(f'[design_space] {name} requires min, max and step')
            values = [axis[key] for key in ('min', 'max', 'step')]
            if any(isinstance(value, bool) or not np.isfinite(value) for value in values):
                raise ValueError(f'[design_space] {name} range must be finite')
            if axis['min'] < 0 or axis['max'] < axis['min'] or axis['step'] <= 0:
                raise ValueError(f'[design_space] {name} requires 0 <= min <= max and step > 0')
        revision = file_revision(destination)
    except (TypeError, ValueError, OSError) as error:
        return {'error': str(error), 'n_generated': 0}
    grid = {key_map.get(k, k): np.round(np.arange(v["min"], v["max"] + 1e-9, v["step"]), 7)
              for k, v in ranges.items()}
    cols = list(grid.keys()); arrays = list(grid.values())
    sizes = [len(a) for a in arrays]
    n_tot = 1
    for s in sizes: n_tot *= s
    if n_tot == 0 or n_tot > np.iinfo(np.int64).max:
        return {'error': '[design_space] Grid combination count exceeds the supported integer range', 'n_generated': 0}
    rng = np.random.default_rng(42); collected = []; seen = set(existing_set)
    if n_points > n_tot:
        return {
            "error": f"Requested n_points={n_points} exceeds total design-space combinations={n_tot}.",
            "n_generated": 0,
            "out_path": out_path,
            "existing_excluded": len(existing_set),
        }

    max_batches = 100
    batch_count = 0
    while len(collected) < n_points and batch_count < max_batches:
        batch_count += 1
        before = len(collected)
        flat = rng.integers(0, n_tot, size=200_000); tmp = flat.copy(); indices = []
        for sz in reversed(sizes): indices.append(tmp % sz); tmp //= sz
        indices.reverse()
        for j in range(len(flat)):
            combo = tuple(float(arrays[ci][indices[ci][j]]) for ci in range(len(cols)))
            if combo not in seen: collected.append(combo); seen.add(combo)
            if len(collected) >= n_points: break
        if len(collected) == before:
            break

    if len(collected) < n_points:
        df = pd.DataFrame(collected, columns=cols); write_csv_atomic(df, destination, revision)
        return {
            "error": "Could not generate requested number of unseen points within bounded attempts.",
            "n_generated": len(df),
            "n_requested": n_points,
            "out_path": out_path,
            "existing_excluded": len(existing_set),
        }

    df = pd.DataFrame(collected, columns=cols); write_csv_atomic(df, destination, revision)
    return {"n_generated": len(df), "out_path": out_path, "existing_excluded": len(existing_set)}
