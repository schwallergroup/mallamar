# Submit robot batches, collect results and run quality control.

import json, os, subprocess, sys, time, logging, numpy as np, pandas as pd
from pathlib import Path
from agent_lg.services.batches import BatchJournal, CommandJournal
from agent_lg.services.results import ResultMatcher
from agent_lg.services.protocols import load_protocol, positive_number, validate_concentrations
from agent_lg.services.storage import file_revision, locked_dataset, read_csv, write_csv_atomic, write_json_atomic
from agent_lg.config import (
    QC_EXPERIMENTS,
    QC_AFTER_N_EXPERIMENTS,
    QC_RUN_DURATION_S,
    QC_RUN_STEP_MAX_S,
    ROBOT_SCRIPT,
    SCRIPTS_DIR,
)
from agent_lg.rewards import (
    FAILURE_MODE_COL,
    FAILURE_NO_RESULT,
    RAW_COLS,
    recompute_all_rewards,
    score_with_frozen,
    sync_results_into_pool,
)

logger = logging.getLogger(__name__)


@locked_dataset
def _record_unresolved(poll_path: str, pending: list, col_map: dict):
    """Record submitted experiments whose measured result never appeared."""
    rows = []
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for i, conc in enumerate(pending, 1):
        row = {"exp_id": f"noresult_{stamp}_{i}",
               "exp_date_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "exp_dir": "",
               FAILURE_MODE_COL: FAILURE_NO_RESULT}
        for k, v in conc.items():
            if k in col_map:
                row[col_map[k]] = v
        for c in RAW_COLS:
            row.setdefault(c, "")
        row["n_osc"] = 0
        rows.append(row)
    if not rows:
        return 0

    revision = file_revision(poll_path)
    new = pd.DataFrame(rows)
    p = Path(poll_path)
    if p.exists():
        old = read_csv(p) # unreadable results must remain untouched
        new = pd.concat([old, new], ignore_index=True)
    write_csv_atomic(new, p, revision)
    return len(rows)


def _results_db_path(db_path: str):
    """Resolve the robot results file independently of the candidate pool."""
    if not os.getenv("OCEAN_CAMPAIGN_DB"):
        return db_path
    return os.getenv("OCEAN_RESULTS_DB") or str(Path(db_path).with_name("experiments_db.txt"))


def _tool_submit_protocol(json_path: str = None, proto_path: str = None,
                          _batch_id=None, _db_path=None):
    """Submit a suggestions JSON to the robot by running ROBOT_SCRIPT."""
    json_path = json_path or proto_path
    if not json_path:
        return {"error": "Missing input path. Provide `json_path` (or legacy `proto_path`)."}

    if not Path(json_path).exists():
        return {"error": f"Suggestions JSON not found: {json_path}"}

    journal = None
    try:
        load_protocol(json_path)
        if _batch_id and _db_path:
            journal = BatchJournal(_db_path, _batch_id)
            existing_ids = set()
            results_path = Path(_results_db_path(_db_path))
            if results_path.exists():
                existing_ids = set(read_csv(results_path)["exp_id"].dropna().astype(str))
            record = journal.load()
            mailbox_worker = SCRIPTS_DIR / 'robot.py'
            if record.get('state') in ('submitting', 'uncertain') and journal.command_path.exists():
                if not ROBOT_SCRIPT or Path(ROBOT_SCRIPT).resolve() != mailbox_worker:
                    raise RuntimeError('Partial mailbox submission requires the original Mallamar robot worker')
            cached = journal.begin_submission(json_path, existing_ids)
            if cached is not None:
                return cached
    except Exception as error:
        return {"error": str(error), "abort": True, "step": "submission_validation"}

    if ROBOT_SCRIPT is None:
        return {
            "status": "NOT_CONFIGURED",
            "message": "Set ROBOT_SCRIPT in agent_lg/config.py to enable robot submission.",
            "json_path": json_path,
        }

    worker_environment = os.environ.copy()
    worker_environment.pop('MALLAMAR_COMMAND_JOURNAL', None)
    mailbox_worker = SCRIPTS_DIR / 'robot.py'
    if journal and os.getenv('OCEAN_MAILBOX_ROOT') and Path(ROBOT_SCRIPT).resolve() == mailbox_worker:
        try:
            record = journal.load()
            CommandJournal(journal.command_path, os.environ['OCEAN_MAILBOX_ROOT']).bind_protocol(record['protocol_sha256'])
            worker_environment['MALLAMAR_COMMAND_JOURNAL'] = str(journal.command_path)
        except Exception as error:
            return {'error': str(error), 'abort': True, 'step': 'command_journal'}

    command = ([sys.executable, "-m", "agent_lg.scripts.robot", str(Path(json_path).resolve())]
               if Path(ROBOT_SCRIPT).resolve() == mailbox_worker
               else [sys.executable, ROBOT_SCRIPT, json_path])
    try:
        #! Run the robot script as a subprocess, passing the JSON path as an argument.
        # No wall-clock timeout here: liveness is governed by the robot heartbeat
        # inside run_experiment (io_mailbox.py), which exits 75 on a genuine stall.
        result = subprocess.run(
            command,
            capture_output=True, text=True, env=worker_environment,
        )
    except FileNotFoundError:
        return {"error": f"ROBOT_SCRIPT not found: {ROBOT_SCRIPT}"}

    if result.returncode == 75: # stall sentinel: robot went silent (no heartbeat)
        outcome = {
            "status": "no_heartbeat",
            "message": "Robot went silent (no heartbeat); it may STILL be running. "
                         "Do NOT assume failure or mock mode — hold and check the robot.",
            "stderr": result.stderr[-500:],
            "json_path": json_path,
        }
        if journal:
            journal.finish_submission(outcome)
        return outcome
    outcome = {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-500:] if result.returncode != 0 else "",
        "json_path": json_path,
    }
    if journal:
        journal.finish_submission(outcome)
    return outcome


def _tool_wait_for_result(concs: list, db_path: str,
                          poll_s: float = 30.0, timeout_s: float = 36_000.0,
                          qc_after_n: int | None = None, qc_volume_uL: float = 1900.0,
                          **_ignored):
    """Wait for all submitted experiments to appear in db_path."""
    if qc_after_n is None:
        qc_after_n = QC_AFTER_N_EXPERIMENTS
    try:
        if _env_to := os.getenv("OCEAN_RESULT_TIMEOUT_S"):
            timeout_s = positive_number(_env_to, "OCEAN_RESULT_TIMEOUT_S")
            print(f"[wait_for_result] timeout_s={timeout_s:.0f}s (OCEAN_RESULT_TIMEOUT_S)")
        poll_s, timeout_s = positive_number(poll_s, "poll_s"), positive_number(timeout_s, "timeout_s")
        if isinstance(qc_after_n, bool) or int(qc_after_n) != float(qc_after_n) or int(qc_after_n) < 0:
            raise ValueError("qc_after_n must be a non-negative integer")
        qc_after_n = int(qc_after_n)
        if qc_after_n:
            positive_number(qc_volume_uL, "qc_volume_uL")
    except (TypeError, ValueError, OverflowError) as error:
        return {"abort": True, "error": str(error), "step": "collection_validation"}
    col_map = {"MA": "conc_MA_M", "NaBrO3": "conc_NaBrO3_M",
               "H2SO4": "conc_H2SO4_M", "Ru": "conc_Ru_M", "Ru(bpy)3": "conc_Ru_M"}
    col_map.update({column: column for column in col_map.values()})

    try:
        validate_concentrations(concs)
        journal = BatchJournal(db_path, _ignored["_batch_id"]) if _ignored.get("_batch_id") else None
        record = journal.load() if journal else {}
        if journal and record.get("state") not in ("submitted", "collected", "committed"):
            raise RuntimeError("Batch has no confirmed submission; inspect the robot before collecting")
        if record.get("state") == "committed":
            return record["collection_result"]
        matcher = ResultMatcher(concs, record.get("existing_ids", []))
        concs = matcher.concentrations
    except Exception as error:
        return {"abort": True, "error": str(error), "step": "collection_validation"}

    def _safe_int(val, default=0):
        try:
            f = float(val)
            return int(f) if not np.isnan(f) else default
        except (TypeError, ValueError, OverflowError):
            return default

    def _safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    def extract(row, conc): # extract relevant info from the db row to return to the agent, e.g. n_osc, period_mean_s, etc. Adjust as needed based on your db schema and what info you want to return.
        return {"conc": conc, "exp_id": str(row["exp_id"]),
                "n_osc": _safe_int(row.get("n_osc", 0)),
                "period_mean_s": _safe_float(row.get("interp_period_mean_s", np.nan)),
                "maxima_mean_V": _safe_float(row.get("interp_maxima_mean_V", np.nan)),
                "total_time_s": _safe_float(row.get("total_time_s", np.nan))}

    pending, found = matcher.pending, []
    qc_triggered = False
    deadline = time.monotonic() + timeout_s

    # In single-pool mode the robot writes a SEPARATE experiments_db.txt sibling;
    # poll that for new measured rows (not the pool itself, which the robot never
    # writes). Legacy mode → poll db_path as before.
    poll_path = _results_db_path(db_path)

    # Do a first pass immediately (don't sleep first) — if submit_protocol already
    # filled the DB, we match right away and exit.
    first_pass = True
    cached_collection = record.get("collection_result")
    if cached_collection:
        found, pending = cached_collection["found"], cached_collection["missing"]
    while not cached_collection and time.monotonic() < deadline:
        if not first_pass:
            time.sleep(poll_s)
        first_pass = False
        try:
            df = read_csv(poll_path)
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        except (OSError, TypeError, ValueError) as error:
            return {"abort": True, "error": str(error), "step": "results_read"}
        try:
            matched_rows = matcher.collect(df)
        except (TypeError, ValueError) as error:
            return {"abort": True, "error": str(error), "step": "results_validation"}
        for row, conc in matched_rows:
            found.append(extract(row, conc))
        pending = matcher.pending
        print(f"[wait_for_result] {len(found)}/{len(concs)} results, {len(pending)} pending …")

        if qc_after_n > 0 and not qc_triggered and len(found) >= qc_after_n: # Mid batch QC
            qc_triggered = True
            print(f"[wait_for_result] {qc_after_n} results reached — triggering mid-batch QC …")
            qc_result = _tool_run_qc(volume_uL=qc_volume_uL, db_path=db_path,
                                     _batch_id=f"{journal.batch_id}:mid_qc" if journal else None)
            if qc_result.get("status") != "ok":
                print(f"[wait_for_result] QC FAILED — aborting. Details: {qc_result}")
                return {"n_found": len(found), "n_missing": len(pending),
                        "found": found, "missing": pending,
                        "abort": True, "abort_reason": "QC failed mid-batch", "qc": qc_result}
            print("[wait_for_result] QC passed — continuing.")

        if not pending:
            break

    # Deadline expired with experiments still unaccounted for. The robot reported
    # `finished` (or nothing at all) and no row ever appeared, so we cannot know what
    # happened to them — but leaving them unrecorded is what made a dropped experiment
    # cost the full timeout and then report partial success as if complete: the batch
    # looked done, bo_iter advanced on the survivors, and the missing points stayed
    # candidates forever.

    # Record them as attempted failures (n_osc=0, so the existing hard penalty applies
    # and the campaign moves on) but tagged `no_result`, NOT `chemistry` — nobody
    # measured this composition, so it must never be read as evidence about it.
    if pending and not cached_collection:
        try:
            n_written = _record_unresolved(poll_path, pending, col_map)
            print(f"[wait_for_result] TIMEOUT after {timeout_s:.0f}s with {len(pending)} "
                  f"experiment(s) unaccounted for — recorded {n_written} as "
                  f"n_osc=0 / failure_mode={FAILURE_NO_RESULT} so the campaign is not "
                  f"blocked. These are NOT chemistry evidence; check the instrument.")
        except Exception as e:
            return {"abort": True, "error": str(e), "step": "record_missing_results",
                    "n_found": len(found), "n_missing": len(pending), "found": found, "missing": pending}

    outcome = {"n_found": len(found), "n_missing": len(pending),
               "found": found, "missing": pending, "abort": False}
    if journal:
        outcome["batch_id"] = journal.batch_id
        journal.collected(outcome)

    # Single-pool mode: pull the robot's freshly-measured rows from the SEPARATE
    # experiments_db.txt into the pool's matching candidate rows (fill interp_*), so a
    # measured candidate leaves heldout on the next BO read.
    if os.getenv("OCEAN_CAMPAIGN_DB"):
        try:
            n_synced = sync_results_into_pool(db_path, poll_path)
            if n_synced:
                print(f"[wait_for_result] synced {n_synced} measured results into the pool.")
            if journal and found:
                synchronized_ids = set(read_csv(db_path)["exp_id"].dropna().astype(str))
                if not {item["exp_id"] for item in found}.issubset(synchronized_ids):
                    raise ValueError("Current batch results are not present in the campaign pool; check experiment identity")
        except Exception as e:
            return {**outcome, "abort": True, "error": str(e), "step": "pool_sync"}

    # Then compute rewards from interp_* per the chosen mode (OCEAN_REWARD_MODE):
    #   off       → leave rewards alone (dummy/test runs; default)
    #   recompute → refit z-norm over ALL observed each batch, re-score everything
    #   freeze    → fit z-norm ONCE on the first batch, score only new rows with it
    reward_mode = os.getenv("OCEAN_REWARD_MODE", "off")
    if reward_mode not in ("off", "recompute", "freeze"):
        return {**outcome, "abort": True, "error": f"Unknown reward mode: {reward_mode}", "step": "reward_validation"}
    if reward_mode in ("recompute", "freeze"):
        try:
            frame = read_csv(db_path)
            missing_columns = [column for column in RAW_COLS if column not in frame.columns]
            if missing_columns:
                raise ValueError(f"Missing raw reward columns: {missing_columns}")
            if reward_mode == "recompute":
                n_scored = recompute_all_rewards(db_path)
            else:
                norm_path = Path(db_path).with_name("reward_norm.json")
                n_scored = score_with_frozen(db_path, norm_path)
            print(f"[wait_for_result] reward mode={reward_mode}: scored {n_scored} rows.")
            updated = read_csv(db_path)
            for column in ("reward_oscillation", "reward_signal"):
                values = pd.to_numeric(updated.loc[updated["exp_id"].astype(str).isin(
                    [item["exp_id"] for item in found]), column], errors="coerce")
                if len(values) != len(found) or not np.isfinite(values).all():
                    raise ValueError("Collected measurements were not fully scored in the campaign dataset")
        except Exception as e:
            return {**outcome, "abort": True, "error": str(e), "step": "reward_calculation"}

    if journal:
        journal.committed(outcome)
    return outcome


def _tool_run_qc(volume_uL: float, db_path: str,
                 poll_s: float = 30.0, timeout_s: float = 36_000.0, _batch_id=None):
    """Run the full QC cycle atomically: build JSON, submit to robot, wait for all results."""
    if not QC_EXPERIMENTS:
        return {
            "status": "NOT_CONFIGURED",
            "message": "Fill QC_EXPERIMENTS in agent_lg/config.py with your calibration concentrations.",
        }

    try:
        validate_concentrations(QC_EXPERIMENTS)
        positive_number(volume_uL, "QC volume_uL")
        poll_s, timeout_s = positive_number(poll_s, "poll_s"), positive_number(timeout_s, "timeout_s")
    except (TypeError, ValueError) as error:
        return {"status": "error", "error": str(error)}

    # Write QC JSON (same format as BO suggestions so robot.py handles both identically)
    # Include QC-specific run parameters if configured
    proto = {
        "label": "qc",
        "volume_uL": volume_uL,
        "suggestions": QC_EXPERIMENTS,
        "stall_timeout_s": 7200, # 2 hours for long BZ experiments
        "run_duration_s": QC_RUN_DURATION_S,
        "run_step_max_s": QC_RUN_STEP_MAX_S,
    }
    out_path = Path(db_path).parent / ".mallamar" / "qc_protocol.json"
    write_json_atomic(out_path, proto)

    # Single-pool: QC results land in the robot's SEPARATE experiments_db.txt
    # (= db_path when not in single-pool mode). Poll that file, and snapshot its
    # existing exp_ids BEFORE submit so we only match NEW QC rows.
    qc_poll_path = _results_db_path(db_path)
    try:
        seen_ids = set(read_csv(qc_poll_path)["exp_id"].astype(str).tolist())
    except FileNotFoundError:
        seen_ids = set()
    except Exception as error:
        return {"status": "error", "error": f"Cannot snapshot QC results: {error}"}

    # Submit to robot
    if ROBOT_SCRIPT is None:
        return {"status": "NOT_CONFIGURED", "message": "Set ROBOT_SCRIPT in agent_lg/config.py."}
    submission = _tool_submit_protocol(str(out_path), _batch_id=_batch_id, _db_path=db_path)
    if submission.get("status") == "no_heartbeat":
        return {"status": "no_heartbeat",
                "message": "Robot went silent during QC (no heartbeat); it may STILL be "
                           "running. Do NOT assume QC failed or that this is a mock/dry-run "
                           "environment — hold and check the robot.",
                "stderr": submission.get("stderr", "")}
    if submission.get("status") != "ok":
        return {"status": "failed", "error": submission.get("error", "QC submission failed"),
                "stderr": submission.get("stderr", "")}
    journal = BatchJournal(db_path, _batch_id) if _batch_id else None
    if journal:
        record = journal.load()
        if record.get("state") in ("collected", "committed"):
            return record["collection_result"]
        seen_ids = set(record["existing_ids"]) # include rows preceding submission, not its own completed QC

    matcher = ResultMatcher(QC_EXPERIMENTS, seen_ids)
    pending, found = matcher.pending, []
    deadline = time.monotonic() + timeout_s

    # Do a first pass immediately (don't sleep first) — if robot already
    # filled the DB, we match right away and exit.
    first_pass = True
    while time.monotonic() < deadline:
        if not first_pass:
            time.sleep(poll_s)
        first_pass = False
        try:
            df = read_csv(qc_poll_path)
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        except (OSError, TypeError, ValueError) as error:
            return {"status": "error", "error": str(error), "step": "results_read"}
        try:
            for row, conc in matcher.collect(df, mark_unmatched=True):
                n_osc = float(row.get("n_osc", 0))
                if not np.isfinite(n_osc) or n_osc < 0 or n_osc != int(n_osc):
                    raise ValueError("QC n_osc must be a finite non-negative integer")
                found.append({"conc": conc, "n_osc": int(n_osc)})
        except (TypeError, ValueError, OverflowError) as error:
            return {"status": "error", "error": str(error), "step": "results_validation"}
        pending = matcher.pending
        print(f"[run_qc] {len(found)}/{len(QC_EXPERIMENTS)} QC results, {len(pending)} pending …")
        if not pending:
            break

    if pending:
        return {"status": "timeout", "n_found": len(found), "n_missing": len(pending), "missing": pending}

    failed = [r for r in found if r["n_osc"] == 0] #! Added check for failture
    if failed:
        outcome = {"status": "error", "reason": "QC n_osc=0", "failed": failed,
                "n_found": len(found), "n_missing": 0}
        if journal:
            journal.collected(outcome)
        return outcome

    outcome = {"status": "ok", "n_found": len(found), "n_missing": 0}
    if journal:
        journal.committed(outcome)
    return outcome
