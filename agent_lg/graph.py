# Route campaign steps and track completed experiment batches.

import json, os, logging, json as _json
from pathlib import Path
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END, add_messages
from langgraph.types import interrupt
from agent_lg.agents import (
    NUDGE_MSG,
    CampaignPrompts,
    build_bo_decision_context,
    make_agent_node,
    make_delegate_tools,
)
from agent_lg.policy import (
    agentic_axes_on,
    hv_improved,
    improved_by_new_pareto,
    resolve_improve_signal,
    update_streak,
)
from agent_lg.tools.analysis import _compute_iteration_hv
from agent_lg.tools.literature import _fetch_literature_context, _RETRIEVAL_QUERIES
from agent_lg.tools_registry import ORCHESTRATOR_DIRECT_TOOLS, _dispatch
from agent_lg.tools.bo import write_final_batch
from agent_lg.services.method import CampaignMethod, CampaignRecipes
from agent_lg.services.pools import check_campaign_pool
from agent_lg.services.validation import CampaignSafetyError
from agent_lg.services.literature import LocalCorpus
from agent_lg.tools.hypothesis import load_corpus

logger = logging.getLogger(__name__)


class CampaignState(TypedDict):
    """Extended graph state with message history + shared context + multi-iteration tracking."""
    messages: Annotated[list, add_messages]
    literature_context: str
    bo_iter: int # incremented by progress_tracker_node after each wait_for_result
    pareto_hv_history: list # [{"iter": N, "n_pareto": K, "hv_2d": float, "n_observed": M}]
    db_path: str # persisted so progress_tracker can compute HV without parsing messages
    target_bo_iters: int # total BO iterations requested; rendered into the prompt so the
    pending_json_path: str # most recent protocol/json path returned by BO Specialist
    pending_concs: list # most recent suggestion list returned by BO Specialist
    #  agent-BO decision state (computed in progress_tracker_node)
    non_improving_streak: int # consecutive iters with <=5% relative HV gain
    last_iter_improved: bool # did the most recent iter improve HV by >5%?
    current_acqf: str # fixed qLogNParEGO, retained for progress reporting
    streak_of_current_acqf: int # consecutive non-improving iters on current_acqf
    n_switches_cumulative: int # running count of acqf switches (audit)
    lengthscale_history: list # per-iter GP diagnostic, never controls training
    campaign_exp_ids: list # exp_ids THIS campaign measured (campaign-scoped HV)
    campaign_id: str
    last_committed_batch: str
    collected_batch_id: str
    runtime_environment: dict # retained to detect incompatible legacy checkpoints


def nudge(_state: MessagesState):
    """Inject a reminder when the model returns text with no tool calls (local model quirk)."""
    print("[WARN] Model returned no tool calls — nudging to continue.")
    return {"messages": [HumanMessage(content=NUDGE_MSG)]}


def correct_early_finish(state: "CampaignState"):
    """Tell the model the true iteration counts after a premature finish_campaign."""
    done, target = _campaign_progress(state)
    return {"messages": [HumanMessage(content=_EARLY_FINISH_MSG.format(
        done=done, target=target, left=max(target - done, 0)))]}


# Retry silent model turns while iterations remain, with a finite limit.
MAX_NUDGES_WHILE_INCOMPLETE = 4

_EARLY_FINISH_MSG = (
    "STOP — you called finish_campaign but the campaign is NOT complete: "
    "{done} of {target} BO iterations have finished (each iteration is one "
    "wait_for_result that returned results). Count from `Current BO iter` in your "
    "Campaign Progress section, not from your own tally. Continue the BO LOOP from "
    "STEP 4 for the remaining {left} iteration(s). Do not call finish_campaign again "
    "until `Current BO iter` equals `Target BO iters`."
)


def _campaign_progress(state):
    """(iterations completed, iterations requested). target 0 == not specified."""
    return int(state.get("bo_iter", 0) or 0), int(state.get("target_bo_iters", 0) or 0)


def _trailing_nudges(msgs):
    """Consecutive nudge messages at the tail, ignoring the model's replies to them."""
    n = 0
    for m in reversed(msgs):
        if isinstance(m, HumanMessage) and m.content == NUDGE_MSG:
            n += 1
        elif isinstance(m, AIMessage):
            continue
        else:
            break
    return n


def route_after_agent(state):
    """Route to tools, or nudge if no tool calls (local model fallback)."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    msgs = state["messages"]
    if len(msgs) >= 2 and isinstance(msgs[-2], HumanMessage) and msgs[-2].content == NUDGE_MSG:
        # Normally one failed nudge ends the run. But ending while iterations remain
        # discards budget silently, so keep nudging (bounded) until the campaign is done.
        done, target = _campaign_progress(state)
        if target and done < target and _trailing_nudges(msgs) < MAX_NUDGES_WHILE_INCOMPLETE:
            print(f"[WARN] No tool calls, but only {done}/{target} BO iterations are done "
                  f"— nudging again rather than ending the campaign "
                  f"(attempt {_trailing_nudges(msgs) + 1}/{MAX_NUDGES_WHILE_INCOMPLETE}).")
            return "nudge"
        if target and done < target:
            print(f"[ERROR] Ending with {done}/{target} BO iterations complete — the model "
                  f"stopped calling tools after {MAX_NUDGES_WHILE_INCOMPLETE} nudges. "
                  f"{target - done} iteration(s) of budget were NOT used.")
        return END
    return "nudge"


def route_after_progress_tracker(state: CampaignState):
    """End if finish_campaign was called, otherwise loop back to agent."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "finish_campaign":
            done, target = _campaign_progress(state)
            if target and done < target:
                already_corrected = any(
                    isinstance(m, HumanMessage) and m.content.startswith("STOP — you called")
                    for m in state["messages"])
                if not already_corrected:
                    print(f"[WARN] finish_campaign called at {done}/{target} BO iterations "
                          f"— sending the real counts back and continuing.")
                    return "correct_early_finish"
                print(f"[ERROR] finish_campaign at {done}/{target} iterations, after a "
                      f"correction. Honouring it; {target - done} iteration(s) unused.")
            return END
        if isinstance(msg, AIMessage):
            break
    return "agent"


def require_tool_success(name, content):
    try:
        result = json.loads(content)
    except (TypeError, ValueError) as error:
        raise CampaignSafetyError(f"[{name}] Unreadable tool output; inspect the campaign before retrying") from error
    if not isinstance(result, dict) or result.get("error") or result.get("abort"):
        raise CampaignSafetyError(f"[{name}] Campaign paused: {result}")
    if name in ("submit_protocol", "run_qc") and result.get("status") != "ok":
        raise CampaignSafetyError(f"[{name}] Campaign paused: {result}")
    if name == "wait_for_result" and result.get("n_found", 0) == 0:
        raise CampaignSafetyError("[wait_for_result] No measurements were collected; inspect the instrument before continuing. Existing missing-result penalties are preserved")
    return result


def batch_arguments(state):
    database = state.get("db_path", "")
    campaign = state.get("campaign_id") or str(Path(database).resolve())
    return {"_batch_id": f"{campaign}:{int(state.get('bo_iter', 0)) + 1}", "_db_path": database}


def progress_tracker_node(state: CampaignState):
    """Update campaign progress and policy signals after tool execution."""
    msgs = state["messages"]

    recent_tool_msgs = []
    for msg in reversed(msgs):
        if isinstance(msg, ToolMessage):
            recent_tool_msgs.append(msg)
        else:
            break

    wait_result = None
    for msg in recent_tool_msgs:
        if msg.name == "wait_for_result":
            try:
                wait_result = json.loads(msg.content)
            except Exception:
                pass
            break

    if wait_result is None:
        return {}
    require_tool_success("wait_for_result", json.dumps(wait_result))
    completed_batch = wait_result.get("batch_id") or state.get("collected_batch_id", "")
    if completed_batch and completed_batch == state.get("last_committed_batch"):
        return {} # replaying the same committed batch must not spend another iteration
    if wait_result.get("n_found", 0) == 0:
        return {} # preserve the existing zero-result progress rule

    new_iter = state.get("bo_iter", 0) + 1
    db = state.get("db_path", "")
    history = list(state.get("pareto_hv_history", []))

    prev_hv = history[-1].get("hv_2d") if history else None # HV before this iter

    # exp_ids measured in THIS iteration (from wait_for_result's `found` list) and
    # cumulatively by this campaign — used for the Pareto-membership improvement
    # signal and campaign-scoped HV respectively.
    batch_ids = [str(f.get("exp_id")) for f in (wait_result.get("found") or [])
                 if isinstance(f, dict) and f.get("exp_id")]
    campaign_ids = list(dict.fromkeys(list(state.get("campaign_exp_ids") or []) + batch_ids))

    hv_info = {"iter": new_iter}
    if db and Path(db).exists():
        hv_info.update(_compute_iteration_hv(db, batch_ids=batch_ids,
                                             campaign_ids=campaign_ids))
        if hv_info.get("error"):
            raise CampaignSafetyError(f"[progress_tracker] Could not read campaign metrics: {hv_info['error']}")
    else:
        hv_info.update({"n_observed": 0, "n_pareto": 0, "hv_2d": 0.0})

    # `pareto_exp_ids` is provenance, not progress — keep it out of the history the
    # Orchestrator reads (it would grow the prompt every iter for no benefit).
    front_ids = hv_info.pop("pareto_exp_ids", None)
    history.append(hv_info)

    # Agent-BO decision state: did this iteration improve, and what's the stall
    # streak? Pure arithmetic on the HV history / front membership 

    improve_signal = resolve_improve_signal()
    if improve_signal == "pareto" and hv_info.get("n_new_pareto") is not None:
        improved = improved_by_new_pareto(hv_info.get("n_new_pareto"))
        signal_used = "pareto"
    else:
        improved = hv_improved(hv_info.get("hv_2d", 0.0), prev_hv)
        signal_used = "hv5pct"
        if improve_signal == "pareto":
            print("[WARN] Pareto improvement signal requested but n_new_pareto is "
                  "unavailable (does the pool have an exp_id column?) — falling back "
                  "to the 5% relative-HV rule, which fired 0/8 on real BZ batches. "
                  "Progress reporting will use the relative-HV rule.")
    new_streak = update_streak(state.get("non_improving_streak", 0), improved)

    # Read the current specialist result or its direct BO fallback.
    acqf_now = None
    bo_payload = {}
    fb_payload = {}
    for msg in reversed(msgs):
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name == "run_bo_iteration" and not fb_payload:
            try:
                fb_payload = json.loads(msg.content) or {}
            except Exception:
                fb_payload = {}
        if msg.name == "call_bo_specialist":
            try:
                bo_payload = json.loads(msg.content) or {}
            except Exception:
                bo_payload = {}
            break
    if not isinstance(bo_payload, dict):
        bo_payload = {}
    if not isinstance(fb_payload, dict):
        fb_payload = {}
    cfg = bo_payload.get("config") or fb_payload.get("config") or ""
    if cfg:
        acqf_now = "parego" # tool dispatch validates the single configuration

    current_acqf_out = acqf_now or state.get("current_acqf")
    streak_of_current = new_streak
    switched, n_switches = False, 0 # historical audit fields remain readable

    # Lengthscales are diagnostics; embedding training stays enabled.
    ls_history = list(state.get("lengthscale_history", []))
    _ls = bo_payload.get("mean_lengthscale")
    if _ls is None:
        _ls = fb_payload.get("mean_lengthscale")
    if _ls is not None:
        ls_history.append(float(_ls))
    print(
        f"[Progress] BO iter {new_iter} complete — "
        f"n_observed={hv_info.get('n_observed','?')}, "
        f"n_pareto={hv_info.get('n_pareto','?')}, "
        f"hv_2d={hv_info.get('hv_2d','?')}, "
        f"new_pareto_in_batch={hv_info.get('n_new_pareto','?')} [{signal_used}], "
        f"improved={improved}, non_improving_streak={new_streak}, "
        f"acqf={current_acqf_out}, streak_of_current_acqf={streak_of_current}"
    )


    _bo_mode = os.getenv("OCEAN_BO_MODE")
    if agentic_axes_on() and db:
        oracle_acqf = "parego" if acqf_now else None
        policy_violation = False if acqf_now else None
        # decision_source: a run_bo_iteration ToolMessage at the orchestrator
        # level (before the last call_bo_specialist) means the fallback fired.
        decision_source = "agent"
        for msg in reversed(msgs):
            if isinstance(msg, ToolMessage) and msg.name == "call_bo_specialist":
                break
            if isinstance(msg, ToolMessage) and msg.name == "run_bo_iteration":
                decision_source = "fallback"; break
        record = {
            "iter": new_iter, "mode": _bo_mode,
            "axes_on": [n for n, v in (("filter", os.getenv("OCEAN_BO_FILTER")),
                                       ("hypothesis", os.getenv("OCEAN_BO_HYPOTHESIS")),
                                       ("inject", os.getenv("OCEAN_BO_INJECT"))) if v],
            #  performance
            "n_observed": hv_info.get("n_observed"), "n_pareto": hv_info.get("n_pareto"),
            "hv_2d": hv_info.get("hv_2d"),
            # Campaign-scoped HV (excludes pre-campaign rows in the pool) + the
            # improvement signal that actually drove this iter's decision.
            "hv_campaign": hv_info.get("hv_campaign"),
            "n_observed_campaign": hv_info.get("n_observed_campaign"),
            "n_new_pareto": hv_info.get("n_new_pareto"),
            # The rule that actually decided `improved`; `improve_signal_requested`
            # differs from it only when the pareto signal was unavailable.
            "improve_signal": signal_used,
            "improve_signal_requested": improve_signal,
            "batch_exp_ids": batch_ids,
            "pareto_exp_ids": front_ids,
            #  decision quality
            "acqf": acqf_now, "decision_source": decision_source,
            "last_iter_improved": improved, "non_improving_streak": new_streak,
            "current_acqf": current_acqf_out, "streak_of_current_acqf": streak_of_current,
            "switched_this_iter": switched, "n_switches_cumulative": n_switches,
            "oracle_acqf": oracle_acqf, "policy_violation": policy_violation,
            #  cost / reliability / faithfulness (from call_bo_specialist)
            "wall_s": bo_payload.get("wall_s"),
            "n_tool_calls": bo_payload.get("n_tool_calls"),
            "n_llm_calls": bo_payload.get("n_llm_calls"),
            "total_tokens": bo_payload.get("total_tokens"),
            "filter_overrode_bo": bo_payload.get("filter_overrode_bo"),
            "reasoning": bo_payload.get("reasoning"),
            #  hypothesis + injection axes
            # inject_points carries the requested conc, the snapped pool conc and the
            # snap distance. The benchmark omitted this, making injected BZ points
            # unrecoverable after the fact; don't repeat that.
            "hypothesis": bo_payload.get("hypothesis"),
            "n_agent_injected": bo_payload.get("n_agent_injected"),
            "inject_points": bo_payload.get("inject_points"),
            "mean_lengthscale": _ls,
        }
        try:
            ap = os.getenv("OCEAN_AUDIT_PATH") or str(Path(db).parent / "agent_bo_audit.jsonl")
            with open(ap, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[audit] write failed: {e}")

    return {
        "last_committed_batch": completed_batch,
        "bo_iter": new_iter,
        "pareto_hv_history": history,
        "non_improving_streak": new_streak,
        "last_iter_improved": improved,
        "current_acqf": current_acqf_out,
        "streak_of_current_acqf": streak_of_current,
        "n_switches_cumulative": n_switches,
        "lengthscale_history": ls_history,
        "campaign_exp_ids": campaign_ids,
        "pending_json_path": "",
        "pending_concs": [],
    }


def _request_human_approval(proto: str):
    """Pause the graph for human approval of a robot submission."""
    print(f"\n[APPROVAL REQUIRED] Agent wants to submit: {proto}")
    approved = interrupt({
        "type": "human_review",
        "message": "Approve robot submission?",
        "protocol": proto,
    })
    if isinstance(approved, str):
        approved = approved.strip().lower() in ("yes", "y", "approve", "ok", "true", "1")
    return bool(approved)


def _bo_fallback(state: "CampaignState", bo_mode: str, tool_id: str):
    """Retry the fixed BO method when the specialist returns no usable batch."""

    try:
        CampaignRecipes().check_environment()
        if bo_mode != "fixed":
            raise ValueError('[config] BO fallback supports only the fixed method')
        config = CampaignRecipes.config_key
        CampaignRecipes().resolve(config)
    except ValueError as error:
        raise CampaignSafetyError(f'[fallback] {error}; no substitute acquisition was run') from error
    volume = float(os.getenv("OCEAN_VOLUME_UL", "1900"))
    print(f"[FALLBACK] BO Specialist produced no usable proposal — "
          f"running the fixed qLogNParEGO configuration ({config}).")

    content = _dispatch("run_bo_iteration", {"config": config, "volume_uL": volume})
    try:
        payload = json.loads(content)
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or payload.get("error") or not payload.get("json_path"):
        print(f"[FALLBACK] run_bo_iteration did not yield a proposal: {str(content)[:200]}")
        return None
    return {
        "message": ToolMessage(content=content, tool_call_id=f"{tool_id}_fallback",
                                 name="run_bo_iteration"),
        "json_path": payload.get("json_path") or payload.get("proto_path"),
        "concs": payload.get("primaries") or payload.get("suggestions") or [],
        "config": config,
    }


def make_tools_node(auto_approve: bool = False, delegate_tool_map: dict = None, campaign_prompts=None, context_store=None):
    """Returns the `tools` node for the Orchestrator."""
    _delegates = delegate_tool_map or {}

    def tools(state: CampaignState):
        if context_store is not None:
            context_store['value'] = state.get('literature_context', '') # checkpoint is authoritative even when retrieval was skipped
        try:
            CampaignRecipes().check_runtime(state.get('runtime_environment', {}))
        except ValueError as error:
            raise CampaignSafetyError(str(error)) from error
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        results = []
        pending_json_path = state.get("pending_json_path", "")
        pending_concs = list(state.get("pending_concs", []))
        state_updates = {}

        for tc in tool_calls:
            name = tc["name"]
            args = dict(tc.get("args", {}))
            tool_id = tc.get("id", f"tc_{name}")

            if name in ("call_bo_specialist", "finish_campaign") and pending_concs:
                raise CampaignSafetyError(f"[{name}] Pending batch must be collected before proposing or finishing")

            if name in ('call_bo_specialist', 'run_bo_iteration'):
                try:
                    check_campaign_pool(state.get('db_path'), state.get('campaign_exp_ids', []))
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                    raise CampaignSafetyError(f'[pool_integrity] {error}') from error

            if name == "submit_protocol":
                if not pending_json_path:
                    results.append(ToolMessage(
                        content=json.dumps({
                            "error": "No pending BO proposal available. Call call_bo_specialist first."
                        }),
                        tool_call_id=tool_id,
                        name=name,
                    ))
                    continue
                args["proto_path"] = pending_json_path
                args["json_path"] = pending_json_path
                args.update(batch_arguments(state))

            if name == "wait_for_result":
                if not pending_concs:
                    results.append(ToolMessage(
                        content=json.dumps({
                            "error": "No pending BO suggestions available. Call call_bo_specialist first."
                        }),
                        tool_call_id=tool_id,
                        name=name,
                    ))
                    continue
                args["concs"] = pending_concs
                args["db_path"] = state.get("db_path", args.get("db_path", ""))
                args["_batch_id"] = batch_arguments(state)["_batch_id"]

            if name == "run_qc":
                args["_batch_id"] = batch_arguments(state)["_batch_id"] + ":qc"
                args["db_path"] = state.get("db_path", args.get("db_path", ""))

            # Pass the configured selection mode and measured progress directly.
            if name == "call_bo_specialist" and (agentic_axes_on() or CampaignMethod.from_environment()):
                # Attribute hypothesis updates without relying on the model to pass the index.
                os.environ["OCEAN_BO_ITER"] = str(state.get("bo_iter", 0))
                args["task"] = str(args.get("task", "")) + build_bo_decision_context(state, campaign_prompts)

            if name == "submit_protocol" and not auto_approve: #! Manual approval for protocol submission
                if not _request_human_approval(args.get("proto_path", "unknown")):
                    print("[APPROVAL] Submission rejected.")
                    results.append(ToolMessage(
                        content=json.dumps({"status": "REJECTED",
                                            "message": "Human operator rejected this submission. "
                                                       "The batch was NOT sent to the robot — do not "
                                                       "call wait_for_result for it."}),
                        tool_call_id=tool_id,
                        name=name,
                    ))
                    # Drop the batch. Leaving it pending let a later wait_for_result
                    # (or auto_submitter) poll for experiments that were never sent —
                    # the campaign then hangs for the full 4 h timeout_s.
                    pending_json_path = ""
                    pending_concs = []
                    state_updates["pending_json_path"] = ""
                    state_updates["pending_concs"] = []
                    continue

            # Inject db_path for finish_campaign to enable fallback report generation
            if name == "finish_campaign":
                args["db_path"] = state.get("db_path", "")
                args["_campaign_id"] = state.get("campaign_id") or str(Path(args["db_path"]).resolve())

            print(f"\n[Tool] {name}({json.dumps(args)[:200]})")
            if name in _delegates:
                content = str(_delegates[name](args))
            else:
                content = _dispatch(name, args)
            if name in ("submit_protocol", "wait_for_result", "run_qc", "finish_campaign"):
                outcome = require_tool_success(name, content)
                if name == "wait_for_result":
                    state_updates["collected_batch_id"] = outcome.pop("batch_id", "")
                    content = json.dumps(outcome) # internal recovery IDs stay out of the model's context
                    pending_json_path, pending_concs = "", []
                    state_updates.update(pending_json_path="", pending_concs=[])
            # delegate tools (call_analyzer / call_bo_specialist) print their own
            # response inside the sub-agent; don't echo the same text again here.
            if name not in _delegates:
                print(f"       → {content[:300]}")
            results.append(ToolMessage(content=content, tool_call_id=tool_id, name=name))

            if name == "call_bo_specialist":
                state_updates["collected_batch_id"] = ""
                try:
                    payload = json.loads(content)
                except Exception:
                    payload = {}
                method = CampaignMethod.from_environment()
                if method:
                    try:
                        if not isinstance(payload, dict):
                            raise ValueError('[method] Specialist did not return a structured selection')
                        method.validate_selection(payload)
                    except ValueError as error:
                        raise CampaignSafetyError(str(error)) from error
                if isinstance(payload, dict):
                    pending_json_path = payload.get("json_path") or payload.get("proto_path") or pending_json_path
                    pending_concs = payload.get("suggestions") or pending_concs
                    state_updates["pending_json_path"] = pending_json_path
                    state_updates["pending_concs"] = pending_concs

                    # Write-back (gated on "any axis is on"): the batch the agent
                    # actually chose is what gets submitted. Without this the robot
                    # runs BO's original batch while wait_for_result polls for the
                    # agent's — see tools/bo.py::write_final_batch. MUST NOT be gated
                    # on OCEAN_BO_MODE: the filter and injection axes change the batch
                    # under their own env vars, so keying write-back to the acqf mode
                    # left `OCEAN_BO_INJECT=1` alone re-opening that 4 h phantom poll.
                    # Falls back to BO's JSON on any problem, so the loop can't break here.
                    if agentic_axes_on() and pending_json_path and pending_concs:
                        _final = write_final_batch(pending_json_path, pending_concs)
                        if _final:
                            pending_json_path = _final
                            state_updates["pending_json_path"] = _final
                        else:
                            if method:
                                raise CampaignSafetyError('[method] Could not save the selected batch; no substitute BO batch was submitted')
                            # Write-back refused, so the robot will be sent BO's ORIGINAL
                            # json. The polled concentrations must follow it, or we are
                            # back to blocker #1: robot runs one batch, wait_for_result
                            # polls for another, 4 h timeout. Losing the agent's selection
                            # for one iteration is the cheap failure; a mismatch is not.
                            _bo_concs = None
                            try:
                                _orig = _json.loads(Path(pending_json_path).read_text())
                                _entries = _orig if isinstance(_orig, list) else [_orig]
                                _bo_concs = next((e.get("suggestions") for e in reversed(_entries)
                                                  if isinstance(e, dict) and e.get("suggestions")),
                                                 None)
                            except Exception:
                                _bo_concs = None
                            if _bo_concs:
                                print(f"[write-back] refused — reverting the polled batch to "
                                      f"BO's original {len(_bo_concs)} suggestion(s) so "
                                      f"submission and polling agree.")
                                pending_concs = _bo_concs
                                state_updates["pending_concs"] = _bo_concs

                # Explicit fixed mode permits a direct BO retry on specialist failure.
                _bo_mode = os.getenv("OCEAN_BO_MODE")
                if _bo_mode and not method and not (pending_json_path and pending_concs):
                    fb = _bo_fallback(state, _bo_mode, tool_id)
                    if fb is not None:
                        results.append(fb["message"])
                        pending_json_path = fb["json_path"] or pending_json_path
                        pending_concs = fb["concs"] or pending_concs
                        state_updates["pending_json_path"] = pending_json_path
                        state_updates["pending_concs"] = pending_concs

                if not (pending_json_path and pending_concs):
                    raise CampaignSafetyError("[call_bo_specialist] No usable batch was returned; inspect the BO worker or specialist response before continuing")

            if name == "finish_campaign":
                try:
                    summary = json.loads(content).get("summary", "")
                except Exception:
                    summary = ""
                print(f"\n{'='*60}\n[DONE] Campaign finished.\n{summary}\n{'='*60}")

        return {"messages": results, **state_updates}
    return tools


def build_graph(base_model, auto_approve: bool = False, checkpointer=None):
    """Build the multi-agent orchestrator graph."""
    context_store = {"value": ""}
    campaign_prompts = CampaignPrompts.from_environment()

    def retrieval_node(state: CampaignState):
        method = CampaignMethod.from_environment()
        if method and not method.n_agent:
            context_store['value'] = ''
            return {'literature_context': ''} # pure BO needs no paper retrieval
        if existing := state.get("literature_context", ""):
            print("[Retrieval] Reusing cached literature context from checkpoint.")
            context_store["value"] = existing
            return {}
        if LocalCorpus().path is not None:
            print('[Retrieval] Loading the configured local papers.')
            ctx = load_corpus()
        else:
            print('[Retrieval] Fetching literature context from arXiv and ChemRxiv.')
            ctx = _fetch_literature_context()
        context_store["value"] = ctx
        print(f'[Retrieval] Loaded {len(ctx)} characters.')
        return {"literature_context": ctx}

    delegate_tools = make_delegate_tools(base_model, context_store, campaign_prompts=campaign_prompts)
    orch_tools = ORCHESTRATOR_DIRECT_TOOLS + delegate_tools
    orchestrator_model = base_model.bind_tools(orch_tools)

    delegate_tool_map = {t.name: t.invoke for t in delegate_tools}

    orch_node = make_agent_node(orchestrator_model, campaign_prompts.orchestrator, "Orchestrator",
                                campaign_prompts=campaign_prompts)
    tools_node = make_tools_node(auto_approve=auto_approve, delegate_tool_map=delegate_tool_map,
                                 campaign_prompts=campaign_prompts, context_store=context_store)

    def auto_submitter_node(state: CampaignState):
        """Submit the pending batch when the specialist has only suggested it."""
        pending_json = state.get("pending_json_path", "")
        pending_concs_s = state.get("pending_concs", [])
        db_path = state.get("db_path", "")

        if not (pending_json and pending_concs_s and db_path):
            return {}

        # Did the LLM itself submit/wait? Look at recent tool messages.

        # A submit_protocol that was REJECTED (or that lost the robot heartbeat) is NOT
        # a submission. Keying off the tool NAME alone meant a rejected attempt marked
        # the batch as "already submitted", so this node skipped submitting and injected
        # wait_for_result anyway — polling for six experiments that were never sent,
        # for the full 4 h timeout. Caught by the reject-first loopback test.
        def _submission_succeeded(message):
            result = require_tool_success("submit_protocol", message.content)
            return result.get("status") == "ok"

        # Submission and polling belong to the latest proposal, even when filenames repeat.
        messages = state["messages"]
        proposal_index = max((index for index, message in enumerate(messages)
                              if isinstance(message, ToolMessage)
                              and message.name in ("call_bo_specialist", "run_bo_iteration")),
                             default=-1)
        recent_tools = [message for message in messages[proposal_index + 1:]
                        if isinstance(message, ToolMessage)]
        already_submitted = any(getattr(m, "name", "") == "submit_protocol"
                                and _submission_succeeded(m) for m in recent_tools)
        already_waited = any(getattr(m, "name", "") == "wait_for_result" for m in recent_tools)

        if already_submitted and already_waited:
            return {} # both already done by the LLM

        msgs = []
        collected_batch_id = state.get("collected_batch_id", "")
        # submit_protocol — only if LLM didn't already do it.
        # Route through the SAME human-approval gate as the tools node so this
        # auto-injected path can no longer bypass sign-off (it previously did).
        if not already_submitted:
            if not auto_approve and not _request_human_approval(pending_json):
                print("[auto_submitter] Submission rejected by human.")
                msgs.append(ToolMessage(
                    content=json.dumps({"status": "REJECTED",
                                        "message": "Human operator rejected this submission."}),
                    tool_call_id="auto_submit", name="submit_protocol"))
                return {"messages": msgs, "pending_json_path": "", "pending_concs": []}
            print(f"\n[auto_submitter] injecting submit_protocol for {pending_json}")
            submit_content = _dispatch("submit_protocol",
                                        {"json_path": pending_json, "proto_path": pending_json,
                                         **batch_arguments(state)})
            require_tool_success("submit_protocol", submit_content)
            print(f"[auto_submitter] submit_protocol → {submit_content[:200]}")
            msgs.append(ToolMessage(content=submit_content,
                                    tool_call_id="auto_submit",
                                    name="submit_protocol"))

        # wait_for_result — always chain if not yet done (this is the common LLM skip)
        if not already_waited:
            # Normalize gollum's {"path": ..., "conc": {"MA": ...}} format to flat conc dicts.
            normalized_concs = []
            for s in pending_concs_s:
                if isinstance(s, dict) and isinstance(s.get("conc"), dict):
                    normalized_concs.append(s["conc"])
                elif isinstance(s, dict):
                    normalized_concs.append(s)
            print(f"\n[auto_submitter] injecting wait_for_result for {len(normalized_concs)} concs")
            wait_content = _dispatch("wait_for_result",
                                      {"concs": normalized_concs, "db_path": db_path,
                                       "qc_after_n": 0, "qc_volume_uL": 1900.0,
                                       "_batch_id": batch_arguments(state)["_batch_id"]})
            outcome = require_tool_success("wait_for_result", wait_content)
            collected_batch_id = outcome.pop("batch_id", "")
            wait_content = json.dumps(outcome)
            print(f"[auto_submitter] wait_for_result → {wait_content[:200]}")
            msgs.append(ToolMessage(content=wait_content,
                                    tool_call_id="auto_wait",
                                    name="wait_for_result"))

        if not msgs:
            return {}

        # Clear pending so we don't double-submit next turn
        return {"messages": msgs, "pending_json_path": "", "pending_concs": [],
                "collected_batch_id": collected_batch_id}

    g = StateGraph(CampaignState)
    g.add_node("retrieve", retrieval_node)
    g.add_node("agent", orch_node)
    g.add_node("tools", tools_node)
    g.add_node("auto_submitter", auto_submitter_node)
    g.add_node("nudge", nudge)
    g.add_node("correct_early_finish", correct_early_finish)
    g.add_node("progress_tracker", progress_tracker_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "nudge": "nudge", END: END})
    g.add_edge("nudge", "agent")
    g.add_edge("tools", "auto_submitter")
    g.add_edge("auto_submitter", "progress_tracker")
    g.add_edge("correct_early_finish", "agent")
    g.add_conditional_edges("progress_tracker", route_after_progress_tracker,
                            {"agent": "agent",
                             "correct_early_finish": "correct_early_finish",
                             END: END})

    return g.compile(checkpointer=checkpointer)
