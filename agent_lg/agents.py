# Build specialist agents and their campaign decision context.

import json, os, time, logging
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from agent_lg.tools_registry import (
    ANALYZER_TOOLS,
    BO_TOOLS,
    _dispatch,
    _parse_tool_calls_from_text,
    filter_suggestions,
    update_hypothesis,
    propose_experiments,
)
from agent_lg.tools import hypothesis as _H
from agent_lg.policy import resolve_improve_signal, agentic_axes_on
from agent_lg.services.method import CampaignMethod

from agent_lg.services.profiling import profile_step, profiled

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str, **kwargs):
    """Read <name>.md from the prompts directory and substitute any {key} placeholders."""
    text = (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))
    return text

NUDGE_MSG = (
    "Do NOT stop or ask questions. Call the next tool in the workflow immediately. "
    "If the campaign is fully complete, call finish_campaign."
)


class CampaignPrompts:
    """Prompt text and settings captured when a campaign is built."""

    def __init__(self, locked_config=None, experiment_mode="fixed", context_window=20):
        if experiment_mode != "fixed" or locked_config not in (None, "t5gemma-s"):
            raise ValueError('[CampaignPrompts] Campaigns use the single fixed t5gemma-s configuration')
        self.experiment_mode = "fixed"
        self.allowed_configs = "['t5gemma-s']"
        self.context_window = int(context_window)
        self.bz_context = _load_prompt("bz_context")
        self.orchestrator = _load_prompt("orchestrator", bz_context=self.bz_context,
                                         experiment_mode=self.experiment_mode,
                                         allowed_configs=self.allowed_configs)
        self.analyzer = _load_prompt("analyzer", bz_context=self.bz_context)
        self.bo = _load_prompt("bo_specialist", bz_context=self.bz_context)
        method = CampaignMethod.from_environment()
        if method:
            self.orchestrator += f"\n\nConfigured campaign_mode: {method.mode}; batch_size: 6."
            if not method.n_agent:
                self.bo += "\n\nConfigured campaign_mode: bo; retain the six ALICE primaries unchanged."
                self.analyzer += "\n\nConfigured campaign_mode: bo; only analyze campaign data and write requested reports. Literature search is disabled."

    @classmethod
    def from_environment(cls):
        return cls(context_window=os.getenv("OCEAN_CONTEXT_WINDOW", "20"))


_default_prompts = CampaignPrompts.from_environment()
BZ_CONTEXT = _default_prompts.bz_context
ORCHESTRATOR_PROMPT = _default_prompts.orchestrator
ANALYZER_PROMPT = _default_prompts.analyzer
BO_PROMPT = _default_prompts.bo
_experiment_mode = _default_prompts.experiment_mode
_allowed_configs = _default_prompts.allowed_configs


# Bound model context while retaining the complete checkpoint history.
# Current progress and decision inputs are added again on each turn.
ORCH_CONTEXT_WINDOW = int(os.getenv("OCEAN_CONTEXT_WINDOW", "20"))


def _window_messages(msgs: list, keep: int = None):
    """Keep the task and recent messages without orphaning tool results."""
    keep = ORCH_CONTEXT_WINDOW if keep is None else keep
    if keep <= 0 or len(msgs) <= keep + 1:
        return msgs, 0 # short campaign — unchanged from before

    start = len(msgs) - keep
    # Advance past any leading tool results so every ToolMessage kept still has its
    # parent AIMessage in the window.
    while start < len(msgs) and isinstance(msgs[start], ToolMessage):
        start += 1
    head = [msgs[0]] if start > 0 else []
    return head + msgs[start:], max(start - 1, 0)


def _flag(value):
    """Render a boolean as the lowercase spelling used in the prompts."""
    return "true" if value else "false"


def _current_hypothesis():
    """Current hypothesis text, or None when hypothesis mode is off/unavailable."""
    if not (os.getenv("OCEAN_BO_HYPOTHESIS") or os.getenv("OCEAN_BO_INJECT")):
        return None
    return _H.store().text or None


def build_bo_decision_context(state: dict, campaign_prompts=None):
    """Authoritative decision context appended to the BO Specialist's task message."""
    campaign_prompts = campaign_prompts or CampaignPrompts.from_environment()
    history = list(state.get("pareto_hv_history") or [])
    last_hv = history[-1].get("hv_2d") if history else None
    prev_hv = history[-2].get("hv_2d") if len(history) >= 2 else None
    delta = (round(last_hv - prev_hv, 4)
               if isinstance(last_hv, (int, float)) and isinstance(prev_hv, (int, float))
               else None)
    n_new_pareto = history[-1].get("n_new_pareto") if history else None

    signal = resolve_improve_signal()

    lines = [
        "",
        "",
        "--- AUTHORITATIVE DECISION CONTEXT (precomputed; use these EXACT values;",
        "    they override anything stated earlier in this task) ---",
        f"experiment_mode        = {campaign_prompts.experiment_mode}",
        f"ALLOWED CONFIGS        = {campaign_prompts.allowed_configs}",
        "HV Decision Context:",
        f"  last_completed_hv      = {last_hv}",
        f"  previous_completed_hv  = {prev_hv}",
        f"  last_delta_hv          = {delta}",
        f"  last_iter_improved     = {_flag(state.get('last_iter_improved', False))}",
        f"  non_improving_streak   = {state.get('non_improving_streak', 0)}",
        f"  current_acqf           = {state.get('current_acqf') or 'none (first decision)'}",
        f"  streak_of_current_acqf = {state.get('streak_of_current_acqf', 0)}",
        f"  bo_iter                = {state.get('bo_iter', 0)}",
    ]
    # How `last_iter_improved` was decided. Under the 'pareto' signal it means "the
    # last batch put >=1 point on the Pareto front" — NOT ">5% HV". Say so, because
    # HV can fall while improving when rewards are recomputed each batch
    # (OCEAN_REWARD_MODE=recompute refits the z-norm; see agent_lg/rewards.py).
    if signal == "pareto":
        lines += [
            f"  improvement_rule       = new Pareto point in batch "
            f"(new_pareto_last_batch = {n_new_pareto})",
            "  NOTE: hv_2d may DECREASE while last_iter_improved=true — rewards are "
            "re-normalised each batch. Trust last_iter_improved, not the raw HV delta.",
        ]
    else:
        lines.append("  improvement_rule       = HV grew >5% relative")

    lines.append(f"filter_enabled: {_flag(os.getenv('OCEAN_BO_FILTER'))}")

    method = CampaignMethod.from_environment()
    if method and not method.n_agent:
        lines += ['campaign_mode: bo; batch_size: 6; keep_bo_count: 6; llm_proposal_count: 0',
                  'SELECTION: return all six ALICE primaries in their original order; do not use hypothesis, filtering or injection.']

    # hypothesisdriven mode (+ optional candidate injection)
    # Ground the decision task in literature and observed experiment data.
    # data on the first pass (write a hypothesis), then current hypothesis + fresh
    # data every later pass (revise it). SOFT prioritisation only — the ablation
    # showed soft beats hard veto (69.59 vs 69.13 HV).
    _hyp_on = bool(os.getenv("OCEAN_BO_HYPOTHESIS")) or bool(os.getenv("OCEAN_BO_INJECT"))
    _inject_on = bool(os.getenv("OCEAN_BO_INJECT"))
    if _hyp_on:
        db = state.get("db_path", "") or os.getenv("OCEAN_CAMPAIGN_DB", "")
        current = _H.store().text
        # Data grounding is load-bearing: without it the benchmark agent
        # hallucinated off-pool chemistry (Mn(II), MeMA) and scored worst.
        obs = _H.observed_table(db) if db else "(no campaign db in state)"
        lines.append(f"hypothesis_enabled: true")
        lines.append(f"inject_enabled: {_flag(_inject_on)}")
        if not current:
            lines += [
                "",
                "HYPOTHESIS-DRIVEN MODE (first iteration — no hypothesis yet).",
                "LITERATURE CORPUS (read it; abstracts, results and conclusions):",
                _H.load_corpus(),
                "",
                "OBSERVED DATA so far (concentration → reward):",
                obs,
                "",
                "STEP H: call update_hypothesis(hypothesis='...') with a concrete, "
                "testable hypothesis about WHERE the good chemistry is. Allow MULTIPLE "
                "operating windows — there may be more than one. Ground it in the "
                "corpus AND the observed data above.",
            ]
        else:
            lines += [
                "",
                "HYPOTHESIS-DRIVEN MODE.",
                "YOUR CURRENT HYPOTHESIS:",
                current,
                "",
                "OBSERVED DATA so far (concentration → reward):",
                obs,
                "",
                "STEP H: REVISE it against the new observations — call "
                "update_hypothesis(hypothesis='...') with the updated version. Keep "
                "multiple windows where the data supports them; drop what it "
                "contradicts. Cite specific measured points. CARRY FORWARD the windows "
                "and contradicted regions you already established and ADD to them — a "
                "shorter hypothesis that discards earlier evidence is a worse one. "
                "Re-sending the text above unchanged is rejected: revise it, or skip "
                "the call if the new data genuinely changes nothing.",
            ]
        if _inject_on:
            _n_agent = os.getenv("OCEAN_INJECT_N_AGENT", "3")
            method = CampaignMethod.from_environment()
            if method:
                lines.append(f"campaign_mode: {method.mode}; batch_size: 6; keep_bo_count: {method.n_bo}; llm_proposal_count: {method.n_agent}")
            lines += [
                "",
                f"SELECTION — INJECTION: the final batch = kept BO picks + "
                f"{_n_agent} point(s) proposed by YOU.",
                "  After run_bo_iteration, call propose_experiments("
                "keep_bo_indices=[...], explore_concentrations=[{\"MA\":_,"
                "\"NaBrO3\":_,\"H2SO4\":_}, ...], suggestions=<from run_bo_iteration>, "
                "reserves=<from run_bo_iteration>, reasoning='...').",
                "  Each proposal is snapped to the nearest UNMEASURED pool candidate. "
                "Use it to probe hypothesis windows BO is NOT proposing. Say which "
                "window each point targets. You MUST call this tool successfully; do not accept BO's batch as a substitute.",
            ]
        else:
            lines += [
                "",
                "SELECTION (soft-prioritise by your hypothesis — do NOT hard-veto). "
                "Prefer hypothesis-consistent candidates, but do NOT drop a strong BO "
                "pick just because it mildly conflicts; keep coverage of every "
                "hypothesised window.",
            ]

    return "\n".join(lines)


def make_agent_node(model, system_prompt: str, label: str = "agent", campaign_prompts=None):
    """Returns an agent node that calls `model` with `system_prompt`."""
    campaign_prompts = campaign_prompts or CampaignPrompts.from_environment()
    def agent(state):
        extras = []
        ctx = state.get("literature_context", "")
        bo_iter = state.get("bo_iter", 0)
        hv_history = state.get("pareto_hv_history", [])

        if ctx:
            extras.append(f"## Retrieved Literature\n{ctx}")

        # `Target BO iters` gates the orchestrator's loop exit (prompts/orchestrator.md).
        # It used to live only in the opening task string, so by iteration 12 the exit
        # condition rested on the model recalling a number from turn 1. Rendered here it
        # is restated every turn. Omitted when unset so a programmatic run_agent caller
        # that never passes it sees the previous text.
        _target = state.get("target_bo_iters") or 0
        _target_line = f"Target BO iters: {_target}\n" if _target else ""

        if hv_history:
            rows = "\n".join(
                f"  iter {e['iter']}: n_observed={e.get('n_observed','?')}, "
                f"n_pareto={e.get('n_pareto','?')}, hv_2d={e.get('hv_2d','?')}"
                for e in hv_history
            )
            progress = (
                f"## Campaign Progress\n"
                f"Current BO iter: {bo_iter}\n"
                f"{_target_line}"
                f"Pareto HV history (updated automatically after each wait_for_result):\n{rows}"
            )
            # Gated agent-BO decision context: only when an agentic axis is on, so
            # These fields are precomputed in progress_tracker_node; the
            # BO Specialist decision tables consume them instead of re-deriving
            # streaks by eye. Only rendered here (orchestrator has the real
            # CampaignState); the orchestrator relays them in the BO task.
            if agentic_axes_on():
                progress += (
                    f"\n\n### BO Decision Context (precomputed — use these exact values)\n"
                    f"experiment_mode: {campaign_prompts.experiment_mode}\n"
                    f"non_improving_streak: {state.get('non_improving_streak', 0)}\n"
                    f"last_iter_improved: {_flag(state.get('last_iter_improved', False))}\n"
                    f"current_acqf: {state.get('current_acqf', 'parego')}\n"
                    f"streak_of_current_acqf: {state.get('streak_of_current_acqf', 0)}\n"
                    f"filter_enabled: {_flag(os.getenv('OCEAN_BO_FILTER'))}"
                )
            extras.append(progress)
        elif bo_iter > 0:
            extras.append(f"## Campaign Progress\nCurrent BO iter: {bo_iter} "
                          f"(no HV data yet)\n{_target_line}".rstrip())

        # Bound what the ORCHESTRATOR sends to the model (see _window_messages). Subagents
        # are invoked with a fresh single-message task and discarded, so they never
        # accumulate and are left alone. state["messages"] is NOT modified: the
        # checkpoint, the audit, auto_submitter's `state["messages"][-8:]` scan and
        # progress_tracker_node's reverse scan all keep the complete history.
        history, n_elided = (_window_messages(state["messages"], keep=campaign_prompts.context_window)
                             if label == "Orchestrator" else (state["messages"], 0))
        if n_elided:
            extras.append(
                f"## Note on this transcript\n"
                f"{n_elided} earlier message(s) are not shown — they are superseded by the "
                f"campaign state above, which is regenerated every turn and is "
                f"authoritative. The full history is preserved in the checkpoint and the "
                f"audit log. Do not try to recall elided turns; use Campaign Progress for "
                f"where the campaign stands, and the messages below for what you are doing "
                f"right now.")

        sys_content = system_prompt + ("\n\n" + "\n\n".join(extras) if extras else "") # Append extras to system prompt so all agents have the same context info available at each turn
        # The system prompt is identical every turn; dumping it is just noise.
        # Print only the agent label as an "active turn" marker. Set OCEAN_DEBUG_PROMPTS=1
        # to restore the full prompt dump when debugging prompt content.
        # if os.getenv("OCEAN_DEBUG_PROMPTS") == "1":
            # print(f"\n[{label}] System prompt:\n{sys_content[:1000]})§")
        # else:
            # print(f"\n[{label}]")

        if n_elided:
            print(f"[{label}] context window: sending {len(history)} of "
                  f"{len(state['messages'])} messages ({n_elided} elided)")
        messages = [SystemMessage(content=sys_content)] + history # Prepend system prompt to the (windowed) conversation history

        # NOTE: This is where the model is actually generates the suggestions
        with profile_step("llm_call", agent=label):
            response = model.invoke(messages)
        if response.content:
            print(f"\n[{label}] {response.content[:400]}")

        # Text fallback: patch tool_calls from text if the model didn't use the API
        if not getattr(response, "tool_calls", None):
            parsed = _parse_tool_calls_from_text(response.content or "")
            if parsed:
                print(f"[{label}/FALLBACK] Parsed {len(parsed)} tool call(s) from text.")
                response = AIMessage(
                    id=response.id,
                    content=response.content,
                    tool_calls=[
                        {"id": f"fb_{i}", "name": p["name"], "args": p["arguments"], "type": "tool_call"}
                        for i, p in enumerate(parsed)
                    ],
                )

        return {"messages": [response]}
    return agent


def _successful_tool_names(messages):
    """Return tools whose latest recorded call did not return an error."""
    successful = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage) or not getattr(msg, "name", None):
            continue
        try:
            payload = json.loads(msg.content)
        except (TypeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            successful.discard(msg.name)
        else:
            successful.add(msg.name)
    return successful


def build_subagent_graph(
    model,
    system_prompt: str,
    label: str,
    tool_overrides: dict = None,
    required_tools=None,
    completion_check=None,
    required_tool_guidance=None,
    max_required_nudges: int = 6,
    stop_after_required_tools: bool = False,
):
    """Build a simple agent+tools loop for a subagent (no checkpointing, no interrupt)."""
    agent_node = make_agent_node(model, system_prompt, label)

    def tools_node(state: MessagesState): # This node is only called if the agent made at least one tool call in this turn
        last = state["messages"][-1] # The agent's most recent message, which contains the tool_calls
        results = []
        for tc in (getattr(last, "tool_calls", None) or []):
            name, args, tid = tc["name"], tc.get("args", {}), tc.get("id", tc["name"]) 
            if name == "propose_experiments":
                for message in reversed(state["messages"] + results):
                    if isinstance(message, ToolMessage) and message.name == "run_bo_iteration":
                        batch = json.loads(message.content)
                        args = {**args, "suggestions": batch.get("primaries") or batch.get("suggestions") or [],
                                "reserves": batch.get("reserves") or []} # use recorded BO candidates
                        break
            print(f"\n[{label}/Tool] {name}({json.dumps(args)[:200]})")
            if tool_overrides and name in tool_overrides:
                content = str(tool_overrides[name](args))
            else:
                content = _dispatch(name, args) # Calls the actual tool function based on the name and args, returns the result as a string
            print(f" → {content[:200]}")
            results.append(ToolMessage(content=content, tool_call_id=tid, name=name)) # Convert the tool call result into a ToolMessage to be processed by the agent node on the next turn
        return {"messages": results}

    def route(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            if stop_after_required_tools:
                required = required_tools(state) if callable(required_tools) else required_tools
                required = set(required or ())
                seen = _successful_tool_names(state["messages"])

                # Pool-mode completion is authoritative: it verifies a valid BO
                # result, the requested injection split, and a stored hypothesis.
                # A model may legitimately carry the previous hypothesis forward
                # instead of calling update_hypothesis again in this iteration.
                # Requiring every historical tool name here would let weaker local
                # models loop forever on optional diversity calls after the batch is
                # already final. 
                # NOTE: Without a completion callback, retain the stricter transcript-based check.

                workflow_complete = (
                    bool(completion_check(state))
                    if completion_check
                    else bool(required) and required.issubset(seen)
                )
                if workflow_complete:
                    print(f"[{label}] Required workflow complete; ignoring optional "
                          "follow-up tool calls.")
                    return END
            return "tools"
        msgs = state["messages"]
        required = required_tools(state) if callable(required_tools) else required_tools
        required = set(required or ())
        if required:
            seen = _successful_tool_names(msgs)
            calls_present = required.issubset(seen)
            calls_succeeded = bool(completion_check(state)) if completion_check else True
            if not (calls_present and calls_succeeded):
                n_nudges = sum(
                    isinstance(msg, HumanMessage) and
                    str(msg.content).startswith(NUDGE_MSG)
                    for msg in msgs
                )
                if n_nudges >= max_required_nudges:
                    print(f"[{label}/WARN] Required workflow still incomplete after "
                          f"{n_nudges} nudges; missing={sorted(required - seen)}.")
                    return END
                return "nudge"
        # A subagent should stop once it has used tools and returned a plain-text summary.
        if any(isinstance(msg, ToolMessage) for msg in msgs):
            return END
        if len(msgs) >= 2 and isinstance(msgs[-2], HumanMessage) and msgs[-2].content == NUDGE_MSG:
            return END
        return "nudge"

    def nudge_node(state: MessagesState):
        required = required_tools(state) if callable(required_tools) else required_tools
        required = set(required or ())
        seen = _successful_tool_names(state["messages"])
        missing = sorted(required - seen)
        suffix = (f" Required tools not yet completed: {missing}."
                  if required else "")
        if required_tool_guidance:
            guidance = (required_tool_guidance(missing, state)
                        if callable(required_tool_guidance)
                        else str(required_tool_guidance))
            if guidance:
                suffix += " " + guidance.strip()
        print(f"[{label}/WARN] Workflow incomplete — nudging.{suffix}")
        return {"messages": [HumanMessage(content=NUDGE_MSG + suffix)]}

    g = StateGraph(MessagesState) # The state is just the conversation history (list of messages), which gets updated by the agent and tools nodes each turn
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("nudge", nudge_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "nudge": "nudge", END: END})
    g.add_edge("nudge", "agent")
    g.add_edge("tools", "agent")
    return g.compile()


def make_delegate_tools(base_model, context_store: dict, campaign_prompts=None):
    """Build the call_analyzer and call_bo_specialist delegate tools."""
    campaign_prompts = campaign_prompts or CampaignPrompts.from_environment()
    def _with_context(task: str):
        ctx = context_store.get("value", "")
        if not ctx:
            return task
        return f"[Literature Context]\n{ctx}\n\n[Task]\n{task}"

    method = CampaignMethod.from_environment()
    pure_bo = method is not None and method.n_agent == 0
    analyzer_tools = [tool for tool in ANALYZER_TOOLS
                      if not pure_bo or tool.name not in {'search_literature', 'search_chemrxiv'}]
    analyzer_model = base_model.bind_tools(analyzer_tools) # The Analyzer needs access to all the analysis tools, but the BO Specialist only needs call_analyzer
    analyzer_graph = build_subagent_graph(analyzer_model, campaign_prompts.analyzer, "Analyzer") # The Analyzer agent can analyze the campaign state and answer questions about it, but it doesn't make BO decisions
    @tool
    def call_analyzer(task: str) -> str:
        """Delegate a data-analysis or reporting task to the Analyzer agent.
        It has access to: search_literature, read_dataset, analyze_space_coverage,
        get_current_pareto_front, analyze_final_results, write_report.
        Describe the full task in the `task` string including relevant file paths."""
        result = analyzer_graph.invoke({"messages": [HumanMessage(content=_with_context(task))]})
        last = result["messages"][-1]
        return last.content if isinstance(last.content, str) else json.dumps(last.content)

    def _call_analyzer_or_recover(args: dict):
        """Recover common BO-specialist tool-call mixups from weaker local models."""
        if "task" in args:
            return call_analyzer.invoke({"task": args["task"]})

        if "config" in args and "volume_uL" in args:
            print("[BO-Specialist/WARN] Recovered malformed call_analyzer -> run_bo_iteration")
            return _dispatch("run_bo_iteration", args)

        if "suggestions" in args:
            print("[BO-Specialist/WARN] Recovered malformed call_analyzer -> check_suggestion_diversity")
            return _dispatch("check_suggestion_diversity", args)

        return json.dumps({
            "error": "call_analyzer requires a task string",
            "received_args": args,
        })

    # Bound specialist model retries when agent features are enabled.
    # Robot heartbeat waits remain independent of the LLM timeout.
    bo_base = base_model
    if agentic_axes_on() and base_model.__class__.__name__ == "ChatOpenAI":
        try:
            _to = float(os.getenv("OCEAN_BO_LLM_TIMEOUT", "120"))
            _rt = int(os.getenv("OCEAN_BO_LLM_RETRIES", "1"))
            bo_base = ChatOpenAI(
                model=base_model.model_name,
                base_url=base_model.openai_api_base,
                api_key=base_model.openai_api_key,
                timeout=_to, max_retries=_rt,
                max_tokens=getattr(base_model, "max_tokens", 4096),
            )
            print(f"[BO-Specialist] fast LLM client: timeout={_to}s, max_retries={_rt}")
        except Exception as e:
            print(f"[BO-Specialist] could not build fast client ({e}); using base model.")
            bo_base = base_model
    # Pure BO exposes only the tools needed to prepare and check ALICE's batch.
    _bo_tools = [tool for tool in BO_TOOLS if not pure_bo or tool.name in {
        'list_available_configs', 'prepare_bo_dataset', 'run_bo_iteration',
        'check_suggestion_diversity',
    }] + [call_analyzer]
    if os.getenv("OCEAN_BO_FILTER"):
        _bo_tools.append(filter_suggestions)
    # Injection modes ground the agent's proposals in its hypothesis.
    if os.getenv("OCEAN_BO_HYPOTHESIS") or os.getenv("OCEAN_BO_INJECT"):
        _bo_tools.append(update_hypothesis)
    if os.getenv("OCEAN_BO_INJECT"):
        _bo_tools.append(propose_experiments)
    bo_model = bo_base.bind_tools(_bo_tools) # The BO Specialist needs access to the BO tools plus call_analyzer so it can query the Analyzer for insights before making decisions
    def required_bo_tools(state):
        method = CampaignMethod.from_environment()
        if method is None:
            return None
        return {'run_bo_iteration', 'propose_experiments'} if method.n_agent else {'run_bo_iteration'}

    bo_graph = build_subagent_graph(
        bo_model, campaign_prompts.bo, "BO-Specialist",
        tool_overrides={"call_analyzer": _call_analyzer_or_recover},
        required_tools=required_bo_tools,
        required_tool_guidance="Complete the tools required by the configured mode. Pure BO retains all six primaries; injection modes require successful propose_experiments.",
    )

    @tool
    def call_bo_specialist(task: str) -> str:
        """Delegate a BO decision task to the BO Specialist agent.
        It queries the Analyzer, then runs the fixed T5Gemma / qLogNParEGO configuration.
        Provide db_path, volume_uL, current BO iteration index, and n_observed."""
        _t0 = time.time()
        result = bo_graph.invoke({"messages": [HumanMessage(content=_with_context(task))]}) # The BO Specialist subgraph is invoked with the task description, which may include references to the current campaign state and a request for a BO recommendation. The subgraph processes the task using the BO Specialist agent and any necessary tool calls (including call_analyzer if it needs insights from the Analyzer), and produces a final response.
        _wall_s = round(time.time() - _t0, 2)
        last = result["messages"][-1]

        # Per-decision agentic metrics (cost + reliability + faithfulness) for the
        # paper pipeline. Cheap to compute; surfaced in the structured result and
        # persisted to the audit log by progress_tracker_node when OCEAN_BO_MODE set.
        _msgs = result["messages"]
        n_tool_calls = sum(1 for m in _msgs if isinstance(m, ToolMessage))
        n_llm_calls = sum(1 for m in _msgs if isinstance(m, AIMessage))
        total_tokens = 0
        for m in _msgs:
            um = getattr(m, "usage_metadata", None) or {}
            total_tokens += int(um.get("total_tokens", 0) or 0)

        run_payload = {}
        diversity_payload = {}
        filter_payload = {}
        inject_payload = {}
        for msg in result["messages"]:
            if not isinstance(msg, ToolMessage):
                continue
            try:
                payload = json.loads(msg.content)
            except Exception:
                payload = None
            if not isinstance(payload, dict):
                continue
            if msg.name == "run_bo_iteration":
                run_payload = payload
                inject_payload, filter_payload = {}, {} # a new BO run invalidates earlier selections
            elif msg.name == "check_suggestion_diversity":
                diversity_payload = payload
            elif msg.name == "filter_suggestions":
                filter_payload = payload
            elif msg.name == "propose_experiments" and not payload.get("no_op"):
                inject_payload = payload

        if run_payload.get('error') or run_payload.get('abort') or run_payload.get('returncode') not in (None, 0):
            return json.dumps({'error': run_payload.get('error') or run_payload.get('stderr') or '[method] BO worker failed', 'abort': True})
        if run_payload:
            # Batch precedence: injection (BO picks + agent-proposed) beats the
            # chemistry filter (selection among BO's own candidates), which beats
            # BO's raw batch. Injection is the later, more specific decision.
            suggestions = (inject_payload.get("suggestions")
                           or filter_payload.get("suggestions")
                           or run_payload.get("suggestions")
                           or run_payload.get("primaries") or [])
            method = CampaignMethod.from_environment()
            structured = {
                "json_path": run_payload.get("json_path") or run_payload.get("proto_path"),
                "proto_path": run_payload.get("proto_path") or run_payload.get("json_path"),
                "suggestions": suggestions,
                "bo_primaries": run_payload.get("primaries") or run_payload.get("suggestions") or [],
                "config": run_payload.get("config"),
                "mean_lengthscale": run_payload.get("mean_lengthscale"),
                "diversity": diversity_payload,
                "filter": filter_payload,
                # Keep the snapped LLM points in the selection and audit record.
                "inject": inject_payload,
                "n_agent_injected": inject_payload.get("n_agent_injected", 0 if method and not method.n_agent else None),
                "inject_points": inject_payload.get("agent_points"),
                # Current hypothesis text (persisted separately in hypothesis.json).
                "hypothesis": _current_hypothesis(),
                "summary": last.content if isinstance(last.content, str) else json.dumps(last.content),
                # agentic metrics (cost / reliability / faithfulness)
                "wall_s": _wall_s,
                "n_tool_calls": n_tool_calls,
                "n_llm_calls": n_llm_calls,
                "total_tokens": total_tokens,
                "filter_overrode_bo": bool(filter_payload.get("overrode_bo")),
                "reasoning": (last.content if isinstance(last.content, str)
                              else json.dumps(last.content))[:1000],
            }
            if method:
                try:
                    method.validate_selection(structured)
                except ValueError as error:
                    return json.dumps({'error': str(error), 'abort': True})
                structured['campaign_mode'] = method.mode
            return json.dumps(structured)

        if CampaignMethod.from_environment():
            return json.dumps({'error': '[method] No recorded BO result for the configured campaign mode', 'abort': True})
        return last.content if isinstance(last.content, str) else json.dumps(last.content)

    return [call_analyzer, call_bo_specialist]
