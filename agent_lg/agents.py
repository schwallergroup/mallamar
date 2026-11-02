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


# Limit model context while retaining full checkpoint history and refreshing current state.
ORCH_CONTEXT_WINDOW = int(os.getenv("OCEAN_CONTEXT_WINDOW", "20"))


def _window_messages(msgs: list, keep: int = None):
    """Keep the task and recent messages without orphaning tool results."""
    keep = ORCH_CONTEXT_WINDOW if keep is None else keep
    if keep <= 0 or len(msgs) <= keep + 1:
        return msgs, 0

    start = len(msgs) - keep
    # Keep each retained tool result with its parent model message.
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
    # Pareto improvement means a new front member, even if recomputed hypervolume decreases.
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

    # Ground hypothesis-driven selection in literature, observations and prior reasoning.
    _hyp_on = bool(os.getenv("OCEAN_BO_HYPOTHESIS")) or bool(os.getenv("OCEAN_BO_INJECT"))
    _inject_on = bool(os.getenv("OCEAN_BO_INJECT"))
    if _hyp_on:
        db = state.get("db_path", "") or os.getenv("OCEAN_CAMPAIGN_DB", "")
        current = _H.store().text
        # Include measured outcomes to ground proposals in observed chemistry.
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
                "corpus AND the observed data above. Save the prior even at zero observations.",
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
                "Call update_hypothesis every iteration, including when the evidence "
                "supports retaining the text unchanged.",
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

        # Restate the iteration budget each turn so the orchestrator can enforce completion.
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
            # Pass measured progress and policy signals from the orchestrator to the BO specialist.
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

        # Window orchestrator messages without changing checkpoint or audit history.
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

        sys_content = system_prompt + ("\n\n" + "\n\n".join(extras) if extras else "") # Add current campaign context to the system prompt.

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
        if msg.name == "update_hypothesis" and (not isinstance(payload, dict) or payload.get("stored") is not True):
            successful.discard(msg.name)
        elif isinstance(payload, dict) and payload.get("error"):
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
    allowed_tools=None,
):
    """Build a simple agent+tools loop for a subagent (no checkpointing, no interrupt)."""
    agent_node = make_agent_node(model, system_prompt, label)
    nudge_message = "Call the next available tool for your task. When your task is complete, return a final answer."
    allowed_names = set(allowed_tools or ())
    if allowed_tools is None:
        for available_tool in ANALYZER_TOOLS + BO_TOOLS + [filter_suggestions, update_hypothesis, propose_experiments]:
            allowed_names.add(available_tool.name)
        allowed_names.update(tool_overrides or {})

    def tools_node(state: MessagesState): # This node is only called if the agent made at least one tool call in this turn
        last = state["messages"][-1] # The agent's most recent message, which contains the tool_calls
        results = []
        for tc in (getattr(last, "tool_calls", None) or []):
            name, args, tid = tc["name"], tc.get("args", {}), tc.get("id", tc["name"])
            if name not in allowed_names:
                results.append(ToolMessage(content=json.dumps({"error": f"{name} is not available to {label}"}),
                                           tool_call_id=tid, name=name))
                continue
            if name == "propose_experiments":
                for message in reversed(state["messages"] + results):
                    if isinstance(message, ToolMessage) and message.name == "run_bo_iteration":
                        batch = json.loads(message.content)
                        args = {**args, "suggestions": batch.get("primaries") or batch.get("suggestions") or [],
                                "reserves": batch.get("reserves") or []} # use recorded BO candidates
                        break
            if name == "check_suggestion_diversity":
                for message in reversed(state["messages"] + results):
                    if not isinstance(message, ToolMessage) or message.name not in {"propose_experiments", "run_bo_iteration"}:
                        continue
                    batch = json.loads(message.content)
                    failed = (batch.get("error") or batch.get("abort") or batch.get("no_op")
                              or batch.get("returncode") not in (None, 0))
                    if message.name == "propose_experiments" and failed:
                        continue
                    args = {"suggestions": []}
                    if not failed:
                        if message.name == "propose_experiments":
                            points = batch.get("suggestions", [])
                        else:
                            points = batch.get("primaries") or batch.get("suggestions", [])
                        for point in points:
                            args["suggestions"].append(point.get("conc", point))
                    break # the latest BO run bounds the current batch
            print(f"\n[{label}/Tool] {name}({json.dumps(args)[:200]})")
            if tool_overrides and name in tool_overrides:
                content = str(tool_overrides[name](args))
            else:
                content = _dispatch(name, args) # Dispatch the requested tool.
            print(f" → {content[:200]}")
            results.append(ToolMessage(content=content, tool_call_id=tid, name=name)) # Return the tool result to the agent.
        return {"messages": results}

    def route(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            if stop_after_required_tools:
                required = required_tools(state) if callable(required_tools) else required_tools
                required = set(required or ())
                seen = _successful_tool_names(state["messages"])

                # Use the completion callback when supplied; otherwise require successful tool calls.

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
                    str(msg.content).startswith(nudge_message)
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
        if len(msgs) >= 2 and isinstance(msgs[-2], HumanMessage) and str(msgs[-2].content).startswith(nudge_message):
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
        return {"messages": [HumanMessage(content=nudge_message + suffix)]}

    g = StateGraph(MessagesState) # Conversation messages form the subagent state.
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
    analyzer_model = base_model.bind_tools(analyzer_tools) # Bind the Analyzer's permitted tools.
    analyzer_names = []
    for available_tool in analyzer_tools:
        analyzer_names.append(available_tool.name)
    analyzer_graph = build_subagent_graph(analyzer_model, campaign_prompts.analyzer, "Analyzer",
                                         allowed_tools=analyzer_names)
    @tool
    def call_analyzer(task: str) -> str:
        """Delegate analysis or reporting; include the dataset path and task."""
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

    # Bound specialist LLM calls independently of robot heartbeat waits.
    bo_base = base_model
    if agentic_axes_on() and base_model.__class__.__name__ == "ChatOpenAI":
        timeout = float(os.getenv("OCEAN_BO_LLM_TIMEOUT", "120"))
        retries = int(os.getenv("OCEAN_BO_LLM_RETRIES", "1"))
        bo_base = ChatOpenAI(
            model=base_model.model_name,
            base_url=base_model.openai_api_base,
            api_key=base_model.openai_api_key,
            timeout=timeout, max_retries=retries,
            max_tokens=getattr(base_model, "max_tokens", 4096),
        )
        print(f"[BO-Specialist] fast LLM client: timeout={timeout}s, max_retries={retries}")
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
    bo_model = bo_base.bind_tools(_bo_tools) # Bind BO tools and Analyzer delegation.
    def required_bo_tools(state):
        method = CampaignMethod.from_environment()
        if method is None:
            return None
        if method.n_agent:
            return {'update_hypothesis', 'run_bo_iteration', 'propose_experiments'}
        return {'run_bo_iteration'}

    def hypothesis_saved(state):
        method = CampaignMethod.from_environment()
        if method is None or not method.n_agent:
            return True
        history = _H.store().history
        return (bool(history) and history[-1].get('bo_iter') == int(os.getenv('OCEAN_BO_ITER', '0'))
                and 'update_hypothesis' in _successful_tool_names(state['messages']))

    bo_names = []
    for available_tool in _bo_tools:
        bo_names.append(available_tool.name)
    bo_graph = build_subagent_graph(
        bo_model, campaign_prompts.bo, "BO-Specialist", allowed_tools=bo_names,
        tool_overrides={"call_analyzer": _call_analyzer_or_recover},
        required_tools=required_bo_tools,
        completion_check=hypothesis_saved,
        required_tool_guidance="Pure BO retains all six primaries. LLM modes must save a hypothesis for this iteration (unchanged text is allowed) and successfully propose_experiments.",
    )

    @tool
    def call_bo_specialist(task: str) -> str:
        """Delegate batch selection with the pool path, volume, iteration and observation count."""
        _t0 = time.time()
        result = bo_graph.invoke({"messages": [HumanMessage(content=_with_context(task))]}) # Run the specialist workflow for this task.
        if not hypothesis_saved(result):
            return json.dumps({'error': '[hypothesis] Save a hypothesis for the current iteration before accepting a batch', 'abort': True})
        _wall_s = round(time.time() - _t0, 2)
        last = result["messages"][-1]

        # Collect per-decision usage and reliability metrics for the campaign audit.
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
            # Prefer injected candidates, then filtered candidates, then the original BO batch.
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
