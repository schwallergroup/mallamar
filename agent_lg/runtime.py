# Start and resume campaigns with scoped settings and durable ownership.

import hashlib, os, sys, uuid, logging, threading
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from agent_lg import config as agent_config
from agent_lg.config import CHECKPOINT_DB, AVAILABLE_CONFIGS, configure_configs
from agent_lg.graph import build_graph
from agent_lg.preflight import check_agentic_config, PreflightError
from agent_lg.tools_registry import TOOL_MAP
from agent_lg.services.validation import CampaignSafetyError
from agent_lg.services.storage import FileLock
from agent_lg.services.campaigns import CampaignSettings, CAMPAIGN_ENVIRONMENT
from agent_lg.services.method import CampaignRecipes, CampaignMethod
from agent_lg.services.pools import check_campaign_pool
from agent_lg.services.logging import setup_logging, cleanup_logging

logger = logging.getLogger(__name__)


class CampaignSession:
    """Keep environment, configuration paths and console routing scoped to one call."""
    controller = threading.RLock()

    def __enter__(self):
        if not self.controller.acquire(blocking=False):
            raise CampaignSafetyError('[session] Another agent is active in this Python process; run parallel campaigns in separate processes')
        try:
            keys = (*CAMPAIGN_ENVIRONMENT, 'MALLAMAR_ARTIFACTS_RECORD', 'MALLAMAR_MODEL_REVISIONS', 'MALLAMAR_DRY_RUN', 'OCEAN_BO_ITER')
            self.environment = {key: os.getenv(key) for key in keys}
            self.configs_dir = agent_config.CONFIGS_DIR
            self.configs = [(name, model_config, model_config['path']) for name, model_config in AVAILABLE_CONFIGS.items()]
            self.wait_tool = TOOL_MAP['wait_for_result']
            self.stdout, self.stderr = sys.stdout, sys.stderr
            return self
        except BaseException:
            self.controller.release()
            raise

    def __exit__(self, *error):
        try:
            for key, value in self.environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            agent_config.CONFIGS_DIR = self.configs_dir
            AVAILABLE_CONFIGS.clear()
            for name, model_config, path in self.configs:
                model_config['path'] = path
                AVAILABLE_CONFIGS[name] = model_config
            TOOL_MAP['wait_for_result'] = self.wait_tool
            if sys.stdout is not self.stdout or sys.stderr is not self.stderr:
                cleanup_logging(self.stdout, self.stderr)
        finally:
            self.controller.release()

def campaign_session(function):
    @wraps(function)
    def call(*args, **kwargs):
        with CampaignSession():
            return function(*args, **kwargs)
    return call

class CampaignLock:
    """One controller per checkpoint thread, dataset and hardware mailbox."""

    def __init__(self, checkpoint_db, thread_id, db_path):
        identifier = hashlib.sha256(thread_id.encode()).hexdigest()[:16]
        self.paths = [f"{checkpoint_db}.{identifier}", f"{db_path}.campaign"]
        if os.getenv("OCEAN_MAILBOX_ROOT"):
            self.paths.append(Path(os.environ["OCEAN_MAILBOX_ROOT"]) / "campaign_controller")
        self.stack = ExitStack()

    def __enter__(self):
        try:
            for path in self.paths:
                self.stack.enter_context(FileLock(path))
        except BaseException as error:
            self.stack.close()
            if isinstance(error, Exception):
                raise CampaignSafetyError(f"[CampaignLock] Could not acquire campaign ownership: {error}") from error
            raise # interrupts also release any locks already acquired
        return self

    def __exit__(self, *error):
        self.stack.close()


def make_base_model(backend: str, 
                    model_name: str,
                    base_url: str = "http://localhost:11434/v1",
                    api_key: str = "ollama"):
    """Return an unbound chat model (no tools attached yet)."""
    if backend == "anthropic":
        return ChatAnthropic(model=model_name, max_tokens=8192)
    elif backend == "openai":
        return ChatOpenAI(model=model_name, base_url=base_url,
                          api_key=api_key, timeout=1200, max_tokens=4096)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'anthropic' or 'openai'.")


def _pending_interrupts(state):
    """Return the interrupt payload(s) the graph is currently paused on (empty if none)."""
    payloads = []
    for task in getattr(state, "tasks", ()) or ():
        for intr in getattr(task, "interrupts", ()) or ():
            payloads.append(getattr(intr, "value", intr)) # extract the value of the interrupt if available
    return payloads


def _prompt_approval(payload):
    """Block for an inline operator y/n decision on a pending robot submission."""
    if isinstance(payload, dict):
        proto = payload.get("protocol", "?")
        msg = payload.get("message", "Approve robot submission?")
    else:
        proto, msg = payload, "Approve robot submission?"
    print(f"\n[HUMAN APPROVAL] {msg}\nProtocol: {proto}")
    while True:
        ans = input("Approve this submission? (y/n): ").strip().lower()
        if ans in ("y", "yes", "approve", "ok"):
            print("[APPROVAL] Approved.")
            return True
        if ans in ("n", "no", "reject"):
            print("[APPROVAL] Rejected.")
            return False
        print("Please type 'y' or 'n'.")


# main campain driver
@campaign_session
def run_agent(
    user_message: str, # agent instructions
    db_path: str = "", # dataset
    model_name: str = "claude-opus-4-6",
    backend: str = "anthropic",
    base_url: str = "http://localhost:11434/v1", # local Ollamma by default for easy testing
    api_key: str = "ollama",
    thread_id: str | None = None,
    auto_approve: bool = False,
    checkpoint_db: str = CHECKPOINT_DB,
    dry_run: bool = False,
    reward_mode: str = "off",
    recursion_limit: int = 1000,
    n_bo_iters: int = 0,
    approval_input=None,
    campaign_mode: str | None = None,
    papers_dir: str | None = None,
):
    """Run 6 BO (bo), 3 BO + 3 LLM (bo-llm, default) or 6 LLM (llm); resume keeps saved settings."""
    load_dotenv(Path.cwd() / ".env") # use the caller's workspace
    if checkpoint_db == ":memory:":
        raise PreflightError("[campaign] Use a checkpoint file; in-memory campaigns cannot recover after a crash")
    
    db_path = str(Path(db_path).expanduser().resolve()) if db_path else db_path
    checkpoint_db = str(Path(checkpoint_db).expanduser().resolve())
    thread_id = thread_id or str(uuid.uuid4()) 
    saved_settings = CampaignSettings(checkpoint_db, thread_id)

    try:
        record = saved_settings.load()
        saved_record_present = record is not None # check campaign already exist
        if record is not None:
            options = saved_settings.restore(record)
            db_path, user_message = options['db_path'], options['user_message']
            model_name, backend, base_url = options['model_name'], options['backend'], options['base_url']
            auto_approve, dry_run, reward_mode = options['auto_approve'], options['dry_run'], options['reward_mode']
            recursion_limit, n_bo_iters = options['recursion_limit'], options['n_bo_iters']
    except (OSError, ValueError) as error:
        raise CampaignSafetyError(f'[settings] {error}') from error
    
    try:
        if record is None:
            CampaignRecipes().check_environment()
            CampaignMethod.configure(campaign_mode)
            if papers_dir is not None:
                os.environ['OCEAN_CORPUS_DIR'] = str(Path(papers_dir).expanduser().resolve())
        else:
            method = CampaignMethod.from_environment()
            if campaign_mode is not None and (method is None or method.mode != campaign_mode):
                raise ValueError('[method] campaign_mode differs from the original campaign')
            if papers_dir is not None and str(Path(papers_dir).expanduser().resolve()) != os.getenv('OCEAN_CORPUS_DIR'):
                raise ValueError('[method] papers_dir differs from the original campaign')
    except ValueError as error:
        raise PreflightError(str(error)) from error
    
    if not db_path:
        raise PreflightError('[campaign] db_path is required for a campaign')
    pool = os.getenv('OCEAN_CAMPAIGN_DB')

    if pool and Path(pool).expanduser().resolve() != Path(db_path).expanduser().resolve():
        raise PreflightError('[campaign] OCEAN_CAMPAIGN_DB differs from db_path; provide one campaign pool')
    os.environ['OCEAN_CAMPAIGN_DB'] = db_path # the Python API and CLI share the explicit pool

    if record is None:
        os.environ.pop('MALLAMAR_ARTIFACTS_RECORD', None)
    else:
        os.environ['MALLAMAR_ARTIFACTS_RECORD'] = str(saved_settings.path)
    if record is None and not os.getenv('OCEAN_HYPOTHESIS_PATH'):
        os.environ['OCEAN_HYPOTHESIS_PATH'] = str(Path(db_path).parent / '.mallamar' / 'hypotheses' / (saved_settings.path.stem + '.json'))
    log_file, orig_stdout, orig_stderr = setup_logging(db_path)

    try:
        if dry_run: # preliminary testing 
            os.environ["MALLAMAR_DRY_RUN"] = "1" # real mailbox commands are disabled for this call
            def _mock_wait_for_result(concs, db_path, **_):
                print(f"[DRY-RUN] wait_for_result mocked — returning {len(concs)} fake result(s).")
                return {
                    "n_found": len(concs),
                    "n_missing": 0,
                    "found": [
                        {"conc": c, "exp_id": f"dry_run_{i}", "n_osc": 3,
                         "period_mean_s": 60.0, "maxima_mean_V": 0.5, "total_time_s": 600.0}
                        for i, c in enumerate(concs)
                    ],
                    "missing": [],
                    "abort": False,
                }
            TOOL_MAP["wait_for_result"] = lambda i: _mock_wait_for_result(**i)
            print("[DRY-RUN] wait_for_result patched — robot results will be mocked.")

       # Restore config paths and validate campaign inputs before building the agents.
        os.environ["OCEAN_REWARD_MODE"] = reward_mode
        configure_configs()
        check_agentic_config(db_path=db_path)

        with CampaignLock(checkpoint_db, thread_id, db_path):
            if record is None:
                with SqliteSaver.from_conn_string(checkpoint_db) as saver:
                    if saver.get({'configurable': {'thread_id': thread_id}}):
                        raise CampaignSafetyError('[settings] Legacy checkpoint has no saved settings. Reconcile its original configuration before adopting it; defaults cannot recover it safely')

                # Create a new settings record here if none exists
                record = saved_settings.create({'db_path': str(Path(db_path).expanduser().resolve()), 'user_message': user_message,
                        'model_name': model_name, 'backend': backend, 'base_url': base_url,
                        'auto_approve': auto_approve, 'dry_run': dry_run, 'reward_mode': reward_mode,
                        'recursion_limit': recursion_limit, 'n_bo_iters': n_bo_iters})
            elif saved_settings.load() != record:
                raise CampaignSafetyError('[settings] Settings changed while acquiring ownership; inspect the campaign')

            saved_settings.restore(record) # use the same absolute deployment paths on start and restart
            configure_configs()
            os.environ['MALLAMAR_ARTIFACTS_RECORD'] = str(saved_settings.path)

            print(f"AGENT START [{backend.upper()} {model_name}]")
            print(f"Thread ID: {thread_id}")
            print(f"Checkpoint: {checkpoint_db}")
            print(f"Auto-approve: {auto_approve}")
            print(f"Reward mode: {reward_mode}")

            # Create the LLM and identify the campaign for checkpointed execution.
            model = make_base_model(backend, model_name, base_url, api_key)
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}

            with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
                # * the graph is built with the model and checkpointer, ready for execution.
                graph = build_graph(model, auto_approve=auto_approve, checkpointer=checkpointer)

                existing = checkpointer.get(config)
                if saved_record_present and not existing:
                    try:
                        saved_settings.require_checkpoint_if_submitted(record)
                    except (OSError, ValueError) as error:
                        raise CampaignSafetyError(f'[settings] {error}') from error
                    
                if existing:
                    values = graph.get_state(config).values # include completed node writes awaiting the next SQLite checkpoint
                    recorded_db = values.get('db_path')
                    if recorded_db is not None and Path(recorded_db).resolve() != Path(db_path).expanduser().resolve():
                        raise CampaignSafetyError('[settings] Dataset differs from the saved checkpoint')
                    try:
                        CampaignRecipes().check_runtime(values.get('runtime_environment', {}))
                    except ValueError as error:
                        raise CampaignSafetyError(str(error)) from error
                    try:
                        check_campaign_pool(db_path, values.get('campaign_exp_ids', []),
                                            pending_batch=bool(values.get('pending_concs')))
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                        raise CampaignSafetyError(f'[pool_integrity] {error}') from error
                    print(f"[INFO] Resuming existing campaign (thread {thread_id})")

                initial_input = {
                    "messages": [HumanMessage(content=user_message)],
                    "literature_context": "",
                    "campaign_id": thread_id,
                    "bo_iter": 0,
                    "pareto_hv_history": [],
                    "db_path": db_path,
                    # Keep the iteration target in state for agent prompts and recovery.
                    "target_bo_iters": n_bo_iters,
                }

                # If the thread is already paused at a human-approval interrupt 
                # (like the operator quit and re-ran), resume from there; otherwise start fresh.
                pending = _pending_interrupts(graph.get_state(config)) if existing else []
                next_input = (Command(resume=approval_input if approval_input is not None else _prompt_approval(pending[0])) if pending
                              else None if existing else initial_input) # resume the checkpoint

                # Drive the graph, pausing for an inline operator y/n on every interrupt
                # (unless --auto-approve, in which case the gate never fires).
                while True:
                    for _ in graph.stream(next_input, config=config, stream_mode="values"):
                        pass
                    pending = _pending_interrupts(graph.get_state(config))
                    if not pending:
                        break # graph completed — nothing awaiting approval
                    next_input = Command(resume=_prompt_approval(pending[0]))

                print(f"\nAGENT DONE (thread: {thread_id})")
                return thread_id

    finally:
        cleanup_logging(orig_stdout, orig_stderr)
        try:
            orig_stdout.write(f"[Logging] Console output saved to: {log_file}\n")
        except (OSError, ValueError):
            pass # optional logging must not replace a campaign failure



def resume_agent(
    thread_id: str,
    human_input=None,
    model_name: str = None,
    backend: str = None,
    base_url: str = None,
    api_key: str = "ollama",
    auto_approve: bool = None,
    checkpoint_db: str = CHECKPOINT_DB,
    reward_mode: str = None,
    db_path: str = None,
    recursion_limit: int = None,
    campaign_mode: str | None = None,
    papers_dir: str | None = None,
):
    """Resume using saved settings; explicit method changes are rejected."""
    saved_settings = CampaignSettings(checkpoint_db, thread_id)
    try:
        record = saved_settings.load()
        if record is None:
            raise ValueError('No saved campaign settings; legacy checkpoints need their original configuration verified')
        options = record['settings']
        method = CampaignMethod.from_environment(record['environment'])
        if campaign_mode is not None and (method is None or method.mode != campaign_mode):
            raise ValueError('[method] campaign_mode differs from the original campaign')
        if papers_dir is not None and str(Path(papers_dir).expanduser().resolve()) != record['environment'].get('OCEAN_CORPUS_DIR'):
            raise ValueError('[method] papers_dir differs from the original campaign')
        overrides = {'model_name': model_name, 'backend': backend, 'base_url': base_url,
                     'auto_approve': auto_approve, 'reward_mode': reward_mode,
                     'db_path': db_path, 'recursion_limit': recursion_limit}
        for key, value in overrides.items():
            if key == 'db_path' and value is not None:
                value = str(Path(value).expanduser().resolve())
            if value is not None and value != options[key]:
                raise ValueError(f'{key} differs from the original campaign; resume uses {options[key]!r}')
    except (OSError, ValueError) as error:
        raise CampaignSafetyError(f'[settings] {error}') from error
    return run_agent(**options, thread_id=thread_id, checkpoint_db=checkpoint_db,
                     api_key=api_key, approval_input=human_input)

