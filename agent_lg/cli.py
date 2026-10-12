# Parse campaign commands; execution and recovery live in runtime.py.

import argparse, json, os, sys, logging, shlex, subprocess
from pathlib import Path
from dotenv import load_dotenv
from agent_lg.config import CHECKPOINT_DB, AVAILABLE_CONFIGS
from agent_lg.runtime import run_agent, resume_agent, campaign_session
from agent_lg.preflight import PreflightError
from agent_lg.services.validation import CampaignSafetyError
from agent_lg.services.method import CampaignMethod
from agent_lg.services.status import campaign_status

logger = logging.getLogger(__name__)


@campaign_session
def main():
    load_dotenv(Path.cwd() / ".env")
    parser = argparse.ArgumentParser(
        description="LangGraph SDL agent for closed-loop BZ reaction MOBO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default=None,
                        help="Path to experiments DB txt file")
    parser.add_argument("--campaign-mode", choices=list(CampaignMethod.modes), default=None,
                        help="bo: 6 BO; bo-llm: 3 BO + 3 LLM (default); llm: 6 LLM. Resume uses the recorded mode.")
    parser.add_argument("--papers", default=None, help="Local campaign papers (PDF/TXT/MD); required for LLM selection modes, optional for bo")
    parser.add_argument("--volume", type=float, default=1900,
                        help="Target mixture volume in µL")
    parser.add_argument("--n_bo_iters", type=int, default=5,
                        help="Number of BO iterations to run")
    parser.add_argument("--backend", default=None,
                        choices=["anthropic", "openai"],
                        help="LLM backend")
    parser.add_argument("--model", default=None,
                        help="Model name (default: claude-opus-4-6 / qwen2.5:14b)")
    parser.add_argument("--base-url", default=None,
                        help="Base URL for OpenAI-compatible backend")
    parser.add_argument("--api-key", default="ollama",
                        help="API key for OpenAI-compatible backend")
    parser.add_argument("--thread-id", default=None,
                        help="Thread ID to resume an existing campaign")
    parser.add_argument('--status', action='store_true', help='Inspect saved campaign recovery state without changing any files')
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing campaign (requires --thread-id)")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument("--approve", action="store_true", default=None,
                        help="When resuming an interrupted submission: approve it")
    approval.add_argument("--reject", action="store_true", default=None,
                        help="When resuming an interrupted submission: reject it")
    parser.add_argument("--auto-approve", action="store_true", default=None,
                        help="Skip human approval gate (for dry-runs and testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock result waiting and block live mailbox commands; use a fake controller for full pipeline tests")
    parser.add_argument("--reward-mode", choices=["off", "recompute", "freeze"], default=None,
                        help="LIVE reward calc after each batch from raw interp_*: "
                             "'recompute' refits the z-norm over ALL observed each batch; "
                             "'freeze' fits it ONCE on the first batch and reuses it. "
                             "'off' (default) leaves rewards untouched (dummy/test runs).")
    parser.add_argument("--single-pool", action="store_true",
                        help="Compatibility flag: BO always uses the explicit --db campaign pool.")
    parser.add_argument("--lock-config", default=None,
                            choices=['t5gemma-s'],
                            help="Compatibility option; campaigns already use the fixed t5gemma-s configuration.")
    parser.add_argument("--checkpoint-db", default=CHECKPOINT_DB,
                    help="Path to SQLite checkpoint database")
    parser.add_argument("--recursion-limit", type=int, default=None,
                    help="Maximum number of graph execution steps before raising recursion error")

    args = parser.parse_args()
    if args.status:
        if not args.thread_id:
            parser.error('--status requires --thread-id')
        report = campaign_status(args.checkpoint_db, args.thread_id)
        print(json.dumps(report, indent=2))
        return

    if not args.resume:
        args.backend = args.backend or 'anthropic'
        args.base_url = args.base_url or 'http://localhost:11434/v1'
        args.reward_mode = args.reward_mode or 'off'
        args.recursion_limit = args.recursion_limit or 1000
        args.auto_approve = bool(args.auto_approve)
    if args.model is None and not args.resume:
        args.model = "claude-opus-4-6" if args.backend == "anthropic" else "qwen2.5:14b"

    # Keep existing command lines valid; run_agent also binds its explicit db_path.
    if args.single_pool:
        if not args.db:
            parser.error("--single-pool requires --db (the campaign pool file)")
        os.environ["OCEAN_CAMPAIGN_DB"] = args.db
        print(f"[single-pool] BO will suggest from --db pool: {args.db}")

    if args.lock_config:
        if args.lock_config not in AVAILABLE_CONFIGS:
            parser.error(f"--lock-config {args.lock_config!r} not in AVAILABLE_CONFIGS."
                        f"Valid keys: {sorted(AVAILABLE_CONFIGS.keys())}")
        os.environ["OCEAN_LOCK_CONFIG"] = args.lock_config
        print(f"[lock-config] Using the fixed {args.lock_config!r} configuration.")

    # Volume for the deterministic BO fallback (graph._bo_fallback) so it can run
    # run_bo_iteration without the agent's task string. Harmless when unused.
    os.environ["OCEAN_VOLUME_UL"] = str(args.volume)

    if args.resume:
        if not args.thread_id:
            parser.error("--resume requires --thread-id")
        human_input = False if args.reject else True if args.approve else None
        try:
            resume_agent(
                thread_id = args.thread_id,
                human_input = human_input,
                model_name = args.model,
                backend = args.backend,
                base_url = args.base_url,
                api_key = args.api_key,
                auto_approve = args.auto_approve,
                checkpoint_db = args.checkpoint_db,
                reward_mode = args.reward_mode,
                db_path = args.db,
                recursion_limit = args.recursion_limit,
                campaign_mode = args.campaign_mode,
                papers_dir = args.papers,
            )
        except PreflightError as e:
            print(str(e), file=sys.stderr)
            raise SystemExit(2)
        except CampaignSafetyError as error:
            print(f"[PAUSED] {error}\nCheckpoint and batch records retained. Resolve the reported issue before resuming.", file=sys.stderr)
            raise SystemExit(75)
        return

    if not args.db:
        parser.error("--db is required for a new campaign")

    db = args.db
    task = (
        f"Run a closed-loop BZ reaction MOBO campaign.\n"
        f"Experiments DB : {db}\n"
        f"Volume : {args.volume} µL\n"
        f"BO iterations : {args.n_bo_iters}\n\n"
        f"Follow the configured workflow: dataset QC, design space validation, "
        f"fixed T5Gemma + qLogNParEGO with Sobol initialization, "
        f"the configured 6 BO, 3 BO + 3 LLM or 6 LLM selection + diversity check, then run all {args.n_bo_iters} iterations "
        f"with per-result assessment and end with full analysis + written report."
    )

    # A misconfigured axis is an operator error, not a crash — show the explanation
    # and exit, without a traceback.
    try:
        thread_id = run_agent(
            user_message = task,
            db_path = args.db,
            model_name = args.model,
            backend = args.backend,
            base_url = args.base_url,
            api_key = args.api_key,
            thread_id = args.thread_id,
            auto_approve = args.auto_approve,
            checkpoint_db = args.checkpoint_db,
            dry_run = args.dry_run,
            reward_mode = args.reward_mode,
            recursion_limit = args.recursion_limit,
            n_bo_iters = args.n_bo_iters,
            campaign_mode = args.campaign_mode,
            papers_dir = args.papers,
        )
    except PreflightError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)

    except CampaignSafetyError as error:
        print(f"[PAUSED] {error}\nCheckpoint and batch records retained. Resolve the reported issue before resuming.", file=sys.stderr)
        raise SystemExit(75)

    arguments = ['python', '-m', 'agent_lg', '--resume', '--thread-id', thread_id,
                 '--checkpoint-db', str(Path(args.checkpoint_db).expanduser().resolve())]
    command = subprocess.list2cmdline(arguments) if os.name == 'nt' else shlex.join(arguments)
    print(f"\nTo resume this campaign later:\n{command} --approve\n{command} --reject\n")

