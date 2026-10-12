# Execute a suggestion batch through the robot mailbox.
import sys, logging
from agent_lg.scripts.io_mailbox import run_experiment
from agent_lg.services.protocols import load_protocol, robot_experiments
from agent_lg.services.storage import WorkerRun

logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m agent_lg.scripts.robot <suggestions_json_path>")
        sys.exit(1)
    json_path = sys.argv[1]
    try:
        data, _ = load_protocol(json_path)
        suggestions = robot_experiments(data)
    except (OSError, TypeError, ValueError) as error:
        print(f"[robot] Invalid protocol: {error}")
        sys.exit(1)

    print(f"[robot] Loaded {len(suggestions)} suggestions from {json_path}")
    for index, experiment in enumerate(suggestions, 1):
        print(f"[robot] Running experiment {index}/{len(suggestions)}: {experiment}")
        try:
            snapshot = run_experiment(experiment, index=index)
        except TimeoutError as error:
            print(f"[robot] STALL: {error}")
            sys.exit(75) # uncertain hardware state must not be treated as an experiment failure
        if isinstance(snapshot, dict) and not snapshot.get("ok", True):
            error = (snapshot.get("result") or {}).get("error", "")
            print(f"[robot] Experiment {index} FAILED on the instrument "
                  f"(state={snapshot.get('state')!r}): {error!r} — no result row will be written for it.")
        else:
            print(f"[robot] Experiment {index} done.")
    print("[robot] All experiments complete.")
    sys.exit(0)


if __name__ == "__main__":
    with WorkerRun("robot"):
        main()
