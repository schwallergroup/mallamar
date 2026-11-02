# Run BO in a separate process and read its campaign output.
import math, os, subprocess, sys, logging, time
from pathlib import Path
from agent_lg.services.profiling import profile_step
from agent_lg.services.protocols import load_protocol, select_protocol, validate_concentrations
from agent_lg.services.storage import atomic_write, file_revision
from agent_lg.services.artifacts import check_recipe
from agent_lg.services.campaigns import PATH_SETTINGS
from agent_lg.services.method import CampaignRecipes

logger = logging.getLogger(__name__)


def worker_diagnostic(results_root, stdout, stderr):
    path = Path(results_root) / ".mallamar" / f"bo-worker-{os.getpid()}-{time.time_ns()}.log"
    try:
        atomic_write(path, lambda temporary: temporary.write_text(f"STDOUT\n{stdout}\nSTDERR\n{stderr}", encoding="utf-8"))
        return str(path)
    except OSError as error:
        logger.warning(f"[BOProcessRunner] Could not retain worker diagnostics: {error}")
        return None


class BOProcessRunner:
    """Keeps model fitting isolated from the agent process."""

    def run(self, config, data_path, volume_uL, results_root, config_name):
        try:
            CampaignRecipes().check_environment()
            config = str(CampaignRecipes().resolve(config) if os.getenv('OCEAN_CAMPAIGN_DB') else Path(config).resolve())
            data_path = str(Path(data_path).resolve())
            results_root = str(Path(results_root).resolve())
            if not Path(config).is_file() or not Path(data_path).is_file():
                raise ValueError("BO config and dataset must be existing files")
            if not math.isfinite(float(volume_uL)) or float(volume_uL) <= 0:
                raise ValueError("BO volume must be finite and positive")
            dataset_revision = file_revision(data_path)
            check_recipe(config, data_path)
            timeout = os.getenv("MALLAMAR_BO_TIMEOUT_S")
            timeout = float(timeout) if timeout else None
            if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
                raise ValueError("MALLAMAR_BO_TIMEOUT_S must be finite and positive")
        except (OSError, TypeError, ValueError) as error:
            return {"returncode": 1, "error": str(error), "config": config_name}
        worker_directory = Path.cwd() # relative imports and model paths follow the caller
        worker_command = [sys.executable,
               "-m", "agent_lg.scripts.suggest_next",
               "--config", config,
               f"--data.init_args.data_path={data_path}",
               "--volume", str(volume_uL), "--n_iters", "1",
               # Keep each campaign output in its own directory.
               "--out_root", str(results_root)]
        # Installed and editable packages share the same worker entry point.
        worker_environment = os.environ.copy()
        for name in PATH_SETTINGS:
            if worker_environment.get(name):
                worker_environment[name] = str(Path(worker_environment[name]).resolve())
        try:
            with profile_step("bo_worker"):
                worker_result = subprocess.run(worker_command, capture_output=True, text=True, cwd=str(worker_directory),
                                               env=worker_environment, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"returncode": 1, "error": f"BO worker failed: {error}", "config": config_name}
        primaries, reserves, json_path, mean_lengthscale = [], [], None, None
        try:
            for line in worker_result.stdout.splitlines():
                if line.startswith("JSON:"):
                    json_path = line[len("JSON:"):].strip()
                elif line.startswith("LENGTHSCALE:"):
                    mean_lengthscale = float(line[len("LENGTHSCALE:"):].strip())
                    if not math.isfinite(mean_lengthscale) or mean_lengthscale <= 0:
                        raise ValueError("Invalid GP lengthscale")
                for prefix, suggestions in [("PROTO:", primaries), ("RESERVE:", reserves)]:
                    if line.startswith(prefix):
                        payload = line[len(prefix):].strip()
                        protocol_path, separator, concentration_text = payload.partition(" | ")
                        if not separator or not protocol_path.strip():
                            raise ValueError("Malformed BO suggestion output")
                        concentrations = {}
                        for token in concentration_text.split():
                            key, separator, value = token.partition("=")
                            if not separator or key in concentrations:
                                raise ValueError("Malformed or duplicate concentration field")
                            concentrations[key.strip()] = float(value)
                        suggestions.append({"path": protocol_path.strip(), "conc": concentrations})
                        break
            if worker_result.returncode == 0:
                if file_revision(data_path) != dataset_revision:
                    raise ValueError("Campaign pool changed during BO; rerun against the current pool before submission")
                validate_concentrations(primaries)
                if reserves:
                    validate_concentrations(reserves)
                json_file = Path(json_path or "")
                if not json_file.is_absolute():
                    json_file = worker_directory / json_file
                if not json_file.resolve().is_relative_to(Path(results_root).resolve()):
                    raise ValueError("BO JSON is outside its configured results directory")
                data, saved = load_protocol(json_file)
                latest = select_protocol(data)
                for reported, stored in [(primaries, saved), (reserves, latest.get("reserve", []))]:
                    if len(reported) != len(stored):
                        raise ValueError("Worker output and saved batch have different lengths")
                    for entry, record in zip(reported, stored):
                        concentrations = record.get("conc", record)
                        if set(entry["conc"]) != set(concentrations) or any(
                                not math.isclose(value, float(concentrations[key]), rel_tol=1e-7, abs_tol=1e-12)
                                for key, value in entry["conc"].items()):
                            raise ValueError("Worker output and saved batch disagree")
        except (OSError, TypeError, ValueError, KeyError) as error:
            return {"returncode": 1, "error": f"Invalid BO worker output: {error}", "config": config_name,
                    "worker_log": worker_diagnostic(results_root, worker_result.stdout, worker_result.stderr),
                    "stderr": worker_result.stderr[-500:]}
        outcome = {"returncode": worker_result.returncode, "n_primary": len(primaries), "n_reserve": len(reserves),
                "primaries": primaries, "reserves": reserves, "json_path": json_path,
                "config": config_name, "mean_lengthscale": mean_lengthscale,
                "stderr": worker_result.stderr[-500:] if worker_result.returncode != 0 else ""}
        if worker_result.returncode:
            outcome["worker_log"] = worker_diagnostic(results_root, worker_result.stdout, worker_result.stderr)
        return outcome
