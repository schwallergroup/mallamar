# Default model configurations, reaction settings and deployment paths.

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parent
HERE = Path.cwd() # campaign outputs belong to the caller's workspace
DEFAULT_CONFIGS_DIR = PACKAGE_DIR / "configs"
SCRIPTS_DIR = PACKAGE_DIR / "scripts"

# Pass --checkpoint-db to use an explicit campaign location.
CHECKPOINT_DB = str(HERE / "agent_checkpoints.sqlite")

# Relative overrides use the caller's working directory; ~ uses their home directory.
def _configs_directory():
    return Path(os.getenv("MALLAMAR_CONFIGS_DIR") or DEFAULT_CONFIGS_DIR).expanduser().resolve()

# Default
CONFIGS_DIR = _configs_directory()
AVAILABLE_CONFIGS = {
    "t5gemma-s": {
        "path": str(CONFIGS_DIR / "t5gemma_s_joint.yaml"),
        "embedding_dim": 512,
        "conda_env": "mallamar",
        "description": "T5Gemma s-s-prefixlm, 512d. Sobol initialization with fixed qLogNParEGO.",
    },
}

CONC_COLS = ["conc_MA_M", "conc_NaBrO3_M", "conc_H2SO4_M", "conc_Ru_M"]
CONC_TOLERANCE = 1e-4 # absolute tolerance in M
REF_POINT = [-50.0, -25.0] # for HV, adapt depending on the objectives

# Worker sends concentrations through the mailbox; the hardware listener builds robot steps.
ROBOT_SCRIPT = str(SCRIPTS_DIR / "robot.py")

BO_KEY_MAP = {
    "NaBrO3_M": "NaBrO3",
    "MA_M": "MA",
    "H2SO4_M": "H2SO4",
    "Ru(bpy)3_M": "Ru(bpy)3",
}

# QC before each batch uses these concentrations in M.
QC_EXPERIMENTS = [
    {"MA": 0.3, "NaBrO3": 0.2, "H2SO4": 1.5, "Ru": 0.35e-3},
]

# Zero disables extra QC within a batch; the orchestrator requests QC before each batch.
QC_AFTER_N_EXPERIMENTS = 0 # within a batch if needed
QC_RUN_DURATION_S = 3600 # seconds
QC_RUN_STEP_MAX_S = 3600 


# Update the existing registry so callers see paths restored from campaign settings.
def configure_configs():
    global CONFIGS_DIR
    CONFIGS_DIR = _configs_directory()
    for model_config in AVAILABLE_CONFIGS.values():
        model_config["path"] = str(CONFIGS_DIR / Path(model_config["path"]).name)
    return CONFIGS_DIR


configure_configs()
