# Check robot inputs before sending any experiment to the mailbox.
import json, logging, math
from pathlib import Path
from agent_lg.config import CONC_COLS

logger = logging.getLogger(__name__)

CONCENTRATION_ALIASES = {
    "MA": ("MA", "conc_MA_M"), "NaBrO3": ("NaBrO3", "conc_NaBrO3_M"),
    "H2SO4": ("H2SO4", "conc_H2SO4_M"), "Ru": ("Ru(bpy)3", "Ru", "conc_Ru_M"),
}
BUILDER_PARAMETERS = ("target_volume_uL", "set_temperature_C", "stir_rpm", "run_duration_s", "run_step_max_s")


def _finite_number(value, name):
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = float(value)
        if not math.isfinite(number):
            raise ValueError()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"[protocol] {name} must be a finite number") from error
    return number


def positive_number(value, name):
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"[protocol] {name} must be positive")
    return number


def validate_concentrations(suggestions):
    if not isinstance(suggestions, list) or not suggestions:
        raise ValueError("[protocol] Expected a non-empty list of experiments")
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            raise ValueError("[protocol] Each experiment must be an object")
        concentrations = suggestion.get("conc", suggestion)
        if not isinstance(concentrations, dict):
            raise ValueError("[protocol] Concentrations must be an object")
        for name, aliases in CONCENTRATION_ALIASES.items():
            values = [_finite_number(concentrations[key], name) for key in aliases if key in concentrations]
            if not values or values[0] < 0:
                raise ValueError(f"[protocol] {name} must be finite and non-negative")
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"[protocol] Conflicting concentration aliases for {name}")
    return suggestions


def protocol_parameters(entry):
    parameters = {name: entry[name] for name in (*BUILDER_PARAMETERS, "stall_timeout_s")
                  if entry.get(name) is not None}
    if entry.get("volume_uL") is not None:
        if entry.get("target_volume_uL") is not None and (
                _finite_number(entry["volume_uL"], "volume_uL") !=
                _finite_number(entry["target_volume_uL"], "target_volume_uL")):
            raise ValueError("[protocol] Conflicting volume parameters")
        parameters["target_volume_uL"] = entry["volume_uL"]
    for name, value in parameters.items():
        number = _finite_number(value, name)
        if name == "set_temperature_C":
            continue # negative Celsius values are valid
        if number < 0 or (name != "stir_rpm" and number == 0):
            raise ValueError(f"[protocol] {name} must be {'non-negative' if name == 'stir_rpm' else 'positive'}")
    return parameters


def select_protocol(data):
    entries = data if isinstance(data, list) else [data]
    protocol = next((entry for entry in reversed(entries)
                     if isinstance(entry, dict) and entry.get("suggestions")), None)
    if protocol is None:
        raise ValueError("[protocol] No experiments in protocol")
    return protocol


def validate_protocol(data):
    protocol = select_protocol(data)
    validate_concentrations(protocol["suggestions"])
    if protocol.get("reserve"):
        validate_concentrations(protocol["reserve"])
    protocol_parameters(protocol)
    for experiment in protocol["suggestions"]:
        protocol_parameters(experiment)
    return protocol


def load_protocol(path):
    path = Path(path)
    if path.suffix.lower() != ".json" or not path.is_file():
        raise ValueError(f"[protocol] Expected an existing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = validate_protocol(data)
    return data, protocol["suggestions"]


def normalize_experiment(suggestion):
    validate_concentrations([suggestion])
    concentrations = suggestion.get("conc", suggestion)
    normalized = {column: next(concentrations[key] for key in aliases if key in concentrations)
                  for column, aliases in zip(CONC_COLS, CONCENTRATION_ALIASES.values())}
    return {**normalized, **protocol_parameters(suggestion)}


def robot_experiments(data):
    protocol = validate_protocol(data) # check the whole batch before the first command
    parameters = protocol_parameters(protocol)
    return [{**normalize_experiment(suggestion), **parameters} for suggestion in protocol["suggestions"]]
