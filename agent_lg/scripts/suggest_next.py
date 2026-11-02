# Command-line entry point for campaign suggestions.
import argparse, logging, yaml
from pathlib import Path
from agent_lg.services.suggestions import SuggestionService
from agent_lg.services.artifacts import check_recipe
from agent_lg.services.storage import WorkerRun
from agent_lg.services.validation import SuggestionInputs
from agent_lg.services.method import CampaignRecipes

logger = logging.getLogger(__name__)


# Kept callable for existing scripts and agent integrations.
def run(config, target_volume_uL: float):
    return SuggestionService(config).suggest(target_volume_uL)

def main():
    parser = argparse.ArgumentParser(description="ALICE campaign suggestions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--volume", type=float, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_iters", type=int)
    parser.add_argument("--reserve_size", type=int)
    parser.add_argument("--out_root")
    args, overrides = parser.parse_known_args()
    config_path = Path(args.config).expanduser().resolve()
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        parser.error(f'Cannot read recipe {config_path}: {error}')
    if not isinstance(config, dict):
        parser.error("Config must be a YAML mapping")
    for override in overrides:
        if not override.startswith("--") or "=" not in override:
            parser.error(f"Expected --section.key=value, got {override!r}")
        key, value = override[2:].split("=", 1)
        node = config
        parts = key.split(".")
        if not all(part.isidentifier() for part in parts):
            parser.error(f'Invalid override key: {key!r}')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                parser.error(f'Override {key!r} crosses a value that is not a mapping')
        try:
            node[parts[-1]] = yaml.safe_load(value)
        except yaml.YAMLError as error:
            parser.error(f'Invalid override {key!r}: {error}')
    for key in ("seed", "n_iters", "reserve_size", "out_root"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config.setdefault("seed", 1)
    config.setdefault("n_iters", 1)
    config.setdefault("reserve_size", 6)
    if args.batch_size is not None:
        if not isinstance(config.get('bo'), dict) or not isinstance(config['bo'].get('init_args', {}), dict):
            parser.error('bo.init_args must be a mapping')
        config['bo'].setdefault('init_args', {})['batch_size'] = args.batch_size
    try:
        CampaignRecipes().check_environment()
        SuggestionInputs(config, args.volume)
        CampaignRecipes().validate(config)
    except ValueError as error:
        parser.error(str(error))
    check_recipe(config_path, config['data']['init_args']['data_path'])
    with WorkerRun('bo'):
        run(config, target_volume_uL=args.volume)


if __name__ == "__main__":
    main()
