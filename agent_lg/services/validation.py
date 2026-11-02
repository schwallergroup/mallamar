# Apply campaign restrictions and validate robot volume before calling ALICE.
from alice.utils.config import BOInputs, integer_input, positive_input, validate_bo_data
from alice.utils.config import validate_class_spec as alice_class_spec


class CampaignSafetyError(RuntimeError):
    """Keep the checkpoint and experiment evidence intact until the issue is resolved."""

def _campaign_class_path(path, name):
    if path.startswith('gollum.'):
        raise ValueError(f'[recipe] {name}.class_path uses retired Gollum paths; use ALICE class paths')


def validate_class_spec(value, name, allow_short_name=False):
    return alice_class_spec(value, name, allow_short_name, path_validator=_campaign_class_path)


def validate_recipe(config):
    if not isinstance(config, dict):
        raise ValueError('[recipe] Configuration must be a mapping')
    for section in ('data', 'surrogate_model', 'acquisition', 'bo'):
        validate_class_spec(config.get(section), section, allow_short_name=section == 'acquisition')
    data = config['data'].get('init_args', {})
    for section in ('featurizer', 'initializer'):
        validate_class_spec(data.get(section), f'data.init_args.{section}')
    if 'exclude_top' in data:
        raise ValueError('[recipe] Remove exclude_top; campaign rows must retain their identity')
    if 'pooling_method' in data['featurizer'].get('init_args', {}):
        raise ValueError('[recipe] Tokenization has no pooling_method; configure pooling on the embedding model')
    validate_bo_data(config)
    if 'freeze_embedder' in config and not isinstance(config['freeze_embedder'], bool):
        raise ValueError('[recipe] freeze_embedder must be a boolean')
    surrogate = config['surrogate_model'].get('init_args', {})
    if 'finetuning_model' in surrogate:
        validate_class_spec(surrogate['finetuning_model'], 'surrogate_model.init_args.finetuning_model')
    model = surrogate.get('finetuning_model', {}).get('init_args', {})
    if config.get('freeze_embedder') or surrogate.get('freeze_embedder') or model.get('trainable') is False:
        raise ValueError('[recipe] Embedding freezing was removed; keep embedding training enabled')


class SuggestionInputs(BOInputs):
    """Apply campaign restrictions before validating generic BO inputs."""

    def __init__(self, config, target_volume_uL):
        validate_recipe(config)
        positive_input(target_volume_uL, 'target_volume_uL')
        super().__init__(config)
