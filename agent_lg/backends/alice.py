# Select ALICE's pool data class and apply campaign-specific validation.
from copy import deepcopy
import os
from agent_lg.services.validation import validate_recipe


def alice_config(config):
    """Copy the config and require available candidates for live campaigns."""
    validate_recipe(config)
    result = deepcopy(config) # callers keep their original settings
    result['data']['class_path'] = 'alice.data.module.CampaignDataModule'
    result['data']['init_args']['require_candidates'] = bool(os.getenv('OCEAN_CAMPAIGN_DB'))
    return result
