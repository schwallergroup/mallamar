# Keep batch modes and the fixed ALICE configuration stable across restarts.
import os, yaml
from pathlib import Path
from agent_lg.config import CONC_COLS, AVAILABLE_CONFIGS
from agent_lg.services.protocols import normalize_experiment, validate_concentrations
from agent_lg.services.validation import validate_recipe


class CampaignMethod:
    modes = {'bo': 0, 'bo-llm': 3, 'llm': 6}
    batch_size = 6

    def __init__(self, mode='bo-llm'):
        if not isinstance(mode, str) or mode not in self.modes:
            raise ValueError('[method] Choose bo (6 BO), bo-llm (3 BO + 3 LLM) or llm (6 LLM)')
        self.mode = mode
        self.n_agent = self.modes[mode]
        self.n_bo = self.batch_size - self.n_agent

    @classmethod
    def from_environment(cls, environment=None):
        environment = os.environ if environment is None else environment
        mode = environment.get('MALLAMAR_CAMPAIGN_MODE')
        if not mode:
            return None # historical records and low-level BO calls have no published mode
        method = cls(mode)
        enabled = '1' if method.n_agent else None
        expected = {'OCEAN_BO_HYPOTHESIS': enabled, 'OCEAN_BO_INJECT': enabled,
                    'OCEAN_BATCH_SIZE': '6', 'OCEAN_INJECT_N_AGENT': str(method.n_agent)}
        if any(environment.get(key) != value for key, value in expected.items()):
            raise ValueError('[method] Batch settings disagree with the saved campaign mode')
        if environment.get('OCEAN_BO_FILTER'):
            raise ValueError('[method] Chemistry filtering is not part of the published modes')
        return method

    @classmethod
    def configure(cls, mode=None):
        # Legacy injection counts are startup aliases; the explicit mode owns its settings.
        count = os.getenv('OCEAN_INJECT_N_AGENT')
        inferred = {'0': 'bo', '3': 'bo-llm', '6': 'llm'}.get(count)
        if count and inferred is None:
            raise ValueError('[method] OCEAN_INJECT_N_AGENT must be 0, 3 or 6')
        selected = mode if mode is not None else (os.getenv('MALLAMAR_CAMPAIGN_MODE') or inferred or 'bo-llm')
        method = cls(selected)
        if count and count != str(method.n_agent) and mode != 'bo':
            raise ValueError('[method] Injection count conflicts with the selected campaign mode')
        if os.getenv('OCEAN_BATCH_SIZE') not in (None, '', '6') or os.getenv('OCEAN_BO_FILTER'):
            raise ValueError('[method] Published modes use six experiments and no chemistry filter')
        os.environ.update(MALLAMAR_CAMPAIGN_MODE=method.mode, OCEAN_BATCH_SIZE='6',
                          OCEAN_INJECT_N_AGENT=str(method.n_agent))
        for key in ('OCEAN_BO_HYPOTHESIS', 'OCEAN_BO_INJECT'):
            if method.n_agent:
                os.environ[key] = '1'
            else:
                os.environ.pop(key, None)
        if not method.n_agent:
            os.environ['OCEAN_N_INIT'] = '6' # cold-start batches also contain six BO points
        return method

    def validate_selection(self, payload):
        injection = payload.get('inject', {})
        suggestions = payload.get('suggestions', [])
        if (payload.get('error') or payload.get('abort') or payload.get('returncode') not in (None, 0)
                or not isinstance(injection, dict)
                or injection.get('error') or injection.get('no_op')
                or not isinstance(suggestions, list) or len(suggestions) != self.batch_size
                or (self.n_agent and (injection.get('n_bo_kept') != self.n_bo
                    or injection.get('n_agent_injected') != self.n_agent
                    or injection.get('n_padded') != 0
                    or injection.get('suggestions') != suggestions))):
            raise ValueError(f'[method] Incomplete {self.mode} selection; require {self.n_bo} BO + {self.n_agent} LLM, with no BO padding')
        validate_concentrations(suggestions)
        concentrations = [normalize_experiment(point) for point in suggestions]
        if len({tuple(point[column] for column in CONC_COLS) for point in concentrations}) != self.batch_size:
            raise ValueError('[method] A batch must contain six distinct pool points')
        primaries = payload.get('bo_primaries', [])
        if not isinstance(primaries, list):
            raise ValueError('[method] Missing BO candidate record')
        primary_concentrations = [normalize_experiment(point) for point in primaries]
        if not self.n_agent:
            if injection or payload.get('filter') or concentrations != primary_concentrations:
                raise ValueError('[method] Pure BO must retain all six ALICE primaries in their original order')
            return suggestions
        if any(point not in primary_concentrations for point in concentrations[:self.n_bo]):
            raise ValueError('[method] Kept BO points must come from the recorded primary candidates')
        agent_points = injection.get('agent_points', [])
        if (not isinstance(agent_points, list) or len(agent_points) != self.n_agent
                or any(not isinstance(point, dict) or not point.get('matched_conc') for point in agent_points)):
            raise ValueError('[method] Missing LLM pool-snapping record')
        matched = [normalize_experiment(point['matched_conc']) for point in agent_points]
        if ([tuple(point[column] for column in CONC_COLS) for point in matched]
                != [tuple(point[column] for column in CONC_COLS) for point in concentrations[self.n_bo:]]):
            raise ValueError('[method] LLM selection does not match its pool-snapping record')
        return suggestions


class CampaignRecipes:
    """One T5Gemma configuration, with qLogNParEGO and embedding training enabled."""

    config_key = 't5gemma-s'
    model_name = 'google/t5gemma-s-s-prefixlm'
    acquisition_class = 'botorch.acquisition.multi_objective.parego.qLogNParEGO'

    def resolve(self, name):
        path = Path(AVAILABLE_CONFIGS[self.config_key]['path']).expanduser().resolve()
        if not isinstance(name, str) or not name.strip():
            raise ValueError('[config] Supply t5gemma-s or its joint YAML path')
        if name != self.config_key and Path(name).expanduser().resolve() != path:
            raise ValueError('[config] Only t5gemma-s / t5gemma_s_joint.yaml is supported; configuration switching was removed')
        if not path.is_file():
            raise ValueError(f'[config] Joint configuration not found: {path}')
        try:
            config = yaml.safe_load(path.read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f'[config] Cannot read {path}: {error}') from error
        self.validate(config)
        return path

    def validate(self, config):
        validate_recipe(config)
        data = config['data']['init_args']
        featurizer = data['featurizer'].get('init_args', {})
        model = config['surrogate_model'].get('init_args', {}).get('finetuning_model', {}).get('init_args', {})
        if (featurizer.get('model_name') != self.model_name or model.get('model_name') != self.model_name
                or featurizer.get('representation') != 'get_tokens'):
            raise ValueError('[config] Campaigns use google/t5gemma-s-s-prefixlm token embeddings')
        if config['acquisition']['class_path'] != self.acquisition_class:
            raise ValueError('[config] Campaign acquisition is fixed to qLogNParEGO')
        if config['bo']['init_args'].get('batch_size') != 6:
            raise ValueError('[config] Published campaign batches contain six experiments')
        if model.get('trainable') is not True:
            raise ValueError('[config] Embedding training must remain enabled')

    def check_environment(self, environment=None):
        environment = os.environ if environment is None else environment
        CampaignMethod.from_environment(environment)
        if environment.get('OCEAN_BO_MODE') not in (None, '', 'fixed'):
            raise ValueError('[config] Acquisition switching was removed; unset OCEAN_BO_MODE or use fixed')
        if environment.get('OCEAN_LOCK_CONFIG') not in (None, '', self.config_key):
            raise ValueError('[config] Only OCEAN_LOCK_CONFIG=t5gemma-s is supported')
        if environment.get('OCEAN_FREEZE_MODE') or environment.get('OCEAN_FREEZE_EMBEDDER'):
            raise ValueError('[config] Embedding freezing was removed; restore an unfrozen campaign or use its original code')

    def check_runtime(self, runtime):
        # Older unfrozen checkpoints stored a null freeze flag; never reactivate it.
        if not isinstance(runtime, dict) or set(runtime) - {'OCEAN_FREEZE_EMBEDDER'}:
            raise ValueError('[config] Unknown mutable campaign setting in checkpoint')
        self.check_environment(runtime)
