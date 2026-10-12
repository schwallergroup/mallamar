# Save method settings before the first graph step, without saving credentials.
import hashlib, json, logging, os
from pathlib import Path
from agent_lg.services.storage import FileLock, write_json_atomic
from agent_lg.services.artifacts import capture_artifacts, verify_artifacts
from agent_lg.services.validation import integer_input
from agent_lg.services.method import CampaignRecipes

logger = logging.getLogger(__name__)

# Retired freeze keys remain in saved records only to reject incompatible restarts.
CAMPAIGN_ENVIRONMENT = (
    'OCEAN_REWARD_MODE', 'OCEAN_CAMPAIGN_DB', 'OCEAN_RESULTS_DB', 'OCEAN_MAILBOX_ROOT',
    'OCEAN_BO_MODE', 'OCEAN_BO_HYPOTHESIS', 'OCEAN_BO_INJECT', 'OCEAN_BO_FILTER',
    'OCEAN_FREEZE_MODE', 'OCEAN_FREEZE_EMBEDDER', 'OCEAN_IMPROVE_SIGNAL', 'OCEAN_LOCK_CONFIG',
    'OCEAN_CORPUS_DIR', 'OCEAN_HYPOTHESIS_PATH', 'OCEAN_HYP_STATIC', 'OCEAN_VOLUME_UL',
    'OCEAN_BATCH_SIZE', 'OCEAN_N_INIT', 'OCEAN_SEED', 'OCEAN_INJECT_N_AGENT',
    'OCEAN_CONTEXT_WINDOW', 'OCEAN_BO_LLM_TIMEOUT', 'OCEAN_BO_LLM_RETRIES',
    'OCEAN_RESULT_TIMEOUT_S', 'OCEAN_RESULTS_ROOT', 'OCEAN_AUDIT_PATH', 'OCEAN_LOG_DIR',
    'MALLAMAR_CAMPAIGN_MODE', 'MALLAMAR_REFERENCE_DATASETS', 'MALLAMAR_BO_TIMEOUT_S', 'MALLAMAR_CONFIGS_DIR',
)

PATH_SETTINGS = {
    'OCEAN_CAMPAIGN_DB', 'OCEAN_RESULTS_DB', 'OCEAN_MAILBOX_ROOT', 'OCEAN_CORPUS_DIR',
    'OCEAN_HYPOTHESIS_PATH', 'OCEAN_RESULTS_ROOT', 'OCEAN_AUDIT_PATH', 'OCEAN_LOG_DIR', 'MALLAMAR_CONFIGS_DIR',
}


class CampaignSettings:
    """One immutable settings record for a checkpoint thread."""

    def __init__(self, checkpoint_db, thread_id):
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError('[CampaignSettings] thread_id must be a nonempty string')
        self.checkpoint_db = str(Path(checkpoint_db).expanduser().resolve())
        self.thread_id = thread_id
        identifier = hashlib.sha256((self.checkpoint_db + '\0' + thread_id).encode()).hexdigest()
        self.path = Path(self.checkpoint_db).parent / '.mallamar' / 'campaigns' / (identifier + '.json')

    def load(self):
        if not self.path.exists():
            return None
        record = json.loads(self.path.read_text(encoding='utf-8'))
        return self.validate(record)

    def validate(self, record):
        if (not isinstance(record, dict) or record.get('version') != 1 or record.get('thread_id') != self.thread_id
                or record.get('checkpoint_db') != self.checkpoint_db):
            raise ValueError('[CampaignSettings] Settings do not match this checkpoint thread')
        required = {'db_path', 'user_message', 'model_name', 'backend', 'base_url',
                    'auto_approve', 'dry_run', 'reward_mode', 'recursion_limit', 'n_bo_iters'}
        if not isinstance(record.get('settings'), dict) or not required.issubset(record['settings']):
            raise ValueError('[CampaignSettings] Incomplete saved settings; original file preserved')
        if not isinstance(record.get('environment'), dict):
            raise ValueError('[CampaignSettings] Invalid saved environment; original file preserved')
        keys = set(record['environment'])
        required_environment = set(CAMPAIGN_ENVIRONMENT) - {'MALLAMAR_CONFIGS_DIR', 'MALLAMAR_CAMPAIGN_MODE'}
        if not required_environment.issubset(keys) or keys - set(CAMPAIGN_ENVIRONMENT):
            raise ValueError('[CampaignSettings] Incomplete saved environment; original file preserved')
        if any(value is not None and not isinstance(value, str) for value in record['environment'].values()):
            raise ValueError('[CampaignSettings] Saved environment values must be strings or null; original file preserved')
        CampaignRecipes().check_environment(record['environment'])
        settings = record['settings']
        for key in ('db_path', 'user_message', 'model_name', 'backend'):
            if not isinstance(settings[key], str) or not settings[key].strip():
                raise ValueError(f'[CampaignSettings] {key} must be a nonempty string; original file preserved')
        if settings['base_url'] is not None and not isinstance(settings['base_url'], str):
            raise ValueError('[CampaignSettings] base_url must be text or null; original file preserved')
        if any(not isinstance(settings[key], bool) for key in ('auto_approve', 'dry_run')):
            raise ValueError('[CampaignSettings] Approval and dry-run settings must be booleans; original file preserved')
        if settings['reward_mode'] not in ('off', 'freeze', 'recompute'):
            raise ValueError('[CampaignSettings] Unknown saved reward_mode; original file preserved')
        integer_input(settings['recursion_limit'], 'recursion_limit')
        if settings['n_bo_iters'] is not None:
            integer_input(settings['n_bo_iters'], 'n_bo_iters', minimum=0)
        return record

    def create(self, settings):
        with FileLock(self.path):
            if self.load() is not None:
                raise ValueError('[CampaignSettings] Settings already exist; resume this campaign')
            environment = {key: os.getenv(key) for key in CAMPAIGN_ENVIRONMENT}
            for key in PATH_SETTINGS:
                if environment[key]:
                    environment[key] = str(Path(environment[key]).expanduser().resolve())
            if environment['MALLAMAR_REFERENCE_DATASETS']:
                environment['MALLAMAR_REFERENCE_DATASETS'] = os.pathsep.join(
                    str(Path(value).expanduser().resolve()) for value in environment['MALLAMAR_REFERENCE_DATASETS'].split(os.pathsep) if value)
            record = {'version': 1, 'thread_id': self.thread_id, 'checkpoint_db': self.checkpoint_db,
                      'settings': settings, 'environment': environment, 'artifacts': capture_artifacts()}
            self.validate(record)
            write_json_atomic(self.path, record)
        return record

    def restore(self, record):
        self.validate(record) # reject the whole record before changing even one environment variable
        verify_artifacts(record.get('artifacts'))
        for key in CAMPAIGN_ENVIRONMENT:
            value = record['environment'].get(key)
            if value is None:
                os.environ.pop(key, None) # an absent axis must remain off after restart
            else:
                os.environ[key] = value
        return record['settings']

    def require_checkpoint_if_submitted(self, record):
        database = record['settings']['db_path']
        directory = Path(database).parent / '.mallamar'
        completion_id = hashlib.sha256(self.thread_id.encode()).hexdigest()
        if (directory / 'completion' / (completion_id + '.json')).exists():
            raise ValueError('[CampaignSettings] Completion exists but its checkpoint is missing; restore the checkpoint before resuming')
        for path in (directory / 'batches').glob('*.json'):
            if path.name.endswith('.commands.json'):
                continue
            batch = json.loads(path.read_text(encoding='utf-8'))
            if batch.get('db_path') == database and batch.get('batch_id', '').startswith(self.thread_id + ':'):
                raise ValueError('[CampaignSettings] Submission evidence exists but its checkpoint is missing; restore the checkpoint instead of starting another loop')
