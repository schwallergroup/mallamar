# Keep campaign recipes, prompts, papers and model revisions stable across restarts.
import hashlib, json, logging, os
from pathlib import Path
from huggingface_hub import HfApi, try_to_load_from_cache
from agent_lg.config import DEFAULT_CONFIGS_DIR as CONFIGS_DIR
from agent_lg.services.storage import FileLock, write_json_atomic
from agent_lg.services.literature import LocalCorpus

logger = logging.getLogger(__name__)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files(scopes):
    result = {}
    for scope in scopes:
        directory = Path(scope['path'])
        for pattern in scope['patterns']:
            for path in sorted(directory.glob(pattern)):
                if path.is_file():
                    result[str(path.resolve())] = file_digest(path)
    return result


def capture_artifacts():
    scopes = [{'path': str(Path(os.getenv('MALLAMAR_CONFIGS_DIR') or CONFIGS_DIR).expanduser().resolve()), 'patterns': ['*.yaml', '*.yml']},
              {'path': str((Path(__file__).parents[1] / 'prompts').resolve()), 'patterns': ['*.md']}]
    corpus = LocalCorpus()
    if corpus.path is not None:
        scopes.append({'path': str(corpus.path), 'patterns': list(corpus.patterns)})
    return {'version': 1, 'scopes': scopes, 'files': artifact_files(scopes)}


def verify_artifacts(record):
    if record is None:
        logger.warning('[artifacts] Legacy settings have no file fingerprints; historical recipe identity cannot be verified')
        return
    if (not isinstance(record, dict) or record.get('version') != 1
            or not isinstance(record.get('scopes'), list) or not isinstance(record.get('files'), dict)):
        raise ValueError('[artifacts] Invalid fingerprint record; original retained')
    for scope in record['scopes']:
        if (not isinstance(scope, dict) or not isinstance(scope.get('path'), str)
                or not isinstance(scope.get('patterns'), list)
                or not all(isinstance(pattern, str) for pattern in scope['patterns'])):
            raise ValueError('[artifacts] Invalid fingerprint scope; original retained')
    current = artifact_files(record['scopes'])
    changed = sorted(path for path in set(current) | set(record['files']) if current.get(path) != record['files'].get(path))
    if changed:
        raise ValueError(f'[artifacts] Campaign files changed: {changed[:8]}; restore the recorded recipes/prompts/papers before resuming')


def model_names(config):
    names = set()
    if isinstance(config, dict):
        for key, value in config.items():
            if key == 'model_name' and isinstance(value, str):
                names.add(value)
            else:
                names.update(model_names(value))
    elif isinstance(config, list):
        for value in config:
            names.update(model_names(value))
    return sorted(names)


def resolve_model(name):
    local = Path(name)
    if local.is_dir():
        files = {str(path.relative_to(local)): file_digest(path) for path in sorted(local.rglob('*')) if path.is_file()}
        return {'local_path': str(local.resolve()), 'files': files}
    cached = try_to_load_from_cache(name, 'config.json')
    if isinstance(cached, str):
        revision = Path(cached).parent.name
    else:
        if os.getenv('HF_HUB_OFFLINE') in ('1', 'true', 'True') or os.getenv('TRANSFORMERS_OFFLINE') in ('1', 'true', 'True'):
            raise ValueError(f'[models] No cached revision for {name}; download it before an offline campaign')
        revision = HfApi().model_info(name).sha
    if not isinstance(revision, str) or len(revision) != 40 or any(char not in '0123456789abcdef' for char in revision):
        raise ValueError(f'[models] Cannot identify an immutable revision for {name}')
    return {'revision': revision}


def check_recipe(config_path, data_path):
    if not os.getenv('OCEAN_CAMPAIGN_DB'):
        return
    verify_current_campaign(data_path)
    path = Path(config_path).resolve()
    identifier = hashlib.sha256(str(path).encode()).hexdigest()
    record_path = Path(data_path).resolve().parent / '.mallamar' / 'recipes' / (identifier + '.json')
    record = {'version': 1, 'path': str(path), 'sha256': file_digest(path)}
    with FileLock(record_path):
        if record_path.exists():
            saved = json.loads(record_path.read_text())
            if saved != record:
                raise ValueError('[recipe] BO configuration changed since its first use; restore it before fitting')
        else:
            write_json_atomic(record_path, record)


def prepare_models(config):
    pool = os.getenv('OCEAN_CAMPAIGN_DB')
    if not pool:
        os.environ.pop('MALLAMAR_MODEL_REVISIONS', None)
        return
    record_path = Path(pool).resolve().parent / '.mallamar' / 'models.json'
    with FileLock(record_path):
        saved = json.loads(record_path.read_text()) if record_path.exists() else {'version': 1, 'models': {}}
        if not isinstance(saved, dict) or saved.get('version') != 1 or not isinstance(saved.get('models'), dict):
            raise ValueError('[models] Invalid revision record; original retained')
        original = json.dumps(saved, sort_keys=True)
        revisions = {}
        for name in model_names(config):
            identity = saved['models'].get(name)
            if identity is None:
                identity = resolve_model(name)
                saved['models'][name] = identity
            if not isinstance(identity, dict):
                raise ValueError('[models] Invalid model identity')
            if 'local_path' in identity:
                if identity != resolve_model(name):
                    raise ValueError('[models] Local model files changed; restore the recorded model')
            else:
                revision = identity.get('revision')
                if not isinstance(revision, str) or len(revision) != 40 or any(char not in '0123456789abcdef' for char in revision):
                    raise ValueError('[models] Invalid model revision')
                revisions[name] = revision # load the saved commit even if the upstream main branch advances
        if original != json.dumps(saved, sort_keys=True):
            write_json_atomic(record_path, saved)
    os.environ['MALLAMAR_MODEL_REVISIONS'] = json.dumps(revisions)


def verify_current_campaign(data_path):
    settings_path = os.getenv('MALLAMAR_ARTIFACTS_RECORD')
    if not settings_path:
        return # direct offline calls have no campaign settings record
    record = json.loads(Path(settings_path).read_text(encoding='utf-8'))
    if Path(record['settings']['db_path']).resolve() != Path(data_path).resolve():
        raise ValueError('[artifacts] Fingerprint record belongs to another campaign pool')
    verify_artifacts(record.get('artifacts'))
