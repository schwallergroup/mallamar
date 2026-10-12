# Inspect recovery evidence without changing the pool, locks or checkpoint database.
import hashlib, json, logging, shutil, sqlite3
from contextlib import closing
from tempfile import TemporaryDirectory
from pathlib import Path
from agent_lg.services.campaigns import CampaignSettings
from agent_lg.services.artifacts import file_digest, verify_artifacts
from agent_lg.services.ownership import lock_status
from agent_lg.services.pools import CampaignPool

logger = logging.getLogger(__name__)


def read_record(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def checkpoint_id(path, thread_id):
    files = [path, Path(str(path) + '-wal')]
    revisions = {str(item): file_digest(item) if item.exists() else None for item in files}
    with TemporaryDirectory(prefix='mallamar-status-') as temporary:
        copied = Path(temporary) / path.name
        for item in files:
            if revisions[str(item)] is not None:
                shutil.copyfile(item, Path(temporary) / item.name)
        if any((file_digest(item) if item.exists() else None) != revisions[str(item)] for item in files):
            raise ValueError('Checkpoint changed during inspection; retry for a consistent snapshot')
        with closing(sqlite3.connect(copied.as_uri() + '?mode=ro', uri=True)) as connection:
            rows = connection.execute('SELECT checkpoint_id FROM checkpoints WHERE thread_id=? ORDER BY checkpoint_id DESC LIMIT 1', (thread_id,)).fetchall()
        return rows[0][0] if rows else None


def campaign_status(checkpoint_db, thread_id):
    saved = CampaignSettings(checkpoint_db, thread_id)
    report = {'thread_id': thread_id, 'checkpoint_db': saved.checkpoint_db,
              'read_only': True, 'issues': [], 'batches': [], 'locks': [], 'commands': [], 'workers': []}
    try:
        record = saved.load()
        if record is None:
            raise ValueError('No saved campaign settings')
    except (OSError, ValueError, TypeError, KeyError) as error:
        report['issues'].append(str(error))
        return report
    options, environment = record['settings'], record['environment']
    report.update(dataset=options['db_path'], reward_mode=options['reward_mode'])
    report['artifact_fingerprints'] = 'recorded' if record.get('artifacts') else 'legacy_unverified'
    try:
        verify_artifacts(record.get('artifacts'))
    except (OSError, ValueError, TypeError, KeyError) as error:
        report['issues'].append(str(error))
    checkpoint = Path(saved.checkpoint_db)
    if checkpoint.exists():
        try:
            report['checkpoint_id'] = checkpoint_id(checkpoint, thread_id)
        except (sqlite3.Error, OSError, ValueError) as error:
            report['issues'].append(f'Checkpoint unreadable: {error}')
    else:
        report['issues'].append('Checkpoint database is missing')
    directory = Path(options['db_path']).resolve().parent / '.mallamar'
    for path in sorted((directory / 'batches').glob('*.json')):
        try:
            item = read_record(path)
            if path.name.endswith('.commands.json'):
                batch_path = path.with_name(path.name.replace('.commands.json', '.json'))
                if not batch_path.exists() or read_record(batch_path).get('db_path') != options['db_path']:
                    continue
                for key, command in item.get('commands', {}).items():
                    report['commands'].append({'journal': str(path), 'key': key, 'state': command.get('state'),
                                               'seq': command.get('payload', {}).get('seq')})
            elif item.get('db_path') == options['db_path']:
                report['batches'].append({'journal': str(path), 'batch_id': item.get('batch_id'), 'state': item.get('state')})
        except (OSError, ValueError, TypeError, AttributeError) as error:
            report['issues'].append(f'Unreadable batch record {path}: {error}')
    completion = directory / 'completion' / (hashlib.sha256(thread_id.encode()).hexdigest() + '.json')
    if completion.exists():
        try:
            report['completion'] = read_record(completion).get('state')
        except (OSError, ValueError) as error:
            report['issues'].append(f'Completion unreadable: {error}')
    model_path = directory / 'models.json'
    if model_path.exists():
        try:
            report['models'] = read_record(model_path).get('models')
        except (OSError, ValueError) as error:
            report['issues'].append(f'Model record unreadable: {error}')
    for path in sorted((directory / 'workers').glob('*.json')):
        worker = lock_status(path)
        report['workers'].append(worker)
    try:
        frame = CampaignPool(options['db_path']).check(read_only=True)
        rewards = frame[['reward_oscillation', 'reward_signal']]
        report['pool'] = {'integrity': 'passed', 'observed': int(rewards.notna().all(axis=1).sum()),
                          'heldout': int(rewards.isna().all(axis=1).sum())}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        report['pool'] = {'integrity': 'needs_recovery'}
        report['issues'].append(str(error))
    identifier = hashlib.sha256(thread_id.encode()).hexdigest()[:16]
    locks = {Path(f'{saved.checkpoint_db}.{identifier}.mallamar.lock'),
             Path(f"{options['db_path']}.campaign.mallamar.lock")}
    for root in {Path(options['db_path']).parent, checkpoint.parent}:
        locks.update(root.glob('*.mallamar.lock'))
        if (root / '.mallamar').exists():
            locks.update((root / '.mallamar').rglob('*.mallamar.lock'))
    mailbox = environment.get('OCEAN_MAILBOX_ROOT')
    if mailbox:
        locks.update(Path(mailbox).rglob('*.mallamar.lock'))
        locks.update((Path(mailbox) / 'mailbox').glob('*.lock'))
    report['locks'] = [lock_status(path) for path in sorted(locks) if path.exists()]
    report['recovery_note'] = ('Read-only snapshot; files may change while a controller is active. '
                               'An exited controller does not prove its workers or robot have stopped. '
                               'This command never clears locks or resends commands.')
    return report
