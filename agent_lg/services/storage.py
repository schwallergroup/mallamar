# Preserve campaign files when writing fails or another writer changes them.
import errno, hashlib, json, logging, os, stat, tempfile, time
from agent_lg.services.ownership import acquire_lock, release_lock, owner_record
from functools import wraps
from pathlib import Path
import pandas as pd
from agent_lg.services.profiling import profile_step

logger = logging.getLogger(__name__)


def read_csv(path, **options):
    if isinstance(path, (str, os.PathLike)):
        path = Path(path).expanduser().resolve()
    options.setdefault("dtype", {"exp_id": "object"}) # IDs are labels; pandas must not turn 001 into 1
    with profile_step("csv_read"):
        return pd.read_csv(path, **options)


def read_campaign_csv(path, **options):
    requested = Path(path).expanduser().resolve()
    if requested.suffix.lower() not in (".csv", ".txt", ".tsv"):
        raise ValueError("Expected a tabular CSV/TXT/TSV dataset, not a protocol JSON or document")
    campaign = os.getenv("OCEAN_CAMPAIGN_DB")
    if campaign:
        campaign = Path(campaign).expanduser().resolve()
        permitted = {campaign, campaign.with_name('experiments_db.txt')}
        if os.getenv("OCEAN_RESULTS_DB"):
            permitted.add(Path(os.environ["OCEAN_RESULTS_DB"]).expanduser().resolve())
        permitted.update(Path(item).expanduser().resolve() for item in os.getenv("MALLAMAR_REFERENCE_DATASETS", "").split(os.pathsep) if item)
        if requested not in permitted:
            raise ValueError(f"Dataset is outside the current campaign: {path}. Explicit reference datasets must be listed in MALLAMAR_REFERENCE_DATASETS")
    return read_csv(requested, **options)


class FileLock:
    """Coordinate cooperating writers; an abandoned lock requires operator review."""

    def __init__(self, path, timeout_s=5):
        self.path = Path(str(Path(path).expanduser().resolve()) + ".mallamar.lock")
        self.timeout_s = timeout_s

    def __enter__(self):
        self.owner = acquire_lock(self.path, self.timeout_s)
        return self

    def __exit__(self, *error):
        release_lock(self.path, self.owner)


def file_revision(path):
    path = Path(path).expanduser().resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def fsync_directory(path):
    if os.name != "posix":
        return
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.fsync(directory) # persist directory entries after creating or replacing a file
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(directory)


def atomic_write(path, writer, validator=None, expected_revision=None, check_revision=False):
    path = Path(path).expanduser().resolve() # writer callbacks may change the caller's cwd
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent) # create a temporary file in the same directory as the target path
    temporary = Path(temporary_name)
    os.close(descriptor)
    try:
        if path.exists():
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode)) # retain access for cooperating lab processes
        writer(temporary)
        if validator is not None:
            validator(temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        if check_revision and file_revision(path) != expected_revision:
            raise RuntimeError(f"[atomic_write] {path} changed while preparing its update; original preserved")
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path, payload):
    atomic_write(path, lambda temporary: temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8"),
                 lambda temporary: json.loads(temporary.read_text(encoding="utf-8")))


def write_csv_atomic(frame, path, revision):
    def validate(temporary):
        saved = pd.read_csv(temporary)
        if list(saved.columns) != list(frame.columns) or len(saved) != len(frame):
            raise ValueError("[write_csv_atomic] Incomplete CSV update; original preserved")
    atomic_write(path, lambda temporary: frame.to_csv(temporary, index=False), validate,
                 expected_revision=revision, check_revision=True)


def locked_dataset(function):
    @wraps(function)
    def update(path, *args, **kwargs):
        with FileLock(path):
            return function(path, *args, **kwargs)
    return update


class WorkerRun:
    """Record a built-in worker's lifetime without changing its command or result."""

    def __init__(self, role):
        self.role = role
        self.path = None

    def __enter__(self):
        pool = os.getenv('OCEAN_CAMPAIGN_DB')
        command = os.getenv('MALLAMAR_COMMAND_JOURNAL')
        if pool:
            directory = Path(pool).resolve().parent / '.mallamar' / 'workers'
        elif command:
            directory = Path(command).resolve().parents[1] / 'workers'
        else:
            return self
        self.record = dict(owner_record(), role=self.role, state='running')
        self.path = directory / (self.record['token'] + '.json')
        write_json_atomic(self.path, self.record)
        return self

    def __exit__(self, error_type, error, traceback):
        if self.path is None:
            return
        success = error is None or (isinstance(error, SystemExit) and error.code in (None, 0))
        self.record['state'] = 'completed' if success else 'failed'
        try:
            write_json_atomic(self.path, self.record)
        except OSError as problem:
            logger.warning(f'[WorkerRun] Could not record worker exit: {problem}; inspect its process before recovery')
